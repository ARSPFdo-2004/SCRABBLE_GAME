from __future__ import annotations

import json
from pathlib import Path
import unittest

from scrabble_plotter.board import BOARD_SIZE, CELL_SIZE_MM, parse_square_label
from scrabble_plotter.calibration import (
    PREMIUM_DOUBLE_LETTER,
    PREMIUM_DOUBLE_WORD,
    PlotterCalibration,
    path_for_storage,
    resolve_stored_path,
)
from scrabble_plotter.gemini_agent import (
    GeminiDetectedWord,
    PlotterAgentAction,
    _parse_json_object,
    format_detected_words_numbered,
    parse_detected_words,
)
from scrabble_plotter.main import build_parser
from scrabble_plotter.overlay import cell_label_positions, grid_segments, project_board_points
from scrabble_plotter.scanner import (
    CameraTile,
    CameraWord,
    PaddleOcrTextBox,
    detect_camera_tiles,
    detect_board_corners,
    detect_tile_corners,
    format_camera_words_numbered,
    identify_directional_words,
    identify_directional_tile_words,
    parse_camera_letter_data,
    parse_easyocr_text_boxes,
    parse_paddleocr_text_boxes,
    parse_tesseract_data,
    scan_board_image,
    scan_camera_letters,
    scan_camera_words,
)
from scrabble_plotter.scoring import LETTER_VALUES, score_board, square_label
from scrabble_plotter.serial_sender import format_move_command, format_reset_command, format_steps_command


class SquareParserTests(unittest.TestCase):
    def test_valid_labels(self) -> None:
        examples = {
            "A1": (0, 0, "A1"),
            "a1": (0, 0, "A1"),
            "H8": (7, 7, "H8"),
            "L12": (11, 11, "L12"),
        }

        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                square = parse_square_label(raw)
                self.assertEqual((square.col, square.row, square.label), expected)

    def test_invalid_labels(self) -> None:
        for label in ("M1", "A0", "A13", "O15", "11A", ""):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_square_label(label)


class PlotterCalibrationTests(unittest.TestCase):
    def test_fixed_coordinate_mapping_with_default_offset(self) -> None:
        calibration = PlotterCalibration()

        expected = {
            "A1": (15.0, 15.0),
            "A2": (15.0, 45.0),
            "B1": (45.0, 15.0),
            "L12": (345.0, 345.0),
        }

        for label, target in expected.items():
            with self.subTest(label=label):
                self.assertEqual(calibration.square_center_in_machine(parse_square_label(label)), target)

    def test_fixed_coordinate_mapping_with_board_corner_offset(self) -> None:
        calibration = PlotterCalibration(offset_x_mm=10.0, offset_y_mm=20.0)

        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A1")), (25.0, 35.0))
        self.assertEqual(calibration.square_center_in_machine(parse_square_label("L12")), (355.0, 365.0))

    def test_fixed_coordinate_mapping_with_custom_cell_size(self) -> None:
        calibration = PlotterCalibration(offset_x_mm=10.0, offset_y_mm=20.0, cell_size_mm=25.0)

        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A1")), (22.5, 32.5))
        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A2")), (22.5, 57.5))

    def test_persistence_round_trip(self) -> None:
        calibration = PlotterCalibration(
            camera_index=2,
            offset_x_mm=12.5,
            offset_y_mm=24.5,
            cell_size_mm=25.0,
            cart_x_mm=111.0,
            cart_y_mm=222.0,
            image_corners=[[0.0, 0.0], [360.0, 0.0], [360.0, 360.0], [0.0, 360.0]],
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.board_size, BOARD_SIZE)
        self.assertEqual(loaded.cell_size_mm, 25.0)
        self.assertEqual(loaded.camera_index, 2)
        self.assertEqual(loaded.offset_x_mm, 12.5)
        self.assertEqual(loaded.offset_y_mm, 24.5)
        self.assertEqual(loaded.x_steps_per_mm, 80.0)
        self.assertEqual(loaded.y_steps_per_mm, 80.0)
        self.assertEqual(loaded.cart_x_mm, 111.0)
        self.assertEqual(loaded.cart_y_mm, 222.0)
        self.assertEqual(len(loaded.image_corners), 4)

    def test_persistence_round_trip_includes_step_settings(self) -> None:
        calibration = PlotterCalibration(
            x_steps_per_mm=100.0,
            y_steps_per_mm=200.0,
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.x_steps_per_mm, 100.0)
        self.assertEqual(loaded.y_steps_per_mm, 200.0)

    def test_persistence_round_trip_includes_scan_settings(self) -> None:
        layout = [[PREMIUM_DOUBLE_WORD if row == col else "normal" for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
        calibration = PlotterCalibration(
            premium_layout=layout,
            ocr_confidence_threshold=72.5,
            ocr_cell_size_px=96,
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.premium_layout[0][0], PREMIUM_DOUBLE_WORD)
        self.assertEqual(loaded.premium_layout[1][0], "normal")
        self.assertEqual(loaded.ocr_confidence_threshold, 72.5)
        self.assertEqual(loaded.ocr_cell_size_px, 96)

    def test_missing_scan_settings_default_for_old_calibration_files(self) -> None:
        loaded = PlotterCalibration.from_dict({"board_size": BOARD_SIZE})

        self.assertEqual(len(loaded.premium_layout), BOARD_SIZE)
        self.assertEqual(len(loaded.premium_layout[0]), BOARD_SIZE)
        self.assertEqual(loaded.ocr_confidence_threshold, 50.0)
        self.assertEqual(loaded.ocr_cell_size_px, 80)

    def test_image_path_can_be_stored_relative_to_calibration_file(self) -> None:
        calibration_path = Path("project/calibration/board.json")
        image_path = Path("project/images/board.jpg")

        stored_path = path_for_storage(image_path, calibration_path)

        self.assertEqual(Path(stored_path), Path("../images/board.jpg"))
        self.assertEqual(resolve_stored_path(stored_path, calibration_path), image_path.resolve())

    def test_square_center_in_image_uses_camera_corners(self) -> None:
        calibration = PlotterCalibration(
            image_corners=[[0.0, 0.0], [360.0, 0.0], [360.0, 360.0], [0.0, 360.0]],
        )

        self.assertEqual(calibration.square_center_in_image(parse_square_label("A1")), (15.0, 15.0))
        self.assertEqual(calibration.square_center_in_image(parse_square_label("L12")), (345.0, 345.0))

    def test_cart_position_in_machine(self) -> None:
        calibration = PlotterCalibration(cart_x_mm=90.0, cart_y_mm=140.0)

        self.assertEqual(calibration.cart_position_in_machine(), (90.0, 140.0))


class OverlayTests(unittest.TestCase):
    def test_project_board_points(self) -> None:
        corners = [(0.0, 0.0), (360.0, 0.0), (360.0, 360.0), (0.0, 360.0)]

        points = project_board_points(corners, [(0.5, 0.5), (11.5, 11.5)])

        self.assertEqual(points[0], (15.0, 15.0))
        self.assertEqual(points[1], (345.0, 345.0))

    def test_grid_segments_and_labels(self) -> None:
        corners = [(0.0, 0.0), (360.0, 0.0), (360.0, 360.0), (0.0, 360.0)]

        self.assertEqual(len(grid_segments(corners)), (BOARD_SIZE + 1) * 2)
        labels = dict(cell_label_positions(corners))
        self.assertEqual(labels["A1"], (15.0, 15.0))
        self.assertEqual(labels["L12"], (345.0, 345.0))


class GCodeFormattingTests(unittest.TestCase):
    def test_gcode_formatting(self) -> None:
        self.assertEqual(format_move_command(10.0, 20.5, 1500.0), "G0 X10 Y20.5 F1500")

    def test_reset_command_formatting(self) -> None:
        self.assertEqual(format_reset_command(), "HOMEZERO")

    def test_steps_command_formatting(self) -> None:
        self.assertEqual(format_steps_command(80.0, 100.5), "STEPS X80 Y100.5")


class ScoringTests(unittest.TestCase):
    def test_standard_letter_values(self) -> None:
        self.assertEqual(LETTER_VALUES["A"], 1)
        self.assertEqual(LETTER_VALUES["Z"], 10)

    def test_scores_horizontal_and_vertical_words(self) -> None:
        board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = "C"
        board[0][1] = "A"
        board[0][2] = "T"
        board[0][0] = "C"
        board[1][0] = "A"
        board[2][0] = "R"

        result = score_board(board)
        words = {word.word: word.score for word in result.words}

        self.assertEqual(words["CAT"], 5)
        self.assertEqual(words["CAR"], 5)
        self.assertEqual(result.total_score, 10)

    def test_applies_letter_and_word_premiums(self) -> None:
        board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = "Q"
        board[0][1] = "I"

        premiums = [["normal" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        premiums[0][0] = PREMIUM_DOUBLE_LETTER
        premiums[0][1] = PREMIUM_DOUBLE_WORD

        result = score_board(board, premium_layout=premiums)

        self.assertEqual(result.words[0].word, "QI")
        self.assertEqual(result.words[0].base_score, 21)
        self.assertEqual(result.words[0].score, 42)

    def test_blank_tiles_score_zero(self) -> None:
        board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = "Z"
        board[0][1] = "A"

        result = score_board(board, blank_squares={"A1"})

        self.assertEqual(result.words[0].word, "ZA")
        self.assertEqual(result.words[0].score, 1)


class ScannerTests(unittest.TestCase):
    def test_tesseract_parser_chooses_best_letter(self) -> None:
        letter, confidence = parse_tesseract_data({"text": ["", "B", "A"], "conf": ["-1", "42", "91"]})

        self.assertEqual(letter, "A")
        self.assertEqual(confidence, 91.0)

    def test_camera_letter_parser_filters_and_scales_captured_letters(self) -> None:
        letters = parse_camera_letter_data(
            {
                "text": ["cat", "3", "Z"],
                "conf": ["82", "91", "49"],
                "left": [20, 80, 120],
                "top": [40, 100, 140],
                "width": [60, 20, 30],
                "height": [30, 20, 30],
            },
            confidence_threshold=50.0,
            scale=2.0,
        )

        self.assertEqual(len(letters), 1)
        self.assertEqual(letters[0].text, "CAT")
        self.assertEqual((letters[0].left, letters[0].top, letters[0].width, letters[0].height), (10, 20, 30, 15))

    def test_easyocr_parser_filters_text_boxes(self) -> None:
        boxes = parse_easyocr_text_boxes(
            [
                _easyocr_result("ta-c", 0.91, 10, 20, 60, 24),
                _easyocr_result("Z", 0.49, 100, 20, 30, 24),
            ],
            confidence_threshold=50.0,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].text, "TAC")
        self.assertEqual(boxes[0].confidence, 91.0)
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height), (10.0, 20.0, 60.0, 24.0))

    def test_easyocr_parser_normalizes_o_and_i_confusions(self) -> None:
        boxes = parse_easyocr_text_boxes(
            [
                _easyocr_result("0IL", 0.91, 10, 20, 60, 24),
                _easyocr_result("1", 0.89, 100, 20, 14, 24),
                _easyocr_result("|", 0.88, 130, 20, 10, 24),
            ],
            confidence_threshold=50.0,
        )

        self.assertEqual([box.text for box in boxes], ["OIL", "I", "I"])

    def test_scan_camera_letters_uses_easyocr_reader(self) -> None:
        image = _synthetic_white_on_black_text_image()

        scan = scan_camera_letters(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [
                _easyocr_result("A", 0.82, 12, 92, 43, 38),
                _easyocr_result("B", 0.91, 75, 92, 43, 38),
            ],
        )

        self.assertEqual(scan.text(), "A")
        self.assertEqual(len(scan.letters), 1)
        self.assertEqual((scan.letters[0].left, scan.letters[0].top), (12, 92))

    def test_paddleocr_v2_parser_filters_text_boxes(self) -> None:
        boxes = parse_paddleocr_text_boxes(
            [
                [
                    [[[10, 20], [70, 20], [70, 44], [10, 44]], ("ta-c", 0.91)],
                    [[[100, 20], [130, 20], [130, 44], [100, 44]], ("Z", 0.49)],
                ]
            ],
            confidence_threshold=50.0,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].text, "TAC")
        self.assertEqual(boxes[0].confidence, 91.0)
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height), (10.0, 20.0, 60.0, 24.0))

    def test_paddleocr_v3_parser_reads_result_payload(self) -> None:
        boxes = parse_paddleocr_text_boxes(
            {
                "res": {
                    "rec_texts": ["DOG"],
                    "rec_scores": [0.88],
                    "rec_polys": [
                        [[20, 10], [42, 10], [42, 120], [20, 120]],
                    ],
                }
            },
            confidence_threshold=50.0,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].text, "DOG")
        self.assertEqual(boxes[0].confidence, 88.0)
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].width, boxes[0].height), (20.0, 10.0, 22.0, 110.0))

    def test_identifies_left_to_right_and_top_to_bottom_letter_runs(self) -> None:
        boxes = [
            _paddle_box("C", 92, 12, 20),
            _paddle_box("A", 88, 42, 20),
            _paddle_box("T", 86, 72, 20),
            _ocr_box("O", 90, 12, 92, 22, 22),
            _ocr_box("I", 88, 46, 92, 6, 22),
            _ocr_box("L", 87, 70, 92, 20, 22),
            _paddle_box("D", 94, 150, 12),
            _paddle_box("O", 90, 150, 42),
            _paddle_box("G", 89, 150, 72),
        ]

        words = identify_directional_words(boxes)

        self.assertEqual(
            [(word.word, word.direction) for word in words],
            [
                ("CAT", "horizontal_left_to_right"),
                ("OIL", "horizontal_left_to_right"),
                ("DOG", "vertical_top_to_bottom"),
            ],
        )

    def test_identifies_words_from_individual_tiles(self) -> None:
        tiles = [
            _camera_tile("C", 92, 12, 20),
            _camera_tile("A", 88, 42, 20),
            _camera_tile("T", 86, 72, 20),
            _camera_tile("D", 94, 150, 12),
            _camera_tile("O", 90, 150, 42),
            _camera_tile("G", 89, 150, 72),
        ]

        words = identify_directional_tile_words(tiles)

        self.assertEqual(
            [(word.word, word.direction) for word in words],
            [
                ("CAT", "horizontal_left_to_right"),
                ("DOG", "vertical_top_to_bottom"),
            ],
        )

    def test_detect_tile_corners_finds_separate_tiles(self) -> None:
        image = _synthetic_tile_image(
            [
                ("", 20, 30),
                ("", 80, 30),
                ("", 140, 30),
            ]
        )

        corners = detect_tile_corners(image)

        self.assertEqual(len(corners), 3)
        centers = [(round(sum(point[0] for point in tile) / 4), round(sum(point[1] for point in tile) / 4)) for tile in corners]
        self.assertEqual(centers, [(40, 50), (100, 50), (160, 50)])

    def test_detect_camera_tiles_reads_letter_inside_each_tile(self) -> None:
        image = _synthetic_dark_tile_image(
            [
                ("T", 20, 30),
                ("A", 80, 30),
                ("C", 140, 30),
            ]
        )
        letters = iter(["T", "A", "C"])

        tiles = detect_camera_tiles(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda crop: [_easyocr_result(next(letters), 0.91, 0, 0, 68, 68)],
        )

        self.assertEqual([tile.letter for tile in tiles], ["T", "A", "C"])

    def test_scan_camera_words_uses_easyocr_reader(self) -> None:
        image = _synthetic_multi_word_white_on_black_text_image()

        scan = scan_camera_words(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [
                _easyocr_result("CAT", 0.91, 18, 30, 85, 40),
                _easyocr_result("0IL", 0.88, 138, 30, 80, 40),
                _easyocr_result("DOG", 0.90, 20, 98, 90, 38),
            ],
        )

        self.assertEqual(scan.tiles, [])
        self.assertEqual([box.text for box in scan.text_boxes], ["CAT", "OIL"])
        self.assertEqual(
            [(word.word, word.direction) for word in scan.words],
            [
                ("CAT", "horizontal_left_to_right"),
                ("OIL", "horizontal_left_to_right"),
            ],
        )

    def test_format_camera_words_as_numbered_list(self) -> None:
        text = format_camera_words_numbered(
            [
                CameraWord("CAT", "horizontal_right_to_left", 92.0, 10, 20, 60, 24),
                CameraWord("CAR", "horizontal_left_to_right", 91.0, 10, 60, 60, 24),
                CameraWord("DOG", "vertical_top_to_bottom", 88.0, 150, 12, 20, 90),
            ]
        )

        self.assertEqual(
            text,
            "1. CAT - horizontal right-to-left (92%)\n"
            "2. CAR - horizontal left-to-right (91%)\n"
            "3. DOG - vertical top-to-bottom (88%)",
        )

    def test_detect_board_corners_finds_perspective_board_area(self) -> None:
        image, expected = _synthetic_camera_board_image()

        corners = detect_board_corners(image)

        self.assertEqual(len(corners), 4)
        for actual, target in zip(corners, expected):
            self.assertAlmostEqual(actual[0], target[0], delta=10.0)
            self.assertAlmostEqual(actual[1], target[1], delta=10.0)

    def test_scan_uses_mocked_ocr_and_confidence_threshold(self) -> None:
        cv2 = _require_cv2_for_tests()
        cell_size = 40
        image = _synthetic_board_image(cell_size, {"A1": "A"})
        calibration = PlotterCalibration(
            image_corners=[
                [0.0, 0.0],
                [float(BOARD_SIZE * cell_size), 0.0],
                [float(BOARD_SIZE * cell_size), float(BOARD_SIZE * cell_size)],
                [0.0, float(BOARD_SIZE * cell_size)],
            ],
            ocr_cell_size_px=cell_size,
            ocr_confidence_threshold=50.0,
        )

        scan = scan_board_image(image, calibration, ocr_reader=lambda crop, square: ("A", 49.0))
        cells = {cell.square: cell for cell in scan.cells}

        self.assertTrue(cells["A1"].occupied)
        self.assertEqual(cells["A1"].letter, "")
        self.assertFalse(cells["B1"].occupied)

    def test_scan_grid_orientation_maps_a1_and_l12(self) -> None:
        cell_size = 40
        image = _synthetic_board_image(cell_size, {"A1": "A", "L12": "Z"})
        calibration = PlotterCalibration(
            image_corners=[
                [0.0, 0.0],
                [float(BOARD_SIZE * cell_size), 0.0],
                [float(BOARD_SIZE * cell_size), float(BOARD_SIZE * cell_size)],
                [0.0, float(BOARD_SIZE * cell_size)],
            ],
            ocr_cell_size_px=cell_size,
            ocr_confidence_threshold=50.0,
        )

        scan = scan_board_image(
            image,
            calibration,
            ocr_reader=lambda crop, square: ({"A1": "A", "L12": "Z"}.get(square, ""), 99.0),
        )
        board = scan.board_letters()

        self.assertEqual(board[0][0], "A")
        self.assertEqual(board[11][11], "Z")
        self.assertEqual(square_label(11, 11), "L12")


class GeminiAgentTests(unittest.TestCase):
    def test_agent_action_accepts_valid_square(self) -> None:
        action = PlotterAgentAction.from_payload(
            {"action": "move_square", "square": "a1", "reason": "target selected"}
        )

        self.assertEqual(action.action, "move_square")
        self.assertEqual(action.square, "A1")

    def test_agent_action_rejects_invalid_square(self) -> None:
        with self.assertRaises(ValueError):
            PlotterAgentAction.from_payload({"action": "move_square", "square": "M1"})

    def test_agent_action_accepts_cart_reset_and_none(self) -> None:
        for raw in ("go_cart", "reset", "none"):
            with self.subTest(action=raw):
                action = PlotterAgentAction.from_payload({"action": raw, "square": "A1"})
                self.assertEqual(action.action, raw)
                self.assertIsNone(action.square)

    def test_json_parser_handles_code_fence(self) -> None:
        payload = _parse_json_object('```json\n{"action":"go_cart","reason":"done"}\n```')

        self.assertEqual(payload["action"], "go_cart")

    def test_parse_detected_words_normalizes_and_filters_results(self) -> None:
        words = parse_detected_words(
            {
                "words": [
                    {"word": "ta-c", "direction": "right to left", "confidence": 120},
                    {"word": "do", "direction": "top-to-bottom", "confidence": 0.85},
                    {"word": "ta-c", "direction": "horizontal_right_to_left", "confidence": 95},
                    {"word": "A", "direction": "vertical_top_to_bottom", "confidence": 90},
                    {"word": "BAD", "direction": "diagonal", "confidence": 90},
                ]
            }
        )

        self.assertEqual([(word.word, word.direction, word.confidence) for word in words], [
            ("TAC", "horizontal_right_to_left", 100.0),
            ("DO", "vertical_top_to_bottom", 85.0),
        ])

    def test_format_detected_words_as_numbered_list(self) -> None:
        text = format_detected_words_numbered(
            [
                GeminiDetectedWord("TAC", "horizontal_right_to_left", 92.0),
                GeminiDetectedWord("DOG", "vertical_top_to_bottom", 88.0),
            ]
        )

        self.assertEqual(
            text,
            "1. TAC - horizontal right-to-left (92%)\n"
            "2. DOG - vertical top-to-bottom (88%)",
        )


class CliTests(unittest.TestCase):
    def test_gui_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gui"])
        self.assertEqual(args.command, "gui")

    def test_scan_image_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scan-image", "--image", "board.jpg", "--calibration", "board.json"])
        self.assertEqual(args.command, "scan-image")
        self.assertEqual(args.image, "board.jpg")
        self.assertEqual(args.calibration, "board.json")

    def test_set_offset_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["set-offset", "--calibration", "board.json", "--x", "10", "--y", "20"])
        self.assertEqual(args.command, "set-offset")
        self.assertEqual(args.x, 10.0)
        self.assertEqual(args.y, 20.0)

    def test_move_gcode_command_does_not_replace_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "move",
                "--calibration",
                "board.json",
                "--square",
                "A1",
                "--port",
                "COM3",
                "--command",
                "G1",
                "--dry-run",
            ]
        )
        self.assertEqual(args.command, "move")
        self.assertEqual(args.gcode_command, "G1")

def _synthetic_board_image(cell_size: int, letters: dict[str, str]):
    import numpy as np

    cv2 = _require_cv2_for_tests()
    size = BOARD_SIZE * cell_size
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    for square, letter in letters.items():
        parsed = parse_square_label(square)
        x = parsed.col * cell_size + int(cell_size * 0.25)
        y = parsed.row * cell_size + int(cell_size * 0.72)
        cv2.putText(
            image,
            letter,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def _paddle_box(text: str, confidence: float, left: int, top: int) -> PaddleOcrTextBox:
    return _ocr_box(text, confidence, left, top, 20, 20)


def _ocr_box(
    text: str,
    confidence: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> PaddleOcrTextBox:
    return PaddleOcrTextBox(
        text=text,
        confidence=confidence,
        points=[
            (float(left), float(top)),
            (float(left + width), float(top)),
            (float(left + width), float(top + height)),
            (float(left), float(top + height)),
        ],
    )


def _easyocr_result(
    text: str,
    confidence: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> list[object]:
    return [
        [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ],
        text,
        confidence,
    ]


def _camera_tile(letter: str, confidence: float, left: int, top: int) -> CameraTile:
    return CameraTile(
        letter=letter,
        confidence=confidence,
        corners=[
            (float(left), float(top)),
            (float(left + 20), float(top)),
            (float(left + 20), float(top + 20)),
            (float(left), float(top + 20)),
        ],
    )


def _synthetic_tile_image(tiles: list[tuple[str, int, int]]):
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((150, 220, 3), 180, dtype=np.uint8)
    tile_size = 40
    for letter, left, top in tiles:
        cv2.rectangle(image, (left, top), (left + tile_size, top + tile_size), (245, 245, 238), -1)
        cv2.rectangle(image, (left, top), (left + tile_size, top + tile_size), (25, 25, 25), 2)
        if letter:
            cv2.putText(
                image,
                letter,
                (left + 11, top + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return image


def _synthetic_dark_tile_image(tiles: list[tuple[str, int, int]]):
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((150, 220, 3), 180, dtype=np.uint8)
    tile_size = 40
    for letter, left, top in tiles:
        cv2.rectangle(image, (left, top), (left + tile_size, top + tile_size), (8, 8, 8), -1)
        cv2.rectangle(image, (left, top), (left + tile_size, top + tile_size), (245, 245, 245), 2)
        if letter:
            cv2.putText(
                image,
                letter,
                (left + 11, top + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return image


def _synthetic_white_on_black_text_image():
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((150, 280, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (120, 75), (0, 0, 0), -1)
    cv2.putText(image, "CAT", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (150, 20), (260, 75), (255, 255, 255), -1)
    cv2.putText(image, "DOG", (158, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(image, (10, 90), (58, 135), (0, 0, 0), -1)
    cv2.putText(image, "A", (21, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (72, 90), (122, 135), (255, 255, 255), -1)
    cv2.putText(image, "B", (84, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def _synthetic_multi_word_white_on_black_text_image():
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((150, 280, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (120, 75), (0, 0, 0), -1)
    cv2.putText(image, "CAT", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (130, 20), (230, 75), (0, 0, 0), -1)
    cv2.putText(image, "OIL", (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (10, 90), (125, 140), (255, 255, 255), -1)
    cv2.putText(image, "DOG", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def _synthetic_camera_board_image():
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((600, 800, 3), 240, dtype=np.uint8)
    points = np.array([[160, 80], [650, 100], [620, 560], [130, 520]], dtype=np.int32)
    cv2.fillConvexPoly(image, points, (245, 245, 245))
    cv2.polylines(image, [points], True, (20, 20, 20), 5, cv2.LINE_AA)
    for index in range(BOARD_SIZE + 1):
        alpha = index / BOARD_SIZE
        left = (points[0] * (1 - alpha) + points[3] * alpha).astype(int)
        right = (points[1] * (1 - alpha) + points[2] * alpha).astype(int)
        top = (points[0] * (1 - alpha) + points[1] * alpha).astype(int)
        bottom = (points[3] * (1 - alpha) + points[2] * alpha).astype(int)
        cv2.line(image, tuple(left), tuple(right), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(image, tuple(top), tuple(bottom), (90, 90, 90), 1, cv2.LINE_AA)
    return image, [(160.0, 80.0), (650.0, 100.0), (620.0, 560.0), (130.0, 520.0)]


def _require_cv2_for_tests():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise unittest.SkipTest("OpenCV is required for scanner tests.") from exc
    return cv2


if __name__ == "__main__":
    unittest.main()
