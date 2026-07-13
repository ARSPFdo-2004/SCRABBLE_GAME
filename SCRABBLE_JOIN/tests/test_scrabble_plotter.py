from __future__ import annotations

import json
from pathlib import Path
import unittest

from scrabble_plotter.ai_player import (
    AiMoveCandidate,
    AiMoveChoice,
    HORIZONTAL,
    VERTICAL,
    generate_ai_move_candidates,
    top_ai_move_candidates,
)
from scrabble_plotter.board import BOARD_SIZE, CELL_SIZE_MM, parse_square_label
from scrabble_plotter.calibration import (
    MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT,
    PREMIUM_DOUBLE_LETTER,
    PREMIUM_DOUBLE_WORD,
    PREMIUM_TRIPLE_LETTER,
    PREMIUM_TRIPLE_WORD,
    PlotterCalibration,
    path_for_storage,
    resolve_stored_path,
)
from scrabble_plotter.camera import open_camera_capture, read_camera_frame
from scrabble_plotter.openai_agent import (
    OpenAIDetectedWord,
    OpenAIPlotterAgent,
    PlotterAgentAction,
    _parse_json_object,
    format_detected_words_numbered,
    normalize_openai_endpoint,
    parse_detected_words,
)
from scrabble_plotter.gui import (
    AI_CHALLENGE_LIGHT_MS,
    ChallengeCandidate,
    PICK_DROP_MAGNET_DELAY_MS,
    PICK_DROP_MOVE_DELAY_MS,
    PICK_DROP_Z_SETTLE_DELAY_MS,
    ScrabblePlotterApp,
    Z_DOWN_COMMAND,
    Z_UP_COMMAND,
)
from scrabble_plotter.main import build_parser
from scrabble_plotter.overlay import cell_label_positions, grid_segments, project_board_points
from scrabble_plotter.scanner import (
    CameraLetterScanResult,
    CameraGridCell,
    CameraOcrGrid,
    CameraTile,
    CameraWord,
    CameraWordScanResult,
    PaddleOcrTextBox,
    board_square_from_image_point,
    build_camera_ocr_grid,
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
    read_cell_with_tesseract,
    scan_board_image,
    scan_camera_letters,
    scan_camera_words,
    score_frame_quality,
    select_best_frame,
)
from scrabble_plotter.scoring import LETTER_VALUES, score_board, square_label
from scrabble_plotter.serial_sender import (
    BoardActuatorSender,
    GCodeSender,
    SerialConfig,
    format_move_command,
    format_reset_command,
    format_steps_command,
)
from scrabble_plotter.word_bank import (
    generated_reference_word_set,
    generated_reference_words,
    matched_matrix_words,
)


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

    def test_machine_label_orientation_can_mirror_columns(self) -> None:
        calibration = PlotterCalibration(
            machine_label_orientation=MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT,
        )

        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A1")), (345.0, 15.0))
        self.assertEqual(calibration.square_center_in_machine(parse_square_label("L1")), (15.0, 15.0))
        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A2")), (345.0, 45.0))

    def test_fixed_coordinate_mapping_with_custom_cell_size(self) -> None:
        calibration = PlotterCalibration(offset_x_mm=10.0, offset_y_mm=20.0, cell_size_mm=25.0)

        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A1")), (22.5, 32.5))
        self.assertEqual(calibration.square_center_in_machine(parse_square_label("A2")), (22.5, 57.5))

    def test_persistence_round_trip(self) -> None:
        calibration = PlotterCalibration(
            camera_index=2,
            tile_rack_camera_index=3,
            offset_x_mm=12.5,
            offset_y_mm=24.5,
            cell_size_mm=25.0,
            cart_x_mm=111.0,
            cart_y_mm=222.0,
            tile_cart_url="192.168.4.1",
            tile_cart_player_1_command="backward",
            tile_cart_player_2_command="forward",
            image_corners=[[0.0, 0.0], [360.0, 0.0], [360.0, 360.0], [0.0, 360.0]],
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.board_size, BOARD_SIZE)
        self.assertEqual(loaded.cell_size_mm, 25.0)
        self.assertEqual(loaded.camera_index, 2)
        self.assertEqual(loaded.tile_rack_camera_index, 3)
        self.assertEqual(loaded.offset_x_mm, 12.5)
        self.assertEqual(loaded.offset_y_mm, 24.5)
        self.assertEqual(loaded.machine_label_orientation, "A1-top-left")
        self.assertEqual(loaded.x_steps_per_mm, 80.0)
        self.assertEqual(loaded.y_steps_per_mm, 80.0)
        self.assertEqual(loaded.cart_x_mm, 111.0)
        self.assertEqual(loaded.cart_y_mm, 222.0)
        self.assertEqual(loaded.tile_cart_url, "192.168.4.1")
        self.assertEqual(loaded.tile_cart_player_1_command, "backward")
        self.assertEqual(loaded.tile_cart_player_2_command, "forward")
        self.assertEqual(len(loaded.image_corners), 4)

    def test_tile_cart_motion_defaults_round_trip(self) -> None:
        calibration = PlotterCalibration()

        loaded = PlotterCalibration.from_dict(calibration.to_dict())

        self.assertEqual(loaded.tile_cart_speed_pwm, 200)
        self.assertEqual(loaded.tile_cart_distance_cm, 5.0)

    def test_persistence_round_trip_includes_step_settings(self) -> None:
        calibration = PlotterCalibration(
            x_steps_per_mm=100.0,
            y_steps_per_mm=200.0,
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.x_steps_per_mm, 100.0)
        self.assertEqual(loaded.y_steps_per_mm, 200.0)

    def test_persistence_round_trip_includes_machine_label_orientation(self) -> None:
        calibration = PlotterCalibration(
            machine_label_orientation=MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT,
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.machine_label_orientation, MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT)
        self.assertEqual(loaded.square_center_in_machine(parse_square_label("A1")), (345.0, 15.0))

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

    def test_persistence_round_trip_includes_actuator_settings(self) -> None:
        calibration = PlotterCalibration(
            actuator_port="COM8",
            actuator_baud=57600,
            actuator_timeout=1.5,
            actuator_countdown_seconds=45,
        )

        payload = json.loads(json.dumps(calibration.to_dict()))
        loaded = PlotterCalibration.from_dict(payload)

        self.assertEqual(loaded.actuator_port, "COM8")
        self.assertEqual(loaded.actuator_baud, 57600)
        self.assertEqual(loaded.actuator_timeout, 1.5)
        self.assertEqual(loaded.actuator_countdown_seconds, 45)

    def test_missing_scan_settings_default_for_old_calibration_files(self) -> None:
        loaded = PlotterCalibration.from_dict({"board_size": BOARD_SIZE})

        self.assertEqual(len(loaded.premium_layout), BOARD_SIZE)
        self.assertEqual(len(loaded.premium_layout[0]), BOARD_SIZE)
        self.assertEqual(loaded.ocr_confidence_threshold, 50.0)
        self.assertEqual(loaded.ocr_cell_size_px, 80)
        self.assertEqual(loaded.actuator_port, "")
        self.assertEqual(loaded.actuator_baud, 115200)
        self.assertEqual(loaded.actuator_timeout, 2.0)
        self.assertEqual(loaded.actuator_countdown_seconds, 30)

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

    def test_premiums_can_be_limited_to_newly_placed_squares(self) -> None:
        board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = "Q"
        board[0][1] = "I"

        premiums = [["normal" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        premiums[0][0] = PREMIUM_DOUBLE_LETTER
        premiums[0][1] = PREMIUM_DOUBLE_WORD

        result = score_board(board, premium_layout=premiums, premium_squares={"B1"})

        self.assertEqual(result.words[0].word, "QI")
        self.assertEqual(result.words[0].base_score, 11)
        self.assertEqual(result.words[0].score, 22)

    def test_blank_tiles_score_zero(self) -> None:
        board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = "Z"
        board[0][1] = "A"

        result = score_board(board, blank_squares={"A1"})

        self.assertEqual(result.words[0].word, "ZA")
        self.assertEqual(result.words[0].score, 1)


class AiPlayerTests(unittest.TestCase):
    def _empty_board(self) -> list[list[str]]:
        return [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def test_empty_board_opening_can_be_any_contiguous_rack_word(self) -> None:
        candidates = top_ai_move_candidates(
            self._empty_board(),
            ["C", "A", "T"],
            words=["CAT"],
        )

        self.assertTrue(candidates)
        first = candidates[0]
        self.assertEqual((first.candidate_id, first.word, first.direction, first.start_square, first.end_square), ("C001", "CAT", HORIZONTAL, "A1", "C1"))
        self.assertEqual([(placement.rack_label, placement.square) for placement in first.placements], [("TR1", "A1"), ("TR2", "B1"), ("TR3", "C1")])

    def test_non_empty_board_reuses_existing_anchor_letter(self) -> None:
        board = self._empty_board()
        board[0][0] = "C"
        board[0][1] = "A"

        candidates = generate_ai_move_candidates(board, ["T"], words=["CAT"])
        anchored = [candidate for candidate in candidates if candidate.word == "CAT" and candidate.start_square == "A1"]

        self.assertEqual(len(anchored), 1)
        self.assertEqual([(placement.rack_label, placement.square, placement.letter) for placement in anchored[0].placements], [("TR1", "C1", "T")])

    def test_cross_words_must_match_reference_words(self) -> None:
        board = self._empty_board()
        board[0][1] = "Z"

        invalid = generate_ai_move_candidates(board, ["C", "A", "T"], words=["CAT"])
        valid = generate_ai_move_candidates(board, ["C", "A", "T"], words=["CAT", "ZA"])

        self.assertEqual(invalid, [])
        self.assertTrue(any(candidate.word == "CAT" and "ZA" in candidate.cross_words for candidate in valid))

    def test_occupied_square_conflict_is_rejected(self) -> None:
        board = self._empty_board()
        board[0][0] = "D"

        candidates = generate_ai_move_candidates(board, ["C", "A", "T"], words=["CAT"])

        self.assertEqual(candidates, [])

    def test_duplicate_rack_letters_use_distinct_slots(self) -> None:
        candidates = top_ai_move_candidates(
            self._empty_board(),
            ["A", "A"],
            words=["AA"],
        )

        self.assertTrue(candidates)
        self.assertEqual([placement.rack_label for placement in candidates[0].placements], ["TR1", "TR2"])

    def test_no_candidate_when_rack_cannot_make_word(self) -> None:
        candidates = generate_ai_move_candidates(
            self._empty_board(),
            ["Z"],
            words=["CAT"],
        )

        self.assertEqual(candidates, [])


class CameraTests(unittest.TestCase):
    def test_word_detection_board_mask_keeps_only_calibrated_area(self) -> None:
        import numpy as np

        app = object.__new__(ScrabblePlotterApp)
        app._calibration = PlotterCalibration(
            image_corners=[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]
        )
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)

        masked = ScrabblePlotterApp._frame_limited_to_calibrated_board(app, frame)

        self.assertEqual(masked[50, 50].tolist(), [200, 200, 200])
        self.assertEqual(masked[10, 10].tolist(), [0, 0, 0])

    def test_read_camera_frame_retries_until_usable_frame(self) -> None:
        frame = _FakeFrame()
        camera = _FakeCapture(
            reads=[
                (False, None),
                (True, _FakeFrame(shape=(0, 2, 3))),
                (True, frame),
            ]
        )

        self.assertIs(read_camera_frame(camera, attempts=3), frame)

    def test_open_camera_capture_releases_backend_that_returns_no_frames(self) -> None:
        bad_capture = _FakeCapture(reads=[(False, None), (False, None)])
        good_frame = _FakeFrame()
        good_capture = _FakeCapture(reads=[(True, good_frame)])
        cv2 = _FakeCv2([bad_capture, good_capture])

        result = open_camera_capture(
            cv2,
            2,
            width=640,
            height=480,
            camera_fourcc="MJPG",
            zoom_out=True,
            warmup_attempts=2,
            read_delay_seconds=0,
            backend_candidates=[("bad", 10), ("good", 20)],
        )

        self.assertTrue(bad_capture.released)
        self.assertIs(result.capture, good_capture)
        self.assertEqual(result.backend_name, "good")
        self.assertIs(result.first_frame, good_frame)
        self.assertEqual(cv2.open_calls, [(2, 10), (2, 20)])
        self.assertEqual(
            good_capture.set_calls,
            [
                (1, 1234),
                (2, 640),
                (3, 480),
                (4, 0.0),
            ],
        )

    def test_open_camera_capture_reports_when_no_backend_returns_frames(self) -> None:
        closed_capture = _FakeCapture(opened=False)
        empty_capture = _FakeCapture(reads=[(False, None)])
        cv2 = _FakeCv2([closed_capture, empty_capture])

        with self.assertRaisesRegex(RuntimeError, "Unable to capture frames from camera 9"):
            open_camera_capture(
                cv2,
                9,
                warmup_attempts=1,
                read_delay_seconds=0,
                backend_candidates=[("closed", 10), ("empty", 20)],
            )

        self.assertTrue(closed_capture.released)
        self.assertTrue(empty_capture.released)



class ScannerTests(unittest.TestCase):
    def test_tesseract_parser_chooses_best_letter(self) -> None:
        letter, confidence = parse_tesseract_data({"text": ["", "B", "A"], "conf": ["-1", "42", "91"]})

        self.assertEqual(letter, "A")
        self.assertEqual(confidence, 91.0)

    def test_cell_ocr_tries_multiple_preprocessed_views(self) -> None:
        cv2 = _require_cv2_for_tests()
        _require_tesseract_for_tests()

        self.assertEqual(
            read_cell_with_tesseract(_synthetic_letter_cell("T", cv2), "A1")[0],
            "T",
        )
        self.assertEqual(
            read_cell_with_tesseract(_synthetic_letter_cell("O", cv2), "B1")[0],
            "O",
        )

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

    def test_scan_camera_letters_keeps_non_white_on_black_text_when_needed(self) -> None:
        image = _synthetic_white_on_black_text_image()

        scan = scan_camera_letters(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [
                _easyocr_result("B", 0.91, 75, 92, 43, 38),
            ],
        )

        self.assertEqual(scan.text(), "B")
        self.assertEqual(len(scan.letters), 1)

    def test_frame_quality_prefers_sharp_text_frame(self) -> None:
        cv2 = _require_cv2_for_tests()
        sharp = _synthetic_white_on_black_text_image()
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

        sharp_quality = score_frame_quality(sharp)
        blurred_quality = score_frame_quality(blurred)
        selected, selected_quality = select_best_frame([blurred, sharp])

        self.assertGreater(sharp_quality.score, blurred_quality.score)
        self.assertGreater(selected_quality.score, blurred_quality.score)
        self.assertTrue((selected == sharp).all())

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

    def test_letter_runs_do_not_jump_over_blank_sized_gaps(self) -> None:
        boxes = [
            _paddle_box("C", 92, 12, 20),
            _paddle_box("T", 86, 72, 20),
            _paddle_box("D", 94, 150, 12),
            _paddle_box("G", 89, 150, 72),
        ]

        words = identify_directional_words(boxes)

        self.assertEqual(words, [])

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

    def test_tile_words_do_not_jump_over_blank_sized_gaps(self) -> None:
        tiles = [
            _camera_tile("C", 92, 12, 20),
            _camera_tile("T", 86, 72, 20),
        ]

        words = identify_directional_tile_words(tiles)

        self.assertEqual(words, [])

    def test_generated_reference_word_bank_contains_1000_words(self) -> None:
        words = generated_reference_words()
        bank = generated_reference_word_set()

        self.assertEqual(len(words), 1000)
        self.assertEqual(len(bank), 1000)
        for word in ("CAT", "DOG", "OIL", "WOOD", "WORD", "TREE"):
            with self.subTest(word=word):
                self.assertIn(word, bank)
        self.assertNotIn("ZZQ", bank)

    def test_matrix_words_match_only_left_to_right_and_top_to_bottom(self) -> None:
        matrix = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        matrix[0][0] = "C"
        matrix[0][1] = "A"
        matrix[0][2] = "T"
        matrix[0][5] = "Z"
        matrix[0][6] = "Z"
        matrix[0][7] = "Q"
        matrix[1][4] = "D"
        matrix[2][4] = "O"
        matrix[3][4] = "G"

        words = matched_matrix_words(matrix)

        self.assertEqual(
            [(word.word, word.direction, word.start_cell, word.end_cell) for word in words],
            [
                ("CAT", "horizontal_left_to_right", "A1", "C1"),
                ("DOG", "vertical_top_to_bottom", "E2", "E4"),
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

    def test_detect_camera_tiles_reads_light_tiles_on_dark_board(self) -> None:
        image = _synthetic_light_tile_board_image(
            [
                ("C", 20, 30),
                ("A", 80, 30),
                ("T", 140, 30),
            ]
        )
        letters = iter(["C", "A", "T"])

        tiles = detect_camera_tiles(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda crop: [_easyocr_result(next(letters), 0.91, 0, 0, 68, 68)],
        )

        self.assertEqual([tile.letter for tile in tiles], ["C", "A", "T"])

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

    def test_scan_camera_words_displays_only_reference_matches(self) -> None:
        image = _synthetic_multi_word_white_on_black_text_image()

        scan = scan_camera_words(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [
                _easyocr_result("CAT", 0.91, 18, 30, 85, 40),
                _easyocr_result("ZZQ", 0.88, 138, 30, 80, 40),
            ],
        )

        self.assertEqual([box.text for box in scan.text_boxes], ["CAT", "ZZQ"])
        self.assertEqual([(word.word, word.direction) for word in scan.words], [("CAT", "horizontal_left_to_right")])

    def test_scan_camera_words_suppresses_contained_partial_words(self) -> None:
        image = _synthetic_multi_word_white_on_black_text_image()

        scan = scan_camera_words(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [
                _easyocr_result("CUT", 0.91, 18, 30, 85, 40),
                _easyocr_result("UT", 0.88, 50, 30, 54, 40),
            ],
        )

        self.assertEqual([(word.word, word.direction) for word in scan.words], [("CUT", "horizontal_left_to_right")])

    def test_format_camera_words_as_numbered_list(self) -> None:
        text = format_camera_words_numbered(
            [
                CameraWord("CAR", "horizontal_left_to_right", 91.0, 10, 60, 60, 24),
                CameraWord("DOG", "vertical_top_to_bottom", 88.0, 150, 12, 20, 90),
            ]
        )

        self.assertEqual(
            text,
            "Horizontal left-to-right:\n"
            "1. CAR (91%)\n"
            "\n"
            "Vertical top-to-bottom:\n"
            "1. DOG (88%)",
        )

    def test_detect_board_corners_finds_perspective_board_area(self) -> None:
        image, expected = _synthetic_camera_board_image()

        corners = detect_board_corners(image)

        self.assertEqual(len(corners), 4)
        for actual, target in zip(corners, expected):
            self.assertAlmostEqual(actual[0], target[0], delta=10.0)
            self.assertAlmostEqual(actual[1], target[1], delta=10.0)

    def test_camera_ocr_grid_maps_tiles_to_board_cells(self) -> None:
        image, board_corners = _synthetic_camera_board_image()
        tile_corners = project_board_points(
            board_corners,
            [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
        )
        tile = CameraTile("A", 91.0, tile_corners)

        grid = build_camera_ocr_grid(image, tiles=[tile])

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(grid.board_letters()[0][0], "A")

    def test_camera_ocr_grid_splits_word_box_across_cells(self) -> None:
        image, board_corners = _synthetic_camera_board_image()
        word_points = project_board_points(
            board_corners,
            [(0.1, 0.2), (2.9, 0.2), (2.9, 0.8), (0.1, 0.8)],
        )

        grid = build_camera_ocr_grid(
            image,
            text_boxes=[PaddleOcrTextBox("CAT", 91.0, word_points)],
        )

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(grid.board_letters()[0][:3], ["C", "A", "T"])

    def test_board_square_from_image_point_maps_detected_grid_click(self) -> None:
        _, board_corners = _synthetic_camera_board_image()
        a1_center, l12_center = project_board_points(
            board_corners,
            [(0.5, 0.5), (11.5, 11.5)],
        )

        self.assertEqual(board_square_from_image_point(board_corners, *a1_center), "A1")
        self.assertEqual(board_square_from_image_point(board_corners, *l12_center), "L12")
        self.assertIsNone(board_square_from_image_point(board_corners, 10.0, 10.0))

    def test_scan_camera_words_returns_board_aligned_grid(self) -> None:
        image, board_corners = _synthetic_camera_board_image()
        word_points = project_board_points(
            board_corners,
            [(0.1, 0.2), (2.9, 0.2), (2.9, 0.8), (0.1, 0.8)],
        )

        scan = scan_camera_words(
            image,
            confidence_threshold=50.0,
            ocr_reader=lambda frame: [[[[x, y] for x, y in word_points], "CAT", 0.91]],
        )

        self.assertEqual([(word.word, word.direction) for word in scan.words], [("CAT", "horizontal_left_to_right")])
        self.assertIsNotNone(scan.grid)
        assert scan.grid is not None
        self.assertEqual(scan.grid.board_letters()[0][:3], ["C", "A", "T"])

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


class OpenAIAgentTests(unittest.TestCase):
    def _candidate(self, candidate_id: str = "C001") -> AiMoveCandidate:
        return AiMoveCandidate(
            candidate_id=candidate_id,
            word="CAT",
            direction=HORIZONTAL,
            start_row=0,
            start_col=0,
            score=5,
            placements=(),
            squares=("A1", "B1", "C1"),
        )

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
                    {"word": "cat", "direction": "left to right", "confidence": 120},
                    {"word": "do", "direction": "top-to-bottom", "confidence": 0.85},
                    {"word": "cat", "direction": "horizontal_left_to_right", "confidence": 95},
                    {"word": "ZZQ", "direction": "horizontal_left_to_right", "confidence": 90},
                    {"word": "A", "direction": "vertical_top_to_bottom", "confidence": 90},
                    {"word": "BAD", "direction": "diagonal", "confidence": 90},
                ]
            }
        )

        self.assertEqual([(word.word, word.direction, word.confidence) for word in words], [
            ("CAT", "horizontal_left_to_right", 100.0),
            ("DO", "vertical_top_to_bottom", 85.0),
        ])

    def test_format_detected_words_as_numbered_list(self) -> None:
        text = format_detected_words_numbered(
            [
                OpenAIDetectedWord("CAT", "horizontal_left_to_right", 92.0),
                OpenAIDetectedWord("DOG", "vertical_top_to_bottom", 88.0),
            ]
        )

        self.assertEqual(
            text,
            "Horizontal left-to-right:\n"
            "1. CAT (92%)\n"
            "\n"
            "Vertical top-to-bottom:\n"
            "1. DOG (88%)",
        )

    def test_choose_ai_move_accepts_valid_candidate_id(self) -> None:
        class Agent(OpenAIPlotterAgent):
            def _post(self, payload, timeout):  # type: ignore[no-untyped-def]
                self.payload = payload
                return {"choices": [{"message": {"content": '{"action":"play","candidate_id":"C001","reason":"best"}'}}]}

        agent = Agent("key")
        candidate = self._candidate("C001")

        choice = agent.choose_ai_move([candidate], ["C", "A", "T"])

        self.assertEqual(choice.action, "play")
        self.assertIs(choice.candidate, candidate)
        self.assertEqual(choice.reason, "best")

    def test_choose_ai_move_accepts_pass(self) -> None:
        class Agent(OpenAIPlotterAgent):
            def _post(self, payload, timeout):  # type: ignore[no-untyped-def]
                return {"choices": [{"message": {"content": '{"action":"pass","reason":"no good move"}'}}]}

        choice = Agent("key").choose_ai_move([self._candidate()], ["C", "A", "T"])

        self.assertEqual(choice.action, "pass")
        self.assertIsNone(choice.candidate)
        self.assertEqual(choice.reason, "no good move")

    def test_choose_ai_move_rejects_unknown_candidate_id(self) -> None:
        class Agent(OpenAIPlotterAgent):
            def _post(self, payload, timeout):  # type: ignore[no-untyped-def]
                return {"choices": [{"message": {"content": '{"action":"play","candidate_id":"C999","reason":"bad"}'}}]}

        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            Agent("key").choose_ai_move([self._candidate()], ["C", "A", "T"])

    def test_choose_ai_move_rejects_malformed_json(self) -> None:
        class Agent(OpenAIPlotterAgent):
            def _post(self, payload, timeout):  # type: ignore[no-untyped-def]
                return {"choices": [{"message": {"content": "not json"}}]}

        with self.assertRaises(json.JSONDecodeError):
            Agent("key").choose_ai_move([self._candidate()], ["C", "A", "T"])

    def test_choose_ai_move_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "OpenAI API key"):
            OpenAIPlotterAgent("")

    def test_endpoint_normalization_accepts_base_v1_url(self) -> None:
        self.assertEqual(
            normalize_openai_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
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


class _FakeSerialConnection:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.writes: list[str] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data.decode("utf-8"))

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        if not self.responses:
            return b""
        return self.responses.pop(0).encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _FakeVar:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: object) -> None:
        self.value = str(value)


class PickDropTests(unittest.TestCase):
    def test_pick_and_drop_runs_requested_order_from_rack_to_board(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.statuses = []
        app.logs = []
        app.calls = []
        app.delays = []

        class ImmediateRoot:
            def after(self, delay, callback):  # type: ignore[no-untyped-def]
                app.delays.append(delay)
                callback()

        app.root = ImmediateRoot()
        app._set_status = app.statuses.append
        app._log = app.logs.append
        app._show_error = lambda exc: (_ for _ in ()).throw(exc)
        app._send_pick_drop_ordered_command = lambda command: app.calls.append(("command", command))
        app._send_pick_drop_ordered_target_move = lambda target: app.calls.append(("move", target))

        ScrabblePlotterApp._pick_and_drop_targets(app, "TR1", "H8")

        self.assertEqual(
            app.calls,
            [
                ("command", Z_UP_COMMAND),
                ("move", "TR1"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M1"),
                ("command", Z_UP_COMMAND),
                ("move", "H8"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M0"),
                ("command", Z_UP_COMMAND),
            ],
        )
        self.assertEqual(app.statuses[-1], "Picked current tile and dropped on H8.")
        self.assertEqual(app.delays[0], PICK_DROP_Z_SETTLE_DELAY_MS)
        self.assertEqual(app.delays[1], PICK_DROP_MOVE_DELAY_MS)
        self.assertEqual(app.delays[3], PICK_DROP_MAGNET_DELAY_MS)
        self.assertEqual(app.delays[6], PICK_DROP_Z_SETTLE_DELAY_MS)
        self.assertEqual(app.delays[7], PICK_DROP_MAGNET_DELAY_MS)

    def test_pick_and_drop_accepts_board_pickup_and_rejects_bad_targets(self) -> None:
        app = object.__new__(ScrabblePlotterApp)

        ScrabblePlotterApp._validate_pick_drop_targets(app, "A1", "H8")
        ScrabblePlotterApp._validate_pick_drop_targets(app, "TR7", "H8")

        with self.assertRaises(ValueError):
            ScrabblePlotterApp._validate_pick_drop_targets(app, "TR8", "H8")
        with self.assertRaises(ValueError):
            ScrabblePlotterApp._validate_pick_drop_targets(app, "TR1", "TR2")

    def test_pick_and_drop_applies_current_z_height_before_sequence(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.pick_square_var = _FakeVar("TR1")
        app.drop_square_var = _FakeVar("H8")
        app.z_height_angle = _FakeVar("120")
        app.calls = []
        app._send_pick_drop_aux_command = lambda command: app.calls.append(("aux", command))
        app._pick_and_drop_targets = lambda pickup, drop: app.calls.append(("pick_drop", pickup, drop))
        app._show_error = lambda exc: (_ for _ in ()).throw(exc)

        ScrabblePlotterApp.pick_and_drop(app)

        self.assertEqual(app.calls, [("aux", "ZH120"), ("pick_drop", "TR1", "H8")])

    def test_pick_and_drop_rejects_overlapping_sequence(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._pick_drop_running = True
        app.pick_square_var = _FakeVar("TR1")
        app.drop_square_var = _FakeVar("H8")
        app.errors = []
        app._show_error = app.errors.append

        ScrabblePlotterApp.pick_and_drop(app)

        self.assertEqual(len(app.errors), 1)
        self.assertIn("already running", str(app.errors[0]))

    def test_pick_drop_auxiliary_commands_do_not_use_rack_move_sender(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.calls = []
        app._send_tile_rack_move_command = lambda command: app.calls.append(("rack", command))
        app._send_pick_drop_aux_command = lambda command: app.calls.append(("aux", command))

        ScrabblePlotterApp._send_pick_drop_ordered_command(app, "ZU")

        self.assertEqual(app.calls, [("aux", "ZU")])

    def test_tile_rack_move_uses_plotter_sender_not_actuator_sender(self) -> None:
        class Sender:
            def __init__(self) -> None:
                self.commands = []

            def send_command(self, command: str) -> list[str]:
                self.commands.append(command)
                return ["ok"]

        app = object.__new__(ScrabblePlotterApp)
        sender = Sender()
        app.logs = []
        app._serial_config = lambda: SerialConfig("COM5", 115200, 0.01, False)
        app._get_sender_for_config = lambda config: sender
        app._send_actuator_command = lambda command: (_ for _ in ()).throw(
            AssertionError("rack movement must not use the board actuator")
        )
        app._send_auxiliary_command = lambda command: (_ for _ in ()).throw(
            AssertionError("rack movement must not use auxiliary Z/magnet sender")
        )
        app._log = app.logs.append

        ScrabblePlotterApp._send_tile_rack_move_command(app, "G0 X335 Y30 F1500")

        self.assertEqual(sender.commands, ["G0 X335 Y30 F1500"])

    def test_tile_rack_move_raises_on_plotter_controller_error(self) -> None:
        class Sender:
            def send_command(self, command: str) -> list[str]:
                return ["err unknown command"]

        app = object.__new__(ScrabblePlotterApp)
        app.logs = []
        app._serial_config = lambda: SerialConfig("COM5", 115200, 0.01, False)
        app._get_sender_for_config = lambda config: Sender()
        app._log = app.logs.append

        with self.assertRaisesRegex(RuntimeError, "err unknown command"):
            ScrabblePlotterApp._send_tile_rack_move_command(app, "G0 X335 Y30 F1500")



class SerialSenderTests(unittest.TestCase):
    def test_auxiliary_command_can_reuse_open_connection_without_startup_g90(self) -> None:
        connection = _FakeSerialConnection(["OK Z UP\n"])
        sender = GCodeSender(SerialConfig(port="COM20", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.send_command("ZU", startup_g90=False)

        self.assertEqual(connection.writes, ["ZU\n"])
        self.assertEqual(responses, ["OK Z UP"])
        self.assertFalse(connection.closed)

    def test_auxiliary_gui_command_uses_current_sender(self) -> None:
        class RecordingSender:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool | None]] = []

            def send_command(self, command: str, *, startup_g90: bool | None = None) -> list[str]:
                self.calls.append((command, startup_g90))
                return ["OK MAGNET ON"]

        app = type("App", (), {})()
        app.sender = RecordingSender()
        app._get_sender = lambda: app.sender

        responses = ScrabblePlotterApp._write_auxiliary_serial_line(app, "M1")

        self.assertEqual(app.sender.calls, [("M1", False)])
        self.assertEqual(responses, ["OK MAGNET ON"])

    def test_board_actuator_sender_never_sends_startup_g90(self) -> None:
        connection = _FakeSerialConnection(["ok board up\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.board_up()

        self.assertEqual(connection.writes, ["BOARD_UP\n"])
        self.assertEqual(responses, ["ok board up"])
        self.assertFalse(connection.closed)

    def test_board_actuator_sender_formats_word_command(self) -> None:
        connection = _FakeSerialConnection(["ok word set HELLO\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.set_word("hello")

        self.assertEqual(connection.writes, ["WORD_SET HELLO\n"])
        self.assertEqual(responses, ["ok word set HELLO"])

    def test_board_actuator_sender_formats_word_list_command(self) -> None:
        connection = _FakeSerialConnection(["ok word list 2\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.set_words(["cat", "dog"])

        self.assertEqual(connection.writes, ["WORD_LIST CAT,DOG\n"])
        self.assertEqual(responses, ["ok word list 2"])

    def test_board_actuator_sender_formats_word_clear_command(self) -> None:
        connection = _FakeSerialConnection(["ok word clear\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.clear_words()

        self.assertEqual(connection.writes, ["WORD_CLEAR\n"])
        self.assertEqual(responses, ["ok word clear"])

    def test_board_actuator_sender_formats_light_cell_commands(self) -> None:
        connection = _FakeSerialConnection(["ok led cells 2\n", "ok led clear\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        light_responses = sender.light_cells(["a1", "b2"])
        clear_responses = sender.clear_lights()

        self.assertEqual(connection.writes, ["LED_CELLS A1,B2\n", "LED_CLEAR\n"])
        self.assertEqual(light_responses, ["ok led cells 2"])
        self.assertEqual(clear_responses, ["ok led clear"])


    def test_board_actuator_sender_formats_score_command(self) -> None:
        connection = _FakeSerialConnection(["ok score p1 12 p2 8\n"])
        sender = BoardActuatorSender(SerialConfig(port="COM21", baud=115200, timeout=0.01, startup_g90=True))
        sender._connection = connection

        responses = sender.set_scores(12, 8)

        self.assertEqual(connection.writes, ["SCORE P1 12 P2 8\n"])
        self.assertEqual(responses, ["ok score p1 12 p2 8"])



class GameStateTests(unittest.TestCase):
    def test_tile_cart_url_for_command_adds_http_and_route(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_cart_url_var = _FakeVar("192.168.4.1")
        app.tile_cart_speed_pwm_var = _FakeVar("200")
        app.tile_cart_distance_cm_var = _FakeVar("5")

        url = ScrabblePlotterApp._tile_cart_url_for_command(app, "forward")

        self.assertEqual(url, "http://192.168.4.1/forward?speed=200&distance=5.0")

    def test_tile_cart_rejects_unknown_command(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_cart_url_var = _FakeVar("http://192.168.4.1")
        app.tile_cart_speed_pwm_var = _FakeVar("200")
        app.tile_cart_distance_cm_var = _FakeVar("5")

        with self.assertRaisesRegex(ValueError, "forward, backward, or stop"):
            ScrabblePlotterApp._tile_cart_url_for_command(app, "left")

    def test_tile_cart_move_rejects_zero_speed(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_cart_url_var = _FakeVar("http://192.168.4.1")
        app.tile_cart_speed_pwm_var = _FakeVar("0")
        app.tile_cart_distance_cm_var = _FakeVar("5")

        with self.assertRaisesRegex(ValueError, "between 1 and 255"):
            ScrabblePlotterApp._tile_cart_url_for_command(app, "forward")

    def test_tile_cart_timeout_covers_expected_motor_movement(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_cart_speed_pwm_var = _FakeVar("35")
        app.tile_cart_distance_cm_var = _FakeVar("10")

        timeout = ScrabblePlotterApp._tile_cart_request_timeout(app, "forward")

        self.assertAlmostEqual(timeout, 6.3)

    def _game_app(self):
        app = object.__new__(ScrabblePlotterApp)
        app._letter_vars = [[_FakeVar("") for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        app._player_1_cells = set()
        app._player_2_cells = set()
        app._player_1_words = []
        app._player_2_words = []
        app._current_player = 1
        app._previous_board_state = {}
        app._turn_start_board_state = {}
        app._turn_scan_started_state = {}
        app._turn_scan_switch_after = False
        app._turn_scan_detected_words = []
        app._pending_turn_scan = False
        app._current_player_display_var = _FakeVar("Current Player: 1")
        app.logs = []
        app.table_updates = []
        app.start_turns = []
        app._log = app.logs.append
        app._update_words_table = lambda: app.table_updates.append((list(app._player_1_words), list(app._player_2_words)))
        app._update_grid_colors = lambda: None
        app._start_turn = lambda: app.start_turns.append(app._current_player)
        app._calibration_from_form = lambda: PlotterCalibration()
        app._blank_squares_from_form = lambda: set()
        return app

    def test_assign_words_uses_turn_start_snapshot_for_correct_player(self) -> None:
        app = self._game_app()
        app._letter_vars[0][0].set("C")
        app._letter_vars[0][1].set("A")
        app._letter_vars[0][2].set("T")
        app._letter_vars[1][4].set("D")
        app._letter_vars[2][4].set("O")
        app._letter_vars[3][4].set("G")
        previous_state = {"A1": "C", "B1": "A", "C1": "T"}

        ScrabblePlotterApp._assign_new_words_to_current_player(app, 2, previous_state=previous_state)

        self.assertEqual(app._player_1_words, [])
        self.assertEqual(app._player_2_words, ["DOG"])
        self.assertEqual(app._player_2_cells, {"E2", "E3", "E4"})
        self.assertNotIn("CAT", app._player_2_words)

    def test_assign_uses_current_scan_detected_words_when_board_scoring_misses_them(self) -> None:
        app = self._game_app()
        app._turn_scan_detected_words = ["DOG"]

        ScrabblePlotterApp._assign_new_words_to_current_player(app, 2, previous_state={})

        self.assertEqual(app._player_1_words, [])
        self.assertEqual(app._player_2_words, ["DOG"])
        self.assertEqual(app.table_updates, [([], ["DOG"])])

    def test_detected_scan_words_do_not_duplicate_existing_player_words(self) -> None:
        app = self._game_app()
        app._player_1_words = ["CAT"]
        app._turn_scan_detected_words = ["CAT", "DOG"]

        ScrabblePlotterApp._assign_new_words_to_current_player(app, 2, previous_state={})

        self.assertEqual(app._player_1_words, ["CAT"])
        self.assertEqual(app._player_2_words, ["DOG"])

    def test_scan_start_captures_current_player_two(self) -> None:
        app = self._game_app()
        app._current_player = 2
        app._current_player_display_var.set("Current Player: 2")
        app._previous_board_state = {"A1": "C"}
        app._turn_start_board_state = {"A1": "C"}

        player_id = ScrabblePlotterApp._begin_player_scan(app, switch_after_scan=True)

        self.assertEqual(player_id, 2)
        self.assertTrue(app._pending_turn_scan)
        self.assertEqual(app._turn_scan_player, 2)
        self.assertTrue(app._turn_scan_switch_after)
        self.assertEqual(app._turn_scan_started_state, {"A1": "C"})

    def test_turn_scan_complete_assigns_to_captured_player_two(self) -> None:
        app = self._game_app()
        app._current_player = 2
        app._turn_scan_player = 2
        app._turn_scan_switch_after = True
        app._turn_scan_started_state = {}
        app._letter_vars[1][4].set("D")
        app._letter_vars[2][4].set("O")
        app._letter_vars[3][4].set("G")

        ScrabblePlotterApp._on_turn_scan_complete(app)

        self.assertEqual(app._player_1_words, [])
        self.assertEqual(app._player_2_words, ["DOG"])
        self.assertEqual(app._current_player, 1)
        self.assertEqual(app.start_turns, [1])


class AiPlayerGuiTests(unittest.TestCase):
    def _candidate(self) -> AiMoveCandidate:
        from scrabble_plotter.ai_player import AiPlacement

        return AiMoveCandidate(
            candidate_id="C001",
            word="CAT",
            direction=HORIZONTAL,
            start_row=0,
            start_col=0,
            score=5,
            placements=(
                AiPlacement(rack_slot=1, letter="C", row=0, col=0),
                AiPlacement(rack_slot=2, letter="A", row=0, col=1),
                AiPlacement(rack_slot=3, letter="T", row=0, col=2),
            ),
            squares=("A1", "B1", "C1"),
        )

    def _turn_app(self, player: int = 1):
        app = object.__new__(ScrabblePlotterApp)
        app._current_player = player
        app._current_player_display_var = _FakeVar(f"Current Player: {player}")
        app._previous_board_state = {}
        app._turn_start_board_state = {}
        app._timer_running = True
        app._ai_player_running = False
        app._pick_drop_running = False
        app._ai_turn_after_id = None
        app.player_1_ai_enabled_var = _FakeVar("1")
        app.scheduled = []

        class Root:
            def after(self, delay, callback):  # type: ignore[no-untyped-def]
                app.scheduled.append((delay, callback))
                return "after-ai"

        app.root = Root()
        return app


    def test_player_two_turn_does_not_schedule_ai(self) -> None:
        app = self._turn_app(player=2)

        ScrabblePlotterApp._start_turn(app)

        self.assertEqual(app.scheduled, [])

    def test_turn_starts_while_tile_cart_moves_in_background(self) -> None:
        app = self._turn_app(player=1)
        app._timer_running = False
        app._timer_display_var = _FakeVar("Time Remaining: 2:00")
        app.tile_cart_enabled_var = _FakeVar(True)
        app._tile_cart_player_position = 2
        app.statuses = []
        app.logs = []
        app._set_status = app.statuses.append
        app._log = app.logs.append
        app.cart_moves = []
        app._start_cart_movement_countdown = lambda: app.cart_moves.append(app._current_player)

        ScrabblePlotterApp._start_turn(app)

        self.assertTrue(app._timer_running)
        self.assertEqual([delay for delay, _callback in app.scheduled], [200, 700])
        self.assertEqual(app.cart_moves, [1])
        self.assertEqual(app._current_player_display_var.get(), "Current Player: 1")
        self.assertTrue(app._timer_display_var.get().startswith("Time Remaining:"))
        self.assertIn("moving the tile cart in the background", app.statuses[-1])

    def test_ai_pick_drop_steps_use_each_rack_slot_and_square_in_order(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.z_height_angle = _FakeVar("80")
        app.calls = []
        app._send_pick_drop_ordered_command = lambda command: app.calls.append(("command", command))
        app._send_pick_drop_ordered_target_move = lambda target: app.calls.append(("move", target))
        app._send_pick_drop_ordered_reset = lambda: app.calls.append(("reset", "HOMEZERO"))

        steps = ScrabblePlotterApp._ai_pick_drop_steps(app, self._candidate())
        for _label, action in steps:
            action()

        self.assertEqual(
            app.calls,
            [
                ("command", "ZH80"),
                ("command", Z_UP_COMMAND),
                ("move", "TR1"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M1"),
                ("command", Z_UP_COMMAND),
                ("move", "A1"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M0"),
                ("command", Z_UP_COMMAND),
                ("command", Z_UP_COMMAND),
                ("move", "TR2"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M1"),
                ("command", Z_UP_COMMAND),
                ("move", "B1"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M0"),
                ("command", Z_UP_COMMAND),
                ("command", Z_UP_COMMAND),
                ("move", "TR3"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M1"),
                ("command", Z_UP_COMMAND),
                ("move", "C1"),
                ("command", Z_DOWN_COMMAND),
                ("command", "M0"),
                ("command", Z_UP_COMMAND),
                ("reset", "HOMEZERO"),
            ],
        )

    def test_ai_success_updates_board_clears_rack_and_switches_to_player_two(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._letter_vars = [[_FakeVar("") for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        app.tile_rack_letter_vars = [_FakeVar(letter) for letter in ["C", "A", "T", "", "", "", ""]]
        app._player_1_cells = set()
        app._player_2_cells = set()
        app._player_1_words = []
        app._player_2_words = []
        app._previous_board_state = {}
        app._turn_start_board_state = {}
        app._turn_scan_started_state = {}
        app._turn_scan_detected_words = []
        app._pending_turn_scan = False
        app._ai_player_running = True
        app._pick_drop_running = True
        app._pending_ai_move_candidate = self._candidate()
        app._current_player = 1
        app._current_player_display_var = _FakeVar("Current Player: 1")
        app.starts = []
        app.statuses = []
        app.logs = []
        app._calibration_from_form = lambda: PlotterCalibration()
        app._blank_squares_from_form = lambda: set()
        app._update_grid_colors = lambda: None
        app._update_words_table = lambda: None
        app._set_status = app.statuses.append
        app._log = app.logs.append
        app._start_turn = lambda: app.starts.append(app._current_player)

        ScrabblePlotterApp._finish_ai_move_success(app, self._candidate())

        self.assertEqual([app._letter_vars[0][col].get() for col in range(3)], ["C", "A", "T"])
        self.assertEqual([variable.get() for variable in app.tile_rack_letter_vars[:3]], ["", "", ""])
        self.assertEqual(app._player_1_words, ["CAT"])
        self.assertEqual(app._current_player, 2)
        self.assertEqual(app.starts, [2])

    def test_ai_pick_drop_failure_leaves_turn_on_player_one(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._current_player = 1
        app._ai_player_running = True
        app._pick_drop_running = True
        app._pending_ai_move_candidate = self._candidate()
        app.errors = []
        app._show_error = app.errors.append

        ScrabblePlotterApp._handle_pick_drop_step_error(app, RuntimeError("controller failed"))

        self.assertFalse(app._ai_player_running)
        self.assertFalse(app._pick_drop_running)
        self.assertEqual(app._current_player, 1)
        self.assertEqual(len(app.errors), 1)

    def test_manual_rack_letters_normalize_for_ai(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_rack_letter_vars = [_FakeVar(value) for value in ["c", "aa", "7", "T", "", "z!", "q"]]
        app.tile_rack_word_suggestions_var = _FakeVar("")
        app._tile_rack_move_state_ready = True

        ScrabblePlotterApp._normalize_tile_rack_letter_entries(app)

        self.assertEqual([variable.get() for variable in app.tile_rack_letter_vars], ["C", "A", "", "T", "", "Z", "Q"])
        self.assertEqual(ScrabblePlotterApp._ai_rack_letters(app), ["C", "A", "", "T", "", "Z", "Q"])

    def test_rack_scan_updates_visible_tile_rack_grid(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_rack_letter_vars = [_FakeVar("") for _ in range(7)]
        app.tile_rack_word_suggestions_var = _FakeVar("")
        app.tile_rack_status_var = _FakeVar("")
        app._tile_rack_move_state_ready = True
        app._tile_rack_calibrated_corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 70.0), (0.0, 70.0)]
        app._tile_rack_letters_from_camera = lambda frame, scan: ["C", "A", "T", "", "", "", ""]
        app._tile_rack_slot_position = lambda index: (100.0, 20.0 + index)
        app.logs = []
        app._log = app.logs.append
        app._refresh_camera_preview = lambda: None

        count = ScrabblePlotterApp._update_tile_rack_letters_from_scan(
            app,
            object(),
            CameraLetterScanResult(letters=[], grid=None),
            source="test scan",
        )

        self.assertEqual(count, 3)
        self.assertEqual([variable.get() for variable in app.tile_rack_letter_vars[:3]], ["C", "A", "T"])
        self.assertIn("TR1: C", app.tile_rack_status_var.get())

    def test_clear_manual_rack_letters(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.tile_rack_letter_vars = [_FakeVar(value) for value in ["C", "A", "T", "", "", "", ""]]
        app.tile_rack_word_suggestions_var = _FakeVar("old")
        app._tile_rack_move_state_ready = True
        app.statuses = []
        app._set_status = app.statuses.append

        ScrabblePlotterApp.clear_tile_rack_letters(app)

        self.assertEqual([variable.get() for variable in app.tile_rack_letter_vars], ["", "", "", "", "", "", ""])
        self.assertEqual(app.tile_rack_word_suggestions_var.get(), "")
        self.assertEqual(app.statuses[-1], "Rack letters cleared.")


class BoardActuatorGuiTests(unittest.TestCase):
    def test_startup_sends_board_up_when_actuator_port_is_saved(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_port_var = _FakeVar("COM9")
        app.calls = []
        app.logs = []
        app._send_actuator_command = lambda command: app.calls.append(command) or ["ok board up"]
        app._log = app.logs.append

        ScrabblePlotterApp._send_startup_board_up(app)

        self.assertEqual(app.calls, ["BOARD_UP"])
        self.assertIn("Startup board up command sent.", app.logs)

    def test_startup_skips_board_up_when_no_actuator_port_is_saved(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_port_var = _FakeVar("")
        app.calls = []
        app.logs = []
        app._send_actuator_command = lambda command: app.calls.append(command)
        app._log = app.logs.append

        ScrabblePlotterApp._send_startup_board_up(app)

        self.assertEqual(app.calls, [])
        self.assertIn("Startup board up skipped", app.logs[0])

    def test_close_sends_board_down_and_closes_serial_connections(self) -> None:
        class Closable:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class Root:
            def __init__(self) -> None:
                self.destroyed = False

            def destroy(self) -> None:
                self.destroyed = True

        app = object.__new__(ScrabblePlotterApp)
        app.actuator_port_var = _FakeVar("COM9")
        app.commands = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok board down"]
        app._log = lambda message: None
        app.stop_camera = lambda clear_preview=False: None
        app._actuator_sender = Closable()
        app._actuator_sender_key = ("COM9", 115200, 2.0)
        app._sender = Closable()
        app.root = Root()

        ScrabblePlotterApp._on_close(app)

        self.assertEqual(app.commands, ["BOARD_DOWN"])
        self.assertIsNone(app._actuator_sender)
        self.assertIsNone(app._actuator_sender_key)
        self.assertTrue(app._sender.closed)
        self.assertTrue(app.root.destroyed)

    def test_camera_challenge_words_are_unique_normalized_camera_words(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._last_camera_word_scan = type(
            "Scan",
            (),
            {
                "words": [
                    CameraWord("cat", "horizontal_left_to_right", 90.0, 0, 0, 10, 10),
                    CameraWord("CAT", "horizontal_left_to_right", 89.0, 0, 0, 10, 10),
                    CameraWord("do-g", "vertical_top_to_bottom", 88.0, 0, 0, 10, 10),
                ],
                "grid": None,
            },
        )()

        words = ScrabblePlotterApp._camera_challenge_words(app)

        self.assertEqual(words, ["CAT", "DOG"])

    def test_send_camera_words_to_actuator_uses_word_list_command(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.commands = []
        app.statuses = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok word list 2"]
        app._set_status = app.statuses.append

        ScrabblePlotterApp._send_camera_words_to_actuator(app, ["CAT", "DOG"])

        self.assertEqual(app.commands, ["WORD_LIST CAT,DOG"])
        self.assertEqual(app.statuses[-1], "Sent 2 camera word(s) to the board actuator challenge list.")

    def test_try_send_camera_words_to_actuator_clears_when_latest_scan_has_no_words(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_port_var = _FakeVar("COM9")
        app._last_camera_word_scan = type("Scan", (), {"words": [], "grid": None})()
        app.commands = []
        app.logs = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok word clear"]
        app._log = app.logs.append

        ScrabblePlotterApp._try_send_camera_words_to_actuator(app)

        self.assertEqual(app.commands, ["WORD_CLEAR"])
        self.assertIn("Cleared board actuator challenge words", app.logs[-1])

    def test_send_actuator_command_raises_on_arduino_error_response(self) -> None:
        class Sender:
            def send_command(self, command: str) -> list[str]:
                return ["err arbitrary"]

        app = object.__new__(ScrabblePlotterApp)
        app.statuses = []
        app.logs = []
        app._get_actuator_sender = lambda: Sender()
        app._set_status = app.statuses.append
        app._log = app.logs.append

        with self.assertRaises(RuntimeError) as raised:
            ScrabblePlotterApp._send_actuator_command(app, "CHALLENGE_START")

        self.assertEqual(app.statuses[-1], "Sent actuator command: CHALLENGE_START")

    def test_start_challenge_sends_all_words_and_light_grid_commands(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._letter_vars = [[_FakeVar("") for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        app._letter_vars[9][2].set("C")
        app._letter_vars[9][3].set("D")
        app._last_camera_word_scan = type(
            "Scan",
            (),
            {"words": [CameraWord("cat", "horizontal_left_to_right", 90.0, 0, 0, 10, 10)], "grid": None},
        )()
        app._last_camera_letter_scan = None
        app._last_scan = None
        app._pending_actuator_challenge_after_word_scan = False
        app.commands = []
        app.statuses = []
        app.errors = []
        app._capture_best_photo_for_ocr = lambda purpose: (_ for _ in ()).throw(RuntimeError("Start the camera first."))
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._show_error = app.errors.append

        ScrabblePlotterApp.start_actuator_challenge(app)

        self.assertEqual(app.errors, [])
        self.assertEqual(
            app.commands,
            [
                "WORD_CLEAR",
                "WORD_LIST CD",
                "WORD_CELLS 0 C10,D10",
                "CHALLENGE_START",
                "LED_CELLS C10,D10",
            ],
        )

    def test_challenge_setup_maps_every_display_word_to_its_lights(self) -> None:
        candidates = [
            ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 1),
            ChallengeCandidate("DOG", ("D2", "D3", "D4"), 5, 2),
        ]

        commands = ScrabblePlotterApp._challenge_setup_commands(candidates)

        self.assertEqual(
            commands,
            [
                "WORD_CLEAR",
                "WORD_LIST CAT,DOG",
                "WORD_CELLS 0 A1,B1,C1",
                "WORD_CELLS 1 D2,D3,D4",
            ],
        )

    def test_player_one_challenge_list_excludes_player_two_words(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._player_1_words = ["CAT", "ZZQ"]
        app._player_2_words = ["DOG"]
        app._build_challenge_candidates = lambda limit=100: [
            ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 2),
            ChallengeCandidate("DOG", ("D2", "D3", "D4"), 5, 2),
        ]
        app._plain_word_score = lambda word: len(word)

        candidates = ScrabblePlotterApp._player_1_challenge_candidates(app)

        self.assertEqual([candidate.word for candidate in candidates], ["CAT", "ZZQ"])
        self.assertTrue(all(candidate.owner == 1 for candidate in candidates))
        self.assertEqual(candidates[0].squares, ("A1", "B1", "C1"))
        self.assertEqual(candidates[1].squares, ())

    def test_player_one_challenge_list_keeps_only_latest_three_words(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._player_1_words = ["CAT", "DOG", "BIRD", "FISH", "LION"]
        app._build_challenge_candidates = lambda limit=100: []
        app._plain_word_score = lambda word: len(word)

        candidates = ScrabblePlotterApp._player_1_challenge_candidates(app)

        self.assertEqual([candidate.word for candidate in candidates], ["BIRD", "FISH", "LION"])

    def test_player_two_words_are_available_for_player_one_to_challenge(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._player_1_words = ["CAT"]
        app._player_2_words = ["DOG", "ZZQ"]
        app._build_challenge_candidates = lambda limit=100: [
            ChallengeCandidate("DOG", ("D2", "D3", "D4"), 5, 1),
        ]
        app._plain_word_score = lambda word: len(word)

        candidates = ScrabblePlotterApp._player_challenge_candidates(app, 2)

        self.assertEqual([candidate.word for candidate in candidates], ["DOG", "ZZQ"])
        self.assertTrue(all(candidate.owner == 2 for candidate in candidates))
        self.assertEqual(candidates[0].squares, ("D2", "D3", "D4"))

    def test_player_one_words_sync_in_navigation_order(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_port_var = _FakeVar("COM9")
        app._challenge_candidate_index = 7
        app.commands = []
        app._player_1_challenge_candidates = lambda: [
            ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 1),
            ChallengeCandidate("ZZQ", (), 29, 1),
        ]
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]

        ScrabblePlotterApp._sync_player_1_challenge_words(app)

        self.assertEqual(
            app.commands,
            [
                "WORD_CLEAR",
                "WORD_ADD CAT",
                "WORD_CELLS 0 A1,B1,C1",
                "WORD_ADD ZZQ",
            ],
        )
        self.assertEqual(app._challenge_candidate_index, 0)

    def test_challenge_button_choice_response_is_parsed(self) -> None:
        app = object.__new__(ScrabblePlotterApp)

        choice = ScrabblePlotterApp._challenge_choice_from_responses(
            app,
            ["ok challenge chosen 1 DOG"],
        )

        self.assertEqual(choice, (1, "DOG"))

    def test_timer_button_response_is_parsed(self) -> None:
        self.assertTrue(
            ScrabblePlotterApp._timer_button_pressed_from_responses(["ok timer pressed"])
        )
        self.assertFalse(
            ScrabblePlotterApp._timer_button_pressed_from_responses(["ok timer none"])
        )

    def test_timer_button_ends_the_active_turn(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._game_started = True
        app._timer_running = True
        app._turn_scanning = False
        app._current_player = 2
        app._current_player_display_var = _FakeVar("Current Player: 2")
        app.logs = []
        app.ended_turns = 0
        app._log = app.logs.append

        def end_turn() -> None:
            app.ended_turns += 1

        app._end_turn = end_turn

        ScrabblePlotterApp._handle_timer_button_press(app)

        self.assertEqual(app.ended_turns, 1)
        self.assertIn("Player 2 pressed the timer button", app.logs[-1])

    def test_timer_button_is_ignored_for_player_one(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._game_started = True
        app._timer_running = True
        app._turn_scanning = False
        app._current_player = 1
        app._current_player_display_var = _FakeVar("Current Player: 1")
        app.logs = []
        app.ended_turns = 0
        app._log = app.logs.append
        app._end_turn = lambda: setattr(app, "ended_turns", app.ended_turns + 1)

        ScrabblePlotterApp._handle_timer_button_press(app)

        self.assertEqual(app.ended_turns, 0)
        self.assertIn("only enabled for Player 2", app.logs[-1])

    def test_start_challenge_rejects_board_without_words(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._letter_vars = [[_FakeVar("") for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        app._last_camera_letter_scan = None
        app._last_camera_word_scan = None
        app._last_scan = None
        app._pending_actuator_challenge_after_word_scan = False
        app.commands = []
        app.errors = []
        app._capture_best_photo_for_ocr = lambda purpose: (_ for _ in ()).throw(RuntimeError("Start the camera first."))
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._show_error = app.errors.append

        ScrabblePlotterApp.start_actuator_challenge(app)

        self.assertEqual(app.commands, [])
        self.assertEqual(len(app.errors), 1)
        self.assertIn("No board words", str(app.errors[0]))

    def test_pending_challenge_uses_fresh_camera_word_scan_grid(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._camera_word_scan_token = 4
        app._camera_word_scan_running = True
        app._pending_actuator_challenge_after_word_scan = True
        app._letter_vars = [[_FakeVar("") for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        app._last_camera_letter_scan = None
        app._last_scan = None
        app.commands = []
        app.statuses = []
        app.errors = []
        app._lock_scan_to_manual_board_grid = lambda scan: None
        app._set_camera_words_text = lambda text: None
        app._refresh_camera_preview = lambda: None
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._show_error = app.errors.append
        def apply_grid(grid):
            for cell in grid.cells:
                app._letter_vars[cell.row][cell.col].set(cell.letter)
            return len(grid.cells)
        app._apply_camera_letter_scan_to_locked_board_grid = lambda scan: 0
        app._apply_camera_ocr_grid_to_board_form = apply_grid

        scan = CameraWordScanResult(
            words=[CameraWord("cat", "horizontal_left_to_right", 90.0, 0, 0, 10, 10)],
            grid=CameraOcrGrid(
                corners=[],
                cells=[
                    CameraGridCell(row=9, col=2, square="C10", letter="C", confidence=90.0, source="test"),
                    CameraGridCell(row=9, col=3, square="D10", letter="A", confidence=90.0, source="test"),
                    CameraGridCell(row=9, col=4, square="E10", letter="T", confidence=90.0, source="test"),
                ],
            ),
        )

        ScrabblePlotterApp._handle_camera_word_scan_result(app, scan, announce=False, scan_token=4)

        self.assertFalse(app._pending_actuator_challenge_after_word_scan)
        self.assertEqual(app.errors, [])
        self.assertEqual(
            app.commands,
            [
                "WORD_CLEAR",
                "WORD_LIST CAT",
                "WORD_CELLS 0 C10,D10,E10",
                "CHALLENGE_START",
                "LED_CELLS C10,D10,E10",
            ],
        )





    def test_reveal_actuator_word_sends_word_choose_only(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.commands = []
        app.errors = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._show_error = app.errors.append

        ScrabblePlotterApp.reveal_actuator_word(app)

        self.assertEqual(app.commands, ["WORD_CHOOSE"])
        self.assertEqual(app.errors, [])

    def test_light_test_controls_send_direct_light_commands(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.commands = []
        app._send_actuator_button_command = app.commands.append

        ScrabblePlotterApp.actuator_light_test(app)
        ScrabblePlotterApp.actuator_lights_off(app)

        self.assertEqual(app.commands, ["LED_TEST", "LED_CLEAR"])

    def test_light_letters_sends_manually_entered_squares(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_light_squares_var = _FakeVar("a1, B2; A1")
        app.actuator_word_var = _FakeVar("")
        app.commands = []
        app.statuses = []
        app.errors = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok led cells 2"]
        app._set_status = app.statuses.append
        app._show_error = app.errors.append

        ScrabblePlotterApp.actuator_light_letters(app)

        self.assertEqual(app.errors, [])
        self.assertEqual(app.commands, ["LED_CELLS A1,B2"])
        self.assertIn("Lit 2", app.statuses[-1])

    def test_light_letters_uses_word_squares_when_manual_squares_are_empty(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.actuator_light_squares_var = _FakeVar("")
        app.actuator_word_var = _FakeVar("dog")
        app._build_challenge_candidates = lambda limit=10: [
            ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 1),
            ChallengeCandidate("DOG", ("D2", "D3", "D4"), 5, 2),
        ]
        app.commands = []
        app.statuses = []
        app.errors = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok led cells 3"]
        app._set_status = app.statuses.append
        app._show_error = app.errors.append

        ScrabblePlotterApp.actuator_light_letters(app)

        self.assertEqual(app.errors, [])
        self.assertEqual(app.commands, ["LED_CELLS D2,D3,D4"])
        self.assertIn("DOG", app.statuses[-1])

    def test_gui_next_and_previous_word_update_lit_squares(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._challenge_candidates = [
            ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 1),
            ChallengeCandidate("DOG", ("D2", "D3", "D4"), 5, 2),
        ]
        app._challenge_candidate_index = 0
        app.commands = []
        app.statuses = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append

        ScrabblePlotterApp.next_actuator_word(app)
        ScrabblePlotterApp.previous_actuator_word(app)

        self.assertEqual(
            app.commands,
            [
                "WORD_NEXT",
                "LED_CELLS D2,D3,D4",
                "WORD_PREV",
                "LED_CELLS A1,B1,C1",
            ],
        )

    def test_challenge_choice_valid_word_awards_owner(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._challenge_candidates = [ChallengeCandidate("CAT", ("A1", "B1", "C1"), 5, 1)]
        app._current_player = 2
        app._current_player_display_var = _FakeVar("Current Player: 2")
        app._player_1_score = 0
        app._player_2_score = 0
        app._player_1_score_var = _FakeVar("")
        app._player_2_score_var = _FakeVar("")
        app.actuator_port_var = _FakeVar("COM9")
        app.commands = []
        app.logs = []
        app.statuses = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._log = app.logs.append

        ScrabblePlotterApp._handle_challenge_choice(app, 0, "CAT")

        self.assertEqual(app._player_1_score, 5)
        self.assertEqual(app._player_2_score, 0)
        self.assertEqual(app.commands, ["SCORE P1 5 P2 0"])
        self.assertIn("valid", app.statuses[-1])

    def test_live_timer_sends_each_remaining_second_once(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._last_actuator_timer_second = None
        app.actuator_port_var = _FakeVar("COM9")
        app.commands = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]

        ScrabblePlotterApp._send_timer_to_actuator(app, 119)
        ScrabblePlotterApp._send_timer_to_actuator(app, 119)
        ScrabblePlotterApp._send_timer_to_actuator(app, 118)

        self.assertEqual(app.commands, ["TIMER 119", "TIMER 118"])

    def test_challenge_choice_invalid_word_awards_challenger(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._challenge_candidates = [ChallengeCandidate("ZZQ", ("A1", "B1", "C1"), 29, 1)]
        app._current_player = 2
        app._current_player_display_var = _FakeVar("Current Player: 2")
        app._player_1_score = 0
        app._player_2_score = 0
        app._player_1_score_var = _FakeVar("")
        app._player_2_score_var = _FakeVar("")
        app.actuator_port_var = _FakeVar("COM9")
        app.commands = []
        app.logs = []
        app.statuses = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._log = app.logs.append

        ScrabblePlotterApp._handle_challenge_choice(app, 0, "ZZQ")

        self.assertEqual(app._player_1_score, 0)
        self.assertEqual(app._player_2_score, 29)
        self.assertEqual(app.commands[-1], "SCORE P1 0 P2 29")
        self.assertIn("not valid", app.statuses[-1])

    def test_player_one_wins_score_for_invalid_player_two_word(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._challenge_candidates = [ChallengeCandidate("ZZQ", ("D2", "D3", "D4"), 29, 2)]
        app._current_player = 1
        app._current_player_display_var = _FakeVar("Current Player: 1")
        app._player_1_score = 0
        app._player_2_score = 0
        app._player_1_score_var = _FakeVar("")
        app._player_2_score_var = _FakeVar("")
        app.actuator_port_var = _FakeVar("COM9")
        app.commands = []
        app.logs = []
        app.statuses = []
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._log = app.logs.append

        ScrabblePlotterApp._handle_challenge_choice(app, 0, "ZZQ")

        self.assertEqual(app._player_1_score, 29)
        self.assertEqual(app._player_2_score, 0)
        self.assertEqual(app.commands[-1], "SCORE P1 29 P2 0")

    def test_player_one_ai_automatically_challenges_invalid_player_two_word(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app.commands = []
        app.logs = []
        app.statuses = []
        app.scheduled = []
        app.completed = []
        app.handled = []
        app.root = type(
            "Root",
            (),
            {"after": lambda _self, delay, callback: app.scheduled.append((delay, callback))},
        )()
        app._build_challenge_candidates = lambda limit=100: [
            ChallengeCandidate("ZZQ", ("D2", "D3", "D4"), 29, 2),
        ]
        app._send_actuator_command = lambda command: app.commands.append(command) or ["ok"]
        app._set_status = app.statuses.append
        app._log = app.logs.append
        app._handle_challenge_choice = lambda index, word: app.handled.append((index, word))

        started = ScrabblePlotterApp._start_ai_challenges(
            app,
            ["ZZQ"],
            lambda: app.completed.append(True),
        )

        self.assertTrue(started)
        self.assertEqual(
            app.commands,
            [
                "WORD_CLEAR",
                "WORD_LIST ZZQ",
                "WORD_CELLS 0 D2,D3,D4",
                "CHALLENGE_START",
                "LED_CELLS D2,D3,D4",
            ],
        )
        self.assertEqual(app.scheduled[0][0], AI_CHALLENGE_LIGHT_MS)
        app.scheduled.pop(0)[1]()
        self.assertEqual(app.handled, [(0, "ZZQ")])


    def test_letter_capture_does_not_clear_tile_rack_side_from_board_grid(self) -> None:
        app = object.__new__(ScrabblePlotterApp)
        app._camera_letter_scan_token = 7
        app._live_letter_scan_running = True
        app._last_live_letter_scan_error = "old"
        app.captured_letters_var = _FakeVar("")
        app.statuses = []
        app.logs = []
        app._lock_scan_to_manual_board_grid = lambda scan: None
        app._apply_camera_letter_scan_to_locked_board_grid = lambda scan: 0
        app._apply_camera_ocr_grid_to_board_form = lambda grid: 0
        app._clear_tile_rack_side_from_main_ocr_grid = lambda: (_ for _ in ()).throw(
            AssertionError("tile rack side cleanup should not run during letter capture")
        )
        app._set_status = app.statuses.append
        app._refresh_camera_preview = lambda: None
        app._log = app.logs.append
        scan = CameraLetterScanResult(letters=[], grid=None)

        ScrabblePlotterApp._handle_camera_letter_scan_result(app, scan, announce=True, scan_token=7)

        self.assertFalse(app._live_letter_scan_running)
        self.assertEqual(app.captured_letters_var.get(), "")
        self.assertIn("No letters captured", app.statuses[-1])

class _FakeFrame:
    def __init__(self, shape: tuple[int, ...] = (2, 2, 3)):
        self.shape = shape


class _FakeCapture:
    def __init__(self, opened: bool = True, reads: list[tuple[bool, object | None]] | None = None):
        self.opened = opened
        self.reads = list(reads or [])
        self.released = False
        self.set_calls: list[tuple[int, object]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: object) -> bool:
        self.set_calls.append((prop, value))
        return True


class _FakeCv2:
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_ZOOM = 4

    def __init__(self, captures: list[_FakeCapture]):
        self.captures = list(captures)
        self.open_calls: list[tuple[int, int | None]] = []

    def VideoCapture(self, camera_index: int, backend: int | None = None) -> _FakeCapture:
        self.open_calls.append((camera_index, backend))
        return self.captures.pop(0)

    def VideoWriter_fourcc(self, *characters: str) -> int:
        self.fourcc_call = characters
        return 1234


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


def _synthetic_light_tile_board_image(tiles: list[tuple[str, int, int]]):
    import numpy as np

    cv2 = _require_cv2_for_tests()
    image = np.full((150, 220, 3), 20, dtype=np.uint8)
    tile_size = 40
    for letter, left, top in tiles:
        cv2.rectangle(image, (left, top), (left + tile_size, top + tile_size), (238, 238, 210), -1)
        if letter:
            cv2.putText(
                image,
                letter,
                (left + 11, top + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (20, 20, 20),
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


def _require_tesseract_for_tests():
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise unittest.SkipTest("Tesseract is required for cell OCR tests.") from exc


def _synthetic_letter_cell(letter: str, cv2):
    import numpy as np

    cell_size = 80
    image = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        letter,
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    return image


if __name__ == "__main__":
    unittest.main()
