from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .board import BOARD_SIZE
from .calibration import PlotterCalibration, board_corner_points
from .scoring import normalize_letter, square_label
from .word_bank import (
    DIRECTION_LABELS,
    HORIZONTAL_LEFT_TO_RIGHT,
    VERTICAL_TOP_TO_BOTTOM,
    filter_matching_words,
    format_words_by_direction,
)


OcrReader = Callable[[Any, str], tuple[str, float]]
EasyOcrReader = Callable[[Any], Any]
PaddleOcrReader = EasyOcrReader
_EASY_OCR_READER = None
_EASY_OCR_LOCK = threading.Lock()


@dataclass(frozen=True)
class ScanCell:
    row: int
    col: int
    square: str
    letter: str
    confidence: float
    occupied: bool
    blank: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "square": self.square,
            "letter": self.letter,
            "confidence": self.confidence,
            "occupied": self.occupied,
            "blank": self.blank,
        }


@dataclass(frozen=True)
class BoardScanResult:
    cells: list[ScanCell]
    board_size: int = BOARD_SIZE

    def board_letters(self) -> list[list[str]]:
        board = [["" for _ in range(self.board_size)] for _ in range(self.board_size)]
        for cell in self.cells:
            if 0 <= cell.row < self.board_size and 0 <= cell.col < self.board_size:
                board[cell.row][cell.col] = cell.letter
        return board

    def blank_squares(self) -> set[str]:
        return {cell.square for cell in self.cells if cell.blank}

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "board": self.board_letters(),
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class CapturedLetter:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CameraLetterScanResult:
    letters: list[CapturedLetter]
    grid: CameraOcrGrid | None = None

    def text(self) -> str:
        return " ".join(letter.text for letter in self.letters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text(),
            "letters": [letter.to_dict() for letter in self.letters],
            "grid": self.grid.to_dict() if self.grid is not None else None,
        }


@dataclass(frozen=True)
class CameraWord:
    word: str
    direction: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def direction_label(self) -> str:
        return DIRECTION_LABELS.get(self.direction, self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "direction": self.direction,
            "confidence": self.confidence,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CameraTile:
    letter: str
    confidence: float
    corners: list[tuple[float, float]]

    @property
    def text(self) -> str:
        return self.letter

    @property
    def left(self) -> float:
        return min(point[0] for point in self.corners)

    @property
    def top(self) -> float:
        return min(point[1] for point in self.corners)

    @property
    def right(self) -> float:
        return max(point[0] for point in self.corners)

    @property
    def bottom(self) -> float:
        return max(point[1] for point in self.corners)

    @property
    def width(self) -> float:
        return max(1.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "letter": self.letter,
            "confidence": self.confidence,
            "corners": [[x, y] for x, y in self.corners],
        }


@dataclass(frozen=True)
class CameraGridCell:
    row: int
    col: int
    square: str
    letter: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "square": self.square,
            "letter": self.letter,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class CameraOcrGrid:
    corners: list[tuple[float, float]]
    cells: list[CameraGridCell] = field(default_factory=list)
    board_size: int = BOARD_SIZE

    def board_letters(self) -> list[list[str]]:
        board = [["" for _ in range(self.board_size)] for _ in range(self.board_size)]
        for cell in self.cells:
            if 0 <= cell.row < self.board_size and 0 <= cell.col < self.board_size:
                board[cell.row][cell.col] = cell.letter
        return board

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "corners": [[x, y] for x, y in self.corners],
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class CameraWordScanResult:
    words: list[CameraWord]
    tiles: list[CameraTile] = field(default_factory=list)
    text_boxes: list[EasyOcrTextBox] = field(default_factory=list)
    grid: CameraOcrGrid | None = None

    def text(self) -> str:
        return " ".join(word.word for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text(),
            "words": [word.to_dict() for word in self.words],
            "tiles": [tile.to_dict() for tile in self.tiles],
            "text_boxes": [box.to_dict() for box in self.text_boxes],
            "grid": self.grid.to_dict() if self.grid is not None else None,
        }


@dataclass(frozen=True)
class EasyOcrTextBox:
    text: str
    confidence: float
    points: list[tuple[float, float]]

    @property
    def left(self) -> float:
        return min(point[0] for point in self.points)

    @property
    def top(self) -> float:
        return min(point[1] for point in self.points)

    @property
    def right(self) -> float:
        return max(point[0] for point in self.points)

    @property
    def bottom(self) -> float:
        return max(point[1] for point in self.points)

    @property
    def width(self) -> float:
        return max(1.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "points": [[x, y] for x, y in self.points],
        }


PaddleOcrTextBox = EasyOcrTextBox


@dataclass(frozen=True)
class CameraFrameQuality:
    score: float
    sharpness: float
    contrast: float
    brightness: float


def scan_image_file(
    image_path: str | Path,
    calibration: PlotterCalibration,
    ocr_reader: OcrReader | None = None,
) -> BoardScanResult:
    cv2 = _require_cv2()
    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise ValueError(f"Unable to load image at '{image_path}'.")
    return scan_board_image(image, calibration, ocr_reader=ocr_reader)


def scan_board_image(
    frame,
    calibration: PlotterCalibration,
    ocr_reader: OcrReader | None = None,
) -> BoardScanResult:  # type: ignore[no-untyped-def]
    calibration.validate_ready_for_scan()
    warped = warp_board_image(frame, calibration)
    cell_size = int(calibration.ocr_cell_size_px)
    reader = ocr_reader or read_cell_with_tesseract
    cells: list[ScanCell] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            crop = warped[row * cell_size : (row + 1) * cell_size, col * cell_size : (col + 1) * cell_size]
            occupied = cell_looks_occupied(crop)
            letter = ""
            confidence = 0.0
            if occupied:
                letter, confidence = reader(crop, square_label(row, col))
                letter = normalize_letter(letter)
                if confidence < calibration.ocr_confidence_threshold:
                    letter = ""
            cells.append(
                ScanCell(
                    row=row,
                    col=col,
                    square=square_label(row, col),
                    letter=letter,
                    confidence=float(confidence),
                    occupied=occupied,
                )
            )

    return BoardScanResult(cells=cells)


def score_frame_quality(frame) -> CameraFrameQuality:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return CameraFrameQuality(score=0.0, sharpness=0.0, contrast=0.0, brightness=0.0)

    scale = min(1.0, 720.0 / float(max(width, height)))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean, stddev = cv2.meanStdDev(gray)
    brightness = float(mean[0][0])
    contrast = float(stddev[0][0])
    brightness_factor = max(0.15, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_factor = max(0.15, min(1.0, contrast / 64.0))
    score = sharpness * (0.55 + 0.45 * brightness_factor) * (0.65 + 0.35 * contrast_factor)
    return CameraFrameQuality(
        score=float(score),
        sharpness=sharpness,
        contrast=contrast,
        brightness=brightness,
    )


def select_best_frame(frames: list[Any]) -> tuple[Any, CameraFrameQuality]:
    candidates: list[tuple[CameraFrameQuality, Any]] = []
    for frame in frames:
        if frame is None:
            continue
        candidates.append((score_frame_quality(frame), frame))
    if not candidates:
        raise ValueError("No camera frames were available to choose from.")
    quality, frame = max(candidates, key=lambda item: item[0].score)
    return frame, quality


def scan_camera_letters(
    frame,
    confidence_threshold: float = 50.0,
    ocr_reader: EasyOcrReader | None = None,
    calibration_corners: list | None = None,
) -> CameraLetterScanResult:  # type: ignore[no-untyped-def]
    if ocr_reader is None:
        tiles = detect_camera_tiles(frame, confidence_threshold=min(confidence_threshold, 15.0))
        if tiles:
            return CameraLetterScanResult(
                letters=captured_letters_from_camera_tiles(tiles),
                grid=build_camera_ocr_grid(frame, tiles=tiles, calibration_corners=calibration_corners),
            )

    raw_result = ocr_reader(frame) if ocr_reader is not None else read_frame_with_easyocr(frame)
    boxes = parse_easyocr_text_boxes(raw_result, confidence_threshold=confidence_threshold)
    boxes = camera_character_text_boxes(frame, boxes)
    return CameraLetterScanResult(
        letters=captured_letters_from_text_boxes(boxes),
        grid=build_camera_ocr_grid(frame, text_boxes=boxes, calibration_corners=calibration_corners),
    )


def scan_camera_words(
    frame,
    confidence_threshold: float = 50.0,
    ocr_reader: EasyOcrReader | None = None,
    calibration_corners: list | None = None,
) -> CameraWordScanResult:  # type: ignore[no-untyped-def]
    detected_tiles = (
        detect_camera_tiles(frame, confidence_threshold=min(confidence_threshold, 15.0))
        if ocr_reader is None
        else []
    )
    raw_result = ocr_reader(frame) if ocr_reader is not None else read_frame_with_easyocr(frame)
    boxes = parse_easyocr_text_boxes(raw_result, confidence_threshold=confidence_threshold)
    boxes = camera_character_text_boxes(frame, boxes)
    text_box_tiles = camera_tiles_from_text_boxes(boxes)
    detected_words = filter_matching_words(_dedupe_camera_words(
        identify_directional_tile_words(detected_tiles) + identify_directional_words(boxes)
    ))
    detected_words = _remove_contained_partial_words(detected_words)
    grid = build_camera_ocr_grid(frame, tiles=detected_tiles, text_boxes=boxes, calibration_corners=calibration_corners)
    return CameraWordScanResult(
        words=detected_words,
        tiles=detected_tiles or text_box_tiles,
        text_boxes=boxes,
        grid=grid,
    )


def detect_camera_tiles(
    frame,
    confidence_threshold: float = 50.0,
    ocr_reader: EasyOcrReader | None = None,
) -> list[CameraTile]:  # type: ignore[no-untyped-def]
    tiles: list[CameraTile] = []
    for corners in detect_tile_corners(frame):
        tile_image = warp_tile_image(frame, corners)
        letter, confidence = read_tile_letter_with_easyocr(tile_image, ocr_reader=ocr_reader)
        if letter and confidence >= confidence_threshold:
            tiles.append(CameraTile(letter=letter, confidence=confidence, corners=corners))
    return sorted(tiles, key=lambda tile: (tile.top, tile.left))


def detect_tile_corners(frame) -> list[list[tuple[float, float]]]:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = float(frame.shape[0] * frame.shape[1])
    candidates: list[tuple[float, list[tuple[float, float]]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.0007 or area > frame_area * 0.10:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect)

        score = _tile_quad_score(quad, area)
        if score <= 0:
            continue
        candidates.append((score, order_corners(quad)))

    candidates.extend(_bright_tile_corner_candidates(frame))
    return _dedupe_tile_corners(candidates)


def _bright_tile_corner_candidates(frame) -> list[tuple[float, list[tuple[float, float]]]]:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    mask = cv2.inRange(gray, 120, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = float(frame.shape[0] * frame.shape[1])
    minimum_area = max(180.0, frame_area * 0.0012)
    maximum_area = max(minimum_area + 1.0, frame_area * 0.025)
    candidates: list[tuple[float, list[tuple[float, float]]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area or area > maximum_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect = min(width, height) / max(width, height)
        fill = area / float(width * height)
        if aspect < 0.65 or fill < 0.55:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect)
        candidates.append((area * aspect * fill, order_corners(quad)))

    return candidates


def warp_tile_image(frame, corners: list[tuple[float, float]], size_px: int = 96):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    destination = _to_float32(
        [
            [0.0, 0.0],
            [float(size_px - 1), 0.0],
            [float(size_px - 1), float(size_px - 1)],
            [0.0, float(size_px - 1)],
        ]
    )
    transform = cv2.getPerspectiveTransform(_to_float32(corners), destination)
    return cv2.warpPerspective(frame, transform, (size_px, size_px))


def captured_letters_from_text_boxes(text_boxes: list[EasyOcrTextBox]) -> list[CapturedLetter]:
    letters = [
        CapturedLetter(
            text=box.text,
            confidence=box.confidence,
            left=int(round(box.left)),
            top=int(round(box.top)),
            width=max(1, int(round(box.width))),
            height=max(1, int(round(box.height))),
        )
        for box in text_boxes
    ]
    return sorted(letters, key=lambda letter: (letter.top, letter.left))


def captured_letters_from_camera_tiles(tiles: list[CameraTile]) -> list[CapturedLetter]:
    letters = [
        CapturedLetter(
            text=tile.letter,
            confidence=tile.confidence,
            left=int(round(tile.left)),
            top=int(round(tile.top)),
            width=max(1, int(round(tile.width))),
            height=max(1, int(round(tile.height))),
        )
        for tile in tiles
    ]
    return sorted(letters, key=lambda letter: (letter.top, letter.left))


def camera_tiles_from_text_boxes(text_boxes: list[EasyOcrTextBox]) -> list[CameraTile]:
    tiles = [
        CameraTile(letter=box.text, confidence=box.confidence, corners=box.points)
        for box in text_boxes
        if len(box.text) == 1
    ]
    return sorted(tiles, key=lambda tile: (tile.top, tile.left))


def build_camera_ocr_grid(
    frame,
    tiles: list[CameraTile] | None = None,
    text_boxes: list[EasyOcrTextBox] | None = None,
    board_size: int = BOARD_SIZE,
    calibration_corners: list | None = None,
) -> CameraOcrGrid | None:  # type: ignore[no-untyped-def]
    # Prefer saved calibration corners over auto-detection to avoid grid drift.
    if calibration_corners and len(calibration_corners) == 4:
        corners: list[tuple[float, float]] | None = [
            (float(pt[0]), float(pt[1])) for pt in calibration_corners
        ]
    else:
        corners = detect_board_grid_corners(frame)
    if corners is None:
        return None
    cells = camera_grid_cells_from_ocr(
        corners,
        tiles=tiles or [],
        text_boxes=text_boxes or [],
        board_size=board_size,
    )
    return CameraOcrGrid(corners=corners, cells=cells, board_size=board_size)


def camera_grid_cells_from_ocr(
    corners: list[tuple[float, float]],
    tiles: list[CameraTile] | None = None,
    text_boxes: list[EasyOcrTextBox] | None = None,
    board_size: int = BOARD_SIZE,
) -> list[CameraGridCell]:
    if len(corners) != 4:
        return []

    cv2 = _require_cv2()
    image_to_board = cv2.getPerspectiveTransform(
        _to_float32(corners),
        _to_float32(board_corner_points(board_size)),
    )

    selected: dict[tuple[int, int], CameraGridCell] = {}
    for tile in tiles or []:
        letter = normalize_letter(tile.letter[:1])
        if not letter:
            continue
        board_point = _transform_image_points([(tile.center_x, tile.center_y)], image_to_board)[0]
        cell = _camera_grid_cell_from_board_point(
            board_point,
            letter=letter,
            confidence=tile.confidence,
            source="tile",
            board_size=board_size,
        )
        _store_camera_grid_cell(selected, cell)

    for box in text_boxes or []:
        for cell in _camera_grid_cells_from_text_box(box, image_to_board, board_size):
            _store_camera_grid_cell(selected, cell)

    return [
        selected[key]
        for key in sorted(selected.keys())
    ]


def board_square_from_image_point(
    corners: list[tuple[float, float]] | list[list[float]],
    x: float,
    y: float,
    board_size: int = BOARD_SIZE,
) -> str | None:
    if len(corners) != 4:
        return None

    cv2 = _require_cv2()
    image_to_board = cv2.getPerspectiveTransform(
        _to_float32(corners),
        _to_float32(board_corner_points(board_size)),
    )
    board_point = _transform_image_points([(float(x), float(y))], image_to_board)[0]
    board_x, board_y = board_point
    tolerance = 0.05
    if (
        board_x < -tolerance
        or board_y < -tolerance
        or board_x > board_size + tolerance
        or board_y > board_size + tolerance
    ):
        return None

    col = max(0, min(board_size - 1, int(board_x)))
    row = max(0, min(board_size - 1, int(board_y)))
    return square_label(row, col)


def detect_board_grid_corners(frame) -> list[tuple[float, float]] | None:  # type: ignore[no-untyped-def]
    corners = _detect_board_corners_from_dark_grid(frame)
    if corners is not None:
        return corners
    corners = _detect_board_corners_from_edges(frame)
    if corners is not None:
        return corners
    return _detect_rectangular_board_corners(frame)


def filter_white_text_on_black_background(frame, text_boxes: list[EasyOcrTextBox]) -> list[EasyOcrTextBox]:  # type: ignore[no-untyped-def]
    return [
        box
        for box in text_boxes
        if _looks_like_white_text_on_black_background(frame, box)
    ]


def camera_character_text_boxes(frame, text_boxes: list[EasyOcrTextBox]) -> list[EasyOcrTextBox]:  # type: ignore[no-untyped-def]
    white_on_black = filter_white_text_on_black_background(frame, text_boxes)
    return white_on_black if white_on_black else text_boxes


def read_tile_letter_with_easyocr(
    tile_image,
    ocr_reader: EasyOcrReader | None = None,
) -> tuple[str, float]:  # type: ignore[no-untyped-def]
    inner = _tile_inner_crop(tile_image)
    raw_result = ocr_reader(inner) if ocr_reader is not None else read_frame_with_easyocr(inner)
    boxes = parse_easyocr_text_boxes(raw_result, confidence_threshold=0.0)
    boxes = camera_character_text_boxes(inner, boxes)
    best_letter = ""
    best_confidence = 0.0
    best_score = 0.0
    for box in boxes:
        letter = normalize_letter(box.text[:1])
        if not letter:
            continue
        score = box.confidence * box.width * box.height
        if score > best_score:
            best_letter = letter
            best_confidence = box.confidence
            best_score = score
    return best_letter, best_confidence


def read_tile_letter_with_paddleocr(
    tile_image,
    ocr_reader: PaddleOcrReader | None = None,
) -> tuple[str, float]:  # type: ignore[no-untyped-def]
    return read_tile_letter_with_easyocr(tile_image, ocr_reader=ocr_reader)


def read_frame_with_easyocr(frame):  # type: ignore[no-untyped-def]
    reader = _easy_ocr_reader()
    with _EASY_OCR_LOCK:
        try:
            return reader.readtext(
                frame,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                batch_size=4,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.2,
                decoder="beamsearch",
                beamWidth=5,
            )
        except TypeError:
            return reader.readtext(frame)


def read_frame_with_paddleocr(frame):  # type: ignore[no-untyped-def]
    return read_frame_with_easyocr(frame)


def parse_easyocr_text_boxes(
    raw_result: Any,
    confidence_threshold: float = 50.0,
) -> list[EasyOcrTextBox]:
    boxes: list[EasyOcrTextBox] = []
    for text_box in _iter_easyocr_text_boxes(raw_result):
        if text_box.confidence >= confidence_threshold and text_box.text:
            boxes.append(text_box)
    return boxes


def parse_paddleocr_text_boxes(
    raw_result: Any,
    confidence_threshold: float = 50.0,
) -> list[PaddleOcrTextBox]:
    boxes: list[PaddleOcrTextBox] = []
    for text_box in _iter_paddleocr_text_boxes(raw_result):
        if text_box.confidence >= confidence_threshold and text_box.text:
            boxes.append(text_box)
    return boxes


def identify_directional_words(
    text_boxes: list[EasyOcrTextBox],
    min_word_length: int = 2,
) -> list[CameraWord]:
    words: list[CameraWord] = []

    for box in text_boxes:
        if len(box.text) < min_word_length:
            continue
        if _looks_vertical(box):
            words.append(_camera_word_from_boxes(box.text, VERTICAL_TOP_TO_BOTTOM, [box]))
        else:
            words.append(_camera_word_from_boxes(box.text, HORIZONTAL_LEFT_TO_RIGHT, [box]))

    single_letter_boxes = [box for box in text_boxes if len(box.text) == 1]
    for run in _letter_runs(single_letter_boxes, line_axis="row"):
        if len(run) >= min_word_length:
            ordered = sorted(run, key=lambda item: item.center_x)
            words.append(
                _camera_word_from_boxes(
                    "".join(box.text for box in ordered),
                    HORIZONTAL_LEFT_TO_RIGHT,
                    ordered,
                )
            )
    for run in _letter_runs(single_letter_boxes, line_axis="column"):
        if len(run) >= min_word_length:
            ordered = sorted(run, key=lambda item: item.center_y)
            words.append(
                _camera_word_from_boxes(
                    "".join(box.text for box in ordered),
                    VERTICAL_TOP_TO_BOTTOM,
                    ordered,
                )
            )

    return _dedupe_camera_words(words)


def identify_directional_tile_words(
    tiles: list[CameraTile],
    min_word_length: int = 2,
) -> list[CameraWord]:
    words: list[CameraWord] = []
    letter_tiles = [tile for tile in tiles if len(tile.letter) == 1]
    for run in _letter_runs(letter_tiles, line_axis="row"):
        if len(run) >= min_word_length:
            ordered = sorted(run, key=lambda item: item.center_x)
            words.append(
                _camera_word_from_boxes(
                    "".join(tile.letter for tile in ordered),
                    HORIZONTAL_LEFT_TO_RIGHT,
                    ordered,
                )
            )
    for run in _letter_runs(letter_tiles, line_axis="column"):
        if len(run) >= min_word_length:
            ordered = sorted(run, key=lambda item: item.center_y)
            words.append(
                _camera_word_from_boxes(
                    "".join(tile.letter for tile in ordered),
                    VERTICAL_TOP_TO_BOTTOM,
                    ordered,
                )
            )
    return _dedupe_camera_words(words)


def format_camera_words_numbered(words: list[CameraWord]) -> str:
    return format_words_by_direction(words)


def parse_camera_letter_data(
    data: dict[str, list[Any]],
    confidence_threshold: float = 50.0,
    scale: float = 1.0,
) -> list[CapturedLetter]:
    letters: list[CapturedLetter] = []
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    for index, raw_text in enumerate(texts):
        text = _normalize_ocr_text(raw_text)
        if not text:
            continue
        try:
            confidence = float(confidences[index])
        except (IndexError, TypeError, ValueError):
            confidence = 0.0
        if confidence < confidence_threshold:
            continue

        letters.append(
            CapturedLetter(
                text=text,
                confidence=confidence,
                left=_scaled_int(_value_at(lefts, index), scale),
                top=_scaled_int(_value_at(tops, index), scale),
                width=max(1, _scaled_int(_value_at(widths, index), scale)),
                height=max(1, _scaled_int(_value_at(heights, index), scale)),
            )
        )
    return letters


def detect_board_corners(frame) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    corners = detect_board_grid_corners(frame)
    if corners is None:
        raise ValueError("Could not identify the board area in the camera image.")
    return corners


def _detect_board_corners_from_edges(frame) -> list[tuple[float, float]] | None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(frame.shape[0] * frame.shape[1])
    best_quad = None
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.05:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect)
        score = _quad_score(quad, area)
        if score > best_score:
            best_quad = quad
            best_score = score

    if best_quad is None:
        return None
    return order_corners(best_quad)


def _detect_board_corners_from_dark_grid(frame) -> list[tuple[float, float]] | None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)
    best: tuple[float, list[tuple[float, float]]] | None = None

    for threshold in (25, 35, 45, 55, 70, 90):
        mask = cv2.inRange(gray, 0, threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.06:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = order_corners(approx.reshape(4, 2))
            else:
                corners = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))

            score = _perspective_board_score(corners, area, frame_width, frame_height)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, corners)

    return best[1] if best is not None else None


def _detect_rectangular_board_corners(frame) -> list[tuple[float, float]] | None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for threshold in (30, 45, 60, 80, 100):
        mask = cv2.inRange(gray, 0, threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.06:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue
            aspect = width / float(height)
            fill = area / float(width * height)
            if 0.65 <= aspect <= 1.45 and fill >= 0.35:
                score = area * fill * (min(width, height) / max(width, height))
                if best is None or score > best[0]:
                    best = (score, (x, y, width, height))

    if best is None:
        return None

    x, y, width, height = best[1]
    side = float(min(width, height))
    return [
        (float(x), float(y)),
        (float(x) + side, float(y)),
        (float(x) + side, float(y) + side),
        (float(x), float(y) + side),
    ]


def order_corners(points) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    import numpy as np

    array = np.array(points, dtype="float32").reshape(4, 2)
    sums = array.sum(axis=1)
    diffs = np.diff(array, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = array[int(sums.argmin())]
    ordered[2] = array[int(sums.argmax())]
    ordered[1] = array[int(diffs.argmin())]
    ordered[3] = array[int(diffs.argmax())]
    return [(float(x), float(y)) for x, y in ordered]


def _camera_grid_cells_from_text_box(
    box: EasyOcrTextBox,
    image_to_board,
    board_size: int,
) -> list[CameraGridCell]:  # type: ignore[no-untyped-def]
    letters = [normalize_letter(character) for character in box.text]
    letters = [letter for letter in letters if letter]
    if not letters:
        return []

    if len(letters) == 1:
        board_point = _transform_image_points([(box.center_x, box.center_y)], image_to_board)[0]
        cell = _camera_grid_cell_from_board_point(
            board_point,
            letter=letters[0],
            confidence=box.confidence,
            source="text",
            board_size=board_size,
        )
        return [cell] if cell is not None else []

    board_points = _transform_image_points(box.points, image_to_board)
    xs = [point[0] for point in board_points]
    ys = [point[1] for point in board_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    cells: list[CameraGridCell] = []

    if span_y > span_x * 1.15:
        board_x = sum(xs) / len(xs)
        step = span_y / len(letters)
        for index, letter in enumerate(letters):
            board_y = min_y + (index + 0.5) * step
            cell = _camera_grid_cell_from_board_point(
                (board_x, board_y),
                letter=letter,
                confidence=box.confidence,
                source="word",
                board_size=board_size,
            )
            if cell is not None:
                cells.append(cell)
    else:
        board_y = sum(ys) / len(ys)
        step = span_x / len(letters)
        for index, letter in enumerate(letters):
            board_x = min_x + (index + 0.5) * step
            cell = _camera_grid_cell_from_board_point(
                (board_x, board_y),
                letter=letter,
                confidence=box.confidence,
                source="word",
                board_size=board_size,
            )
            if cell is not None:
                cells.append(cell)

    return cells


def _camera_grid_cell_from_board_point(
    board_point: tuple[float, float],
    letter: str,
    confidence: float,
    source: str,
    board_size: int,
) -> CameraGridCell | None:
    board_x, board_y = board_point
    tolerance = 0.35
    if (
        board_x < -tolerance
        or board_y < -tolerance
        or board_x >= board_size + tolerance
        or board_y >= board_size + tolerance
    ):
        return None

    col = max(0, min(board_size - 1, int(board_x)))
    row = max(0, min(board_size - 1, int(board_y)))
    return CameraGridCell(
        row=row,
        col=col,
        square=square_label(row, col),
        letter=letter,
        confidence=float(confidence),
        source=source,
    )


def _store_camera_grid_cell(
    selected: dict[tuple[int, int], CameraGridCell],
    cell: CameraGridCell | None,
) -> None:
    if cell is None:
        return
    key = (cell.row, cell.col)
    existing = selected.get(key)
    if existing is None or _camera_grid_cell_score(cell) >= _camera_grid_cell_score(existing):
        selected[key] = cell


def _camera_grid_cell_score(cell: CameraGridCell) -> float:
    priority = {"tile": 8.0, "text": 4.0, "word": 0.0}.get(cell.source, 0.0)
    return float(cell.confidence) + priority


def _transform_image_points(points: list[tuple[float, float]], transform) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    transformed = cv2.perspectiveTransform(
        _to_float32([[list(point) for point in points]]),
        transform,
    )
    return [(float(point[0]), float(point[1])) for point in transformed[0]]


def warp_board_image(frame, calibration: PlotterCalibration):  # type: ignore[no-untyped-def]
    calibration.validate_ready_for_scan()
    cv2 = _require_cv2()
    width = calibration.board_size * int(calibration.ocr_cell_size_px)
    destination = _to_float32(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(width)],
            [0.0, float(width)],
        ]
    )
    transform = cv2.getPerspectiveTransform(_to_float32(calibration.image_corners), destination)
    return cv2.warpPerspective(frame, transform, (width, width))


def cell_looks_occupied(cell_image) -> bool:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = _cell_gray_inner(cell_image, margin_ratio=0.18)
    if gray.size == 0:
        return False
        
    # Stretch contrast within the cell to handle glare without distorting letters
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 180)
    edge_density = float((edges > 0).sum()) / float(edges.size)
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_density = float((dark_mask > 0).sum()) / float(dark_mask.size)
    return edge_density > 0.018 or dark_density > 0.20


def read_cell_with_tesseract(cell_image, square: str) -> tuple[str, float]:  # type: ignore[no-untyped-def]
    try:
        pytesseract = _require_pytesseract()
        processed = preprocess_cell_for_ocr(cell_image)
        config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        return parse_tesseract_data(data)
    except Exception as exc:
        if "tesseract is not installed" in str(exc).lower() or "not in your path" in str(exc).lower():
            return read_tile_letter_with_easyocr(cell_image)
        raise


def parse_tesseract_data(data: dict[str, list[Any]]) -> tuple[str, float]:
    best_letter = ""
    best_confidence = 0.0
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    for raw_text, raw_confidence in zip(texts, confidences):
        letter = normalize_letter(str(raw_text).strip()[:1])
        if not letter:
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > best_confidence:
            best_letter = letter
            best_confidence = confidence
    return best_letter, best_confidence


def preprocess_cell_for_ocr(cell_image):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = _cell_gray_inner(cell_image, margin_ratio=0.12)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )


def preprocess_frame_for_ocr(frame):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    height, width = gray.shape[:2]
    scale = max(1.0, 1200.0 / float(width)) if width > 0 else 1.0
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return processed, scale


def _cell_gray_inner(cell_image, margin_ratio: float):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    if len(cell_image.shape) == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image
    height, width = gray.shape[:2]
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    return gray[margin_y : height - margin_y, margin_x : width - margin_x]


def _tile_inner_crop(tile_image, margin_ratio: float = 0.14):  # type: ignore[no-untyped-def]
    height, width = tile_image.shape[:2]
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    return tile_image[margin_y : height - margin_y, margin_x : width - margin_x]


def _normalize_ocr_text(raw_text: Any) -> str:
    text = str(raw_text).strip()
    if not text:
        return ""

    normalized: list[str] = []
    for char in text:
        if char == "0":
            normalized.append("O")
            continue
        if char in {"1", "|", "!", "]", "["}:
            normalized.append("I")
            continue
        if char == "l" and len(text) == 1:
            normalized.append("I")
            continue
        upper = char.upper()
        if "A" <= upper <= "Z":
            normalized.append(upper)
    return "".join(normalized)


def _iter_easyocr_text_boxes(raw_result: Any):
    line = _parse_easyocr_line(raw_result)
    if line is not None:
        yield line
        return

    if isinstance(raw_result, (list, tuple)):
        for item in raw_result:
            yield from _iter_easyocr_text_boxes(item)


def _parse_easyocr_line(value: Any) -> EasyOcrTextBox | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None

    points = _coerce_point_box(value[0])
    if not points:
        return None

    text = _normalize_ocr_text(value[1])
    if not text:
        return None

    return EasyOcrTextBox(
        text=text,
        confidence=_confidence_percent(value[2]),
        points=points,
    )


def _iter_paddleocr_text_boxes(raw_result: Any):
    v3_payload = _paddleocr_result_payload(raw_result)
    if v3_payload is not None:
        yield from _iter_v3_paddleocr_text_boxes(v3_payload)
        return

    line = _parse_v2_paddleocr_line(raw_result)
    if line is not None:
        yield line
        return

    if isinstance(raw_result, (list, tuple)):
        for item in raw_result:
            yield from _iter_paddleocr_text_boxes(item)


def _iter_v3_paddleocr_text_boxes(payload: dict[str, Any]):
    texts = _payload_value(payload, ("rec_texts", "texts"))
    scores = _payload_value(payload, ("rec_scores", "scores"))
    polys = _payload_value(payload, ("rec_polys", "dt_polys", "polys"))
    rects = _payload_value(payload, ("rec_boxes", "boxes"))

    for index, raw_text in enumerate(_list_like(texts)):
        text = _normalize_ocr_text(raw_text)
        if not text:
            continue
        points = _coerce_point_box(_value_at(_list_like(polys), index))
        if not points:
            points = _coerce_rect_box(_value_at(_list_like(rects), index))
        if not points:
            continue
        yield PaddleOcrTextBox(
            text=text,
            confidence=_confidence_percent(_value_at(_list_like(scores), index)),
            points=points,
        )


def _parse_v2_paddleocr_line(value: Any) -> PaddleOcrTextBox | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None

    points = _coerce_point_box(value[0])
    if not points:
        return None

    text_payload = value[1]
    if not isinstance(text_payload, (list, tuple)) or len(text_payload) < 2:
        return None

    text = _normalize_ocr_text(text_payload[0])
    if not text:
        return None

    return PaddleOcrTextBox(
        text=text,
        confidence=_confidence_percent(text_payload[1]),
        points=points,
    )


def _paddleocr_result_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        payload = value.get("res", value)
        return payload if isinstance(payload, dict) else None

    for attribute in ("json", "to_dict"):
        if not hasattr(value, attribute):
            continue
        payload = getattr(value, attribute)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict):
            payload = payload.get("res", payload)
            return payload if isinstance(payload, dict) else None
    return None


def _letter_runs(boxes: list[Any], line_axis: str) -> list[list[Any]]:
    if not boxes:
        return []

    if line_axis == "row":
        sorted_boxes = sorted(boxes, key=lambda box: box.center_y)
        tolerance = _median([box.height for box in boxes]) * 0.65
        position = lambda box: box.center_y
        run_sort = lambda box: box.center_x
        edge_gap = lambda current, previous: max(0.0, current.left - previous.right)
        gap_limit = max(24.0, _median([box.width for box in boxes]) * 1.6)
    else:
        sorted_boxes = sorted(boxes, key=lambda box: box.center_x)
        tolerance = _median([box.width for box in boxes]) * 0.65
        position = lambda box: box.center_x
        run_sort = lambda box: box.center_y
        edge_gap = lambda current, previous: max(0.0, current.top - previous.bottom)
        gap_limit = max(24.0, _median([box.height for box in boxes]) * 1.6)

    tolerance = max(8.0, tolerance)
    clusters: list[list[Any]] = []
    for box in sorted_boxes:
        if not clusters:
            clusters.append([box])
            continue
        cluster = clusters[-1]
        cluster_center = sum(position(item) for item in cluster) / len(cluster)
        if abs(position(box) - cluster_center) <= tolerance:
            cluster.append(box)
        else:
            clusters.append([box])

    runs: list[list[Any]] = []
    for cluster in clusters:
        ordered = sorted(cluster, key=run_sort)
        current: list[Any] = []
        previous: Any | None = None
        for box in ordered:
            if previous is not None and edge_gap(box, previous) > gap_limit:
                if current:
                    runs.append(current)
                current = []
            current.append(box)
            previous = box
        if current:
            runs.append(current)
    return runs


def _remove_contained_partial_words(words: list[CameraWord]) -> list[CameraWord]:
    kept: list[CameraWord] = []
    for word in sorted(words, key=lambda item: (-len(item.word), item.top, item.left, item.word)):
        if _is_contained_partial_word(word, kept):
            continue
        kept.append(word)
    return sorted(kept, key=lambda item: (_direction_sort_rank(item.direction), item.top, item.left, item.word))


def _is_contained_partial_word(candidate: CameraWord, longer_words: list[CameraWord]) -> bool:
    candidate_word = candidate.word
    for longer in longer_words:
        if candidate.direction != longer.direction:
            continue
        if len(candidate_word) >= len(longer.word):
            continue
        if candidate_word not in longer.word:
            continue
        if _word_boxes_same_line(candidate, longer) and _word_axis_contains(longer, candidate):
            return True
    return False


def _word_boxes_same_line(first: CameraWord, second: CameraWord) -> bool:
    if first.direction == HORIZONTAL_LEFT_TO_RIGHT:
        first_center = first.top + first.height / 2.0
        second_center = second.top + second.height / 2.0
        tolerance = max(first.height, second.height) * 0.75
        return abs(first_center - second_center) <= max(8.0, tolerance)

    first_center = first.left + first.width / 2.0
    second_center = second.left + second.width / 2.0
    tolerance = max(first.width, second.width) * 0.75
    return abs(first_center - second_center) <= max(8.0, tolerance)


def _word_axis_contains(container: CameraWord, contained: CameraWord) -> bool:
    tolerance = 8.0
    if container.direction == HORIZONTAL_LEFT_TO_RIGHT:
        container_start = float(container.left) - tolerance
        container_end = float(container.left + container.width) + tolerance
        contained_start = float(contained.left)
        contained_end = float(contained.left + contained.width)
    else:
        container_start = float(container.top) - tolerance
        container_end = float(container.top + container.height) + tolerance
        contained_start = float(contained.top)
        contained_end = float(contained.top + contained.height)
    return contained_start >= container_start and contained_end <= container_end


def _camera_word_from_boxes(word: str, direction: str, boxes: list[EasyOcrTextBox]) -> CameraWord:
    normalized_word = _normalize_ocr_text(word)
    left = min(box.left for box in boxes)
    top = min(box.top for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    confidence = sum(box.confidence for box in boxes) / len(boxes)
    return CameraWord(
        word=normalized_word,
        direction=direction,
        confidence=confidence,
        left=int(round(left)),
        top=int(round(top)),
        width=max(1, int(round(right - left))),
        height=max(1, int(round(bottom - top))),
    )


def _dedupe_camera_words(words: list[CameraWord]) -> list[CameraWord]:
    deduped: list[CameraWord] = []
    seen: set[tuple[str, str, int, int]] = set()
    for word in sorted(words, key=lambda item: (_direction_sort_rank(item.direction), item.top, item.left, item.word)):
        if len(word.word) < 2:
            continue
        key = (
            word.word,
            word.direction,
            int(round((word.left + word.width / 2) / 20.0)),
            int(round((word.top + word.height / 2) / 20.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(word)
    return deduped


def _direction_sort_rank(direction: str) -> int:
    if direction == HORIZONTAL_LEFT_TO_RIGHT:
        return 0
    return 1


def _looks_vertical(box: EasyOcrTextBox) -> bool:
    return box.height > box.width * 1.15


def _looks_like_white_text_on_black_background(frame, box: EasyOcrTextBox) -> bool:  # type: ignore[no-untyped-def]
    values = _masked_grayscale_values(frame, box.points)
    if not values:
        return False

    total = float(len(values))
    dark_ratio = sum(1 for value in values if value <= 90) / total
    bright_ratio = sum(1 for value in values if value >= 170) / total
    p05 = _percentile(values, 0.05)
    p95 = _percentile(values, 0.95)
    contrast = p95 - p05

    return (
        dark_ratio >= 0.35
        and 0.006 <= bright_ratio <= 0.55
        and dark_ratio > bright_ratio
        and contrast >= 80
    )


def _masked_grayscale_values(frame, points: list[tuple[float, float]]) -> list[int]:  # type: ignore[no-untyped-def]
    if not points:
        return []

    cv2 = _require_cv2()

    height, width = frame.shape[:2]
    raw_left = float(min(point[0] for point in points))
    raw_top = float(min(point[1] for point in points))
    raw_right = float(max(point[0] for point in points))
    raw_bottom = float(max(point[1] for point in points))
    pad_x = max(3, int(round((raw_right - raw_left) * 0.20)))
    pad_y = max(3, int(round((raw_bottom - raw_top) * 0.20)))
    left = max(0, int(raw_left) - pad_x)
    top = max(0, int(raw_top) - pad_y)
    right = min(width, int(raw_right) + pad_x + 1)
    bottom = min(height, int(raw_bottom) + pad_y + 1)
    if right <= left or bottom <= top:
        return []

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    values = gray.reshape(-1)
    if hasattr(values, "tolist"):
        return [int(value) for value in values.tolist()]
    return [int(value) for value in values]


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _coerce_point_box(value: Any) -> list[tuple[float, float]]:
    raw_points = _to_plain_value(value)
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 4:
        return []

    points: list[tuple[float, float]] = []
    for point in raw_points[:4]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return []
    return points


def _coerce_rect_box(value: Any) -> list[tuple[float, float]]:
    raw_rect = _to_plain_value(value)
    if not isinstance(raw_rect, (list, tuple)) or len(raw_rect) < 4:
        return []
    try:
        left, top, right, bottom = [float(raw_rect[index]) for index in range(4)]
    except (TypeError, ValueError):
        return []
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _payload_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return []


def _list_like(value: Any) -> list[Any]:
    plain = _to_plain_value(value)
    if isinstance(plain, list):
        return plain
    if isinstance(plain, tuple):
        return list(plain)
    return []


def _to_plain_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _confidence_percent(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if 0.0 < confidence <= 1.0:
        confidence *= 100.0
    return min(100.0, max(0.0, confidence))


def _median(values: list[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _value_at(values: list[Any], index: int) -> Any:
    try:
        return values[index]
    except IndexError:
        return 0


def _scaled_int(value: Any, scale: float) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    if scale <= 0:
        scale = 1.0
    return int(round(numeric_value / scale))


def _to_float32(values):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.array(values, dtype=np.float32)


def _perspective_board_score(
    corners: list[tuple[float, float]],
    contour_area: float,
    frame_width: int,
    frame_height: int,
) -> float:
    area = abs(_polygon_area(corners))
    if area <= 0:
        return 0.0
    frame_area = float(frame_width * frame_height)
    if area < frame_area * 0.06 or area > frame_area * 0.96:
        return 0.0

    sides = [
        _distance(corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ]
    shortest = min(sides)
    longest = max(sides)
    if shortest <= 0 or longest / shortest > 2.2:
        return 0.0

    fill = min(contour_area, area) / max(contour_area, area)
    if fill < 0.30:
        return 0.0
    return area * fill * (shortest / longest)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return total / 2.0


def _quad_score(quad, contour_area: float) -> float:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    area = abs(float(cv2.contourArea(quad)))
    if area <= 0:
        return 0.0
    x, y, width, height = cv2.boundingRect(_to_float32(quad).astype("int32"))
    if width <= 0 or height <= 0:
        return 0.0
    aspect = min(width, height) / max(width, height)
    fill = min(contour_area, area) / max(contour_area, area)
    return area * max(0.2, aspect) * max(0.2, fill)


def _tile_quad_score(quad, contour_area: float) -> float:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    area = abs(float(cv2.contourArea(quad)))
    if area <= 0:
        return 0.0

    sides = _quad_side_lengths(quad)
    shortest = min(sides)
    longest = max(sides)
    if shortest < 18.0 or longest / shortest > 1.8:
        return 0.0

    x, y, width, height = cv2.boundingRect(_to_float32(quad).astype("int32"))
    if width <= 0 or height <= 0:
        return 0.0
    aspect = min(width, height) / max(width, height)
    if aspect < 0.55:
        return 0.0

    fill = min(contour_area, area) / max(contour_area, area)
    if fill < 0.40:
        return 0.0

    return area * aspect * fill


def _dedupe_tile_corners(
    candidates: list[tuple[float, list[tuple[float, float]]]],
) -> list[list[tuple[float, float]]]:
    selected: list[tuple[float, list[tuple[float, float]]]] = []
    for score, corners in sorted(candidates, key=lambda item: item[0], reverse=True):
        center = _quad_center(corners)
        average_side = _average_quad_side(corners)
        duplicate = False
        for _, existing in selected:
            existing_center = _quad_center(existing)
            distance = _distance(center, existing_center)
            if distance < max(12.0, min(average_side, _average_quad_side(existing)) * 0.55):
                duplicate = True
                break
        if not duplicate:
            selected.append((score, corners))

    return [
        corners
        for _, corners in sorted(selected, key=lambda item: (_quad_center(item[1])[1], _quad_center(item[1])[0]))
    ]


def _quad_side_lengths(points) -> list[float]:  # type: ignore[no-untyped-def]
    ordered = order_corners(points)
    return [
        _distance(ordered[index], ordered[(index + 1) % 4])
        for index in range(4)
    ]


def _average_quad_side(points: list[tuple[float, float]]) -> float:
    sides = _quad_side_lengths(points)
    return sum(sides) / len(sides)


def _quad_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for board scanning. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2


def _require_pytesseract():
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for offline OCR. Install scrabble_plotter/requirements.txt."
        ) from exc
    return pytesseract


def _easy_ocr_reader():
    global _EASY_OCR_READER
    if _EASY_OCR_READER is None:
        easyocr = _require_easyocr()
        _EASY_OCR_READER = easyocr.Reader(["en"], gpu=True)
    return _EASY_OCR_READER


def _paddle_ocr_engine():
    return _easy_ocr_reader()


def _require_easyocr():
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is required for camera letter and word detection. "
            "Install scrabble_plotter/requirements.txt, then start the camera again."
        ) from exc
    return easyocr


def _require_paddleocr():
    return _require_easyocr()
