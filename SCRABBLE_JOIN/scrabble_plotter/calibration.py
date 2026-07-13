from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .board import BOARD_SIZE, CELL_SIZE_MM, Square


PREMIUM_NORMAL = "normal"
PREMIUM_DOUBLE_LETTER = "double_letter"
PREMIUM_TRIPLE_LETTER = "triple_letter"
PREMIUM_DOUBLE_WORD = "double_word"
PREMIUM_TRIPLE_WORD = "triple_word"
VALID_PREMIUM_CODES = {
    PREMIUM_NORMAL,
    PREMIUM_DOUBLE_LETTER,
    PREMIUM_TRIPLE_LETTER,
    PREMIUM_DOUBLE_WORD,
    PREMIUM_TRIPLE_WORD,
}
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 50.0
DEFAULT_OCR_CELL_SIZE_PX = 80
DEFAULT_ACTUATOR_BAUD = 115200
DEFAULT_ACTUATOR_TIMEOUT = 2.0
DEFAULT_ACTUATOR_COUNTDOWN_SECONDS = 30
MACHINE_LABEL_ORIENTATION_A1_TOP_LEFT = "A1-top-left"
MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT = "A1-top-right"
VALID_MACHINE_LABEL_ORIENTATIONS = {
    MACHINE_LABEL_ORIENTATION_A1_TOP_LEFT,
    MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT,
}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for camera calibration. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2


@dataclass
class PlotterCalibration:
    board_size: int = BOARD_SIZE
    label_orientation: str = "A1-top-left"
    machine_label_orientation: str = MACHINE_LABEL_ORIENTATION_A1_TOP_LEFT
    image_path: str | None = None
    image_corners: list[list[float]] = field(default_factory=list)
    camera_index: int = 0
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    cell_size_mm: float = CELL_SIZE_MM
    cell_margin_mm: float = 0.0
    x_steps_per_mm: float = 80.0
    y_steps_per_mm: float = 80.0
    cart_x_mm: float = 0.0
    cart_y_mm: float = 0.0
    tile_cart_pitch_x_mm: float = 16.0
    tile_cart_pitch_y_mm: float = 16.0
    tile_rack_x_mm: float = 335.0
    tile_rack_y_mm: float = 30.0
    tile_rack_pitch_mm: float = 10.0
    premium_layout: list[list[str]] = field(default_factory=lambda: default_premium_layout())
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD
    ocr_cell_size_px: int = DEFAULT_OCR_CELL_SIZE_PX
    actuator_port: str = ""
    actuator_baud: int = DEFAULT_ACTUATOR_BAUD
    actuator_timeout: float = DEFAULT_ACTUATOR_TIMEOUT
    actuator_countdown_seconds: int = DEFAULT_ACTUATOR_COUNTDOWN_SECONDS
    tile_cart_url: str = "http://192.168.4.1"
    tile_cart_player_1_command: str = "backward"
    tile_cart_player_2_command: str = "forward"
    tile_cart_speed_pwm: int = 200
    tile_cart_distance_cm: float = 5.0

    def set_image_corners(
        self,
        image_path: str,
        corners: list[tuple[float, float]],
        calibration_path: str | Path | None = None,
    ) -> None:
        self.set_camera_corners(corners)
        self.image_path = path_for_storage(image_path, calibration_path)

    def set_camera_corners(self, corners: list[tuple[float, float]]) -> None:
        if len(corners) != 4:
            raise ValueError("Exactly 4 board corners are required.")
        self.image_corners = [[float(x), float(y)] for x, y in corners]

    def set_plotter_offset(self, offset_x_mm: float, offset_y_mm: float) -> None:
        self.offset_x_mm = float(offset_x_mm)
        self.offset_y_mm = float(offset_y_mm)

    def validate_ready_for_move(self) -> None:
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"Calibration board size must be {BOARD_SIZE}.")
        if self.cell_size_mm <= 0:
            raise ValueError("Cell size must be greater than 0.")
        if self.x_steps_per_mm <= 0 or self.y_steps_per_mm <= 0:
            raise ValueError("Stepper scale must be greater than 0.")
        if self.machine_label_orientation not in VALID_MACHINE_LABEL_ORIENTATIONS:
            expected = ", ".join(sorted(VALID_MACHINE_LABEL_ORIENTATIONS))
            raise ValueError(f"Machine label orientation must be one of: {expected}.")

    def validate_ready_for_scan(self) -> None:
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"Calibration board size must be {BOARD_SIZE}.")
        if len(self.image_corners) != 4:
            raise ValueError("Camera board calibration is missing.")
        if self.ocr_cell_size_px <= 0:
            raise ValueError("OCR cell size must be greater than 0.")
        if self.ocr_confidence_threshold < 0 or self.ocr_confidence_threshold > 100:
            raise ValueError("OCR confidence threshold must be from 0 to 100.")

    def square_center_in_image(self, square: Square) -> tuple[float, float]:
        if len(self.image_corners) != 4:
            raise ValueError("Camera board calibration is missing.")

        cv2 = _require_cv2()
        transform = cv2.getPerspectiveTransform(
            _to_float32(board_corner_points(self.board_size)),
            _to_float32(self.image_corners),
        )
        center = _to_float32([[list(square.center_in_board_space())]])
        transformed = cv2.perspectiveTransform(center, transform)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    def square_center_in_machine(self, square: Square) -> tuple[float, float]:
        pitch_mm = self.cell_size_mm + self.cell_margin_mm
        machine_col = self._machine_col_for_square(square)
        x = self.offset_x_mm + machine_col * pitch_mm + self.cell_size_mm / 2.0
        y = self.offset_y_mm + square.row * pitch_mm + self.cell_size_mm / 2.0
        return (x, y)

    def _machine_col_for_square(self, square: Square) -> int:
        if self.machine_label_orientation == MACHINE_LABEL_ORIENTATION_A1_TOP_RIGHT:
            return self.board_size - 1 - square.col
        if self.machine_label_orientation == MACHINE_LABEL_ORIENTATION_A1_TOP_LEFT:
            return square.col
        expected = ", ".join(sorted(VALID_MACHINE_LABEL_ORIENTATIONS))
        raise ValueError(f"Machine label orientation must be one of: {expected}.")

    def cart_position_in_machine(self) -> tuple[float, float]:
        return (self.cart_x_mm, self.cart_y_mm)

    def tile_cart_position(self, index: int) -> tuple[float, float]:
        if index < 1 or index > 16:
            raise ValueError(f"Tile cart index must be between 1 and 16, got {index}")
        row = (index - 1) // 4
        col = (index - 1) % 4
        return (
            self.cart_x_mm + col * self.tile_cart_pitch_x_mm,
            self.cart_y_mm + row * self.tile_cart_pitch_y_mm,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "label_orientation": self.label_orientation,
            "machine_label_orientation": self.machine_label_orientation,
            "image_path": self.image_path,
            "image_corners": self.image_corners,
            "camera_index": self.camera_index,
            "offset_x_mm": self.offset_x_mm,
            "offset_y_mm": self.offset_y_mm,
            "cell_size_mm": self.cell_size_mm,
            "x_steps_per_mm": self.x_steps_per_mm,
            "y_steps_per_mm": self.y_steps_per_mm,
            "cart_x_mm": self.cart_x_mm,
            "cart_y_mm": self.cart_y_mm,
            "tile_cart_pitch_x_mm": self.tile_cart_pitch_x_mm,
            "tile_cart_pitch_y_mm": self.tile_cart_pitch_y_mm,
            "tile_rack_x_mm": self.tile_rack_x_mm,
            "tile_rack_y_mm": self.tile_rack_y_mm,
            "tile_rack_pitch_mm": self.tile_rack_pitch_mm,
            "premium_layout": normalize_premium_layout(self.premium_layout, self.board_size),
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "ocr_cell_size_px": self.ocr_cell_size_px,
            "actuator_port": self.actuator_port,
            "actuator_baud": self.actuator_baud,
            "actuator_timeout": self.actuator_timeout,
            "actuator_countdown_seconds": self.actuator_countdown_seconds,
            "tile_cart_url": self.tile_cart_url,
            "tile_cart_player_1_command": self.tile_cart_player_1_command,
            "tile_cart_player_2_command": self.tile_cart_player_2_command,
            "tile_cart_speed_pwm": self.tile_cart_speed_pwm,
            "tile_cart_distance_cm": self.tile_cart_distance_cm,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlotterCalibration":
        return cls(
            board_size=int(payload.get("board_size", BOARD_SIZE)),
            label_orientation=payload.get("label_orientation", "A1-top-left"),
            machine_label_orientation=payload.get(
                "machine_label_orientation",
                payload.get("label_orientation", MACHINE_LABEL_ORIENTATION_A1_TOP_LEFT),
            ),
            image_path=payload.get("image_path"),
            image_corners=payload.get("image_corners", []),
            camera_index=int(payload.get("camera_index", 0)),
            offset_x_mm=float(payload.get("offset_x_mm", 0.0)),
            offset_y_mm=float(payload.get("offset_y_mm", 0.0)),
            cell_size_mm=float(payload.get("cell_size_mm", CELL_SIZE_MM)),
            x_steps_per_mm=float(payload.get("x_steps_per_mm", 80.0)),
            y_steps_per_mm=float(payload.get("y_steps_per_mm", 80.0)),
            cart_x_mm=float(payload.get("cart_x_mm", 0.0)),
            cart_y_mm=float(payload.get("cart_y_mm", 0.0)),
            tile_cart_pitch_x_mm=float(payload.get("tile_cart_pitch_x_mm", 16.0)),
            tile_cart_pitch_y_mm=float(payload.get("tile_cart_pitch_y_mm", 16.0)),
            tile_rack_x_mm=float(payload.get("tile_rack_x_mm", 335.0)),
            tile_rack_y_mm=float(payload.get("tile_rack_y_mm", 30.0)),
            tile_rack_pitch_mm=float(payload.get("tile_rack_pitch_mm", 10.0)),
            premium_layout=normalize_premium_layout(
                payload.get("premium_layout", default_premium_layout()),
                int(payload.get("board_size", BOARD_SIZE)),
            ),
            ocr_confidence_threshold=float(
                payload.get("ocr_confidence_threshold", DEFAULT_OCR_CONFIDENCE_THRESHOLD)
            ),
            ocr_cell_size_px=int(payload.get("ocr_cell_size_px", DEFAULT_OCR_CELL_SIZE_PX)),
            actuator_port=str(payload.get("actuator_port") or ""),
            actuator_baud=int(payload.get("actuator_baud", DEFAULT_ACTUATOR_BAUD)),
            actuator_timeout=float(payload.get("actuator_timeout", DEFAULT_ACTUATOR_TIMEOUT)),
            actuator_countdown_seconds=int(
                payload.get("actuator_countdown_seconds", DEFAULT_ACTUATOR_COUNTDOWN_SECONDS)
            ),
            tile_cart_url=str(payload.get("tile_cart_url") or "http://192.168.4.1"),
            tile_cart_player_1_command=str(payload.get("tile_cart_player_1_command") or "backward"),
            tile_cart_player_2_command=str(payload.get("tile_cart_player_2_command") or "forward"),
            tile_cart_speed_pwm=int(payload.get("tile_cart_speed_pwm", 200)),
            tile_cart_distance_cm=float(payload.get("tile_cart_distance_cm", 5.0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PlotterCalibration":
        calibration_path = Path(path)
        if not calibration_path.exists():
            return cls()
        return cls.from_dict(json.loads(calibration_path.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        calibration_path = Path(path)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )


def board_corner_points(board_size: int = BOARD_SIZE) -> list[list[float]]:
    size = float(board_size)
    return [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]]


def default_premium_layout(board_size: int = BOARD_SIZE) -> list[list[str]]:
    layout = [[PREMIUM_NORMAL for _ in range(board_size)] for _ in range(board_size)]
    if board_size == 12:
        # TW (Triple Word)
        for r, c in [(0, 0), (0, 11), (11, 0), (11, 11), (0, 5), (0, 6), (5, 0), (6, 0), (11, 5), (11, 6), (5, 11), (6, 11)]:
            layout[r][c] = PREMIUM_TRIPLE_WORD
        
        # DW (Double Word)
        for r, c in [(1, 1), (2, 2), (3, 3), (4, 4), (1, 10), (2, 9), (3, 8), (4, 7), (10, 1), (9, 2), (8, 3), (7, 4), (10, 10), (9, 9), (8, 8), (7, 7)]:
            layout[r][c] = PREMIUM_DOUBLE_WORD

        # TL (Triple Letter)
        for r, c in [(1, 5), (1, 6), (5, 1), (6, 1), (10, 5), (10, 6), (5, 10), (6, 10), (5, 5), (6, 6), (5, 6), (6, 5)]:
            layout[r][c] = PREMIUM_TRIPLE_LETTER

        # DL (Double Letter)
        for r, c in [(0, 3), (0, 8), (3, 0), (8, 0), (11, 3), (11, 8), (3, 11), (8, 11), (2, 6), (2, 5), (3, 3), (3, 8), (8, 3), (8, 8), (6, 2), (5, 2), (6, 9), (5, 9)]:
            layout[r][c] = PREMIUM_DOUBLE_LETTER

    return layout


def normalize_premium_layout(payload: Any, board_size: int = BOARD_SIZE) -> list[list[str]]:
    if board_size != BOARD_SIZE:
        board_size = BOARD_SIZE

    if not isinstance(payload, list):
        return default_premium_layout(board_size)

    layout = default_premium_layout(board_size)
    for row_index in range(min(board_size, len(payload))):
        row = payload[row_index]
        if not isinstance(row, list):
            continue
        for col_index in range(min(board_size, len(row))):
            code = str(row[col_index])
            layout[row_index][col_index] = code if code in VALID_PREMIUM_CODES else PREMIUM_NORMAL
    return layout


def _to_float32(values: list[list[float]] | list[list[list[float]]]):
    import numpy as np

    return np.array(values, dtype=np.float32)


def resolve_stored_path(path: str | Path, calibration_path: str | Path | None = None) -> Path:
    stored_path = Path(path).expanduser()
    if stored_path.is_absolute() or calibration_path is None:
        return stored_path.resolve()
    return (Path(calibration_path).expanduser().parent / stored_path).resolve()


def path_for_storage(path: str | Path, calibration_path: str | Path | None = None) -> str:
    source_path = Path(path).expanduser()
    if calibration_path is None:
        return str(source_path)

    base_dir = Path(calibration_path).expanduser().parent
    try:
        return os.path.relpath(source_path.resolve(), base_dir.resolve())
    except ValueError:
        return str(source_path)
