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
    image_path: str | None = None
    image_corners: list[list[float]] = field(default_factory=list)
    camera_index: int = 0
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    cell_size_mm: float = CELL_SIZE_MM
    x_steps_per_mm: float = 80.0
    y_steps_per_mm: float = 80.0
    cart_x_mm: float = 0.0
    cart_y_mm: float = 0.0
    premium_layout: list[list[str]] = field(default_factory=lambda: default_premium_layout())
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD
    ocr_cell_size_px: int = DEFAULT_OCR_CELL_SIZE_PX

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
        x = self.offset_x_mm + (square.col + 0.5) * self.cell_size_mm
        y = self.offset_y_mm + (square.row + 0.5) * self.cell_size_mm
        return (x, y)

    def cart_position_in_machine(self) -> tuple[float, float]:
        return (self.cart_x_mm, self.cart_y_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "label_orientation": self.label_orientation,
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
            "premium_layout": normalize_premium_layout(self.premium_layout, self.board_size),
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "ocr_cell_size_px": self.ocr_cell_size_px,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlotterCalibration":
        return cls(
            board_size=int(payload.get("board_size", BOARD_SIZE)),
            label_orientation=payload.get("label_orientation", "A1-top-left"),
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
            premium_layout=normalize_premium_layout(
                payload.get("premium_layout", default_premium_layout()),
                int(payload.get("board_size", BOARD_SIZE)),
            ),
            ocr_confidence_threshold=float(
                payload.get("ocr_confidence_threshold", DEFAULT_OCR_CONFIDENCE_THRESHOLD)
            ),
            ocr_cell_size_px=int(payload.get("ocr_cell_size_px", DEFAULT_OCR_CELL_SIZE_PX)),
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
    return [[PREMIUM_NORMAL for _ in range(board_size)] for _ in range(board_size)]


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
