"""Scrabble board image-to-G-code sender."""

from .board import BOARD_SIZE, CELL_SIZE_MM, Square, parse_square_label
from .calibration import PlotterCalibration

__all__ = ["BOARD_SIZE", "CELL_SIZE_MM", "PlotterCalibration", "Square", "parse_square_label"]
