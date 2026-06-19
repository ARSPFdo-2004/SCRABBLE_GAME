from __future__ import annotations

import re
from dataclasses import dataclass

BOARD_SIZE = 12
CELL_SIZE_MM = 30.0
SQUARE_LABEL_PATTERN = re.compile(r"^([A-La-l])(1[0-2]|[1-9])$")


@dataclass(frozen=True)
class Square:
    col: int
    row: int

    @property
    def label(self) -> str:
        return f"{chr(ord('A') + self.col)}{self.row + 1}"

    def center_in_board_space(self) -> tuple[float, float]:
        return (self.col + 0.5, self.row + 0.5)


def parse_square_label(label: str) -> Square:
    normalized = label.strip().upper()
    match = SQUARE_LABEL_PATTERN.match(normalized)
    if not match:
        raise ValueError(
            f"Invalid square label '{label}'. Use Letter+number from A1 to L12."
        )

    col = ord(match.group(1)) - ord("A")
    row = int(match.group(2)) - 1
    return Square(col=col, row=row)
