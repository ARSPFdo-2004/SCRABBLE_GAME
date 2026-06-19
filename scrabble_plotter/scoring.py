from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .board import BOARD_SIZE
from .calibration import (
    PREMIUM_DOUBLE_LETTER,
    PREMIUM_DOUBLE_WORD,
    PREMIUM_NORMAL,
    PREMIUM_TRIPLE_LETTER,
    PREMIUM_TRIPLE_WORD,
    normalize_premium_layout,
)


LETTER_VALUES: dict[str, int] = {
    "A": 1,
    "B": 3,
    "C": 3,
    "D": 2,
    "E": 1,
    "F": 4,
    "G": 2,
    "H": 4,
    "I": 1,
    "J": 8,
    "K": 5,
    "L": 1,
    "M": 3,
    "N": 1,
    "O": 1,
    "P": 3,
    "Q": 10,
    "R": 1,
    "S": 1,
    "T": 1,
    "U": 1,
    "V": 4,
    "W": 4,
    "X": 8,
    "Y": 4,
    "Z": 10,
}

PREMIUM_SHORT_LABELS: dict[str, str] = {
    PREMIUM_NORMAL: "",
    PREMIUM_DOUBLE_LETTER: "DL",
    PREMIUM_TRIPLE_LETTER: "TL",
    PREMIUM_DOUBLE_WORD: "DW",
    PREMIUM_TRIPLE_WORD: "TW",
}
PREMIUM_FROM_SHORT_LABEL: dict[str, str] = {
    "": PREMIUM_NORMAL,
    "N": PREMIUM_NORMAL,
    "DL": PREMIUM_DOUBLE_LETTER,
    "TL": PREMIUM_TRIPLE_LETTER,
    "DW": PREMIUM_DOUBLE_WORD,
    "TW": PREMIUM_TRIPLE_WORD,
}


@dataclass(frozen=True)
class ScoredTile:
    square: str
    letter: str
    blank: bool
    face_value: int
    letter_multiplier: int
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "square": self.square,
            "letter": self.letter,
            "blank": self.blank,
            "face_value": self.face_value,
            "letter_multiplier": self.letter_multiplier,
            "score": self.score,
        }


@dataclass(frozen=True)
class ScoredWord:
    word: str
    direction: str
    squares: list[str]
    base_score: int
    word_multiplier: int
    score: int
    tiles: list[ScoredTile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "direction": self.direction,
            "squares": self.squares,
            "base_score": self.base_score,
            "word_multiplier": self.word_multiplier,
            "score": self.score,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


@dataclass(frozen=True)
class ScoreResult:
    total_score: int
    words: list[ScoredWord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "words": [word.to_dict() for word in self.words],
        }


def score_board(
    board_letters: list[list[str | None]],
    premium_layout: list[list[str]] | None = None,
    blank_squares: set[str] | None = None,
) -> ScoreResult:
    normalized_board = normalize_board_letters(board_letters)
    normalized_premiums = normalize_premium_layout(
        premium_layout if premium_layout is not None else None,
        BOARD_SIZE,
    )
    blanks = {square.upper() for square in (blank_squares or set())}
    words: list[ScoredWord] = []

    for row in range(BOARD_SIZE):
        words.extend(_score_line(normalized_board, normalized_premiums, blanks, row, 0, 0, 1, "horizontal"))
    for col in range(BOARD_SIZE):
        words.extend(_score_line(normalized_board, normalized_premiums, blanks, 0, col, 1, 0, "vertical"))

    return ScoreResult(total_score=sum(word.score for word in words), words=words)


def normalize_board_letters(board_letters: list[list[str | None]]) -> list[list[str]]:
    board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for row_index in range(min(BOARD_SIZE, len(board_letters))):
        row = board_letters[row_index]
        for col_index in range(min(BOARD_SIZE, len(row))):
            board[row_index][col_index] = normalize_letter(row[col_index])
    return board


def normalize_letter(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if len(text) != 1 or text < "A" or text > "Z":
        return ""
    return text


def square_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def premium_short_label(code: str) -> str:
    return PREMIUM_SHORT_LABELS.get(code, "")


def premium_from_short_label(label: str) -> str:
    return PREMIUM_FROM_SHORT_LABEL.get(label.strip().upper(), PREMIUM_NORMAL)


def _score_line(
    board: list[list[str]],
    premium_layout: list[list[str]],
    blank_squares: set[str],
    start_row: int,
    start_col: int,
    row_step: int,
    col_step: int,
    direction: str,
) -> list[ScoredWord]:
    words: list[ScoredWord] = []
    run: list[tuple[int, int, str]] = []
    row = start_row
    col = start_col

    while row < BOARD_SIZE and col < BOARD_SIZE:
        letter = board[row][col]
        if letter:
            run.append((row, col, letter))
        else:
            words.extend(_score_run(run, premium_layout, blank_squares, direction))
            run = []
        row += row_step
        col += col_step

    words.extend(_score_run(run, premium_layout, blank_squares, direction))
    return words


def _score_run(
    run: list[tuple[int, int, str]],
    premium_layout: list[list[str]],
    blank_squares: set[str],
    direction: str,
) -> list[ScoredWord]:
    if len(run) < 2:
        return []

    tiles: list[ScoredTile] = []
    word_multiplier = 1
    base_score = 0

    for row, col, letter in run:
        square = square_label(row, col)
        premium = premium_layout[row][col]
        blank = square in blank_squares
        face_value = 0 if blank else LETTER_VALUES.get(letter, 0)
        letter_multiplier = _letter_multiplier(premium)
        word_multiplier *= _word_multiplier(premium)
        tile_score = face_value * letter_multiplier
        base_score += tile_score
        tiles.append(
            ScoredTile(
                square=square,
                letter=letter,
                blank=blank,
                face_value=face_value,
                letter_multiplier=letter_multiplier,
                score=tile_score,
            )
        )

    return [
        ScoredWord(
            word="".join(letter for _, _, letter in run),
            direction=direction,
            squares=[tile.square for tile in tiles],
            base_score=base_score,
            word_multiplier=word_multiplier,
            score=base_score * word_multiplier,
            tiles=tiles,
        )
    ]


def _letter_multiplier(premium: str) -> int:
    if premium == PREMIUM_DOUBLE_LETTER:
        return 2
    if premium == PREMIUM_TRIPLE_LETTER:
        return 3
    return 1


def _word_multiplier(premium: str) -> int:
    if premium == PREMIUM_DOUBLE_WORD:
        return 2
    if premium == PREMIUM_TRIPLE_WORD:
        return 3
    return 1
