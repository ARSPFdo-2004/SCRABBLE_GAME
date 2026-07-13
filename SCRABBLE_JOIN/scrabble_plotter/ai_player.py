from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .board import BOARD_SIZE
from .calibration import (
    PREMIUM_DOUBLE_LETTER,
    PREMIUM_DOUBLE_WORD,
    PREMIUM_NORMAL,
    PREMIUM_TRIPLE_LETTER,
    PREMIUM_TRIPLE_WORD,
    normalize_premium_layout,
)
from .scoring import LETTER_VALUES, normalize_board_letters, normalize_letter, square_label
from .word_bank import generated_reference_words, normalize_word


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
AI_CANDIDATE_LIMIT = 50


@dataclass(frozen=True)
class RackTile:
    slot: int
    letter: str

    @property
    def rack_label(self) -> str:
        return f"TR{self.slot}"


@dataclass(frozen=True)
class AiPlacement:
    rack_slot: int
    letter: str
    row: int
    col: int

    @property
    def rack_label(self) -> str:
        return f"TR{self.rack_slot}"

    @property
    def square(self) -> str:
        return square_label(self.row, self.col)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "rack": self.rack_label,
            "square": self.square,
            "letter": self.letter,
        }


@dataclass(frozen=True)
class AiMoveCandidate:
    candidate_id: str
    word: str
    direction: str
    start_row: int
    start_col: int
    score: int
    placements: tuple[AiPlacement, ...]
    squares: tuple[str, ...]
    cross_words: tuple[str, ...] = ()

    @property
    def start_square(self) -> str:
        return square_label(self.start_row, self.start_col)

    @property
    def end_square(self) -> str:
        row_step, col_step = _direction_step(self.direction)
        end_row = self.start_row + row_step * (len(self.word) - 1)
        end_col = self.start_col + col_step * (len(self.word) - 1)
        return square_label(end_row, end_col)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "word": self.word,
            "direction": self.direction,
            "start": self.start_square,
            "end": self.end_square,
            "score": self.score,
            "placements": [placement.to_prompt_dict() for placement in self.placements],
            "cross_words": list(self.cross_words),
        }


@dataclass(frozen=True)
class AiMoveChoice:
    action: str
    candidate_id: str | None = None
    reason: str = ""
    candidate: AiMoveCandidate | None = None

    def validate(self, candidates: Iterable[AiMoveCandidate]) -> "AiMoveChoice":
        action = self.action.strip().lower()
        if action not in {"play", "pass"}:
            raise ValueError(f"OpenAI returned unsupported AI action: {self.action}")
        if action == "pass":
            return replace(self, action="pass", candidate_id=None, candidate=None)

        candidate_id = (self.candidate_id or "").strip().upper()
        if not candidate_id:
            raise ValueError("OpenAI chose play without a candidate id.")
        candidates_by_id = {candidate.candidate_id.upper(): candidate for candidate in candidates}
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"OpenAI chose unknown candidate id: {candidate_id}")
        return replace(self, action="play", candidate_id=candidate.candidate_id, candidate=candidate)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], candidates: Iterable[AiMoveCandidate]) -> "AiMoveChoice":
        return cls(
            action=str(payload.get("action", "pass")).strip().lower(),
            candidate_id=payload.get("candidate_id"),
            reason=str(payload.get("reason", "")).strip(),
        ).validate(candidates)


def top_ai_move_candidates(
    board_letters: list[list[str | None]],
    rack_letters: list[str] | list[RackTile],
    premium_layout: list[list[str]] | None = None,
    *,
    limit: int = AI_CANDIDATE_LIMIT,
    words: Iterable[str] | None = None,
) -> list[AiMoveCandidate]:
    candidates = generate_ai_move_candidates(
        board_letters,
        rack_letters,
        premium_layout=premium_layout,
        words=words,
    )
    return candidates[: max(0, int(limit))]


def generate_ai_move_candidates(
    board_letters: list[list[str | None]],
    rack_letters: list[str] | list[RackTile],
    premium_layout: list[list[str]] | None = None,
    *,
    words: Iterable[str] | None = None,
) -> list[AiMoveCandidate]:
    board = normalize_board_letters(board_letters)
    rack_tiles = normalize_rack_tiles(rack_letters)
    if not rack_tiles:
        return []

    premiums = normalize_premium_layout(premium_layout, BOARD_SIZE)
    bank = frozenset(normalize_word(word) for word in (words or generated_reference_words()))
    board_has_tiles = _board_has_tiles(board)

    candidates: list[AiMoveCandidate] = []
    for word in sorted(bank):
        if len(word) < 2 or len(word) > BOARD_SIZE:
            continue
        if not _rack_can_supply_missing_letters(word, board, rack_tiles):
            continue
        for direction in (HORIZONTAL, VERTICAL):
            row_step, col_step = _direction_step(direction)
            max_start_row = BOARD_SIZE - (len(word) - 1) * row_step
            max_start_col = BOARD_SIZE - (len(word) - 1) * col_step
            for row in range(max_start_row):
                for col in range(max_start_col):
                    candidate = _candidate_at(
                        board,
                        rack_tiles,
                        premiums,
                        bank,
                        word,
                        direction,
                        row,
                        col,
                        board_has_tiles,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

    candidates.sort(key=_candidate_sort_key)
    return [
        replace(candidate, candidate_id=f"C{index:03d}")
        for index, candidate in enumerate(candidates, start=1)
    ]


def normalize_rack_tiles(rack_letters: list[str] | list[RackTile]) -> tuple[RackTile, ...]:
    tiles: list[RackTile] = []
    for index, value in enumerate(rack_letters, start=1):
        if isinstance(value, RackTile):
            slot = int(value.slot)
            letter = normalize_letter(value.letter)
        else:
            slot = index
            letter = normalize_letter(str(value))
        if slot < 1 or slot > 7 or not letter:
            continue
        tiles.append(RackTile(slot=slot, letter=letter))
    return tuple(sorted(tiles, key=lambda tile: tile.slot))


def _candidate_at(
    board: list[list[str]],
    rack_tiles: tuple[RackTile, ...],
    premium_layout: list[list[str]],
    bank: frozenset[str],
    word: str,
    direction: str,
    start_row: int,
    start_col: int,
    board_has_tiles: bool,
) -> AiMoveCandidate | None:
    row_step, col_step = _direction_step(direction)
    before_row = start_row - row_step
    before_col = start_col - col_step
    after_row = start_row + row_step * len(word)
    after_col = start_col + col_step * len(word)
    if _letter_at(board, before_row, before_col) or _letter_at(board, after_row, after_col):
        return None

    remaining_tiles = list(rack_tiles)
    placements: list[AiPlacement] = []
    main_positions: list[tuple[int, int, str, bool]] = []
    reused_existing = False

    for index, letter in enumerate(word):
        row = start_row + row_step * index
        col = start_col + col_step * index
        
        # The user requested to never play in A1 (0, 0), B1 (0, 1), or A2 (1, 0)
        if (row, col) in ((0, 0), (0, 1), (1, 0)):
            return None

        existing = board[row][col]
        if existing:
            if existing != letter:
                return None
            reused_existing = True
            main_positions.append((row, col, letter, False))
            continue

        rack_index = _find_rack_tile_index(remaining_tiles, letter)
        if rack_index is None:
            return None
        tile = remaining_tiles.pop(rack_index)
        placement = AiPlacement(rack_slot=tile.slot, letter=letter, row=row, col=col)
        placements.append(placement)
        main_positions.append((row, col, letter, True))

    if not placements:
        return None
    if board_has_tiles:
        if not (reused_existing or _touches_existing_tile(board, placements)):
            return None
    else:
        covers_b2 = any(p.row == 1 and p.col == 1 for p in placements)
        if not covers_b2:
            return None

    temporary_board = [row.copy() for row in board]
    for placement in placements:
        temporary_board[placement.row][placement.col] = placement.letter

    cross_words: list[str] = []
    cross_score = 0
    for placement in placements:
        cross = _cross_word_for_placement(
            temporary_board,
            premium_layout,
            bank,
            placement,
            direction,
        )
        if cross is None:
            return None
        cross_word, score = cross
        if cross_word:
            cross_words.append(cross_word)
            cross_score += score

    main_score = _score_positions(main_positions, premium_layout)
    squares = tuple(square_label(row, col) for row, col, _letter, _is_new in main_positions)
    return AiMoveCandidate(
        candidate_id="",
        word=word,
        direction=direction,
        start_row=start_row,
        start_col=start_col,
        score=main_score + cross_score,
        placements=tuple(placements),
        squares=squares,
        cross_words=tuple(cross_words),
    )


def _cross_word_for_placement(
    board: list[list[str]],
    premium_layout: list[list[str]],
    bank: frozenset[str],
    placement: AiPlacement,
    main_direction: str,
) -> tuple[str, int] | None:
    cross_direction = VERTICAL if main_direction == HORIZONTAL else HORIZONTAL
    row_step, col_step = _direction_step(cross_direction)
    row = placement.row
    col = placement.col
    while _letter_at(board, row - row_step, col - col_step):
        row -= row_step
        col -= col_step

    positions: list[tuple[int, int, str, bool]] = []
    while True:
        letter = _letter_at(board, row, col)
        if not letter:
            break
        positions.append((row, col, letter, row == placement.row and col == placement.col))
        row += row_step
        col += col_step

    if len(positions) <= 1:
        return "", 0
    word = "".join(letter for _row, _col, letter, _is_new in positions)
    if word not in bank:
        return None
    return word, _score_positions(positions, premium_layout)


def _score_positions(
    positions: list[tuple[int, int, str, bool]],
    premium_layout: list[list[str]],
) -> int:
    base_score = 0
    word_multiplier = 1
    for row, col, letter, is_new in positions:
        premium = premium_layout[row][col] if is_new else PREMIUM_NORMAL
        base_score += LETTER_VALUES.get(letter, 0) * _letter_multiplier(premium)
        word_multiplier *= _word_multiplier(premium)
    return base_score * word_multiplier


def _rack_can_supply_missing_letters(
    word: str,
    board: list[list[str]],
    rack_tiles: tuple[RackTile, ...],
) -> bool:
    rack_counter = Counter(tile.letter for tile in rack_tiles)
    word_counter = Counter(word)
    board_counter = Counter(letter for row in board for letter in row if letter)
    for letter, count in word_counter.items():
        if rack_counter[letter] + board_counter[letter] < count:
            return False
    return True


def _candidate_sort_key(candidate: AiMoveCandidate) -> tuple[Any, ...]:
    direction_order = 0 if candidate.direction == HORIZONTAL else 1
    rack_slots = tuple(placement.rack_slot for placement in candidate.placements)
    return (
        -candidate.score,
        -len(candidate.word),
        candidate.start_row,
        candidate.start_col,
        direction_order,
        candidate.word,
        rack_slots,
    )


def _touches_existing_tile(board: list[list[str]], placements: Iterable[AiPlacement]) -> bool:
    for placement in placements:
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if _letter_at(board, placement.row + row_delta, placement.col + col_delta):
                return True
    return False


def _find_rack_tile_index(rack_tiles: list[RackTile], letter: str) -> int | None:
    for index, tile in enumerate(rack_tiles):
        if tile.letter == letter:
            return index
    return None


def _board_has_tiles(board: list[list[str]]) -> bool:
    return any(letter for row in board for letter in row)


def _letter_at(board: list[list[str]], row: int, col: int) -> str:
    if row < 0 or col < 0 or row >= BOARD_SIZE or col >= BOARD_SIZE:
        return ""
    return board[row][col]


def _direction_step(direction: str) -> tuple[int, int]:
    if direction == HORIZONTAL:
        return 0, 1
    if direction == VERTICAL:
        return 1, 0
    raise ValueError(f"Unsupported AI direction: {direction}")


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
