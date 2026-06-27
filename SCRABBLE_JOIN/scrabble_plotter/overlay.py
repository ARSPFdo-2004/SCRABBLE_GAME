from __future__ import annotations

from .board import BOARD_SIZE
from .calibration import PlotterCalibration, board_corner_points


Point = tuple[float, float]
Segment = tuple[Point, Point]


def project_board_points(
    corners: list[list[float]] | list[tuple[float, float]],
    points: list[Point],
    board_size: int = BOARD_SIZE,
) -> list[Point]:
    if len(corners) != 4:
        raise ValueError("Exactly 4 board corners are required.")

    cv2 = _require_cv2()
    transform = cv2.getPerspectiveTransform(
        _to_float32(board_corner_points(board_size)),
        _to_float32([[float(x), float(y)] for x, y in corners]),
    )
    projected = cv2.perspectiveTransform(_to_float32([[list(point) for point in points]]), transform)
    return [(float(point[0]), float(point[1])) for point in projected[0]]


def grid_segments(
    corners: list[list[float]] | list[tuple[float, float]],
    board_size: int = BOARD_SIZE,
) -> list[Segment]:
    segment_points: list[Point] = []
    for index in range(board_size + 1):
        segment_points.extend(
            [
                (float(index), 0.0),
                (float(index), float(board_size)),
                (0.0, float(index)),
                (float(board_size), float(index)),
            ]
        )

    projected = project_board_points(corners, segment_points, board_size)
    return [
        (projected[index], projected[index + 1])
        for index in range(0, len(projected), 2)
    ]


def cell_label_positions(
    corners: list[list[float]] | list[tuple[float, float]],
    board_size: int = BOARD_SIZE,
) -> list[tuple[str, Point]]:
    labels: list[str] = []
    points: list[Point] = []
    for row in range(board_size):
        for col in range(board_size):
            labels.append(f"{chr(ord('A') + col)}{row + 1}")
            points.append((col + 0.5, row + 0.5))

    projected = project_board_points(corners, points, board_size)
    return list(zip(labels, projected))


def draw_board_overlay(
    frame,
    calibration: PlotterCalibration,
    board_letters: list[list[str]] | None = None,
):  # type: ignore[no-untyped-def]
    if len(calibration.image_corners) != 4:
        return frame

    return draw_grid_overlay(
        frame,
        calibration.image_corners,
        board_size=calibration.board_size,
        board_letters=board_letters,
        show_empty_labels=True,
    )


def draw_camera_ocr_grid_overlay(frame, ocr_grid):  # type: ignore[no-untyped-def]
    if ocr_grid is None or len(getattr(ocr_grid, "corners", [])) != 4:
        return frame

    board_letters = ocr_grid.board_letters() if hasattr(ocr_grid, "board_letters") else None
    board_confidences = _grid_confidences(ocr_grid)
    return draw_grid_overlay(
        frame,
        ocr_grid.corners,
        board_size=getattr(ocr_grid, "board_size", BOARD_SIZE),
        board_letters=board_letters,
        board_confidences=board_confidences,
        show_empty_labels=False,
    )


def draw_grid_overlay(
    frame,
    corners: list[list[float]] | list[tuple[float, float]],
    board_size: int = BOARD_SIZE,
    board_letters: list[list[str]] | None = None,
    board_confidences: list[list[float]] | None = None,
    show_empty_labels: bool = True,
):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    overlay = frame.copy()

    for start, end in grid_segments(corners, board_size):
        cv2.line(overlay, _int_point(start), _int_point(end), (0, 255, 255), 1, cv2.LINE_AA)

    outline = project_board_points(
        corners,
        [(0.0, 0.0), (float(board_size), 0.0), (float(board_size), float(board_size)), (0.0, float(board_size))],
        board_size,
    )
    for index, point in enumerate(outline):
        cv2.line(overlay, _int_point(point), _int_point(outline[(index + 1) % 4]), (0, 200, 0), 3, cv2.LINE_AA)

    label_points = cell_label_positions(corners, board_size)
    for index, (label, point) in enumerate(label_points):
        x, y = _int_point(point)
        row = index // board_size
        col = index % board_size
        letter = _letter_at(board_letters, row, col)
        if letter:
            _draw_centered_letter_confidence(
                overlay,
                letter,
                _confidence_at(board_confidences, row, col),
                (x, y),
                letter_scale=0.78,
                confidence_scale=0.34,
                text_color=(0, 0, 0),
                fill_color=(70, 255, 130),
            )
        elif show_empty_labels:
            cv2.putText(
                overlay,
                label,
                (x - 10, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def draw_captured_letters_overlay(frame, captured_letters):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    overlay = frame.copy()
    for captured in captured_letters:
        x1 = int(captured.left)
        y1 = int(captured.top)
        x2 = int(captured.left + captured.width)
        y2 = int(captured.top + captured.height)
        text = str(captured.text).strip().upper()
        if not text:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 255, 130), 2, cv2.LINE_AA)
        label = f"{text} {captured.confidence:.0f}%"
        _draw_text_label(overlay, label, (x1, max(0, y1 - 6)))
    return overlay


def draw_camera_ocr_overlay(
    frame,
    captured_letters=None,
    detected_words=None,
    detected_tiles=None,
    ocr_grid=None,
    show_letters: bool = True,
    show_word_labels: bool = True,
):  # type: ignore[no-untyped-def]
    overlay = frame.copy()
    if show_letters:
        overlay = draw_camera_ocr_grid_overlay(overlay, ocr_grid)
        overlay = draw_captured_letters_overlay(overlay, captured_letters or [])
        overlay = draw_detected_tiles_overlay(overlay, detected_tiles or [])
    return draw_detected_words_overlay(overlay, detected_words or [], show_labels=show_word_labels)


def draw_detected_tiles_overlay(frame, detected_tiles):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    overlay = frame.copy()
    for tile in detected_tiles:
        points = [_int_point(point) for point in tile.corners]
        if len(points) != 4:
            continue
        cv2.polylines(overlay, [_to_int32(points)], True, (70, 255, 130), 2, cv2.LINE_AA)
        letter = str(tile.letter).strip().upper()
        if letter:
            center = (
                int(round(sum(point[0] for point in points) / len(points))),
                int(round(sum(point[1] for point in points) / len(points))),
            )
            _draw_centered_letter_confidence(
                overlay,
                letter,
                getattr(tile, "confidence", 0.0),
                center,
                letter_scale=0.62,
                confidence_scale=0.34,
                text_color=(0, 0, 0),
                fill_color=(70, 255, 130),
            )
    return overlay


def draw_detected_words_overlay(frame, detected_words, show_labels: bool = True):  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    overlay = frame.copy()
    for index, detected in enumerate(detected_words, start=1):
        x1 = int(detected.left)
        y1 = int(detected.top)
        x2 = int(detected.left + detected.width)
        y2 = int(detected.top + detected.height)
        word = str(detected.word).strip().upper()
        if not word:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 220, 255), 2, cv2.LINE_AA)
        if show_labels:
            _draw_text_label(overlay, f"{index}. {word}", (x1, max(0, y1 - 26)), fill_color=(80, 220, 255))
    return overlay


def _grid_confidences(ocr_grid) -> list[list[float]] | None:  # type: ignore[no-untyped-def]
    board_size = getattr(ocr_grid, "board_size", BOARD_SIZE)
    cells = getattr(ocr_grid, "cells", None)
    if cells is None:
        return None

    confidences = [[0.0 for _ in range(board_size)] for _ in range(board_size)]
    for cell in cells:
        row = int(getattr(cell, "row", -1))
        col = int(getattr(cell, "col", -1))
        if 0 <= row < board_size and 0 <= col < board_size:
            try:
                confidences[row][col] = max(confidences[row][col], float(getattr(cell, "confidence", 0.0)))
            except (TypeError, ValueError):
                pass
    return confidences


def _letter_at(board_letters: list[list[str]] | None, row: int, col: int) -> str:
    if board_letters is None or row >= len(board_letters) or col >= len(board_letters[row]):
        return ""
    letter = str(board_letters[row][col]).strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        return letter
    return ""


def _confidence_at(board_confidences: list[list[float]] | None, row: int, col: int) -> float:
    if board_confidences is None or row >= len(board_confidences) or col >= len(board_confidences[row]):
        return 0.0
    try:
        return float(board_confidences[row][col])
    except (TypeError, ValueError):
        return 0.0


def _confidence_label(confidence: float) -> str:
    if confidence <= 0:
        return ""
    return f"{confidence:.0f}%"


def _draw_centered_letter_confidence(
    frame,
    letter: str,
    confidence: float,
    center: tuple[int, int],
    letter_scale: float,
    confidence_scale: float,
    text_color: tuple[int, int, int],
    fill_color: tuple[int, int, int],
) -> None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    letter = str(letter).strip().upper()
    confidence_text = _confidence_label(confidence)
    letter_thickness = 2
    confidence_thickness = 1
    (letter_width, letter_height), letter_baseline = cv2.getTextSize(letter, font, letter_scale, letter_thickness)
    if confidence_text:
        (confidence_width, confidence_height), confidence_baseline = cv2.getTextSize(
            confidence_text,
            font,
            confidence_scale,
            confidence_thickness,
        )
        gap = 2
    else:
        confidence_width = confidence_height = confidence_baseline = gap = 0

    text_width = max(letter_width, confidence_width)
    total_height = letter_height + letter_baseline + gap + confidence_height + confidence_baseline
    pad_x = 4
    pad_y = 3
    left = int(center[0] - text_width / 2) - pad_x
    top = int(center[1] - total_height / 2) - pad_y
    right = left + text_width + pad_x * 2
    bottom = top + total_height + pad_y * 2
    cv2.rectangle(frame, (left, top), (right, bottom), fill_color, -1)

    letter_x = int(center[0] - letter_width / 2)
    letter_y = top + pad_y + letter_height
    cv2.putText(frame, letter, (letter_x, letter_y), font, letter_scale, text_color, letter_thickness, cv2.LINE_AA)

    if confidence_text:
        confidence_x = int(center[0] - confidence_width / 2)
        confidence_y = letter_y + letter_baseline + gap + confidence_height
        cv2.putText(
            frame,
            confidence_text,
            (confidence_x, confidence_y),
            font,
            confidence_scale,
            text_color,
            confidence_thickness,
            cv2.LINE_AA,
        )


def _draw_centered_text(
    frame,
    text: str,
    center: tuple[int, int],
    scale: float,
    text_color: tuple[int, int, int],
    fill_color: tuple[int, int, int],
    thickness: int,
) -> None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(center[0] - text_width / 2)
    y = int(center[1] + text_height / 2)
    pad = 4
    cv2.rectangle(
        frame,
        (x - pad, y - text_height - pad),
        (x + text_width + pad, y + baseline + pad),
        fill_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def _draw_text_label(
    frame,
    text: str,
    origin: tuple[int, int],
    fill_color: tuple[int, int, int] = (70, 255, 130),
) -> None:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, int(origin[0]))
    y = max(text_height + 4, int(origin[1]))
    cv2.rectangle(
        frame,
        (x, y - text_height - 5),
        (x + text_width + 8, y + baseline + 4),
        fill_color,
        -1,
    )
    cv2.putText(frame, text, (x + 4, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _int_point(point: Point) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))


def _to_float32(values):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.array(values, dtype=np.float32)


def _to_int32(values):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.array(values, dtype=np.int32)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for camera overlay. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2
