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

    cv2 = _require_cv2()
    board_size = calibration.board_size
    overlay = frame.copy()

    for start, end in grid_segments(calibration.image_corners, board_size):
        cv2.line(overlay, _int_point(start), _int_point(end), (0, 255, 255), 1, cv2.LINE_AA)

    outline = project_board_points(
        calibration.image_corners,
        [(0.0, 0.0), (float(board_size), 0.0), (float(board_size), float(board_size)), (0.0, float(board_size))],
        board_size,
    )
    for index, point in enumerate(outline):
        cv2.line(overlay, _int_point(point), _int_point(outline[(index + 1) % 4]), (0, 200, 0), 3, cv2.LINE_AA)

    label_points = cell_label_positions(calibration.image_corners, board_size)
    for index, (label, point) in enumerate(label_points):
        x, y = _int_point(point)
        row = index // board_size
        col = index % board_size
        letter = _letter_at(board_letters, row, col)
        if letter:
            _draw_centered_text(overlay, letter, (x, y), 0.95, (0, 0, 0), (70, 255, 130), 2)
        else:
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


def draw_camera_ocr_overlay(frame, captured_letters=None, detected_words=None, detected_tiles=None):  # type: ignore[no-untyped-def]
    overlay = draw_captured_letters_overlay(frame, captured_letters or [])
    overlay = draw_detected_tiles_overlay(overlay, detected_tiles or [])
    return draw_detected_words_overlay(overlay, detected_words or [])


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
            _draw_centered_text(overlay, letter, center, 0.65, (0, 0, 0), (70, 255, 130), 2)
    return overlay


def draw_detected_words_overlay(frame, detected_words):  # type: ignore[no-untyped-def]
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
        _draw_text_label(overlay, f"{index}. {word}", (x1, max(0, y1 - 26)), fill_color=(80, 220, 255))
    return overlay


def _letter_at(board_letters: list[list[str]] | None, row: int, col: int) -> str:
    if board_letters is None or row >= len(board_letters) or col >= len(board_letters[row]):
        return ""
    letter = str(board_letters[row][col]).strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        return letter
    return ""


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
