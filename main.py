from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np


DEFAULT_OUTPUT_PATH = "camera_ocr_result.jpg"
BOARD_SIZE = 12
EMPTY_CELL = "."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the best webcam frame and analyze it for characters with EasyOCR."
    )
    parser.add_argument("--camera-index", type=int, default=1, help="Camera index to open, usually 0.")
    parser.add_argument("--frames", type=int, default=24, help="Number of camera frames to sample.")
    parser.add_argument("--duration", type=float, default=1.5, help="Maximum capture time in seconds.")
    parser.add_argument("--width", type=int, help="Optional camera capture width.")
    parser.add_argument("--height", type=int, help="Optional camera capture height.")
    parser.add_argument("--threshold", type=float, default=0.10, help="Minimum OCR confidence from 0 to 1.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Annotated image output path.")
    parser.add_argument("--image", help="Analyze an existing image instead of opening the camera.")
    parser.add_argument("--grid-size", type=int, default=BOARD_SIZE, help="Board matrix size.")
    parser.add_argument("--live", action="store_true", help="Open a live camera preview with OCR overlay.")
    parser.add_argument(
        "--live-interval",
        type=float,
        default=2.5,
        help="Seconds between background OCR updates in live mode.",
    )
    parser.add_argument("--show", action="store_true", help="Show the annotated image in an OpenCV window.")
    return parser


def open_camera(camera_index: int, width: int | None = None, height: int | None = None) -> Any:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_index}.")

    if width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def capture_best_frame(
    camera_index: int,
    frame_count: int,
    duration_seconds: float,
    width: int | None = None,
    height: int | None = None,
) -> tuple[Any, dict[str, float]]:
    camera = open_camera(camera_index, width=width, height=height)

    try:
        best_frame = None
        best_quality: dict[str, float] | None = None
        deadline = time.monotonic() + max(0.1, duration_seconds)
        attempts = 0

        while attempts < max(1, frame_count) and time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.03)
                continue

            attempts += 1
            quality = score_frame_quality(frame)
            if best_quality is None or quality["score"] > best_quality["score"]:
                best_frame = frame.copy()
                best_quality = quality
            time.sleep(0.025)

        if best_frame is None or best_quality is None:
            raise RuntimeError("The camera opened, but no frame could be captured.")
        return best_frame, best_quality
    finally:
        camera.release()


def score_frame_quality(frame: Any) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return {"score": 0.0, "sharpness": 0.0, "contrast": 0.0, "brightness": 0.0}

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
    return {
        "score": float(score),
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
    }


def analyze_characters(frame: Any, threshold: float, reader: Any | None = None) -> list[dict[str, Any]]:
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False)
    tile_detections = analyze_tile_characters(frame, reader, threshold)
    if tile_detections:
        return tile_detections

    detections: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []

    for variant_name, variant, scale in ocr_frame_variants(frame):
        raw_results = read_easyocr(reader, variant)
        for box, text, confidence in raw_results:
            confidence = float(confidence)
            cleaned_text = "".join(character for character in str(text).upper() if character.isalnum())
            if not cleaned_text:
                continue
            detection = {
                "box": scale_box_to_original(box, scale),
                "text": cleaned_text,
                "confidence": confidence,
                "variant": variant_name,
            }
            fallback_candidates.append(detection)
            if confidence >= threshold:
                add_detection(detections, detection)

    if not detections:
        for detection in sorted(fallback_candidates, key=lambda item: item["confidence"], reverse=True)[:10]:
            if detection["confidence"] >= 0.01:
                add_detection(detections, detection)

    return sorted(detections, key=lambda item: (box_bounds(item["box"])[1], box_bounds(item["box"])[0]))


def analyze_tile_characters(frame: Any, reader: Any, threshold: float) -> list[dict[str, Any]]:
    try:
        package_root = Path(__file__).resolve().parent / "Scrabble-2"
        if package_root.exists() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from scrabble_plotter.scanner import detect_camera_tiles
    except Exception:
        return []

    tile_threshold = min(max(threshold * 100.0, 5.0), 15.0)
    try:
        tiles = detect_camera_tiles(
            frame,
            confidence_threshold=tile_threshold,
            ocr_reader=lambda crop: read_easyocr(reader, crop),
        )
    except Exception:
        return []

    detections = [
        {
            "box": [[float(x), float(y)] for x, y in tile.corners],
            "text": tile.letter,
            "confidence": min(1.0, max(0.0, tile.confidence / 100.0)),
            "variant": "tile",
        }
        for tile in tiles
        if tile.letter
    ]
    return sorted(detections, key=lambda item: (box_bounds(item["box"])[1], box_bounds(item["box"])[0]))


def read_easyocr(reader: Any, frame: Any) -> list[Any]:
    try:
        return reader.readtext(
            frame,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            batch_size=4,
            contrast_ths=0.05,
            adjust_contrast=0.8,
            text_threshold=0.25,
            low_text=0.15,
            link_threshold=0.15,
            paragraph=False,
        )
    except TypeError:
        return reader.readtext(frame)


def ocr_frame_variants(frame: Any) -> list[tuple[str, Any, float]]:
    variants: list[tuple[str, Any, float]] = [("original", frame, 1.0)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    height, width = gray.shape[:2]
    scale = max(1.0, min(3.0, 1600.0 / float(max(1, width))))

    blurred = cv2.GaussianBlur(frame, (0, 0), 1.1)
    sharpened = cv2.addWeighted(frame, 1.7, blurred, -0.7, 0)
    variants.append(("sharpened", sharpened, 1.0))

    if scale > 1.0:
        variants.append(("upscaled", resize_for_ocr(frame, scale), scale))
        gray = resize_for_ocr(gray, scale)
    else:
        variants.append(("grayscale", gray, 1.0))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("enhanced", enhanced, scale))

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    variants.append(("threshold", thresholded, scale))
    variants.append(("inverted_threshold", cv2.bitwise_not(thresholded), scale))
    return variants


def resize_for_ocr(frame: Any, scale: float) -> Any:
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interpolation)


def scale_box_to_original(box: Any, scale: float) -> list[list[float]]:
    divisor = scale if scale > 0 else 1.0
    return [[float(point[0]) / divisor, float(point[1]) / divisor] for point in box]


def add_detection(detections: list[dict[str, Any]], detection: dict[str, Any]) -> None:
    for index, existing in enumerate(detections):
        if boxes_overlap(existing["box"], detection["box"]) >= 0.45:
            if detection["confidence"] > existing["confidence"]:
                detections[index] = detection
            return
    detections.append(detection)


def boxes_overlap(first_box: Any, second_box: Any) -> float:
    first_left, first_top, first_right, first_bottom = box_bounds(first_box)
    second_left, second_top, second_right, second_bottom = box_bounds(second_box)
    overlap_left = max(first_left, second_left)
    overlap_top = max(first_top, second_top)
    overlap_right = min(first_right, second_right)
    overlap_bottom = min(first_bottom, second_bottom)
    overlap_width = max(0.0, overlap_right - overlap_left)
    overlap_height = max(0.0, overlap_bottom - overlap_top)
    intersection = overlap_width * overlap_height
    first_area = max(1.0, (first_right - first_left) * (first_bottom - first_top))
    second_area = max(1.0, (second_right - second_left) * (second_bottom - second_top))
    return intersection / max(1.0, first_area + second_area - intersection)


def box_bounds(box: Any) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def best_frame_from_samples(frames: list[Any]) -> tuple[Any, dict[str, float]]:
    candidates = [(frame, score_frame_quality(frame)) for frame in frames if frame is not None]
    if not candidates:
        raise RuntimeError("No frames are available for OCR.")
    return max(candidates, key=lambda item: item[1]["score"])


@dataclass(frozen=True)
class FlexibleGrid:
    board_size: int
    corners: list[tuple[float, float]]
    image_to_grid: Any
    grid_to_image: Any

    def map_point(self, x: float, y: float) -> tuple[int, int] | None:
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.image_to_grid)[0][0]
        grid_x = float(mapped[0])
        grid_y = float(mapped[1])
        if grid_x < -0.15 or grid_y < -0.15 or grid_x > self.board_size + 0.15 or grid_y > self.board_size + 0.15:
            return None
        col = max(0, min(self.board_size - 1, int(grid_x)))
        row = max(0, min(self.board_size - 1, int(grid_y)))
        return row, col


def map_detections_to_matrix(
    frame: Any,
    detections: list[dict[str, Any]],
    board_size: int,
) -> tuple[list[list[str]], list[dict[str, Any]], FlexibleGrid | None]:
    matrix = [[EMPTY_CELL for _ in range(board_size)] for _ in range(board_size)]
    mapped = [dict(detection) for detection in detections]
    grid = detect_flexible_grid(frame, board_size=board_size)
    if grid is None:
        return matrix, mapped, None

    confidence_by_cell = [[-1.0 for _ in range(board_size)] for _ in range(board_size)]
    for detection in mapped:
        text = str(detection.get("text", "")).strip()
        if not text:
            continue

        center_x, center_y = detection_center(detection)
        cell = grid.map_point(center_x, center_y)
        if cell is None:
            continue

        row, col = cell
        letter = text[:1].upper()
        confidence = float(detection.get("confidence", 0.0))
        detection["row"] = row
        detection["col"] = col
        detection["cell"] = f"{chr(ord('A') + col)}{row + 1}"
        if confidence >= confidence_by_cell[row][col]:
            matrix[row][col] = letter
            confidence_by_cell[row][col] = confidence

    return matrix, mapped, grid


def detect_flexible_grid(frame: Any, board_size: int) -> FlexibleGrid | None:
    corners = detect_board_corners_from_dark_grid(frame)
    if corners is None:
        corners = detect_rectangular_board_corners(frame)
    if corners is None:
        return None

    source = np.array(corners, dtype=np.float32)
    destination = np.array(
        [
            [0.0, 0.0],
            [float(board_size), 0.0],
            [float(board_size), float(board_size)],
            [0.0, float(board_size)],
        ],
        dtype=np.float32,
    )
    return FlexibleGrid(
        board_size=board_size,
        corners=corners,
        image_to_grid=cv2.getPerspectiveTransform(source, destination),
        grid_to_image=cv2.getPerspectiveTransform(destination, source),
    )


def detect_board_corners_from_dark_grid(frame: Any) -> list[tuple[float, float]] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)
    best: tuple[float, list[tuple[float, float]]] | None = None

    for threshold in (25, 35, 45, 55, 65):
        mask = cv2.inRange(gray, 0, threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.08:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = order_corners(approx.reshape(4, 2))
            else:
                corners = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))

            score = perspective_grid_score(corners, area, frame_width, frame_height)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, corners)

    return best[1] if best is not None else None


def detect_rectangular_board_corners(frame: Any) -> list[tuple[float, float]] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for threshold in (30, 40, 50, 60):
        mask = cv2.inRange(gray, 0, threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.08:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue
            aspect = width / float(height)
            fill = area / float(width * height)
            if 0.75 <= aspect <= 1.25 and fill >= 0.45:
                score = area * fill
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


def perspective_grid_score(
    corners: list[tuple[float, float]],
    contour_area: float,
    frame_width: int,
    frame_height: int,
) -> float:
    area = abs(polygon_area(corners))
    if area <= 0:
        return 0.0
    frame_area = float(frame_width * frame_height)
    if area < frame_area * 0.08 or area > frame_area * 0.95:
        return 0.0

    sides = [
        distance(corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ]
    shortest = min(sides)
    longest = max(sides)
    if shortest <= 0 or longest / shortest > 1.9:
        return 0.0

    fill = min(contour_area, area) / max(contour_area, area)
    if fill < 0.35:
        return 0.0
    return area * fill * (shortest / longest)


def order_corners(points: Any) -> list[tuple[float, float]]:
    array = np.array(points, dtype=np.float32).reshape(4, 2)
    sums = array.sum(axis=1)
    diffs = np.diff(array, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = array[int(sums.argmin())]
    ordered[2] = array[int(sums.argmax())]
    ordered[1] = array[int(diffs.argmin())]
    ordered[3] = array[int(diffs.argmax())]
    return [(float(x), float(y)) for x, y in ordered]


def polygon_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return total / 2.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def detection_center(detection: dict[str, Any]) -> tuple[float, float]:
    left, top, right, bottom = box_bounds(detection["box"])
    return (left + right) / 2.0, (top + bottom) / 2.0


def format_matrix(matrix: list[list[str]]) -> str:
    header = "    " + " ".join(chr(ord("A") + col) for col in range(len(matrix)))
    lines = [header]
    for index, row in enumerate(matrix, start=1):
        lines.append(f"{index:>2}: " + " ".join(cell if cell else EMPTY_CELL for cell in row))
    return "\n".join(lines)


def draw_detections(
    frame: Any,
    detections: list[dict[str, Any]],
    grid: FlexibleGrid | None = None,
) -> Any:
    annotated = frame.copy()
    if grid is not None:
        draw_board_grid(annotated, grid)
    for detection in detections:
        points = np.array(detection["box"], dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        x = int(points[:, 0].min())
        y = max(20, int(points[:, 1].min()) - 8)
        cell = detection.get("cell")
        label_text = f"{cell}:{detection['text']}" if cell else detection["text"]
        label = f"{label_text} {detection['confidence']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_board_grid(frame: Any, grid: FlexibleGrid) -> None:
    for index in range(grid.board_size + 1):
        vertical = grid_points_to_image(grid, [(float(index), 0.0), (float(index), float(grid.board_size))])
        horizontal = grid_points_to_image(grid, [(0.0, float(index)), (float(grid.board_size), float(index))])
        cv2.line(frame, vertical[0], vertical[1], (0, 180, 255), 1, cv2.LINE_AA)
        cv2.line(frame, horizontal[0], horizontal[1], (0, 180, 255), 1, cv2.LINE_AA)


def grid_points_to_image(grid: FlexibleGrid, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    array = np.array([[list(point)] for point in points], dtype=np.float32)
    transformed = cv2.perspectiveTransform(array, grid.grid_to_image).reshape(-1, 2)
    return [(int(round(point[0])), int(round(point[1]))) for point in transformed]


def draw_live_overlay(
    frame: Any,
    detections: list[dict[str, Any]],
    quality: dict[str, float] | None,
    matrix: list[list[str]] | None,
    grid: FlexibleGrid | None,
    message: str,
) -> Any:
    annotated = draw_detections(frame, detections, grid=grid)
    words = tile_words_from_detections(detections)
    word_text = ", ".join(word for word, _, _ in words)
    character_text = " ".join(character for detection in detections for character in detection["text"])
    filled_cells = sum(1 for row in matrix or [] for cell in row if cell != EMPTY_CELL)
    lines = [
        "Live OCR: q/esc quit | space scan | s save",
        f"Words: {word_text or '-'}",
        f"Characters: {character_text or '-'}",
        f"12x12 cells filled: {filled_cells}",
    ]
    if quality is not None:
        lines.append(
            f"Quality: sharpness {quality['sharpness']:.0f}, "
            f"contrast {quality['contrast']:.1f}, brightness {quality['brightness']:.1f}"
        )
    if message:
        lines.append(message)

    line_height = 24
    panel_height = 12 + line_height * len(lines)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], panel_height), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (12, 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_live_camera(args: argparse.Namespace, threshold: float) -> int:
    camera = open_camera(args.camera_index, width=args.width, height=args.height)
    reader = easyocr.Reader(["en"], gpu=False)
    output_path = Path(args.output)
    window_name = "Live Camera OCR"
    recent_frames: list[Any] = []
    lock = threading.Lock()
    state: dict[str, Any] = {
        "detections": [],
        "quality": None,
        "message": "Aiming...",
        "running": False,
        "last_started_at": 0.0,
        "matrix": None,
        "grid": None,
    }

    def start_ocr_scan(reason: str) -> None:
        with lock:
            if state["running"]:
                return
            samples = [frame.copy() for frame in recent_frames[-max(1, args.frames) :]]
            state["running"] = True
            state["message"] = f"OCR running ({reason})..."
            state["last_started_at"] = time.monotonic()

        try:
            frame, quality = best_frame_from_samples(samples)
        except Exception as exc:
            with lock:
                state["message"] = str(exc)
                state["running"] = False
            return

        def worker() -> None:
            try:
                detections = analyze_characters(frame, threshold=threshold, reader=reader)
                matrix, detections, grid = map_detections_to_matrix(
                    frame,
                    detections,
                    board_size=max(1, int(args.grid_size)),
                )
                annotated = draw_detections(frame, detections, grid=grid)
                cv2.imwrite(str(output_path), annotated)
                words = tile_words_from_detections(detections)
                summary = ", ".join(word for word, _, _ in words)
                if not summary:
                    summary = " ".join(detection["text"] for detection in detections) or "none"
                print("\n12x12 matrix:")
                print(format_matrix(matrix))
                with lock:
                    state["detections"] = detections
                    state["quality"] = quality
                    state["matrix"] = matrix
                    state["grid"] = grid
                    state["message"] = f"Detected: {summary}. Saved {output_path.name}."
                    state["running"] = False
            except Exception as exc:
                with lock:
                    state["message"] = f"OCR error: {exc}"
                    state["running"] = False

        threading.Thread(target=worker, daemon=True).start()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Live camera started. Press Space to scan now, S to save, Q or Esc to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.03)
                continue

            recent_frames.append(frame.copy())
            if len(recent_frames) > max(3, args.frames):
                recent_frames = recent_frames[-max(3, args.frames) :]

            now = time.monotonic()
            with lock:
                should_auto_scan = (
                    not state["running"]
                    and now - float(state["last_started_at"]) >= max(0.5, args.live_interval)
                )
                detections = list(state["detections"])
                quality = state["quality"]
                matrix = state["matrix"]
                grid = state["grid"]
                message = str(state["message"])

            if should_auto_scan:
                start_ocr_scan("auto")

            display = draw_live_overlay(frame, detections, quality, matrix, grid, message)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                start_ocr_scan("manual")
            if key in (ord("s"), ord("S")):
                cv2.imwrite(str(output_path), display)
                with lock:
                    state["message"] = f"Saved current view to {output_path.name}."
    finally:
        camera.release()
        cv2.destroyWindow(window_name)
    return 0


def print_detection_summary(
    detections: list[dict[str, Any]],
    matrix: list[list[str]] | None = None,
) -> None:
    characters = [character for detection in detections for character in detection["text"]]
    if characters:
        print("Characters:", " ".join(characters))
    else:
        print("Characters: none detected")

    if matrix is not None:
        print("12x12 matrix:")
        print(format_matrix(matrix))

    tile_words = tile_words_from_detections(detections)
    if tile_words:
        print("Detected words:")
        for word, direction, confidence in tile_words:
            print(f"- {word} ({direction}, {confidence:.2f})")

    if detections:
        print("Detected text boxes:")
        for detection in detections:
            variant = detection.get("variant", "ocr")
            print(f"- {detection['text']} ({detection['confidence']:.2f}, {variant})")


def tile_words_from_detections(detections: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    if not any(detection.get("variant") == "tile" for detection in detections):
        return []
    try:
        package_root = Path(__file__).resolve().parent / "Scrabble-2"
        if package_root.exists() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from scrabble_plotter.scanner import CameraTile, identify_directional_tile_words
    except Exception:
        return []

    tiles = [
        CameraTile(
            letter=str(detection["text"])[:1],
            confidence=float(detection["confidence"]) * 100.0,
            corners=[(float(point[0]), float(point[1])) for point in detection["box"]],
        )
        for detection in detections
        if detection.get("variant") == "tile" and str(detection.get("text", ""))
    ]
    return [
        (word.word, word.direction_label, min(1.0, max(0.0, word.confidence / 100.0)))
        for word in identify_directional_tile_words(tiles)
    ]


def main() -> int:
    args = build_parser().parse_args()
    threshold = min(1.0, max(0.0, args.threshold))

    if args.live:
        return run_live_camera(args, threshold)

    if args.image:
        frame = cv2.imread(str(Path(args.image)))
        if frame is None:
            raise RuntimeError(f"Unable to load image at '{args.image}'.")
        quality = score_frame_quality(frame)
        print(f"Analyzing image: {Path(args.image).resolve()}")
    else:
        frame, quality = capture_best_frame(
            camera_index=args.camera_index,
            frame_count=args.frames,
            duration_seconds=args.duration,
            width=args.width,
            height=args.height,
        )
        print(f"Captured best frame from camera {args.camera_index}.")

    print(
        "Frame quality: "
        f"score={quality['score']:.0f}, sharpness={quality['sharpness']:.0f}, "
        f"contrast={quality['contrast']:.1f}, brightness={quality['brightness']:.1f}"
    )

    detections = analyze_characters(frame, threshold=threshold)
    matrix, detections, grid = map_detections_to_matrix(
        frame,
        detections,
        board_size=max(1, int(args.grid_size)),
    )
    print_detection_summary(detections, matrix=matrix)

    annotated = draw_detections(frame, detections, grid=grid)
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated image to {output_path.resolve()}")

    if args.show:
        cv2.imshow("Camera OCR Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
