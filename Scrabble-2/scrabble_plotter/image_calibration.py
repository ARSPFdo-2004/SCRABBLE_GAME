from __future__ import annotations

from pathlib import Path


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for image calibration. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2


def collect_board_corners(image_path: str) -> list[tuple[float, float]]:
    cv2 = _require_cv2()
    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise ValueError(f"Unable to load image at '{image_path}'.")
    return collect_board_corners_from_frame(image)


def collect_board_corners_from_frame(image) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    cv2 = _require_cv2()
    corners: list[tuple[float, float]] = []
    display = image.copy()
    window_name = "Scrabble Board Calibration"
    prompt = [
        "Click top-left corner",
        "Click top-right corner",
        "Click bottom-right corner",
        "Click bottom-left corner",
    ]

    def handle_click(event, x, y, flags, param):  # type: ignore[no-untyped-def]
        nonlocal display
        if event != cv2.EVENT_LBUTTONDOWN or len(corners) >= 4:
            return
        corners.append((float(x), float(y)))
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            str(len(corners)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_click)

    try:
        while True:
            frame = display.copy()
            status = prompt[min(len(corners), 3)] if len(corners) < 4 else "Press Enter to confirm"
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                raise KeyboardInterrupt("Image calibration cancelled.")
            if key in (10, 13) and len(corners) == 4:
                return corners
    finally:
        cv2.destroyWindow(window_name)
