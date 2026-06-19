from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_CAMERA_WARMUP_ATTEMPTS = 10
DEFAULT_CAMERA_READ_DELAY_SECONDS = 0.04


@dataclass(frozen=True)
class CameraOpenResult:
    capture: Any
    backend_name: str
    first_frame: Any


def open_camera_capture(
    cv2: Any,
    camera_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    camera_fourcc: str | None = None,
    zoom_out: bool = False,
    camera_zoom: float | None = None,
    warmup_attempts: int = DEFAULT_CAMERA_WARMUP_ATTEMPTS,
    read_delay_seconds: float = DEFAULT_CAMERA_READ_DELAY_SECONDS,
    backend_candidates: Iterable[tuple[str, int | None]] | None = None,
) -> CameraOpenResult:
    errors: list[str] = []
    for backend_name, backend in backend_candidates or camera_backend_candidates(cv2):
        camera = _video_capture(cv2, camera_index, backend)
        if not camera.isOpened():
            _release_camera(camera)
            errors.append(f"{backend_name} did not open")
            continue

        _configure_camera(
            cv2,
            camera,
            width=width,
            height=height,
            camera_fourcc=camera_fourcc,
            zoom_out=zoom_out,
            camera_zoom=camera_zoom,
        )
        first_frame = read_camera_frame(
            camera,
            attempts=warmup_attempts,
            delay_seconds=read_delay_seconds,
        )
        if first_frame is None:
            _release_camera(camera)
            errors.append(f"{backend_name} opened but returned no frames")
            continue

        return CameraOpenResult(camera, backend_name, first_frame)

    detail = "; ".join(errors) if errors else "no camera backend was available"
    raise RuntimeError(
        f"Unable to capture frames from camera {camera_index}. "
        f"{detail}. Try a different camera number, close other apps using the webcam, "
        "or reconnect the camera."
    )


def camera_backend_candidates(cv2: Any) -> list[tuple[str, int | None]]:
    if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        return [("DirectShow", int(cv2.CAP_DSHOW))]
    return [("default", None)]


def read_camera_frame(
    camera: Any,
    *,
    attempts: int = 1,
    delay_seconds: float = 0.0,
) -> Any | None:
    for attempt in range(max(1, int(attempts))):
        ok, frame = camera.read()
        if ok and _frame_has_pixels(frame):
            return frame
        if delay_seconds > 0 and attempt + 1 < max(1, int(attempts)):
            time.sleep(delay_seconds)
    return None


def _video_capture(cv2: Any, camera_index: int, backend: int | None) -> Any:
    if backend is None:
        return cv2.VideoCapture(camera_index)
    return cv2.VideoCapture(camera_index, backend)


def _configure_camera(
    cv2: Any,
    camera: Any,
    *,
    width: int | None,
    height: int | None,
    camera_fourcc: str | None,
    zoom_out: bool,
    camera_zoom: float | None,
) -> None:
    fourcc = (camera_fourcc or "").strip()
    if len(fourcc) == 4:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    if width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if camera_zoom is not None:
        camera.set(cv2.CAP_PROP_ZOOM, float(camera_zoom))
    elif zoom_out:
        camera.set(cv2.CAP_PROP_ZOOM, 0.0)


def _frame_has_pixels(frame: Any) -> bool:
    if frame is None:
        return False
    shape = getattr(frame, "shape", None)
    if shape is None:
        return True
    return len(shape) >= 2 and shape[0] > 0 and shape[1] > 0


def _release_camera(camera: Any) -> None:
    try:
        camera.release()
    except Exception:
        pass
