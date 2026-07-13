from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlsplit


DEFAULT_DASHBOARD_PORT = 8765
LIVE_BOARD_SIZE = 12


def build_match_snapshot(app, now: float | None = None) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Return the small, read-only game state consumed by the website."""
    current_time = time.time() if now is None else float(now)
    current_player = _player_id(getattr(app, "_current_player", 1))
    timer_running = bool(getattr(app, "_timer_running", False))
    cart_countdown_active = bool(getattr(app, "_cart_countdown_running", False))

    turn_remaining = None
    if timer_running:
        turn_remaining = _remaining_seconds(getattr(app, "_turn_end_time", 0.0), current_time)

    cart_remaining = None
    if cart_countdown_active:
        cart_remaining = _remaining_seconds(
            getattr(app, "_cart_countdown_end_time", 0.0),
            current_time,
        )

    if bool(getattr(app, "_turn_scanning", False)) or bool(
        getattr(app, "_pending_turn_scan", False)
    ):
        game_status = "scanning"
    elif bool(getattr(app, "_ai_player_running", False)) or bool(
        getattr(app, "_pick_drop_running", False)
    ):
        game_status = "robot_moving"
    elif timer_running:
        game_status = "playing"
    elif bool(getattr(app, "_game_started", False)):
        game_status = "paused"
    else:
        game_status = "not_started"

    board_state = getattr(app, "_previous_board_state", {})
    if not isinstance(board_state, dict):
        board_state = {}
    player_1_cells = set(getattr(app, "_player_1_cells", set()))
    player_2_cells = set(getattr(app, "_player_2_cells", set()))
    board: list[list[str]] = []
    board_owners: list[list[int]] = []
    for row in range(LIVE_BOARD_SIZE):
        board_row: list[str] = []
        owner_row: list[int] = []
        for col in range(LIVE_BOARD_SIZE):
            square = f"{chr(ord('A') + col)}{row + 1}"
            letter = str(board_state.get(square, "")).strip().upper()
            board_row.append(letter[:1] if letter[:1].isalpha() else "")
            if square in player_1_cells:
                owner_row.append(1)
            elif square in player_2_cells:
                owner_row.append(2)
            else:
                owner_row.append(0)
        board.append(board_row)
        board_owners.append(owner_row)

    return {
        "player_1_score": _non_negative_int(getattr(app, "_player_1_score", 0)),
        "player_2_score": _non_negative_int(getattr(app, "_player_2_score", 0)),
        "current_player": current_player,
        "turn_countdown_seconds": turn_remaining,
        "turn_countdown_active": timer_running,
        "cart_countdown_seconds": cart_remaining,
        "cart_countdown_active": cart_countdown_active,
        "board_size": LIVE_BOARD_SIZE,
        "board": board,
        "board_owners": board_owners,
        "game_status": game_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class LiveDashboardServer:
    def __init__(
        self,
        snapshot_provider: Callable[[], dict[str, object]],
        host: str = "0.0.0.0",
        port: int = DEFAULT_DASHBOARD_PORT,
    ):
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def bound_port(self) -> int:
        if self._httpd is None:
            return self.port
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        if self._httpd is not None:
            return

        snapshot_provider = self.snapshot_provider

        class MatchRequestHandler(BaseHTTPRequestHandler):
            server_version = "ScrablifyDashboard/1.0"

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._send_common_headers(0)
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path.rstrip("/") or "/"
                if path == "/api/match":
                    try:
                        payload = snapshot_provider()
                        self._send_json(200, payload)
                    except Exception:
                        self._send_json(500, {"error": "Unable to read the live match state."})
                    return
                if path == "/health":
                    self._send_json(200, {"status": "ok"})
                    return
                self._send_json(404, {"error": "Not found"})

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_common_headers(len(body))
                self.end_headers()
                self.wfile.write(body)

            def _send_common_headers(self, content_length: int) -> None:
                self.send_header("Content-Length", str(content_length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Private-Network", "true")

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), MatchRequestHandler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="scrablify-live-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None


def dashboard_urls(port: int = DEFAULT_DASHBOARD_PORT) -> list[str]:
    addresses: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("8.8.8.8", 80))
            route_address = route_socket.getsockname()[0]
            if route_address and not route_address.startswith("127."):
                addresses.append(route_address)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address and not address.startswith("127.") and address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    return [f"http://{address}:{int(port)}" for address in addresses]


def _remaining_seconds(end_time: object, now: float) -> int:
    try:
        return max(0, int(float(end_time) - now))
    except (TypeError, ValueError):
        return 0


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _player_id(value: object) -> int:
    try:
        player = int(value)
    except (TypeError, ValueError):
        return 1
    return player if player in (1, 2) else 1
