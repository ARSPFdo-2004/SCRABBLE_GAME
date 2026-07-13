from __future__ import annotations

import json
import unittest
import urllib.request

from scrabble_plotter.live_dashboard import LiveDashboardServer, build_match_snapshot


class _GameState:
    _player_1_score = 31
    _player_2_score = 27
    _current_player = 2
    _timer_running = True
    _turn_end_time = 165.9
    _cart_countdown_running = True
    _cart_countdown_end_time = 112.2
    _game_started = True
    _turn_scanning = False
    _pending_turn_scan = False
    _ai_player_running = False
    _pick_drop_running = False
    _previous_board_state = {"A1": "C", "B1": "A", "C1": "T", "L12": "Z"}
    _player_1_cells = {"A1", "B1", "C1"}
    _player_2_cells = {"L12"}


class MatchSnapshotTests(unittest.TestCase):
    def test_snapshot_contains_live_scores_and_countdowns(self) -> None:
        snapshot = build_match_snapshot(_GameState(), now=100.0)

        self.assertEqual(snapshot["player_1_score"], 31)
        self.assertEqual(snapshot["player_2_score"], 27)
        self.assertEqual(snapshot["current_player"], 2)
        self.assertEqual(snapshot["turn_countdown_seconds"], 65)
        self.assertEqual(snapshot["cart_countdown_seconds"], 12)
        self.assertEqual(snapshot["game_status"], "playing")
        self.assertEqual(snapshot["board_size"], 12)
        self.assertEqual(snapshot["board"][0][:3], ["C", "A", "T"])
        self.assertEqual(snapshot["board_owners"][0][:3], [1, 1, 1])
        self.assertEqual(snapshot["board"][11][11], "Z")
        self.assertEqual(snapshot["board_owners"][11][11], 2)

    def test_scanning_status_overrides_paused_timer(self) -> None:
        state = _GameState()
        state._timer_running = False
        state._turn_scanning = True

        snapshot = build_match_snapshot(state, now=100.0)

        self.assertIsNone(snapshot["turn_countdown_seconds"])
        self.assertEqual(snapshot["game_status"], "scanning")


class LiveDashboardServerTests(unittest.TestCase):
    def test_match_endpoint_is_json_and_allows_the_website_origin(self) -> None:
        server = LiveDashboardServer(
            lambda: build_match_snapshot(_GameState(), now=100.0),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.bound_port}/api/match",
                timeout=2.0,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
                self.assertEqual(payload["player_1_score"], 31)
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
