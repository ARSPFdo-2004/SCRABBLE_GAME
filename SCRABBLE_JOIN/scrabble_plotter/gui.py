from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .board import BOARD_SIZE, CELL_SIZE_MM, parse_square_label
from .calibration import PlotterCalibration
from .camera import open_camera_capture, read_camera_frame
from .gemini_agent import (
    GeminiPlotterAgent,
    PlotterAgentAction,
)
from .image_calibration import collect_board_corners_from_frame
from .overlay import draw_camera_ocr_overlay
from .scanner import (
    BoardScanResult,
    CameraLetterScanResult,
    CameraWordScanResult,
    board_square_from_image_point,
    format_camera_words_numbered,
    scan_board_image,
    scan_camera_letters,
    scan_camera_words,
    select_best_frame,
)
from .word_bank import format_words_by_direction, matched_matrix_words, normalize_word
from .scoring import (
    ScoreResult,
    normalize_letter,
    premium_from_short_label,
    premium_short_label,
    score_board,
)
from .serial_sender import (
    BoardActuatorSender,
    GCodeSender,
    SerialConfig,
    format_move_command,
    list_serial_ports,
)


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = APP_ROOT / "scrabble_plotter_calibration.json"
LIVE_LETTER_SCAN_INTERVAL_SECONDS = 4.0
LIVE_WORD_SCAN_INTERVAL_SECONDS = 4.0
BEST_CAPTURE_FRAME_COUNT = 18
BEST_CAPTURE_TIMEOUT_SECONDS = 1.0
BEST_CAPTURE_FRAME_DELAY_SECONDS = 0.025
CAMERA_READ_FAILURE_LIMIT = 10
Z_DOWN_COMMAND = "ZU"
Z_UP_COMMAND = "ZD"
PICK_DROP_DEFAULT_DELAY_MS = 1000
PICK_DROP_MAGNET_DELAY_MS = 1000
PICK_DROP_MOVE_DELAY_MS = 1000
PICK_DROP_Z_SETTLE_DELAY_MS = 1000


class ScrabblePlotterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Scrabble Join")
        self.root.geometry("1240x820")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(800, self._send_startup_board_up)
        self.root.after(900, self._install_tile_rack_calibrate_button)
        self.root.after(1000, self._install_tile_rack_letters_panel)
        self.root.after(1050, self._install_live_board_calibrate_button)
        self.root.after(1150, self._install_board_cell_margin_feature)
        self._z_axis_controls_frame = None
        self.z_height_angle = tk.IntVar(value=80)
        self.root.after(500, self._add_z_axis_controls_to_move_section)

        self.calibration_path = tk.StringVar(value=str(DEFAULT_CALIBRATION_PATH.resolve()))
        self._calibration = PlotterCalibration.load(self.calibration_path.get())

        self.camera_index_var = tk.StringVar(value=str(self._calibration.camera_index))
        self.offset_x_var = tk.StringVar(value=str(self._calibration.offset_x_mm))
        self.offset_y_var = tk.StringVar(value=str(self._calibration.offset_y_mm))
        self.cell_size_var = tk.StringVar(value=str(self._calibration.cell_size_mm))
        self.x_steps_per_mm_var = tk.StringVar(value=str(self._calibration.x_steps_per_mm))
        self.y_steps_per_mm_var = tk.StringVar(value=str(self._calibration.y_steps_per_mm))
        self.cart_x_var = tk.StringVar(value=str(self._calibration.cart_x_mm))
        self.cart_y_var = tk.StringVar(value=str(self._calibration.cart_y_mm))
        self.ocr_confidence_threshold_var = tk.StringVar(value=str(self._calibration.ocr_confidence_threshold))
        self.ocr_cell_size_px_var = tk.StringVar(value=str(self._calibration.ocr_cell_size_px))
        self.square_var = tk.StringVar(value="H8")
        self.pick_square_var = tk.StringVar(value="TR1")
        self.drop_square_var = tk.StringVar(value="H8")
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.feed_rate_var = tk.StringVar(value="1500")
        self.command_var = tk.StringVar(value="G0")
        self.timeout_var = tk.StringVar(value="2.0")
        self.startup_g90_var = tk.BooleanVar(value=True)
        self.actuator_port_var = tk.StringVar(value=self._calibration.actuator_port)
        self.actuator_baud_var = tk.StringVar(value=str(self._calibration.actuator_baud))
        self.actuator_timeout_var = tk.StringVar(value=str(self._calibration.actuator_timeout))
        self.actuator_countdown_seconds_var = tk.StringVar(
            value=str(self._calibration.actuator_countdown_seconds)
        )
        self.actuator_word_var = tk.StringVar(value="")
        self.gemini_api_key_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))
        self.gemini_model_var = tk.StringVar(value="gemini-2.5-flash")
        self.gemini_objective_var = tk.StringVar(value="Choose the next board square.")
        self.gemini_include_camera_var = tk.BooleanVar(value=True)
        self.live_letter_scan_var = tk.BooleanVar(value=False)
        self.live_word_scan_var = tk.BooleanVar(value=True)
        self._last_agent_action: PlotterAgentAction | None = None
        self.blank_squares_var = tk.StringVar()
        self.captured_letters_var = tk.StringVar()
        self._letter_vars: list[list[tk.StringVar]] = [
            [tk.StringVar() for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
        self._letter_entries: list[list[tk.Entry]] = []
        self._premium_vars: list[list[tk.StringVar]] = [
            [
                tk.StringVar(value=premium_short_label(self._calibration.premium_layout[row][col]))
                for col in range(BOARD_SIZE)
            ]
            for row in range(BOARD_SIZE)
        ]
        self._last_scan: BoardScanResult | None = None
        self._last_camera_letter_scan: CameraLetterScanResult | None = None
        self._last_camera_word_scan: CameraWordScanResult | None = None
        self._camera_word_scan_running = False
        self._pending_actuator_challenge_after_word_scan = False
        self.camera_words_text: tk.Text | None = None

        self.status_var = tk.StringVar(
            value="Start the camera to find visible words."
        )
        self.preview_image = None
        self._preview_display_size = (0, 0)
        self._preview_source_size = (0, 0)
        self._camera = None
        self._camera_after_id: str | None = None
        self._latest_frame = None
        self._captured_photo_frame = None
        self._camera_failed_reads = 0
        self._camera_letter_scan_token = 0
        self._camera_word_scan_token = 0
        self._last_live_letter_scan_at = 0.0
        self._live_letter_scan_running = False
        self._last_live_letter_scan_error: str | None = None
        self._last_live_word_scan_at = 0.0
        self._last_live_word_scan_error: str | None = None
        self._sender: GCodeSender | None = None
        self._sender_key: tuple[str, int, float, bool] | None = None
        self._actuator_sender: BoardActuatorSender | None = None
        self._actuator_sender_key: tuple[str, int, float] | None = None

        self._build_ui()
        self.refresh_ports()
        self.load_calibration_into_form()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(frame, text="Live Camera", padding=10)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_frame.columnconfigure(4, weight=1)
        preview_frame.rowconfigure(2, weight=1)

        ttk.Label(preview_frame, text="Camera").grid(row=0, column=0, sticky="w")
        camera_selector = ttk.Combobox(
            preview_frame,
            textvariable=self.camera_index_var,
            values=[str(index) for index in range(6)],
            width=8,
        )
        camera_selector.grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Button(preview_frame, text="Start", command=self.start_camera).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(preview_frame, text="Stop", command=self.stop_camera).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(preview_frame, text="Calibrate Board", command=self.calibrate_board_from_camera).grid(
            row=0, column=4, sticky="w"
        )

        live_options = ttk.Frame(preview_frame)
        live_options.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            live_options,
            text="Live letters",
            variable=self.live_letter_scan_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            live_options,
            text="Live words",
            variable=self.live_word_scan_var,
        ).grid(row=0, column=1, sticky="w", padx=(14, 0))

        self.preview_label = ttk.Label(preview_frame, text="Camera stopped", anchor="center")
        self.preview_label.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
        self.preview_label.bind("<Button-1>", self._handle_camera_preview_click)

        controls_container = ttk.Frame(frame)
        controls_container.grid(row=0, column=1, sticky="nsew")
        controls_container.columnconfigure(0, weight=1)
        controls_container.rowconfigure(0, weight=1)

        controls_canvas = tk.Canvas(controls_container, highlightthickness=0)
        controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(controls_container, orient="vertical", command=controls_canvas.yview)
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = ttk.Frame(controls_canvas)
        controls.columnconfigure(0, weight=1)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def sync_controls_scroll_region(event) -> None:  # type: ignore[no-untyped-def]
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def sync_controls_width(event) -> None:  # type: ignore[no-untyped-def]
            controls_canvas.itemconfigure(controls_window, width=event.width)

        def scroll_controls_with_wheel(event) -> str:  # type: ignore[no-untyped-def]
            delta = event.delta
            if delta == 0:
                return "break"
            controls_canvas.yview_scroll(int(-delta / 120), "units")
            return "break"

        controls.bind("<Configure>", sync_controls_scroll_region)
        controls_canvas.bind("<Configure>", sync_controls_width)
        controls_canvas.bind_all("<MouseWheel>", scroll_controls_with_wheel)

        calibration_box = ttk.LabelFrame(controls, text="Board Settings", padding=10)
        calibration_box.grid(row=0, column=0, sticky="ew")
        calibration_box.columnconfigure(1, weight=1)

        ttk.Label(calibration_box, text="Calibration File").grid(row=0, column=0, sticky="w")
        ttk.Entry(calibration_box, textvariable=self.calibration_path).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(calibration_box, text="Browse", command=self.choose_calibration_file).grid(row=0, column=2, padx=(8, 0))

        fields = [
            ("Offset X mm", self.offset_x_var),
            ("Offset Y mm", self.offset_y_var),
            ("Cell Size mm", self.cell_size_var),
            ("X Steps/mm", self.x_steps_per_mm_var),
            ("Y Steps/mm", self.y_steps_per_mm_var),
            ("Cart X mm", self.cart_x_var),
            ("Cart Y mm", self.cart_y_var),
            ("OCR Min Confidence", self.ocr_confidence_threshold_var),
            ("OCR Cell Size px", self.ocr_cell_size_px_var),
        ]
        for index, (label, variable) in enumerate(fields, start=1):
            ttk.Label(calibration_box, text=label).grid(row=index, column=0, sticky="w", pady=2)
            entry = ttk.Entry(calibration_box, textvariable=variable)
            entry.grid(row=index, column=1, columnspan=2, sticky="ew", padx=(8, 0))

        ttk.Button(calibration_box, text="Save Board Settings", command=self.save_board_settings).grid(
            row=len(fields) + 1, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )
        ttk.Button(calibration_box, text="Send Step Settings", command=self.send_step_settings).grid(
            row=len(fields) + 2, column=0, columnspan=3, sticky="ew"
        )

        move_box = ttk.LabelFrame(controls, text="Move Plotter", padding=10)
        move_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        move_box.columnconfigure(1, weight=1)

        move_fields = [
            ("Square", self.square_var),
            ("Pickup", self.pick_square_var),
            ("Drop", self.drop_square_var),
            ("COM Port", self.port_var),
            ("Baud", self.baud_var),
            ("Feed Rate", self.feed_rate_var),
            ("G-code Command", self.command_var),
            ("Timeout", self.timeout_var),
        ]
        for index, (label, variable) in enumerate(move_fields):
            ttk.Label(move_box, text=label).grid(row=index, column=0, sticky="w", pady=2)
            ttk.Entry(move_box, textvariable=variable).grid(row=index, column=1, sticky="ew", padx=(8, 8))
            if label == "COM Port":
                ttk.Button(move_box, text="Refresh", command=self.refresh_ports).grid(row=index, column=2, sticky="ew")

        next_move_row = len(move_fields)
        ttk.Checkbutton(move_box, text="Send G90 before move", variable=self.startup_g90_var).grid(
            row=next_move_row, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Button(move_box, text="Preview G-code", command=self.preview_move).grid(
            row=next_move_row + 1, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )
        ttk.Button(move_box, text="Send Move", command=self.send_move).grid(
            row=next_move_row + 2, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(move_box, text="Reset To Start", command=self.reset_to_start).grid(
            row=next_move_row + 3, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Button(move_box, text="Go To Cart", command=self.go_to_cart).grid(
            row=next_move_row + 4, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Button(move_box, text="Pick And Drop", command=self.pick_and_drop).grid(
            row=next_move_row + 5, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )

        self._build_board_actuator_controls(controls, row=2)

        scan_box = ttk.LabelFrame(controls, text="Camera OCR", padding=10)
        scan_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        scan_box.columnconfigure(0, weight=1)

        scan_buttons = ttk.Frame(scan_box)
        scan_buttons.grid(row=0, column=0, sticky="ew")
        scan_buttons.columnconfigure(0, weight=1)
        scan_buttons.columnconfigure(1, weight=1)
        scan_buttons.columnconfigure(2, weight=1)
        ttk.Button(scan_buttons, text="Take Picture", command=self.take_picture_from_camera).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(scan_buttons, text="Find Words", command=self.identify_words_with_easyocr).grid(
            row=0, column=1, sticky="ew", padx=(0, 6)
        )
        ttk.Button(scan_buttons, text="Live View", command=self.resume_live_camera).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(scan_buttons, text="Capture Letters", command=self.capture_letters_from_camera).grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0)
        )
        ttk.Button(scan_buttons, text="Scan Board", command=self.scan_board_from_camera).grid(
            row=1, column=1, sticky="ew", padx=(0, 6), pady=(6, 0)
        )
        ttk.Button(scan_buttons, text="Calculate Score", command=self.calculate_score_from_board).grid(
            row=1, column=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(scan_buttons, text="Tile Rack Position", command=self.open_tile_rack_position_window).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )

        ttk.Label(scan_box, text="Captured Letters").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(scan_box, textvariable=self.captured_letters_var).grid(row=2, column=0, sticky="ew")

        board_grid = ttk.Frame(scan_box)
        board_grid.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for col in range(BOARD_SIZE):
            ttk.Label(board_grid, text=chr(ord("A") + col), width=3, anchor="center").grid(row=0, column=col + 1)
        self._letter_entries = []
        for row in range(BOARD_SIZE):
            ttk.Label(board_grid, text=str(row + 1), width=3, anchor="e").grid(row=row + 1, column=0, padx=(0, 3))
            entry_row: list[tk.Entry] = []
            for col in range(BOARD_SIZE):
                entry = tk.Entry(
                    board_grid,
                    textvariable=self._letter_vars[row][col],
                    width=3,
                    justify="center",
                    relief="solid",
                    borderwidth=1,
                )
                entry.grid(row=row + 1, column=col + 1, padx=1, pady=1)
                entry.bind(
                    "<Button-1>",
                    lambda event, row=row, col=col: self._handle_ocr_board_cell_click(row, col),
                    add="+",
                )
                entry.bind("<FocusOut>", lambda event: self._normalize_board_entries())
                entry.bind("<Return>", lambda event: self.calculate_score_from_board())
                entry_row.append(entry)
            self._letter_entries.append(entry_row)

        ttk.Label(scan_box, text="Matched Words").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.camera_words_text = tk.Text(scan_box, height=6, wrap="word", state="disabled")
        self.camera_words_text.grid(row=5, column=0, sticky="ew")

        ttk.Label(scan_box, text="Blank Squares").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(scan_box, textvariable=self.blank_squares_var).grid(row=7, column=0, sticky="ew")

        premium_box = ttk.LabelFrame(scan_box, text="Premium Layout (DL TL DW TW)", padding=6)
        premium_box.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        for col in range(BOARD_SIZE):
            ttk.Label(premium_box, text=chr(ord("A") + col), width=3, anchor="center").grid(row=0, column=col + 1)
        for row in range(BOARD_SIZE):
            ttk.Label(premium_box, text=str(row + 1), width=3, anchor="e").grid(row=row + 1, column=0, padx=(0, 3))
            for col in range(BOARD_SIZE):
                entry = tk.Entry(
                    premium_box,
                    textvariable=self._premium_vars[row][col],
                    width=3,
                    justify="center",
                    relief="solid",
                    borderwidth=1,
                )
                entry.grid(row=row + 1, column=col + 1, padx=1, pady=1)
                entry.bind("<FocusOut>", lambda event: self._normalize_premium_entries())
        ttk.Button(scan_box, text="Save Scan Settings", command=self.save_board_settings).grid(
            row=9, column=0, sticky="ew", pady=(10, 0)
        )

        agent_box = ttk.LabelFrame(controls, text="Gemini Agent", padding=10)
        agent_box.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        agent_box.columnconfigure(1, weight=1)

        agent_fields = [
            ("API Key", self.gemini_api_key_var),
            ("Model", self.gemini_model_var),
            ("Objective", self.gemini_objective_var),
        ]
        for index, (label, variable) in enumerate(agent_fields):
            ttk.Label(agent_box, text=label).grid(row=index, column=0, sticky="w", pady=2)
            show = "*" if label == "API Key" else ""
            ttk.Entry(agent_box, textvariable=variable, show=show).grid(
                row=index, column=1, columnspan=2, sticky="ew", padx=(8, 0)
            )

        ttk.Checkbutton(agent_box, text="Use latest camera frame", variable=self.gemini_include_camera_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Button(agent_box, text="Ask Gemini", command=self.ask_gemini).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )
        ttk.Button(agent_box, text="Run Gemini Action", command=self.run_gemini_action).grid(
            row=5, column=0, columnspan=3, sticky="ew"
        )

        log_box = ttk.LabelFrame(controls, text="Status", padding=10)
        log_box.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(1, weight=1)
        controls.rowconfigure(5, weight=1)

        ttk.Label(log_box, textvariable=self.status_var, wraplength=380).grid(row=0, column=0, sticky="ew")
        self.log_text = tk.Text(log_box, height=14, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def _build_board_actuator_controls(self, parent: ttk.Frame, row: int) -> None:
        actuator_box = ttk.LabelFrame(parent, text="Board Actuator Arduino", padding=10)
        actuator_box.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        actuator_box.columnconfigure(1, weight=1)

        ttk.Label(actuator_box, text="COM Port").grid(row=0, column=0, sticky="w", pady=2)
        self.actuator_port_combo = ttk.Combobox(
            actuator_box,
            textvariable=self.actuator_port_var,
            values=[],
            width=12,
        )
        self.actuator_port_combo.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=2)
        ttk.Button(actuator_box, text="Refresh", command=self.refresh_ports).grid(
            row=0, column=2, sticky="ew", pady=2
        )

        fields = [
            ("Baud", self.actuator_baud_var),
            ("Timeout", self.actuator_timeout_var),
            ("Countdown s", self.actuator_countdown_seconds_var),
            ("Word", self.actuator_word_var),
        ]
        for index, (label, variable) in enumerate(fields, start=1):
            ttk.Label(actuator_box, text=label).grid(row=index, column=0, sticky="w", pady=2)
            ttk.Entry(actuator_box, textvariable=variable).grid(
                row=index, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=2
            )

        buttons = [
            ("Save", self.save_actuator_settings),
            ("Test", self.test_actuator_connection),
            ("Status", self.request_actuator_status),
            ("Board Up", self.actuator_board_up),
            ("Board Down", self.actuator_board_down),
            ("Countdown", self.start_actuator_countdown),
            ("Stop Timer", self.stop_actuator_countdown),
            ("Challenge", self.start_actuator_challenge),
            ("Cancel Chal.", self.cancel_actuator_challenge),
            ("Prev Word", self.previous_actuator_word),
            ("Next Word", self.next_actuator_word),
            ("Send Word", self.send_actuator_word),
            ("Camera Words", self.send_camera_words_to_actuator),
            ("Light Letters", self.send_letter_leds_to_actuator),
            ("LED Test", self.test_actuator_leds),
            ("Reveal Word", self.reveal_actuator_word),
            ("Clear LEDs", self.clear_actuator_leds),
            ("Display On", self.actuator_display_on),
            ("Display Off", self.actuator_display_off),
        ]
        button_start_row = len(fields) + 1
        for index, (label, command) in enumerate(buttons):
            button_row = button_start_row + index // 3
            button_col = index % 3
            padx = (0, 6) if button_col < 2 else (0, 0)
            ttk.Button(actuator_box, text=label, command=command).grid(
                row=button_row,
                column=button_col,
                sticky="ew",
                padx=padx,
                pady=(8 if index < 3 else 6, 0),
            )

    def choose_calibration_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose calibration file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=Path(self.calibration_path.get()).name or "scrabble_plotter_calibration.json",
        )
        if not path:
            return
        self.calibration_path.set(path)
        self.load_calibration_into_form()

    def load_calibration_into_form(self) -> None:
        self._calibration = PlotterCalibration.load(self.calibration_path.get())
        self.camera_index_var.set(str(self._calibration.camera_index))
        self.offset_x_var.set(str(self._calibration.offset_x_mm))
        self.offset_y_var.set(str(self._calibration.offset_y_mm))
        self.cell_size_var.set(str(self._calibration.cell_size_mm))
        self.x_steps_per_mm_var.set(str(self._calibration.x_steps_per_mm))
        self.y_steps_per_mm_var.set(str(self._calibration.y_steps_per_mm))
        self.cart_x_var.set(str(self._calibration.cart_x_mm))
        self.cart_y_var.set(str(self._calibration.cart_y_mm))
        self.ocr_confidence_threshold_var.set(str(self._calibration.ocr_confidence_threshold))
        self.ocr_cell_size_px_var.set(str(self._calibration.ocr_cell_size_px))
        self.actuator_port_var.set(self._calibration.actuator_port)
        self.actuator_baud_var.set(str(self._calibration.actuator_baud))
        self.actuator_timeout_var.set(str(self._calibration.actuator_timeout))
        self.actuator_countdown_seconds_var.set(str(self._calibration.actuator_countdown_seconds))
        self._load_premium_layout_into_form()
        if len(self._calibration.image_corners) == 4:
            self._set_status("Loaded board calibration. Start the camera to find visible words.")
        else:
            self._set_status("Start the camera to find visible words.")

    def save_board_settings(self) -> None:
        try:
            self._calibration = self._calibration_from_form()
            self._calibration.save(self.calibration_path.get())
            self._set_status("Saved board settings.")
            self._log(
                f"Board offset saved: X={self._calibration.offset_x_mm:.3f}, "
                f"Y={self._calibration.offset_y_mm:.3f}; "
                f"steps/mm X={self._calibration.x_steps_per_mm:.3f}, "
                f"Y={self._calibration.y_steps_per_mm:.3f}; "
                f"cart X={self._calibration.cart_x_mm:.3f}, "
                f"Y={self._calibration.cart_y_mm:.3f}; "
                f"actuator port={self._calibration.actuator_port or 'not set'}"
            )
        except Exception as exc:
            self._show_error(exc)

    def save_actuator_settings(self) -> None:
        try:
            self._calibration = self._calibration_from_form()
            self._calibration.save(self.calibration_path.get())
            self._set_status("Saved board actuator settings.")
            self._log(
                "Board actuator saved: "
                f"port={self._calibration.actuator_port or 'not set'}, "
                f"baud={self._calibration.actuator_baud}, "
                f"timeout={self._calibration.actuator_timeout:.3f}, "
                f"countdown={self._calibration.actuator_countdown_seconds}s"
            )
        except Exception as exc:
            self._show_error(exc)

    def send_step_settings(self) -> None:
        try:
            calibration = self._calibration_from_form()
            calibration.save(self.calibration_path.get())
            sender = self._get_sender()
            command, responses = sender.send_step_config(
                calibration.x_steps_per_mm,
                calibration.y_steps_per_mm,
            )
            self._set_status("Sent step settings to the controller.")
            self._log(f"Sent: {command}")
            if responses:
                self._log("Responses: " + " | ".join(responses))
        except Exception as exc:
            self._show_error(exc)

    def start_camera(self) -> None:
        try:
            cv2 = _require_cv2()
            self.stop_camera(clear_preview=False)
            camera_index = int(self.camera_index_var.get())
            opened_camera = open_camera_capture(cv2, camera_index)

            self._camera = opened_camera.capture
            self._calibration = self._calibration_from_form()
            self._calibration.camera_index = camera_index
            self._calibration.save(self.calibration_path.get())
            self._latest_frame = opened_camera.first_frame.copy()
            self._captured_photo_frame = None
            self._camera_failed_reads = 0
            self._invalidate_camera_scans()
            self._last_live_letter_scan_at = 0.0
            self._last_live_letter_scan_error = None
            self._last_camera_letter_scan = None
            self._last_live_word_scan_at = 0.0
            self._last_live_word_scan_error = None
            self._last_camera_word_scan = None
            self.captured_letters_var.set("")
            self._set_camera_words_text("")
            self._show_frame(opened_camera.first_frame)
            self._set_status(f"Camera {camera_index} started ({opened_camera.backend_name}).")
            self._schedule_camera_update()
        except Exception as exc:
            self._show_error(exc)

    def stop_camera(self, clear_preview: bool = True) -> None:
        if self._camera_after_id is not None:
            self.root.after_cancel(self._camera_after_id)
            self._camera_after_id = None
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._camera_failed_reads = 0
        if clear_preview:
            self._latest_frame = None
            self._captured_photo_frame = None
            self._invalidate_camera_scans()
            self.preview_label.configure(image="", text="Camera stopped")
            self.preview_image = None
            self._preview_display_size = (0, 0)
            self._preview_source_size = (0, 0)

    def calibrate_board_from_camera(self) -> None:
        if self._latest_frame is None:
            raise_user_error("Start the camera first.")
            return

        try:
            corners = collect_board_corners_from_frame(self._latest_frame.copy())
            self._calibration = self._calibration_from_form()
            self._calibration.set_camera_corners(corners)
            self._calibration.save(self.calibration_path.get())
            self._set_status("Saved board corners for board scanning.")
            self._log(f"Board corners: {corners}")
            self._show_frame(self._latest_frame)
        except Exception as exc:
            self._show_error(exc)

    def preview_move(self) -> None:
        try:
            calibration = self._calibration_from_form()
            calibration.validate_ready_for_move()
            square = parse_square_label(self.square_var.get())
            x, y = calibration.square_center_in_machine(square)
            gcode = format_move_command(x, y, self._optional_float(self.feed_rate_var.get()), self.command_var.get())
            self._set_status(f"{square.label} -> X={x:.3f}, Y={y:.3f}")
            self._log(gcode)
        except Exception as exc:
            self._show_error(exc)

    def send_move(self) -> None:
        try:
            self._send_move_target(self.square_var.get())
        except Exception as exc:
            self._show_error(exc)

    def _send_move_target(self, target: str) -> None:
        target = str(target).strip().upper()
        if self._is_tile_rack_target(target):
            self._send_tile_rack_target_move(target)
            return
        self._send_square_move(target)

    def _is_tile_rack_target(self, target: str) -> bool:
        return (
            len(target) >= 3
            and target[:2] == "TR"
            and target[2:].isdigit()
            and 1 <= int(target[2:]) <= 7
        )

    def _send_tile_rack_target_move(self, target: str) -> None:
        self._ensure_tile_rack_move_state()
        slot_index = int(target[2:]) - 1
        x, y = self._tile_rack_slot_position(slot_index)
        feed = float(self.tile_rack_feed_var.get())
        command = f"G0 X{x:g} Y{y:g} F{feed:g}"
        self._send_tile_rack_move_command(command)
        self._set_status(f"Moved to {target} at X{x:g} Y{y:g}.")
        self._log(command)

    def _move_to_tile_rack_target_from_button(self, target: str) -> None:
        try:
            self._send_tile_rack_target_move(target)
        except Exception as exc:
            self._show_error(exc)

    def _install_tile_rack_calibrate_button(self) -> None:
        if getattr(self, "_tile_rack_calibrate_button_installed", False):
            return

        board_button = self._find_button_by_text(self.root, "Calibrate Board")
        if board_button is None:
            self.root.after(700, self._install_tile_rack_calibrate_button)
            return

        parent = board_button.master
        grid_info = board_button.grid_info()
        try:
            row = int(grid_info.get("row", 0))
            column = int(grid_info.get("column", 0)) + int(grid_info.get("columnspan", 1))
        except Exception:
            row = 0
            column = 0

        button = ttk.Button(parent, text="Calibrate Tile Rack", command=self.calibrate_tile_rack)
        button.grid(row=row, column=column, sticky=grid_info.get("sticky", ""), padx=(8, 0), pady=grid_info.get("pady", 0))
        self._tile_rack_calibrate_button_installed = True

    def _find_button_by_text(self, widget, text: str):  # type: ignore[no-untyped-def]
        try:
            if widget.winfo_class() in {"TButton", "Button"} and str(widget.cget("text")) == text:
                return widget
        except Exception:
            pass
        for child in widget.winfo_children():
            found = self._find_button_by_text(child, text)
            if found is not None:
                return found
        return None

    def _install_tile_rack_preview_click_handler(self) -> None:
        preview = getattr(self, "preview_label", None)
        if preview is None:
            self.root.after(700, self._install_tile_rack_preview_click_handler)
            return
        try:
            preview.bind("<Button-1>", self._handle_preview_click_with_tile_rack_calibration)
        except Exception:
            self.root.after(700, self._install_tile_rack_preview_click_handler)

    def _install_tile_rack_letters_panel(self) -> None:
        if getattr(self, "_tile_rack_letters_panel_installed", False):
            return

        anchor = self._find_button_by_text(self.root, "Tile Rack Position") or self._find_button_by_text(
            self.root, "Calibrate Tile Rack"
        )
        if anchor is None:
            self.root.after(700, self._install_tile_rack_letters_panel)
            return

        self._ensure_tile_rack_move_state()
        parent = anchor.master
        max_row = 0
        for child in parent.winfo_children():
            try:
                info = child.grid_info()
                max_row = max(max_row, int(info.get("row", 0)) + int(info.get("rowspan", 1)) - 1)
            except Exception:
                continue

        panel = ttk.LabelFrame(parent, text="Tile Rack Letters")
        panel.grid(row=max_row + 1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self._build_tile_rack_letter_grid(panel)
        ttk.Button(panel, text="Suggest Rack Words", command=self.suggest_tile_rack_words).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4)
        )
        ttk.Label(
            panel,
            textvariable=self.tile_rack_word_suggestions_var,
            anchor="nw",
            justify="left",
            wraplength=260,
        ).grid(row=8, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        self._tile_rack_letters_panel_installed = True

    def _install_board_cell_margin_feature(self) -> None:
        if getattr(self, "_board_cell_margin_feature_installed", False):
            return

        if not hasattr(self, "board_cell_margin_var"):
            self.board_cell_margin_var = tk.StringVar(value="0")

        self._wrap_calibration_form_with_cell_margin()
        if self._place_board_cell_margin_input():
            self._board_cell_margin_feature_installed = True
        else:
            self.root.after(700, self._install_board_cell_margin_feature)

    def _wrap_calibration_form_with_cell_margin(self) -> None:
        if getattr(self, "_board_cell_margin_calibration_wrapped", False):
            return

        original = getattr(self, "_calibration_from_form", None)
        if not callable(original):
            return

        def calibration_from_form_with_margin():  # type: ignore[no-untyped-def]
            calibration = original()
            try:
                calibration.cell_margin_mm = float(self.board_cell_margin_var.get())
            except Exception:
                calibration.cell_margin_mm = 0.0
            return calibration

        self._calibration_from_form = calibration_from_form_with_margin
        self._board_cell_margin_calibration_wrapped = True

    def _place_board_cell_margin_input(self) -> bool:
        anchor = self._find_widget_by_text_contains(self.root, "cell size")
        if anchor is None:
            anchor = self._find_button_by_text(self.root, "Calibrate Board")
        if anchor is None:
            return False

        parent = anchor.master
        try:
            grid_info = anchor.grid_info()
            row = int(grid_info.get("row", 0))
            column = int(grid_info.get("column", 0)) + int(grid_info.get("columnspan", 1)) + 1
            sticky = grid_info.get("sticky", "w")
        except Exception:
            row = 0
            column = 0
            sticky = "w"

        ttk.Label(parent, text="Cell margin").grid(row=row, column=column, sticky=sticky, padx=(8, 4), pady=2)
        ttk.Entry(parent, textvariable=self.board_cell_margin_var, width=8).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            padx=(0, 6),
            pady=2,
        )
        return True

    def _find_widget_by_text_contains(self, widget, text: str):  # type: ignore[no-untyped-def]
        wanted = text.lower()
        try:
            value = str(widget.cget("text")).lower()
            if wanted in value:
                return widget
        except Exception:
            pass
        for child in widget.winfo_children():
            found = self._find_widget_by_text_contains(child, text)
            if found is not None:
                return found
        return None

    def _install_live_board_calibrate_button(self) -> None:
        if getattr(self, "_live_board_calibrate_button_installed", False):
            return

        button = self._find_button_by_text(self.root, "Calibrate Board")
        if button is None:
            self.root.after(700, self._install_live_board_calibrate_button)
            return

        try:
            button.configure(command=self.calibrate_board_on_live_camera)
            self._live_board_calibrate_button_installed = True
        except Exception:
            self.root.after(700, self._install_live_board_calibrate_button)

    def calibrate_board_on_live_camera(self) -> None:
        frame = self._current_camera_ocr_frame()
        if frame is None:
            self._show_error(RuntimeError("Start the camera first so the board can be calibrated on the live image."))
            return

        self._board_corner_selection_active = True
        self._board_corner_selection_points = []
        self._bind_live_board_corner_clicks()
        self._set_status("Click board top-left corner on the live camera.")
        log = getattr(self, "_log", None)
        if callable(log):
            log("Board live-camera calibration started.")
        self._refresh_camera_preview()

    def _bind_live_board_corner_clicks(self) -> None:
        preview = getattr(self, "preview_label", None)
        if preview is None:
            raise RuntimeError("Start the camera first so the board can be calibrated on the live image.")

        if not getattr(self, "_board_live_click_binding_active", False):
            try:
                self._board_previous_preview_click_binding = preview.bind("<Button-1>")
            except Exception:
                self._board_previous_preview_click_binding = ""
        preview.bind("<Button-1>", self._handle_board_live_corner_click)
        self._board_live_click_binding_active = True

    def _restore_live_board_corner_clicks(self) -> None:
        preview = getattr(self, "preview_label", None)
        if preview is None:
            return

        previous_binding = getattr(self, "_board_previous_preview_click_binding", "")
        try:
            preview.bind("<Button-1>", previous_binding)
        except Exception:
            pass
        self._board_live_click_binding_active = False
        self._board_previous_preview_click_binding = ""

    def _handle_board_live_corner_click(self, event) -> None:  # type: ignore[no-untyped-def]
        image_point = self._preview_click_to_frame_point(event.x, event.y)
        if image_point is None:
            self._set_status("Click inside the live camera image.")
            return

        points = list(getattr(self, "_board_corner_selection_points", []))
        points.append((float(image_point[0]), float(image_point[1])))
        self._board_corner_selection_points = points

        corner_names = ("top-left", "top-right", "bottom-right", "bottom-left")
        if len(points) < 4:
            next_name = corner_names[len(points)]
            self._set_status(f"Board corner {len(points)} saved. Click {next_name} corner next.")
            self._refresh_camera_preview()
            return

        self._apply_live_board_corner_calibration(points[:4])
        self._board_corner_selection_active = False
        self._board_corner_selection_points = []
        self._restore_live_board_corner_clicks()
        self._set_status("Board calibrated from live camera corners.")
        self._refresh_camera_preview()

    def _apply_live_board_corner_calibration(self, corners) -> None:  # type: ignore[no-untyped-def]
        normalized = [[float(x), float(y)] for x, y in corners]
        grid = self._current_camera_ocr_grid_for_manual_lock()
        if grid is None:
            try:
                from .scanner import build_camera_ocr_grid

                frame = self._current_camera_ocr_frame()
                grid = build_camera_ocr_grid(frame) if frame is not None else None
            except Exception:
                grid = None

        if grid is not None:
            self._set_grid_corners(grid, normalized)
            self._manual_board_ocr_grid_lock = self._clone_camera_grid(grid)
            self._fallback_camera_ocr_grid = self._clone_camera_grid(grid)
        else:
            import types

            simple_grid = types.SimpleNamespace(corners=normalized, board_size=BOARD_SIZE)
            self._manual_board_ocr_grid_lock = simple_grid
            self._fallback_camera_ocr_grid = simple_grid

        log = getattr(self, "_log", None)
        if callable(log):
            log(f"Board live-camera calibration corners: {normalized}")

    def _install_manual_board_grid_lock(self) -> None:
        if getattr(self, "_manual_board_grid_lock_installed", False):
            return

        button = self._find_button_by_text(self.root, "Calibrate Board")
        if button is None:
            self.root.after(700, self._install_manual_board_grid_lock)
            return

        original = getattr(self, "_original_calibrate_camera_board", None)
        if original is None:
            original = getattr(self, "calibrate_camera_board", None)
            self._original_calibrate_camera_board = original
        if not callable(original):
            return

        try:
            button.configure(command=self._calibrate_camera_board_and_lock_grid)
            self._manual_board_grid_lock_installed = True
        except Exception:
            self.root.after(700, self._install_manual_board_grid_lock)

    def _calibrate_camera_board_and_lock_grid(self) -> None:
        original = getattr(self, "_original_calibrate_camera_board", None)
        if callable(original):
            original()
        self.root.after(300, lambda: self._capture_manual_board_grid_lock(remaining_attempts=20))

    def _capture_manual_board_grid_lock(self, remaining_attempts: int = 0) -> None:
        grid = self._current_camera_ocr_grid_for_manual_lock()
        if grid is None:
            if remaining_attempts > 0:
                self.root.after(300, lambda: self._capture_manual_board_grid_lock(remaining_attempts - 1))
            return

        self._manual_board_ocr_grid_lock = self._clone_camera_grid(grid)
        self._set_status("Manual board grid locked for board scans.")
        self._log("Manual board grid locked. Future board scans will reuse this grid layout.")

    def _current_camera_ocr_grid_for_manual_lock(self):  # type: ignore[no-untyped-def]
        for source_name in ("_last_camera_word_scan", "_last_camera_letter_scan"):
            scan = getattr(self, source_name, None)
            grid = getattr(scan, "grid", None)
            if grid is not None and getattr(grid, "corners", None):
                return grid

        for source_name in (
            "_manual_camera_ocr_grid",
            "_manual_board_ocr_grid",
            "_calibrated_camera_ocr_grid",
            "_camera_ocr_grid",
            "_fallback_camera_ocr_grid",
        ):
            grid = getattr(self, source_name, None)
            if grid is not None and getattr(grid, "corners", None):
                return grid

        try:
            grid = self._current_camera_ocr_grid()
            if grid is not None and getattr(grid, "corners", None):
                return grid
        except Exception:
            pass
        return None

    def _clone_camera_grid(self, grid):  # type: ignore[no-untyped-def]
        try:
            import copy

            return copy.deepcopy(grid)
        except Exception:
            return grid

    def _lock_scan_to_manual_board_grid(self, scan) -> None:  # type: ignore[no-untyped-def]
        locked_grid = getattr(self, "_manual_board_ocr_grid_lock", None)
        if locked_grid is None:
            return

        grid = getattr(scan, "grid", None)
        if grid is None:
            return

        locked_corners = getattr(locked_grid, "corners", None)
        if not locked_corners or len(locked_corners) != 4:
            return

        self._set_grid_corners(grid, locked_corners)

    def _apply_camera_letter_scan_to_locked_board_grid(self, scan) -> int:  # type: ignore[no-untyped-def]
        locked_grid = getattr(self, "_manual_board_ocr_grid_lock", None) or getattr(self, "_fallback_camera_ocr_grid", None)
        corners = getattr(locked_grid, "corners", None)
        entries = getattr(self, "_letter_entries", None)
        if not corners or len(corners) != 4 or not entries:
            return 0

        placements: dict[tuple[int, int], str] = {}
        rack_corners = getattr(self, "_tile_rack_calibrated_corners", None)
        for item in getattr(scan, "letters", []) or []:
            center = self._tile_rack_camera_item_center(item)
            if center is None:
                continue
            if rack_corners and len(rack_corners) == 4:
                rack_slot = self._tile_rack_slot_index_from_camera_corners(rack_corners, center[0], center[1])
                if rack_slot is not None:
                    continue
            row_col = self._board_row_col_from_camera_corners(corners, center[0], center[1])
            if row_col is None:
                continue
            letter = self._tile_rack_camera_item_letter(item)
            if letter:
                placements[row_col] = letter

        if not placements:
            return 0

        self._clear_main_board_ocr_entries()
        for (row, col), letter in placements.items():
            self._set_main_ocr_entry(row, col, letter)
        return len(placements)

    def _board_row_col_from_camera_corners(self, corners, x: float, y: float) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return None

        try:
            source = np.array(corners, dtype=np.float32)
            destination = np.array(
                [(0.0, 0.0), (float(BOARD_SIZE), 0.0), (float(BOARD_SIZE), float(BOARD_SIZE)), (0.0, float(BOARD_SIZE))],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(source, destination)
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(point, transform)[0][0]
        except Exception:
            return None

        grid_x = float(mapped[0])
        grid_y = float(mapped[1])
        if grid_x < -0.05 or grid_x > BOARD_SIZE + 0.05 or grid_y < -0.05 or grid_y > BOARD_SIZE + 0.05:
            return None
        col = max(0, min(BOARD_SIZE - 1, int(grid_x)))
        row = max(0, min(BOARD_SIZE - 1, int(grid_y)))
        return row, col

    def _clear_main_board_ocr_entries(self) -> None:
        entries = getattr(self, "_letter_entries", None)
        if not entries:
            return
        for row in range(min(BOARD_SIZE, len(entries))):
            for col in range(min(BOARD_SIZE, len(entries[row]))):
                self._clear_main_ocr_entry(row, col)

    def _set_main_ocr_entry(self, row: int, col: int, letter: str) -> None:
        try:
            entry = self._letter_entries[row][col]
        except Exception:
            return
        try:
            old_state = str(entry.cget("state"))
        except Exception:
            old_state = ""
        try:
            if old_state == "readonly":
                entry.configure(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, str(letter).upper()[:1])
        finally:
            if old_state == "readonly":
                entry.configure(state=old_state)

    def _handle_preview_click_with_tile_rack_calibration(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(self, "_tile_rack_corner_selection_active", False):
            self._handle_tile_rack_corner_preview_click(event)
            return
        self._handle_preview_click(event)

    def _handle_tile_rack_corner_preview_click(self, event) -> None:  # type: ignore[no-untyped-def]
        image_point = self._preview_click_to_frame_point(event.x, event.y)
        if image_point is None:
            self._set_status("Click inside the camera image.")
            return

        points = list(getattr(self, "_tile_rack_corner_selection_points", []))
        points.append((float(image_point[0]), float(image_point[1])))
        self._tile_rack_corner_selection_points = points

        corner_names = ("top-left", "top-right", "bottom-right", "bottom-left")
        if len(points) < 4:
            next_name = corner_names[len(points)]
            self.tile_rack_status_var.set(f"Rack corner {len(points)} saved. Click {next_name} corner next.")
            self._set_status(f"Rack corner {len(points)} saved. Click {next_name} corner next.")
            self._refresh_camera_preview()
            return

        self._tile_rack_calibrated_corners = points[:4]
        self._tile_rack_calibrated_rect = None
        self._last_tile_rack_camera_rect = None
        self._tile_rack_corner_selection_active = False
        self._tile_rack_corner_selection_points = []
        self._restore_live_tile_rack_corner_clicks()
        source = getattr(self, "_tile_rack_calibration_source", "camera image")
        self.tile_rack_status_var.set("Tile rack calibrated from clicked image corners.")
        self._set_status("Tile rack calibrated from the clicked corners.")
        self._log(f"Tile rack calibrated from {source}: {self._tile_rack_calibrated_corners}")
        self._refresh_camera_preview()

    def _bind_live_tile_rack_corner_clicks(self) -> None:
        preview = getattr(self, "preview_label", None)
        if preview is None:
            raise RuntimeError("Start the camera first so the tile rack can be calibrated on the live image.")

        if not getattr(self, "_tile_rack_live_click_binding_active", False):
            try:
                self._tile_rack_previous_preview_click_binding = preview.bind("<Button-1>")
            except Exception:
                self._tile_rack_previous_preview_click_binding = ""
        preview.bind("<Button-1>", self._handle_tile_rack_live_corner_click)
        self._tile_rack_live_click_binding_active = True

    def _restore_live_tile_rack_corner_clicks(self) -> None:
        preview = getattr(self, "preview_label", None)
        if preview is None:
            return

        previous_binding = getattr(self, "_tile_rack_previous_preview_click_binding", "")
        try:
            preview.bind("<Button-1>", previous_binding)
        except Exception:
            pass
        self._tile_rack_live_click_binding_active = False
        self._tile_rack_previous_preview_click_binding = ""

    def _handle_tile_rack_live_corner_click(self, event) -> None:  # type: ignore[no-untyped-def]
        self._handle_tile_rack_corner_preview_click(event)

    def calibrate_tile_rack(self) -> None:
        self._ensure_tile_rack_move_state()
        try:
            frame = getattr(self, "_last_displayed_camera_frame_for_ocr", None)
            if frame is None:
                frame = getattr(self, "_latest_frame", None)
            if frame is None:
                frame, quality = self._capture_best_photo_for_ocr("tile rack calibration")
                self._captured_photo_frame = frame.copy()
                source = f"best camera frame, sharpness {quality.sharpness:.0f}"
            else:
                source = "live camera"

            self._tile_rack_corner_selection_active = True
            self._tile_rack_corner_selection_points = []
            self._tile_rack_calibrated_corners = None
            self._tile_rack_calibrated_rect = None
            self._last_tile_rack_camera_rect = None
            self._tile_rack_calibration_source = source
            self._bind_live_tile_rack_corner_clicks()
            self.tile_rack_status_var.set(
                "Click tile rack corners on the live camera: top-left, top-right, bottom-right, bottom-left."
            )
            self._set_status("Click top-left tile rack corner on the live camera.")
            self._log(f"Tile rack live-camera calibration started from {source}.")
            refresh = getattr(self, "_refresh_camera_preview", None)
            if callable(refresh):
                refresh()
        except Exception as exc:
            self._show_error(exc)

    def _open_tile_rack_image_corner_window(self, frame, source: str) -> None:  # type: ignore[no-untyped-def]
        existing_window = getattr(self, "_tile_rack_image_corner_window", None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return

        self._tile_rack_corner_image_frame = frame.copy()
        self._tile_rack_corner_selection_points = []
        self._tile_rack_corner_selection_active = False
        self._tile_rack_calibrated_corners = None
        self._tile_rack_calibrated_rect = None
        self._last_tile_rack_camera_rect = None
        self._tile_rack_calibration_source = source

        window = tk.Toplevel(self.root)
        self._tile_rack_image_corner_window = window
        window.title("Calibrate Tile Rack")
        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=0)
        window.rowconfigure(1, weight=1)

        ttk.Label(
            window,
            text="Click the tile rack corners on the image in this order: top-left, top-right, bottom-right, bottom-left.",
            wraplength=760,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))

        self._tile_rack_corner_image_label = ttk.Label(window, cursor="crosshair")
        self._tile_rack_corner_image_label.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=6)
        self._tile_rack_corner_image_label.bind("<Button-1>", self._handle_tile_rack_image_corner_click)

        side = ttk.Frame(window)
        side.grid(row=1, column=1, sticky="ns", padx=(6, 10), pady=6)
        self._tile_rack_corner_status_var = tk.StringVar(value="Next: top-left")
        ttk.Label(side, textvariable=self._tile_rack_corner_status_var, wraplength=190).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        corner_names = ("top-left", "top-right", "bottom-right", "bottom-left")
        self._tile_rack_image_corner_value_vars = []
        for index, name in enumerate(corner_names):
            ttk.Label(side, text=name).grid(row=index + 1, column=0, sticky="w", pady=3)
            value_var = tk.StringVar(value="not selected")
            self._tile_rack_image_corner_value_vars.append(value_var)
            ttk.Label(side, textvariable=value_var, width=18).grid(row=index + 1, column=1, sticky="w", pady=3)

        ttk.Button(side, text="Reset Corners", command=self._reset_tile_rack_image_corner_selection).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(10, 4)
        )
        ttk.Button(side, text="Close", command=window.destroy).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=4
        )

        self.tile_rack_status_var.set("Click top-left tile rack corner in the calibration image.")
        self._set_status("Click top-left tile rack corner in the calibration image.")
        self._refresh_tile_rack_corner_window_image()

    def _handle_tile_rack_image_corner_click(self, event) -> None:  # type: ignore[no-untyped-def]
        frame = getattr(self, "_tile_rack_corner_image_frame", None)
        if frame is None:
            return

        frame_height, frame_width = frame.shape[:2]
        image_width, image_height = getattr(self, "_tile_rack_corner_display_size", (0, 0))
        if image_width <= 0 or image_height <= 0:
            return
        if event.x < 0 or event.y < 0 or event.x > image_width or event.y > image_height:
            self._set_status("Click inside the tile rack calibration image.")
            return

        points = list(getattr(self, "_tile_rack_corner_selection_points", []))
        if len(points) >= 4:
            return

        x = float(event.x) * float(frame_width) / float(image_width)
        y = float(event.y) * float(frame_height) / float(image_height)
        points.append((x, y))
        self._tile_rack_corner_selection_points = points

        corner_names = ("top-left", "top-right", "bottom-right", "bottom-left")
        value_vars = getattr(self, "_tile_rack_image_corner_value_vars", [])
        if len(points) <= len(value_vars):
            value_vars[len(points) - 1].set(f"X{int(round(x))} Y{int(round(y))}")

        if len(points) < 4:
            next_name = corner_names[len(points)]
            self._tile_rack_corner_status_var.set(f"Next: {next_name}")
            self.tile_rack_status_var.set(f"Rack corner {len(points)} saved. Click {next_name}.")
            self._set_status(f"Rack corner {len(points)} saved. Click {next_name}.")
        else:
            self._apply_tile_rack_image_corner_selection()

        self._refresh_tile_rack_corner_window_image()

    def _apply_tile_rack_image_corner_selection(self) -> None:
        points = list(getattr(self, "_tile_rack_corner_selection_points", []))
        if len(points) != 4:
            return

        self._tile_rack_calibrated_corners = points[:4]
        self._tile_rack_calibrated_rect = None
        self._last_tile_rack_camera_rect = None
        self._tile_rack_corner_selection_active = False
        source = getattr(self, "_tile_rack_calibration_source", "camera image")
        if hasattr(self, "_tile_rack_corner_status_var"):
            self._tile_rack_corner_status_var.set("Done. Tile rack corners calibrated.")
        self.tile_rack_status_var.set("Tile rack calibrated from image corners.")
        self._set_status("Tile rack calibrated from image corners.")
        self._log(f"Tile rack calibrated from {source}: {self._tile_rack_calibrated_corners}")
        refresh = getattr(self, "_refresh_camera_preview", None)
        if callable(refresh):
            refresh()

    def _reset_tile_rack_image_corner_selection(self) -> None:
        self._tile_rack_corner_selection_points = []
        for value_var in getattr(self, "_tile_rack_image_corner_value_vars", []):
            value_var.set("not selected")
        if hasattr(self, "_tile_rack_corner_status_var"):
            self._tile_rack_corner_status_var.set("Next: top-left")
        self.tile_rack_status_var.set("Click top-left tile rack corner in the calibration image.")
        self._set_status("Click top-left tile rack corner in the calibration image.")
        self._refresh_tile_rack_corner_window_image()

    def _refresh_tile_rack_corner_window_image(self) -> None:
        frame = getattr(self, "_tile_rack_corner_image_frame", None)
        label = getattr(self, "_tile_rack_corner_image_label", None)
        if frame is None or label is None:
            return

        cv2 = _require_cv2()
        display = self._draw_tile_rack_pending_corner_overlay(frame.copy())
        points = list(getattr(self, "_tile_rack_corner_selection_points", []))
        if len(points) == 4:
            display = self._draw_tile_rack_corner_grid_overlay(display, points)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk

        image = Image.fromarray(rgb)
        image.thumbnail((760, 560))
        self._tile_rack_corner_display_size = image.size
        self._tile_rack_corner_photo = ImageTk.PhotoImage(image)
        label.configure(image=self._tile_rack_corner_photo)

    def _open_tile_rack_corner_window(self, corners, source: str) -> None:  # type: ignore[no-untyped-def]
        window = tk.Toplevel(self.root)
        self._tile_rack_corner_window = window
        window.title("Calibrate Tile Rack")
        window.resizable(False, False)
        window.columnconfigure(1, weight=1)
        window.columnconfigure(2, weight=1)

        ttk.Label(window, text="Corner").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(window, text="X").grid(row=0, column=1, sticky="w", padx=6, pady=(10, 4))
        ttk.Label(window, text="Y").grid(row=0, column=2, sticky="w", padx=6, pady=(10, 4))

        labels = ("Top left", "Top right", "Bottom right", "Bottom left")
        self._tile_rack_corner_vars = []
        for index, label in enumerate(labels):
            x, y = corners[index]
            x_var = tk.StringVar(value=f"{float(x):.0f}")
            y_var = tk.StringVar(value=f"{float(y):.0f}")
            self._tile_rack_corner_vars.append((x_var, y_var))
            ttk.Label(window, text=label).grid(row=index + 1, column=0, sticky="w", padx=10, pady=3)
            ttk.Entry(window, textvariable=x_var, width=8).grid(row=index + 1, column=1, sticky="ew", padx=6, pady=3)
            ttk.Entry(window, textvariable=y_var, width=8).grid(row=index + 1, column=2, sticky="ew", padx=6, pady=3)

        ttk.Button(window, text="Apply Tile Rack Corners", command=self._apply_tile_rack_corner_calibration).grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 4)
        )
        ttk.Button(window, text="Use Other Side Default", command=self._set_tile_rack_other_side_default_corners).grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=4
        )
        ttk.Button(window, text="Use Brown Rack Detection", command=self._set_tile_rack_detected_corners).grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=10, pady=4
        )
        ttk.Label(window, text=f"Source: {source}", wraplength=260).grid(
            row=8, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 10)
        )

    def _apply_tile_rack_corner_calibration(self) -> None:
        try:
            corners = []
            for x_var, y_var in self._tile_rack_corner_vars:
                corners.append((float(x_var.get()), float(y_var.get())))
            if len(corners) != 4:
                raise ValueError("Give all four tile rack corners.")

            self._tile_rack_calibrated_corners = corners
            self._tile_rack_calibrated_rect = None
            self._last_tile_rack_camera_rect = None
            self.tile_rack_status_var.set("Tile rack corners calibrated. Rack grid adjusted to the given corners.")
            self._set_status("Tile rack corners calibrated.")
            self._log(f"Tile rack corner calibration: {corners}")
            refresh = getattr(self, "_refresh_camera_preview", None)
            if callable(refresh):
                refresh()
        except Exception as exc:
            self._show_error(exc)

    def _set_tile_rack_other_side_default_corners(self) -> None:
        try:
            frame = getattr(self, "_captured_photo_frame", None)
            if frame is None:
                frame, _quality = self._capture_best_photo_for_ocr("tile rack side default")
            height, width = frame.shape[:2]

            current = self._read_tile_rack_corner_values()
            current_center_x = sum(point[0] for point in current) / 4.0 if current else width * 0.2
            rack_width = max(30.0, abs(current[1][0] - current[0][0]) if current else width * 0.08)
            rack_height = max(120.0, abs(current[3][1] - current[0][1]) if current else height * 0.65)
            margin_x = max(5.0, width * 0.03)
            margin_y = max(5.0, height * 0.15)

            if current_center_x < width / 2.0:
                x = width - margin_x - rack_width
            else:
                x = margin_x
            y = margin_y
            self._write_tile_rack_corner_values(
                [
                    (x, y),
                    (x + rack_width, y),
                    (x + rack_width, y + rack_height),
                    (x, y + rack_height),
                ]
            )
        except Exception as exc:
            self._show_error(exc)

    def _set_tile_rack_detected_corners(self) -> None:
        try:
            frame = getattr(self, "_captured_photo_frame", None)
            if frame is None:
                frame, _quality = self._capture_best_photo_for_ocr("tile rack brown detection")
            rect = self._detect_tile_rack_brown_rect(frame)
            if rect is None:
                raise ValueError("Could not find the brown tile rack area clearly.")
            x, y, width, height = rect
            self._write_tile_rack_corner_values(
                [
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                ]
            )
        except Exception as exc:
            self._show_error(exc)

    def _read_tile_rack_corner_values(self) -> list[tuple[float, float]]:
        corners = []
        for x_var, y_var in getattr(self, "_tile_rack_corner_vars", []):
            corners.append((float(x_var.get()), float(y_var.get())))
        return corners

    def _write_tile_rack_corner_values(self, corners) -> None:  # type: ignore[no-untyped-def]
        for index, (x, y) in enumerate(corners):
            x_var, y_var = self._tile_rack_corner_vars[index]
            x_var.set(f"{float(x):.0f}")
            y_var.set(f"{float(y):.0f}")

    def _build_tile_rack_letter_grid(self, parent) -> None:  # type: ignore[no-untyped-def]
        self._ensure_tile_rack_move_state()
        for index, variable in enumerate(self.tile_rack_letter_vars):
            ttk.Label(parent, text=f"TR{index + 1}").grid(row=index, column=0, sticky="e", padx=(6, 4), pady=1)
            entry = ttk.Entry(parent, textvariable=variable, width=4, justify="center")
            entry.grid(row=index, column=1, sticky="w", padx=(0, 6), pady=1)
            entry.configure(state="readonly")

    def suggest_tile_rack_words(self) -> None:
        self._ensure_tile_rack_move_state()
        try:
            rack_letters = self._tile_rack_letters_for_word_suggestions()
            if len(rack_letters) < 2:
                raise ValueError("Detect or enter at least 2 tile rack letters first.")

            words = self._tile_rack_words_from_letters(rack_letters)
            if not words:
                message = f"No words found from rack letters: {''.join(rack_letters)}"
                self.tile_rack_word_suggestions_var.set(message)
                self._set_status(message)
                return

            shown_words = words[:80]
            display = "\n".join(
                f"{index + 1}. {word} ({len(word)} letters, {self._scrabble_word_score(word)} pts)"
                for index, word in enumerate(shown_words)
            )
            if len(words) > len(shown_words):
                display += f"\n... {len(words) - len(shown_words)} more"
            self.tile_rack_word_suggestions_var.set(display)
            self._set_status(f"Suggested {len(words)} word(s) from rack letters: {''.join(rack_letters)}.")
            self._log("Tile rack word suggestions:\n" + display)
        except Exception as exc:
            self._show_error(exc)

    def _tile_rack_letters_for_word_suggestions(self) -> list[str]:
        letters: list[str] = []
        for variable in getattr(self, "tile_rack_letter_vars", []):
            try:
                value = variable.get()
            except Exception:
                value = ""
            for character in str(value).upper():
                if character.isalpha():
                    letters.append(character)
                    break
        return letters[:7]

    def _tile_rack_words_from_letters(self, rack_letters: list[str]) -> list[str]:
        from collections import Counter

        available = Counter(letter.upper() for letter in rack_letters if letter.isalpha())
        words = []
        seen = set()
        for word in self._tile_rack_candidate_words():
            normalized = "".join(character for character in str(word).upper() if character.isalpha())
            if len(normalized) < 2 or len(normalized) > len(rack_letters) or normalized in seen:
                continue
            needed = Counter(normalized)
            if all(needed[letter] <= available[letter] for letter in needed):
                seen.add(normalized)
                words.append(normalized)

        return sorted(words, key=lambda word: (-len(word), -self._scrabble_word_score(word), word))

    def _tile_rack_candidate_words(self) -> list[str]:
        cached = getattr(self, "_tile_rack_candidate_word_cache", None)
        if cached is not None:
            return cached

        words = self._tile_rack_candidate_words_from_modules()
        if not words:
            words = self._tile_rack_candidate_words_from_files()
        if not words:
            words = self._fallback_tile_rack_words()

        self._tile_rack_candidate_word_cache = words
        return words

    def _tile_rack_candidate_words_from_modules(self) -> list[str]:
        import importlib

        module_names = (
            "scrabble_plotter.words",
            "scrabble_plotter.dictionary",
            "scrabble_plotter.scoring",
            "scrabble_plotter.solver",
            "scrabble_plotter.word_finder",
        )
        attribute_names = (
            "VALID_WORDS",
            "WORDS",
            "WORD_LIST",
            "DICTIONARY",
            "SCRABBLE_WORDS",
            "COMMON_WORDS",
        )
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attribute_name in attribute_names:
                values = getattr(module, attribute_name, None)
                if values:
                    try:
                        return list(values)
                    except Exception:
                        continue
        return []

    def _tile_rack_candidate_words_from_files(self) -> list[str]:
        from pathlib import Path

        names = (
            "words.txt",
            "wordlist.txt",
            "word_list.txt",
            "dictionary.txt",
            "scrabble_words.txt",
            "sowpods.txt",
            "twl.txt",
            "twl06.txt",
            "enable1.txt",
        )
        roots = []
        try:
            package_root = Path(__file__).resolve().parent
            roots.extend([package_root, package_root.parent, package_root / "data", package_root.parent / "data"])
        except Exception:
            pass
        try:
            roots.append(Path.cwd())
        except Exception:
            pass

        for root in roots:
            for name in names:
                path = root / name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    words = [
                        line.strip().upper()
                        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        if line.strip()
                    ]
                except Exception:
                    continue
                if words:
                    return words
        return []

    def _scrabble_word_score(self, word: str) -> int:
        scores = {
            "A": 1,
            "E": 1,
            "I": 1,
            "O": 1,
            "U": 1,
            "L": 1,
            "N": 1,
            "S": 1,
            "T": 1,
            "R": 1,
            "D": 2,
            "G": 2,
            "B": 3,
            "C": 3,
            "M": 3,
            "P": 3,
            "F": 4,
            "H": 4,
            "V": 4,
            "W": 4,
            "Y": 4,
            "K": 5,
            "J": 8,
            "X": 8,
            "Q": 10,
            "Z": 10,
        }
        return sum(scores.get(character.upper(), 0) for character in word)

    def _fallback_tile_rack_words(self) -> list[str]:
        return """
        A AN AM AS AT BE BY DO GO HE HI IF IN IS IT ME MY NO OF OH ON OR OX SO TO UP US WE
        ACE ACT ADD AGE AGO AID AIM AIR ALE ALL AND ANT ANY APE ARC ARE ARM ART ASH ASK ATE
        BAD BAG BAN BAR BAT BAY BED BEE BET BID BIG BIN BIT BOB BOG BOW BOX BOY BUD BUG BUN
        BUS BUT BUY CAB CAD CAM CAN CAP CAR CAT COB COD COG CON COP COT COW COY CUB CUE CUP
        CUT DAB DAD DAM DAY DEN DID DIE DIG DIM DIN DIP DOG DOT DRY DUE DUG EAR EAT EGG EGO
        ELF END ERA FAR FAT FED FEE FEN FEW FIG FIN FIR FIT FIX FLY FOG FOR FOX FRY FUN FUR
        GAP GAS GEL GET GIN GOD GOT GUM GUN GUY HAD HAM HAS HAT HAY HEN HER HIM HIP HIS HIT
        HOP HOT HOW HUG HUT ICE ILL INK JAM JAR JAW JET JOB JOG JOY KEY KID KIN KIT LAB LAD
        LAG LAP LAW LAY LED LEG LET LID LIE LIP LOG LOT LOW MAD MAN MAP MAT MAY MEN MET MIX
        MOM MUD MUG NAP NET NEW NOD NOR NOT NOW NUT OAK OAR ODD OFF OLD ONE OUR OUT OWL PAD
        PAL PAN PAR PAT PAW PAY PEA PEN PET PIE PIN PIT POD POP POT RAG RAM RAN RAT RAW RAY
        RED RID RIG RIP ROB ROD ROT ROW RUB RUG RUN SAD SAG SAT SAW SAY SEA SEE SET SHE SHY
        SIN SIP SIR SIT SIX SKY SLY SON SUN TAB TAG TAN TAP TAR TAX TEA TEN THE TIE TIN TOE
        TON TOP TOY TRY TUB TWO USE VAN VAT VET VIA WAR WAS WAY WEB WET WHO WHY WIN WIT WON
        YES YET YOU ZAP ZIP ZOO
        ABLE ACID ACRE ACTS AGES AIMS AIRS ALSO AREA ARMS ARMY ARTS BACK BAGS BAKE BALL BAND
        BARE BARK BARN BASE BATH BEAD BEAM BEAN BEAR BEAT BEND BEST BIRD BITE BOAT BOLD BONE
        BOOK BORN BOWL CAFE CAGE CAKE CALL CALM CAME CAMP CARD CARE CART CASE CASH CAST CATS
        CHAT CLAY COAL COAT CODE COLD COME COOK COOL CORD CORE COST CUBE CURE DARK DATE DAWN
        DAYS DEAL DEAR DECK DEEP DICE DIRT DOOR DOWN DRAW DROP DUST EACH EARN EAST EASY ECHO
        EDGE FAIR FALL FARM FAST FEAR FEED FEEL FILE FILL FILM FIND FINE FIRE FISH FIVE FLAG
        FLAT FLOW FOOD FOOT FORM FOUR FREE FROM GAME GATE GEAR GIFT GIRL GIVE GOAL GOLD GOOD
        GRAY GROW HAIR HALF HAND HARD HARM HATE HAVE HEAD HEAR HEAT HELP HILL HOLD HOME HOPE
        HOUR IDEA INTO IRON JACK JOIN JUMP KEEP KIND KING KITE LACE LACK LADY LAKE LAND LANE
        LAST LATE LEAD LEAF LEFT LEND LIFE LINE LINK LIST LIVE LOAD LOCK LONG LOOK LORD LOVE
        MADE MAIL MAIN MAKE MANY MARK MATE MEAL MEAN MEET MILE MILK MIND MINE MINT MOON MORE
        MOST MOVE NAME NEAR NEED NEST NICE NINE NOTE OPEN OVER PACK PAGE PAID PAIR PARK PART
        PAST PATH PEAK PEAR PICK PILE PINK PLAN PLAY PLOT PLUS POEM POND POOL PORT POST PULL
        PUSH RACE RAIN READ REAL REST RICE RIDE RING ROAD ROCK ROLE ROPE ROSE RULE SAND SAVE
        SEAT SEED SEEN SELF SEND SHIP SHOP SIDE SIGN SING SINK SITE SLOW SNOW SOFT SOLD SOME
        SONG SOON SORT STAR STAY STEP STOP SUCH SUIT SURE TAKE TALE TALL TAPE TEAM TELL TEND
        TENT THAN THAT THEM THEN THEY THIN THIS TIME TONE TOOL TREE TRIP TRUE TURN UNIT UPON
        USED USER VAST VERY VIEW WALK WALL WANT WARM WASH WAVE WEAK WEAR WEEK WELL WENT WERE
        WEST WHAT WHEN WILL WIND WING WIRE WISE WITH WORD WORK YARD YEAR
        ABOUT ABOVE ACTOR AFTER AGAIN AGREE ALARM ALIVE ALLOW ALONE AMONG ANGLE APPLE APPLY
        BASIC BEACH BEGIN BLACK BOARD BRAIN BREAD BRING BROWN BUILD CHAIR CHARM CHECK CLEAN
        CLEAR CLOSE CLOUD COAST COUNT COVER CREAM CROSS DANCE DREAM DRINK EARLY EARTH EMPTY
        ENTER FIELD FIRST FLOOR FOUND FRAME FRESH FRONT FRUIT GLASS GRASS GREAT GREEN GROUP
        GUARD GUESS HAPPY HEART HEAVY HOUSE HUMAN IMAGE LARGE LATER LAUGH LEARN LIGHT LOCAL
        MAGIC MAJOR MARCH MATCH MONEY MONTH MUSIC NEVER NIGHT NORTH OCEAN OTHER PAPER PARTY
        PEACE PHONE PIECE PLACE PLAIN PLANT POINT POWER PRESS PRICE QUICK RADIO READY RIGHT
        RIVER ROUND SCALE SCORE SHARE SHORT SMALL SMART SOUND SOUTH SPACE SPEAK SPEED SPELL
        SPEND SPORT STAND START STATE STONE STORE STORY SWEET TABLE TEACH THANK THEIR THERE
        THING THINK THREE TODAY TOTAL TOUCH TRAIN UNDER UNTIL VALUE VIDEO VISIT VOICE WATER
        WHERE WHITE WHOLE WORLD WRITE YOUNG
        """.split()

    def capture_tile_rack_letters(self) -> None:
        self._ensure_tile_rack_move_state()
        try:
            live_frame = getattr(self, "_last_displayed_camera_frame_for_ocr", None)
            if live_frame is None:
                live_frame = getattr(self, "_latest_frame", None)
            if live_frame is not None:
                frame = live_frame.copy()
                source = "live camera"
            else:
                frame, quality = self._capture_best_photo_for_ocr("tile rack letters")
                source = f"best live camera frame, sharpness {quality.sharpness:.0f}"

            scan = scan_camera_letters(frame)
            letters = self._tile_rack_letters_from_camera(frame, scan)
            detected_count = 0
            lines = []
            for index, variable in enumerate(self.tile_rack_letter_vars):
                letter = letters[index] if index < len(letters) else ""
                variable.set(letter)
                if letter:
                    detected_count += 1
                x, y = self._tile_rack_slot_position(index)
                lines.append(f"TR{index + 1}: {letter or '-'}  X{x:g} Y{y:g}")

            result = "\n".join(lines)
            self.tile_rack_status_var.set(result)
            self._set_status(f"Captured {detected_count} tile rack letter(s) from the {source}.")
            self._log("Tile rack letters:\n" + result)
            refresh = getattr(self, "_refresh_camera_preview", None)
            if callable(refresh):
                refresh()
        except Exception as exc:
            self._show_error(exc)

    def _update_tile_rack_letters_from_scan(self, frame, scan, source: str = "camera OCR") -> int:  # type: ignore[no-untyped-def]
        self._ensure_tile_rack_move_state()
        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if not corners or len(corners) != 4:
            return 0

        letters = self._tile_rack_letters_from_camera(frame, scan)
        detected_count = 0
        lines = []
        for index, variable in enumerate(self.tile_rack_letter_vars):
            letter = letters[index] if index < len(letters) else ""
            variable.set(letter)
            if letter:
                detected_count += 1
            x, y = self._tile_rack_slot_position(index)
            lines.append(f"TR{index + 1}: {letter or '-'}  X{x:g} Y{y:g}")

        result = "\n".join(lines)
        self.tile_rack_status_var.set(result)
        self._log(f"Tile rack letters from {source}:\n" + result)
        refresh = getattr(self, "_refresh_camera_preview", None)
        if callable(refresh):
            refresh()
        return detected_count

    def _has_calibrated_tile_rack_grid(self) -> bool:
        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        return bool(corners and len(corners) == 4)

    def _clear_tile_rack_side_from_main_ocr_grid(self) -> None:
        if not self._has_calibrated_tile_rack_grid():
            return

        entries = getattr(self, "_letter_entries", None)
        if not entries:
            return

        rack_corners = getattr(self, "_tile_rack_calibrated_corners", None)
        rack_x_values = [float(point[0]) for point in rack_corners]
        rack_y_values = [float(point[1]) for point in rack_corners]
        rack_center_x = sum(rack_x_values) / len(rack_x_values)
        rack_top = min(rack_y_values)
        rack_bottom = max(rack_y_values)

        grid = None
        try:
            grid = self._current_camera_ocr_grid()
        except Exception:
            grid = None

        grid_corners = getattr(grid, "corners", None)
        if grid_corners and len(grid_corners) == 4:
            grid_x_values = [float(point[0]) for point in grid_corners]
            grid_center_x = sum(grid_x_values) / len(grid_x_values)
            rack_column = 0 if rack_center_x < grid_center_x else BOARD_SIZE - 1
            top_left, top_right, bottom_right, bottom_left = [
                (float(point[0]), float(point[1])) for point in grid_corners
            ]
            top_edge = top_left if rack_column == 0 else top_right
            bottom_edge = bottom_left if rack_column == 0 else bottom_right
            for row in range(min(BOARD_SIZE, len(entries))):
                row_amount = (row + 0.5) / BOARD_SIZE
                row_y = top_edge[1] + row_amount * (bottom_edge[1] - top_edge[1])
                if rack_top - 8.0 <= row_y <= rack_bottom + 8.0:
                    self._clear_main_ocr_entry(row, rack_column)
            return

        frame = self._current_camera_ocr_frame()
        frame_width = frame.shape[1] if frame is not None else 0
        rack_column = 0 if frame_width <= 0 or rack_center_x < frame_width / 2.0 else BOARD_SIZE - 1
        rack_letters = {
            variable.get().strip().upper()
            for variable in getattr(self, "tile_rack_letter_vars", [])
            if variable.get().strip()
        }
        for row in range(min(BOARD_SIZE, len(entries))):
            try:
                text = entries[row][rack_column].get().strip().upper()
            except Exception:
                text = ""
            if not rack_letters or text in rack_letters:
                self._clear_main_ocr_entry(row, rack_column)

    def _clear_main_ocr_entry(self, row: int, col: int) -> None:
        try:
            entry = self._letter_entries[row][col]
        except Exception:
            return
        try:
            old_state = str(entry.cget("state"))
        except Exception:
            old_state = ""
        try:
            if old_state == "readonly":
                entry.configure(state="normal")
            entry.delete(0, tk.END)
        finally:
            if old_state == "readonly":
                entry.configure(state=old_state)

    def _separate_grid_from_tile_rack(self, grid) -> None:  # type: ignore[no-untyped-def]
        if not self._has_calibrated_tile_rack_grid():
            return

        grid_corners = getattr(grid, "corners", None)
        rack_corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if not grid_corners or len(grid_corners) != 4 or not rack_corners or len(rack_corners) != 4:
            return

        rack_x_values = [float(point[0]) for point in rack_corners]
        grid_x_values = [float(point[0]) for point in grid_corners]
        rack_center_x = sum(rack_x_values) / len(rack_x_values)
        grid_center_x = sum(grid_x_values) / len(grid_x_values)
        rack_width = max(rack_x_values) - min(rack_x_values)
        white_strip_gap = max(10.0, rack_width * 0.15)

        top_left, top_right, bottom_right, bottom_left = [
            (float(point[0]), float(point[1])) for point in grid_corners
        ]

        if rack_center_x < grid_center_x:
            boundary_x = max(rack_x_values) + white_strip_gap
            if boundary_x <= min(grid_x_values) or boundary_x >= max(grid_x_values):
                return
            new_top_left = self._point_on_segment_at_x(top_left, top_right, boundary_x)
            new_bottom_left = self._point_on_segment_at_x(bottom_left, bottom_right, boundary_x)
            new_corners = [new_top_left, top_right, bottom_right, new_bottom_left]
        else:
            boundary_x = min(rack_x_values) - white_strip_gap
            if boundary_x <= min(grid_x_values) or boundary_x >= max(grid_x_values):
                return
            new_top_right = self._point_on_segment_at_x(top_left, top_right, boundary_x)
            new_bottom_right = self._point_on_segment_at_x(bottom_left, bottom_right, boundary_x)
            new_corners = [top_left, new_top_right, new_bottom_right, bottom_left]

        self._set_grid_corners(grid, new_corners)

    def _separate_main_grid_from_tile_rack(self, scan) -> None:  # type: ignore[no-untyped-def]
        if not self._has_calibrated_tile_rack_grid():
            return

        grid = getattr(scan, "grid", None)
        grid_corners = getattr(grid, "corners", None)
        rack_corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if grid is None or not grid_corners or len(grid_corners) != 4 or not rack_corners or len(rack_corners) != 4:
            return

        rack_x_values = [float(point[0]) for point in rack_corners]
        grid_x_values = [float(point[0]) for point in grid_corners]
        rack_center_x = sum(rack_x_values) / len(rack_x_values)
        grid_center_x = sum(grid_x_values) / len(grid_x_values)
        rack_width = max(rack_x_values) - min(rack_x_values)
        white_strip_gap = max(10.0, rack_width * 0.15)

        top_left, top_right, bottom_right, bottom_left = [
            (float(point[0]), float(point[1])) for point in grid_corners
        ]

        if rack_center_x < grid_center_x:
            boundary_x = max(rack_x_values) + white_strip_gap
            if boundary_x <= min(grid_x_values) or boundary_x >= max(grid_x_values):
                return
            new_top_left = self._point_on_segment_at_x(top_left, top_right, boundary_x)
            new_bottom_left = self._point_on_segment_at_x(bottom_left, bottom_right, boundary_x)
            new_corners = [new_top_left, top_right, bottom_right, new_bottom_left]
        else:
            boundary_x = min(rack_x_values) - white_strip_gap
            if boundary_x <= min(grid_x_values) or boundary_x >= max(grid_x_values):
                return
            new_top_right = self._point_on_segment_at_x(top_left, top_right, boundary_x)
            new_bottom_right = self._point_on_segment_at_x(bottom_left, bottom_right, boundary_x)
            new_corners = [top_left, new_top_right, new_bottom_right, bottom_left]

        self._set_grid_corners(grid, new_corners)

    def _point_on_segment_at_x(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        x_value: float,
    ) -> list[float]:
        start_x, start_y = start
        end_x, end_y = end
        if abs(end_x - start_x) < 1e-6:
            return [float(x_value), float(start_y)]
        amount = (float(x_value) - start_x) / (end_x - start_x)
        amount = max(0.0, min(1.0, amount))
        return [float(x_value), start_y + amount * (end_y - start_y)]

    def _set_grid_corners(self, grid, corners) -> None:  # type: ignore[no-untyped-def]
        normalized = [[float(x), float(y)] for x, y in corners]
        try:
            grid.corners = normalized
            return
        except Exception:
            pass
        try:
            current = getattr(grid, "corners", None)
            if current is not None:
                current[:] = normalized
        except Exception:
            pass

    def _frame_without_tile_rack_area(self, frame):  # type: ignore[no-untyped-def]
        if not self._has_calibrated_tile_rack_grid():
            return frame
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return frame

        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        masked = frame.copy()
        polygon = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in corners], dtype=np.int32)
        if len(masked.shape) == 3:
            fill_value = tuple(int(value) for value in np.median(masked.reshape(-1, masked.shape[2]), axis=0))
        else:
            fill_value = int(np.median(masked))
        cv2.fillConvexPoly(masked, polygon, fill_value)
        return masked

    def _restore_tile_rack_area_from_frame(self, display, clean_frame):  # type: ignore[no-untyped-def]
        if not self._has_calibrated_tile_rack_grid():
            return display
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return display

        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if display.shape[:2] != clean_frame.shape[:2]:
            return display
        polygon = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in corners], dtype=np.int32)
        mask = np.zeros(display.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        restored = display.copy()
        restored[mask > 0] = clean_frame[mask > 0]
        return restored

    def _tile_rack_letters_from_camera(self, frame, scan) -> list[str]:  # type: ignore[no-untyped-def]
        letters = [""] * 7
        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if corners and len(corners) == 4:
            for item in self._tile_rack_camera_items(scan):
                center = self._tile_rack_camera_item_center(item)
                if center is None:
                    continue
                slot_index = self._tile_rack_slot_index_from_camera_corners(corners, center[0], center[1])
                if slot_index is None or letters[slot_index]:
                    continue
                letter = self._tile_rack_camera_item_letter(item)
                if letter:
                    letters[slot_index] = letter
            return letters

        rect = getattr(self, "_tile_rack_calibrated_rect", None) or getattr(self, "_last_tile_rack_camera_rect", None)
        if rect is None:
            rect = self._detect_tile_rack_brown_rect(frame)
        if rect is not None:
            self._last_tile_rack_camera_rect = rect
        if rect is not None:
            for item in self._tile_rack_camera_items(scan):
                center = self._tile_rack_camera_item_center(item)
                if center is None:
                    continue
                slot_index = self._tile_rack_slot_index_from_camera_point(rect, center[0], center[1])
                if slot_index is None or letters[slot_index]:
                    continue
                letter = self._tile_rack_camera_item_letter(item)
                if letter:
                    letters[slot_index] = letter
            if any(letters):
                return letters

        fallback = []
        for item in sorted(self._tile_rack_camera_items(scan), key=self._tile_rack_camera_sort_key):
            letter = self._tile_rack_camera_item_letter(item)
            if letter:
                fallback.append(letter)
            if len(fallback) >= 7:
                break
        for index, letter in enumerate(fallback[:7]):
            letters[index] = letter
        return letters

    def _tile_rack_camera_items(self, scan) -> list[object]:  # type: ignore[no-untyped-def]
        items: list[object] = []
        for name in ("letters", "tiles", "text_boxes"):
            values = getattr(scan, name, None)
            if values:
                items.extend(list(values))
        return items

    def _tile_rack_camera_item_letter(self, item) -> str:  # type: ignore[no-untyped-def]
        for name in ("text", "letter"):
            value = getattr(item, name, None)
            if not value:
                continue
            for character in str(value).upper():
                if character.isalpha():
                    return character
        return ""

    def _tile_rack_camera_sort_key(self, item) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        center = self._tile_rack_camera_item_center(item)
        if center is None:
            return (0.0, 0.0)
        return (center[1], center[0])

    def _tile_rack_camera_item_center(self, item) -> tuple[float, float] | None:  # type: ignore[no-untyped-def]
        center_x = getattr(item, "center_x", None)
        center_y = getattr(item, "center_y", None)
        if center_x is not None and center_y is not None:
            return float(center_x), float(center_y)

        left = getattr(item, "left", None)
        top = getattr(item, "top", None)
        if left is not None and top is not None:
            width = getattr(item, "width", 0) or 0
            height = getattr(item, "height", 0) or 0
            return float(left) + float(width) / 2.0, float(top) + float(height) / 2.0

        points = getattr(item, "points", None) or getattr(item, "corners", None)
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return sum(xs) / len(xs), sum(ys) / len(ys)
        return None

    def _detect_tile_rack_brown_rect(self, frame):  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return None

        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        lower_brown = np.array([5, 35, 25], dtype=np.uint8)
        upper_brown = np.array([35, 255, 215], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(frame.shape[0] * frame.shape[1])
        best_rect = None
        best_score = 0.0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(width * height)
            if area < frame_area * 0.001:
                continue
            if height < width * 1.4:
                continue
            score = area * (height / max(width, 1))
            if score > best_score:
                best_rect = (int(x), int(y), int(width), int(height))
                best_score = score
        return best_rect

    def _draw_tile_rack_grid_overlay(self, frame):  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
        except Exception:
            return frame

        if getattr(self, "_tile_rack_corner_selection_active", False):
            return self._draw_tile_rack_pending_corner_overlay(frame)

        corners = getattr(self, "_tile_rack_calibrated_corners", None)
        if corners and len(corners) == 4:
            return self._draw_tile_rack_corner_grid_overlay(frame, corners)

        rect = getattr(self, "_tile_rack_calibrated_rect", None) or getattr(self, "_last_tile_rack_camera_rect", None)
        if rect is None:
            rect = self._detect_tile_rack_brown_rect(frame)
        if rect is None:
            return frame

        self._last_tile_rack_camera_rect = rect
        x, y, width, height = rect
        green = (0, 255, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), green, 3, cv2.LINE_AA)
        for index in range(1, 7):
            line_y = int(round(y + height * index / 7.0))
            cv2.line(overlay, (x, line_y), (x + width, line_y), green, 2, cv2.LINE_AA)

        if hasattr(self, "tile_rack_letter_vars"):
            for index, variable in enumerate(self.tile_rack_letter_vars):
                try:
                    letter = variable.get()
                except Exception:
                    letter = ""
                if not letter:
                    continue
                center_y = int(round(y + height * (index + 0.5) / 7.0))
                cv2.putText(
                    overlay,
                    str(letter).upper()[:1],
                    (x + max(6, width // 2 - 10), center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    green,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _draw_tile_rack_pending_corner_overlay(self, frame):  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
        except Exception:
            return frame

        points = list(getattr(self, "_tile_rack_corner_selection_points", []))
        if not points:
            return frame

        overlay = frame.copy()
        green = (0, 255, 0)
        for index, point in enumerate(points):
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            cv2.circle(overlay, (x, y), 6, green, -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(index + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                green,
                2,
                cv2.LINE_AA,
            )
            if index > 0:
                previous = points[index - 1]
                cv2.line(
                    overlay,
                    (int(round(float(previous[0]))), int(round(float(previous[1])))),
                    (x, y),
                    green,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _draw_board_pending_corner_overlay(self, frame):  # type: ignore[no-untyped-def]
        if not getattr(self, "_board_corner_selection_active", False):
            return frame
        try:
            cv2 = _require_cv2()
        except Exception:
            return frame

        points = list(getattr(self, "_board_corner_selection_points", []))
        if not points:
            return frame

        overlay = frame.copy()
        color = (255, 0, 255)
        for index, point in enumerate(points):
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            cv2.circle(overlay, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(index + 1),
                (x + 9, y - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            if index > 0:
                previous = points[index - 1]
                cv2.line(
                    overlay,
                    (int(round(float(previous[0]))), int(round(float(previous[1])))),
                    (x, y),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _draw_tile_rack_corner_grid_overlay(self, frame, corners):  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return frame

        overlay = frame.copy()
        green = (0, 255, 0)
        source = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)], dtype=np.float32)
        destination = np.array(corners, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)

        def project(point):  # type: ignore[no-untyped-def]
            mapped = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), transform)[0][0]
            return int(round(float(mapped[0]))), int(round(float(mapped[1])))

        outline = [project(point) for point in [(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)]]
        for index, point in enumerate(outline):
            cv2.line(overlay, point, outline[(index + 1) % 4], green, 3, cv2.LINE_AA)

        for index in range(1, 7):
            left = project((0.0, float(index)))
            right = project((1.0, float(index)))
            cv2.line(overlay, left, right, green, 2, cv2.LINE_AA)

        if hasattr(self, "tile_rack_letter_vars"):
            for index, variable in enumerate(self.tile_rack_letter_vars):
                try:
                    letter = variable.get()
                except Exception:
                    letter = ""
                if not letter:
                    continue
                letter_point = project((0.45, index + 0.5))
                cv2.putText(
                    overlay,
                    str(letter).upper()[:1],
                    letter_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    green,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _tile_rack_slot_index_from_camera_corners(self, corners, x: float, y: float) -> int | None:  # type: ignore[no-untyped-def]
        try:
            cv2 = _require_cv2()
            import numpy as np
        except Exception:
            return None

        try:
            source = np.array(corners, dtype=np.float32)
            destination = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(source, destination)
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(point, transform)[0][0]
        except Exception:
            return None

        grid_x = float(mapped[0])
        grid_y = float(mapped[1])
        if grid_x < -0.05 or grid_x > 1.05 or grid_y < -0.05 or grid_y > 7.05:
            return None
        return max(0, min(6, int(grid_y)))

    def _tile_rack_slot_index_from_camera_point(self, rect, x: float, y: float) -> int | None:  # type: ignore[no-untyped-def]
        rect_x, rect_y, rect_width, rect_height = rect
        if x < rect_x or x > rect_x + rect_width or y < rect_y or y > rect_y + rect_height:
            return None
        slot_height = rect_height / 7.0
        if slot_height <= 0:
            return None
        return max(0, min(6, int((y - rect_y) / slot_height)))

    def _tile_rack_slot_position(self, slot_index: int) -> tuple[float, float]:
        if slot_index < 0 or slot_index >= 7:
            raise ValueError("Tile rack target must be TR1 through TR7.")
        rack_x = float(self.tile_rack_tr1_x_var.get())
        rack_y = float(self.tile_rack_tr1_y_var.get())
        tile_size = float(self.tile_rack_tile_size_var.get())
        return rack_x, rack_y + tile_size * slot_index

    def _ensure_tile_rack_move_state(self) -> None:
        if getattr(self, "_tile_rack_move_state_ready", False):
            if not hasattr(self, "tile_rack_letter_vars"):
                self.tile_rack_letter_vars = [tk.StringVar(value="") for _ in range(7)]
            if not hasattr(self, "tile_rack_word_suggestions_var"):
                self.tile_rack_word_suggestions_var = tk.StringVar(value="")
            return

        default_feed = "1500"
        feed_var = getattr(self, "feed_rate_var", None)
        if feed_var is not None:
            try:
                default_feed = str(feed_var.get())
            except Exception:
                default_feed = "1500"

        self.tile_rack_tr1_x_var = tk.StringVar(value="335")
        self.tile_rack_tr1_y_var = tk.StringVar(value="30")
        self.tile_rack_tile_size_var = tk.StringVar(value="10")
        self.tile_rack_feed_var = tk.StringVar(value=default_feed)
        self.tile_rack_status_var = tk.StringVar(value="Enter TR1 to TR7 in Move Plotter to go to rack slots.")
        self.tile_rack_letter_vars = [tk.StringVar(value="") for _ in range(7)]
        self.tile_rack_word_suggestions_var = tk.StringVar(value="")
        self._tile_rack_move_state_ready = True

    def _send_tile_rack_move_command(self, command: str) -> None:
        for method_name in (
            "_send_commands",
            "_send_serial_commands",
            "_send_controller_commands",
            "_send_gcode_commands",
            "_send_command_lines",
            "_send_gcode_lines",
            "_send_to_plotter",
            "_send_to_controller",
            "_send_serial",
            "_send_gcode",
            "_send_plotter_command",
            "_send_controller_command",
            "_send_raw_command",
            "_send_manual_command",
            "_send_command",
            "_send_gcode_command",
            "send_commands",
            "send_serial_commands",
            "send_controller_commands",
            "send_gcode_commands",
            "send_command_lines",
            "send_gcode_lines",
            "send_to_plotter",
            "send_to_controller",
            "send_serial",
            "send_gcode",
            "send_plotter_command",
            "send_controller_command",
            "send_raw_command",
            "send_manual_command",
            "send_command",
            "send_gcode_command",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            if "commands" in method_name or "lines" in method_name:
                payloads = (
                    ([command],),
                    ([command], "tile rack move"),
                    (command,),
                    (command, "tile rack move"),
                )
            else:
                payloads = (
                    (command,),
                    (command, "tile rack move"),
                )
            for payload in payloads:
                try:
                    method(*payload)
                    return
                except TypeError:
                    continue

        for method_name in dir(self):
            lowered = method_name.lower()
            if (
                method_name in {
                    "send_move",
                    "reset_to_start",
                    "_send_square_move",
                    "_send_move_target",
                    "_send_reset",
                    "_send_tile_rack_move_command",
                    "_send_tile_rack_move_with_gcode_sender",
                    "_send_tile_rack_target_move",
                    "_move_to_tile_rack_target_from_button",
                }
                or ("send" not in lowered and "command" not in lowered and "gcode" not in lowered)
            ):
                continue
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            if "commands" in lowered or "lines" in lowered:
                payloads = (([command],), (command,))
            else:
                payloads = ((command,),)
            for payload in payloads:
                try:
                    method(*payload)
                    return
                except TypeError:
                    continue
                except Exception:
                    continue

        raise RuntimeError("Could not find the plotter command sender for tile rack movement.")

    def _pick_and_drop_from_tile_rack_target(self) -> bool:
        rack_target, board_target = self._tile_rack_pick_drop_targets()
        if rack_target is None:
            return False
        if board_target is None:
            raise ValueError("Enter the board drop square as A1 to L12.")

        self._set_status(f"Picking from {rack_target} and dropping on {board_target}...")
        self._log(f"Rack pickup through normal pick/drop: {rack_target} -> {board_target}")
        self._send_pick_drop_aux_command(Z_UP_COMMAND)
        self._send_tile_rack_target_move(rack_target)
        self._send_pick_drop_aux_command(Z_DOWN_COMMAND)
        self._send_pick_drop_aux_command("R1")
        self._send_pick_drop_aux_command(Z_UP_COMMAND)
        self._send_square_move(board_target)
        self._send_pick_drop_aux_command(Z_DOWN_COMMAND)
        self._send_pick_drop_aux_command("R0")
        self._send_pick_drop_aux_command(Z_UP_COMMAND)
        self._set_status(f"Picked from {rack_target} and dropped on {board_target}.")
        return True

    def _tile_rack_pick_drop_targets(self) -> tuple[str | None, str | None]:
        values: list[tuple[str, str]] = []
        for name, source in vars(self).items():
            if not hasattr(source, "get"):
                continue
            try:
                value = source.get()
            except Exception:
                continue
            text = str(value).strip()
            if text:
                values.append((name.lower(), text))

        source_words = ("pick", "pickup", "from", "source")
        drop_words = ("drop", "to", "target", "destination", "place")

        rack_target = None
        for name, text in values:
            if any(word in name for word in source_words) and self._is_tile_rack_target(text.upper()):
                rack_target = text.upper()
                break
        if rack_target is None:
            for _name, text in values:
                if self._is_tile_rack_target(text.upper()):
                    rack_target = text.upper()
                    break
        if rack_target is None:
            return None, None

        board_target = None
        for name, text in values:
            if any(word in name for word in drop_words) and self._looks_like_board_square(text):
                board_target = text.upper()
                break
        if board_target is None:
            for name, text in values:
                if "rack" not in name and self._looks_like_board_square(text):
                    board_target = text.upper()
                    break
        return rack_target, board_target

    def _looks_like_board_square(self, value: object) -> bool:
        text = str(value).strip().upper()
        if len(text) < 2:
            return False
        column = text[0]
        row = text[1:]
        return column in "ABCDEFGHIJKL" and row.isdigit() and 1 <= int(row) <= 12

    def _send_pick_drop_aux_command(self, command: str) -> None:
        writer = getattr(self, "_write_auxiliary_serial_line", None)
        if callable(writer):
            writer(command)
            self._log(f"Sent auxiliary command: {command}")
            return

        sender = getattr(self, "_send_auxiliary_command", None)
        if callable(sender):
            sender(command)
            return

        raise RuntimeError("Could not find the Z movement or relay command sender.")

    def _send_startup_z_up(self) -> None:
        try:
            self._send_pick_drop_aux_command(Z_UP_COMMAND)
            self._log("Startup Z up command sent.")
        except Exception as exc:
            log = getattr(self, "_log", None)
            if callable(log):
                log(f"Startup Z up command skipped: {exc}")

    def _pick_current_position_and_drop_to_board(self) -> None:
        board_target = self._pick_drop_board_target()
        if board_target is None:
            raise ValueError("Enter the board drop square, for example A1.")

        self._set_status(f"Picking current tile and dropping on {board_target}...")
        self._log(f"Pick/drop current position -> {board_target}")
        steps = [
            ("Z down", lambda: self._send_pick_drop_ordered_command(Z_DOWN_COMMAND)),
            ("Magnet on", lambda: self._send_pick_drop_ordered_command("R1")),
            ("Z up", lambda: self._send_pick_drop_ordered_command(Z_UP_COMMAND)),
            (f"Move to {board_target} to drop", lambda: self._send_pick_drop_ordered_board_move(board_target)),
            ("Z down", lambda: self._send_pick_drop_ordered_command(Z_DOWN_COMMAND)),
            ("Magnet off", lambda: self._send_pick_drop_ordered_command("R0")),
            ("Z up", lambda: self._send_pick_drop_ordered_command(Z_UP_COMMAND)),
        ]
        self._run_pick_drop_steps(steps, board_target)

    def pick_and_drop(self) -> None:
        try:
            if getattr(self, "_pick_drop_running", False):
                raise RuntimeError("Pick/drop is already running. Wait until it finishes before starting again.")
            pickup_target = self.pick_square_var.get().strip().upper()
            drop_target = self.drop_square_var.get().strip().upper()
            self._validate_pick_drop_targets(pickup_target, drop_target)
            self._apply_pick_drop_z_height()
            self._pick_drop_running = True
            self._pick_and_drop_targets(pickup_target, drop_target)
        except Exception as exc:
            self._pick_drop_running = False
            self._show_error(exc)

    def _validate_pick_drop_targets(self, pickup_target: str, drop_target: str) -> None:
        if not (self._is_tile_rack_target(pickup_target) or self._looks_like_board_square(pickup_target)):
            raise ValueError("Enter pickup as a board square like A1 or rack slot TR1 to TR7.")
        if not self._looks_like_board_square(drop_target):
            raise ValueError("Enter drop as a board square like H8.")

    def _apply_pick_drop_z_height(self) -> None:
        angle = int(float(self.z_height_angle.get()))
        angle = max(0, min(180, angle))
        self.z_height_angle.set(angle)
        self._send_pick_drop_aux_command(f"ZH{angle}")

    def _pick_and_drop_targets(self, pickup_target: str, drop_target: str) -> None:
        self._set_status(f"Picking from {pickup_target} and dropping on {drop_target}...")
        self._log(f"Pick/drop: {pickup_target} -> {drop_target}")
        steps = [
            ("Z up", lambda: self._send_pick_drop_ordered_command(Z_UP_COMMAND)),
            (f"Move to pickup {pickup_target}", lambda: self._send_pick_drop_ordered_target_move(pickup_target)),
            ("Z down", lambda: self._send_pick_drop_ordered_command(Z_DOWN_COMMAND)),
            ("Magnet on", lambda: self._send_pick_drop_ordered_command("M1")),
            ("Z up", lambda: self._send_pick_drop_ordered_command(Z_UP_COMMAND)),
            (f"Move to drop {drop_target}", lambda: self._send_pick_drop_ordered_target_move(drop_target)),
            ("Z down", lambda: self._send_pick_drop_ordered_command(Z_DOWN_COMMAND)),
            ("Magnet off", lambda: self._send_pick_drop_ordered_command("M0")),
            ("Z up", lambda: self._send_pick_drop_ordered_command(Z_UP_COMMAND)),
        ]
        self._run_pick_drop_steps(steps, drop_target)

    def _send_pick_drop_ordered_command(self, command: str) -> None:
        self._send_pick_drop_aux_command(command)

    def _send_pick_drop_ordered_target_move(self, target: str) -> None:
        target = target.strip().upper()
        if self._is_tile_rack_target(target):
            self._send_tile_rack_target_move(target)
            return
        self._send_pick_drop_ordered_board_move(target)

    def _send_pick_drop_ordered_board_move(self, square_label: str) -> None:
        self._send_square_move(square_label)

    def _pick_drop_feed_rate(self) -> float:
        for name in ("feed_rate_var", "feed_var", "plotter_feed_rate_var", "tile_rack_feed_var"):
            source = getattr(self, name, None)
            if source is None:
                continue
            try:
                return float(source.get() if hasattr(source, "get") else source)
            except Exception:
                continue
        return 1500.0

    def _run_pick_drop_steps(self, steps, board_target: str, index: int = 0) -> None:  # type: ignore[no-untyped-def]
        if index >= len(steps):
            self._pick_drop_running = False
            self._set_status(f"Picked current tile and dropped on {board_target}.")
            return

        label, action = steps[index]
        try:
            self._set_status(f"Pick/drop step {index + 1}/{len(steps)}: {label}")
            self._log(f"Pick/drop step {index + 1}: {label}")
            action()
        except Exception as exc:
            self._pick_drop_running = False
            self._show_error(exc)
            return

        delay_ms = self._pick_drop_step_delay_ms(label)
        self.root.after(delay_ms, lambda: self._run_pick_drop_steps(steps, board_target, index + 1))

    def _pick_drop_step_delay_ms(self, label: str) -> int:
        lowered = label.lower()
        if lowered.startswith("z "):
            return PICK_DROP_Z_SETTLE_DELAY_MS
        if lowered.startswith("magnet "):
            return PICK_DROP_MAGNET_DELAY_MS
        if lowered.startswith("move "):
            return PICK_DROP_MOVE_DELAY_MS
        return PICK_DROP_DEFAULT_DELAY_MS

    def _pick_drop_board_target(self) -> str | None:
        values: list[tuple[str, str]] = []
        for name, source in vars(self).items():
            if not hasattr(source, "get"):
                continue
            try:
                value = source.get()
            except Exception:
                continue
            text = str(value).strip()
            if text:
                values.append((name.lower(), text))

        drop_words = ("drop", "to", "target", "destination", "place")
        for name, text in values:
            if any(word in name for word in drop_words) and self._looks_like_board_square(text):
                return text.upper()

        square_var = getattr(self, "square_var", None)
        if square_var is not None:
            try:
                square_text = str(square_var.get()).strip()
            except Exception:
                square_text = ""
            if self._looks_like_board_square(square_text):
                return square_text.upper()

        for _name, text in values:
            if self._looks_like_board_square(text):
                return text.upper()
        return None

    def _handle_ocr_board_cell_click(self, row: int, col: int) -> None:
        square = f"{chr(ord('A') + col)}{row + 1}"
        self._move_plotter_to_clicked_square(square, source="OCR grid")

    def _handle_camera_preview_click(self, event) -> None:  # type: ignore[no-untyped-def]
        grid = self._current_camera_ocr_grid()
        if grid is None:
            self._set_status("No auto-aligned OCR grid is visible yet. Capture letters or find words first.")
            return

        point = self._preview_click_to_frame_point(int(event.x), int(event.y))
        if point is None:
            self._set_status("Click inside the camera image to choose a board square.")
            return

        square = board_square_from_image_point(grid.corners, point[0], point[1], board_size=grid.board_size)
        if square is None:
            self._set_status("Click inside the detected board grid to move the plotter.")
            return

        self._move_plotter_to_clicked_square(square, source="camera grid")

    def _move_plotter_to_clicked_square(self, square: str, source: str) -> None:
        try:
            self.square_var.set(square)
            self._send_square_move(square)
            self._log(f"Clicked {source} square {square}.")
        except Exception as exc:
            self._show_error(exc)

    def _send_tile_rack_move_with_gcode_sender(self, command: str) -> bool:
        return False

        try:
            config = self._tile_rack_move_serial_config()
        except Exception:
            return False

        if not self._tile_rack_move_config_has_port(config):
            return False

        try:
            sender = GCodeSender(config)
            if hasattr(sender, "__enter__") and hasattr(sender, "__exit__"):
                with sender as active_sender:
                    return self._send_tile_rack_move_to_sender(active_sender, command)
            try:
                return self._send_tile_rack_move_to_sender(sender, command)
            finally:
                close = getattr(sender, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def _send_tile_rack_move_to_sender(self, sender, command: str) -> bool:  # type: ignore[no-untyped-def]
        for method_name in ("send_commands", "send_command", "send", "write"):
            method = getattr(sender, method_name, None)
            if not callable(method):
                continue
            payloads = ([command], command) if method_name == "send_commands" else (command, [command])
            for payload in payloads:
                try:
                    method(payload)
                    return True
                except TypeError:
                    continue
        return False

    def _tile_rack_move_serial_config(self) -> SerialConfig:
        import dataclasses
        import inspect

        for method_name in (
            "_serial_config_from_form",
            "_serial_config_from_inputs",
            "_get_serial_config",
            "serial_config_from_form",
            "get_serial_config",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                config = method()
            except Exception:
                continue
            if isinstance(config, SerialConfig):
                return config

        if dataclasses.is_dataclass(SerialConfig):
            field_names = [field.name for field in dataclasses.fields(SerialConfig)]
        else:
            signature = inspect.signature(SerialConfig)
            field_names = [name for name in signature.parameters if name != "self"]

        values = {}
        for name in field_names:
            value = self._tile_rack_move_serial_value(name)
            if value is not None:
                values[name] = value
        return SerialConfig(**values)

    def _tile_rack_move_serial_value(self, name: str):  # type: ignore[no-untyped-def]
        key = name.lower()
        if key in {"port", "serial_port", "com_port"}:
            return self._tile_rack_selected_port()
        if key in {"baud", "baudrate", "baud_rate"}:
            return int(self._tile_rack_read_gui_value("baud_var", "baudrate_var", "baud_rate_var", default="115200"))
        if "timeout" in key:
            return float(self._tile_rack_read_gui_value("timeout_var", "serial_timeout_var", default="2.0"))
        if key in {"dry_run", "dryrun"}:
            value = self._tile_rack_read_gui_value("dry_run_var", "dryrun_var", default=False)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if "delay" in key or "settle" in key:
            return float(self._tile_rack_read_gui_value("settle_seconds_var", "startup_delay_var", default="2.0"))
        return None

    def _tile_rack_move_config_has_port(self, config: SerialConfig) -> bool:
        for name in ("port", "serial_port", "com_port"):
            if hasattr(config, name):
                value = getattr(config, name)
                if value is not None and str(value).strip():
                    return True
        return False

    def _tile_rack_selected_port(self) -> str | None:
        port = self._tile_rack_read_gui_value(
            "port_var",
            "serial_port_var",
            "com_port_var",
            "plotter_port_var",
            "selected_port_var",
            "port_combo",
            "serial_port_combo",
            "com_port_combo",
        )
        if port is None:
            return None
        text = str(port).strip()
        if not text:
            return None
        if text.upper().startswith("COM"):
            return text.split()[0].rstrip(":,;")
        if text.startswith("/dev/") or text.lower().startswith("usb"):
            return text
        for name, source in vars(self).items():
            lowered = name.lower()
            if "port" not in lowered and "com" not in lowered and "serial" not in lowered:
                continue
            if not hasattr(source, "get"):
                continue
            try:
                value = source.get()
            except Exception:
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if text.upper().startswith("COM"):
                return text.split()[0].rstrip(":,;")
            if text.startswith("/dev/") or text.lower().startswith("usb"):
                return text
        return text

    def _tile_rack_read_gui_value(self, *names: str, default=None):  # type: ignore[no-untyped-def]
        for name in names:
            source = getattr(self, name, None)
            if source is None:
                continue
            try:
                value = source.get() if hasattr(source, "get") else source
            except Exception:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return default

    def open_tile_rack_position_window(self, force_open: bool = False) -> None:
        self._ensure_tile_rack_move_state()
        existing_window = getattr(self, "_tile_rack_position_window", None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._tile_rack_position_window = window
        window.title("Tile Rack Position")
        window.resizable(False, False)
        window.columnconfigure(1, weight=1)

        ttk.Label(window, text="TR1 X").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Entry(window, textvariable=self.tile_rack_tr1_x_var, width=12).grid(
            row=0, column=1, sticky="ew", padx=10, pady=(10, 4)
        )

        ttk.Label(window, text="TR1 Y").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_tr1_y_var, width=12).grid(
            row=1, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(window, text="Tile size").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_tile_size_var, width=12).grid(
            row=2, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(window, text="Feed rate").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_feed_var, width=12).grid(
            row=3, column=1, sticky="ew", padx=10, pady=4
        )

        buttons = ttk.LabelFrame(window, text="Rack Slots")
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4))
        for index in range(7):
            target = f"TR{index + 1}"
            ttk.Button(
                buttons,
                text=target,
                command=lambda target=target: self._move_to_tile_rack_target_from_button(target),
                width=5,
            ).grid(row=index // 4, column=index % 4, sticky="ew", padx=2, pady=2)

        ttk.Button(window, text="Capture Rack Letters", command=self.capture_tile_rack_letters).grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4)
        )

        rack_letters_frame = ttk.LabelFrame(window, text="Tile Rack Letters")
        rack_letters_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        self._build_tile_rack_letter_grid(rack_letters_frame)

        ttk.Button(window, text="Suggest Rack Words", command=self.suggest_tile_rack_words).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4)
        )

        suggestions_frame = ttk.LabelFrame(window, text="Suggested Words")
        suggestions_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        ttk.Label(
            suggestions_frame,
            textvariable=self.tile_rack_word_suggestions_var,
            width=36,
            anchor="nw",
            justify="left",
            wraplength=280,
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        ttk.Label(window, textvariable=self.tile_rack_status_var, wraplength=280).grid(
            row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 10)
        )
        return

        self._ensure_tile_rack_position_state()
        if not force_open and not getattr(self, "_tile_rack_manual_corners", None):
            self.select_tile_rack_corners(open_position_after=True)
            return

        existing_window = getattr(self, "_tile_rack_position_window", None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._tile_rack_position_window = window
        window.title("Tile Rack Position")
        window.resizable(False, False)
        window.columnconfigure(1, weight=1)

        ttk.Label(window, text="Rack X").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Entry(window, textvariable=self.tile_rack_position_x_var, width=12).grid(
            row=0, column=1, sticky="ew", padx=10, pady=(10, 4)
        )

        ttk.Label(window, text="Rack Y").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_position_y_var, width=12).grid(
            row=1, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(window, text="Slot gap").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_position_gap_var, width=12).grid(
            row=2, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(window, text="Feed rate").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_position_feed_var, width=12).grid(
            row=3, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Button(window, text="Move To Tile Rack", command=self.move_to_tile_rack_position).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4)
        )
        tr_frame = ttk.LabelFrame(window, text="Rack Positions")
        tr_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        self._build_tile_rack_move_buttons(tr_frame)

        ttk.Button(window, text="Detect Rack Letters", command=self.detect_tile_rack_position_letters).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=4
        )
        ttk.Button(window, text="Select Rack Corners", command=self.select_tile_rack_corners).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=10, pady=4
        )
        rack_ocr_frame = ttk.LabelFrame(window, text="Tile Rack OCR")
        rack_ocr_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        self._build_tile_rack_ocr_grid(rack_ocr_frame)

        ttk.Label(window, textvariable=self.tile_rack_position_status_var, wraplength=300).grid(
            row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 10)
        )

    def _ensure_tile_rack_position_state(self) -> None:
        if getattr(self, "_tile_rack_position_state_ready", False):
            return

        default_feed = "1500"
        feed_var = getattr(self, "feed_rate_var", None)
        if feed_var is not None:
            try:
                default_feed = str(feed_var.get())
            except Exception:
                default_feed = "1500"

        self.tile_rack_position_x_var = tk.StringVar(value="335")
        self.tile_rack_position_y_var = tk.StringVar(value="30")
        self.tile_rack_position_gap_var = tk.StringVar(value="10")
        self.tile_rack_position_feed_var = tk.StringVar(value=default_feed)
        self.tile_rack_position_status_var = tk.StringVar(value="Set the tile rack X/Y position, then move to it.")
        self.tile_rack_ocr_letter_vars = [tk.StringVar(value="") for _ in range(7)]
        self._tile_rack_manual_corners = getattr(self, "_tile_rack_manual_corners", None)
        self._tile_rack_position_state_ready = True

    def _build_tile_rack_move_buttons(self, parent) -> None:  # type: ignore[no-untyped-def]
        for index in range(7):
            target = f"TR{index + 1}"
            ttk.Button(
                parent,
                text=target,
                command=lambda target=target: self._move_to_tile_rack_label(target),
                width=5,
            ).grid(row=index // 4, column=index % 4, sticky="ew", padx=2, pady=2)

    def _move_to_tile_rack_label(self, target: str) -> None:
        try:
            self._send_move_target(target)
        except Exception as exc:
            self._show_error(exc)

    def _build_tile_rack_ocr_grid(self, parent) -> None:  # type: ignore[no-untyped-def]
        self._ensure_tile_rack_position_state()
        for index, variable in enumerate(self.tile_rack_ocr_letter_vars):
            ttk.Label(parent, text=f"TR{index + 1}").grid(row=index, column=0, sticky="e", padx=(6, 4), pady=1)
            entry = ttk.Entry(parent, textvariable=variable, width=4, justify="center")
            entry.grid(row=index, column=1, sticky="w", padx=(0, 6), pady=1)
            entry.configure(state="readonly")

    def move_to_tile_rack_position(self) -> None:
        self._ensure_tile_rack_position_state()
        try:
            x = float(self.tile_rack_position_x_var.get())
            y = float(self.tile_rack_position_y_var.get())
            feed = float(self.tile_rack_position_feed_var.get())
            command = self._send_absolute_plotter_move(x, y, feed)
            message = f"Moved to tile rack position X{x:g} Y{y:g}."
            self.tile_rack_position_status_var.set(message)
            self._set_status(message)
            self._log(command)
        except Exception as exc:
            self._show_error(exc)

    def detect_tile_rack_position_letters(self) -> None:
        import time

        self._ensure_tile_rack_position_state()
        try:
            rack_x = float(self.tile_rack_position_x_var.get())
            rack_y = float(self.tile_rack_position_y_var.get())
            slot_gap = float(self.tile_rack_position_gap_var.get())
            lines = []
            detected_count = 0
            for variable in self.tile_rack_ocr_letter_vars:
                variable.set("")
            for index in range(7):
                slot_x = rack_x
                slot_y = rack_y + slot_gap * index
                self._set_status(f"Scanning tile rack slot {index + 1} at X{slot_x:g} Y{slot_y:g}...")
                self._send_absolute_plotter_move(slot_x, slot_y, float(self.tile_rack_position_feed_var.get()))
                time.sleep(0.8)
                frame, _quality = self._capture_best_photo_for_ocr(f"tile rack slot {index + 1}")
                scan = scan_camera_letters(frame)
                letter = self._tile_rack_position_single_letter_from_scan(scan)
                self.tile_rack_ocr_letter_vars[index].set("" if letter == "-" else letter)
                if letter != "-":
                    detected_count += 1
                lines.append(f"Slot {index + 1}: {letter}  X{slot_x:g} Y{slot_y:g}")

            result = "\n".join(lines)
            self.tile_rack_position_status_var.set(result)
            self._set_status(f"Detected {detected_count} tile rack letter(s) separately.")
            self._log("Tile rack letters:\n" + result)
            self._refresh_camera_preview()
        except Exception as exc:
            self._show_error(exc)

    def _tile_rack_position_single_letter_from_scan(self, scan: CameraLetterScanResult) -> str:
        letters = self._tile_rack_position_letters_from_scan(scan)
        return letters[0] if letters else "-"

    def _tile_rack_position_letters_from_scan(self, scan: CameraLetterScanResult) -> list[str]:
        letters: list[str] = []
        for captured in sorted(self._tile_rack_scan_items(scan), key=self._tile_rack_position_letter_sort_key):
            text = self._tile_rack_item_text(captured)
            for character in str(text).upper():
                if character.isalpha():
                    letters.append(character)
                    if len(letters) >= 7:
                        return letters
        return letters

    def _tile_rack_scan_items(self, scan) -> list[object]:  # type: ignore[no-untyped-def]
        items: list[object] = []
        for name in ("letters", "tiles", "text_boxes"):
            values = getattr(scan, name, None)
            if values:
                items.extend(list(values))
        return items

    def _tile_rack_item_text(self, item) -> str:  # type: ignore[no-untyped-def]
        for name in ("text", "letter"):
            value = getattr(item, name, None)
            if value:
                return str(value)
        return ""

    def _tile_rack_position_letter_sort_key(self, captured) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        center = self._tile_rack_position_letter_center(captured)
        if center is None:
            return (0.0, 0.0)
        return (center[1], center[0])

    def _tile_rack_position_letter_center(self, captured) -> tuple[float, float] | None:  # type: ignore[no-untyped-def]
        center_x = getattr(captured, "center_x", None)
        center_y = getattr(captured, "center_y", None)
        if center_x is not None and center_y is not None:
            return (float(center_x), float(center_y))

        left = getattr(captured, "left", None)
        top = getattr(captured, "top", None)
        if left is not None and top is not None:
            width = getattr(captured, "width", 0) or 0
            height = getattr(captured, "height", 0) or 0
            return (float(left) + float(width) / 2.0, float(top) + float(height) / 2.0)

        points = getattr(captured, "points", None) or getattr(captured, "corners", None)
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        return None

    def select_tile_rack_corners(self, open_position_after: bool = False) -> None:
        self._ensure_tile_rack_position_state()
        try:
            from PIL import Image, ImageTk

            frame = self._current_camera_ocr_frame()
            if frame is None:
                frame, _quality = self._capture_best_photo_for_ocr("tile rack corner selection")
        except Exception as exc:
            self._show_error(exc)
            return

        try:
            cv2 = _require_cv2()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            image = Image.fromarray(rgb)
        except Exception as exc:
            self._show_error(exc)
            return

        max_width = 760
        max_height = 520
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        display_image = image.resize(display_size)

        window = tk.Toplevel(self.root)
        window.title("Select Tile Rack Corners")
        ttk.Label(window, text="Click the four tile rack corners: top-left, top-right, bottom-right, bottom-left.").grid(
            row=0, column=0, sticky="ew", padx=10, pady=(10, 4)
        )
        canvas = tk.Canvas(window, width=display_size[0], height=display_size[1], cursor="crosshair")
        canvas.grid(row=1, column=0, padx=10, pady=4)
        photo = ImageTk.PhotoImage(display_image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

        clicked: list[tuple[float, float]] = []

        def redraw_points() -> None:
            canvas.delete("corner")
            for index, point in enumerate(clicked):
                x = point[0] * scale
                y = point[1] * scale
                canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="lime", width=2, tags="corner")
                canvas.create_text(x + 12, y - 10, text=str(index + 1), fill="lime", tags="corner")
            if len(clicked) > 1:
                scaled = [(point[0] * scale, point[1] * scale) for point in clicked]
                for start, end in zip(scaled, scaled[1:]):
                    canvas.create_line(start[0], start[1], end[0], end[1], fill="lime", width=2, tags="corner")
                if len(clicked) == 4:
                    canvas.create_line(scaled[-1][0], scaled[-1][1], scaled[0][0], scaled[0][1], fill="lime", width=2, tags="corner")

        def handle_click(event) -> None:  # type: ignore[no-untyped-def]
            if len(clicked) >= 4:
                return
            clicked.append((float(event.x) / scale, float(event.y) / scale))
            redraw_points()
            if len(clicked) == 4:
                self._tile_rack_manual_corners = clicked.copy()
                self._last_tile_rack_camera_rect = None
                self.tile_rack_position_status_var.set("Tile rack corners selected. The green rack grid will use these corners.")
                self._set_status("Tile rack corners selected.")
                self._refresh_camera_preview()
                if open_position_after:
                    window.after(400, lambda: (window.destroy(), self.open_tile_rack_position_window(force_open=True)))
                else:
                    window.after(400, window.destroy)

        def clear_points() -> None:
            clicked.clear()
            self._tile_rack_manual_corners = None
            self._last_tile_rack_camera_rect = None
            canvas.delete("corner")
            self.tile_rack_position_status_var.set("Tile rack corner selection cleared.")
            self._refresh_camera_preview()

        canvas.bind("<Button-1>", handle_click)
        ttk.Button(window, text="Clear Rack Corners", command=clear_points).grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))

    def _update_tile_rack_ocr_from_current_frame(self, scan: CameraLetterScanResult) -> None:
        if not getattr(self, "_tile_rack_position_state_ready", False):
            return
        frame = self._current_camera_ocr_frame()
        if frame is None:
            return
        letters = self._tile_rack_letters_from_brown_grid(frame, scan)
        self._set_tile_rack_ocr_letters(letters)

    def _set_tile_rack_ocr_letters(self, letters: list[str]) -> None:
        self._ensure_tile_rack_position_state()
        for index in range(7):
            letter = letters[index] if index < len(letters) and letters[index] else ""
            self.tile_rack_ocr_letter_vars[index].set(letter)

    def _clear_tile_rack_ocr_letters(self) -> None:
        self._ensure_tile_rack_position_state()
        for variable in self.tile_rack_ocr_letter_vars:
            variable.set("")

    def detect_tile_rack_position_letters(self) -> None:
        self._ensure_tile_rack_position_state()
        try:
            if self._captured_photo_frame is None:
                frame, quality = self._capture_best_photo_for_ocr("tile rack OCR grid")
                source = f"best camera frame, sharpness {quality.sharpness:.0f}"
            else:
                frame = self._captured_photo_frame.copy()
                source = "captured picture"

            scan = scan_camera_letters(frame)
            letters = self._tile_rack_letters_from_brown_grid(frame, scan)
            self._clear_tile_rack_ocr_letters()
            self._set_tile_rack_ocr_letters(letters)
            rack_x = float(self.tile_rack_position_x_var.get())
            rack_y = float(self.tile_rack_position_y_var.get())
            slot_gap = float(self.tile_rack_position_gap_var.get())
            lines = []
            detected_count = 0
            for index in range(7):
                letter = letters[index] if index < len(letters) and letters[index] else "-"
                if letter != "-":
                    detected_count += 1
                slot_x = rack_x
                slot_y = rack_y + slot_gap * index
                lines.append(f"TR{index + 1}: {letter}  X{slot_x:g} Y{slot_y:g}")

            result = "\n".join(lines)
            self.tile_rack_position_status_var.set(result)
            self._set_status(f"Detected {detected_count} tile rack letter(s) from the {source}.")
            self._log("Tile rack OCR grid:\n" + result)
            self._refresh_camera_preview()
        except Exception as exc:
            self._show_error(exc)

    def _tile_rack_letters_from_brown_grid(self, frame, scan: CameraLetterScanResult) -> list[str]:  # type: ignore[no-untyped-def]
        manual_corners = getattr(self, "_tile_rack_manual_corners", None)
        if manual_corners and len(manual_corners) == 4:
            return self._tile_rack_letters_from_corner_grid(manual_corners, scan)

        rect = self._detect_tile_rack_brown_rect(frame)
        letters = [""] * 7
        self._last_tile_rack_camera_rect = rect
        if rect is None:
            fallback_letters = self._tile_rack_position_letters_from_scan(scan)
            for index, letter in enumerate(fallback_letters[:7]):
                letters[index] = letter
            return letters

        for captured in self._tile_rack_scan_items(scan):
            center = self._tile_rack_position_letter_center(captured)
            if center is None:
                continue
            slot_index = self._tile_rack_slot_index_from_point(rect, center[0], center[1])
            if slot_index is None or letters[slot_index]:
                continue
            text = self._tile_rack_item_text(captured)
            for character in str(text).upper():
                if character.isalpha():
                    letters[slot_index] = character
                    break
        return letters

    def _tile_rack_letters_from_corner_grid(self, corners, scan: CameraLetterScanResult) -> list[str]:  # type: ignore[no-untyped-def]
        letters = [""] * 7
        for captured in self._tile_rack_scan_items(scan):
            center = self._tile_rack_position_letter_center(captured)
            if center is None:
                continue
            slot_index = self._tile_rack_slot_index_from_corner_grid(corners, center[0], center[1])
            if slot_index is None or letters[slot_index]:
                continue
            text = self._tile_rack_item_text(captured)
            for character in str(text).upper():
                if character.isalpha():
                    letters[slot_index] = character
                    break
        return letters

    def _tile_rack_slot_index_from_corner_grid(self, corners, x: float, y: float) -> int | None:  # type: ignore[no-untyped-def]
        cv2 = _require_cv2()
        import numpy as np

        source = np.array(corners, dtype=np.float32)
        destination = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, transform)[0][0]
        grid_x = float(mapped[0])
        grid_y = float(mapped[1])
        if grid_x < -0.05 or grid_x > 1.05 or grid_y < -0.05 or grid_y > 7.05:
            return None
        return max(0, min(6, int(grid_y)))

    def _detect_tile_rack_brown_rect(self, frame):  # type: ignore[no-untyped-def]
        cv2 = _require_cv2()
        import numpy as np

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if len(frame.shape) != 3:
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

        lower_brown = np.array([5, 35, 25], dtype=np.uint8)
        upper_brown = np.array([35, 255, 210], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(frame.shape[0] * frame.shape[1])
        best_rect = None
        best_score = 0.0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(width * height)
            if area < frame_area * 0.002:
                continue
            if height < width * 1.4:
                continue
            score = area * (height / max(width, 1))
            if score > best_score:
                best_rect = (int(x), int(y), int(width), int(height))
                best_score = score
        return best_rect

    def _tile_rack_slot_index_from_point(self, rect, x: float, y: float) -> int | None:  # type: ignore[no-untyped-def]
        rect_x, rect_y, rect_width, rect_height = rect
        if x < rect_x or x > rect_x + rect_width or y < rect_y or y > rect_y + rect_height:
            return None
        slot_height = rect_height / 7.0
        if slot_height <= 0:
            return None
        return max(0, min(6, int((y - rect_y) / slot_height)))

    def _draw_tile_rack_green_grid_overlay(self, frame):  # type: ignore[no-untyped-def]
        cv2 = _require_cv2()
        manual_corners = getattr(self, "_tile_rack_manual_corners", None)
        if manual_corners and len(manual_corners) == 4:
            return self._draw_tile_rack_corner_grid_overlay(frame, manual_corners)

        rect = getattr(self, "_last_tile_rack_camera_rect", None) or self._detect_tile_rack_brown_rect(frame)
        if rect is None:
            return frame

        x, y, width, height = rect
        green = (0, 255, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), green, 3, cv2.LINE_AA)
        for index in range(1, 7):
            line_y = int(round(y + height * index / 7.0))
            cv2.line(overlay, (x, line_y), (x + width, line_y), green, 2, cv2.LINE_AA)

        for index in range(7):
            center_y = int(round(y + height * (index + 0.5) / 7.0))
            letter = ""
            if getattr(self, "_tile_rack_position_state_ready", False):
                letter = self.tile_rack_ocr_letter_vars[index].get()
            if letter:
                cv2.putText(
                    overlay,
                    letter,
                    (x + max(8, width // 2 - 10), center_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    green,
                    2,
                    cv2.LINE_AA,
                )
        return overlay

    def _draw_tile_rack_corner_grid_overlay(self, frame, corners):  # type: ignore[no-untyped-def]
        cv2 = _require_cv2()
        import numpy as np

        overlay = frame.copy()
        green = (0, 255, 0)
        source = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)], dtype=np.float32)
        destination = np.array(corners, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)

        def project(point):  # type: ignore[no-untyped-def]
            mapped = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), transform)[0][0]
            return int(round(float(mapped[0]))), int(round(float(mapped[1])))

        outline = [project(point) for point in [(0.0, 0.0), (1.0, 0.0), (1.0, 7.0), (0.0, 7.0)]]
        for index, point in enumerate(outline):
            cv2.line(overlay, point, outline[(index + 1) % 4], green, 3, cv2.LINE_AA)

        for index in range(1, 7):
            left = project((0.0, float(index)))
            right = project((1.0, float(index)))
            cv2.line(overlay, left, right, green, 2, cv2.LINE_AA)

        for index in range(7):
            letter_point = project((0.48, index + 0.5))
            if getattr(self, "_tile_rack_position_state_ready", False):
                letter = self.tile_rack_ocr_letter_vars[index].get()
                if letter:
                    cv2.putText(overlay, letter, letter_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2, cv2.LINE_AA)
        return overlay

    def _send_tile_rack_position_command(self, command: str) -> None:
        self._send_plotter_raw_command(command)
        return

        sender = getattr(self, "_send_tile_rack_command", None)
        if callable(sender):
            sender(command, prefer_arduino=False)
            return

        for method_name in (
            "_send_raw_command",
            "_send_manual_command",
            "_send_command",
            "_send_gcode_command",
            "send_raw_command",
            "send_manual_command",
            "send_command",
            "send_gcode_command",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                method(command)
                return
            except TypeError:
                continue

        raise RuntimeError("Could not find the plotter command sender for the tile rack position.")

    def open_tile_rack_window(self) -> None:
        raise_user_error("The tile rack feature has been removed.")
        return

        self._ensure_tile_rack_state()
        existing_window = getattr(self, "_tile_rack_window", None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._tile_rack_window = window
        window.title("Tile Rack")
        window.resizable(False, False)
        window.columnconfigure(1, weight=1)

        ttk.Label(window, text="Rack letters").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        rack_entry = ttk.Entry(window, textvariable=self.tile_rack_letters_var, width=18)
        rack_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 4))
        ttk.Button(window, text="Capture Rack", command=self.capture_tile_rack_from_camera).grid(
            row=0, column=2, sticky="ew", padx=10, pady=(10, 4)
        )

        ttk.Label(window, text="Word").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_word_var, width=18).grid(
            row=1, column=1, sticky="ew", padx=10, pady=4
        )
        ttk.Button(window, text="Suggest Word", command=self.suggest_tile_rack_word).grid(
            row=1, column=2, sticky="ew", padx=10, pady=4
        )

        ttk.Label(window, text="Start square").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(window, textvariable=self.tile_rack_start_square_var, width=8).grid(
            row=2, column=1, sticky="w", padx=10, pady=4
        )
        ttk.Button(window, text="Check Word", command=self.check_tile_rack_word).grid(
            row=2, column=2, sticky="ew", padx=10, pady=4
        )

        motion_frame = ttk.LabelFrame(window, text="Rack pickup settings")
        motion_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 4))
        for column in range(6):
            motion_frame.columnconfigure(column, weight=1)

        ttk.Label(motion_frame, text="Left rack X").grid(row=0, column=0, sticky="w", padx=6, pady=(8, 2))
        ttk.Entry(motion_frame, textvariable=self.tile_rack_x_var, width=8).grid(row=0, column=1, padx=6, pady=(8, 2))
        ttk.Label(motion_frame, text="Top Y").grid(row=0, column=2, sticky="w", padx=6, pady=(8, 2))
        ttk.Entry(motion_frame, textvariable=self.tile_rack_y_var, width=8).grid(row=0, column=3, padx=6, pady=(8, 2))
        ttk.Label(motion_frame, text="Vertical gap").grid(row=0, column=4, sticky="w", padx=6, pady=(8, 2))
        ttk.Entry(motion_frame, textvariable=self.tile_rack_spacing_var, width=8).grid(row=0, column=5, padx=6, pady=(8, 2))

        ttk.Label(motion_frame, text="MG995").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(motion_frame, text="Automatic lift/lower").grid(row=1, column=1, columnspan=3, sticky="w", padx=6, pady=2)
        ttk.Label(motion_frame, text="Servo wait").grid(row=1, column=4, sticky="w", padx=6, pady=2)
        ttk.Label(motion_frame, text=f"{DEFAULT_MG995_SERVO_WAIT_SECONDS:g}s").grid(row=1, column=5, sticky="w", padx=6, pady=2)

        ttk.Label(motion_frame, text="XY feed").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(motion_frame, textvariable=self.tile_rack_xy_feed_var, width=8).grid(row=2, column=1, padx=6, pady=2)
        ttk.Label(motion_frame, text="Arduino").grid(row=2, column=2, sticky="w", padx=6, pady=2)
        ttk.Label(motion_frame, text="Uses selected controller port").grid(row=2, column=3, sticky="w", padx=6, pady=2)
        ttk.Button(motion_frame, text="Go To Rack", command=self.move_to_tile_rack_start).grid(
            row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=2
        )

        ttk.Label(motion_frame, text="Move wait").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(motion_frame, text=f"{DEFAULT_TILE_RACK_MOVE_WAIT_SECONDS:g}s").grid(row=3, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(motion_frame, text="Magnet").grid(row=4, column=0, sticky="w", padx=6, pady=(2, 8))
        ttk.Label(motion_frame, text="Automatic on/off").grid(row=4, column=1, columnspan=3, sticky="w", padx=6, pady=(2, 8))

        self.tile_rack_make_button = ttk.Button(
            window,
            text="Make Word On Board",
            command=self.make_tile_rack_word_on_board,
            state="disabled",
        )
        self.tile_rack_make_button.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 4))

        ttk.Label(window, textvariable=self.tile_rack_status_var, wraplength=520).grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 10)
        )
        self._update_tile_rack_word_option()

    def _ensure_tile_rack_state(self) -> None:
        if getattr(self, "_tile_rack_state_ready", False):
            return

        default_feed = "1500"
        feed_var = getattr(self, "feed_rate_var", None)
        if feed_var is not None:
            try:
                default_feed = str(feed_var.get())
            except Exception:
                default_feed = "1500"

        self.tile_rack_letters_var = tk.StringVar(value="")
        self.tile_rack_word_var = tk.StringVar(value="")
        self.tile_rack_start_square_var = tk.StringVar(value="G6")
        self.tile_rack_x_var = tk.StringVar(value="0")
        self.tile_rack_y_var = tk.StringVar(value="40")
        self.tile_rack_spacing_var = tk.StringVar(value="25")
        self.tile_rack_xy_feed_var = tk.StringVar(value=default_feed)
        self.tile_rack_servo_up_var = tk.StringVar(value="SERVO_UP")
        self.tile_rack_servo_down_var = tk.StringVar(value="SERVO_DOWN")
        self.tile_rack_servo_wait_var = tk.StringVar(value="0.4")
        self.tile_rack_move_wait_var = tk.StringVar(value="1.0")
        self.tile_rack_arduino_port_var = tk.StringVar(value="")
        self.tile_rack_magnet_on_var = tk.StringVar(value="M3")
        self.tile_rack_magnet_off_var = tk.StringVar(value="M5")
        self.tile_rack_status_var = tk.StringVar(
            value="Capture or type up to 7 rack letters, then enter a word to place."
        )
        self._tile_rack_normalizing = False
        self._tile_rack_word_normalizing = False
        self._tile_rack_state_ready = True

        self.tile_rack_letters_var.trace_add("write", lambda *_: self._normalize_tile_rack_letters())
        self.tile_rack_word_var.trace_add("write", lambda *_: self._normalize_tile_rack_word())
        self.tile_rack_start_square_var.trace_add("write", lambda *_: self._update_tile_rack_word_option())

    def _normalize_tile_rack_letters(self) -> None:
        if self._tile_rack_normalizing:
            return
        self._tile_rack_normalizing = True
        try:
            normalized = normalize_rack_letters(self.tile_rack_letters_var.get())
            if normalized != self.tile_rack_letters_var.get():
                self.tile_rack_letters_var.set(normalized)
        finally:
            self._tile_rack_normalizing = False
        self._update_tile_rack_word_option()

    def _normalize_tile_rack_word(self) -> None:
        if self._tile_rack_word_normalizing:
            return
        self._tile_rack_word_normalizing = True
        try:
            normalized = normalize_word(self.tile_rack_word_var.get())
            if normalized != self.tile_rack_word_var.get():
                self.tile_rack_word_var.set(normalized)
        finally:
            self._tile_rack_word_normalizing = False
        self._update_tile_rack_word_option()

    def capture_tile_rack_from_camera(self) -> None:
        self._ensure_tile_rack_state()
        try:
            if self._captured_photo_frame is None:
                frame, quality = self._capture_best_photo_for_ocr("tile rack")
                source = f"best camera frame (sharpness {quality.sharpness:.0f})"
            else:
                frame = self._captured_photo_frame.copy()
                source = "captured picture"
            self._set_tile_rack_status(f"Scanning rack letters from the {source}...")
            thread = threading.Thread(
                target=self._scan_tile_rack_worker,
                args=(frame.copy(),),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            self._show_error(exc)

    def _scan_tile_rack_worker(self, frame) -> None:  # type: ignore[no-untyped-def]
        try:
            scan = scan_camera_letters(frame)
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._show_error(exc))
            return
        frame_shape = getattr(frame, "shape", None)
        frame_width = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) >= 2 else 0
        self.root.after(0, lambda scan=scan, frame_width=frame_width: self._handle_tile_rack_scan_result(scan, frame_width))

    def _handle_tile_rack_scan_result(self, scan: CameraLetterScanResult, frame_width: int = 0) -> None:
        rack_letters = self._rack_letters_from_camera_scan(scan, frame_width)
        self.tile_rack_letters_var.set(rack_letters)
        if rack_letters:
            message = f"Left-side vertical tile rack captured {len(rack_letters)} letter(s): {rack_letters}"
        else:
            message = "No left-side rack letters were detected. Type the rack letters or capture a clearer rack image."
        self._set_tile_rack_status(message)
        self._refresh_camera_preview()

    def _rack_letters_from_camera_scan(self, scan: CameraLetterScanResult, frame_width: int = 0) -> str:
        rack_letters = self._filter_left_side_rack_letters(scan, frame_width)
        detected = []
        for captured in sorted(rack_letters, key=self._camera_letter_vertical_sort_key):
            text = getattr(captured, "text", "")
            detected.extend(character for character in str(text).upper() if character.isalpha())
        return normalize_rack_letters("".join(detected))

    def _filter_left_side_rack_letters(self, scan: CameraLetterScanResult, frame_width: int) -> list[object]:
        letters = list(scan.letters)
        if not letters:
            return []

        board_left = None
        if scan.grid is not None and getattr(scan.grid, "corners", None):
            board_left = min(float(point[0]) for point in scan.grid.corners)
        else:
            calibration = getattr(self, "_calibration", None)
            image_corners = getattr(calibration, "image_corners", None)
            if image_corners and len(image_corners) == 4:
                board_left = min(float(point[0]) for point in image_corners)

        filtered = []
        for captured in letters:
            center = self._camera_letter_center(captured)
            if center is None:
                continue
            x, _ = center
            if board_left is not None:
                if x < board_left:
                    filtered.append(captured)
            elif frame_width > 0 and x <= frame_width * 0.40:
                filtered.append(captured)

        if filtered:
            return filtered[:7]
        if board_left is not None or frame_width > 0:
            return []
        return letters[:7]

    def _camera_letter_center(self, captured) -> tuple[float, float] | None:  # type: ignore[no-untyped-def]
        center_x = getattr(captured, "center_x", None)
        center_y = getattr(captured, "center_y", None)
        if center_x is not None and center_y is not None:
            return (float(center_x), float(center_y))
        left = getattr(captured, "left", None)
        top = getattr(captured, "top", None)
        if left is not None and top is not None:
            width = getattr(captured, "width", 0) or 0
            height = getattr(captured, "height", 0) or 0
            return (float(left) + float(width) / 2.0, float(top) + float(height) / 2.0)
        points = getattr(captured, "points", None) or getattr(captured, "corners", None)
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    def _camera_letter_vertical_sort_key(self, captured) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        center = self._camera_letter_center(captured)
        if center is None:
            return (0.0, 0.0)
        return (center[1], center[0])

    def suggest_tile_rack_word(self) -> None:
        self._ensure_tile_rack_state()
        rack_letters = normalize_rack_letters(self.tile_rack_letters_var.get())
        word = self._best_tile_rack_word(rack_letters)
        if not word:
            self._set_tile_rack_status(
                "No playable word was found in the loaded word list. Enter a word manually to check it."
            )
            return
        self.tile_rack_word_var.set(word)
        self._set_tile_rack_status(f"Suggested word: {word}")

    def _best_tile_rack_word(self, rack_letters: str) -> str:
        candidates = self._tile_rack_word_candidates()
        playable = [
            normalize_word(candidate)
            for candidate in candidates
            if can_build_word_from_rack(str(candidate), rack_letters)
        ]
        playable = sorted({word for word in playable if word}, key=lambda word: (-len(word), word))
        return playable[0] if playable else ""

    def _tile_rack_word_candidates(self) -> list[str]:
        candidates: list[str] = []
        for name in (
            "word_bank",
            "_word_bank",
            "valid_words",
            "_valid_words",
            "dictionary_words",
            "_dictionary_words",
        ):
            source = getattr(self, name, None)
            if isinstance(source, dict):
                candidates.extend(str(key) for key in source.keys())
            elif isinstance(source, (list, set, tuple)):
                candidates.extend(str(value) for value in source)

        for name in ("detected_words_var", "camera_words_var", "words_var"):
            variable = getattr(self, name, None)
            if variable is None:
                continue
            try:
                value = variable.get()
            except Exception:
                continue
            candidates.extend(value.replace(",", " ").split())
        return candidates

    def check_tile_rack_word(self) -> None:
        self._ensure_tile_rack_state()
        self._update_tile_rack_word_option(show_status=True)

    def _update_tile_rack_word_option(self, show_status: bool = False) -> None:
        if not getattr(self, "_tile_rack_state_ready", False):
            return
        rack_letters = normalize_rack_letters(self.tile_rack_letters_var.get())
        word = normalize_word(self.tile_rack_word_var.get())
        can_make = can_build_word_from_rack(word, rack_letters)
        placement_error: Exception | None = None
        squares: list[str] = []
        if can_make:
            try:
                squares = horizontal_word_squares(self.tile_rack_start_square_var.get(), word, direction="left")
            except Exception as exc:
                placement_error = exc
        button = getattr(self, "tile_rack_make_button", None)
        if button is not None:
            button.configure(state="normal")
        if show_status:
            if can_make and placement_error is None:
                self._set_tile_rack_status(f"{word} can be made and will be placed at {' '.join(squares)}.")
            elif placement_error is not None:
                self._set_tile_rack_status(str(placement_error))
            elif not rack_letters:
                self._set_tile_rack_status("Capture or type rack letters first.")
            elif not word:
                self._set_tile_rack_status("Enter the word to build from the rack.")
            else:
                self._set_tile_rack_status(f"{word} cannot be made from rack letters {rack_letters}.")

    def make_tile_rack_word_on_board(self) -> None:
        self._ensure_tile_rack_state()
        try:
            rack_letters = normalize_rack_letters(self.tile_rack_letters_var.get())
            if not rack_letters:
                rack_letters = self._capture_tile_rack_letters_for_placement()
            word = normalize_word(self.tile_rack_word_var.get())
            if not word:
                word = self._best_tile_rack_word(rack_letters)
                if word:
                    self.tile_rack_word_var.set(word)
            if not can_build_word_from_rack(word, rack_letters):
                raise ValueError(f"{word or 'That word'} cannot be made from rack letters {rack_letters}.")
            slot_indices = rack_slot_indices_for_word(word, rack_letters)
            target_squares = horizontal_word_squares(self.tile_rack_start_square_var.get(), word, direction="left")
            settings = self._tile_rack_motion_settings()
            self._place_tile_rack_word(word, slot_indices, target_squares, settings)
        except Exception as exc:
            self._show_error(exc)

    def _capture_tile_rack_letters_for_placement(self) -> str:
        if self._captured_photo_frame is None:
            frame, _quality = self._capture_best_photo_for_ocr("tile rack placement")
        else:
            frame = self._captured_photo_frame.copy()

        scan = scan_camera_letters(frame)
        frame_shape = getattr(frame, "shape", None)
        frame_width = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) >= 2 else 0
        rack_letters = self._rack_letters_from_camera_scan(scan, frame_width)
        self.tile_rack_letters_var.set(rack_letters)
        if not rack_letters:
            raise ValueError("No left-side rack letters were detected for placement.")
        self._set_tile_rack_status(f"Detected rack letters for placement: {rack_letters}")
        return rack_letters

    def move_to_tile_rack_start(self) -> None:
        self._ensure_tile_rack_state()
        try:
            settings = self._tile_rack_motion_settings()
            self._send_tile_rack_servo_up(settings)
            self._send_tile_rack_xy_move(
                float(settings["rack_x"]),
                float(settings["rack_y"]),
                float(settings["xy_feed"]),
            )
            self._set_tile_rack_status("Plotter moved to tile rack slot 1 with the magnet lifted.")
        except Exception as exc:
            self._show_error(exc)

    def _tile_rack_motion_settings(self) -> dict[str, float | str]:
        return {
            "rack_x": float(self.tile_rack_x_var.get()),
            "rack_y": float(self.tile_rack_y_var.get()),
            "slot_spacing": float(self.tile_rack_spacing_var.get()),
            "xy_feed": float(self.tile_rack_xy_feed_var.get()),
            "servo_up_commands": MG995_SERVO_UP_COMMANDS,
            "servo_down_commands": MG995_SERVO_DOWN_COMMANDS,
            "servo_wait": DEFAULT_MG995_SERVO_WAIT_SECONDS,
            "move_wait": DEFAULT_TILE_RACK_MOVE_WAIT_SECONDS,
            "arduino_port": "",
            "magnet_on_commands": MAGNET_ON_COMMANDS,
            "magnet_off_commands": MAGNET_OFF_COMMANDS,
        }

    def _place_tile_rack_word(
        self,
        word: str,
        slot_indices: list[int],
        target_squares: list[str],
        settings: dict[str, float | str],
    ) -> None:
        with self._tile_rack_serial_session():
            self._set_tile_rack_status(f"Placing {word} from the tile rack...")
            self._log(f"Tile rack placement started for {word}.")
            self._send_tile_rack_servo_up(settings)
            self._send_tile_rack_magnet_off(settings)

            for letter, slot_index, target_square in zip(word, slot_indices, target_squares):
                rack_x, rack_y = vertical_rack_slot_position(
                    float(settings["rack_x"]),
                    float(settings["rack_y"]),
                    slot_index,
                    float(settings["slot_spacing"]),
                )

                self._log(f"Picking {letter} from rack slot {slot_index + 1}.")
                self._send_tile_rack_servo_up(settings)
                self._send_tile_rack_xy_move(rack_x, rack_y, float(settings["xy_feed"]))
                self._wait_after_tile_rack_move(settings)
                self._send_tile_rack_servo_down(settings)
                self._send_tile_rack_magnet_on(settings)
                self._send_tile_rack_delay(settings)
                self._send_tile_rack_servo_up(settings)

                self._log(f"Placing {letter} on {target_square}.")
                self._send_tile_rack_square_move(target_square, settings)
                self._wait_after_tile_rack_move(settings)
                self._send_tile_rack_servo_down(settings)
                self._send_tile_rack_magnet_off(settings)
                self._send_tile_rack_delay(settings)
                self._send_tile_rack_servo_up(settings)

            self._set_tile_rack_status(f"Placed {word} right-to-left from {target_squares[0]}.")
            self._log(f"Tile rack placement complete for {word}: {' '.join(target_squares)}.")

    def _send_tile_rack_xy_move(self, x: float, y: float, feed: float) -> None:
        self._send_tile_rack_command(f"G0 X{x:g} Y{y:g} F{feed:g}", prefer_arduino=False)

    def _send_tile_rack_square_move(self, square: str, settings: dict[str, float | str]) -> None:
        command = self._tile_rack_square_move_command(square, float(settings["xy_feed"]))
        if command:
            self._send_tile_rack_command(command, prefer_arduino=False)
            return
        self._send_square_move(square)

    def _tile_rack_square_move_command(self, square: str, feed: float) -> str:
        calibration = None
        for method_name in ("_calibration_from_form", "_calibration"):
            source = getattr(self, method_name, None)
            try:
                calibration = source() if callable(source) else source
            except Exception:
                calibration = None
            if calibration is not None:
                break

        if calibration is not None:
            for args in (
                (calibration, square, feed),
                (square, calibration, feed),
                (calibration, square),
                (square, calibration),
            ):
                try:
                    command = format_move_command(*args)
                    if command:
                        return str(command)
                except TypeError:
                    continue
                except Exception:
                    continue

            for method_name in (
                "square_center",
                "center_for_square",
                "coordinates_for_square",
                "square_to_xy",
                "xy_for_square",
                "square_center_mm",
                "plotter_coordinates_for_square",
                "square_to_plotter_xy",
                "board_square_to_plotter_position",
            ):
                method = getattr(calibration, method_name, None)
                if not callable(method):
                    continue
                try:
                    x, y = method(square)
                except Exception:
                    continue
                for args in ((x, y, feed), (x, y), (float(x), float(y), feed)):
                    try:
                        command = format_move_command(*args)
                        if command:
                            return str(command)
                    except TypeError:
                        continue
                    except Exception:
                        continue
                return f"G0 X{float(x):g} Y{float(y):g} F{feed:g}"
        return ""

    def _send_tile_rack_z_move(self, z: float, feed: float) -> None:
        self._send_tile_rack_command(f"G0 Z{z:g} F{feed:g}", prefer_arduino=False)

    def _send_tile_rack_servo_up(self, settings: dict[str, float | str]) -> None:
        self._send_tile_rack_aux_command_sequence(settings["servo_up_commands"])
        self._send_tile_rack_delay(settings)

    def _send_tile_rack_servo_down(self, settings: dict[str, float | str]) -> None:
        self._send_tile_rack_aux_command_sequence(settings["servo_down_commands"])
        self._send_tile_rack_delay(settings)

    def _send_tile_rack_magnet_on(self, settings: dict[str, float | str]) -> None:
        self._send_tile_rack_aux_command_sequence(settings["magnet_on_commands"])

    def _send_tile_rack_magnet_off(self, settings: dict[str, float | str]) -> None:
        self._send_tile_rack_aux_command_sequence(settings["magnet_off_commands"])

    def _send_tile_rack_delay(self, settings: dict[str, float | str]) -> None:
        import time

        wait_seconds = max(0.0, float(settings["servo_wait"]))
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _wait_after_tile_rack_move(self, settings: dict[str, float | str]) -> None:
        import time

        wait_seconds = max(0.0, float(settings["move_wait"]))
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _send_tile_rack_aux_command(self, command: str) -> None:
        self._send_tile_rack_command(command, prefer_arduino=False)

    def _send_tile_rack_aux_command_sequence(self, commands) -> None:  # type: ignore[no-untyped-def]
        last_error: Exception | None = None
        sent_any = False
        for command in commands:
            try:
                self._send_tile_rack_aux_command(str(command))
                sent_any = True
            except Exception as exc:
                last_error = exc
        if not sent_any and last_error is not None:
            raise last_error

    def _tile_rack_serial_session(self):  # type: ignore[no-untyped-def]
        import contextlib

        return contextlib.nullcontext()

        if getattr(self, "_tile_rack_active_sender", None) is not None:
            return contextlib.nullcontext()

        try:
            config = self._tile_rack_serial_config(prefer_arduino=False)
        except Exception:
            return contextlib.nullcontext()
        if not self._serial_config_has_port(config, prefer_arduino=False):
            return contextlib.nullcontext()

        try:
            sender = GCodeSender(config)
        except TypeError:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def session():
            active_sender = sender
            aux_sender = None
            active_aux_sender = None
            entered = False
            aux_entered = False
            try:
                if hasattr(sender, "__enter__") and hasattr(sender, "__exit__"):
                    active_sender = sender.__enter__()
                    entered = True
                self._tile_rack_active_sender = active_sender
                plotter_port = self._serial_config_port(config)
                aux_port = self._selected_tile_rack_serial_port(prefer_arduino=True)
                if aux_port and aux_port != plotter_port:
                    aux_config = self._tile_rack_serial_config(prefer_arduino=True)
                    aux_sender = GCodeSender(aux_config)
                    active_aux_sender = aux_sender
                    if hasattr(aux_sender, "__enter__") and hasattr(aux_sender, "__exit__"):
                        active_aux_sender = aux_sender.__enter__()
                        aux_entered = True
                    self._tile_rack_active_aux_sender = active_aux_sender
                else:
                    self._tile_rack_active_aux_sender = active_sender
                yield
            finally:
                self._tile_rack_active_sender = None
                self._tile_rack_active_aux_sender = None
                if aux_sender is not None:
                    if aux_entered:
                        aux_sender.__exit__(None, None, None)
                    else:
                        close = getattr(aux_sender, "close", None)
                        if callable(close):
                            close()
                if entered:
                    sender.__exit__(None, None, None)
                else:
                    close = getattr(sender, "close", None)
                    if callable(close):
                        close()

        return session()

    def _send_tile_rack_command(self, command: str, prefer_arduino: bool = False) -> None:
        command = command.strip()
        if not command:
            return

        active_sender = (
            getattr(self, "_tile_rack_active_aux_sender", None)
            if prefer_arduino
            else getattr(self, "_tile_rack_active_sender", None)
        )
        if active_sender is not None:
            if self._send_tile_rack_command_with_gcode_sender(command, prefer_arduino=prefer_arduino):
                return
            raise RuntimeError(f"Could not send tile rack command: {command}")

        if prefer_arduino and self._tile_rack_arduino_port():
            if self._send_tile_rack_command_with_gcode_sender(command, prefer_arduino=True):
                return
            raise RuntimeError("Could not send the tile rack MG995/magnet command to the Arduino port.")

        for method_name in (
            "_send_commands",
            "_send_serial_commands",
            "_send_controller_commands",
            "_send_gcode_commands",
            "send_commands",
            "send_serial_commands",
            "send_controller_commands",
            "send_gcode_commands",
            "_send_raw_command",
            "_send_manual_command",
            "_send_command",
            "_send_gcode_command",
            "send_raw_command",
            "send_manual_command",
            "send_command",
            "send_gcode_command",
        ):
            method = getattr(self, method_name, None)
            if callable(method):
                if self._try_tile_rack_sender_method(method, command, prefer_list="commands" in method_name):
                    return

        sender = (
            getattr(self, "_sender", None)
            or getattr(self, "sender", None)
            or getattr(self, "_gcode_sender", None)
            or getattr(self, "gcode_sender", None)
        )
        if sender is not None:
            for method_name in ("send_commands", "send_command", "send", "write"):
                method = getattr(sender, method_name, None)
                if callable(method):
                    if self._try_tile_rack_sender_method(method, command, prefer_list="commands" in method_name):
                        return

        if self._send_tile_rack_command_with_gcode_sender(command, prefer_arduino=prefer_arduino):
            return

        if not self._selected_tile_rack_serial_port(prefer_arduino=prefer_arduino):
            raise RuntimeError("Select the plotter COM port before using the tile rack movement.")

        raise RuntimeError(
            "No raw G-code sender was found for rack Z or magnet commands. "
            "Add a manual command sender or connect this method to the controller command path."
        )

    def _try_tile_rack_sender_method(self, method, command: str, prefer_list: bool = False) -> bool:  # type: ignore[no-untyped-def]
        list_payloads = (
            ([command],),
            ([command], "tile rack"),
            ([command], "Tile rack command"),
        )
        string_payloads = (
            (command,),
            (command, "tile rack"),
            (command, "Tile rack command"),
        )
        payloads = list_payloads + string_payloads if prefer_list else string_payloads + list_payloads
        for payload in payloads:
            try:
                method(*payload)
                return True
            except TypeError:
                continue
        return False

    def _send_tile_rack_command_with_gcode_sender(self, command: str, prefer_arduino: bool = False) -> bool:
        active_sender = (
            getattr(self, "_tile_rack_active_aux_sender", None)
            if prefer_arduino
            else getattr(self, "_tile_rack_active_sender", None)
        )
        if active_sender is not None:
            return self._send_tile_rack_command_to_sender(active_sender, command)

        try:
            config = self._tile_rack_serial_config(prefer_arduino=prefer_arduino)
        except Exception:
            return False
        if not self._serial_config_has_port(config, prefer_arduino=prefer_arduino):
            return False

        try:
            sender = GCodeSender(config)
        except TypeError:
            return False

        context_sender = sender
        if hasattr(sender, "__enter__") and hasattr(sender, "__exit__"):
            with sender as active_sender:
                context_sender = active_sender
                return self._send_tile_rack_command_to_sender(context_sender, command)
        return self._send_tile_rack_command_to_sender(context_sender, command)

    def _send_tile_rack_command_to_sender(self, sender, command: str) -> bool:  # type: ignore[no-untyped-def]
        for method_name in ("send_commands", "send_command", "send", "write"):
            method = getattr(sender, method_name, None)
            if callable(method) and self._try_tile_rack_sender_method(method, command, prefer_list="commands" in method_name):
                return True
        return False

    def _tile_rack_serial_config(self, prefer_arduino: bool = False) -> SerialConfig:
        import dataclasses
        import inspect

        for method_name in (
            "_serial_config_from_form",
            "_serial_config_from_inputs",
            "_serial_config",
            "_get_serial_config",
            "serial_config_from_form",
            "get_serial_config",
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                config = method()
            except Exception:
                continue
            if isinstance(config, SerialConfig):
                return self._tile_rack_config_with_selected_port(config, prefer_arduino=prefer_arduino)

        names = []
        if dataclasses.is_dataclass(SerialConfig):
            names = [field.name for field in dataclasses.fields(SerialConfig)]
        else:
            signature = inspect.signature(SerialConfig)
            names = [name for name in signature.parameters if name != "self"]

        values = {}
        for name in names:
            value = self._tile_rack_serial_config_value(name, prefer_arduino=prefer_arduino)
            if value is not None:
                values[name] = value
        return self._tile_rack_config_with_selected_port(SerialConfig(**values), prefer_arduino=prefer_arduino)

    def _tile_rack_config_with_selected_port(self, config: SerialConfig, prefer_arduino: bool = False) -> SerialConfig:
        import dataclasses

        selected_port = self._selected_tile_rack_serial_port(prefer_arduino=prefer_arduino)
        if not selected_port:
            return config

        for name in ("port", "serial_port", "com_port"):
            if not hasattr(config, name):
                continue
            current = getattr(config, name)
            if current is not None and str(current).strip() and not prefer_arduino:
                return config
            if dataclasses.is_dataclass(config):
                return dataclasses.replace(config, **{name: selected_port})
            try:
                setattr(config, name, selected_port)
            except Exception:
                return config
            return config
        return config

    def _serial_config_has_port(self, config: SerialConfig, prefer_arduino: bool = False) -> bool:
        for name in ("port", "serial_port", "com_port"):
            if hasattr(config, name):
                value = getattr(config, name)
                if value is not None and str(value).strip():
                    return True
        return bool(self._selected_tile_rack_serial_port(prefer_arduino=prefer_arduino))

    def _serial_config_port(self, config: SerialConfig) -> str | None:
        for name in ("port", "serial_port", "com_port"):
            if hasattr(config, name):
                value = getattr(config, name)
                if value is not None and str(value).strip():
                    return self._normalize_serial_port_name(value)
        return None

    def _tile_rack_serial_config_value(self, name: str, prefer_arduino: bool = False):  # type: ignore[no-untyped-def]
        key = name.lower()
        if key in {"port", "serial_port", "com_port"}:
            return self._selected_tile_rack_serial_port(prefer_arduino=prefer_arduino) or self._read_tile_rack_gui_value(
                "port_var",
                "serial_port_var",
                "com_port_var",
                "plotter_port_var",
                "selected_port_var",
            )
        if key in {"baud", "baudrate", "baud_rate"}:
            return int(
                self._read_tile_rack_gui_value(
                    "baud_var",
                    "baudrate_var",
                    "baud_rate_var",
                    default="115200",
                )
            )
        if key == "timeout" or "timeout" in key:
            return float(self._read_tile_rack_gui_value("timeout_var", "serial_timeout_var", default="2.0"))
        if key in {"dry_run", "dryrun"}:
            value = self._read_tile_rack_gui_value("dry_run_var", "dryrun_var", default=False)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if "settle" in key or "delay" in key:
            return float(self._read_tile_rack_gui_value("settle_seconds_var", "startup_delay_var", default="2.0"))
        return None

    def _tile_rack_arduino_port(self) -> str | None:
        source = getattr(self, "tile_rack_arduino_port_var", None)
        if source is None:
            source_value = None
        else:
            try:
                source_value = source.get()
            except Exception:
                source_value = None
        if self._looks_like_serial_port(source_value):
            return self._normalize_serial_port_name(source_value)
        return None

    def _detect_tile_rack_mg995_port(self) -> str | None:
        import time

        try:
            import serial
        except Exception:
            return None

        try:
            ports = list_serial_ports()
        except Exception:
            return None

        port_names: list[str] = []
        selected_plotter_port = self._selected_tile_rack_serial_port(prefer_arduino=False)
        if selected_plotter_port:
            port_names.append(selected_plotter_port)

        for port in ports:
            name = getattr(port, "device", None) or getattr(port, "name", None) or str(port)
            normalized_name = self._normalize_serial_port_name(name)
            if self._looks_like_serial_port(normalized_name) and normalized_name not in port_names:
                port_names.append(normalized_name)

        for port_name in port_names:
            try:
                with serial.Serial(port_name, 115200, timeout=0.35, write_timeout=0.35) as connection:
                    time.sleep(1.8)
                    connection.reset_input_buffer()
                    connection.write(b"MG995_PING\n")
                    connection.flush()
                    deadline = time.monotonic() + 1.2
                    response = ""
                    while time.monotonic() < deadline:
                        line = connection.readline().decode(errors="ignore")
                        response += line
                        if "SCRABBLE_MG995_A1_READY" in response or "MG995" in response:
                            return port_name
            except Exception:
                continue
        return None

    def _selected_tile_rack_serial_port(self, prefer_arduino: bool = False) -> str | None:
        if prefer_arduino:
            arduino_port = self._tile_rack_arduino_port()
            if arduino_port:
                return arduino_port

        port = self._read_tile_rack_gui_value(
            "port_var",
            "serial_port_var",
            "com_port_var",
            "plotter_port_var",
            "selected_port_var",
            "port_combo",
            "serial_port_combo",
            "com_port_combo",
        )
        if self._looks_like_serial_port(port):
            return self._normalize_serial_port_name(port)

        for name, source in vars(self).items():
            lowered = name.lower()
            if not prefer_arduino and "arduino" in lowered:
                continue
            if "port" not in lowered and "com" not in lowered and "serial" not in lowered:
                continue
            if not hasattr(source, "get"):
                continue
            try:
                value = source.get()
            except Exception:
                continue
            if self._looks_like_serial_port(value):
                return self._normalize_serial_port_name(value)
        return None

    def _looks_like_serial_port(self, value) -> bool:  # type: ignore[no-untyped-def]
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        lowered = text.lower()
        return lowered.startswith("com") or lowered.startswith("/dev/") or lowered.startswith("usb")

    def _normalize_serial_port_name(self, value) -> str:  # type: ignore[no-untyped-def]
        text = str(value).strip()
        if text.upper().startswith("COM"):
            token = text.split()[0].rstrip(":,;")
            return token
        return text

    def _read_tile_rack_gui_value(self, *names: str, default=None):  # type: ignore[no-untyped-def]
        for name in names:
            source = getattr(self, name, None)
            if source is None:
                continue
            try:
                value = source.get() if hasattr(source, "get") else source
            except Exception:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return default

    def _set_tile_rack_status(self, message: str) -> None:
        if getattr(self, "_tile_rack_state_ready", False):
            self.tile_rack_status_var.set(message)
        self._set_status(message)

    def reset_to_start(self) -> None:
        try:
            self._send_reset()
        except Exception as exc:
            self._show_error(exc)

    def go_to_cart(self) -> None:
        try:
            self._send_cart_move()
        except Exception as exc:
            self._show_error(exc)

    def scan_board_from_camera(self) -> None:
        try:
            calibration = self._calibration_from_form()
            calibration.validate_ready_for_scan()
            if self._captured_photo_frame is None:
                frame, quality = self._capture_best_photo_for_ocr("board scan")
                source = f"best camera frame (sharpness {quality.sharpness:.0f})"
            else:
                frame = self._captured_photo_frame.copy()
                source = "captured picture"
            self._set_status(f"Scanning the calibrated board from the {source}...")
            self._log(f"Calibrated board scan started from {source}.")
            thread = threading.Thread(
                target=self._scan_board_worker,
                args=(frame.copy(), calibration),
                daemon=True,
            )
            thread.start()
        except RuntimeError as exc:
            if str(exc) == "Start the camera first.":
                raise_user_error(str(exc))
            else:
                self._show_error(exc)
        except Exception as exc:
            self._show_error(exc)

    def take_picture_from_camera(self) -> None:
        try:
            _, quality = self._capture_best_photo_for_ocr("manual picture")
            self._set_status(
                f"Best picture captured. Sharpness {quality.sharpness:.0f}; click Find Words, Capture Letters, or Scan Board."
            )
        except RuntimeError as exc:
            if str(exc) == "Start the camera first.":
                raise_user_error(str(exc))
            else:
                self._show_error(exc)
        except Exception as exc:
            self._show_error(exc)

    def resume_live_camera(self) -> None:
        if self._captured_photo_frame is None:
            self._set_status("Live camera view is already active.")
            return

        self._captured_photo_frame = None
        self._invalidate_camera_scans()
        self._last_camera_letter_scan = None
        self._last_camera_word_scan = None
        self.captured_letters_var.set("")
        self._set_camera_words_text("")
        self._refresh_camera_preview()
        self._set_status("Live camera view resumed.")
        self._log("Live camera view resumed.")

    def _capture_best_photo_for_ocr(self, reason: str):  # type: ignore[no-untyped-def]
        frame, quality = self._capture_best_camera_frame()
        self._captured_photo_frame = frame.copy()
        self._invalidate_camera_scans()
        self._last_camera_letter_scan = None
        self._last_camera_word_scan = None
        self.captured_letters_var.set("")
        self._set_camera_words_text("")
        self._refresh_camera_preview()
        self._log(
            f"Selected best camera frame for {reason}: "
            f"score={quality.score:.0f}, sharpness={quality.sharpness:.0f}, "
            f"contrast={quality.contrast:.1f}, brightness={quality.brightness:.1f}."
        )
        return self._captured_photo_frame.copy(), quality

    def _capture_best_camera_frame(self):  # type: ignore[no-untyped-def]
        frames = []
        if self._latest_frame is not None:
            frames.append(self._latest_frame.copy())

        deadline = time.monotonic() + BEST_CAPTURE_TIMEOUT_SECONDS
        while (
            self._camera is not None
            and len(frames) < BEST_CAPTURE_FRAME_COUNT
            and time.monotonic() < deadline
        ):
            frame = read_camera_frame(self._camera)
            if frame is not None:
                self._camera_failed_reads = 0
                self._latest_frame = frame
                frames.append(frame.copy())
            else:
                self._camera_failed_reads += 1
                if self._camera_failed_reads >= CAMERA_READ_FAILURE_LIMIT:
                    break
            time.sleep(BEST_CAPTURE_FRAME_DELAY_SECONDS)

        if not frames:
            raise RuntimeError("Start the camera first.")

        frame, quality = select_best_frame(frames)
        return frame.copy(), quality

    def capture_letters_from_camera(self) -> None:
        try:
            confidence_threshold = self._ocr_confidence_threshold()
            if self._captured_photo_frame is None:
                frame, quality = self._capture_best_photo_for_ocr("letter capture")
                source = f"best camera frame (sharpness {quality.sharpness:.0f})"
            else:
                frame = self._captured_photo_frame.copy()
                source = "captured picture"
            scan_token = self._next_camera_letter_scan_token()
            self._set_status(f"Capturing letters from the {source}...")
            self._log(f"Camera letter capture started from {source}.")
            thread = threading.Thread(
                target=self._camera_letter_scan_worker,
                args=(frame.copy(), confidence_threshold, True, scan_token),
                daemon=True,
            )
            thread.start()
        except RuntimeError as exc:
            if str(exc) == "Start the camera first.":
                raise_user_error(str(exc))
            else:
                self._show_error(exc)
        except Exception as exc:
            self._show_error(exc)

    def identify_words_with_easyocr(self) -> None:
        if self._camera_word_scan_running:
            self._set_status("EasyOCR word detection is already running.")
            return

        try:
            confidence_threshold = self._ocr_confidence_threshold()
            if self._captured_photo_frame is None:
                frame, quality = self._capture_best_photo_for_ocr("word detection")
                source = f"best camera frame (sharpness {quality.sharpness:.0f})"
            else:
                frame = self._captured_photo_frame.copy()
                source = "captured picture"
            self._camera_word_scan_running = True
            scan_token = self._next_camera_word_scan_token()
            self._set_status(f"Finding words in the {source} with EasyOCR...")
            self._set_camera_words_text("Finding words...")
            self._log(f"EasyOCR word detection started from {source}.")
            thread = threading.Thread(
                target=self._camera_word_scan_worker,
                args=(frame.copy(), confidence_threshold, True, scan_token),
                daemon=True,
            )
            thread.start()
        except RuntimeError as exc:
            if str(exc) == "Start the camera first.":
                raise_user_error(str(exc))
            else:
                self._show_error(exc)
        except Exception as exc:
            self._camera_word_scan_running = False
            self._show_error(exc)

    def identify_words_with_paddleocr(self) -> None:
        self.identify_words_with_easyocr()

    def identify_words_with_gemini(self) -> None:
        self.identify_words_with_easyocr()

    def calculate_score_from_board(self) -> None:
        try:
            self._normalize_board_entries()
            calibration = self._calibration_from_form()
            board_letters = self._board_letters_from_form()
            score = score_board(
                board_letters,
                premium_layout=calibration.premium_layout,
                blank_squares=self._blank_squares_from_form(),
            )
            self._handle_score_result(score)
            self._set_camera_words_text(format_words_by_direction(matched_matrix_words(board_letters)))
        except Exception as exc:
            self._show_error(exc)

    def ask_gemini(self) -> None:
        try:
            calibration = self._calibration_from_form()
            calibration.validate_ready_for_move()
            image_jpeg = self._latest_camera_jpeg() if self.gemini_include_camera_var.get() else None
            self._set_status("Asking Gemini for the next plotter action...")
            self._log("Gemini request started.")
            thread = threading.Thread(
                target=self._ask_gemini_worker,
                args=(calibration, image_jpeg),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            self._show_error(exc)

    def run_gemini_action(self) -> None:
        try:
            if self._last_agent_action is None:
                self.ask_gemini()
                return
            self._execute_agent_action(self._last_agent_action)
            self._last_agent_action = None
        except Exception as exc:
            self._show_error(exc)

    def _scan_board_worker(self, frame, calibration: PlotterCalibration) -> None:  # type: ignore[no-untyped-def]
        try:
            scan = scan_board_image(frame, calibration)
            score = score_board(
                scan.board_letters(),
                premium_layout=calibration.premium_layout,
                blank_squares=scan.blank_squares(),
            )
            self.root.after(0, lambda: self._handle_scan_result(scan, score, announce=True))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._show_error(exc))

    def _camera_letter_scan_worker(
        self,
        frame,
        confidence_threshold: float,
        announce: bool,
        scan_token: int,
    ) -> None:  # type: ignore[no-untyped-def]
        try:
            scan = scan_camera_letters(frame, confidence_threshold=confidence_threshold)
            self.root.after(
                0,
                lambda: self._handle_camera_letter_scan_result(scan, announce=announce, scan_token=scan_token),
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda exc=exc: self._handle_camera_letter_scan_error(
                    exc,
                    announce=announce,
                    scan_token=scan_token,
                ),
            )

    def _camera_word_scan_worker(
        self,
        frame,
        confidence_threshold: float,
        announce: bool,
        scan_token: int,
    ) -> None:  # type: ignore[no-untyped-def]
        try:
            scan = scan_camera_words(frame, confidence_threshold=confidence_threshold)
            self.root.after(
                0,
                lambda: self._handle_camera_word_scan_result(scan, announce=announce, scan_token=scan_token),
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda exc=exc: self._handle_camera_word_scan_error(
                    exc,
                    announce=announce,
                    scan_token=scan_token,
                ),
            )

    def _handle_camera_letter_scan_result(
        self,
        scan: CameraLetterScanResult,
        announce: bool = False,
        scan_token: int | None = None,
    ) -> None:
        if scan_token is not None and scan_token != self._camera_letter_scan_token:
            return
        self._live_letter_scan_running = False
        self._last_live_letter_scan_error = None
        self._last_camera_letter_scan = scan
        captured_text = scan.text()
        self.captured_letters_var.set(captured_text)
        self._lock_scan_to_manual_board_grid(scan)
        aligned_count = 0
        should_map_to_board = announce or self._pending_actuator_challenge_after_word_scan
        if should_map_to_board:
            aligned_count = self._apply_camera_letter_scan_to_locked_board_grid(scan)
            if aligned_count == 0:
                aligned_count = self._apply_camera_ocr_grid_to_board_form(scan.grid)
        if captured_text:
            message = f"Captured {len(scan.letters)} letter group(s): {captured_text}"
        else:
            message = "No letters captured from the camera."
        if scan.grid is not None:
            message += f" Auto grid aligned {len(scan.grid.cells)} letter(s)."
        elif announce:
            message += " Board grid was not visible enough to align."
        self._set_status(message)
        self._refresh_camera_preview()
        if announce:
            if aligned_count:
                self._log(f"Camera OCR grid placed {aligned_count} letter(s) into the board matrix.")
            self._log(message)

    def _handle_camera_letter_scan_error(
        self,
        exc: Exception,
        announce: bool = False,
        scan_token: int | None = None,
    ) -> None:
        if scan_token is not None and scan_token != self._camera_letter_scan_token:
            return
        self._live_letter_scan_running = False
        message = str(exc)
        if announce or self._last_live_letter_scan_error != message:
            self._last_live_letter_scan_error = message
            self._set_status(f"Letter capture paused: {message}")
            self._log(f"Letter capture paused: {message}")
        if "easyocr" in message.lower():
            self.live_letter_scan_var.set(False)

    def _handle_camera_word_scan_result(
        self,
        scan: CameraWordScanResult,
        announce: bool = False,
        scan_token: int | None = None,
    ) -> None:
        if scan_token is not None and scan_token != self._camera_word_scan_token:
            return
        self._camera_word_scan_running = False
        self._last_live_word_scan_error = None
        self._last_camera_word_scan = scan
        self._lock_scan_to_manual_board_grid(scan)
        aligned_count = 0
        should_map_to_board = announce or self._pending_actuator_challenge_after_word_scan
        if should_map_to_board:
            aligned_count = self._apply_camera_letter_scan_to_locked_board_grid(scan)
            if aligned_count == 0:
                aligned_count = self._apply_camera_ocr_grid_to_board_form(scan.grid)
        formatted_words = format_camera_words_numbered(scan.words)
        if scan.grid is not None and scan.grid.cells:
            grid_words = matched_matrix_words(scan.grid.board_letters())
            if grid_words:
                formatted_words = format_words_by_direction(grid_words)
        self._set_camera_words_text(formatted_words)
        if scan.words:
            message = f"EasyOCR matched {len(scan.words)} word(s) from {len(scan.text_boxes)} text box(es)."
            if announce:
                if aligned_count:
                    self._log(f"Camera OCR grid placed {aligned_count} letter(s) into the board matrix.")
                self._log("Matched words:\n" + formatted_words)
        else:
            message = f"EasyOCR found {len(scan.text_boxes)} text box(es), but no words matched the list."
            if announce:
                self._log("Matched words: none")
        if scan.grid is not None:
            message += f" Auto grid aligned {len(scan.grid.cells)} letter(s)."
        elif announce:
            message += " Board grid was not visible enough to align."
        self._set_status(message)
        self._refresh_camera_preview()
        if announce:
            self._try_send_camera_words_to_actuator()
        if self._pending_actuator_challenge_after_word_scan:
            self._finish_pending_actuator_challenge()

    def _handle_camera_word_scan_error(
        self,
        exc: Exception,
        announce: bool = False,
        scan_token: int | None = None,
    ) -> None:
        if scan_token is not None and scan_token != self._camera_word_scan_token:
            return
        self._camera_word_scan_running = False
        if self._pending_actuator_challenge_after_word_scan:
            self._pending_actuator_challenge_after_word_scan = False
        message = str(exc)
        if announce or self._last_live_word_scan_error != message:
            self._last_live_word_scan_error = message
            self._set_status(f"Word detection paused: {message}")
            self._log(f"Word detection paused: {message}")
        if "easyocr" in message.lower():
            self.live_word_scan_var.set(False)

    def _handle_scan_result(self, scan: BoardScanResult, score: ScoreResult, announce: bool = True) -> None:
        self._last_scan = scan
        try:
            threshold = float(self.ocr_confidence_threshold_var.get())
        except ValueError:
            threshold = self._calibration.ocr_confidence_threshold
        cells_by_square = {cell.square: cell for cell in scan.cells}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = cells_by_square.get(f"{chr(ord('A') + col)}{row + 1}")
                letter = cell.letter if cell else ""
                self._letter_vars[row][col].set(letter)
                entry = self._letter_entries[row][col]
                if cell is not None and cell.occupied and (not cell.letter or cell.confidence < threshold):
                    entry.configure(bg="#fff2a8")
                elif cell is not None and cell.letter:
                    entry.configure(bg="#e8f7e8")
                else:
                    entry.configure(bg="white")
        self.blank_squares_var.set("")
        self._set_camera_words_text(format_words_by_direction(matched_matrix_words(scan.board_letters())))
        occupied = sum(1 for cell in scan.cells if cell.occupied)
        recognized = sum(1 for cell in scan.cells if cell.letter)
        if announce:
            self._handle_score_result(score)
            self._set_status(f"Scanned {recognized} letters from {occupied} occupied cells.")
            self._log(f"Scan complete: {recognized} recognized, {occupied} occupied.")
        else:
            self._set_status(
                f"Live scan: {recognized} letters from {occupied} occupied cells. Score {score.total_score}."
            )

    def _apply_camera_ocr_grid_to_board_form(self, grid) -> int:  # type: ignore[no-untyped-def]
        if grid is None:
            return 0

        board_letters = grid.board_letters()
        placed = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                letter = normalize_letter(board_letters[row][col])
                self._letter_vars[row][col].set(letter)
                entry = self._letter_entries[row][col]
                if letter:
                    placed += 1
                    entry.configure(bg="#e8f7e8")
                else:
                    entry.configure(bg="white")
        self._set_camera_words_text(format_words_by_direction(matched_matrix_words(board_letters)))
        return placed

    def _handle_score_result(self, score: ScoreResult) -> None:
        self._set_status(f"Board score: {score.total_score}")
        self._log(self._format_score_result(score))

    def _ask_gemini_worker(self, calibration: PlotterCalibration, image_jpeg: bytes | None) -> None:
        try:
            agent = GeminiPlotterAgent(
                api_key=self.gemini_api_key_var.get(),
                model=self.gemini_model_var.get(),
            )
            action = agent.decide(
                self.gemini_objective_var.get(),
                calibration,
                image_jpeg=image_jpeg,
                timeout=max(5.0, float(self.timeout_var.get()) * 10.0),
            )
            self.root.after(0, lambda: self._handle_agent_action(action))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._show_error(exc))

    def _handle_agent_action(self, action: PlotterAgentAction) -> None:
        self._last_agent_action = action
        summary = self._format_agent_action(action)
        self._set_status(f"Gemini chose: {summary}")
        self._log(f"Gemini chose: {summary}")
        if action.reason:
            self._log(f"Gemini reason: {action.reason}")

    def _execute_agent_action(self, action: PlotterAgentAction) -> None:
        if action.action == "move_square" and action.square:
            self.square_var.set(action.square)
            self._send_square_move(action.square)
            return
        if action.action == "go_cart":
            self._send_cart_move()
            return
        if action.action == "reset":
            self._send_reset()
            return
        self._set_status("Gemini chose no movement.")
        self._log("Gemini action: none")

    def _format_agent_action(self, action: PlotterAgentAction) -> str:
        if action.action == "move_square" and action.square:
            return f"move to {action.square}"
        if action.action == "go_cart":
            return "go to cart"
        if action.action == "reset":
            return "reset to start"
        return "none"

    def _latest_camera_jpeg(self) -> bytes | None:
        frame = self._current_camera_ocr_frame()
        if frame is None:
            return None
        cv2 = _require_cv2()
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode the current camera frame for Gemini.")
        return encoded.tobytes()

    def _send_square_move(self, square_label: str) -> None:
        if getattr(self, "_force_z_up_before_pick_drop_move", False):
            self._send_pick_drop_aux_command("ZU")

        target = str(square_label).strip().upper()
        if self._is_tile_rack_target(target):
            self._send_tile_rack_target_move(target)
            return

        calibration = self._calibration_from_form()
        calibration.validate_ready_for_move()
        square = parse_square_label(square_label)
        x, y = calibration.square_center_in_machine(square)
        sender = self._get_sender()
        gcode, responses = sender.send_move(
            x,
            y,
            feed_rate=self._optional_float(self.feed_rate_var.get()),
            command=self.command_var.get().strip() or "G0",
        )
        self._set_status(f"Sent {square.label} to {sender.config.port}")
        self._log(f"Sent: {gcode}")
        if responses:
            self._log("Responses: " + " | ".join(responses))

    def _send_cart_move(self) -> None:
        calibration = self._calibration_from_form()
        calibration.validate_ready_for_move()
        x, y = calibration.cart_position_in_machine()
        sender = self._get_sender()
        gcode, responses = sender.send_move(
            x,
            y,
            feed_rate=self._optional_float(self.feed_rate_var.get()),
            command=self.command_var.get().strip() or "G0",
        )
        self._set_status(f"Sent cart move to X={x:.3f}, Y={y:.3f}")
        self._log(f"Sent: {gcode}")
        if responses:
            self._log("Responses: " + " | ".join(responses))

    def _send_reset(self) -> None:
        sender = self._get_sender()
        command, responses = sender.send_reset()
        self._set_status(f"Sent reset command to {sender.config.port}")
        self._log(f"Sent: {command}")
        if responses:
            self._log("Responses: " + " | ".join(responses))

    def refresh_ports(self) -> None:
        try:
            ports = list_serial_ports()
            if ports and not self.port_var.get().strip():
                self.port_var.set(ports[0])
            actuator_combo = getattr(self, "actuator_port_combo", None)
            if actuator_combo is not None:
                actuator_combo.configure(values=ports)
            if ports:
                self._log("Available ports: " + ", ".join(ports))
            else:
                self._log("No serial ports detected automatically. You can still type one manually.")
        except Exception as exc:
            self._show_error(exc)

    def _schedule_camera_update(self) -> None:
        self._camera_after_id = self.root.after(30, self._update_camera_frame)

    def _update_camera_frame(self) -> None:
        if self._camera is None:
            return

        frame = read_camera_frame(self._camera)
        if frame is not None:
            self._camera_failed_reads = 0
            self._latest_frame = frame
            if self._captured_photo_frame is None:
                self._maybe_start_live_letter_scan(frame)
                self._maybe_start_live_word_scan(frame)
                self._show_frame(frame)
        else:
            self._camera_failed_reads += 1
            if self._camera_failed_reads >= CAMERA_READ_FAILURE_LIMIT:
                self._handle_camera_frame_loss()
                return
        self._schedule_camera_update()

    def _handle_camera_frame_loss(self) -> None:
        message = (
            "Camera stopped because no video frames were available. "
            "Try another camera number or close other apps using the webcam."
        )
        self.stop_camera(clear_preview=True)
        self.preview_label.configure(image="", text="Camera frame unavailable")
        self.preview_image = None
        self._preview_display_size = (0, 0)
        self._preview_source_size = (0, 0)
        self._set_status(message)
        self._log(message)

    def _maybe_start_live_letter_scan(self, frame) -> None:  # type: ignore[no-untyped-def]
        if not self.live_letter_scan_var.get() or self._live_letter_scan_running:
            return

        now = time.monotonic()
        if now - self._last_live_letter_scan_at < LIVE_LETTER_SCAN_INTERVAL_SECONDS:
            return
        self._last_live_letter_scan_at = now

        try:
            confidence_threshold = self._ocr_confidence_threshold()
        except Exception as exc:
            self._handle_camera_letter_scan_error(exc)
            return

        self._live_letter_scan_running = True
        scan_token = self._next_camera_letter_scan_token()
        thread = threading.Thread(
            target=self._camera_letter_scan_worker,
            args=(frame.copy(), confidence_threshold, False, scan_token),
            daemon=True,
        )
        thread.start()

    def _maybe_start_live_word_scan(self, frame) -> None:  # type: ignore[no-untyped-def]
        if not self.live_word_scan_var.get() or self._camera_word_scan_running:
            return

        now = time.monotonic()
        if now - self._last_live_word_scan_at < LIVE_WORD_SCAN_INTERVAL_SECONDS:
            return
        self._last_live_word_scan_at = now

        try:
            confidence_threshold = self._ocr_confidence_threshold()
        except Exception as exc:
            self._handle_camera_word_scan_error(exc)
            return

        self._camera_word_scan_running = True
        scan_token = self._next_camera_word_scan_token()
        thread = threading.Thread(
            target=self._camera_word_scan_worker,
            args=(frame.copy(), confidence_threshold, False, scan_token),
            daemon=True,
        )
        thread.start()

    def _current_camera_ocr_frame(self):  # type: ignore[no-untyped-def]
        displayed_frame = getattr(self, "_last_displayed_camera_frame_for_ocr", None)
        if displayed_frame is not None:
            return displayed_frame
        if self._latest_frame is not None:
            return self._latest_frame
        return self._captured_photo_frame

    def _current_camera_ocr_grid(self):  # type: ignore[no-untyped-def]
        if self._last_camera_word_scan is not None and self._last_camera_word_scan.grid is not None:
            return self._last_camera_word_scan.grid
        if self._last_camera_letter_scan is not None:
            return self._last_camera_letter_scan.grid
        return None

    def _preview_click_to_frame_point(self, click_x: int, click_y: int) -> tuple[float, float] | None:
        display_width, display_height = self._preview_display_size
        source_width, source_height = self._preview_source_size
        if display_width <= 0 or display_height <= 0 or source_width <= 0 or source_height <= 0:
            return None

        label_width = max(1, self.preview_label.winfo_width())
        label_height = max(1, self.preview_label.winfo_height())
        image_left = max(0.0, (label_width - display_width) / 2.0)
        image_top = max(0.0, (label_height - display_height) / 2.0)
        image_x = float(click_x) - image_left
        image_y = float(click_y) - image_top
        if image_x < 0 or image_y < 0 or image_x > display_width or image_y > display_height:
            return None

        return (
            image_x * source_width / display_width,
            image_y * source_height / display_height,
        )

    def _refresh_camera_preview(self) -> None:
        frame = self._current_camera_ocr_frame()
        if frame is not None:
            self._show_frame(frame)

    def _invalidate_camera_scans(self) -> None:
        self._camera_letter_scan_token += 1
        self._camera_word_scan_token += 1
        self._live_letter_scan_running = False
        self._camera_word_scan_running = False
        self._pending_actuator_challenge_after_word_scan = False

    def _next_camera_letter_scan_token(self) -> int:
        self._camera_letter_scan_token += 1
        return self._camera_letter_scan_token

    def _next_camera_word_scan_token(self) -> int:
        self._camera_word_scan_token += 1
        return self._camera_word_scan_token

    def _show_frame(self, frame) -> None:  # type: ignore[no-untyped-def]
        from PIL import Image, ImageTk

        cv2 = _require_cv2()
        captured_letters = self._last_camera_letter_scan.letters if self._last_camera_letter_scan else []
        detected_words = self._last_camera_word_scan.words if self._last_camera_word_scan else []
        detected_tiles = self._last_camera_word_scan.tiles if self._last_camera_word_scan else []
        ocr_grid = self._current_camera_ocr_grid()
        self._last_displayed_camera_frame_for_ocr = frame.copy()
        display = draw_camera_ocr_overlay(
            frame.copy(),
            captured_letters,
            detected_words,
            detected_tiles,
            ocr_grid=ocr_grid,
        )
        display = self._draw_board_pending_corner_overlay(display)
        display = self._draw_tile_rack_grid_overlay(display)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._preview_source_size = (int(display.shape[1]), int(display.shape[0]))
        width = self.preview_label.winfo_width()
        height = self.preview_label.winfo_height()
        image.thumbnail((max(width, 700), max(height, 520)))
        self._preview_display_size = image.size
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _calibration_from_form(self) -> PlotterCalibration:
        calibration = PlotterCalibration.load(self.calibration_path.get())
        calibration.board_size = BOARD_SIZE
        calibration.cell_size_mm = float(self.cell_size_var.get())
        calibration.camera_index = int(self.camera_index_var.get())
        calibration.offset_x_mm = float(self.offset_x_var.get())
        calibration.offset_y_mm = float(self.offset_y_var.get())
        calibration.x_steps_per_mm = float(self.x_steps_per_mm_var.get())
        calibration.y_steps_per_mm = float(self.y_steps_per_mm_var.get())
        calibration.cart_x_mm = float(self.cart_x_var.get())
        calibration.cart_y_mm = float(self.cart_y_var.get())
        calibration.ocr_confidence_threshold = float(self.ocr_confidence_threshold_var.get())
        calibration.ocr_cell_size_px = int(self.ocr_cell_size_px_var.get())
        calibration.actuator_port = self.actuator_port_var.get().strip()
        calibration.actuator_baud = int(self.actuator_baud_var.get())
        calibration.actuator_timeout = float(self.actuator_timeout_var.get())
        calibration.actuator_countdown_seconds = int(float(self.actuator_countdown_seconds_var.get()))
        if calibration.actuator_countdown_seconds <= 0:
            raise ValueError("Actuator countdown seconds must be greater than 0.")
        calibration.premium_layout = self._premium_layout_from_form()
        return calibration

    def _load_premium_layout_into_form(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                self._premium_vars[row][col].set(premium_short_label(self._calibration.premium_layout[row][col]))

    def _premium_layout_from_form(self) -> list[list[str]]:
        self._normalize_premium_entries()
        return [
            [premium_from_short_label(self._premium_vars[row][col].get()) for col in range(BOARD_SIZE)]
            for row in range(BOARD_SIZE)
        ]

    def _board_letters_from_form(self) -> list[list[str]]:
        return [
            [normalize_letter(self._letter_vars[row][col].get()) for col in range(BOARD_SIZE)]
            for row in range(BOARD_SIZE)
        ]

    def _blank_squares_from_form(self) -> set[str]:
        raw = self.blank_squares_var.get().replace(",", " ").split()
        blanks: set[str] = set()
        for label in raw:
            blanks.add(parse_square_label(label).label)
        return blanks

    def _normalize_board_entries(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                self._letter_vars[row][col].set(normalize_letter(self._letter_vars[row][col].get()))

    def _normalize_premium_entries(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                code = premium_from_short_label(self._premium_vars[row][col].get())
                self._premium_vars[row][col].set(premium_short_label(code))

    def _format_score_result(self, score: ScoreResult) -> str:
        if not score.words:
            return "Score: 0 (no words found)"

        lines = [f"Score: {score.total_score}"]
        for word in score.words:
            direction = "H" if word.direction == "horizontal" else "V"
            squares = "-".join(word.squares)
            multiplier = f" x{word.word_multiplier}" if word.word_multiplier != 1 else ""
            lines.append(f"{word.word} {direction} {squares}: {word.base_score}{multiplier} = {word.score}")
        return "\n".join(lines)

    def _set_camera_words_text(self, text: str) -> None:
        if self.camera_words_text is None:
            return
        self.camera_words_text.configure(state="normal")
        self.camera_words_text.delete("1.0", "end")
        self.camera_words_text.insert("1.0", text)
        self.camera_words_text.configure(state="disabled")

    def _ocr_confidence_threshold(self) -> float:
        threshold = float(self.ocr_confidence_threshold_var.get())
        if threshold < 0 or threshold > 100:
            raise ValueError("OCR confidence threshold must be from 0 to 100.")
        return threshold

    def _optional_float(self, raw: str) -> float | None:
        value = raw.strip()
        return float(value) if value else None

    def _actuator_config(self) -> SerialConfig:
        config = SerialConfig(
            port=self.actuator_port_var.get().strip(),
            baud=int(self.actuator_baud_var.get()),
            timeout=float(self.actuator_timeout_var.get()),
            startup_g90=False,
        )
        if not config.port:
            raise ValueError("Enter the Board Actuator Arduino COM port, for example COM4.")
        return config

    def _get_actuator_sender(self) -> BoardActuatorSender:
        config = self._actuator_config()
        sender_key = (config.port, config.baud, config.timeout)
        if self._actuator_sender is not None and self._actuator_sender_key == sender_key:
            return self._actuator_sender

        if self._actuator_sender is not None:
            self._actuator_sender.close()

        self._actuator_sender = BoardActuatorSender(config)
        self._actuator_sender.open()
        self._actuator_sender_key = sender_key
        self._log(f"Connected to Board Actuator Arduino on {config.port} at {config.baud} baud.")
        return self._actuator_sender

    def _send_actuator_command(self, command: str) -> list[str]:
        command = command.strip()
        if not command:
            return []
        sender = self._get_actuator_sender()
        responses = sender.send_command(command)
        self._set_status(f"Sent actuator command: {command}")
        self._log(f"Sent actuator command: {command}")
        if responses:
            self._log("Actuator responses: " + " | ".join(responses))
        for response in responses:
            if response.lower().startswith("err"):
                raise RuntimeError(f"Board Actuator Arduino error: {response}")
        return responses

    def test_actuator_connection(self) -> None:
        try:
            self._send_actuator_command("PING")
            self._set_status("Board Actuator Arduino responded.")
        except Exception as exc:
            self._show_error(exc)

    def request_actuator_status(self) -> None:
        self._send_actuator_button_command("STATUS")

    def actuator_board_up(self) -> None:
        self._send_actuator_button_command("BOARD_UP")

    def actuator_board_down(self) -> None:
        self._send_actuator_button_command("BOARD_DOWN")

    def start_actuator_countdown(self) -> None:
        try:
            seconds = int(float(self.actuator_countdown_seconds_var.get()))
            if seconds <= 0:
                raise ValueError("Actuator countdown seconds must be greater than 0.")
            self._send_actuator_command(f"COUNTDOWN {seconds}")
        except Exception as exc:
            self._show_error(exc)

    def stop_actuator_countdown(self) -> None:
        self._send_actuator_button_command("COUNTDOWN_STOP")

    def start_actuator_challenge(self) -> None:
        try:
            if getattr(self, "_camera_word_scan_running", False):
                self._pending_actuator_challenge_after_word_scan = True
                self._set_status("Waiting for the current camera scan, then the challenge LEDs will update.")
                return

            try:
                frame, quality = self._capture_best_photo_for_ocr("challenge")
                source = f"best camera frame (sharpness {quality.sharpness:.0f})"
            except RuntimeError as exc:
                if str(exc) != "Start the camera first.":
                    raise
                self._send_actuator_challenge_from_current_grid()
                return

            confidence_threshold = self._ocr_confidence_threshold()
            self._camera_word_scan_running = True
            self._pending_actuator_challenge_after_word_scan = True
            scan_token = self._next_camera_word_scan_token()
            self._set_status(f"Reading the board from the {source} before starting challenge...")
            self._set_camera_words_text("Finding challenge words...")
            self._log(f"Challenge camera scan started from {source}.")
            thread = threading.Thread(
                target=self._camera_word_scan_worker,
                args=(frame.copy(), confidence_threshold, False, scan_token),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            self._pending_actuator_challenge_after_word_scan = False
            self._show_error(exc)

    def cancel_actuator_challenge(self) -> None:
        self._send_actuator_button_command("CHALLENGE_CANCEL")

    def previous_actuator_word(self) -> None:
        self._send_actuator_button_command("WORD_PREV")

    def next_actuator_word(self) -> None:
        self._send_actuator_button_command("WORD_NEXT")

    def send_actuator_word(self) -> None:
        try:
            word = self.actuator_word_var.get().strip().upper()
            if not word:
                raise ValueError("Enter a word before sending it to the actuator Arduino.")
            self._send_actuator_command(f"WORD_SET {word}")
        except Exception as exc:
            self._show_error(exc)

    def send_camera_words_to_actuator(self) -> None:
        try:
            words = self._camera_challenge_words()
            if not words:
                raise ValueError("No camera words are available yet. Use Find Words first.")
            self._send_camera_words_to_actuator(words)
        except Exception as exc:
            self._show_error(exc)

    def send_letter_leds_to_actuator(self) -> None:
        try:
            self._send_led_cells_to_actuator(self._current_led_cell_labels(), announce=True)
        except Exception as exc:
            self._show_error(exc)

    def test_actuator_leds(self) -> None:
        self._send_actuator_button_command("LED_TEST")

    def reveal_actuator_word(self) -> None:
        try:
            self._send_actuator_command("WORD_CHOOSE")
        except Exception as exc:
            self._show_error(exc)

    def clear_actuator_leds(self) -> None:
        self._send_actuator_button_command("LED_CLEAR")

    def actuator_display_on(self) -> None:
        self._send_actuator_button_command("DISPLAY_ON")

    def actuator_display_off(self) -> None:
        self._send_actuator_button_command("DISPLAY_OFF")

    def _camera_challenge_words(self) -> list[str]:
        words: list[str] = []
        seen: set[str] = set()

        def add(value: object) -> None:
            word = normalize_word(value)
            if len(word) < 2 or word in seen:
                return
            seen.add(word)
            words.append(word)

        scan = self._last_camera_word_scan
        if scan is not None:
            for detected in scan.words:
                add(detected.word)
            if scan.grid is not None:
                for detected in matched_matrix_words(scan.grid.board_letters()):
                    add(detected.word)

        return words[:10]

    def _send_camera_words_to_actuator(self, words: list[str]) -> None:
        command = "WORD_LIST " + ",".join(words)
        self._send_actuator_command(command)
        self._set_status(f"Sent {len(words)} camera word(s) to the board actuator challenge list.")

    def _try_send_camera_words_to_actuator(self) -> None:
        if not self.actuator_port_var.get().strip():
            return
        words = self._camera_challenge_words()
        try:
            if words:
                self._send_camera_words_to_actuator(words)
            else:
                self._send_actuator_command("WORD_CLEAR")
                self._log("Cleared board actuator challenge words because the camera found none.")
        except Exception as exc:
            self._log(f"Camera words were not sent to the board actuator: {exc}")

    def _finish_pending_actuator_challenge(self) -> None:
        self._pending_actuator_challenge_after_word_scan = False
        try:
            self._send_actuator_challenge_from_current_grid()
        except Exception as exc:
            self._show_error(exc)

    def _send_actuator_challenge_from_current_grid(self) -> None:
        labels = self._current_led_cell_labels()
        if not labels:
            raise ValueError(
                "No board letters were found for the challenge LEDs. Start the camera or run Find Words first."
            )

        words = self._camera_challenge_words()
        if words:
            self._send_camera_words_to_actuator(words)
        self._send_actuator_command("CHALLENGE_START")
        self._send_led_cells_to_actuator(labels, announce=False)
        self._set_status(f"Challenge started with {len(labels)} red LED cell(s).")

    def _current_led_cell_labels(self) -> list[str]:
        labels: list[str] = []
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                letter = normalize_letter(self._letter_vars[row][col].get())
                if letter:
                    labels.append(f"{chr(ord('A') + col)}{row + 1}")
        return labels

    def _send_led_cells_to_actuator(self, labels: list[str], announce: bool = True) -> None:
        if not labels:
            self._send_actuator_command("LED_CLEAR")
            if announce:
                self._set_status("Sent letter lights to the board actuator (0 lit cell(s)).")
            return

        command = "LED_CELLS " + ",".join(labels)
        if len(command) >= 180:
            raise ValueError("Too many detected letter cells to send in one Arduino command.")
        self._send_actuator_command(command)
        if announce:
            lit_count = len(labels)
            self._set_status(f"Sent letter lights to the board actuator ({lit_count} lit cell(s)).")

    def _try_send_letter_leds_to_actuator(self, announce: bool = False) -> None:
        if not self.actuator_port_var.get().strip():
            return
        try:
            self._send_led_cells_to_actuator(self._current_led_cell_labels(), announce=announce)
        except Exception as exc:
            self._log(f"Letter lights were not sent to the board actuator: {exc}")

    def _send_actuator_button_command(self, command: str) -> None:
        try:
            self._send_actuator_command(command)
        except Exception as exc:
            self._show_error(exc)

    def _send_startup_board_up(self) -> None:
        if not self.actuator_port_var.get().strip():
            self._log("Startup board up skipped: no Board Actuator Arduino COM port saved.")
            return
        try:
            self._send_actuator_command("BOARD_UP")
            self._log("Startup board up command sent.")
        except Exception as exc:
            self._log(f"Startup board up command skipped: {exc}")

    def _send_shutdown_board_down(self) -> None:
        if not self.actuator_port_var.get().strip():
            return
        try:
            self._send_actuator_command("BOARD_DOWN")
            self._log("Shutdown board down command sent.")
        except Exception as exc:
            self._log(f"Shutdown board down command skipped: {exc}")

    def _serial_config(self) -> SerialConfig:
        config = SerialConfig(
            port=self.port_var.get().strip(),
            baud=int(self.baud_var.get()),
            timeout=float(self.timeout_var.get()),
            startup_g90=self.startup_g90_var.get(),
        )
        if not config.port:
            raise ValueError("Enter a COM port such as COM3.")
        return config

    def _get_sender(self) -> GCodeSender:
        config = self._serial_config()
        sender_key = (config.port, config.baud, config.timeout, config.startup_g90)
        if self._sender is not None and self._sender_key == sender_key:
            return self._sender

        if self._sender is not None:
            self._sender.close()

        self._sender = GCodeSender(config)
        self._sender.open()
        self._sender_key = sender_key
        self._log(f"Connected to {config.port} at {config.baud} baud.")
        return self._sender

    def _on_close(self) -> None:
        self.stop_camera(clear_preview=False)
        try:
            self._send_shutdown_board_down()
        finally:
            if self._actuator_sender is not None:
                self._actuator_sender.close()
                self._actuator_sender = None
                self._actuator_sender_key = None
            if self._sender is not None:
                self._sender.close()
            self.root.destroy()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _add_z_axis_controls_to_move_section(self) -> None:
        if self._z_axis_controls_frame is not None and self._z_axis_controls_frame.winfo_exists():
            return

        parent = self._find_move_plotter_section() or self.root
        controls = ttk.LabelFrame(parent, text="Z Movement / Relay Magnet", padding=10)
        self._z_axis_controls_frame = controls
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Down height").grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Spinbox(
            controls,
            from_=0,
            to=180,
            increment=1,
            textvariable=self.z_height_angle,
            width=6,
        ).grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="Set Height", command=self._send_z_height_command).grid(
            row=0, column=2, sticky="ew", padx=(8, 0)
        )

        ttk.Scale(
            controls,
            from_=0,
            to=180,
            variable=self.z_height_angle,
            orient="horizontal",
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Button(controls, text="Z Down", command=lambda: self._send_auxiliary_command(Z_DOWN_COMMAND)).grid(
            row=2, column=0, sticky="ew", padx=(0, 6), pady=(10, 0)
        )
        ttk.Button(controls, text="Z Up", command=lambda: self._send_auxiliary_command(Z_UP_COMMAND)).grid(
            row=2, column=1, sticky="ew", padx=(0, 6), pady=(10, 0)
        )
        ttk.Button(
            controls,
            text="Magnet ON",
            command=lambda: self._send_auxiliary_command("M1"),
        ).grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=(8, 0))
        ttk.Button(
            controls,
            text="Magnet OFF",
            command=lambda: self._send_auxiliary_command("M0"),
        ).grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=(8, 0))
        ttk.Label(controls, text="A2 controls the relay input").grid(
            row=3, column=2, sticky="w", pady=(8, 0)
        )

        self._place_z_axis_controls(parent, controls)

    def _find_move_plotter_section(self):
        candidates = []
        for widget in self._walk_widgets(self.root):
            text = self._widget_text(widget).lower()
            if "move plotter" in text:
                return widget if self._looks_like_container(widget) else self._nearest_container(widget)
            if "plotter" in text and "move" in text:
                section = widget if self._looks_like_container(widget) else self._nearest_container(widget)
                if section is not None:
                    candidates.append((0, section))
            elif "move" in text:
                section = widget if self._looks_like_container(widget) else self._nearest_container(widget)
                if section is not None:
                    candidates.append((1, section))
            elif "send move" in text:
                section = self._nearest_container(widget)
                if section is not None:
                    candidates.append((0, section))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        return None

    def _walk_widgets(self, widget):
        yield widget
        for child in widget.winfo_children():
            yield from self._walk_widgets(child)

    def _widget_text(self, widget) -> str:
        try:
            return str(widget.cget("text"))
        except Exception:
            return ""

    def _looks_like_container(self, widget) -> bool:
        class_name = widget.winfo_class().lower()
        return "frame" in class_name or "notebook" in class_name or "pane" in class_name

    def _nearest_container(self, widget):
        parent = getattr(widget, "master", None)
        while parent is not None and parent is not self.root:
            text = self._widget_text(parent).lower()
            if "move" in text or "plotter" in text:
                return parent
            parent = getattr(parent, "master", None)
        return getattr(widget, "master", None)

    def _place_z_axis_controls(self, parent, controls) -> None:
        managers = [child.winfo_manager() for child in parent.winfo_children() if child is not controls]
        if "grid" in managers:
            rows = []
            for child in parent.winfo_children():
                if child is controls or child.winfo_manager() != "grid":
                    continue
                try:
                    rows.append(int(child.grid_info().get("row", 0)))
                except Exception:
                    pass
            controls.grid(row=(max(rows) + 1 if rows else 0), column=0, columnspan=3, sticky="ew", pady=(10, 0))
            try:
                parent.columnconfigure(0, weight=1)
            except Exception:
                pass
        elif "pack" in managers:
            controls.pack(fill="x", pady=(10, 0))
        else:
            controls.grid(row=0, column=0, sticky="ew", pady=(10, 0))

    def _send_z_height_command(self) -> None:
        angle = int(float(self.z_height_angle.get()))
        angle = max(0, min(180, angle))
        self.z_height_angle.set(angle)
        self._send_auxiliary_command(f"ZH{angle}")

    def _send_auxiliary_command(self, command: str) -> None:
        try:
            responses = self._write_auxiliary_serial_line(command)
            log = getattr(self, "_log", None)
            if callable(log):
                log(f"Sent auxiliary command: {command}")
                if responses:
                    log("Responses: " + " | ".join(responses))
            set_status = getattr(self, "_set_status", None)
            if callable(set_status):
                set_status(f"Sent {command}")
        except Exception as exc:
            self._show_error(exc)

    def _write_auxiliary_serial_line(self, command: str) -> list[str]:
        sender = self._get_sender()
        return sender.send_command(command, startup_g90=False)

    def _show_error(self, exc: Exception) -> None:
        message = self._format_user_error(exc)
        self._set_status(message)
        self._log("Error: " + message)
        messagebox.showerror("Scrabble Join", message)

    def _format_user_error(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if (
            "cannot configure port" in lowered
            or "permissionerror(13" in lowered
            or "a device attached to the system is not functioning" in lowered
        ):
            return (
                message
                + "\n\n"
                + "This means Windows could not open the selected Arduino COM port. "
                + "Close Arduino IDE Serial Monitor/Plotter and any other program using the board, "
                + "unplug and reconnect the USB cable, press Refresh Ports, then select the Arduino COM port again. "
                + "Use 115200 baud for the updated Arduino sketch. If the same COM port still fails, check Device Manager "
                + "and reinstall the CH340/Arduino USB driver or try another USB cable/port."
            )
        return message


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for live camera preview. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2


def raise_user_error(message: str) -> None:
    messagebox.showwarning("Scrabble Join", message)


def launch_gui() -> None:
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    app = ScrabblePlotterApp(root)
    app._log("Ready.")
    root.mainloop()
