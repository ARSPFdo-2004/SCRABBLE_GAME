from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .board import BOARD_SIZE, CELL_SIZE_MM, parse_square_label
from .calibration import PlotterCalibration
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
    format_camera_words_numbered,
    scan_board_image,
    scan_camera_letters,
    scan_camera_words,
)
from .scoring import (
    ScoreResult,
    normalize_letter,
    premium_from_short_label,
    premium_short_label,
    score_board,
)
from .serial_sender import (
    GCodeSender,
    SerialConfig,
    format_move_command,
    list_serial_ports,
)


LIVE_LETTER_SCAN_INTERVAL_SECONDS = 4.0
LIVE_WORD_SCAN_INTERVAL_SECONDS = 4.0


class ScrabblePlotterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Scrabble Plotter Sender")
        self.root.geometry("1180x780")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.calibration_path = tk.StringVar(value=str(Path("scrabble_plotter_calibration.json").resolve()))
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
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.feed_rate_var = tk.StringVar(value="1500")
        self.command_var = tk.StringVar(value="G0")
        self.timeout_var = tk.StringVar(value="2.0")
        self.startup_g90_var = tk.BooleanVar(value=True)
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
        self.camera_words_text: tk.Text | None = None

        self.status_var = tk.StringVar(
            value="Start the camera to find visible words."
        )
        self.preview_image = None
        self._camera = None
        self._camera_after_id: str | None = None
        self._latest_frame = None
        self._captured_photo_frame = None
        self._camera_letter_scan_token = 0
        self._camera_word_scan_token = 0
        self._last_live_letter_scan_at = 0.0
        self._live_letter_scan_running = False
        self._last_live_letter_scan_error: str | None = None
        self._last_live_word_scan_at = 0.0
        self._last_live_word_scan_error: str | None = None
        self._sender: GCodeSender | None = None
        self._sender_key: tuple[str, int, float, bool] | None = None

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

        ttk.Checkbutton(move_box, text="Send G90 before move", variable=self.startup_g90_var).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Button(move_box, text="Preview G-code", command=self.preview_move).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )
        ttk.Button(move_box, text="Send Move", command=self.send_move).grid(
            row=8, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(move_box, text="Reset To Start", command=self.reset_to_start).grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Button(move_box, text="Go To Cart", command=self.go_to_cart).grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )

        scan_box = ttk.LabelFrame(controls, text="Camera OCR", padding=10)
        scan_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
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
        ttk.Button(scan_buttons, text="Calculate Score", command=self.calculate_score_from_board).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0)
        )

        ttk.Label(scan_box, text="Captured Letters").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(scan_box, textvariable=self.captured_letters_var).grid(row=2, column=0, sticky="ew")

        ttk.Label(scan_box, text="Detected Words").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.camera_words_text = tk.Text(scan_box, height=5, wrap="word", state="disabled")
        self.camera_words_text.grid(row=4, column=0, sticky="ew")

        board_grid = ttk.Frame(scan_box)
        board_grid.grid(row=5, column=0, sticky="ew", pady=(10, 0))
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
                entry.bind("<FocusOut>", lambda event: self._normalize_board_entries())
                entry.bind("<Return>", lambda event: self.calculate_score_from_board())
                entry_row.append(entry)
            self._letter_entries.append(entry_row)

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
        agent_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
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
        log_box.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(1, weight=1)
        controls.rowconfigure(4, weight=1)

        ttk.Label(log_box, textvariable=self.status_var, wraplength=380).grid(row=0, column=0, sticky="ew")
        self.log_text = tk.Text(log_box, height=14, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

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
                f"Y={self._calibration.cart_y_mm:.3f}"
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
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"Unable to open camera {camera_index}.")

            self._camera = camera
            self._calibration = self._calibration_from_form()
            self._calibration.camera_index = camera_index
            self._calibration.save(self.calibration_path.get())
            self._latest_frame = None
            self._captured_photo_frame = None
            self._invalidate_camera_scans()
            self._last_live_letter_scan_at = 0.0
            self._last_live_letter_scan_error = None
            self._last_camera_letter_scan = None
            self._last_live_word_scan_at = 0.0
            self._last_live_word_scan_error = None
            self._last_camera_word_scan = None
            self.captured_letters_var.set("")
            self._set_camera_words_text("")
            self._set_status(f"Camera {camera_index} started.")
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
        if clear_preview:
            self._latest_frame = None
            self._captured_photo_frame = None
            self._invalidate_camera_scans()
            self.preview_label.configure(image="", text="Camera stopped")
            self.preview_image = None

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
            self._send_square_move(self.square_var.get())
        except Exception as exc:
            self._show_error(exc)

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
        self.capture_letters_from_camera()

    def take_picture_from_camera(self) -> None:
        if self._latest_frame is None:
            raise_user_error("Start the camera first.")
            return

        self._captured_photo_frame = self._latest_frame.copy()
        self._invalidate_camera_scans()
        self._last_camera_letter_scan = None
        self._last_camera_word_scan = None
        self.captured_letters_var.set("")
        self._set_camera_words_text("")
        self._refresh_camera_preview()
        self._set_status("Picture captured. Click Find Words to identify the words in it.")
        self._log("Picture captured for EasyOCR word detection.")

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

    def capture_letters_from_camera(self) -> None:
        frame = self._current_camera_ocr_frame()
        if frame is None:
            raise_user_error("Start the camera first.")
            return

        try:
            confidence_threshold = self._ocr_confidence_threshold()
            source = "captured picture" if self._captured_photo_frame is not None else "camera"
            scan_token = self._next_camera_letter_scan_token()
            self._set_status(f"Capturing letters from the {source}...")
            self._log(f"Camera letter capture started from {source}.")
            thread = threading.Thread(
                target=self._camera_letter_scan_worker,
                args=(frame.copy(), confidence_threshold, True, scan_token),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            self._show_error(exc)

    def identify_words_with_easyocr(self) -> None:
        frame = self._current_camera_ocr_frame()
        if frame is None:
            raise_user_error("Start the camera first.")
            return
        if self._camera_word_scan_running:
            self._set_status("EasyOCR word detection is already running.")
            return

        try:
            confidence_threshold = self._ocr_confidence_threshold()
            self._camera_word_scan_running = True
            source = "captured picture" if self._captured_photo_frame is not None else "camera"
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
            score = score_board(
                self._board_letters_from_form(),
                premium_layout=calibration.premium_layout,
                blank_squares=self._blank_squares_from_form(),
            )
            self._handle_score_result(score)
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
        if captured_text:
            message = f"Captured {len(scan.letters)} letter group(s): {captured_text}"
        else:
            message = "No letters captured from the camera."
        self._set_status(message)
        self._refresh_camera_preview()
        if announce:
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
        formatted_words = format_camera_words_numbered(scan.words)
        self._set_camera_words_text(formatted_words)
        if scan.words:
            self._set_status(
                f"EasyOCR found {len(scan.words)} matching word(s) from {len(scan.text_boxes)} text box(es)."
            )
            if announce:
                self._log("Detected words:\n" + formatted_words)
        else:
            self._set_status(f"EasyOCR found {len(scan.text_boxes)} text box(es), but no matching words.")
            if announce:
                self._log("Detected words: none")
        self._refresh_camera_preview()

    def _handle_camera_word_scan_error(
        self,
        exc: Exception,
        announce: bool = False,
        scan_token: int | None = None,
    ) -> None:
        if scan_token is not None and scan_token != self._camera_word_scan_token:
            return
        self._camera_word_scan_running = False
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

        ok, frame = self._camera.read()
        if ok:
            self._latest_frame = frame
            if self._captured_photo_frame is None:
                self._maybe_start_live_letter_scan(frame)
                self._maybe_start_live_word_scan(frame)
                self._show_frame(frame)
        self._schedule_camera_update()

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
        if self._captured_photo_frame is not None:
            return self._captured_photo_frame
        return self._latest_frame

    def _refresh_camera_preview(self) -> None:
        frame = self._current_camera_ocr_frame()
        if frame is not None:
            self._show_frame(frame)

    def _invalidate_camera_scans(self) -> None:
        self._camera_letter_scan_token += 1
        self._camera_word_scan_token += 1
        self._live_letter_scan_running = False
        self._camera_word_scan_running = False

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
        display = draw_camera_ocr_overlay(frame.copy(), captured_letters, detected_words, detected_tiles)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = self.preview_label.winfo_width()
        height = self.preview_label.winfo_height()
        image.thumbnail((max(width, 700), max(height, 520)))
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
        if self._sender is not None:
            self._sender.close()
        self.root.destroy()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _show_error(self, exc: Exception) -> None:
        self._set_status(str(exc))
        self._log("Error: " + str(exc))
        messagebox.showerror("Scrabble Plotter Sender", str(exc))


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for live camera preview. Install scrabble_plotter/requirements.txt."
        ) from exc
    return cv2


def raise_user_error(message: str) -> None:
    messagebox.showwarning("Scrabble Plotter Sender", message)


def launch_gui() -> None:
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    app = ScrabblePlotterApp(root)
    app._log("Ready.")
    root.mainloop()
