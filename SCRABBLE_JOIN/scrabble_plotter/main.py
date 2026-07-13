from __future__ import annotations

import argparse
import json
from pathlib import Path

from .board import parse_square_label
from .calibration import PlotterCalibration
from .gui import launch_gui
from .image_calibration import collect_board_corners
from .scanner import scan_image_file
from .scoring import score_board
from .serial_sender import GCodeSender, SerialConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map 12x12 board squares to XY plotter G-code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("calibrate-image", help="Calibrate the board image.")
    image_parser.add_argument("--image", required=True, help="Path to the board image.")
    image_parser.add_argument(
        "--calibration",
        required=True,
        help="Path to the JSON calibration file to create or update.",
    )

    scan_parser = subparsers.add_parser("scan-image", help="Scan and score a saved board image.")
    scan_parser.add_argument("--image", required=True, help="Path to the board image.")
    scan_parser.add_argument("--calibration", required=True, help="Path to the JSON calibration file.")

    offset_parser = subparsers.add_parser("set-offset", help="Save the board top-left plotter offset.")
    offset_parser.add_argument("--calibration", required=True, help="Path to the JSON calibration file.")
    offset_parser.add_argument("--x", type=float, required=True, help="Top-left board X offset in mm.")
    offset_parser.add_argument("--y", type=float, required=True, help="Top-left board Y offset in mm.")

    move_parser = subparsers.add_parser("move", help="Move the plotter to one square.")
    _add_motion_args(move_parser)
    move_parser.add_argument("--square", required=True, help="Target square, for example H8 or L12.")
    move_parser.add_argument("--dry-run", action="store_true", help="Print the G-code without opening the serial port.")

    interactive_parser = subparsers.add_parser("interactive", help="Interactive square entry loop.")
    _add_motion_args(interactive_parser)

    subparsers.add_parser("gui", help="Open the desktop GUI.")

    return parser


def _add_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration", required=True, help="Path to the JSON calibration file.")
    parser.add_argument("--port", required=True, help="Serial port such as COM3.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--feed-rate", type=float, help="Optional feed rate for the move command.")
    parser.add_argument(
        "--command",
        dest="gcode_command",
        default="G0",
        help="G-code motion command, usually G0 or G1.",
    )
    parser.add_argument("--startup-g90", action="store_true", help="Send G90 before movement commands.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Serial read timeout in seconds.")


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate-image":
        return run_calibrate_image(args.image, args.calibration)
    if args.command == "scan-image":
        return run_scan_image(args.image, args.calibration)
    if args.command == "set-offset":
        return run_set_offset(args)
    if args.command == "move":
        return run_move(args)
    if args.command == "interactive":
        return run_interactive(args)
    if args.command == "gui":
        launch_gui()
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2


def run_calibrate_image(image_path: str, calibration_path: str) -> int:
    calibration = PlotterCalibration.load(calibration_path)
    corners = collect_board_corners(image_path)
    calibration.set_image_corners(image_path, corners, calibration_path)
    calibration.save(calibration_path)
    print(f"Saved image calibration to {Path(calibration_path).resolve()}")
    return 0


def run_scan_image(image_path: str, calibration_path: str) -> int:
    calibration = PlotterCalibration.load(calibration_path)
    scan = scan_image_file(image_path, calibration)
    score = score_board(
        scan.board_letters(),
        premium_layout=calibration.premium_layout,
        blank_squares=scan.blank_squares(),
    )
    print(json.dumps({"scan": scan.to_dict(), "score": score.to_dict()}, indent=2))
    return 0


def run_set_offset(args: argparse.Namespace) -> int:
    calibration = PlotterCalibration.load(args.calibration)
    calibration.set_plotter_offset(args.x, args.y)
    calibration.save(args.calibration)
    print(f"Saved board offset to {Path(args.calibration).resolve()}")
    return 0


def run_move(args: argparse.Namespace) -> int:
    calibration = PlotterCalibration.load(args.calibration)
    calibration.validate_ready_for_move()
    square = parse_square_label(args.square)
    target_x, target_y = calibration.square_center_in_machine(square)

    if args.dry_run:
        from .serial_sender import format_move_command

        gcode = format_move_command(target_x, target_y, args.feed_rate, args.gcode_command)
        print(f"{square.label} -> X={target_x:.3f}, Y={target_y:.3f}")
        print(gcode)
        return 0

    sender = GCodeSender(
        SerialConfig(
            port=args.port,
            baud=args.baud,
            timeout=args.timeout,
            startup_g90=args.startup_g90,
        )
    )
    gcode, responses = sender.send_move(target_x, target_y, args.feed_rate, args.gcode_command)
    print(f"Sent: {gcode}")
    print(f"Target square {square.label} -> X={target_x:.3f}, Y={target_y:.3f}")
    if responses:
        print("Arduino responses:")
        for line in responses:
            print(f"  {line}")
    else:
        print("No Arduino response received before timeout.")
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    calibration = PlotterCalibration.load(args.calibration)
    calibration.validate_ready_for_move()
    sender = GCodeSender(
        SerialConfig(
            port=args.port,
            baud=args.baud,
            timeout=args.timeout,
            startup_g90=args.startup_g90,
        )
    )
    print("Enter a square label like A1 or L12, or type quit to stop.")
    while True:
        raw = _prompt("Square> ").strip()
        if raw.lower() in {"quit", "exit", "q"}:
            return 0
        try:
            square = parse_square_label(raw)
            x, y = calibration.square_center_in_machine(square)
            gcode, responses = sender.send_move(x, y, args.feed_rate, args.gcode_command)
            print(f"Sent: {gcode}")
            if responses:
                print("Responses: " + " | ".join(responses))
        except Exception as exc:
            print(f"Error: {exc}")


def _prompt(message: str) -> str:
    return input(message).strip()
