from __future__ import annotations

import time
from dataclasses import dataclass


def _require_serial():
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial communication. Install scrabble_plotter/requirements.txt."
        ) from exc
    return serial


def list_serial_ports() -> list[str]:
    serial = _require_serial()
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError:
        return []
    return [port.device for port in list_ports.comports()]


def _format_float(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def format_move_command(x: float, y: float, feed_rate: float | None = None, command: str = "G0") -> str:
    parts = [command.upper(), f"X{_format_float(x)}", f"Y{_format_float(y)}"]
    if feed_rate is not None:
        parts.append(f"F{_format_float(feed_rate)}")
    return " ".join(parts)


def format_reset_command() -> str:
    return "HOMEZERO"


def format_steps_command(x_steps_per_mm: float, y_steps_per_mm: float) -> str:
    return f"STEPS X{_format_float(x_steps_per_mm)} Y{_format_float(y_steps_per_mm)}"


@dataclass
class SerialConfig:
    port: str
    baud: int
    timeout: float = 2.0
    startup_g90: bool = False


class GCodeSender:
    def __init__(self, config: SerialConfig):
        self.config = config
        self._connection = None

    def send_move(
        self,
        x: float,
        y: float,
        feed_rate: float | None = None,
        command: str = "G0",
    ) -> tuple[str, list[str]]:
        gcode = format_move_command(x, y, feed_rate=feed_rate, command=command)
        responses = self.send_commands([gcode])
        return gcode, responses

    def send_reset(self) -> tuple[str, list[str]]:
        command = format_reset_command()
        responses = self.send_commands([command])
        return command, responses

    def send_step_config(self, x_steps_per_mm: float, y_steps_per_mm: float) -> tuple[str, list[str]]:
        command = format_steps_command(x_steps_per_mm, y_steps_per_mm)
        responses = self.send_commands([command])
        return command, responses

    def send_command(self, command: str, *, startup_g90: bool | None = None) -> list[str]:
        return self.send_commands([command], startup_g90=startup_g90)

    def open(self) -> None:
        if self._connection is not None:
            return

        serial = _require_serial()
        connection = serial.Serial(self.config.port, self.config.baud, timeout=self.config.timeout)
        try:
            connection.setDTR(False)
            connection.setRTS(False)
        except Exception:
            pass
        time.sleep(2)
        try:
            connection.reset_input_buffer()
            connection.reset_output_buffer()
        except Exception:
            pass
        self._connection = connection

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    def send_commands(self, commands: list[str], *, startup_g90: bool | None = None) -> list[str]:
        should_close = self._connection is None
        if should_close:
            self.open()

        connection = self._connection
        if connection is None:
            raise RuntimeError("Serial connection could not be opened.")

        try:
            should_send_startup_g90 = self.config.startup_g90 if startup_g90 is None else startup_g90
            if should_send_startup_g90:
                self._write_line(connection, "G90")
                self._read_responses(connection)

            responses: list[str] = []
            for command in commands:
                self._write_line(connection, command)
                responses.extend(self._read_responses(connection))
            return responses
        finally:
            if should_close:
                self.close()

    def _write_line(self, connection, line: str) -> None:  # type: ignore[no-untyped-def]
        connection.write((line.strip() + "\n").encode("utf-8"))
        connection.flush()

    def _read_responses(self, connection) -> list[str]:  # type: ignore[no-untyped-def]
        deadline = time.monotonic() + self.config.timeout
        responses: list[str] = []
        while time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                if responses:
                    break
                continue
            message = raw.decode("utf-8", errors="replace").strip()
            if not message:
                continue
            responses.append(message)
            lowered = message.lower()
            if lowered == "ok" or lowered.startswith("ok ") or lowered.startswith("err"):
                break
        return responses


class BoardActuatorSender(GCodeSender):
    """Serial sender for the second Arduino that controls board actuators."""

    def send_command(self, command: str, *, startup_g90: bool | None = None) -> list[str]:
        return super().send_command(command, startup_g90=False)

    def send_commands(self, commands: list[str], *, startup_g90: bool | None = None) -> list[str]:
        return super().send_commands(commands, startup_g90=False)

    def ping(self) -> list[str]:
        return self.send_command("PING")

    def board_up(self) -> list[str]:
        return self.send_command("BOARD_UP")

    def board_down(self) -> list[str]:
        return self.send_command("BOARD_DOWN")

    def countdown(self, seconds: int | None = None) -> list[str]:
        command = "COUNTDOWN" if seconds is None else f"COUNTDOWN {seconds}"
        return self.send_command(command)

    def set_word(self, word: str) -> list[str]:
        return self.send_command(f"WORD_SET {word.strip().upper()}")

    def set_words(self, words: list[str]) -> list[str]:
        payload = ",".join(word.strip().upper() for word in words if word.strip())
        return self.send_command(f"WORD_LIST {payload}")

    def clear_words(self) -> list[str]:
        return self.send_command("WORD_CLEAR")

    def set_led_cells(self, labels: list[str]) -> list[str]:
        payload = ",".join(label.strip().upper() for label in labels if label.strip())
        return self.send_command(f"LED_CELLS {payload}")
