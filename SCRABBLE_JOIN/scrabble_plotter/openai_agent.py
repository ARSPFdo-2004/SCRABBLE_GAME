from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .ai_player import AiMoveCandidate, AiMoveChoice
from .board import BOARD_SIZE, parse_square_label
from .calibration import PlotterCalibration
from .word_bank import (
    DIRECTION_LABELS,
    HORIZONTAL_LEFT_TO_RIGHT,
    VERTICAL_TOP_TO_BOTTOM,
    filter_matching_words,
    format_words_by_direction,
)


DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
PLOTTER_ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "square": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}
WORD_DETECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "direction": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["word", "direction", "confidence"],
            },
        },
    },
    "required": ["words"],
}
AI_MOVE_CHOICE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "candidate_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


@dataclass
class PlotterAgentAction:
    action: str
    square: str | None = None
    reason: str = ""

    def validate(self) -> "PlotterAgentAction":
        allowed_actions = {"move_square", "go_cart", "reset", "none"}
        if self.action not in allowed_actions:
            raise ValueError(f"OpenAI returned unsupported action: {self.action}")
        if self.action == "move_square":
            if not self.square:
                raise ValueError("OpenAI chose move_square without a square.")
            self.square = parse_square_label(self.square).label
        else:
            self.square = None
        return self

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PlotterAgentAction":
        return cls(
            action=str(payload.get("action", "none")).strip().lower(),
            square=payload.get("square"),
            reason=str(payload.get("reason", "")).strip(),
        ).validate()


@dataclass(frozen=True)
class OpenAIDetectedWord:
    word: str
    direction: str
    confidence: float = 0.0

    @property
    def direction_label(self) -> str:
        return DIRECTION_LABELS.get(self.direction, self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "direction": self.direction,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OpenAIDetectedWord":
        word = _normalize_detected_word(payload.get("word", ""))
        direction = _normalize_word_direction(payload.get("direction", ""))
        confidence = _clamped_confidence(payload.get("confidence", 0.0))
        return cls(word=word, direction=direction, confidence=confidence)


class OpenAIPlotterAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        endpoint: str = DEFAULT_OPENAI_ENDPOINT,
    ):
        if not api_key.strip():
            raise ValueError("Enter an OpenAI API key.")
        if not model.strip():
            raise ValueError("Enter an OpenAI model name.")
        if not endpoint.strip():
            raise ValueError("Enter an OpenAI endpoint.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.endpoint = normalize_openai_endpoint(endpoint)

    def decide(
        self,
        objective: str,
        calibration: PlotterCalibration,
        image_jpeg: bytes | None = None,
        timeout: float = 20.0,
    ) -> PlotterAgentAction:
        prompt = self._build_prompt(objective, calibration)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_jpeg:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_jpeg)},
                }
            )

        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        response_payload = self._post(request_payload, timeout=timeout)
        text = _extract_text(response_payload)
        return PlotterAgentAction.from_payload(_parse_json_object(text))

    def identify_camera_words(
        self,
        image_jpeg: bytes,
        captured_letters: str = "",
        timeout: float = 20.0,
    ) -> list[OpenAIDetectedWord]:
        if not image_jpeg:
            raise ValueError("A camera image is required for OpenAI word detection.")

        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_word_prompt(captured_letters)},
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_jpeg)},
            },
        ]
        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        response_payload = self._post(request_payload, timeout=timeout)
        text = _extract_text(response_payload)
        return parse_detected_words(_parse_json_object(text))

    def choose_ai_move(
        self,
        candidates: list[AiMoveCandidate],
        rack_letters: list[str],
        objective: str = "",
        timeout: float = 20.0,
    ) -> AiMoveChoice:
        if not candidates:
            return AiMoveChoice(action="pass", reason="No legal moves were available.")

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_ai_move_prompt(
                                candidates,
                                rack_letters,
                                objective,
                            )
                        }
                    ],
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        response_payload = self._post(request_payload, timeout=timeout)
        text = _extract_text(response_payload)
        return AiMoveChoice.from_payload(_parse_json_object(text), candidates)

    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            if details:
                raise RuntimeError(f"OpenAI request failed: HTTP {exc.code} {exc.reason}: {details}") from exc
            raise RuntimeError(f"OpenAI request failed: HTTP {exc.code} {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    def _build_prompt(self, objective: str, calibration: PlotterCalibration) -> str:
        return (
            "You control a Scrabble plotter through a safe action API. "
            "Return exactly one JSON object and no extra text.\n"
            "Allowed actions:\n"
            '{"action":"move_square","square":"A1","reason":"short reason"}\n'
            '{"action":"go_cart","reason":"short reason"}\n'
            '{"action":"reset","reason":"short reason"}\n'
            '{"action":"none","reason":"short reason"}\n'
            f"The board is {BOARD_SIZE} by {BOARD_SIZE}. Valid squares are A1 through L12. "
            "A1 is top-left in the camera image. The app will convert squares to machine motion; "
            "do not return raw G-code or motor steps.\n"
            f"Current calibration: offset X {calibration.offset_x_mm:.3f} mm, "
            f"offset Y {calibration.offset_y_mm:.3f} mm, "
            f"cell size {calibration.cell_size_mm:.3f} mm, "
            f"cart X {calibration.cart_x_mm:.3f} mm, cart Y {calibration.cart_y_mm:.3f} mm.\n"
            "Valid pickup targets include tile rack slots (TR1-TR7) and tile cart grid slots (TC1-TC16).\n"
            f"User objective: {objective.strip() or 'Choose the safest next action.'}"
        )

    def _build_word_prompt(self, captured_letters: str) -> str:
        local_ocr = captured_letters.strip() or "No local OCR text was available."
        return (
            "Identify complete words visible in the camera image. "
            "Do not return board coordinates or describe the board area. "
            "Only return words in these two orientations:\n"
            "1. horizontal_left_to_right: read letters from left to right in the same row.\n"
            "2. vertical_top_to_bottom: read letters from top to bottom in the same column.\n"
            "Ignore right-to-left horizontal words, bottom-to-top vertical words, diagonals, partial letters, "
            "and uncertain fragments. Use only uppercase A-Z letters. Prefer words of two or more letters.\n"
            "Confidence must be a number from 0 to 100.\n"
            "Return exactly one JSON object and no extra text, in this shape:\n"
            '{"words":[{"word":"WORD","direction":"horizontal_left_to_right","confidence":85}]}\n'
            'If no words are visible, return {"words":[]}.\n'
            f"Local OCR captured these letters as extra context: {local_ocr}"
        )

    def _build_ai_move_prompt(
        self,
        candidates: list[AiMoveCandidate],
        rack_letters: list[str],
        objective: str,
    ) -> str:
        candidate_payload = [candidate.to_prompt_dict() for candidate in candidates]
        rack_text = "".join(letter or "-" for letter in rack_letters) or "empty"
        return (
            "You are Player 1 in a physical Scrabble-style game. "
            "Choose exactly one legal move from the candidate list. "
            "The app has already validated board bounds, rack slots, anchors, cross-words, and plotter safety. "
            "Valid pickup targets include tile rack slots (TR1-TR7) and tile cart grid slots (TC1-TC16). "
            "Do not invent new words, squares, rack slots, G-code, motor steps, or coordinates.\n"
            "Return exactly one JSON object and no extra text.\n"
            'To play a move: {"action":"play","candidate_id":"C001","reason":"short reason"}\n'
            'To pass: {"action":"pass","reason":"short reason"}\n'
            f"Rack letters by slot TR1-TR7: {rack_text}\n"
            f"User objective: {objective.strip() or 'Choose the best scoring legal move.'}\n"
            "Legal candidates, ranked by local score:\n"
            f"{json.dumps(candidate_payload, separators=(',', ':'))}"
        )


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenAI returned no choices.")
    message = choices[0].get("message", {})
    text = str(message.get("content", "")).strip()
    if not text:
        raise ValueError("OpenAI returned an empty response.")
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response must be a JSON object.")
    return payload


def parse_detected_words(payload: dict[str, Any]) -> list[OpenAIDetectedWord]:
    raw_words = payload.get("words", [])
    if not isinstance(raw_words, list):
        raise ValueError("OpenAI word response must include a words list.")

    words: list[OpenAIDetectedWord] = []
    seen: set[tuple[str, str]] = set()
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        try:
            detected = OpenAIDetectedWord.from_payload(raw_word)
        except ValueError:
            continue
        key = (detected.word, detected.direction)
        if key in seen:
            continue
        seen.add(key)
        words.append(detected)
    return filter_matching_words(words)


def format_detected_words_numbered(words: list[OpenAIDetectedWord]) -> str:
    return format_words_by_direction(words)


def _normalize_detected_word(value: Any) -> str:
    word = "".join(char for char in str(value).strip().upper() if "A" <= char <= "Z")
    if len(word) < 2:
        raise ValueError("OpenAI word must contain at least two letters.")
    return word


def _normalize_word_direction(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        HORIZONTAL_LEFT_TO_RIGHT: HORIZONTAL_LEFT_TO_RIGHT,
        "left_to_right": HORIZONTAL_LEFT_TO_RIGHT,
        "ltr": HORIZONTAL_LEFT_TO_RIGHT,
        VERTICAL_TOP_TO_BOTTOM: VERTICAL_TOP_TO_BOTTOM,
        "top_to_bottom": VERTICAL_TOP_TO_BOTTOM,
        "ttb": VERTICAL_TOP_TO_BOTTOM,
    }
    direction = aliases.get(normalized)
    if direction is None:
        raise ValueError(f"Unsupported OpenAI word direction: {value}")
    return direction


def _clamped_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if 0.0 < confidence <= 1.0:
        confidence *= 100.0
    return min(100.0, max(0.0, confidence))


def normalize_openai_endpoint(endpoint: str) -> str:
    cleaned = endpoint.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned + "/chat/completions"
    return cleaned


def _image_data_url(image_jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
