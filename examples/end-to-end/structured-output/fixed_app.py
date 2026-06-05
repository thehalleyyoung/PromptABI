"""Fixed structured-output parser: validates required fields after JSON parse."""

from __future__ import annotations

import json


def parse_answer(text: str) -> dict[str, object]:
    payload = json.loads(text)
    if payload != {"answer": "ok", "confidence": "high"}:
        raise ValueError("answer must match the constrained schema")
    return payload

