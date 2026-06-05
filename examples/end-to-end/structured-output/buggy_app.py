"""Buggy structured-output parser: valid JSON is mistaken for valid schema."""

from __future__ import annotations

import json


def parse_answer(text: str) -> dict[str, object]:
    return json.loads(text)

