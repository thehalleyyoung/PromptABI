"""Fixed tool-call handler: normalizes provider JSON strings before dispatch."""

from __future__ import annotations

import json


ALLOWED_TOOLS = {"lookup_order", "refund_order"}


def handle_tool_call(call: dict[str, object]) -> str:
    name = str(call["name"])
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"unknown tool: {name}")
    raw_arguments = call["arguments"]
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a JSON object or JSON string object")
    if name == "lookup_order":
        return f"lookup:{arguments['order_id']}"
    return f"refund:{arguments['order_id']}:{arguments['reason']}"

