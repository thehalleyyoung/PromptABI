"""Buggy tool-call handler: parses only one complete JSON-object tool call."""

from __future__ import annotations


ALLOWED_TOOLS = {"lookup_order", "refund_order"}


def handle_tool_call(call: dict[str, object]) -> str:
    name = str(call["name"])
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"unknown tool: {name}")
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        raise TypeError("arguments must already be a JSON object")
    if name == "lookup_order":
        return f"lookup:{arguments['order_id']}"
    return f"refund:{arguments['order_id']}:{arguments['reason']}"

