"""Fixed migration shim: only deploys against a target with the same envelope."""

from __future__ import annotations


def first_tool_call(response: dict[str, object]) -> object:
    choices = response["choices"]
    return choices[0]["message"]["tool_calls"][0]  # type: ignore[index]

