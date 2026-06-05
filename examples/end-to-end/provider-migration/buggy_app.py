"""Buggy migration shim: indexes OpenAI fields after a provider migration."""

from __future__ import annotations


def first_tool_call(response: dict[str, object]) -> object:
    choices = response["choices"]
    return choices[0]["message"]["tool_calls"][0]  # type: ignore[index]

