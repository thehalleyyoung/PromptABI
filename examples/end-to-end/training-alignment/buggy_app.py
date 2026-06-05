"""Buggy training transform: emits a target role absent from serving."""

from __future__ import annotations


def target_roles() -> tuple[str, ...]:
    return ("assistant", "critic")

