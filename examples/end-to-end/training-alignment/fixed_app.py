"""Fixed training transform: labels supervised spans as assistant outputs."""

from __future__ import annotations


def target_roles() -> tuple[str, ...]:
    return ("assistant",)

