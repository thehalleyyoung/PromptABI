"""CLI renderers for PromptABI diagnostics."""

from __future__ import annotations

import json

from .session import VerificationResult


def render_text(result: VerificationResult) -> str:
    lines = [
        f"PromptABI verification: {result.config.name}",
        f"checks: {', '.join(result.config.checks) if result.config.checks else '(none)'}",
        f"status: {'PASS' if result.ok else 'FAIL'}",
    ]
    for diagnostic in result.diagnostics:
        lines.append(f"{diagnostic.severity.value.upper()} {diagnostic.rule_id}: {diagnostic.message}")
        if diagnostic.artifact is not None and diagnostic.artifact.path is not None:
            lines.append(f"  artifact: {diagnostic.artifact.name} ({diagnostic.artifact.path})")
        if diagnostic.witness is not None:
            lines.append(f"  witness: {diagnostic.witness.summary}")
    return "\n".join(lines) + "\n"


def render_json(result: VerificationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"

