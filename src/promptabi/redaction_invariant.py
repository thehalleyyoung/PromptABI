"""Prove private-field redaction survives packing (step 277).

Redaction applied to raw examples is worthless if a *later* transform -- packing,
templating, truncation -- reintroduces the raw value (e.g. by concatenating an
un-redacted metadata field, or by a buggy pass that reads from the original
record).  This module proves redaction is an *invariant* of the whole transform
pipeline: given the set of secret values that must never appear, it scans the
output of every pass and proves no secret resurfaces, naming the exact pass that
reintroduced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

REDACTION_INVARIANT_VERSION = "promptabi.redaction-invariant.v1"


class RedactionFindingKind(StrEnum):
    SECRET_PRESENT_AT_INPUT = "secret-present-at-input"
    SECRET_REINTRODUCED = "secret-reintroduced"


@dataclass(frozen=True, slots=True)
class PackingStage:
    name: str
    transform: Callable[[str], str]


@dataclass(frozen=True, slots=True)
class RedactionFinding:
    kind: RedactionFindingKind
    stage: str
    secret_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "stage": self.stage,
            "secret_label": self.secret_label,
        }


@dataclass(frozen=True, slots=True)
class RedactionInvariantResult:
    version: str
    preserved: bool
    findings: tuple[RedactionFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "preserved": self.preserved,
            "findings": [f.to_dict() for f in self.findings],
        }


def prove_redaction_invariant(
    redacted_input: str,
    secrets: dict[str, str],
    stages: tuple[PackingStage, ...],
) -> RedactionInvariantResult:
    """Prove no secret value resurfaces through the packing pipeline.

    ``secrets`` maps a human label to the raw value that must never appear.
    """

    findings: list[RedactionFinding] = []

    for label, value in secrets.items():
        if value and value in redacted_input:
            findings.append(
                RedactionFinding(
                    RedactionFindingKind.SECRET_PRESENT_AT_INPUT,
                    "input",
                    label,
                )
            )

    text = redacted_input
    for stage in stages:
        text = stage.transform(text)
        for label, value in secrets.items():
            if value and value in text:
                findings.append(
                    RedactionFinding(
                        RedactionFindingKind.SECRET_REINTRODUCED,
                        stage.name,
                        label,
                    )
                )

    return RedactionInvariantResult(
        version=REDACTION_INVARIANT_VERSION,
        preserved=not findings,
        findings=tuple(findings),
    )


def render_redaction_invariant_text(result: RedactionInvariantResult) -> str:
    lines = [
        f"PromptABI redaction-survives-packing proof ({result.version})",
        f"result: {'PRESERVED' if result.preserved else 'VIOLATED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.stage}]: secret {f.secret_label!r}")
    return "\n".join(lines) + "\n"
