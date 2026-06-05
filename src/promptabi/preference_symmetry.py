"""Verify preference-pair role symmetry (step 265).

A preference example for DPO/RLHF is a triple ``(prompt, chosen, rejected)``.  The
learning signal is only valid if ``chosen`` and ``rejected`` share an *identical
prompt context and role structure* and differ **only** in the final assistant
response.  If the rejected branch silently has an extra system turn, a different
prompt, or a forged role, the preference gradient is contaminated.

This module proves role/context symmetry: same prompt messages, same role
sequence up to the final assistant turn, and a genuine difference in the final
response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PREFERENCE_SYMMETRY_VERSION = "promptabi.preference-symmetry.v1"


class SymmetryFindingKind(StrEnum):
    PROMPT_MISMATCH = "prompt-mismatch"
    ROLE_SEQUENCE_MISMATCH = "role-sequence-mismatch"
    FINAL_ROLE_NOT_ASSISTANT = "final-role-not-assistant"
    RESPONSES_IDENTICAL = "responses-identical"


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class PreferencePair:
    chosen: tuple[Turn, ...]
    rejected: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class SymmetryFinding:
    kind: SymmetryFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class SymmetryResult:
    version: str
    symmetric: bool
    findings: tuple[SymmetryFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "symmetric": self.symmetric,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_preference_pair(pair: PreferencePair) -> SymmetryResult:
    findings: list[SymmetryFinding] = []
    chosen, rejected = pair.chosen, pair.rejected

    if not chosen or not rejected:
        findings.append(
            SymmetryFinding(
                SymmetryFindingKind.PROMPT_MISMATCH,
                "chosen and rejected branches must both be non-empty",
            )
        )
        return SymmetryResult(PREFERENCE_SYMMETRY_VERSION, False, tuple(findings))

    if chosen[-1].role != "assistant" or rejected[-1].role != "assistant":
        findings.append(
            SymmetryFinding(
                SymmetryFindingKind.FINAL_ROLE_NOT_ASSISTANT,
                "the final turn of each branch must be the assistant response",
            )
        )

    prompt_chosen = chosen[:-1]
    prompt_rejected = rejected[:-1]
    if prompt_chosen != prompt_rejected:
        findings.append(
            SymmetryFinding(
                SymmetryFindingKind.PROMPT_MISMATCH,
                "prompt context differs between chosen and rejected branches",
            )
        )
        # Role-sequence detail for easier debugging.
        rc = tuple(t.role for t in prompt_chosen)
        rr = tuple(t.role for t in prompt_rejected)
        if rc != rr:
            findings.append(
                SymmetryFinding(
                    SymmetryFindingKind.ROLE_SEQUENCE_MISMATCH,
                    f"prompt role sequence {list(rc)} != {list(rr)}",
                )
            )

    if chosen[-1].content == rejected[-1].content:
        findings.append(
            SymmetryFinding(
                SymmetryFindingKind.RESPONSES_IDENTICAL,
                "chosen and rejected responses are identical; no preference signal",
            )
        )

    return SymmetryResult(
        version=PREFERENCE_SYMMETRY_VERSION,
        symmetric=not findings,
        findings=tuple(findings),
    )


def render_symmetry_text(result: SymmetryResult) -> str:
    lines = [
        f"PromptABI preference-pair symmetry ({result.version})",
        f"result: {'SYMMETRIC' if result.symmetric else 'ASYMMETRIC'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
