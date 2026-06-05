"""Verify curriculum-stage prompt-ABI drift (step 260).

Curriculum fine-tuning runs a model through several *stages* (e.g. broad
instruction data, then domain data, then preference data).  If the chat template,
special-token map, or role set silently *drifts* between stages, the model is
trained against inconsistent prompt interfaces and the final artifact's behavior
no longer matches any single declared template.  This is a real and hard-to-spot
training bug.

This module captures each stage's prompt interface as a :class:`StageInterface`
and proves the interface is stable across the curriculum, reporting the exact
field that drifted between any two consecutive stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

CURRICULUM_DRIFT_VERSION = "promptabi.curriculum-drift.v1"


class DriftKind(StrEnum):
    TEMPLATE_DRIFT = "template-drift"
    SPECIAL_TOKEN_DRIFT = "special-token-drift"
    ROLE_SET_DRIFT = "role-set-drift"
    BOS_EOS_DRIFT = "bos-eos-drift"


@dataclass(frozen=True, slots=True)
class StageInterface:
    stage: str
    template_digest: str
    special_tokens: frozenset[str]
    roles: frozenset[str]
    bos: str | None
    eos: str | None


@dataclass(frozen=True, slots=True)
class DriftFinding:
    kind: DriftKind
    from_stage: str
    to_stage: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CurriculumDriftResult:
    version: str
    stable: bool
    findings: tuple[DriftFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "stable": self.stable,
            "findings": [f.to_dict() for f in self.findings],
        }


def _pair(a: StageInterface, b: StageInterface) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    if a.template_digest != b.template_digest:
        findings.append(
            DriftFinding(
                DriftKind.TEMPLATE_DRIFT,
                a.stage,
                b.stage,
                f"template digest {a.template_digest} -> {b.template_digest}",
            )
        )
    if a.special_tokens != b.special_tokens:
        added = sorted(b.special_tokens - a.special_tokens)
        removed = sorted(a.special_tokens - b.special_tokens)
        findings.append(
            DriftFinding(
                DriftKind.SPECIAL_TOKEN_DRIFT,
                a.stage,
                b.stage,
                f"added={added} removed={removed}",
            )
        )
    if a.roles != b.roles:
        findings.append(
            DriftFinding(
                DriftKind.ROLE_SET_DRIFT,
                a.stage,
                b.stage,
                f"{sorted(a.roles)} -> {sorted(b.roles)}",
            )
        )
    if (a.bos, a.eos) != (b.bos, b.eos):
        findings.append(
            DriftFinding(
                DriftKind.BOS_EOS_DRIFT,
                a.stage,
                b.stage,
                f"bos/eos ({a.bos},{a.eos}) -> ({b.bos},{b.eos})",
            )
        )
    return findings


def verify_curriculum(stages: tuple[StageInterface, ...]) -> CurriculumDriftResult:
    if len(stages) < 2:
        return CurriculumDriftResult(CURRICULUM_DRIFT_VERSION, True, ())
    findings: list[DriftFinding] = []
    for a, b in zip(stages, stages[1:]):
        findings.extend(_pair(a, b))
    return CurriculumDriftResult(
        version=CURRICULUM_DRIFT_VERSION,
        stable=not findings,
        findings=tuple(findings),
    )


def render_curriculum_text(result: CurriculumDriftResult) -> str:
    lines = [
        f"PromptABI curriculum-stage drift ({result.version})",
        f"result: {'STABLE' if result.stable else 'DRIFTED'}",
    ]
    for finding in result.findings:
        lines.append(
            f"  ! {finding.kind.value} [{finding.from_stage}->{finding.to_stage}]: "
            f"{finding.detail}"
        )
    return "\n".join(lines) + "\n"
