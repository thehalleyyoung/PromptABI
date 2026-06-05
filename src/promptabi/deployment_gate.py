"""Connect provider conformance to deployment gates (step 296).

A deployment gate decides whether a given provider/revision may be promoted to an
environment.  It binds a conformance result to a policy: required obligations
that must pass, a minimum pass-rate, a maximum allowed severity of open issues,
and an allow/deny list of revisions.  This module evaluates a gate and returns a
promote/block decision with the precise blocking reasons -- the bridge between
conformance evidence and release automation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

DEPLOYMENT_GATE_VERSION = "promptabi.deployment-gate.v1"


class GateDecision(StrEnum):
    PROMOTE = "promote"
    BLOCK = "block"


class GateBlockKind(StrEnum):
    PASS_RATE_TOO_LOW = "pass-rate-too-low"
    REQUIRED_OBLIGATION_FAILED = "required-obligation-failed"
    SEVERITY_EXCEEDED = "severity-exceeded"
    REVISION_DENIED = "revision-denied"


_SEVERITY_ORDER = {"minor": 0, "major": 1, "blocker": 2}


@dataclass(frozen=True, slots=True)
class ConformanceEvidence:
    provider: str
    revision: str
    pass_rate: float
    failed_obligations: frozenset[str]
    max_open_severity: str | None  # "minor" | "major" | "blocker" | None


@dataclass(frozen=True, slots=True)
class GatePolicy:
    environment: str
    min_pass_rate: float
    required_obligations: frozenset[str]
    max_allowed_severity: str  # inclusive ceiling
    denied_revisions: frozenset[str] = field(default=frozenset())


@dataclass(frozen=True, slots=True)
class GateBlock:
    kind: GateBlockKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class GateResult:
    version: str
    decision: GateDecision
    blocks: tuple[GateBlock, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "decision": self.decision.value,
            "blocks": [b.to_dict() for b in self.blocks],
        }


def evaluate_gate(
    evidence: ConformanceEvidence, policy: GatePolicy
) -> GateResult:
    blocks: list[GateBlock] = []

    if evidence.revision in policy.denied_revisions:
        blocks.append(
            GateBlock(
                GateBlockKind.REVISION_DENIED,
                f"revision {evidence.revision!r} is on the deny list",
            )
        )

    if evidence.pass_rate < policy.min_pass_rate:
        blocks.append(
            GateBlock(
                GateBlockKind.PASS_RATE_TOO_LOW,
                f"pass-rate {evidence.pass_rate:.3f} < required "
                f"{policy.min_pass_rate:.3f}",
            )
        )

    unmet = policy.required_obligations & evidence.failed_obligations
    for ob in sorted(unmet):
        blocks.append(
            GateBlock(
                GateBlockKind.REQUIRED_OBLIGATION_FAILED,
                f"required obligation {ob!r} failed",
            )
        )

    if evidence.max_open_severity is not None:
        ceiling = _SEVERITY_ORDER.get(policy.max_allowed_severity, 2)
        observed = _SEVERITY_ORDER.get(evidence.max_open_severity, 2)
        if observed > ceiling:
            blocks.append(
                GateBlock(
                    GateBlockKind.SEVERITY_EXCEEDED,
                    f"open severity {evidence.max_open_severity!r} exceeds "
                    f"ceiling {policy.max_allowed_severity!r}",
                )
            )

    decision = GateDecision.BLOCK if blocks else GateDecision.PROMOTE
    return GateResult(
        version=DEPLOYMENT_GATE_VERSION,
        decision=decision,
        blocks=tuple(blocks),
    )


def render_gate_text(result: GateResult) -> str:
    lines = [
        f"PromptABI deployment gate ({result.version})",
        f"decision: {result.decision.value.upper()}",
    ]
    for b in result.blocks:
        lines.append(f"  ! {b.kind.value}: {b.detail}")
    return "\n".join(lines) + "\n"
