"""Track context-window semantics by provider revision (step 285).

"Context window" is not one number.  A provider revision pins several coupled
quantities: the maximum *total* tokens, the maximum *output* tokens, whether the
prompt and completion share one budget, and how the special/template tokens count
against it.  When a revision silently changes any of these, requests that used to
fit start being rejected or truncated.  This module records context-window
semantics per provider revision and proves whether a planned request fits, plus
detects breaking changes between two revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

CONTEXT_WINDOW_VERSION = "promptabi.context-window.v1"


class ContextFindingKind(StrEnum):
    EXCEEDS_TOTAL = "exceeds-total"
    EXCEEDS_OUTPUT = "exceeds-output"
    SHARED_BUDGET_OVERFLOW = "shared-budget-overflow"


class RevisionChangeKind(StrEnum):
    TOTAL_SHRUNK = "total-shrunk"
    OUTPUT_SHRUNK = "output-shrunk"
    BUDGET_MODEL_CHANGED = "budget-model-changed"


@dataclass(frozen=True, slots=True)
class ContextWindowSemantics:
    provider: str
    revision: str
    max_total_tokens: int
    max_output_tokens: int
    shared_budget: bool  # prompt + output share max_total_tokens
    template_token_overhead: int = 0


@dataclass(frozen=True, slots=True)
class RequestPlan:
    prompt_tokens: int
    requested_output_tokens: int


@dataclass(frozen=True, slots=True)
class ContextFinding:
    kind: ContextFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FitResult:
    version: str
    fits: bool
    findings: tuple[ContextFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fits": self.fits,
            "findings": [f.to_dict() for f in self.findings],
        }


def check_request_fits(
    sem: ContextWindowSemantics, plan: RequestPlan
) -> FitResult:
    findings: list[ContextFinding] = []
    prompt = plan.prompt_tokens + sem.template_token_overhead

    if plan.requested_output_tokens > sem.max_output_tokens:
        findings.append(
            ContextFinding(
                ContextFindingKind.EXCEEDS_OUTPUT,
                f"requested {plan.requested_output_tokens} output > "
                f"{sem.max_output_tokens} max",
            )
        )

    if sem.shared_budget:
        total = prompt + plan.requested_output_tokens
        if total > sem.max_total_tokens:
            findings.append(
                ContextFinding(
                    ContextFindingKind.SHARED_BUDGET_OVERFLOW,
                    f"prompt+output {total} > shared budget {sem.max_total_tokens}",
                )
            )
    else:
        if prompt > sem.max_total_tokens:
            findings.append(
                ContextFinding(
                    ContextFindingKind.EXCEEDS_TOTAL,
                    f"prompt {prompt} > max_total {sem.max_total_tokens}",
                )
            )

    return FitResult(
        version=CONTEXT_WINDOW_VERSION,
        fits=not findings,
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class RevisionChange:
    kind: RevisionChangeKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


def diff_revisions(
    old: ContextWindowSemantics, new: ContextWindowSemantics
) -> tuple[RevisionChange, ...]:
    changes: list[RevisionChange] = []
    if new.max_total_tokens < old.max_total_tokens:
        changes.append(
            RevisionChange(
                RevisionChangeKind.TOTAL_SHRUNK,
                f"{old.max_total_tokens} -> {new.max_total_tokens}",
            )
        )
    if new.max_output_tokens < old.max_output_tokens:
        changes.append(
            RevisionChange(
                RevisionChangeKind.OUTPUT_SHRUNK,
                f"{old.max_output_tokens} -> {new.max_output_tokens}",
            )
        )
    if new.shared_budget != old.shared_budget:
        changes.append(
            RevisionChange(
                RevisionChangeKind.BUDGET_MODEL_CHANGED,
                f"shared_budget {old.shared_budget} -> {new.shared_budget}",
            )
        )
    return tuple(changes)


def render_fit_text(result: FitResult) -> str:
    lines = [
        f"PromptABI context-window fit ({result.version})",
        f"result: {'FITS' if result.fits else 'OVERFLOW'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
