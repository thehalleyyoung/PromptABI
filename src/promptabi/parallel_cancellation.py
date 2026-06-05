"""Parallel tool-call cancellation semantics (step 292).

When a model emits several tool calls in one turn and the client cancels the
turn mid-flight, the contract must be unambiguous: every emitted call is either
answered, or explicitly cancelled; no call is silently dropped; and no result is
returned for a call that was cancelled before dispatch.  This module replays a
parallel tool-call episode with a cancellation point and verifies the
follow-up message accounts for every call exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PARALLEL_CANCELLATION_VERSION = "promptabi.parallel-cancellation.v1"


class CancellationFindingKind(StrEnum):
    UNACCOUNTED_CALL = "unaccounted-call"
    RESULT_FOR_CANCELLED = "result-for-cancelled"
    DOUBLE_ACCOUNTED = "double-accounted"
    UNKNOWN_CALL_REFERENCED = "unknown-call-referenced"


@dataclass(frozen=True, slots=True)
class EmittedCall:
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class CallDisposition:
    call_id: str
    status: str  # "answered" | "cancelled"


@dataclass(frozen=True, slots=True)
class CancellationFinding:
    kind: CancellationFindingKind
    call_id: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "call_id": self.call_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CancellationResult:
    version: str
    consistent: bool
    findings: tuple[CancellationFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "consistent": self.consistent,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_cancellation(
    emitted: tuple[EmittedCall, ...],
    dispositions: tuple[CallDisposition, ...],
    cancelled_before_dispatch: frozenset[str],
) -> CancellationResult:
    findings: list[CancellationFinding] = []
    emitted_ids = {c.call_id for c in emitted}

    seen: dict[str, int] = {}
    for d in dispositions:
        seen[d.call_id] = seen.get(d.call_id, 0) + 1
        if d.call_id not in emitted_ids:
            findings.append(
                CancellationFinding(
                    CancellationFindingKind.UNKNOWN_CALL_REFERENCED,
                    d.call_id,
                    "disposition references a call that was never emitted",
                )
            )
        if (
            d.status == "answered"
            and d.call_id in cancelled_before_dispatch
        ):
            findings.append(
                CancellationFinding(
                    CancellationFindingKind.RESULT_FOR_CANCELLED,
                    d.call_id,
                    "result returned for a call cancelled before dispatch",
                )
            )

    for call_id, count in seen.items():
        if count > 1 and call_id in emitted_ids:
            findings.append(
                CancellationFinding(
                    CancellationFindingKind.DOUBLE_ACCOUNTED,
                    call_id,
                    f"call accounted for {count} times",
                )
            )

    for call in emitted:
        if call.call_id not in seen:
            findings.append(
                CancellationFinding(
                    CancellationFindingKind.UNACCOUNTED_CALL,
                    call.call_id,
                    "emitted call has no answered/cancelled disposition",
                )
            )

    return CancellationResult(
        version=PARALLEL_CANCELLATION_VERSION,
        consistent=not findings,
        findings=tuple(findings),
    )


def render_cancellation_text(result: CancellationResult) -> str:
    lines = [
        f"PromptABI parallel tool-call cancellation ({result.version})",
        f"result: {'CONSISTENT' if result.consistent else 'INCONSISTENT'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.call_id}]: {f.detail}")
    return "\n".join(lines) + "\n"
