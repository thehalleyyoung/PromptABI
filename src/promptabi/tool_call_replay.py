"""Verify multi-turn tool-call replay semantics (step 283).

A tool-using conversation alternates assistant ``tool_call`` turns with ``tool``
result turns.  Replaying such a transcript requires strict semantics: every
tool call must be answered by exactly one tool result carrying the *same call
id*, results must not arrive before their call, and ids must be unique.  Provider
adapters frequently get this wrong (dropping ids, reusing ids, interleaving).

This module replays a transcript and proves the call/result pairing is well
formed, reporting the exact offending turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

TOOL_REPLAY_VERSION = "promptabi.tool-call-replay.v1"


class ToolReplayFindingKind(StrEnum):
    DUPLICATE_CALL_ID = "duplicate-call-id"
    RESULT_BEFORE_CALL = "result-before-call"
    UNANSWERED_CALL = "unanswered-call"
    ORPHAN_RESULT = "orphan-result"
    DOUBLE_ANSWERED = "double-answered"


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str


@dataclass(frozen=True, slots=True)
class ReplayTurn:
    index: int
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolReplayFinding:
    kind: ToolReplayFindingKind
    turn_index: int
    call_id: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "turn_index": self.turn_index,
            "call_id": self.call_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ToolReplayResult:
    version: str
    valid: bool
    findings: tuple[ToolReplayFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def replay_tool_calls(turns: tuple[ReplayTurn, ...]) -> ToolReplayResult:
    findings: list[ToolReplayFinding] = []
    open_calls: dict[str, int] = {}
    answered: set[str] = set()
    seen_ids: set[str] = set()

    for turn in turns:
        for call in turn.calls:
            if call.call_id in seen_ids:
                findings.append(
                    ToolReplayFinding(
                        ToolReplayFindingKind.DUPLICATE_CALL_ID,
                        turn.index,
                        call.call_id,
                        "call id reused",
                    )
                )
            seen_ids.add(call.call_id)
            open_calls[call.call_id] = turn.index

        for result in turn.results:
            if result.call_id not in open_calls:
                findings.append(
                    ToolReplayFinding(
                        ToolReplayFindingKind.ORPHAN_RESULT
                        if result.call_id in answered
                        else ToolReplayFindingKind.RESULT_BEFORE_CALL,
                        turn.index,
                        result.call_id,
                        "result has no preceding open call",
                    )
                )
                continue
            if result.call_id in answered:
                findings.append(
                    ToolReplayFinding(
                        ToolReplayFindingKind.DOUBLE_ANSWERED,
                        turn.index,
                        result.call_id,
                        "call answered more than once",
                    )
                )
            answered.add(result.call_id)

    for call_id, turn_index in open_calls.items():
        if call_id not in answered:
            findings.append(
                ToolReplayFinding(
                    ToolReplayFindingKind.UNANSWERED_CALL,
                    turn_index,
                    call_id,
                    "tool call never received a result",
                )
            )

    return ToolReplayResult(
        version=TOOL_REPLAY_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def render_tool_replay_text(result: ToolReplayResult) -> str:
    lines = [
        f"PromptABI multi-turn tool-call replay ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for f in result.findings:
        lines.append(
            f"  ! {f.kind.value} @turn {f.turn_index} [{f.call_id}]: {f.detail}"
        )
    return "\n".join(lines) + "\n"
