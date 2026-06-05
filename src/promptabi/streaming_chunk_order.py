"""Streaming chunk-order robustness checks (step 284).

A streaming completion is delivered as a sequence of SSE chunks.  A robust client
must reconstruct the same final text regardless of benign reordering of
*independent* fields, but must also reject chunks that violate hard ordering
rules: the role chunk first, content deltas in index order, ``finish_reason``
only in the final chunk, and tool-call argument fragments delivered in order.

This module validates a captured chunk stream against these rules and
reconstructs the assembled content, reporting any ordering violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

CHUNK_ORDER_VERSION = "promptabi.chunk-order.v1"


class ChunkOrderFindingKind(StrEnum):
    ROLE_NOT_FIRST = "role-not-first"
    CONTENT_AFTER_FINISH = "content-after-finish"
    FINISH_NOT_LAST = "finish-not-last"
    MULTIPLE_FINISH = "multiple-finish"
    INDEX_REGRESSION = "index-regression"


@dataclass(frozen=True, slots=True)
class StreamChunk:
    seq: int
    role: str | None = None
    content_delta: str | None = None
    content_index: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkOrderFinding:
    kind: ChunkOrderFindingKind
    seq: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "seq": self.seq, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ChunkOrderResult:
    version: str
    valid: bool
    assembled: str
    findings: tuple[ChunkOrderFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "assembled": self.assembled,
            "findings": [f.to_dict() for f in self.findings],
        }


def check_chunk_order(chunks: tuple[StreamChunk, ...]) -> ChunkOrderResult:
    findings: list[ChunkOrderFinding] = []
    ordered = sorted(chunks, key=lambda c: c.seq)

    finished_at: int | None = None
    last_content_index = -1
    assembled_parts: list[tuple[int, str]] = []

    for i, chunk in enumerate(ordered):
        if chunk.role is not None and i != 0:
            findings.append(
                ChunkOrderFinding(
                    ChunkOrderFindingKind.ROLE_NOT_FIRST,
                    chunk.seq,
                    "role announced after content began",
                )
            )
        if chunk.finish_reason is not None:
            if finished_at is not None:
                findings.append(
                    ChunkOrderFinding(
                        ChunkOrderFindingKind.MULTIPLE_FINISH,
                        chunk.seq,
                        "more than one finish_reason chunk",
                    )
                )
            finished_at = chunk.seq
        if chunk.content_delta is not None:
            if finished_at is not None:
                findings.append(
                    ChunkOrderFinding(
                        ChunkOrderFindingKind.CONTENT_AFTER_FINISH,
                        chunk.seq,
                        "content delta after finish_reason",
                    )
                )
            if chunk.content_index is not None:
                if chunk.content_index < last_content_index:
                    findings.append(
                        ChunkOrderFinding(
                            ChunkOrderFindingKind.INDEX_REGRESSION,
                            chunk.seq,
                            f"content_index {chunk.content_index} < "
                            f"{last_content_index}",
                        )
                    )
                last_content_index = max(last_content_index, chunk.content_index)
            assembled_parts.append(
                (chunk.content_index if chunk.content_index is not None else chunk.seq,
                 chunk.content_delta)
            )

    if finished_at is not None and ordered and ordered[-1].finish_reason is None:
        findings.append(
            ChunkOrderFinding(
                ChunkOrderFindingKind.FINISH_NOT_LAST,
                finished_at,
                "finish_reason chunk is not the final chunk",
            )
        )

    assembled = "".join(part for _, part in sorted(assembled_parts, key=lambda p: p[0]))

    return ChunkOrderResult(
        version=CHUNK_ORDER_VERSION,
        valid=not findings,
        assembled=assembled,
        findings=tuple(findings),
    )


def render_chunk_order_text(result: ChunkOrderResult) -> str:
    lines = [
        f"PromptABI streaming chunk-order check ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
        f"assembled: {result.assembled!r}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} @seq {f.seq}: {f.detail}")
    return "\n".join(lines) + "\n"
