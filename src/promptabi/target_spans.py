"""Verify supervised target spans survive truncation (step 270).

When a packed/rendered example is longer than ``max_length`` it is truncated.
If truncation severs a *supervised target span* -- cutting an assistant response
in half, or dropping it entirely -- the trainer learns from a partial or empty
target, a silent and common data bug.  Left-truncation can additionally drop the
prompt that the surviving target depends on.

This module proves that, after truncation to ``max_length`` from a given side,
every supervised target span is preserved intact, reporting the exact span that
was severed or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

TARGET_SPAN_VERSION = "promptabi.target-span.v1"


class TruncationSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class TargetSpanFindingKind(StrEnum):
    SPAN_DROPPED = "span-dropped"
    SPAN_SEVERED = "span-severed"
    PROMPT_DROPPED = "prompt-dropped"


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open ``[start, end)`` token span."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid span bounds")


@dataclass(frozen=True, slots=True)
class TargetSpanFinding:
    kind: TargetSpanFindingKind
    span: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "span": self.span, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class TargetSpanResult:
    version: str
    preserved: bool
    kept_window: tuple[int, int]
    findings: tuple[TargetSpanFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "preserved": self.preserved,
            "kept_window": list(self.kept_window),
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_target_spans(
    total_length: int,
    max_length: int,
    side: TruncationSide,
    target_spans: tuple[Span, ...],
    prompt_span: Span | None = None,
) -> TargetSpanResult:
    if side is TruncationSide.RIGHT:
        kept = (0, min(max_length, total_length))
    else:
        kept = (max(0, total_length - max_length), total_length)

    findings: list[TargetSpanFinding] = []

    def covered(span: Span) -> str:
        if span.end <= kept[0] or span.start >= kept[1]:
            return "dropped"
        if span.start < kept[0] or span.end > kept[1]:
            return "severed"
        return "intact"

    for span in target_spans:
        status = covered(span)
        if status == "dropped":
            findings.append(
                TargetSpanFinding(
                    TargetSpanFindingKind.SPAN_DROPPED,
                    span.name,
                    f"target span [{span.start},{span.end}) lies outside kept "
                    f"window {list(kept)}",
                )
            )
        elif status == "severed":
            findings.append(
                TargetSpanFinding(
                    TargetSpanFindingKind.SPAN_SEVERED,
                    span.name,
                    f"target span [{span.start},{span.end}) is cut by kept window "
                    f"{list(kept)}",
                )
            )

    if prompt_span is not None and covered(prompt_span) == "dropped":
        findings.append(
            TargetSpanFinding(
                TargetSpanFindingKind.PROMPT_DROPPED,
                prompt_span.name,
                "the prompt the surviving target depends on was truncated away",
            )
        )

    return TargetSpanResult(
        version=TARGET_SPAN_VERSION,
        preserved=not findings,
        kept_window=kept,
        findings=tuple(findings),
    )


def render_target_span_text(result: TargetSpanResult) -> str:
    lines = [
        f"PromptABI supervised target-span check ({result.version})",
        f"kept window: {list(result.kept_window)}",
        f"result: {'PRESERVED' if result.preserved else 'DAMAGED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.span}]: {f.detail}")
    return "\n".join(lines) + "\n"
