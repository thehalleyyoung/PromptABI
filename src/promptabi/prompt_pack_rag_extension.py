"""Verify prompt-pack RAG extension points (step 251).

A pack may declare *extension points* -- named slots where a consumer injects
retrieved (RAG) context at runtime, e.g. ``{{retrieved_docs}}`` inside a system
preamble.  These slots are a classic injection surface: if untrusted retrieved
text can render as a role header or a control delimiter, retrieval becomes a
prompt-injection channel.

This module verifies declared extension points are (a) actually present in the
template, (b) not placed inside a role/control region where injected text would
be interpreted structurally, and (c) covered by a declared sanitizer.  It reuses
the project's notion of forbidden control markers to flag unsafe slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_RAG_VERSION = "promptabi.prompt-pack-rag.v1"

# Markers that must never be forgeable from injected retrieval content.
DEFAULT_CONTROL_MARKERS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
)


class RagFindingKind(StrEnum):
    SLOT_NOT_FOUND = "slot-not-found"
    SLOT_UNSANITIZED = "slot-unsanitized"
    SLOT_IN_CONTROL_REGION = "slot-in-control-region"
    DUPLICATE_SLOT = "duplicate-slot"


@dataclass(frozen=True, slots=True)
class ExtensionPoint:
    name: str
    placeholder: str
    sanitizer: str | None = None


@dataclass(frozen=True, slots=True)
class PackTemplate:
    """A pack template plus the control markers it considers structural."""

    source: str
    control_markers: tuple[str, ...] = DEFAULT_CONTROL_MARKERS

    def control_region_spans(self) -> tuple[tuple[int, int], ...]:
        """Return (start, end) spans for each control-marker header region.

        A header region runs from a control marker to the end of its line (or
        the next control marker), because text on the *same line* as a role
        delimiter is structurally part of that delimiter's header.
        """

        spans: list[tuple[int, int]] = []
        for marker in self.control_markers:
            start = self.source.find(marker)
            while start != -1:
                newline = self.source.find("\n", start + len(marker))
                end = newline if newline != -1 else len(self.source)
                spans.append((start, end))
                start = self.source.find(marker, start + 1)
        return tuple(spans)


@dataclass(frozen=True, slots=True)
class RagFinding:
    kind: RagFindingKind
    slot: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "slot": self.slot, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RagVerification:
    version: str
    safe: bool
    findings: tuple[RagFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "safe": self.safe,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_extension_points(
    template: PackTemplate,
    points: tuple[ExtensionPoint, ...],
) -> RagVerification:
    findings: list[RagFinding] = []
    seen: set[str] = set()

    for point in points:
        if point.name in seen:
            findings.append(
                RagFinding(
                    RagFindingKind.DUPLICATE_SLOT,
                    point.name,
                    "extension point declared more than once",
                )
            )
        seen.add(point.name)

        idx = template.source.find(point.placeholder)
        if idx == -1:
            findings.append(
                RagFinding(
                    RagFindingKind.SLOT_NOT_FOUND,
                    point.name,
                    f"placeholder {point.placeholder!r} not present in template",
                )
            )
            continue

        if point.sanitizer is None:
            findings.append(
                RagFinding(
                    RagFindingKind.SLOT_UNSANITIZED,
                    point.name,
                    "injected RAG content is not routed through a declared sanitizer",
                )
            )

        slot_end = idx + len(point.placeholder)
        for start, end in template.control_region_spans():
            # Slot overlaps a control marker region -> structurally dangerous.
            if idx < end and slot_end > start:
                findings.append(
                    RagFinding(
                        RagFindingKind.SLOT_IN_CONTROL_REGION,
                        point.name,
                        "placeholder overlaps a control-marker region; injected "
                        "text could forge a role/control delimiter",
                    )
                )
                break

    return RagVerification(
        version=PROMPT_PACK_RAG_VERSION,
        safe=not findings,
        findings=tuple(findings),
    )


def render_rag_verification_text(result: RagVerification) -> str:
    lines = [
        f"PromptABI prompt-pack RAG extension points ({result.version})",
        f"result: {'SAFE' if result.safe else 'UNSAFE'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.kind.value} [{finding.slot}]: {finding.detail}")
    return "\n".join(lines) + "\n"
