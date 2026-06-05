"""Model multi-modal placeholders as interface artifacts (step 269).

Multi-modal chat templates embed *placeholders* (``<image>``, ``<|audio|>``) that
the runtime replaces with media features.  These placeholders are interface
artifacts with strict obligations: the number of placeholders in the rendered
text must equal the number of media items provided, their order must match, and
-- critically -- user-authored text must not be able to *forge* a placeholder and
smuggle in or displace a media slot.

This module proves the placeholder-to-media correspondence and detects forgeable
placeholders appearing inside user content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

MULTIMODAL_PLACEHOLDER_VERSION = "promptabi.multimodal-placeholder.v1"


class PlaceholderFindingKind(StrEnum):
    COUNT_MISMATCH = "count-mismatch"
    ORDER_MISMATCH = "order-mismatch"
    FORGED_IN_USER_CONTENT = "forged-in-user-content"
    UNKNOWN_PLACEHOLDER = "unknown-placeholder"


@dataclass(frozen=True, slots=True)
class MediaItem:
    modality: str  # "image" | "audio" | ...


@dataclass(frozen=True, slots=True)
class PlaceholderSpec:
    token: str
    modality: str


@dataclass(frozen=True, slots=True)
class RenderedMultimodalPrompt:
    rendered_text: str
    user_authored_text: str
    media: tuple[MediaItem, ...]


@dataclass(frozen=True, slots=True)
class PlaceholderFinding:
    kind: PlaceholderFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PlaceholderResult:
    version: str
    valid: bool
    findings: tuple[PlaceholderFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def _scan_placeholders(
    text: str, specs: tuple[PlaceholderSpec, ...]
) -> list[tuple[int, PlaceholderSpec]]:
    hits: list[tuple[int, PlaceholderSpec]] = []
    for spec in specs:
        start = text.find(spec.token)
        while start != -1:
            hits.append((start, spec))
            start = text.find(spec.token, start + 1)
    hits.sort(key=lambda h: h[0])
    return hits


def verify_placeholders(
    prompt: RenderedMultimodalPrompt,
    specs: tuple[PlaceholderSpec, ...],
) -> PlaceholderResult:
    findings: list[PlaceholderFinding] = []

    hits = _scan_placeholders(prompt.rendered_text, specs)
    rendered_modalities = tuple(spec.modality for _, spec in hits)
    media_modalities = tuple(m.modality for m in prompt.media)

    if len(rendered_modalities) != len(media_modalities):
        findings.append(
            PlaceholderFinding(
                PlaceholderFindingKind.COUNT_MISMATCH,
                f"{len(rendered_modalities)} placeholders != "
                f"{len(media_modalities)} media items",
            )
        )
    elif rendered_modalities != media_modalities:
        findings.append(
            PlaceholderFinding(
                PlaceholderFindingKind.ORDER_MISMATCH,
                f"placeholder modality order {list(rendered_modalities)} != "
                f"media order {list(media_modalities)}",
            )
        )

    for spec in specs:
        if spec.token in prompt.user_authored_text:
            findings.append(
                PlaceholderFinding(
                    PlaceholderFindingKind.FORGED_IN_USER_CONTENT,
                    f"placeholder {spec.token!r} appears in user-authored content; "
                    "a user could inject or displace a media slot",
                )
            )

    return PlaceholderResult(
        version=MULTIMODAL_PLACEHOLDER_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def render_placeholder_text(result: PlaceholderResult) -> str:
    lines = [
        f"PromptABI multi-modal placeholder check ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
