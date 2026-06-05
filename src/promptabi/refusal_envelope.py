"""Structured-output refusal envelopes (step 290).

When a model refuses (safety, policy) a request that demanded structured output,
a naive client tries to parse the refusal *as* the schema and crashes, or worse,
mistakes the refusal prose for valid data.  A correct contract carries refusals
in a dedicated channel: ``finish_reason == "content_filter"`` or an explicit
``refusal`` field, with ``parsed`` left null.  This module classifies a
structured-output response as valid-data / well-formed-refusal /
ambiguous-refusal (a contract violation) so clients branch safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

REFUSAL_ENVELOPE_VERSION = "promptabi.refusal-envelope.v1"


class RefusalClass(StrEnum):
    VALID_DATA = "valid-data"
    WELL_FORMED_REFUSAL = "well-formed-refusal"
    AMBIGUOUS_REFUSAL = "ambiguous-refusal"


class RefusalFindingKind(StrEnum):
    REFUSAL_IN_DATA_CHANNEL = "refusal-in-data-channel"
    MISSING_REFUSAL_FLAG = "missing-refusal-flag"
    DATA_AND_REFUSAL = "data-and-refusal"


# Heuristic phrases that signal a refusal when they appear as the whole content.
_REFUSAL_MARKERS = (
    "i can't help with that",
    "i cannot help with that",
    "i'm sorry, but i can't",
    "i am unable to assist",
    "i can't assist with that",
)


@dataclass(frozen=True, slots=True)
class StructuredResponse:
    finish_reason: str
    parsed: object | None
    refusal: str | None
    raw_content: str


@dataclass(frozen=True, slots=True)
class RefusalFinding:
    kind: RefusalFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RefusalResult:
    version: str
    classification: RefusalClass
    findings: tuple[RefusalFinding, ...] = field(default=())

    @property
    def safe_to_parse(self) -> bool:
        return self.classification == RefusalClass.VALID_DATA

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "classification": self.classification.value,
            "safe_to_parse": self.safe_to_parse,
            "findings": [f.to_dict() for f in self.findings],
        }


def _looks_like_refusal(text: str) -> bool:
    low = text.strip().lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def classify_refusal(response: StructuredResponse) -> RefusalResult:
    findings: list[RefusalFinding] = []

    explicit_refusal = (
        response.refusal is not None
        or response.finish_reason == "content_filter"
    )
    prose_refusal = response.parsed is None and _looks_like_refusal(
        response.raw_content
    )

    if explicit_refusal and response.parsed is not None:
        findings.append(
            RefusalFinding(
                RefusalFindingKind.DATA_AND_REFUSAL,
                "response carries both parsed data and a refusal flag",
            )
        )
        classification = RefusalClass.AMBIGUOUS_REFUSAL
    elif explicit_refusal:
        classification = RefusalClass.WELL_FORMED_REFUSAL
    elif prose_refusal:
        findings.append(
            RefusalFinding(
                RefusalFindingKind.MISSING_REFUSAL_FLAG,
                "content reads as a refusal but no refusal flag/finish_reason set",
            )
        )
        findings.append(
            RefusalFinding(
                RefusalFindingKind.REFUSAL_IN_DATA_CHANNEL,
                "refusal prose delivered in the data channel",
            )
        )
        classification = RefusalClass.AMBIGUOUS_REFUSAL
    else:
        classification = RefusalClass.VALID_DATA

    return RefusalResult(
        version=REFUSAL_ENVELOPE_VERSION,
        classification=classification,
        findings=tuple(findings),
    )


def render_refusal_text(result: RefusalResult) -> str:
    lines = [
        f"PromptABI refusal-envelope classification ({result.version})",
        f"class: {result.classification.value} "
        f"(safe_to_parse={result.safe_to_parse})",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
