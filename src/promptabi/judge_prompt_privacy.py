"""RLHF judge-prompt privacy checks (step 264).

An RLHF/LLM-as-judge pipeline feeds a *judge prompt* (the rubric + the transcript
being scored) to a model.  If that prompt embeds private fields -- raw PII, a
system secret, an internal user id -- the judging step becomes a data-exfiltration
channel and a privacy violation.  This module inspects a judge prompt against a
declared set of private fields and patterns and proves none leak into the text
the judge sees, distinguishing *redacted* references (hashes, ``[REDACTED]``)
from raw values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

JUDGE_PRIVACY_VERSION = "promptabi.judge-privacy.v1"

_DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit-card", r"\b(?:\d[ -]?){13,16}\b"),
)

_REDACTION_MARKERS = ("[REDACTED]", "<redacted>", "sha256:")


class JudgePrivacyFindingKind(StrEnum):
    PRIVATE_FIELD_LEAK = "private-field-leak"
    PATTERN_LEAK = "pattern-leak"


@dataclass(frozen=True, slots=True)
class JudgePrompt:
    rubric: str
    transcript: str
    private_field_values: tuple[str, ...] = ()

    def text(self) -> str:
        return self.rubric + "\n" + self.transcript


@dataclass(frozen=True, slots=True)
class JudgePrivacyFinding:
    kind: JudgePrivacyFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class JudgePrivacyResult:
    version: str
    private: bool
    findings: tuple[JudgePrivacyFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "private": self.private,
            "findings": [f.to_dict() for f in self.findings],
        }


def _is_redacted_context(text: str, idx: int) -> bool:
    window = text[max(0, idx - 16) : idx + 16]
    return any(marker in window for marker in _REDACTION_MARKERS)


def check_judge_privacy(
    prompt: JudgePrompt,
    extra_patterns: tuple[tuple[str, str], ...] = (),
) -> JudgePrivacyResult:
    findings: list[JudgePrivacyFinding] = []
    text = prompt.text()

    for value in prompt.private_field_values:
        if not value:
            continue
        idx = text.find(value)
        if idx != -1 and not _is_redacted_context(text, idx):
            findings.append(
                JudgePrivacyFinding(
                    JudgePrivacyFindingKind.PRIVATE_FIELD_LEAK,
                    f"raw private value {value!r} appears in the judge prompt",
                )
            )

    for name, pattern in (*_DEFAULT_PATTERNS, *extra_patterns):
        for match in re.finditer(pattern, text):
            if not _is_redacted_context(text, match.start()):
                findings.append(
                    JudgePrivacyFinding(
                        JudgePrivacyFindingKind.PATTERN_LEAK,
                        f"unredacted {name} pattern matched at offset {match.start()}",
                    )
                )

    return JudgePrivacyResult(
        version=JUDGE_PRIVACY_VERSION,
        private=not findings,
        findings=tuple(findings),
    )


def render_judge_privacy_text(result: JudgePrivacyResult) -> str:
    lines = [
        f"PromptABI RLHF judge-prompt privacy ({result.version})",
        f"result: {'PRIVATE' if result.private else 'LEAKING'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
