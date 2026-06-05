"""Standard conformance issue templates (step 295).

When a provider fails a conformance vector, the failure should be reportable in a
*standard, machine-and-human-readable* form so providers can triage and fix it.
This module renders a conformance finding into a standardized issue template --
stable title, severity, reproduction (vector id + obligation), expected vs.
observed, and a deterministic fingerprint for deduplication across runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

CONFORMANCE_ISSUE_TEMPLATE_VERSION = "promptabi.conformance-issue-template.v1"


class IssueSeverity(StrEnum):
    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True, slots=True)
class ConformanceFailure:
    vector_id: str
    obligation: str
    expected: str
    observed: str
    severity: IssueSeverity


@dataclass(frozen=True, slots=True)
class IssueTemplate:
    version: str
    title: str
    severity: IssueSeverity
    fingerprint: str
    body: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "title": self.title,
            "severity": self.severity.value,
            "fingerprint": self.fingerprint,
            "body": self.body,
        }


def _fingerprint(failure: ConformanceFailure) -> str:
    raw = "|".join(
        [
            CONFORMANCE_ISSUE_TEMPLATE_VERSION,
            failure.vector_id,
            failure.obligation,
            failure.severity.value,
        ]
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def render_issue_template(failure: ConformanceFailure) -> IssueTemplate:
    title = f"[conformance:{failure.severity.value}] {failure.obligation} " \
            f"fails for {failure.vector_id}"
    fingerprint = _fingerprint(failure)
    body = "\n".join(
        [
            f"Fingerprint: {fingerprint}",
            f"Vector: {failure.vector_id}",
            f"Obligation: {failure.obligation}",
            "",
            "Expected:",
            f"  {failure.expected}",
            "",
            "Observed:",
            f"  {failure.observed}",
            "",
            "Reproduce:",
            f"  promptabi replay-vector {failure.vector_id}",
        ]
    )
    return IssueTemplate(
        version=CONFORMANCE_ISSUE_TEMPLATE_VERSION,
        title=title,
        severity=failure.severity,
        fingerprint=fingerprint,
        body=body,
    )


def dedup_issues(
    failures: tuple[ConformanceFailure, ...]
) -> tuple[IssueTemplate, ...]:
    seen: dict[str, IssueTemplate] = {}
    for failure in failures:
        tpl = render_issue_template(failure)
        seen.setdefault(tpl.fingerprint, tpl)
    return tuple(seen.values())


def render_issue_template_text(template: IssueTemplate) -> str:
    return (
        f"PromptABI conformance issue ({template.version})\n"
        f"{template.title}\n"
        f"{template.body}\n"
    )
