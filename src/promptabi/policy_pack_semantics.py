"""Prove that local policy packs preserve PromptABI checker semantics.

A policy pack lets a repository remap severities and accept risks, but it must
never silently change *what the checker actually found*.  This module formalizes
and verifies a semantics-preservation theorem for the policy layer:

* **No dropped finding** -- every raw checker finding survives policy application,
  either directly (possibly with a remapped severity) or as an auditable
  ``diagnostic-suppressed`` record that carries its original identity.
* **No fabricated finding** -- every error-level diagnostic emitted after policy
  application traces back to a raw finding or to a known policy meta-diagnostic
  class; a policy cannot invent failures the checker never produced.
* **No semantic downgrade** -- a safety-critical rule cannot be relaxed below the
  configured floor, which would hide real bugs behind a severity override.
* **No unjustified suppression** -- every suppression must carry a justification,
  so an accepted risk is always recorded rather than erased.

The prover runs the *real* ``apply_policy_diagnostics`` transformer, and can also
check an untrusted, externally-claimed transformation to surface counterexamples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from .diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .policies import VerificationPolicy, apply_policy_diagnostics


POLICY_PACK_SEMANTICS_VERSION = "promptabi.policy-pack-semantics.v1"

_SEVERITY_RANK = {
    DiagnosticSeverity.INFO: 0,
    DiagnosticSeverity.WARNING: 1,
    DiagnosticSeverity.ERROR: 2,
}

_KNOWN_META_RULES = frozenset(
    {
        "diagnostic-suppressed",
        "policy-suppression-invalid",
        "policy-suppression-stale-witness",
        "policy-severity-threshold",
        "policy-pack-verified",
    }
)


class PolicyPackSemanticsViolationKind(StrEnum):
    """Concrete ways a policy pack can fail to preserve checker semantics."""

    DROPPED_FINDING = "dropped-finding"
    FABRICATED_FINDING = "fabricated-finding"
    SEMANTIC_DOWNGRADE = "semantic-downgrade"
    UNJUSTIFIED_SUPPRESSION = "unjustified-suppression"


@dataclass(frozen=True, slots=True)
class PolicyPackSemanticsViolation:
    """One semantics-preservation violation with a replayable witness."""

    kind: PolicyPackSemanticsViolationKind
    message: str
    rule_id: str
    witness: WitnessTrace
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PolicyPackSemanticsReport:
    """Result of proving a policy pack preserves checker semantics."""

    raw_count: int
    applied_count: int
    preserved_findings: int
    violations: tuple[PolicyPackSemanticsViolation, ...]

    @property
    def preserves_semantics(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "applied_count": self.applied_count,
            "preserved_findings": self.preserved_findings,
            "preserves_semantics": self.preserves_semantics,
            "raw_count": self.raw_count,
            "version": POLICY_PACK_SEMANTICS_VERSION,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def prove_policy_pack_preserves_semantics(
    raw_diagnostics: tuple[Diagnostic, ...],
    policy: VerificationPolicy,
    *,
    claimed_applied: tuple[Diagnostic, ...] | None = None,
    critical_rule_ids: tuple[str, ...] = (),
    severity_floor: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    today: date | None = None,
) -> PolicyPackSemanticsReport:
    """Prove (or refute) that ``policy`` preserves the semantics of ``raw_diagnostics``."""

    applied = (
        claimed_applied
        if claimed_applied is not None
        else apply_policy_diagnostics(raw_diagnostics, policy, today=today)
    )
    artifact = ArtifactRef(kind="policy-pack", name="local-policy-pack", path="memory://policy-pack")
    violations: list[PolicyPackSemanticsViolation] = []

    applied_structural = {
        _identity(diag) for diag in applied if diag.rule_id != "diagnostic-suppressed"
    }
    applied_originals = {
        (_property(diag, "original_rule_id"), _property(diag, "original_fingerprint"))
        for diag in applied
        if diag.rule_id == "diagnostic-suppressed"
    }
    preserved = 0
    for raw in raw_diagnostics:
        if _identity(raw) in applied_structural or (raw.rule_id, raw.fingerprint) in applied_originals:
            preserved += 1
            continue
        violations.append(
            _violation(
                PolicyPackSemanticsViolationKind.DROPPED_FINDING,
                f"raw finding '{raw.rule_id}' ({raw.fingerprint}) disappeared after policy application",
                raw.rule_id,
                artifact,
                step_in=f"raw {raw.rule_id}#{raw.fingerprint}",
                step_out="absent from applied output",
                suggestion="A policy pack may remap severity or record a justified suppression, never erase a finding.",
            )
        )

    raw_structural = {_identity(diag) for diag in raw_diagnostics}
    for diag in applied:
        if diag.severity is not DiagnosticSeverity.ERROR:
            continue
        if diag.rule_id in _KNOWN_META_RULES:
            continue
        if _identity(diag) in raw_structural:
            continue
        violations.append(
            _violation(
                PolicyPackSemanticsViolationKind.FABRICATED_FINDING,
                f"applied error '{diag.rule_id}' ({diag.fingerprint}) has no raw finding it traces back to",
                diag.rule_id,
                artifact,
                step_in="scan applied error diagnostics",
                step_out=f"{diag.rule_id} not in raw findings or meta classes",
                suggestion="Policy packs must not synthesize checker errors; only the checker may emit them.",
            )
        )

    critical = set(critical_rule_ids)
    floor_rank = _SEVERITY_RANK[severity_floor]
    for rule_id, severity in policy.severity_overrides:
        if rule_id in critical and _SEVERITY_RANK[severity] < floor_rank:
            violations.append(
                _violation(
                    PolicyPackSemanticsViolationKind.SEMANTIC_DOWNGRADE,
                    (
                        f"severity override relaxes safety-critical rule '{rule_id}' to "
                        f"'{severity.value}', below the '{severity_floor.value}' floor"
                    ),
                    rule_id,
                    artifact,
                    step_in=f"override {rule_id}->{severity.value}",
                    step_out=f"below floor {severity_floor.value}",
                    suggestion="Keep safety-critical rules at or above the floor; fix the issue instead of hiding it.",
                )
            )

    for suppression in policy.suppressions:
        if not (suppression.justification or "").strip():
            violations.append(
                _violation(
                    PolicyPackSemanticsViolationKind.UNJUSTIFIED_SUPPRESSION,
                    f"suppression for rule '{suppression.rule_id}' carries no justification",
                    suppression.rule_id,
                    artifact,
                    step_in=f"suppress {suppression.rule_id}",
                    step_out="empty justification",
                    suggestion="Record an accepted-risk justification so suppressions remain auditable.",
                )
            )

    return PolicyPackSemanticsReport(
        raw_count=len(raw_diagnostics),
        applied_count=len(applied),
        preserved_findings=preserved,
        violations=tuple(violations),
    )


def render_policy_pack_semantics_json(report: PolicyPackSemanticsReport) -> str:
    """Render a policy-pack semantics report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_policy_pack_semantics_text(report: PolicyPackSemanticsReport) -> str:
    """Render a policy-pack semantics report for CLI users."""

    status = "PRESERVED" if report.preserves_semantics else "VIOLATED"
    lines = [
        "PromptABI local policy-pack semantics proof",
        f"status: {status}",
        f"raw_findings: {report.raw_count}",
        f"applied_diagnostics: {report.applied_count}",
        f"preserved_findings: {report.preserved_findings}",
    ]
    if report.preserves_semantics:
        lines.append("violations: none")
        return "\n".join(lines) + "\n"
    lines.append(f"violations: {len(report.violations)}")
    for violation in report.violations:
        lines.append(f"VIOLATION {violation.kind.value} [{violation.rule_id}]: {violation.message}")
        lines.append(f"  suggestion: {violation.suggestion}")
    return "\n".join(lines) + "\n"


def _identity(diagnostic: Diagnostic) -> tuple[object, ...]:
    """Severity-independent structural identity of a checker finding."""

    artifact = diagnostic.artifact
    art_key = (artifact.kind, artifact.name, artifact.path) if artifact is not None else None
    span_key = diagnostic.span.to_dict() if diagnostic.span is not None else None
    return (diagnostic.rule_id, art_key, diagnostic.message, repr(span_key))


def _property(diagnostic: Diagnostic, key: str) -> str | None:
    for name, value in diagnostic.properties:
        if name == key:
            return str(value) if value is not None else None
    return None


def _violation(
    kind: PolicyPackSemanticsViolationKind,
    message: str,
    rule_id: str,
    artifact: ArtifactRef,
    *,
    step_in: str,
    step_out: str,
    suggestion: str,
) -> PolicyPackSemanticsViolation:
    return PolicyPackSemanticsViolation(
        kind=kind,
        message=message,
        rule_id=rule_id,
        witness=WitnessTrace(
            summary=f"policy pack violates semantics preservation: {kind.value}",
            steps=(
                WitnessStep(action="compare raw checker findings to policy output", input=step_in, output=step_out),
                WitnessStep(action="emit minimal policy fix", input=kind.value, output=suggestion),
            ),
            artifacts=(artifact,),
            minimal_fixes=(suggestion,),
        ),
        suggestion=suggestion,
    )
