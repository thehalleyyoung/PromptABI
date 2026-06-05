from datetime import date, timedelta

from promptabi import (
    PolicyPackSemanticsViolationKind,
    prove_policy_pack_preserves_semantics,
    render_policy_pack_semantics_text,
)
from promptabi.diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity, WitnessTrace
from promptabi.policies import Suppression, VerificationPolicy


def _diag(rule_id: str, severity: DiagnosticSeverity = DiagnosticSeverity.ERROR) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=f"{rule_id} message",
        artifact=ArtifactRef(kind="tokenizer", name="artifact-a", path="memory://a"),
        witness=WitnessTrace(summary=f"why {rule_id}"),
    )


def test_severity_override_preserves_semantics() -> None:
    raw = (_diag("tokenizer-roundtrip"), _diag("template-role-leak"))
    policy = VerificationPolicy(
        severity_overrides=(("template-role-leak", DiagnosticSeverity.WARNING),),
    )

    report = prove_policy_pack_preserves_semantics(raw, policy)

    assert report.preserves_semantics
    assert report.preserved_findings == 2
    assert "violations: none" in render_policy_pack_semantics_text(report)


def test_justified_suppression_is_preserved_as_auditable_record() -> None:
    target = _diag("tokenizer-roundtrip")
    policy = VerificationPolicy(
        suppressions=(
            Suppression(
                rule_id="tokenizer-roundtrip",
                justification="vendor confirmed safe for this corpus",
                fingerprint=target.fingerprint,
                witness_digest=target.witness_digest,
                owner="team-llm",
                accepted_risk="low",
                expires_on=date.today() + timedelta(days=30),
            ),
        ),
    )

    report = prove_policy_pack_preserves_semantics((target,), policy)

    # The finding is suppressed but its identity survives, so semantics are preserved.
    assert report.preserves_semantics
    assert report.preserved_findings == 1


def test_dropped_finding_is_detected_via_untrusted_transformation() -> None:
    raw = (_diag("tokenizer-roundtrip"), _diag("template-role-leak"))
    # An untrusted policy engine claims to drop a finding entirely.
    claimed_applied = (_diag("template-role-leak"),)

    report = prove_policy_pack_preserves_semantics(
        raw, VerificationPolicy(), claimed_applied=claimed_applied
    )

    assert not report.preserves_semantics
    assert any(
        v.kind == PolicyPackSemanticsViolationKind.DROPPED_FINDING for v in report.violations
    )
    dropped = next(
        v for v in report.violations if v.kind == PolicyPackSemanticsViolationKind.DROPPED_FINDING
    )
    assert dropped.rule_id == "tokenizer-roundtrip"
    assert dropped.witness.minimal_fixes


def test_fabricated_error_is_detected() -> None:
    raw = (_diag("tokenizer-roundtrip"),)
    claimed_applied = (_diag("tokenizer-roundtrip"), _diag("invented-error"))

    report = prove_policy_pack_preserves_semantics(
        raw, VerificationPolicy(), claimed_applied=claimed_applied
    )

    assert not report.preserves_semantics
    assert any(
        v.kind == PolicyPackSemanticsViolationKind.FABRICATED_FINDING for v in report.violations
    )


def test_semantic_downgrade_of_critical_rule_is_detected() -> None:
    raw = (_diag("prompt-injection-guard"),)
    policy = VerificationPolicy(
        severity_overrides=(("prompt-injection-guard", DiagnosticSeverity.INFO),),
    )

    report = prove_policy_pack_preserves_semantics(
        raw, policy, critical_rule_ids=("prompt-injection-guard",)
    )

    assert not report.preserves_semantics
    assert any(
        v.kind == PolicyPackSemanticsViolationKind.SEMANTIC_DOWNGRADE for v in report.violations
    )


def test_unjustified_suppression_is_detected() -> None:
    target = _diag("tokenizer-roundtrip")
    policy = VerificationPolicy(
        suppressions=(
            Suppression(
                rule_id="tokenizer-roundtrip",
                justification="   ",
                fingerprint=target.fingerprint,
            ),
        ),
    )

    report = prove_policy_pack_preserves_semantics((target,), policy)

    assert not report.preserves_semantics
    assert any(
        v.kind == PolicyPackSemanticsViolationKind.UNJUSTIFIED_SUPPRESSION for v in report.violations
    )
