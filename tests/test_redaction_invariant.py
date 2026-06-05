"""Tests for redaction-survives-packing proofs (step 277)."""

from __future__ import annotations

from promptabi.redaction_invariant import (
    PackingStage,
    RedactionFindingKind,
    prove_redaction_invariant,
    render_redaction_invariant_text,
)

SECRETS = {"ssn": "123-45-6789", "email": "alice@example.com"}


def test_redaction_preserved() -> None:
    stages = (
        PackingStage("template", lambda t: f"<s>{t}</s>"),
        PackingStage("pad", lambda t: t + " <pad>"),
    )
    result = prove_redaction_invariant("user [REDACTED] said hi", SECRETS, stages)
    assert result.preserved


def test_secret_present_at_input() -> None:
    result = prove_redaction_invariant("ssn is 123-45-6789", SECRETS, ())
    assert any(
        f.kind is RedactionFindingKind.SECRET_PRESENT_AT_INPUT for f in result.findings
    )


def test_secret_reintroduced_by_buggy_stage() -> None:
    leaky = PackingStage("leak", lambda t: t + " alice@example.com")
    result = prove_redaction_invariant("clean text", SECRETS, (leaky,))
    assert not result.preserved
    finding = next(
        f for f in result.findings
        if f.kind is RedactionFindingKind.SECRET_REINTRODUCED
    )
    assert finding.stage == "leak"
    assert finding.secret_label == "email"


def test_render_text_smoke() -> None:
    result = prove_redaction_invariant("clean", SECRETS, ())
    assert "redaction" in render_redaction_invariant_text(result)
