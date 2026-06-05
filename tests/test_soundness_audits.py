import json

import pytest

from promptabi import (
    CheckMode,
    SoundnessAuditStatus,
    build_soundness_audit_report,
    render_soundness_audit_json,
    render_soundness_audit_markdown,
    render_soundness_audit_text,
    soundness_audits,
)
from promptabi.cli import main
from promptabi.compatibility_matrix import CHECK_RULE_IDS
from promptabi.session import CHECK_MODE_CATALOG


def test_soundness_audit_covers_canonical_built_in_rules() -> None:
    report = build_soundness_audit_report()

    assert report.passed
    assert {audit.check for audit in report.audits} == set(CHECK_RULE_IDS)
    for audit in report.audits:
        assert audit.rule_ids == CHECK_RULE_IDS[audit.check]
        assert audit.canonical
        assert audit.assumptions
        assert audit.supported_fragments
        assert audit.proof_obligations
        assert audit.differential_evidence
        assert all(rule_id in CHECK_MODE_CATALOG for rule_id in audit.rule_ids)
        assert set(audit.modes) == {mode for rule_id in audit.rule_ids for mode in CHECK_MODE_CATALOG[rule_id]}


def test_soundness_status_never_promotes_blind_spots_to_full_soundness() -> None:
    report = build_soundness_audit_report()

    assert any(audit.blind_spots for audit in report.audits)
    assert all(
        audit.status is not SoundnessAuditStatus.SOUND_WITHIN_FRAGMENT
        for audit in report.audits
        if audit.blind_spots
    )
    heuristic = next(audit for audit in report.audits if audit.check == "stop-tokenizer-analysis")
    assert CheckMode.HEURISTIC in heuristic.modes
    assert heuristic.status in {SoundnessAuditStatus.CONDITIONALLY_SOUND, SoundnessAuditStatus.HEURISTIC}


def test_soundness_audit_core_rule_is_substantive_and_filterable() -> None:
    report = build_soundness_audit_report(rule="role-boundary-nonforgeability")

    assert len(report.audits) == 1
    audit = report.audits[0]
    assert audit.check == "role-boundary-nonforgeability"
    assert audit.status is SoundnessAuditStatus.CONDITIONALLY_SOUND
    assert "bounded symbolic role regions" in audit.abstraction
    assert any(obligation.name == "marker-reachability" for obligation in audit.proof_obligations)
    assert any(evidence.name == "proof-sketch-replay" for evidence in audit.differential_evidence)
    assert any(blind_spot.kind == "semantic-obedience" for blind_spot in audit.blind_spots)


def test_soundness_audit_renderers_are_deterministic() -> None:
    report = build_soundness_audit_report(rule="grammar-tokenizer-empty")

    first_json = render_soundness_audit_json(report)
    second_json = render_soundness_audit_json(report)
    payload = json.loads(first_json)
    text = render_soundness_audit_text(report)
    markdown = render_soundness_audit_markdown(report)

    assert first_json == second_json
    assert payload["passed"] is True
    assert payload["audits"][0]["check"] == "grammar-tokenizer-emptiness"
    assert "automaton-witness" in text
    assert markdown.startswith("# PromptABI soundness audit")
    assert "| grammar-tokenizer-emptiness |" in markdown


def test_soundness_audit_public_api_and_cli(capsys) -> None:
    rendered = soundness_audits(rule="static-contracts", output_format="markdown")

    assert isinstance(rendered, str)
    assert "solver-assignment-validity" in rendered

    exit_code = main(["soundness-audit", "--rule", "static-contracts", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["audits"][0]["check"] == "static-contracts"
    assert "z3-backed-smt" in payload["audits"][0]["modes"]
    assert captured.err == ""


def test_soundness_audit_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="unknown soundness audit rule"):
        build_soundness_audit_report(rule="not-a-rule")
