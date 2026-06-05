"""Tests for fine-tune manifest registry gates (step 275)."""

from __future__ import annotations

from promptabi.finetune_registry_gate import (
    FineTuneManifest,
    GateFindingKind,
    GatePolicy,
    evaluate_gate,
    render_gate_text,
)

POLICY = GatePolicy(
    allowed_base_models=frozenset({"meta-llama/Llama-3.1-8B"}),
    required_fields=frozenset({"dataset_digest", "tokenizer_pin", "template_digest"}),
    required_approvals=frozenset({"safety", "legal"}),
)

GOOD = FineTuneManifest(
    name="ft-1",
    base_model="meta-llama/Llama-3.1-8B",
    fields={"dataset_digest": "d", "tokenizer_pin": "t", "template_digest": "tpl"},
    contract_passed=True,
    approvals=frozenset({"safety", "legal"}),
)


def test_admit_good_manifest() -> None:
    assert evaluate_gate(POLICY, GOOD).admitted


def test_base_model_not_allowed() -> None:
    bad = FineTuneManifest("ft", "other/model", GOOD.fields, True, GOOD.approvals)
    result = evaluate_gate(POLICY, bad)
    assert any(f.kind is GateFindingKind.BASE_NOT_ALLOWED for f in result.findings)


def test_missing_field() -> None:
    bad = FineTuneManifest(
        "ft", GOOD.base_model, {"dataset_digest": "d"}, True, GOOD.approvals
    )
    result = evaluate_gate(POLICY, bad)
    assert any(f.kind is GateFindingKind.MISSING_FIELD for f in result.findings)


def test_contract_not_passed() -> None:
    bad = FineTuneManifest("ft", GOOD.base_model, GOOD.fields, False, GOOD.approvals)
    result = evaluate_gate(POLICY, bad)
    assert any(f.kind is GateFindingKind.CONTRACT_NOT_PASSED for f in result.findings)


def test_missing_approval() -> None:
    bad = FineTuneManifest(
        "ft", GOOD.base_model, GOOD.fields, True, frozenset({"safety"})
    )
    result = evaluate_gate(POLICY, bad)
    assert any(f.kind is GateFindingKind.MISSING_APPROVAL for f in result.findings)


def test_render_text_smoke() -> None:
    assert "gate" in render_gate_text(evaluate_gate(POLICY, GOOD))
