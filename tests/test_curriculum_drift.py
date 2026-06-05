"""Tests for curriculum-stage prompt-ABI drift (step 260)."""

from __future__ import annotations

from promptabi.curriculum_drift import (
    DriftKind,
    StageInterface,
    render_curriculum_text,
    verify_curriculum,
)


def _stage(name: str, **overrides) -> StageInterface:
    base = dict(
        stage=name,
        template_digest="t1",
        special_tokens=frozenset({"<|im_start|>", "<|im_end|>"}),
        roles=frozenset({"system", "user", "assistant"}),
        bos="<s>",
        eos="</s>",
    )
    base.update(overrides)
    return StageInterface(**base)  # type: ignore[arg-type]


def test_stable_curriculum() -> None:
    stages = (_stage("sft"), _stage("domain"), _stage("dpo"))
    assert verify_curriculum(stages).stable


def test_template_drift_detected() -> None:
    stages = (_stage("sft"), _stage("domain", template_digest="t2"))
    result = verify_curriculum(stages)
    assert not result.stable
    assert any(f.kind is DriftKind.TEMPLATE_DRIFT for f in result.findings)


def test_special_token_drift_detected() -> None:
    stages = (
        _stage("sft"),
        _stage("domain", special_tokens=frozenset({"<|im_start|>"})),
    )
    result = verify_curriculum(stages)
    assert any(f.kind is DriftKind.SPECIAL_TOKEN_DRIFT for f in result.findings)


def test_role_set_drift_detected() -> None:
    stages = (_stage("sft"), _stage("domain", roles=frozenset({"system", "user"})))
    result = verify_curriculum(stages)
    assert any(f.kind is DriftKind.ROLE_SET_DRIFT for f in result.findings)


def test_bos_eos_drift_detected() -> None:
    stages = (_stage("sft"), _stage("domain", eos="<|endoftext|>"))
    result = verify_curriculum(stages)
    assert any(f.kind is DriftKind.BOS_EOS_DRIFT for f in result.findings)


def test_single_stage_is_trivially_stable() -> None:
    assert verify_curriculum((_stage("only"),)).stable


def test_render_text_smoke() -> None:
    result = verify_curriculum((_stage("a"), _stage("b")))
    assert "curriculum" in render_curriculum_text(result)
