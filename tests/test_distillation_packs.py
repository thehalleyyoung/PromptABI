"""Tests for distillation prompt-pack verification (step 273)."""

from __future__ import annotations

from promptabi.distillation_packs import (
    DistillationFindingKind,
    DistillationPack,
    ModelInterface,
    render_distillation_text,
    verify_distillation,
)

TEACHER = ModelInterface(
    name="teacher",
    prompt_digest="d1",
    roles=frozenset({"system", "user", "assistant"}),
    stop_sequences=frozenset({"<|im_end|>"}),
    tokenizer="tok-a",
)
STUDENT = ModelInterface(
    name="student",
    prompt_digest="d1",
    roles=frozenset({"system", "user", "assistant"}),
    stop_sequences=frozenset({"<|im_end|>", "</s>"}),
    tokenizer="tok-b",
)


def test_compatible_distillation() -> None:
    assert verify_distillation(DistillationPack(TEACHER, STUDENT)).compatible


def test_prompt_digest_mismatch() -> None:
    student = ModelInterface("s", "d2", STUDENT.roles, STUDENT.stop_sequences, "tok")
    result = verify_distillation(DistillationPack(TEACHER, student))
    assert any(
        f.kind is DistillationFindingKind.PROMPT_DIGEST_MISMATCH for f in result.findings
    )


def test_stop_policy_too_weak() -> None:
    student = ModelInterface(
        "s", "d1", STUDENT.roles, frozenset({"</s>"}), "tok"
    )
    result = verify_distillation(DistillationPack(TEACHER, student))
    assert any(
        f.kind is DistillationFindingKind.STOP_POLICY_TOO_WEAK for f in result.findings
    )


def test_tokenizer_mismatch_when_required() -> None:
    result = verify_distillation(
        DistillationPack(TEACHER, STUDENT, require_same_tokenizer=True)
    )
    assert any(
        f.kind is DistillationFindingKind.TOKENIZER_MISMATCH for f in result.findings
    )


def test_render_text_smoke() -> None:
    result = verify_distillation(DistillationPack(TEACHER, STUDENT))
    assert "distillation" in render_distillation_text(result)
