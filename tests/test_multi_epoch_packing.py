"""Tests for multi-epoch packing proofs (step 261)."""

from __future__ import annotations

from promptabi.multi_epoch_packing import (
    PackedSequence,
    PackingViolationKind,
    prove_packing,
    render_packing_text,
)

SEP = 2

# Two documents packed: doc 0 = tokens [10,11, SEP], doc 1 = [12,13, SEP], pad.
GOOD = PackedSequence(
    token_ids=(10, 11, SEP, 12, 13, SEP, 0),
    doc_ids=(0, 0, 0, 1, 1, 1, -1),
    attention_resets=frozenset({0, 3}),
    loss_mask=(1, 1, 0, 1, 1, 0, 0),
    separator_id=SEP,
)


def test_good_packing_is_sound() -> None:
    proof = prove_packing(GOOD, epochs=3)
    assert proof.sound, proof.violations
    assert proof.epochs == 3


def test_missing_separator_detected() -> None:
    bad = PackedSequence(
        token_ids=(10, 11, 12, 13),
        doc_ids=(0, 0, 1, 1),
        attention_resets=frozenset({0, 2}),
        loss_mask=(1, 1, 1, 1),
        separator_id=SEP,
    )
    proof = prove_packing(bad)
    assert not proof.sound
    assert any(
        v.kind is PackingViolationKind.MISSING_SEPARATOR for v in proof.violations
    )


def test_attention_bleed_detected() -> None:
    bad = PackedSequence(
        token_ids=(10, SEP, 12, 13),
        doc_ids=(0, 0, 1, 1),
        attention_resets=frozenset({0}),  # missing reset at pos 2
        loss_mask=(1, 0, 1, 1),
        separator_id=SEP,
    )
    proof = prove_packing(bad)
    assert any(
        v.kind is PackingViolationKind.ATTENTION_BLEED for v in proof.violations
    )


def test_loss_on_padding_detected() -> None:
    bad = PackedSequence(
        token_ids=(10, 11, SEP, 0),
        doc_ids=(0, 0, 0, -1),
        attention_resets=frozenset({0}),
        loss_mask=(1, 1, 0, 1),  # supervises padding
        separator_id=SEP,
    )
    proof = prove_packing(bad)
    assert any(
        v.kind is PackingViolationKind.LOSS_ON_PADDING for v in proof.violations
    )


def test_loss_across_document_detected() -> None:
    bad = PackedSequence(
        token_ids=(10, SEP, 12),
        doc_ids=(0, 0, 1),
        attention_resets=frozenset({0, 2}),
        loss_mask=(1, 1, 1),  # supervises the bridging separator
        separator_id=SEP,
    )
    proof = prove_packing(bad)
    assert any(
        v.kind is PackingViolationKind.LOSS_ACROSS_DOCUMENT for v in proof.violations
    )


def test_length_mismatch_detected() -> None:
    bad = PackedSequence(
        token_ids=(10, 11),
        doc_ids=(0,),
        attention_resets=frozenset({0}),
        loss_mask=(1, 1),
        separator_id=SEP,
    )
    proof = prove_packing(bad)
    assert any(
        v.kind is PackingViolationKind.LENGTH_MISMATCH for v in proof.violations
    )


def test_render_text_smoke() -> None:
    assert "packing" in render_packing_text(prove_packing(GOOD))
