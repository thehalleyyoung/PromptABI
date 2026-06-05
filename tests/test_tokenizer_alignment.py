"""Tests for train/eval tokenizer alignment proofs (step 267)."""

from __future__ import annotations

from promptabi.tokenizer_alignment import (
    AlignmentFindingKind,
    TokenizerFingerprint,
    prove_alignment,
    render_alignment_text,
)


def _fp(release: str, **overrides) -> TokenizerFingerprint:
    base = dict(
        release=release,
        vocab_digest="v1",
        added_tokens=("<|im_start|>", "<|im_end|>"),
        special_tokens={"bos": "<s>", "eos": "</s>"},
        add_bos=True,
        add_eos=False,
        probe_encodings={"hello": (1, 2), "world": (3,)},
    )
    base.update(overrides)
    return TokenizerFingerprint(**base)  # type: ignore[arg-type]


def test_aligned_tokenizers() -> None:
    assert prove_alignment(_fp("train"), _fp("eval")).aligned


def test_vocab_digest_mismatch() -> None:
    result = prove_alignment(_fp("train"), _fp("eval", vocab_digest="v2"))
    assert any(
        f.kind is AlignmentFindingKind.VOCAB_DIGEST_MISMATCH for f in result.findings
    )


def test_added_token_mismatch() -> None:
    result = prove_alignment(_fp("train"), _fp("eval", added_tokens=("<|im_start|>",)))
    assert any(
        f.kind is AlignmentFindingKind.ADDED_TOKEN_MISMATCH for f in result.findings
    )


def test_bos_eos_flag_mismatch() -> None:
    result = prove_alignment(_fp("train"), _fp("eval", add_eos=True))
    assert any(
        f.kind is AlignmentFindingKind.BOS_EOS_FLAG_MISMATCH for f in result.findings
    )


def test_probe_encoding_mismatch() -> None:
    result = prove_alignment(
        _fp("train"), _fp("eval", probe_encodings={"hello": (9, 9), "world": (3,)})
    )
    assert any(
        f.kind is AlignmentFindingKind.PROBE_ENCODING_MISMATCH for f in result.findings
    )


def test_render_text_smoke() -> None:
    assert "alignment" in render_alignment_text(prove_alignment(_fp("a"), _fp("b")))
