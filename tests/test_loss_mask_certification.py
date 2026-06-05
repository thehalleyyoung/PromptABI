"""Tests for loss-mask certification across loaders (step 263)."""

from __future__ import annotations

from promptabi.loss_mask_certification import (
    LoaderMask,
    LossMaskFindingKind,
    TargetSpec,
    certify_loss_masks,
    render_loss_mask_text,
)

SPEC = TargetSpec(length=5, target_positions=frozenset({3, 4}))


def test_agreeing_correct_loaders() -> None:
    loaders = (
        LoaderMask("trl", (0, 0, 0, 1, 1)),
        LoaderMask("custom", (0, 0, 0, 1, 1)),
    )
    assert certify_loss_masks(SPEC, loaders).certified


def test_supervising_prompt_rejected() -> None:
    loaders = (LoaderMask("bad", (1, 0, 0, 1, 1)),)
    result = certify_loss_masks(SPEC, loaders)
    assert any(
        f.kind is LossMaskFindingKind.SUPERVISES_PROMPT for f in result.findings
    )


def test_dropping_target_rejected() -> None:
    loaders = (LoaderMask("bad", (0, 0, 0, 1, 0)),)
    result = certify_loss_masks(SPEC, loaders)
    assert any(f.kind is LossMaskFindingKind.DROPS_TARGET for f in result.findings)


def test_length_mismatch() -> None:
    loaders = (LoaderMask("bad", (0, 1)),)
    result = certify_loss_masks(SPEC, loaders)
    assert any(f.kind is LossMaskFindingKind.LENGTH_MISMATCH for f in result.findings)


def test_loader_disagreement() -> None:
    loaders = (
        LoaderMask("a", (0, 0, 0, 1, 1)),
        LoaderMask("b", (0, 0, 1, 1, 1)),
    )
    result = certify_loss_masks(SPEC, loaders)
    assert any(
        f.kind is LossMaskFindingKind.LOADER_DISAGREEMENT for f in result.findings
    )


def test_render_text_smoke() -> None:
    result = certify_loss_masks(SPEC, (LoaderMask("a", (0, 0, 0, 1, 1)),))
    assert "loss-mask" in render_loss_mask_text(result)
