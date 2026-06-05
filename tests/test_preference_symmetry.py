"""Tests for preference-pair role symmetry (step 265)."""

from __future__ import annotations

from promptabi.preference_symmetry import (
    PreferencePair,
    SymmetryFindingKind,
    Turn,
    render_symmetry_text,
    verify_preference_pair,
)


def test_symmetric_pair() -> None:
    pair = PreferencePair(
        chosen=(Turn("system", "s"), Turn("user", "u"), Turn("assistant", "good")),
        rejected=(Turn("system", "s"), Turn("user", "u"), Turn("assistant", "bad")),
    )
    assert verify_preference_pair(pair).symmetric


def test_prompt_mismatch() -> None:
    pair = PreferencePair(
        chosen=(Turn("user", "u1"), Turn("assistant", "a")),
        rejected=(Turn("user", "u2"), Turn("assistant", "b")),
    )
    result = verify_preference_pair(pair)
    assert any(f.kind is SymmetryFindingKind.PROMPT_MISMATCH for f in result.findings)


def test_role_sequence_mismatch() -> None:
    pair = PreferencePair(
        chosen=(Turn("user", "u"), Turn("assistant", "a")),
        rejected=(
            Turn("system", "s"),
            Turn("user", "u"),
            Turn("assistant", "b"),
        ),
    )
    result = verify_preference_pair(pair)
    assert any(
        f.kind is SymmetryFindingKind.ROLE_SEQUENCE_MISMATCH for f in result.findings
    )


def test_final_role_not_assistant() -> None:
    pair = PreferencePair(
        chosen=(Turn("user", "u"), Turn("user", "x")),
        rejected=(Turn("user", "u"), Turn("user", "y")),
    )
    result = verify_preference_pair(pair)
    assert any(
        f.kind is SymmetryFindingKind.FINAL_ROLE_NOT_ASSISTANT for f in result.findings
    )


def test_identical_responses() -> None:
    pair = PreferencePair(
        chosen=(Turn("user", "u"), Turn("assistant", "same")),
        rejected=(Turn("user", "u"), Turn("assistant", "same")),
    )
    result = verify_preference_pair(pair)
    assert any(
        f.kind is SymmetryFindingKind.RESPONSES_IDENTICAL for f in result.findings
    )


def test_render_text_smoke() -> None:
    pair = PreferencePair(
        chosen=(Turn("user", "u"), Turn("assistant", "a")),
        rejected=(Turn("user", "u"), Turn("assistant", "b")),
    )
    assert "symmetry" in render_symmetry_text(verify_preference_pair(pair))
