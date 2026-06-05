"""Tests for finite unicode normalization constraints (step 237)."""

from __future__ import annotations

import json

import pytest

from promptabi.unicode_normalization import (
    NormalizationFindingKind,
    analyze_normalization,
    render_normalization_json,
    render_normalization_text,
)


def test_ascii_alphabet_is_fully_normalized() -> None:
    report = analyze_normalization(["a", "b", "1"], form="NFC")
    assert report.sound
    assert len(report.normalized_texts) == 3
    assert all(c.is_normalized for c in report.characters)


def test_detects_non_normalized_character() -> None:
    # 'A' + combining ring above is not in NFC.
    report = analyze_normalization(["A\u030a"], form="NFC")
    assert any(
        f.kind is NormalizationFindingKind.NOT_NORMALIZED for f in report.findings
    )
    assert len(report.normalized_texts) == 0


def test_detects_confusables_under_nfkc() -> None:
    # fullwidth 'A' and ascii 'A' collapse under NFKC.
    report = analyze_normalization(["A", "\uff21"], form="NFKC")
    assert report.confusable_classes
    assert any(f.kind is NormalizationFindingKind.CONFUSABLE for f in report.findings)


def test_ligature_collapses_under_nfkc() -> None:
    report = analyze_normalization(["\ufb01"], form="NFKC")  # 'fi' ligature
    not_norm = [
        f for f in report.findings if f.kind is NormalizationFindingKind.NOT_NORMALIZED
    ]
    assert not_norm


def test_solver_agreement_holds() -> None:
    report = analyze_normalization(["a", "A\u030a", "1", "\uff21"], form="NFC")
    assert report.solver_agrees
    assert not any(
        f.kind is NormalizationFindingKind.SOLVER_DISAGREEMENT for f in report.findings
    )
    assert report.idempotent


def test_invalid_form_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_normalization(["a"], form="NFX")


def test_empty_alphabet_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_normalization([])


def test_empty_entry_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_normalization([""])


def test_render_round_trips() -> None:
    report = analyze_normalization(["a", "\uff21"], form="NFKC")
    payload = json.loads(render_normalization_json(report))
    assert payload["form"] == "NFKC"
    assert "normalization" in render_normalization_text(report)
