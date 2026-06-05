"""Tests for benchmark-answer leakage detection (step 266)."""

from __future__ import annotations

from promptabi.benchmark_leakage import (
    BenchmarkItem,
    LeakageKind,
    SyntheticExample,
    detect_leakage,
    jaccard,
    render_leakage_text,
)

BENCH = (
    BenchmarkItem("mmlu-1", "what is the boiling point of water at sea level"),
)


def test_clean_data() -> None:
    synth = (SyntheticExample("s1", "the cat sat on a warm windowsill all day"),)
    assert detect_leakage(synth, BENCH).clean


def test_exact_match_detected() -> None:
    synth = (
        SyntheticExample(
            "s1", "answer: what is the boiling point of water at sea level"
        ),
    )
    report = detect_leakage(synth, BENCH)
    assert not report.clean
    assert any(f.kind is LeakageKind.EXACT_MATCH for f in report.findings)


def test_ngram_overlap_detected() -> None:
    synth = (
        SyntheticExample(
            "s1", "what is the boiling point of water at high altitude"
        ),
    )
    report = detect_leakage(synth, BENCH, ngram=4, threshold=0.4)
    assert any(f.kind is LeakageKind.NGRAM_OVERLAP for f in report.findings)


def test_jaccard_bounds() -> None:
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0


def test_render_text_smoke() -> None:
    assert "leakage" in render_leakage_text(detect_leakage((), BENCH))
