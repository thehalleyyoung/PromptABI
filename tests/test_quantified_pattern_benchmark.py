"""Tests for quantified-pattern approximation benchmarks (step 228)."""

from __future__ import annotations

import json

import pytest

from promptabi.quantified_pattern_benchmark import (
    QuantifiedPattern,
    benchmark_quantified_patterns,
    first_k_strategy,
    render_quantified_pattern_json,
    render_quantified_pattern_text,
)


def test_exact_expansion_detects_violation() -> None:
    # last index violates the cap; an all-index sample must find it.
    pattern = QuantifiedPattern(name="hot-tail", domains=(5, 5, 100), cap=10)
    bench = benchmark_quantified_patterns([pattern], first_k_strategy(3))
    (result,) = bench.results
    assert result.exact_verdict == "violated"
    assert result.approx_verdict == "violated"
    assert result.sound
    assert result.precise


def test_sampling_that_misses_hot_index_is_unsound() -> None:
    # only index 2 can exceed the cap; sampling the first 2 indices misses it.
    pattern = QuantifiedPattern(name="hot-tail", domains=(5, 5, 100), cap=10)
    bench = benchmark_quantified_patterns([pattern], first_k_strategy(2))
    (result,) = bench.results
    assert result.exact_verdict == "violated"
    assert result.approx_verdict == "holds"
    assert result.sound is False
    assert bench.all_sound is False
    assert len(bench.unsound) == 1


def test_obligation_that_holds_is_precise_under_sampling() -> None:
    # no index can exceed the cap -> both verdicts "holds".
    pattern = QuantifiedPattern(name="all-small", domains=(3, 4, 5), cap=10)
    bench = benchmark_quantified_patterns([pattern], first_k_strategy(1))
    (result,) = bench.results
    assert result.exact_verdict == "holds"
    assert result.approx_verdict == "holds"
    assert result.sound
    assert result.precise


def test_precision_rate_and_speedup_reported() -> None:
    patterns = [
        QuantifiedPattern(name="p1", domains=(3, 4, 5), cap=10),
        QuantifiedPattern(name="p2", domains=(5, 5, 100), cap=10),
    ]
    bench = benchmark_quantified_patterns(patterns, first_k_strategy(2))
    assert 0.0 <= bench.precision_rate <= 1.0
    payload = json.loads(render_quantified_pattern_json(bench))
    assert payload["unsound_count"] == 1
    assert "quantified-pattern benchmark" in render_quantified_pattern_text(bench)


def test_empty_domains_rejected() -> None:
    with pytest.raises(ValueError):
        QuantifiedPattern(name="bad", domains=(), cap=1)


def test_first_k_strategy_requires_positive_k() -> None:
    with pytest.raises(ValueError):
        first_k_strategy(0)
