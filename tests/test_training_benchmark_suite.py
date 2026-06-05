"""Tests for training-contract benchmark suites (step 279)."""

from __future__ import annotations

from promptabi.training_benchmark_suite import (
    Scenario,
    default_training_suite,
    render_benchmark_text,
    run_suite,
)


def test_default_suite_passes_completely() -> None:
    report = run_suite(default_training_suite())
    assert report.total == 10
    assert report.correct == 10
    assert report.accuracy == 1.0


def test_suite_detects_a_wrong_expectation() -> None:
    # A scenario whose expected verdict is wrong should be scored as incorrect.
    bogus = Scenario("bogus", "x", lambda: True, expected_pass=False)
    report = run_suite((bogus,))
    assert report.correct == 0


def test_outcomes_carry_categories() -> None:
    report = run_suite(default_training_suite())
    categories = {o.category for o in report.outcomes}
    assert {"loss-mask", "target-span", "preference", "transforms", "leakage"} <= categories


def test_render_text_smoke() -> None:
    report = run_suite(default_training_suite())
    text = render_benchmark_text(report)
    assert "benchmark" in text
    assert "10/10" in text
