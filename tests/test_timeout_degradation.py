"""Tests for timeout-sensitive degradation (step 235)."""

from __future__ import annotations

import json

import pytest

from promptabi.formal import (
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Sum,
    Value,
    Var,
)
from promptabi.timeout_degradation import (
    DegradationFindingKind,
    profile_timeout_degradation,
    render_degradation_json,
    render_degradation_text,
)


def _small_problem() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=3),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(1))),),
        name="small",
    )


def _wide_problem() -> FiniteContractProblem:
    # 4 wide integer variables -> enumeration is slow; a tiny timeout degrades.
    variables = tuple(
        IntRangeDomain(name=f"v{i}", minimum=0, maximum=200) for i in range(4)
    )
    # provably unsatisfiable negation: sum <= 10 forced but each >= 50 impossible
    constraints = (
        NamedConstraint(name="lo", expression=Le(Value(50), Var("v0"))),
        NamedConstraint(
            name="sum-small",
            expression=Le(Sum(Var("v0"), Var("v1"), Var("v2"), Var("v3")), Value(10)),
        ),
    )
    return FiniteContractProblem(variables=variables, constraints=constraints, name="wide")


def test_small_problem_does_not_degrade() -> None:
    report = profile_timeout_degradation(_small_problem(), [0.5, 1.0])
    assert report.sound
    assert report.degraded is False
    assert report.ground_truth_status == "sat"


def test_wide_problem_degrades_soundly_under_tiny_timeout() -> None:
    report = profile_timeout_degradation(_wide_problem(), [1e-6])
    assert report.sound  # timing out is inconclusive, never wrong
    # every conclusive observation matches ground truth
    for obs in report.observations:
        if obs.conclusive:
            assert obs.status == report.ground_truth_status


def test_conclusive_verdicts_match_ground_truth() -> None:
    report = profile_timeout_degradation(_small_problem(), [0.25, 0.5, 1.0])
    assert all(
        obs.status == report.ground_truth_status
        for obs in report.observations
        if obs.conclusive
    )
    assert not any(
        f.kind is DegradationFindingKind.UNSOUND_UNDER_TIMEOUT for f in report.findings
    )


def test_empty_timeouts_rejected() -> None:
    with pytest.raises(ValueError):
        profile_timeout_degradation(_small_problem(), [])


def test_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        profile_timeout_degradation(_small_problem(), [-1.0])


def test_render_round_trips() -> None:
    report = profile_timeout_degradation(_small_problem(), [0.5])
    payload = json.loads(render_degradation_json(report))
    assert payload["sound"] is True
    assert "timeout degradation" in render_degradation_text(report)
