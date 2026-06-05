"""Tests for solver-fragment conformance suites (step 236)."""

from __future__ import annotations

import json

from promptabi.formal import (
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.solver_fragment_conformance import (
    ConformanceCase,
    ConformanceFindingKind,
    ExpectedVerdict,
    render_conformance_json,
    render_conformance_text,
    run_fragment_conformance_suite,
    standard_fragment_conformance_suite,
)


def test_standard_suite_is_conformant() -> None:
    report = run_fragment_conformance_suite()
    assert report.conformant
    assert report.pass_rate == 1.0
    assert len(report.results) == len(standard_fragment_conformance_suite())


def test_standard_suite_covers_all_verdicts() -> None:
    report = run_fragment_conformance_suite()
    verdicts = {result.actual_verdict for result in report.results}
    assert {"sat", "unsat", "abstain"} <= verdicts


def test_unsupported_case_is_out_of_fragment() -> None:
    report = run_fragment_conformance_suite()
    abstain = next(r for r in report.results if r.name == "abstain-unsupported")
    assert abstain.actual_in_fragment is False
    assert abstain.actual_verdict == "abstain"


def test_mislabeled_case_produces_verdict_mismatch() -> None:
    sat = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="sat",
    )
    bad = ConformanceCase("bad", sat, ExpectedVerdict.UNSAT, True)
    report = run_fragment_conformance_suite([bad])
    assert not report.conformant
    assert any(
        f.kind is ConformanceFindingKind.VERDICT_MISMATCH for f in report.findings
    )
    assert report.pass_rate == 0.0


def test_mislabeled_fragment_produces_fragment_mismatch() -> None:
    sat = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="sat",
    )
    bad = ConformanceCase("bad", sat, ExpectedVerdict.SAT, False)
    report = run_fragment_conformance_suite([bad])
    assert any(
        f.kind is ConformanceFindingKind.FRAGMENT_MISMATCH for f in report.findings
    )


def test_render_round_trips() -> None:
    report = run_fragment_conformance_suite()
    payload = json.loads(render_conformance_json(report))
    assert payload["conformant"] is True
    assert "conformance" in render_conformance_text(report)
