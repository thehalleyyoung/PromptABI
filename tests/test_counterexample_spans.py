"""Tests for connecting SMT counterexamples to source spans (step 234)."""

from __future__ import annotations

import json

import pytest

from promptabi.diagnostics import ArtifactRef, SourceSpan
from promptabi.formal import (
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    SolverBackend,
    SolverResult,
    SolverStatus,
    Value,
    Var,
)
from promptabi.counterexample_spans import (
    CounterexampleSpanError,
    VariableProvenance,
    annotate_counterexample,
    render_annotated_counterexample_json,
    render_annotated_counterexample_text,
)


def _sat_result() -> SolverResult:
    return SolverResult(
        status=SolverStatus.SAT,
        backend=SolverBackend.Z3,
        assignment={"context": 900, "reserved_completion": 256},
    )


def _provenance() -> list[VariableProvenance]:
    artifact = ArtifactRef(kind="token-budget", name="overflow-pack", path="budget.json")
    return [
        VariableProvenance(
            variable="context",
            artifact=artifact,
            span=SourceSpan(path="budget.json", start_line=3, start_column=5),
            snippet='"context": {"max": 900}',
        ),
        VariableProvenance(
            variable="reserved_completion",
            artifact=artifact,
            span=SourceSpan(path="budget.json", start_line=4, start_column=5),
        ),
    ]


def test_annotates_each_variable_with_span() -> None:
    annotated = annotate_counterexample(_sat_result(), _provenance())
    assert annotated.fully_located
    locations = {item.variable: item.location() for item in annotated.located}
    assert locations["context"] == "budget.json:3:5"
    assert locations["reserved_completion"] == "budget.json:4:5"


def test_unmapped_variables_are_reported() -> None:
    annotated = annotate_counterexample(_sat_result(), _provenance()[:1])
    assert not annotated.fully_located
    assert annotated.unmapped == ("reserved_completion",)


def test_mapping_input_is_accepted() -> None:
    provenance = {record.variable: record for record in _provenance()}
    annotated = annotate_counterexample(_sat_result(), provenance)
    assert annotated.fully_located


def test_non_sat_result_raises() -> None:
    result = SolverResult(status=SolverStatus.UNSAT, backend=SolverBackend.Z3)
    with pytest.raises(CounterexampleSpanError):
        annotate_counterexample(result, _provenance())


def test_end_to_end_with_real_solver() -> None:
    problem = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-big", expression=Le(Value(3), Var("x"))),),
        name="demo",
    )
    result = problem.solve()
    provenance = [
        VariableProvenance(
            variable="x",
            artifact=ArtifactRef(kind="demo", name="demo"),
            span=SourceSpan(path="demo.json", start_line=1, start_column=1),
        )
    ]
    annotated = annotate_counterexample(result, provenance)
    assert annotated.fully_located
    assert annotated.located[0].variable == "x"


def test_render_round_trips() -> None:
    annotated = annotate_counterexample(_sat_result(), _provenance())
    payload = json.loads(render_annotated_counterexample_json(annotated))
    assert payload["fully_located"] is True
    assert "source map" in render_annotated_counterexample_text(annotated)
