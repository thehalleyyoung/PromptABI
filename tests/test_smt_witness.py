"""Tests for minimized human-readable SMT witnesses (step 226)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.formal import (
    BoolDomain,
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.smt_witness import (
    SmtWitnessError,
    minimize_smt_model,
    render_smt_witness_json,
    render_smt_witness_text,
)
from promptabi.token_budget_arithmetic import (
    compile_token_budget_problem,
    load_token_budget_contract,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "token-budget-arithmetic"


def _problem_with_noise() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(
            IntRangeDomain(name="x", minimum=0, maximum=5),
            IntRangeDomain(name="y", minimum=0, maximum=5),
            BoolDomain(name="flag"),
        ),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="noise-demo",
    )


def test_drops_structurally_irrelevant_variables() -> None:
    witness = minimize_smt_model(_problem_with_noise())
    names = {v.name for v in witness.relevant}
    assert names == {"x"}
    assert set(witness.omitted) == {"y", "flag"}
    assert witness.variables_dropped == 2


def test_flexible_values_reported_for_relevant_variable() -> None:
    witness = minimize_smt_model(_problem_with_noise())
    (x,) = witness.relevant
    assert set(x.flexible_values) == {0, 1, 2}
    assert x.pinned is False
    assert x.constraints == ("x-small",)


def test_narrative_is_human_readable() -> None:
    witness = minimize_smt_model(_problem_with_noise())
    text = "\n".join(witness.narrative)
    assert "noise-demo" in text
    assert "irrelevant" in text
    assert "x" in text


def test_dropped_variables_are_sound_dont_cares() -> None:
    problem = _problem_with_noise()
    witness = minimize_smt_model(problem)
    # Any value of a dropped variable still satisfies the model.
    index = {c.name for c in problem.constraints}
    assert index  # constraints exist
    for name in witness.omitted:
        # dropped variable appears in no constraint expression
        assert all(name not in str(c.expression.to_dict()) for c in problem.constraints)


def test_pinned_variable_when_no_flexibility() -> None:
    problem = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-eq-3", expression=Le(Value(3), Var("x"))),),
        name="pin",
    )
    # x >= 3 has flexible values {3,4,5}; force exact pin with two-sided bound.
    problem = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=3, maximum=3),),
        constraints=(NamedConstraint(name="x-eq-3", expression=Le(Value(3), Var("x"))),),
        name="pin",
    )
    witness = minimize_smt_model(problem)
    (x,) = witness.relevant
    assert x.pinned is True
    assert x.value == 3


def test_raises_when_no_counterexample() -> None:
    # A guarantee that provably holds -> negation is UNSAT -> no model.
    contract = load_token_budget_contract(str(EXAMPLES / "chat-pack.json"))
    problem = compile_token_budget_problem(contract, contract.guarantees[0])
    with pytest.raises(SmtWitnessError):
        minimize_smt_model(problem)


def test_real_overflow_contract_witness() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "overflow-pack.json"))
    problem = compile_token_budget_problem(contract, contract.guarantees[0])
    witness = minimize_smt_model(problem)
    assert witness.relevant  # at least one variable participates in the overflow


def test_render_round_trips() -> None:
    witness = minimize_smt_model(_problem_with_noise())
    payload = json.loads(render_smt_witness_json(witness))
    assert payload["variables_dropped"] == 2
    assert "minimized SMT witness" in render_smt_witness_text(witness)
