"""Tests for proof-carrying solver cache entries (step 232)."""

from __future__ import annotations

import json
from pathlib import Path

from promptabi.formal import (
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.proof_carrying_cache import (
    ProofCarryingCache,
    ProofCarryingEntry,
    ProofKind,
    build_proof,
    validate_proof,
)
from promptabi.proof_carrying_cache import render_proof_carrying_cache_json  # noqa: F401
from promptabi.token_budget_arithmetic import (
    compile_token_budget_problem,
    load_token_budget_contract,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "token-budget-arithmetic"


def _sat_problem() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="sat",
    )


def _unsat_problem() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(
            NamedConstraint(name="x-ge-4", expression=Le(Value(4), Var("x"))),
            NamedConstraint(name="x-le-1", expression=Le(Var("x"), Value(1))),
        ),
        name="unsat",
    )


def test_sat_proof_is_a_model() -> None:
    problem = _sat_problem()
    result = problem.solve()
    entry = build_proof(result, "k")
    assert entry.proof_kind is ProofKind.MODEL
    assert validate_proof(problem, entry)


def test_unsat_proof_is_an_unsat_core() -> None:
    problem = _unsat_problem()
    result = problem.solve()
    entry = build_proof(result, "k")
    assert entry.proof_kind is ProofKind.UNSAT_CORE
    assert validate_proof(problem, entry)


def test_cache_hit_is_validated() -> None:
    cache = ProofCarryingCache()
    problem = _sat_problem()
    _, hit1 = cache.solve(problem)
    assert hit1 is False
    _, hit2 = cache.solve(problem)
    assert hit2 is True
    assert cache.hits == 1
    assert cache.misses == 1


def test_tampered_model_is_rejected() -> None:
    cache = ProofCarryingCache()
    problem = _sat_problem()
    result = problem.solve()
    # inject a bogus model that violates x <= 2
    bogus = ProofCarryingEntry(
        cache_key="ignored",
        status="sat",
        proof_kind=ProofKind.MODEL,
        model={"x": 5},
    )
    cache.inject(problem, bogus, result)
    _, hit = cache.solve(problem)
    assert hit is False  # rejected -> fresh solve
    assert cache.rejected == 1


def test_tampered_unsat_core_is_rejected() -> None:
    cache = ProofCarryingCache()
    problem = _unsat_problem()
    result = problem.solve()
    # a single-constraint "core" that is not actually unsat on its own
    bogus = ProofCarryingEntry(
        cache_key="ignored",
        status="unsat",
        proof_kind=ProofKind.UNSAT_CORE,
        unsat_core=("x-ge-4",),
    )
    cache.inject(problem, bogus, result)
    _, hit = cache.solve(problem)
    assert hit is False
    assert cache.rejected == 1


def test_real_token_budget_proof_validates() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "chat-pack.json"))
    problem = compile_token_budget_problem(contract, contract.guarantees[0])
    result = problem.solve()
    entry = build_proof(result, "k")
    assert validate_proof(problem, entry)


def test_render_json() -> None:
    cache = ProofCarryingCache()
    cache.solve(_sat_problem())
    payload = json.loads(json.dumps(cache.to_dict()))
    assert payload["schema"].startswith("promptabi.proof-carrying-cache")
