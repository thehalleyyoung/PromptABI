"""Tests for solver portfolio replay metadata (step 225)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.token_budget_arithmetic import load_token_budget_contract
from promptabi.solver_portfolio_replay import (
    SolverPortfolioFindingKind,
    SolverStrategy,
    default_portfolio,
    render_solver_portfolio_json,
    replay_solver_portfolio,
    run_solver_portfolio,
    token_budget_portfolio_problems,
    verify_solver_portfolio,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "token-budget-arithmetic"


def _problems(name: str):
    contract = load_token_budget_contract(str(EXAMPLES / name))
    return token_budget_portfolio_problems(contract)


def test_default_portfolio_runs_both_backends() -> None:
    name, problem = _problems("chat-pack.json")[0]
    record = run_solver_portfolio(problem)
    strategies = {attempt.strategy for attempt in record.attempts}
    assert strategies == {"z3-smt", "finite-enumeration"}
    # every guarantee in chat-pack provably holds -> unsat negation
    assert record.metadata.winning_status == "unsat"
    assert record.agreement_ok


def test_portfolio_strategies_agree_on_conclusion() -> None:
    report = verify_solver_portfolio(_problems("chat-pack.json"))
    assert report.ok
    for record in report.records:
        conclusive = [a for a in record.attempts if a.conclusive]
        assert len({a.conclusion for a in conclusive}) == 1


def test_overflow_contract_is_counterexample_in_portfolio() -> None:
    report = verify_solver_portfolio(_problems("overflow-pack.json"))
    assert report.ok
    assert all(r.metadata.winning_status == "sat" for r in report.records)


def test_replay_reproduces_recorded_verdict() -> None:
    name, problem = _problems("chat-pack.json")[0]
    record = run_solver_portfolio(problem)
    assert replay_solver_portfolio(problem, record.metadata)


def test_replay_metadata_carries_query_fingerprint() -> None:
    name, problem = _problems("chat-pack.json")[0]
    record = run_solver_portfolio(problem)
    meta = record.metadata
    assert meta.query_key
    fingerprint_names = {n for n, _ in meta.solver_version_fingerprints}
    assert "promptabi-finite-contract-solver" in fingerprint_names


def test_enumeration_only_portfolio_is_conclusive() -> None:
    # The overflow contract is satisfiable, so finite enumeration finds a
    # counterexample quickly without exhausting the domain.
    problems = _problems("overflow-pack.json")
    report = verify_solver_portfolio(
        problems, portfolio=(SolverStrategy(name="enum", prefer_z3=False),)
    )
    assert report.ok
    assert all(r.metadata.winning_strategy == "enum" for r in report.records)


def test_no_conclusive_strategy_flagged() -> None:
    # An empty portfolio is rejected outright.
    name, problem = _problems("chat-pack.json")[0]
    with pytest.raises(ValueError):
        run_solver_portfolio(problem, portfolio=())


def test_render_json_round_trips() -> None:
    report = verify_solver_portfolio(_problems("chat-pack.json"))
    payload = json.loads(render_solver_portfolio_json(report))
    assert payload["ok"] is True
    assert payload["records"]


def test_cli_solver_portfolio(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["solver-portfolio", "--contract", str(EXAMPLES / "chat-pack.json"), "--format", "json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_default_portfolio_has_smt_first() -> None:
    portfolio = default_portfolio()
    assert portfolio[0].prefer_z3 is True
    assert portfolio[1].prefer_z3 is False
