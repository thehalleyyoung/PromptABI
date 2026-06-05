"""Tests for the solver-lemma cache over normalized artifact products (step 224)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.token_budget_arithmetic import load_token_budget_contract
from promptabi.solver_lemma_cache import (
    SolverLemmaCacheFindingKind,
    canonical_artifact_hash,
    normalize_artifact_product,
    render_solver_lemma_cache_json,
    token_budget_lemma_suite,
    verify_solver_lemma_cache,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "token-budget-arithmetic"


def test_canonical_hash_is_formatting_invariant() -> None:
    a = {"b": 1, "a": [1, 2, 3]}
    b = {"a": [1, 2, 3], "b": 1}
    assert canonical_artifact_hash(a) == canonical_artifact_hash(b)


def test_product_hash_is_order_independent() -> None:
    x = {"name": "x"}
    y = {"name": "y"}
    assert normalize_artifact_product([x, y]) == normalize_artifact_product([y, x])
    assert normalize_artifact_product([x]) != normalize_artifact_product([x, y])


def test_token_budget_suite_is_sound_and_reused() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "chat-pack.json"))
    report = verify_solver_lemma_cache(token_budget_lemma_suite(contract))
    assert report.ok
    assert report.soundness_ok
    # the reordered/reformatted equivalents must be served from the cache
    assert report.hits >= 1
    assert report.reuse_demonstrated
    assert any(record.cache_hit for record in report.records)


def test_distinct_lemmas_report_no_reuse() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "chat-pack.json"))
    # only the base guarantees (no equivalent duplicates) -> all misses
    base = tuple(lemma for lemma in token_budget_lemma_suite(contract) if "reordered" not in lemma.name)
    report = verify_solver_lemma_cache(base, require_reuse=True)
    assert report.hits == 0
    assert any(f.kind is SolverLemmaCacheFindingKind.NO_REUSE for f in report.findings)
    # but soundness still holds and verdicts are computed
    assert report.soundness_ok


def test_overflow_contract_lemmas_remain_refuted_under_cache() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "overflow-pack.json"))
    report = verify_solver_lemma_cache(token_budget_lemma_suite(contract))
    assert report.soundness_ok
    assert all(record.status == "sat" for record in report.records)


def test_render_json_round_trips() -> None:
    contract = load_token_budget_contract(str(EXAMPLES / "chat-pack.json"))
    report = verify_solver_lemma_cache(token_budget_lemma_suite(contract))
    payload = json.loads(render_solver_lemma_cache_json(report))
    assert payload["soundness_ok"] is True
    assert payload["reuse_demonstrated"] is True


def test_cli_solver_cache(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["solver-cache", "--contract", str(EXAMPLES / "chat-pack.json"), "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["hits"] >= 1
