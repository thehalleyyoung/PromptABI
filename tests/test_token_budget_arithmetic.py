"""Tests for token-budget linear arithmetic (step 223)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.token_budget_arithmetic import (
    TokenBudgetProofStatus,
    load_token_budget_contract,
    render_token_budget_report_json,
    render_token_budget_report_text,
    token_budget_contract_from_dict,
    verify_token_budget_contract,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "token-budget-arithmetic"


def test_chat_pack_fits_window() -> None:
    report = verify_token_budget_contract(load_token_budget_contract(str(EXAMPLES / "chat-pack.json")))
    assert report.ok
    assert {r.guarantee for r in report.results} == {"packed-fits-window", "completion-reserved"}
    assert all(r.status is TokenBudgetProofStatus.PROVEN for r in report.results)


def test_overflow_pack_is_refuted_with_overbudget_packing() -> None:
    report = verify_token_budget_contract(load_token_budget_contract(str(EXAMPLES / "overflow-pack.json")))
    assert not report.ok
    result = report.refuted[0]
    assert result.counterexample is not None
    total = result.counterexample["context"] + result.counterexample["reserved_completion"]
    assert total > 1024


def test_coefficients_are_respected() -> None:
    # 3 * messages + 50 overhead must be <= 100; messages in [0, 20] -> refutable
    contract = token_budget_contract_from_dict(
        {
            "name": "weighted",
            "segments": [{"name": "messages", "minimum": 0, "maximum": 20}],
            "guarantees": [
                {
                    "name": "weighted-fits",
                    "left": [{"segment": "messages", "coefficient": 3}, {"const": 50}],
                    "op": "<=",
                    "right": [{"const": 100}],
                }
            ],
        }
    )
    report = verify_token_budget_contract(contract)
    assert not report.ok
    # 3*messages + 50 <= 100 means messages <= 16; messages=17 is the smallest witness
    assert report.results[0].counterexample["messages"] >= 17


def test_assumptions_can_make_a_guarantee_provable() -> None:
    contract = token_budget_contract_from_dict(
        {
            "name": "guarded",
            "segments": [{"name": "messages", "minimum": 0, "maximum": 20}],
            "assumptions": [
                {"name": "small-input", "left": [{"segment": "messages"}], "op": "<=", "right": [{"const": 10}]}
            ],
            "guarantees": [
                {
                    "name": "weighted-fits",
                    "left": [{"segment": "messages", "coefficient": 3}, {"const": 50}],
                    "op": "<=",
                    "right": [{"const": 100}],
                }
            ],
        }
    )
    assert verify_token_budget_contract(contract).ok


def test_render_outputs_round_trip() -> None:
    report = verify_token_budget_contract(load_token_budget_contract(str(EXAMPLES / "overflow-pack.json")))
    text = render_token_budget_report_text(report)
    assert "VIOLATED" in text
    payload = json.loads(render_token_budget_report_json(report))
    assert payload["ok"] is False
    assert payload["contract"] == "overflow-pack"


def test_cli_token_budget(capsys: pytest.CaptureFixture[str]) -> None:
    ok = main(["token-budget", "--contract", str(EXAMPLES / "chat-pack.json")])
    assert ok == 0
    assert "PROVEN" in capsys.readouterr().out

    bad = main(["token-budget", "--contract", str(EXAMPLES / "overflow-pack.json"), "--format", "json"])
    assert bad == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
