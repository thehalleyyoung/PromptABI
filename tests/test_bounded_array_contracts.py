"""Tests for bounded-array static contracts (step 220)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.bounded_array_contracts import (
    BoundedArrayProofStatus,
    bounded_array_contract_from_dict,
    load_bounded_array_contract,
    render_bounded_array_report_json,
    render_bounded_array_report_text,
    verify_bounded_array_contract,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "bounded-array-contracts"


def test_token_budget_example_is_proven() -> None:
    contract = load_bounded_array_contract(str(EXAMPLES / "token-budget-segments.json"))
    report = verify_bounded_array_contract(contract)
    assert report.ok
    assert {r.guarantee for r in report.results} == {
        "no-negative-segment",
        "at-least-one-segment",
        "largest-segment-within-window",
    }
    assert all(r.status is BoundedArrayProofStatus.PROVEN for r in report.results)


def test_enum_chat_role_sequence_is_proven() -> None:
    contract = load_bounded_array_contract(str(EXAMPLES / "chat-role-sequence.json"))
    report = verify_bounded_array_contract(contract)
    assert report.ok
    assert all(r.status is BoundedArrayProofStatus.PROVEN for r in report.results)


def test_unsound_contract_is_refuted_with_counterexample() -> None:
    contract = load_bounded_array_contract(str(EXAMPLES / "unsound-budget.json"))
    report = verify_bounded_array_contract(contract)
    assert not report.ok
    refuted = report.refuted
    assert len(refuted) == 1
    result = refuted[0]
    assert result.status is BoundedArrayProofStatus.REFUTED
    assert result.counterexample is not None
    # the counterexample must actually witness the violation: some segment > 4
    assert any(isinstance(v, int) and v > 4 for v in result.counterexample)


def test_from_dict_round_trips_and_distinct_is_unprovable_without_assumption() -> None:
    contract = bounded_array_contract_from_dict(
        {
            "name": "free-roles",
            "element_domain": {"kind": "enum", "members": ["a", "b", "c"]},
            "max_length": 3,
            "min_length": 1,
            "assumptions": [],
            "guarantees": [{"name": "all-distinct", "kind": "distinct"}],
        }
    )
    report = verify_bounded_array_contract(contract)
    # without a distinctness assumption, a duplicated array is a valid counterexample
    assert not report.ok
    assert report.results[0].status is BoundedArrayProofStatus.REFUTED


def test_render_outputs_are_stable_and_machine_readable() -> None:
    contract = load_bounded_array_contract(str(EXAMPLES / "token-budget-segments.json"))
    report = verify_bounded_array_contract(contract)
    text = render_bounded_array_report_text(report)
    assert "status: PROVEN" in text
    payload = json.loads(render_bounded_array_report_json(report))
    assert payload["ok"] is True
    assert payload["contract"] == "token-budget-segments"
    assert len(payload["results"]) == 3


def test_cli_bounded_array_proven_and_refuted(capsys: pytest.CaptureFixture[str]) -> None:
    ok_code = main(["bounded-array", "--contract", str(EXAMPLES / "token-budget-segments.json")])
    assert ok_code == 0
    out = capsys.readouterr().out
    assert "PROVEN" in out

    bad_code = main(["bounded-array", "--contract", str(EXAMPLES / "unsound-budget.json"), "--format", "json"])
    assert bad_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
