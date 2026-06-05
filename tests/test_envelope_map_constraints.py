"""Tests for provider-envelope finite map constraints (step 222)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.envelope_map_constraints import (
    EnvelopeProofStatus,
    envelope_contract_from_dict,
    load_envelope_contract,
    render_envelope_report_json,
    render_envelope_report_text,
    verify_envelope_contract,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "envelope-map-constraints"


def test_openai_chat_envelope_is_proven() -> None:
    report = verify_envelope_contract(load_envelope_contract(str(EXAMPLES / "openai-chat.json")))
    assert report.ok
    assert {r.guarantee for r in report.results} == {
        "model-is-supported",
        "temperature-within-range",
        "tool_choice-implies-tools",
    }
    assert all(r.status is EnvelopeProofStatus.PROVEN for r in report.results)


def test_unsound_stream_envelope_is_refuted_with_counterexample() -> None:
    report = verify_envelope_contract(load_envelope_contract(str(EXAMPLES / "unsound-stream.json")))
    assert not report.ok
    refuted = report.refuted
    assert len(refuted) == 1
    assert refuted[0].counterexample is not None
    # the counterexample must witness stream=true while logprobs is present
    assert "logprobs" in refuted[0].counterexample


def test_required_field_absence_is_refuted() -> None:
    contract = envelope_contract_from_dict(
        {
            "name": "needs-model",
            "fields": [{"name": "model", "kind": "enum", "members": ["a", "b"], "optional": False}],
            "guarantees": [{"name": "model-present", "kind": "present", "field": "model"}],
        }
    )
    report = verify_envelope_contract(contract)
    # model is required, so it is always present -> guarantee holds
    assert report.ok


def test_mutually_exclusive_guarantee() -> None:
    contract = envelope_contract_from_dict(
        {
            "name": "exclusive",
            "fields": [
                {"name": "a", "kind": "bool", "optional": True},
                {"name": "b", "kind": "bool", "optional": True},
            ],
            "assumptions": [
                {"name": "never-both", "kind": "mutually-exclusive", "fields": ["a", "b"]}
            ],
            "guarantees": [
                {"name": "still-exclusive", "kind": "mutually-exclusive", "fields": ["a", "b"]}
            ],
        }
    )
    assert verify_envelope_contract(contract).ok


def test_requires_together_is_refuted_without_assumption() -> None:
    contract = envelope_contract_from_dict(
        {
            "name": "needs-companion",
            "fields": [
                {"name": "tool_choice", "kind": "enum", "members": ["auto", "none"], "optional": True},
                {"name": "tools", "kind": "bool", "optional": True},
            ],
            "guarantees": [
                {
                    "name": "tool_choice-requires-tools",
                    "kind": "requires-together",
                    "field": "tool_choice",
                    "fields": ["tools"],
                }
            ],
        }
    )
    report = verify_envelope_contract(contract)
    assert not report.ok
    assert report.results[0].status is EnvelopeProofStatus.REFUTED


def test_render_outputs_round_trip() -> None:
    report = verify_envelope_contract(load_envelope_contract(str(EXAMPLES / "openai-chat.json")))
    text = render_envelope_report_text(report)
    assert "status: PROVEN" in text
    payload = json.loads(render_envelope_report_json(report))
    assert payload["ok"] is True
    assert payload["contract"] == "openai-chat-request"


def test_cli_envelope_map(capsys: pytest.CaptureFixture[str]) -> None:
    ok = main(["envelope-map", "--contract", str(EXAMPLES / "openai-chat.json")])
    assert ok == 0
    assert "PROVEN" in capsys.readouterr().out

    bad = main(["envelope-map", "--contract", str(EXAMPLES / "unsound-stream.json"), "--format", "json"])
    assert bad == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
