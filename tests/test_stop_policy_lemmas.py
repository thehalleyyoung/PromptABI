"""Tests for stop-policy prefix/suffix lemmas (step 221)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptabi.stop_policy_lemmas import (
    StopPolicyLemmaKind,
    StopPolicyLemmaStatus,
    load_stop_policy,
    render_stop_policy_lemmas_json,
    render_stop_policy_lemmas_text,
    verify_stop_policy_lemmas,
)
from promptabi.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "stop-policy-lemmas"


def _result(report, kind):
    return next(r for r in report.results if r.kind is kind)


def test_clean_policy_proves_every_lemma() -> None:
    report = verify_stop_policy_lemmas(load_stop_policy(str(EXAMPLES / "clean.json")))
    assert report.ok
    assert all(r.status is StopPolicyLemmaStatus.PROVEN for r in report.results)


def test_shadowed_policy_refutes_prefix_and_substring() -> None:
    report = verify_stop_policy_lemmas(load_stop_policy(str(EXAMPLES / "shadowed.json")))
    assert not report.ok
    prefix = _result(report, StopPolicyLemmaKind.PREFIX_FREE)
    assert prefix.status is StopPolicyLemmaStatus.REFUTED
    assert prefix.certified_by_automaton
    shadowed_pairs = {(w.shadowing, w.shadowed) for w in prefix.witnesses}
    assert ("</s>", "</s></s>") in shadowed_pairs
    assert ("STOP", "STOPNOW") in shadowed_pairs

    substring = _result(report, StopPolicyLemmaKind.SUBSTRING_FREE)
    assert substring.status is StopPolicyLemmaStatus.REFUTED


def test_border_lemma_detects_self_overlap() -> None:
    report = verify_stop_policy_lemmas({"name": "p", "stop_sequences": ["abcab"]})
    border = _result(report, StopPolicyLemmaKind.BORDER_FREE)
    assert border.status is StopPolicyLemmaStatus.REFUTED
    assert any(w.shadowing == "ab" for w in border.witnesses)


def test_empty_stop_sequence_is_refuted() -> None:
    report = verify_stop_policy_lemmas({"name": "p", "stop_sequences": ["", "x"]})
    non_empty = _result(report, StopPolicyLemmaKind.NON_EMPTY)
    assert non_empty.status is StopPolicyLemmaStatus.REFUTED


def test_prefix_free_holds_for_distinct_unrelated_stops() -> None:
    report = verify_stop_policy_lemmas({"name": "p", "stop_sequences": ["END", "HALT", "DONE"]})
    assert report.ok


def test_render_outputs_round_trip() -> None:
    report = verify_stop_policy_lemmas(load_stop_policy(str(EXAMPLES / "shadowed.json")))
    text = render_stop_policy_lemmas_text(report)
    assert "VIOLATED" in text
    payload = json.loads(render_stop_policy_lemmas_json(report))
    assert payload["ok"] is False
    assert payload["policy_name"] == "shadowed-stops"


def test_cli_stop_policy_lemmas(capsys: pytest.CaptureFixture[str]) -> None:
    ok = main(["stop-policy-lemmas", "--policy", str(EXAMPLES / "clean.json")])
    assert ok == 0
    assert "PROVEN" in capsys.readouterr().out

    bad = main(["stop-policy-lemmas", "--policy", str(EXAMPLES / "shadowed.json"), "--format", "json"])
    assert bad == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
