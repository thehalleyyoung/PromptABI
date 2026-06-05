"""Tests for cross-field json schema obligations (step 238)."""

from __future__ import annotations

import json

from promptabi.cross_field_obligations import (
    FieldKind,
    FieldSpec,
    MutuallyExclusive,
    Ordering,
    RequiredIf,
    SumBound,
    check_consistency,
    check_entailment,
    render_obligation_json,
    render_obligation_text,
)


def _bool(name: str) -> FieldSpec:
    return FieldSpec(name=name, kind=FieldKind.BOOL)


def _int(name: str, lo: int, hi: int) -> FieldSpec:
    return FieldSpec(name=name, kind=FieldKind.INT, minimum=lo, maximum=hi)


def test_consistent_obligations() -> None:
    fields = [_bool("tool_choice"), _bool("tools"), _bool("temperature"), _bool("top_p")]
    obligations = [
        RequiredIf("choice-needs-tools", "tool_choice", "tools"),
        MutuallyExclusive("sampling", "temperature", "top_p"),
    ]
    result = check_consistency(fields, obligations)
    assert result.holds
    assert result.status == "sat"


def test_contradictory_obligations_detected() -> None:
    fields = [_int("start", 0, 5), _int("end", 0, 5)]
    obligations = [
        Ordering("start-before-end", "start", "end"),
        Ordering("end-before-start", "end", "start"),
    ]
    result = check_consistency(fields, obligations)
    assert not result.holds
    assert result.status == "unsat"


def test_sum_bound_entailment_holds() -> None:
    fields = [_int("prompt", 0, 100), _int("max", 0, 100)]
    # If prompt+max <= 50 then prompt <= 50 must hold.
    assumptions = [SumBound("budget", ("prompt", "max"), 50)]
    claim = SumBound("prompt-small", ("prompt",), 50)
    result = check_entailment(fields, assumptions, claim)
    assert result.holds
    assert result.status == "unsat"


def test_entailment_fails_with_counterexample() -> None:
    fields = [_int("prompt", 0, 100), _int("max", 0, 100)]
    assumptions = [SumBound("budget", ("prompt", "max"), 80)]
    claim = SumBound("prompt-small", ("prompt",), 50)
    result = check_entailment(fields, assumptions, claim)
    assert not result.holds
    assert result.status == "sat"
    assert result.counterexample is not None
    assert result.counterexample["prompt"] > 50


def test_required_if_entailment() -> None:
    fields = [_bool("a"), _bool("b"), _bool("c")]
    # a->b and b->c entails a->c
    assumptions = [RequiredIf("ab", "a", "b"), RequiredIf("bc", "b", "c")]
    claim = RequiredIf("ac", "a", "c")
    result = check_entailment(fields, assumptions, claim)
    assert result.holds


def test_render_round_trips() -> None:
    fields = [_int("start", 0, 5), _int("end", 0, 5)]
    result = check_consistency(fields, [Ordering("o", "start", "end")])
    payload = json.loads(render_obligation_json(result))
    assert payload["holds"] is True
    assert "obligation" in render_obligation_text(result)
