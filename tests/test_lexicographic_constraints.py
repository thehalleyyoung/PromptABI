"""Tests for lexicographic constraints over ordered messages (step 233)."""

from __future__ import annotations

import json

import pytest

from promptabi.formal import (
    Eq,
    IntRangeDomain,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.lexicographic_constraints import (
    LexOrderContract,
    lex_less,
    lex_less_equal,
    render_lex_order_json,
    render_lex_order_text,
    verify_lex_order,
)


def _ev(expr, assignment) -> bool:
    return bool(expr.evaluate(assignment))


def test_lex_less_semantics() -> None:
    left = [Var("a0"), Var("a1")]
    right = [Var("b0"), Var("b1")]
    expr = lex_less(left, right)
    assert _ev(expr, {"a0": 1, "a1": 5, "b0": 1, "b1": 9}) is True
    assert _ev(expr, {"a0": 2, "a1": 1, "b0": 1, "b1": 9}) is False
    assert _ev(expr, {"a0": 1, "a1": 9, "b0": 1, "b1": 9}) is False  # equal
    assert _ev(expr, {"a0": 0, "a1": 9, "b0": 1, "b1": 0}) is True


def test_lex_less_equal_includes_equality() -> None:
    left = [Var("a0")]
    right = [Var("b0")]
    expr = lex_less_equal(left, right)
    assert _ev(expr, {"a0": 3, "b0": 3}) is True
    assert _ev(expr, {"a0": 4, "b0": 3}) is False


def test_lex_mismatched_widths_rejected() -> None:
    with pytest.raises(ValueError):
        lex_less([Var("a")], [Var("b0"), Var("b1")])


def _two_row_contract(assumptions) -> LexOrderContract:
    variables = (
        IntRangeDomain(name="turn0", minimum=0, maximum=2),
        IntRangeDomain(name="rank0", minimum=0, maximum=3),
        IntRangeDomain(name="turn1", minimum=0, maximum=2),
        IntRangeDomain(name="rank1", minimum=0, maximum=3),
    )
    rows = (
        (Var("turn0"), Var("rank0")),
        (Var("turn1"), Var("rank1")),
    )
    return LexOrderContract(variables=variables, rows=rows, assumptions=assumptions, name="msgs")


def test_provably_ordered_when_turns_increase() -> None:
    assumptions = (
        NamedConstraint(name="t0", expression=Eq(Var("turn0"), Value(0))),
        NamedConstraint(name="t1", expression=Eq(Var("turn1"), Value(1))),
    )
    report = verify_lex_order(_two_row_contract(assumptions))
    assert report.ordered
    assert report.ok
    assert report.witness is None


def test_out_of_order_counterexample_found() -> None:
    # no ordering assumptions -> there exist assignments that violate order
    report = verify_lex_order(_two_row_contract(()))
    assert not report.ordered
    assert report.witness is not None


def test_always_out_of_order_is_detected() -> None:
    assumptions = (
        NamedConstraint(name="t0", expression=Eq(Var("turn0"), Value(2))),
        NamedConstraint(name="t1", expression=Eq(Var("turn1"), Value(0))),
    )
    report = verify_lex_order(_two_row_contract(assumptions))
    assert not report.ordered


def test_render_round_trips() -> None:
    report = verify_lex_order(_two_row_contract(()))
    payload = json.loads(render_lex_order_json(report))
    assert payload["ordered"] is False
    assert "lexicographic order" in render_lex_order_text(report)
