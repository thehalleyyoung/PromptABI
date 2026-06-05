"""Tests for derived migration interpolants (step 231)."""

from __future__ import annotations

import json

import pytest

from promptabi.formal import (
    And,
    BoolDomain,
    Eq,
    FiniteContractProblem,
    NamedConstraint,
    Value,
    Var,
)
from promptabi.migration_interpolant import (
    MigrationInterpolantError,
    MigrationStatus,
    combine_contracts,
    derive_migration_interpolant,
    render_migration_interpolant_json,
    render_migration_interpolant_text,
)

_VARS = (BoolDomain(name="content_present"), BoolDomain(name="tool_calls_present"))


def _contract(name: str, expr) -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=_VARS,
        constraints=(NamedConstraint(name=name, expression=expr),),
        name=name,
    )


def _tool_only_region(name: str) -> FiniteContractProblem:
    # tool_calls present AND content absent (assistant tool-only message)
    return _contract(
        name,
        And(Eq(Var("tool_calls_present"), Value(True)), Eq(Var("content_present"), Value(False))),
    )


def test_unsafe_migration_yields_incompatibility_cube() -> None:
    source = _tool_only_region("source-accepts-tool-only")
    target_reject = _tool_only_region("target-rejects-tool-only")
    interpolant = derive_migration_interpolant(source, target_reject)
    assert interpolant.status is MigrationStatus.UNSAFE
    assert not interpolant.safe
    cube = {literal.variable: literal.value for literal in interpolant.incompatibility_cube}
    assert cube == {"tool_calls_present": True, "content_present": False}
    assert interpolant.witness is not None


def test_safe_migration_yields_interpolant() -> None:
    source = _tool_only_region("source-accepts-tool-only")
    # target only rejects empty assistant messages (no tools, no content)
    target_reject = _contract(
        "target-rejects-empty",
        And(Eq(Var("tool_calls_present"), Value(False)), Eq(Var("content_present"), Value(False))),
    )
    interpolant = derive_migration_interpolant(source, target_reject)
    assert interpolant.status is MigrationStatus.SAFE
    assert interpolant.safe
    assert interpolant.interpolant_terms
    assert interpolant.render_interpolant() != "false"


def test_cube_drops_irrelevant_variable() -> None:
    # source accepts everything with content absent (tool flag irrelevant)
    source = _contract("source", Eq(Var("content_present"), Value(False)))
    target_reject = _contract("target", Eq(Var("content_present"), Value(False)))
    interpolant = derive_migration_interpolant(source, target_reject)
    assert interpolant.status is MigrationStatus.UNSAFE
    cube_vars = {literal.variable for literal in interpolant.incompatibility_cube}
    # tool_calls_present is a don't-care -> dropped from the cube
    assert cube_vars == {"content_present"}


def test_combine_rejects_mismatched_vocabulary() -> None:
    source = _tool_only_region("source")
    other = FiniteContractProblem(
        variables=(BoolDomain(name="x"),),
        constraints=(NamedConstraint(name="c", expression=Eq(Var("x"), Value(True))),),
        name="other",
    )
    with pytest.raises(MigrationInterpolantError):
        combine_contracts(source, other)


def test_render_round_trips() -> None:
    source = _tool_only_region("source")
    target_reject = _tool_only_region("target")
    interpolant = derive_migration_interpolant(source, target_reject)
    payload = json.loads(render_migration_interpolant_json(interpolant))
    assert payload["safe"] is False
    assert "migration interpolant" in render_migration_interpolant_text(interpolant)
