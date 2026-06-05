"""Tests for certified abstention reasons on unsupported formulas (step 227)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from promptabi.formal import (
    Assignment,
    BoolDomain,
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    SolverBackend,
    SolverBudgetOutcome,
    SolverResult,
    SolverStatus,
    Value,
    Var,
    _Z3Context,
)
from promptabi.abstention_certificate import (
    AbstentionFindingKind,
    certify_abstention,
    classify_fragment,
    render_abstention_certificate_json,
    render_abstention_certificate_text,
)


@dataclass(frozen=True)
class _UnsupportedNode:
    """A node deliberately outside the supported fragment."""

    inner: object

    def evaluate(self, assignment: Assignment) -> bool:
        raise TypeError("unsupported construct cannot be evaluated")

    def to_z3(self, context: "_Z3Context") -> object:
        raise TypeError("unsupported construct cannot be encoded to z3")

    def to_dict(self) -> dict[str, object]:
        return {"unsupported": True}


def _supported_problem() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="supported",
    )


def _unsupported_problem() -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(
            NamedConstraint(name="weird", expression=_UnsupportedNode(inner="?")),  # type: ignore[arg-type]
        ),
        name="unsupported",
    )


def test_supported_problem_is_classified_supported() -> None:
    classification = classify_fragment(_supported_problem())
    assert classification.supported is True
    assert classification.unsupported == ()


def test_unsupported_node_is_named() -> None:
    classification = classify_fragment(_unsupported_problem())
    assert classification.supported is False
    constructs = {item.construct for item in classification.unsupported}
    assert "_UnsupportedNode" in constructs


def test_supported_problem_does_not_abstain() -> None:
    cert = certify_abstention(_supported_problem())
    assert cert.ok
    assert cert.justified
    assert cert.abstained_for_fragment is False


def test_unsupported_problem_abstains_justifiably() -> None:
    cert = certify_abstention(_unsupported_problem())
    assert cert.ok
    assert cert.justified
    assert cert.abstained_for_fragment is True


def test_unjustified_abstention_is_flagged() -> None:
    fake = SolverResult(
        status=SolverStatus.UNKNOWN,
        backend=SolverBackend.FINITE_ENUMERATION,
        reason="unsupported solver fragment: fabricated",
        budget_outcome=SolverBudgetOutcome.ABSTAINED,
    )
    cert = certify_abstention(_supported_problem(), fake)
    assert not cert.ok
    assert any(f.kind is AbstentionFindingKind.UNJUSTIFIED_ABSTENTION for f in cert.findings)


def test_missed_abstention_is_flagged() -> None:
    fake = SolverResult(
        status=SolverStatus.SAT,
        backend=SolverBackend.FINITE_ENUMERATION,
        assignment={"x": 0},
        budget_outcome=SolverBudgetOutcome.PROVED,
    )
    cert = certify_abstention(_unsupported_problem(), fake)
    assert not cert.ok
    assert any(f.kind is AbstentionFindingKind.MISSED_ABSTENTION for f in cert.findings)


def test_render_round_trips() -> None:
    cert = certify_abstention(_unsupported_problem())
    payload = json.loads(render_abstention_certificate_json(cert))
    assert payload["justified"] is True
    assert "abstention certificate" in render_abstention_certificate_text(cert)
