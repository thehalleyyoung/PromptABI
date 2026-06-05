"""Certify abstention reasons for unsupported formulas (step 227).

When PromptABI's solver returns ``ABSTAINED`` it is making a *soundness*
promise: "this formula left my supported finite/SMT fragment, so I refuse to
guess."  That promise is only trustworthy if the abstention is **justified** --
the formula really does contain a construct outside the fragment -- and,
symmetrically, if formulas that are *inside* the fragment are never abstained on
for fragment reasons.

This module turns that promise into a checkable certificate.  It

* statically classifies a :class:`~promptabi.formal.FiniteContractProblem`
  against the supported node and domain registries, naming every construct that
  is outside the fragment and where it occurs;
* runs (or accepts) the solver result and reads its abstention status;
* emits an :class:`AbstentionCertificate` that reconciles the two and flags any
  mismatch -- an *unjustified abstention* (the solver gave up on a supported
  formula) or a *missed abstention* (the solver answered a formula it should not
  have been able to model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .formal import (
    And,
    BoolDomain,
    BoundedStringDomain,
    Contains,
    Eq,
    EnumDomain,
    Expression,
    FiniteContractProblem,
    Ge,
    Gt,
    Implies,
    InSet,
    IntRangeDomain,
    Le,
    Length,
    Lt,
    Mul,
    Ne,
    Not,
    Or,
    SolverBudgetOutcome,
    SolverResult,
    SolverStatus,
    Sum,
    Value,
    Var,
)

ABSTENTION_CERTIFICATE_VERSION = "promptabi.abstention-certificate.v1"

SUPPORTED_NODE_TYPES: frozenset[type] = frozenset(
    {Var, Value, Eq, Ne, Le, Lt, Ge, Gt, Sum, Mul, And, Or, Not, Implies, InSet, Length, Contains}
)

SUPPORTED_DOMAIN_TYPES: frozenset[type] = frozenset(
    {BoolDomain, EnumDomain, IntRangeDomain, BoundedStringDomain}
)

# Child-expression accessors per supported node type.
_CHILD_ACCESSORS: dict[type, tuple[str, ...]] = {
    Var: (),
    Value: (),
    Eq: ("left", "right"),
    Ne: ("left", "right"),
    Le: ("left", "right"),
    Lt: ("left", "right"),
    Ge: ("left", "right"),
    Gt: ("left", "right"),
    Sum: ("terms",),
    Mul: ("term",),
    And: ("terms",),
    Or: ("terms",),
    Not: ("term",),
    Implies: ("condition", "consequence"),
    InSet: ("term",),
    Length: ("term",),
    Contains: ("haystack", "needle"),
}


class AbstentionFindingKind(StrEnum):
    UNJUSTIFIED_ABSTENTION = "unjustified-abstention"
    MISSED_ABSTENTION = "missed-abstention"


@dataclass(frozen=True, slots=True)
class UnsupportedConstruct:
    location: str
    construct: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"location": self.location, "construct": self.construct, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FragmentClassification:
    supported: bool
    unsupported: tuple[UnsupportedConstruct, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "unsupported": [item.to_dict() for item in self.unsupported],
        }


@dataclass(frozen=True, slots=True)
class AbstentionFinding:
    kind: AbstentionFindingKind
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class AbstentionCertificate:
    version: str
    problem: str
    classification: FragmentClassification
    solver_status: str
    solver_outcome: str
    abstained_for_fragment: bool
    justified: bool
    findings: tuple[AbstentionFinding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "problem": self.problem,
            "ok": self.ok,
            "justified": self.justified,
            "abstained_for_fragment": self.abstained_for_fragment,
            "solver_status": self.solver_status,
            "solver_outcome": self.solver_outcome,
            "classification": self.classification.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _iter_children(expression: Expression) -> tuple[Expression, ...]:
    accessors = _CHILD_ACCESSORS.get(type(expression))
    if not accessors:
        return ()
    children: list[Expression] = []
    for accessor in accessors:
        value = getattr(expression, accessor, None)
        if isinstance(value, tuple):
            children.extend(value)
        elif value is not None:
            children.append(value)  # type: ignore[arg-type]
    return tuple(children)


def _classify_expression(
    expression: Expression, location: str, out: list[UnsupportedConstruct]
) -> None:
    if type(expression) not in SUPPORTED_NODE_TYPES:
        out.append(
            UnsupportedConstruct(
                location=location,
                construct=type(expression).__name__,
                detail="expression node is outside the supported finite/SMT fragment",
            )
        )
        return
    for index, child in enumerate(_iter_children(expression)):
        _classify_expression(child, f"{location}/{type(expression).__name__}[{index}]", out)


def classify_fragment(problem: FiniteContractProblem) -> FragmentClassification:
    """Statically classify a problem against the supported fragment registries."""

    unsupported: list[UnsupportedConstruct] = []
    for variable in problem.variables:
        if type(variable) not in SUPPORTED_DOMAIN_TYPES:
            unsupported.append(
                UnsupportedConstruct(
                    location=f"variable:{variable.name}",
                    construct=type(variable).__name__,
                    detail="variable domain is outside the supported finite/SMT fragment",
                )
            )
    for constraint in problem.constraints:
        _classify_expression(constraint.expression, f"constraint:{constraint.name}", unsupported)
    return FragmentClassification(supported=not unsupported, unsupported=tuple(unsupported))


def _abstained_for_fragment(result: SolverResult) -> bool:
    if result.status is not SolverStatus.UNKNOWN:
        return False
    if result.budget_outcome is not SolverBudgetOutcome.ABSTAINED:
        return False
    reason = (result.reason or result.budget_reason or "").lower()
    return "unsupported solver fragment" in reason or "outside" in reason or reason == ""


def certify_abstention(
    problem: FiniteContractProblem,
    result: SolverResult | None = None,
    *,
    prefer_z3: bool = True,
) -> AbstentionCertificate:
    """Certify that the solver's (non-)abstention matches the static fragment."""

    classification = classify_fragment(problem)
    solver_result = result if result is not None else problem.solve(prefer_z3=prefer_z3)
    abstained = _abstained_for_fragment(solver_result)

    findings: list[AbstentionFinding] = []
    if classification.supported and abstained:
        findings.append(
            AbstentionFinding(
                kind=AbstentionFindingKind.UNJUSTIFIED_ABSTENTION,
                message=(
                    "solver abstained for fragment reasons but every construct is inside "
                    "the supported fragment"
                ),
            )
        )
    if not classification.supported and not abstained:
        names = ", ".join(sorted({item.construct for item in classification.unsupported}))
        findings.append(
            AbstentionFinding(
                kind=AbstentionFindingKind.MISSED_ABSTENTION,
                message=(
                    f"problem contains unsupported construct(s) [{names}] but the solver "
                    f"returned a conclusive verdict instead of abstaining"
                ),
            )
        )

    justified = (abstained and not classification.supported) or (
        not abstained and classification.supported
    )

    return AbstentionCertificate(
        version=ABSTENTION_CERTIFICATE_VERSION,
        problem=problem.name,
        classification=classification,
        solver_status=solver_result.status.value,
        solver_outcome=solver_result.budget_outcome.value,
        abstained_for_fragment=abstained,
        justified=justified,
        findings=tuple(findings),
    )


def certify_abstentions(
    problems: Sequence[FiniteContractProblem],
    *,
    prefer_z3: bool = True,
) -> tuple[AbstentionCertificate, ...]:
    return tuple(certify_abstention(problem, prefer_z3=prefer_z3) for problem in problems)


def render_abstention_certificate_json(certificate: AbstentionCertificate) -> str:
    return json.dumps(certificate.to_dict(), indent=2, sort_keys=True) + "\n"


def render_abstention_certificate_text(certificate: AbstentionCertificate) -> str:
    lines = [
        f"PromptABI abstention certificate ({certificate.version})",
        f"problem: {certificate.problem}",
        f"fragment: {'supported' if certificate.classification.supported else 'unsupported'}",
        f"solver: {certificate.solver_status} / {certificate.solver_outcome}",
        f"abstained-for-fragment: {certificate.abstained_for_fragment}",
        f"justified: {certificate.justified}",
    ]
    for item in certificate.classification.unsupported:
        lines.append(f"  unsupported: {item.construct} at {item.location}")
    for finding in certificate.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
