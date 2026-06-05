"""Support cross-field json schema obligations (step 238).

JSON Schema decides each property in isolation; the obligations that actually
protect a prompt interface are *cross-field*: "if ``tool_choice`` is set then
``tools`` must be non-empty", "``max_tokens`` + ``prompt_tokens`` must not exceed
the context window", "``temperature`` and ``top_p`` are mutually exclusive",
"``start`` must be less than ``end``".  These obligations are exactly the part a
plain schema validator cannot reason about, yet they are decidable over the
finite/bounded field domains PromptABI already models.

This module gives those obligations a first-class encoding.  Each obligation
compiles to a :mod:`~promptabi.formal` expression over declared fields, and we
offer two proofs over the compiled finite problem:

* :func:`check_consistency` -- the conjunction of obligations is satisfiable, so
  the schema is not self-contradictory (some valid document exists);
* :func:`check_entailment` -- a claimed obligation is *implied* by a set of
  assumed obligations (the negation is UNSAT), so a downstream consumer may rely
  on it without restating it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from .formal import (
    And,
    BoolDomain,
    Eq,
    Expression,
    FiniteContractProblem,
    Implies,
    IntRangeDomain,
    Le,
    Lt,
    NamedConstraint,
    Not,
    Or,
    SolverStatus,
    Sum,
    Value,
    Var,
)

CROSS_FIELD_OBLIGATIONS_VERSION = "promptabi.cross-field-obligations.v1"


class FieldKind(StrEnum):
    BOOL = "bool"
    INT = "int"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    kind: FieldKind
    minimum: int = 0
    maximum: int = 1

    def domain(self):
        if self.kind is FieldKind.BOOL:
            return BoolDomain(name=self.name)
        return IntRangeDomain(name=self.name, minimum=self.minimum, maximum=self.maximum)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "kind": self.kind.value}
        if self.kind is FieldKind.INT:
            payload["minimum"] = self.minimum
            payload["maximum"] = self.maximum
        return payload


class Obligation:
    """Base class for cross-field obligations."""

    name: str

    def to_expression(self) -> Expression:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RequiredIf(Obligation):
    """If ``condition`` field is true/present, ``required`` field must be true."""

    name: str
    condition: str
    required: str

    def to_expression(self) -> Expression:
        return Implies(Eq(Var(self.condition), Value(True)), Eq(Var(self.required), Value(True)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": "required-if",
            "condition": self.condition,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class MutuallyExclusive(Obligation):
    """At most one of two boolean fields may be true."""

    name: str
    left: str
    right: str

    def to_expression(self) -> Expression:
        return Or(Eq(Var(self.left), Value(False)), Eq(Var(self.right), Value(False)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": "mutually-exclusive",
            "left": self.left,
            "right": self.right,
        }


@dataclass(frozen=True, slots=True)
class Ordering(Obligation):
    """Integer field ``less`` must be strictly less than ``greater``."""

    name: str
    less: str
    greater: str

    def to_expression(self) -> Expression:
        return Lt(Var(self.less), Var(self.greater))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": "ordering",
            "less": self.less,
            "greater": self.greater,
        }


@dataclass(frozen=True, slots=True)
class SumBound(Obligation):
    """Sum of integer fields must not exceed ``limit``."""

    name: str
    fields: tuple[str, ...]
    limit: int

    def to_expression(self) -> Expression:
        return Le(Sum(*(Var(f) for f in self.fields)), Value(self.limit))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": "sum-bound",
            "fields": list(self.fields),
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class ObligationResult:
    version: str
    kind: str
    holds: bool
    status: str
    counterexample: dict[str, object] | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": self.kind,
            "holds": self.holds,
            "status": self.status,
            "counterexample": self.counterexample,
            "detail": self.detail,
        }


def _problem(
    fields: Sequence[FieldSpec],
    expressions: Sequence[tuple[str, Expression]],
    *,
    name: str,
) -> FiniteContractProblem:
    return FiniteContractProblem(
        variables=tuple(spec.domain() for spec in fields),
        constraints=tuple(
            NamedConstraint(name=cname, expression=expr) for cname, expr in expressions
        ),
        name=name,
    )


def check_consistency(
    fields: Sequence[FieldSpec],
    obligations: Sequence[Obligation],
    *,
    prefer_z3: bool = True,
) -> ObligationResult:
    """A schema's obligations are consistent iff their conjunction is satisfiable."""

    problem = _problem(
        fields,
        [(o.name, o.to_expression()) for o in obligations],
        name="cross-field-consistency",
    )
    result = problem.solve(prefer_z3=prefer_z3, max_assignments=200000)
    if result.status is SolverStatus.SAT:
        return ObligationResult(
            version=CROSS_FIELD_OBLIGATIONS_VERSION,
            kind="consistency",
            holds=True,
            status=result.status.value,
            counterexample=None,
            detail="obligations are mutually satisfiable",
        )
    if result.status is SolverStatus.UNSAT:
        return ObligationResult(
            version=CROSS_FIELD_OBLIGATIONS_VERSION,
            kind="consistency",
            holds=False,
            status=result.status.value,
            counterexample=None,
            detail="obligations are contradictory: no valid document exists",
        )
    return ObligationResult(
        version=CROSS_FIELD_OBLIGATIONS_VERSION,
        kind="consistency",
        holds=False,
        status=result.status.value,
        counterexample=None,
        detail=f"undecided: {result.reason}",
    )


def check_entailment(
    fields: Sequence[FieldSpec],
    assumptions: Sequence[Obligation],
    claim: Obligation,
    *,
    prefer_z3: bool = True,
) -> ObligationResult:
    """``claim`` is entailed iff assumptions AND not(claim) is unsatisfiable."""

    assumption_expr = (
        And(*(a.to_expression() for a in assumptions))
        if assumptions
        else Value(True)
    )
    problem = _problem(
        fields,
        [
            ("assumptions", assumption_expr),
            ("negated-claim", Not(claim.to_expression())),
        ],
        name="cross-field-entailment",
    )
    result = problem.solve(prefer_z3=prefer_z3, max_assignments=200000)
    if result.status is SolverStatus.UNSAT:
        return ObligationResult(
            version=CROSS_FIELD_OBLIGATIONS_VERSION,
            kind="entailment",
            holds=True,
            status=result.status.value,
            counterexample=None,
            detail=f"{claim.name!r} is implied by the assumed obligations",
        )
    if result.status is SolverStatus.SAT:
        return ObligationResult(
            version=CROSS_FIELD_OBLIGATIONS_VERSION,
            kind="entailment",
            holds=False,
            status=result.status.value,
            counterexample=dict(result.assignment or {}),
            detail=f"{claim.name!r} is not implied; counterexample document found",
        )
    return ObligationResult(
        version=CROSS_FIELD_OBLIGATIONS_VERSION,
        kind="entailment",
        holds=False,
        status=result.status.value,
        counterexample=None,
        detail=f"undecided: {result.reason}",
    )


def render_obligation_json(result: ObligationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


def render_obligation_text(result: ObligationResult) -> str:
    lines = [
        f"PromptABI cross-field obligation ({result.version})",
        f"check: {result.kind}",
        f"result: {'HOLDS' if result.holds else 'FAILS'} ({result.status})",
        f"detail: {result.detail}",
    ]
    if result.counterexample:
        lines.append("counterexample:")
        for key, value in sorted(result.counterexample.items()):
            lines.append(f"  {key} = {value}")
    return "\n".join(lines) + "\n"
