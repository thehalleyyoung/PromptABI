"""Support lexicographic constraints for ordered messages (step 233).

Conversation transcripts, tool-call batches, and packed training windows are
*ordered*: turn indices must strictly increase, ``(turn, role_rank)`` keys must
sort the way the renderer assumes, and a message must never sort before the
message that precedes it.  These are **lexicographic** obligations over tuples of
finite integer keys.

This module compiles lexicographic comparisons into the quantifier-free finite
fragment PromptABI's solver already understands:

    lex_less([a0, a1, ...], [b0, b1, ...]) ==
        (a0 < b0) ∨ (a0 == b0 ∧ ((a1 < b1) ∨ (a1 == b1 ∧ ...)))

It then offers a :class:`LexOrderContract` that declares a sequence of key tuples
and proves they are strictly lexicographically increasing -- or returns the
exact out-of-order pair as a counterexample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .formal import (
    And,
    Eq,
    Expression,
    FiniteContractProblem,
    Lt,
    NamedConstraint,
    Not,
    Or,
    SolverStatus,
    VariableDomain,
)

LEXICOGRAPHIC_VERSION = "promptabi.lexicographic-constraints.v1"


def lex_less(left: Sequence[Expression], right: Sequence[Expression]) -> Expression:
    """Quantifier-free encoding of ``left <_lex right`` over equal-length tuples."""

    if len(left) != len(right):
        raise ValueError("lexicographic comparison requires equal-length key tuples")
    if not left:
        # empty tuples are equal, so strict-less is false
        return Eq(_zero(), _one())  # an always-false finite literal
    # build from the last position backwards
    formula: Expression = Lt(left[-1], right[-1])
    for index in range(len(left) - 2, -1, -1):
        formula = Or(Lt(left[index], right[index]), And(Eq(left[index], right[index]), formula))
    return formula


def lex_less_equal(left: Sequence[Expression], right: Sequence[Expression]) -> Expression:
    if len(left) != len(right):
        raise ValueError("lexicographic comparison requires equal-length key tuples")
    equal = And(*(Eq(a, b) for a, b in zip(left, right))) if left else Eq(_zero(), _zero())
    return Or(lex_less(left, right), equal)


def _zero() -> Expression:
    from .formal import Value

    return Value(0)


def _one() -> Expression:
    from .formal import Value

    return Value(1)


def strictly_increasing(rows: Sequence[Sequence[Expression]]) -> Expression:
    """Conjunction asserting consecutive rows are strictly lex-increasing."""

    if len(rows) < 2:
        return Eq(_zero(), _zero())  # vacuously true
    clauses = [lex_less(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    return And(*clauses)


class LexOrderFindingKind(StrEnum):
    OUT_OF_ORDER = "out-of-order"


@dataclass(frozen=True, slots=True)
class LexOrderContract:
    """Declared key tuples that must be strictly lexicographically increasing."""

    variables: tuple[VariableDomain, ...]
    rows: tuple[tuple[Expression, ...], ...]
    assumptions: tuple[NamedConstraint, ...] = ()
    name: str = "lex-order"

    def __post_init__(self) -> None:
        widths = {len(row) for row in self.rows}
        if len(widths) > 1:
            raise ValueError("all rows must have the same key-tuple width")


@dataclass(frozen=True, slots=True)
class LexOrderReport:
    version: str
    name: str
    ordered: bool
    witness: dict[str, object] | None = None
    findings: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.ordered

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "name": self.name,
            "ordered": self.ordered,
            "ok": self.ok,
            "witness": self.witness,
            "findings": list(self.findings),
        }


def verify_lex_order(contract: LexOrderContract, *, prefer_z3: bool = True) -> LexOrderReport:
    """Prove the rows are strictly lex-increasing under the declared assumptions."""

    increasing = strictly_increasing(contract.rows)
    # search for a counterexample: assumptions hold but ordering is violated
    constraints = list(contract.assumptions)
    constraints.append(NamedConstraint(name="order-violated", expression=Not(increasing)))
    problem = FiniteContractProblem(
        variables=contract.variables,
        constraints=tuple(constraints),
        name=f"{contract.name}:order",
    )
    result = problem.solve(prefer_z3=prefer_z3)
    if result.status is SolverStatus.UNSAT:
        return LexOrderReport(version=LEXICOGRAPHIC_VERSION, name=contract.name, ordered=True)
    if result.status is SolverStatus.SAT and result.assignment is not None:
        return LexOrderReport(
            version=LEXICOGRAPHIC_VERSION,
            name=contract.name,
            ordered=False,
            witness=dict(sorted(result.assignment.items())),
            findings=("rows are not strictly lexicographically increasing under the assumptions",),
        )
    return LexOrderReport(
        version=LEXICOGRAPHIC_VERSION,
        name=contract.name,
        ordered=False,
        findings=(f"solver abstained ({result.status.value})",),
    )


def render_lex_order_json(report: LexOrderReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_lex_order_text(report: LexOrderReport) -> str:
    lines = [
        f"PromptABI lexicographic order check ({report.version})",
        f"contract: {report.name}",
        f"status: {'ORDERED' if report.ordered else 'OUT-OF-ORDER'}",
    ]
    if report.witness:
        rendered = ", ".join(f"{k}={v!r}" for k, v in report.witness.items())
        lines.append(f"witness: {rendered}")
    for finding in report.findings:
        lines.append(f"  ! {finding}")
    return "\n".join(lines) + "\n"
