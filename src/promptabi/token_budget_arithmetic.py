"""Arithmetic over packed token budgets (step 223).

A prompt that ships to a provider is *packed* from several token-consuming
segments -- the system prompt, few-shot exemplars, retrieved context, the user
turn, serialized tool schemas -- plus a reservation for the completion.  The
single most common production incident is a pack whose segments can, for some
admissible input size, exceed the context window and trigger a hard truncation
or a 400.  Proving that "for every admissible segment size the packed total
fits" is *linear integer arithmetic*, which the finite-contract solver
(:mod:`promptabi.formal`) now supports via :class:`~promptabi.formal.Sum` and
:class:`~promptabi.formal.Mul`.

A :class:`TokenBudgetContract` declares the segments (each an integer token
count in a finite range), ``assume`` obligations (bounds known about the input)
and ``guarantee`` obligations (linear inequalities that must always hold).  Each
obligation is a linear comparison ``left op right`` where ``left``/``right`` are
sums of ``coefficient * segment`` terms and integer constants.  Verification
proves ``assumptions => guarantee`` for every admissible packing by asking the
solver whether ``assumptions and not guarantee`` is satisfiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .formal import (
    Eq,
    Expression,
    FiniteContractProblem,
    Ge,
    Gt,
    IntRangeDomain,
    Le,
    Lt,
    Mul,
    NamedConstraint,
    Ne,
    Not,
    SolverStatus,
    Sum,
    Value,
    Var,
    VariableDomain,
)

TOKEN_BUDGET_VERSION = "promptabi.token-budget-arithmetic.v1"

_OPS = {"<=": Le, "<": Lt, ">=": Ge, ">": Gt, "==": Eq, "!=": Ne}


class TokenBudgetProofStatus(StrEnum):
    PROVEN = "proven"
    REFUTED = "refuted"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class TokenSegment:
    """A named token-consuming segment with a finite count range."""

    name: str
    minimum: int = 0
    maximum: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("token segment name must be non-empty")
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("token segment requires 0 <= minimum <= maximum")

    def domain(self) -> VariableDomain:
        return IntRangeDomain(name=self.name, minimum=self.minimum, maximum=self.maximum)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class LinearTerm:
    """One ``coefficient * segment`` term or, when ``segment`` is empty, a constant."""

    coefficient: int = 1
    segment: str = ""
    const: int = 0

    def expression(self) -> Expression:
        if self.segment:
            if self.coefficient == 1:
                return Var(self.segment)
            return Mul(self.coefficient, Var(self.segment))
        return Value(self.const)

    def to_dict(self) -> dict[str, object]:
        if self.segment:
            return {"segment": self.segment, "coefficient": self.coefficient}
        return {"const": self.const}


@dataclass(frozen=True, slots=True)
class TokenBudgetObligation:
    """A linear comparison ``left op right`` over packed token counts."""

    name: str
    left: tuple[LinearTerm, ...]
    op: str
    right: tuple[LinearTerm, ...]

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unsupported token-budget op: {self.op!r}")

    def expression(self) -> Expression:
        op_cls = _OPS[self.op]
        return op_cls(_linear_sum(self.left), _linear_sum(self.right))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "left": [term.to_dict() for term in self.left],
            "op": self.op,
            "right": [term.to_dict() for term in self.right],
        }


@dataclass(frozen=True, slots=True)
class TokenBudgetContract:
    name: str
    segments: tuple[TokenSegment, ...]
    assumptions: tuple[TokenBudgetObligation, ...] = ()
    guarantees: tuple[TokenBudgetObligation, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "segments": [segment.to_dict() for segment in self.segments],
            "assumptions": [obligation.to_dict() for obligation in self.assumptions],
            "guarantees": [obligation.to_dict() for obligation in self.guarantees],
        }


@dataclass(frozen=True, slots=True)
class TokenBudgetGuaranteeResult:
    guarantee: str
    status: TokenBudgetProofStatus
    counterexample: Mapping[str, int] | None
    checked_assignments: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "guarantee": self.guarantee,
            "status": self.status.value,
            "checked_assignments": self.checked_assignments,
        }
        if self.counterexample is not None:
            data["counterexample"] = dict(self.counterexample)
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class TokenBudgetProofReport:
    version: str
    contract: str
    results: tuple[TokenBudgetGuaranteeResult, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return all(result.status is TokenBudgetProofStatus.PROVEN for result in self.results)

    @property
    def refuted(self) -> tuple[TokenBudgetGuaranteeResult, ...]:
        return tuple(r for r in self.results if r.status is TokenBudgetProofStatus.REFUTED)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "contract": self.contract,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def _linear_sum(terms: tuple[LinearTerm, ...]) -> Expression:
    if not terms:
        return Value(0)
    return Sum(*[term.expression() for term in terms])


def compile_token_budget_problem(
    contract: TokenBudgetContract,
    negated_guarantee: TokenBudgetObligation,
) -> FiniteContractProblem:
    """Compile segments, assumptions and a negated guarantee into a problem.

    Satisfiable exactly when some admissible packing meets every assumption yet
    violates ``negated_guarantee`` -- i.e. a concrete over-budget packing.
    """

    variables = tuple(segment.domain() for segment in contract.segments)
    constraints: list[NamedConstraint] = [
        NamedConstraint(
            name=f"{contract.name}-assume-{assumption.name}",
            expression=assumption.expression(),
        )
        for assumption in contract.assumptions
    ]
    constraints.append(
        NamedConstraint(
            name=f"{contract.name}-violates-{negated_guarantee.name}",
            expression=Not(negated_guarantee.expression()),
        )
    )
    return FiniteContractProblem(
        variables=variables,
        constraints=tuple(constraints),
        name=f"{contract.name}-token-budget",
    )


def _verify_guarantee(
    contract: TokenBudgetContract,
    guarantee: TokenBudgetObligation,
    *,
    prefer_z3: bool,
    timeout_seconds: float | None,
) -> TokenBudgetGuaranteeResult:
    problem = compile_token_budget_problem(contract, guarantee)
    result = problem.solve(prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
    if result.status is SolverStatus.UNSAT:
        return TokenBudgetGuaranteeResult(
            guarantee=guarantee.name,
            status=TokenBudgetProofStatus.PROVEN,
            counterexample=None,
            checked_assignments=result.checked_assignments,
        )
    if result.status is SolverStatus.SAT:
        counter = {
            segment.name: int(_finite_int((result.assignment or {}).get(segment.name)))
            for segment in contract.segments
        }
        return TokenBudgetGuaranteeResult(
            guarantee=guarantee.name,
            status=TokenBudgetProofStatus.REFUTED,
            counterexample=counter,
            checked_assignments=result.checked_assignments,
        )
    return TokenBudgetGuaranteeResult(
        guarantee=guarantee.name,
        status=TokenBudgetProofStatus.ABSTAINED,
        counterexample=None,
        checked_assignments=result.checked_assignments,
        reason=result.reason,
    )


def _finite_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0


def verify_token_budget_contract(
    contract: TokenBudgetContract,
    *,
    prefer_z3: bool = True,
    timeout_seconds: float | None = None,
) -> TokenBudgetProofReport:
    """Prove every linear guarantee holds for all admissible packings."""

    results = [
        _verify_guarantee(contract, guarantee, prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
        for guarantee in contract.guarantees
    ]
    return TokenBudgetProofReport(
        version=TOKEN_BUDGET_VERSION,
        contract=contract.name,
        results=tuple(results),
    )


def _term_from_dict(data: Mapping[str, object]) -> LinearTerm:
    if "const" in data:
        return LinearTerm(const=int(data["const"]))  # type: ignore[arg-type]
    if "segment" in data:
        return LinearTerm(segment=str(data["segment"]), coefficient=int(data.get("coefficient", 1)))
    raise ValueError("linear term must declare 'segment' or 'const'")


def _terms(data: Mapping[str, object], key: str) -> tuple[LinearTerm, ...]:
    raw = data.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"token-budget obligation '{key}' must be a list")
    return tuple(_term_from_dict(item) for item in raw)


def _obligation_from_dict(data: Mapping[str, object]) -> TokenBudgetObligation:
    return TokenBudgetObligation(
        name=str(data.get("name")),
        left=_terms(data, "left"),
        op=str(data.get("op")),
        right=_terms(data, "right"),
    )


def token_budget_contract_from_dict(data: Mapping[str, object]) -> TokenBudgetContract:
    if not isinstance(data, Mapping):
        raise ValueError("token-budget contract must be a JSON object")
    segments_raw = data.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError("token-budget contract requires a non-empty 'segments' list")
    segments = tuple(
        TokenSegment(
            name=str(item.get("name")),
            minimum=int(item.get("minimum", 0)),
            maximum=int(item.get("maximum", 0)),
        )
        for item in segments_raw
    )
    return TokenBudgetContract(
        name=str(data.get("name", "token-budget")),
        segments=segments,
        assumptions=tuple(_obligation_from_dict(item) for item in data.get("assumptions", []) or []),
        guarantees=tuple(_obligation_from_dict(item) for item in data.get("guarantees", []) or []),
    )


def load_token_budget_contract(path: str) -> TokenBudgetContract:
    return token_budget_contract_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def render_token_budget_report_json(report: TokenBudgetProofReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_token_budget_report_text(report: TokenBudgetProofReport) -> str:
    lines = [
        f"PromptABI token-budget contract '{report.contract}' ({report.version})",
        f"status: {'PROVEN' if report.ok else 'VIOLATED'}",
        f"guarantees: {len(report.results)}",
    ]
    for result in report.results:
        lines.append("")
        lines.append(f"{result.guarantee}: {result.status.value}")
        if result.counterexample is not None:
            lines.append("  over-budget packing: " + json.dumps(result.counterexample, sort_keys=True))
        if result.reason:
            lines.append(f"  reason: {result.reason}")
    return "\n".join(lines) + "\n"
