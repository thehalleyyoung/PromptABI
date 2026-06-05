"""Bounded arrays for PromptABI static contracts.

PromptABI's finite-contract solver (:mod:`promptabi.formal`) reasons over scalar
Bool, enum, int, and bounded-string variables.  Many real prompt/ABI properties
range over *sequences*: the messages in a chat, the tool calls in a turn, the
stop sequences in a policy, the token budgets of each segment.  This module adds
**bounded arrays** to that solver: an array of at most ``max_length`` elements
drawn from a finite element domain, with universally- and existentially-
quantified element predicates compiled down to the existing finite/Z3 backend by
unrolling each bounded quantifier over the concrete indices.

A :class:`BoundedArrayContract` declares the element domain, the length bound,
``assume`` obligations (facts taken as given about every array) and ``guarantee``
obligations (properties to prove).  Verification proves, for **every** array the
contract admits, that ``assumptions => guarantee`` by asking the real solver
whether ``assumptions and not guarantee`` is satisfiable: ``UNSAT`` is a proof,
``SAT`` returns a concrete violating array, and an unsupported fragment abstains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from .formal import (
    And,
    BoolDomain,
    Eq,
    Expression,
    FiniteContractProblem,
    Ge,
    Gt,
    Implies,
    InSet,
    IntRangeDomain,
    Le,
    Lt,
    Ne,
    NamedConstraint,
    Not,
    Or,
    SolverStatus,
    Value,
    Var,
    VariableDomain,
    EnumDomain,
)


BOUNDED_ARRAY_CONTRACT_VERSION = "promptabi.bounded-array-contracts.v1"

_COMPARISON_OPS = {"<=": Le, "<": Lt, ">=": Ge, ">": Gt}
_EQUALITY_OPS = {"==": Eq, "!=": Ne}
_ORDERING_OPS = frozenset({"<=", "<", ">=", ">"})


class BoundedArrayElementKind(StrEnum):
    """Element domain kinds supported for bounded arrays."""

    INT = "int"
    ENUM = "enum"
    BOOL = "bool"


class BoundedArrayObligationKind(StrEnum):
    """Quantified obligations expressible over a bounded array."""

    ALL_COMPARE = "all-compare"
    ALL_IN = "all-in"
    EXISTS_COMPARE = "exists-compare"
    SORTED_ASCENDING = "sorted-ascending"
    SORTED_DESCENDING = "sorted-descending"
    DISTINCT = "distinct"
    NON_EMPTY = "non-empty"


class BoundedArrayProofStatus(StrEnum):
    """Outcome of verifying one guarantee against a bounded-array contract."""

    PROVEN = "proven"
    REFUTED = "refuted"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class BoundedArrayElementDomain:
    """The finite domain every array element is drawn from."""

    kind: BoundedArrayElementKind
    minimum: int = 0
    maximum: int = 0
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is BoundedArrayElementKind.INT and self.minimum > self.maximum:
            raise ValueError("int element domain requires minimum <= maximum")
        if self.kind is BoundedArrayElementKind.ENUM and not self.members:
            raise ValueError("enum element domain requires at least one member")

    def variable(self, name: str) -> VariableDomain:
        if self.kind is BoundedArrayElementKind.INT:
            return IntRangeDomain(name=name, minimum=self.minimum, maximum=self.maximum)
        if self.kind is BoundedArrayElementKind.ENUM:
            return EnumDomain(name=name, members=self.members)
        return BoolDomain(name=name)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind.value}
        if self.kind is BoundedArrayElementKind.INT:
            data["minimum"] = self.minimum
            data["maximum"] = self.maximum
        elif self.kind is BoundedArrayElementKind.ENUM:
            data["members"] = list(self.members)
        return data


@dataclass(frozen=True, slots=True)
class BoundedArrayObligation:
    """A quantified predicate over a bounded array (an assumption or guarantee)."""

    name: str
    kind: BoundedArrayObligationKind
    op: str | None = None
    value: int | str | bool | None = None
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bounded-array obligation name must be non-empty")
        if self.kind in (
            BoundedArrayObligationKind.ALL_COMPARE,
            BoundedArrayObligationKind.EXISTS_COMPARE,
        ):
            if self.op not in (_COMPARISON_OPS.keys() | _EQUALITY_OPS.keys()):
                raise ValueError(f"comparison obligation requires a valid op, got {self.op!r}")
            if self.value is None:
                raise ValueError("comparison obligation requires a value")
        if self.kind is BoundedArrayObligationKind.ALL_IN and not self.members:
            raise ValueError("all-in obligation requires members")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "kind": self.kind.value}
        if self.op is not None:
            data["op"] = self.op
        if self.value is not None:
            data["value"] = self.value
        if self.members:
            data["members"] = list(self.members)
        return data


@dataclass(frozen=True, slots=True)
class BoundedArrayContract:
    """A bounded array with assumed and guaranteed quantified obligations."""

    name: str
    element_domain: BoundedArrayElementDomain
    max_length: int
    min_length: int = 0
    assumptions: tuple[BoundedArrayObligation, ...] = ()
    guarantees: tuple[BoundedArrayObligation, ...] = ()

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("bounded array max_length must be positive")
        if not 0 <= self.min_length <= self.max_length:
            raise ValueError("bounded array requires 0 <= min_length <= max_length")
        self._validate_obligations()

    def _validate_obligations(self) -> None:
        for obligation in (*self.assumptions, *self.guarantees):
            ordering = obligation.kind in (
                BoundedArrayObligationKind.SORTED_ASCENDING,
                BoundedArrayObligationKind.SORTED_DESCENDING,
            ) or (obligation.op in _ORDERING_OPS)
            if ordering and self.element_domain.kind is not BoundedArrayElementKind.INT:
                raise ValueError(
                    f"obligation '{obligation.name}' uses ordering but element domain is not int"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "element_domain": self.element_domain.to_dict(),
            "max_length": self.max_length,
            "min_length": self.min_length,
            "assumptions": [obligation.to_dict() for obligation in self.assumptions],
            "guarantees": [obligation.to_dict() for obligation in self.guarantees],
        }


@dataclass(frozen=True, slots=True)
class BoundedArrayGuaranteeResult:
    """Result of verifying one guarantee for a bounded-array contract."""

    guarantee: str
    status: BoundedArrayProofStatus
    counterexample: tuple[object, ...] | None
    checked_assignments: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "guarantee": self.guarantee,
            "status": self.status.value,
            "checked_assignments": self.checked_assignments,
        }
        if self.counterexample is not None:
            data["counterexample"] = list(self.counterexample)
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class BoundedArrayContractReport:
    """Verification report for every guarantee in a bounded-array contract."""

    version: str
    contract: str
    results: tuple[BoundedArrayGuaranteeResult, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return all(result.status is BoundedArrayProofStatus.PROVEN for result in self.results)

    @property
    def refuted(self) -> tuple[BoundedArrayGuaranteeResult, ...]:
        return tuple(r for r in self.results if r.status is BoundedArrayProofStatus.REFUTED)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "contract": self.contract,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def verify_bounded_array_contract(
    contract: BoundedArrayContract,
    *,
    prefer_z3: bool = True,
    timeout_seconds: float | None = None,
) -> BoundedArrayContractReport:
    """Prove every guarantee holds for all arrays the contract admits."""

    results = [
        _verify_guarantee(contract, guarantee, prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
        for guarantee in contract.guarantees
    ]
    return BoundedArrayContractReport(
        version=BOUNDED_ARRAY_CONTRACT_VERSION,
        contract=contract.name,
        results=tuple(results),
    )


def compile_bounded_array_problem(
    contract: BoundedArrayContract,
    negated_guarantee: BoundedArrayObligation,
) -> FiniteContractProblem:
    """Compile a contract plus a negated guarantee into a finite-contract problem.

    The returned problem is satisfiable exactly when a bounded array satisfies
    every assumption yet violates ``negated_guarantee`` -- i.e. a counterexample.
    """

    element_vars = [contract.element_domain.variable(_elem(contract.name, i)) for i in range(contract.max_length)]
    presence_vars = [BoolDomain(name=_present(contract.name, i)) for i in range(contract.max_length)]
    variables: tuple[VariableDomain, ...] = tuple(element_vars) + tuple(presence_vars)

    constraints: list[NamedConstraint] = []
    for index, prefix in enumerate(_prefix_constraints(contract)):
        constraints.append(NamedConstraint(name=f"{contract.name}-prefix-{index}", expression=prefix))
    for assumption in contract.assumptions:
        constraints.append(
            NamedConstraint(
                name=f"{contract.name}-assume-{assumption.name}",
                expression=_obligation_expression(contract, assumption),
            )
        )
    constraints.append(
        NamedConstraint(
            name=f"{contract.name}-violates-{negated_guarantee.name}",
            expression=Not(_obligation_expression(contract, negated_guarantee)),
        )
    )
    return FiniteContractProblem(
        variables=variables,
        constraints=tuple(constraints),
        name=f"{contract.name}-bounded-array",
    )


def bounded_array_contract_from_dict(data: Mapping[str, object]) -> BoundedArrayContract:
    """Parse a bounded-array contract from a JSON-style mapping."""

    if not isinstance(data, Mapping):
        raise ValueError("bounded-array contract must be a JSON object")
    domain_raw = data.get("element_domain")
    if not isinstance(domain_raw, Mapping):
        raise ValueError("bounded-array contract requires an 'element_domain' object")
    element_domain = _element_domain_from_dict(domain_raw)
    return BoundedArrayContract(
        name=str(data.get("name", "bounded-array")),
        element_domain=element_domain,
        max_length=_required_int(data, "max_length"),
        min_length=int(data.get("min_length", 0)),
        assumptions=tuple(_obligation_from_dict(item) for item in _obligation_list(data, "assumptions")),
        guarantees=tuple(_obligation_from_dict(item) for item in _obligation_list(data, "guarantees")),
    )


def load_bounded_array_contract(path: str) -> BoundedArrayContract:
    """Load a bounded-array contract from a JSON file."""

    from pathlib import Path

    return bounded_array_contract_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _element_domain_from_dict(data: Mapping[str, object]) -> BoundedArrayElementDomain:
    kind = BoundedArrayElementKind(str(data.get("kind")))
    if kind is BoundedArrayElementKind.INT:
        return BoundedArrayElementDomain(
            kind=kind,
            minimum=_required_int(data, "minimum"),
            maximum=_required_int(data, "maximum"),
        )
    if kind is BoundedArrayElementKind.ENUM:
        members = data.get("members")
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise ValueError("enum element domain requires a list of string members")
        return BoundedArrayElementDomain(kind=kind, members=tuple(members))
    return BoundedArrayElementDomain(kind=kind)


def _obligation_from_dict(data: Mapping[str, object]) -> BoundedArrayObligation:
    if not isinstance(data, Mapping):
        raise ValueError("bounded-array obligation must be a JSON object")
    members = data.get("members")
    return BoundedArrayObligation(
        name=str(data.get("name")),
        kind=BoundedArrayObligationKind(str(data.get("kind"))),
        op=str(data["op"]) if "op" in data and data["op"] is not None else None,
        value=data.get("value"),
        members=tuple(members) if isinstance(members, list) else (),
    )


def _obligation_list(data: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = data.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"bounded-array contract '{key}' must be a list")
    return raw


def _required_int(data: Mapping[str, object], key: str) -> int:
    if key not in data:
        raise ValueError(f"bounded-array contract requires '{key}'")
    return int(data[key])  # type: ignore[arg-type]


def render_bounded_array_report_json(report: BoundedArrayContractReport) -> str:
    """Render the bounded-array verification report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_bounded_array_report_text(report: BoundedArrayContractReport) -> str:
    """Render the bounded-array verification report for CI logs and reviewers."""

    lines = [
        f"PromptABI bounded-array contract '{report.contract}' ({report.version})",
        f"status: {'PROVEN' if report.ok else 'VIOLATED'}",
        f"guarantees: {len(report.results)}",
    ]
    for result in report.results:
        lines.append("")
        lines.append(f"{result.guarantee}: {result.status.value}")
        if result.counterexample is not None:
            lines.append("  counterexample array: " + json.dumps(list(result.counterexample)))
        if result.reason:
            lines.append(f"  reason: {result.reason}")
    return "\n".join(lines) + "\n"


def _verify_guarantee(
    contract: BoundedArrayContract,
    guarantee: BoundedArrayObligation,
    *,
    prefer_z3: bool,
    timeout_seconds: float | None,
) -> BoundedArrayGuaranteeResult:
    problem = compile_bounded_array_problem(contract, guarantee)
    result = problem.solve(prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
    if result.status is SolverStatus.UNSAT:
        return BoundedArrayGuaranteeResult(
            guarantee=guarantee.name,
            status=BoundedArrayProofStatus.PROVEN,
            counterexample=None,
            checked_assignments=result.checked_assignments,
        )
    if result.status is SolverStatus.SAT:
        return BoundedArrayGuaranteeResult(
            guarantee=guarantee.name,
            status=BoundedArrayProofStatus.REFUTED,
            counterexample=_reconstruct_array(contract, result.assignment or {}),
            checked_assignments=result.checked_assignments,
        )
    return BoundedArrayGuaranteeResult(
        guarantee=guarantee.name,
        status=BoundedArrayProofStatus.ABSTAINED,
        counterexample=None,
        checked_assignments=result.checked_assignments,
        reason=result.reason,
    )


def _prefix_constraints(contract: BoundedArrayContract) -> list[Expression]:
    constraints: list[Expression] = []
    # Present indices form a prefix: present[i+1] => present[i].
    for index in range(contract.max_length - 1):
        constraints.append(
            Implies(_present_term(contract.name, index + 1), _present_term(contract.name, index))
        )
    # Enforce the minimum length by forcing the first min_length indices present.
    for index in range(contract.min_length):
        constraints.append(Eq(_present_term(contract.name, index), Value(True)))
    return constraints


def _obligation_expression(
    contract: BoundedArrayContract,
    obligation: BoundedArrayObligation,
) -> Expression:
    name = contract.name
    n = contract.max_length
    kind = obligation.kind
    if kind is BoundedArrayObligationKind.NON_EMPTY:
        return _present_term(name, 0)
    if kind is BoundedArrayObligationKind.ALL_COMPARE:
        op_cls = _op_class(obligation.op)
        return And(
            *[
                Implies(_present_term(name, i), op_cls(_elem_term(name, i), Value(obligation.value)))
                for i in range(n)
            ]
        )
    if kind is BoundedArrayObligationKind.EXISTS_COMPARE:
        op_cls = _op_class(obligation.op)
        return Or(
            *[
                And(_present_term(name, i), op_cls(_elem_term(name, i), Value(obligation.value)))
                for i in range(n)
            ]
        )
    if kind is BoundedArrayObligationKind.ALL_IN:
        return And(
            *[
                Implies(_present_term(name, i), InSet(_elem_term(name, i), obligation.members))
                for i in range(n)
            ]
        )
    if kind is BoundedArrayObligationKind.SORTED_ASCENDING:
        return And(
            *[
                Implies(
                    And(_present_term(name, i), _present_term(name, i + 1)),
                    Le(_elem_term(name, i), _elem_term(name, i + 1)),
                )
                for i in range(n - 1)
            ]
        )
    if kind is BoundedArrayObligationKind.SORTED_DESCENDING:
        return And(
            *[
                Implies(
                    And(_present_term(name, i), _present_term(name, i + 1)),
                    Ge(_elem_term(name, i), _elem_term(name, i + 1)),
                )
                for i in range(n - 1)
            ]
        )
    if kind is BoundedArrayObligationKind.DISTINCT:
        terms: list[Expression] = []
        for i in range(n):
            for j in range(i + 1, n):
                terms.append(
                    Implies(
                        And(_present_term(name, i), _present_term(name, j)),
                        Ne(_elem_term(name, i), _elem_term(name, j)),
                    )
                )
        return And(*terms)
    raise ValueError(f"unsupported bounded-array obligation kind: {kind}")


def _reconstruct_array(
    contract: BoundedArrayContract,
    assignment: Mapping[str, object],
) -> tuple[object, ...]:
    elements: list[object] = []
    for index in range(contract.max_length):
        present = assignment.get(_present(contract.name, index))
        if not _is_true(present):
            continue
        elements.append(assignment.get(_elem(contract.name, index)))
    return tuple(elements)


def _op_class(op: str | None):
    if op in _COMPARISON_OPS:
        return _COMPARISON_OPS[op]
    if op in _EQUALITY_OPS:
        return _EQUALITY_OPS[op]
    raise ValueError(f"unsupported comparison op: {op!r}")


def _elem(name: str, index: int) -> str:
    return f"{name}__elem_{index}"


def _present(name: str, index: int) -> str:
    return f"{name}__present_{index}"


def _elem_term(name: str, index: int) -> Expression:
    return Var(_elem(name, index))


def _present_term(name: str, index: int) -> Expression:
    return Var(_present(name, index))


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"
