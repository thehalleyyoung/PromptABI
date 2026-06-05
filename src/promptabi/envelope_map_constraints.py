"""Finite map constraints for provider envelopes (step 222).

Provider request/response *envelopes* are finite maps: a request body maps a
fixed set of keys (``model``, ``temperature``, ``max_tokens``, ``stream``,
``tool_choice``, ``response_format`` ...) to values drawn from small, finite
domains, and many keys are optional.  Real ABI breakages live in the
*relationships* between those keys: "if ``stream`` is true the envelope must not
also request ``logprobs``", "``tool_choice`` may only be set when ``tools`` is
present", "``temperature`` must lie in ``[0, 2]`` whenever it is present".

This module encodes a provider envelope as a finite map over the existing
finite-contract solver (:mod:`promptabi.formal`): each field becomes a typed
variable plus a ``present`` flag, and each obligation compiles to a constraint.
Verification proves ``assumptions => guarantee`` for **every** envelope the
schema admits by asking the solver whether ``assumptions and not guarantee`` is
satisfiable -- ``UNSAT`` is a proof, ``SAT`` returns a concrete violating
envelope, and an unsupported fragment abstains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .formal import (
    And,
    BoolDomain,
    EnumDomain,
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
)

ENVELOPE_MAP_CONSTRAINTS_VERSION = "promptabi.envelope-map-constraints.v1"

_COMPARISON_OPS = {"<=": Le, "<": Lt, ">=": Ge, ">": Gt}
_EQUALITY_OPS = {"==": Eq, "!=": Ne}


class EnvelopeFieldKind(StrEnum):
    INT = "int"
    ENUM = "enum"
    BOOL = "bool"


class EnvelopeObligationKind(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    COMPARE = "compare"
    IN = "in"
    REQUIRES_TOGETHER = "requires-together"
    MUTUALLY_EXCLUSIVE = "mutually-exclusive"
    REQUIRES_ONE_OF = "requires-one-of"
    IMPLIES = "implies"


class EnvelopeProofStatus(StrEnum):
    PROVEN = "proven"
    REFUTED = "refuted"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class EnvelopeField:
    """One key of a provider envelope and the finite domain of its value."""

    name: str
    kind: EnvelopeFieldKind
    optional: bool = True
    minimum: int = 0
    maximum: int = 0
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("envelope field name must be non-empty")
        if self.kind is EnvelopeFieldKind.INT and self.minimum > self.maximum:
            raise ValueError("int envelope field requires minimum <= maximum")
        if self.kind is EnvelopeFieldKind.ENUM and not self.members:
            raise ValueError("enum envelope field requires at least one member")

    def value_domain(self) -> VariableDomain:
        if self.kind is EnvelopeFieldKind.INT:
            return IntRangeDomain(name=_value_var(self.name), minimum=self.minimum, maximum=self.maximum)
        if self.kind is EnvelopeFieldKind.ENUM:
            return EnumDomain(name=_value_var(self.name), members=self.members)
        return BoolDomain(name=_value_var(self.name))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "kind": self.kind.value, "optional": self.optional}
        if self.kind is EnvelopeFieldKind.INT:
            data["minimum"] = self.minimum
            data["maximum"] = self.maximum
        elif self.kind is EnvelopeFieldKind.ENUM:
            data["members"] = list(self.members)
        return data


@dataclass(frozen=True, slots=True)
class EnvelopePredicate:
    """An atomic predicate over one field (used inside ``implies`` obligations)."""

    kind: EnvelopeObligationKind
    field: str = ""
    op: str | None = None
    value: object | None = None
    members: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind.value, "field": self.field}
        if self.op is not None:
            data["op"] = self.op
        if self.value is not None:
            data["value"] = self.value
        if self.members:
            data["members"] = list(self.members)
        return data


@dataclass(frozen=True, slots=True)
class EnvelopeObligation:
    """One assumption or guarantee over an envelope's finite map."""

    name: str
    kind: EnvelopeObligationKind
    field: str = ""
    op: str | None = None
    value: object | None = None
    members: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    antecedent: EnvelopePredicate | None = None
    consequent: EnvelopePredicate | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "kind": self.kind.value}
        if self.field:
            data["field"] = self.field
        if self.op is not None:
            data["op"] = self.op
        if self.value is not None:
            data["value"] = self.value
        if self.members:
            data["members"] = list(self.members)
        if self.fields:
            data["fields"] = list(self.fields)
        if self.antecedent is not None:
            data["antecedent"] = self.antecedent.to_dict()
        if self.consequent is not None:
            data["consequent"] = self.consequent.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class EnvelopeContract:
    """A provider envelope schema with assume/guarantee obligations."""

    name: str
    fields: tuple[EnvelopeField, ...]
    assumptions: tuple[EnvelopeObligation, ...] = ()
    guarantees: tuple[EnvelopeObligation, ...] = ()

    def field_map(self) -> dict[str, EnvelopeField]:
        return {field_.name: field_ for field_ in self.fields}

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fields": [field_.to_dict() for field_ in self.fields],
            "assumptions": [obligation.to_dict() for obligation in self.assumptions],
            "guarantees": [obligation.to_dict() for obligation in self.guarantees],
        }


@dataclass(frozen=True, slots=True)
class EnvelopeGuaranteeResult:
    guarantee: str
    status: EnvelopeProofStatus
    counterexample: Mapping[str, object] | None
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
class EnvelopeContractReport:
    version: str
    contract: str
    results: tuple[EnvelopeGuaranteeResult, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return all(result.status is EnvelopeProofStatus.PROVEN for result in self.results)

    @property
    def refuted(self) -> tuple[EnvelopeGuaranteeResult, ...]:
        return tuple(r for r in self.results if r.status is EnvelopeProofStatus.REFUTED)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "contract": self.contract,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def _value_var(name: str) -> str:
    return f"value::{name}"


def _present_var(name: str) -> str:
    return f"present::{name}"


def _present_term(name: str) -> Expression:
    return Var(_present_var(name))


def _value_term(name: str) -> Expression:
    return Var(_value_var(name))


def _op_class(op: str | None):
    if op in _COMPARISON_OPS:
        return _COMPARISON_OPS[op]
    if op in _EQUALITY_OPS:
        return _EQUALITY_OPS[op]
    raise ValueError(f"unsupported comparison op: {op!r}")


def _predicate_holds(predicate: EnvelopePredicate) -> Expression:
    """Expression that is true when ``predicate`` holds in an envelope."""

    kind = predicate.kind
    if kind is EnvelopeObligationKind.PRESENT:
        return _present_term(predicate.field)
    if kind is EnvelopeObligationKind.ABSENT:
        return Not(_present_term(predicate.field))
    if kind is EnvelopeObligationKind.COMPARE:
        op_cls = _op_class(predicate.op)
        return And(_present_term(predicate.field), op_cls(_value_term(predicate.field), Value(predicate.value)))
    if kind is EnvelopeObligationKind.IN:
        return And(_present_term(predicate.field), InSet(_value_term(predicate.field), predicate.members))
    raise ValueError(f"unsupported envelope predicate kind: {kind}")


def _obligation_expression(obligation: EnvelopeObligation) -> Expression:
    kind = obligation.kind
    if kind is EnvelopeObligationKind.PRESENT:
        return _present_term(obligation.field)
    if kind is EnvelopeObligationKind.ABSENT:
        return Not(_present_term(obligation.field))
    if kind is EnvelopeObligationKind.COMPARE:
        op_cls = _op_class(obligation.op)
        return Implies(_present_term(obligation.field), op_cls(_value_term(obligation.field), Value(obligation.value)))
    if kind is EnvelopeObligationKind.IN:
        return Implies(_present_term(obligation.field), InSet(_value_term(obligation.field), obligation.members))
    if kind is EnvelopeObligationKind.REQUIRES_TOGETHER:
        if not obligation.field or not obligation.fields:
            raise ValueError("'requires-together' needs 'field' (trigger) and 'fields' (required companions)")
        return Implies(
            _present_term(obligation.field),
            And(*[_present_term(name) for name in obligation.fields]),
        )
    if kind is EnvelopeObligationKind.MUTUALLY_EXCLUSIVE:
        terms: list[Expression] = []
        names = obligation.fields
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                terms.append(Not(And(_present_term(names[i]), _present_term(names[j]))))
        return And(*terms) if terms else Value(True)
    if kind is EnvelopeObligationKind.REQUIRES_ONE_OF:
        return Or(*[_present_term(name) for name in obligation.fields]) if obligation.fields else Value(False)
    if kind is EnvelopeObligationKind.IMPLIES:
        if obligation.antecedent is None or obligation.consequent is None:
            raise ValueError("'implies' obligation requires antecedent and consequent")
        return Implies(_predicate_holds(obligation.antecedent), _predicate_holds(obligation.consequent))
    raise ValueError(f"unsupported envelope obligation kind: {kind}")


def compile_envelope_problem(
    contract: EnvelopeContract,
    negated_guarantee: EnvelopeObligation,
) -> FiniteContractProblem:
    """Compile the schema, assumptions and a negated guarantee into a problem.

    The problem is satisfiable exactly when some admissible envelope satisfies
    every assumption yet violates ``negated_guarantee``.
    """

    variables: list[VariableDomain] = []
    constraints: list[NamedConstraint] = []
    for field_ in contract.fields:
        variables.append(field_.value_domain())
        variables.append(BoolDomain(name=_present_var(field_.name)))
        if not field_.optional:
            constraints.append(
                NamedConstraint(
                    name=f"{contract.name}-required-{field_.name}",
                    expression=Eq(_present_term(field_.name), Value(True)),
                )
            )

    for assumption in contract.assumptions:
        constraints.append(
            NamedConstraint(
                name=f"{contract.name}-assume-{assumption.name}",
                expression=_obligation_expression(assumption),
            )
        )
    constraints.append(
        NamedConstraint(
            name=f"{contract.name}-violates-{negated_guarantee.name}",
            expression=Not(_obligation_expression(negated_guarantee)),
        )
    )
    return FiniteContractProblem(
        variables=tuple(variables),
        constraints=tuple(constraints),
        name=f"{contract.name}-envelope",
    )


def _reconstruct_envelope(contract: EnvelopeContract, assignment: Mapping[str, object]) -> dict[str, object]:
    envelope: dict[str, object] = {}
    for field_ in contract.fields:
        present = assignment.get(_present_var(field_.name))
        if present is True or present == "true" or present == 1:
            envelope[field_.name] = assignment.get(_value_var(field_.name))
    return envelope


def _verify_guarantee(
    contract: EnvelopeContract,
    guarantee: EnvelopeObligation,
    *,
    prefer_z3: bool,
    timeout_seconds: float | None,
) -> EnvelopeGuaranteeResult:
    problem = compile_envelope_problem(contract, guarantee)
    result = problem.solve(prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
    if result.status is SolverStatus.UNSAT:
        return EnvelopeGuaranteeResult(
            guarantee=guarantee.name,
            status=EnvelopeProofStatus.PROVEN,
            counterexample=None,
            checked_assignments=result.checked_assignments,
        )
    if result.status is SolverStatus.SAT:
        return EnvelopeGuaranteeResult(
            guarantee=guarantee.name,
            status=EnvelopeProofStatus.REFUTED,
            counterexample=_reconstruct_envelope(contract, result.assignment or {}),
            checked_assignments=result.checked_assignments,
        )
    return EnvelopeGuaranteeResult(
        guarantee=guarantee.name,
        status=EnvelopeProofStatus.ABSTAINED,
        counterexample=None,
        checked_assignments=result.checked_assignments,
        reason=result.reason,
    )


def verify_envelope_contract(
    contract: EnvelopeContract,
    *,
    prefer_z3: bool = True,
    timeout_seconds: float | None = None,
) -> EnvelopeContractReport:
    """Prove every guarantee holds for all envelopes the schema admits."""

    results = [
        _verify_guarantee(contract, guarantee, prefer_z3=prefer_z3, timeout_seconds=timeout_seconds)
        for guarantee in contract.guarantees
    ]
    return EnvelopeContractReport(
        version=ENVELOPE_MAP_CONSTRAINTS_VERSION,
        contract=contract.name,
        results=tuple(results),
    )


def _field_from_dict(data: Mapping[str, object]) -> EnvelopeField:
    kind = EnvelopeFieldKind(str(data.get("kind")))
    members = data.get("members")
    return EnvelopeField(
        name=str(data.get("name")),
        kind=kind,
        optional=bool(data.get("optional", True)),
        minimum=int(data.get("minimum", 0)),
        maximum=int(data.get("maximum", 0)),
        members=tuple(members) if isinstance(members, list) else (),
    )


def _predicate_from_dict(data: Mapping[str, object] | None) -> EnvelopePredicate | None:
    if data is None:
        return None
    members = data.get("members")
    return EnvelopePredicate(
        kind=EnvelopeObligationKind(str(data.get("kind"))),
        field=str(data.get("field", "")),
        op=str(data["op"]) if data.get("op") is not None else None,
        value=data.get("value"),
        members=tuple(members) if isinstance(members, list) else (),
    )


def _obligation_from_dict(data: Mapping[str, object]) -> EnvelopeObligation:
    members = data.get("members")
    fields = data.get("fields")
    antecedent = data.get("antecedent")
    consequent = data.get("consequent")
    return EnvelopeObligation(
        name=str(data.get("name")),
        kind=EnvelopeObligationKind(str(data.get("kind"))),
        field=str(data.get("field", "")),
        op=str(data["op"]) if data.get("op") is not None else None,
        value=data.get("value"),
        members=tuple(members) if isinstance(members, list) else (),
        fields=tuple(fields) if isinstance(fields, list) else (),
        antecedent=_predicate_from_dict(antecedent if isinstance(antecedent, Mapping) else None),
        consequent=_predicate_from_dict(consequent if isinstance(consequent, Mapping) else None),
    )


def envelope_contract_from_dict(data: Mapping[str, object]) -> EnvelopeContract:
    if not isinstance(data, Mapping):
        raise ValueError("envelope contract must be a JSON object")
    fields_raw = data.get("fields")
    if not isinstance(fields_raw, list) or not fields_raw:
        raise ValueError("envelope contract requires a non-empty 'fields' list")
    return EnvelopeContract(
        name=str(data.get("name", "envelope")),
        fields=tuple(_field_from_dict(item) for item in fields_raw),
        assumptions=tuple(_obligation_from_dict(item) for item in data.get("assumptions", []) or []),
        guarantees=tuple(_obligation_from_dict(item) for item in data.get("guarantees", []) or []),
    )


def load_envelope_contract(path: str) -> EnvelopeContract:
    return envelope_contract_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def render_envelope_report_json(report: EnvelopeContractReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_envelope_report_text(report: EnvelopeContractReport) -> str:
    lines = [
        f"PromptABI envelope contract '{report.contract}' ({report.version})",
        f"status: {'PROVEN' if report.ok else 'VIOLATED'}",
        f"guarantees: {len(report.results)}",
    ]
    for result in report.results:
        lines.append("")
        lines.append(f"{result.guarantee}: {result.status.value}")
        if result.counterexample is not None:
            lines.append("  counterexample envelope: " + json.dumps(result.counterexample, sort_keys=True))
        if result.reason:
            lines.append(f"  reason: {result.reason}")
    return "\n".join(lines) + "\n"
