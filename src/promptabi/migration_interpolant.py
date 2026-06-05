"""Derive interpolants for failed migration checks (step 231).

A provider-migration check asks: is every request/response valid under the
*source* provider still valid under the *target* provider?  PromptABI encodes the
source-acceptance region as one finite contract ``A`` and the
target-rejection region as another finite contract ``B`` over the **shared**
interface variables (``content_present``, ``tool_calls_present``,
``finish_reason``, ...).

* If ``A ∧ B`` is unsatisfiable the migration is **safe**, and there is a Craig
  *interpolant*: a formula ``I`` over the shared variables with ``A ⊨ I`` and
  ``I ∧ B`` unsatisfiable.  PromptABI derives one from the Z3 unsat core (the
  source-side obligations that already exclude every rejected shape) so the
  migration proof can be reused as a reusable lemma.
* If ``A ∧ B`` is satisfiable the migration **fails**.  A single counterexample
  is noisy, so this module *generalizes* it into the largest sound cube over the
  shared variables -- the incompatibility region -- naming exactly which
  interface conditions break the migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence

from .formal import (
    FiniteContractProblem,
    NamedConstraint,
    SolverStatus,
    VariableDomain,
)

MIGRATION_INTERPOLANT_VERSION = "promptabi.migration-interpolant.v1"


class MigrationInterpolantError(ValueError):
    """Raised when two contracts cannot be combined over a shared vocabulary."""


class MigrationStatus(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass(frozen=True, slots=True)
class InterpolantLiteral:
    variable: str
    value: object

    def to_dict(self) -> dict[str, object]:
        return {"variable": self.variable, "value": self.value}

    def render(self) -> str:
        return f"{self.variable} == {self.value!r}"


@dataclass(frozen=True, slots=True)
class MigrationInterpolant:
    version: str
    status: MigrationStatus
    shared_variables: tuple[str, ...]
    # For SAFE: the source-side obligations (unsat-core constraint names) that
    # form the interpolant.  For UNSAFE: the generalized incompatibility cube.
    interpolant_terms: tuple[str, ...] = field(default=())
    incompatibility_cube: tuple[InterpolantLiteral, ...] = field(default=())
    witness: Mapping[str, object] | None = None

    @property
    def safe(self) -> bool:
        return self.status is MigrationStatus.SAFE

    def render_interpolant(self) -> str:
        if self.safe:
            if not self.interpolant_terms:
                return "true"
            return " ∧ ".join(self.interpolant_terms)
        if not self.incompatibility_cube:
            return "false"
        return " ∧ ".join(literal.render() for literal in self.incompatibility_cube)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "status": self.status.value,
            "safe": self.safe,
            "shared_variables": list(self.shared_variables),
            "interpolant_terms": list(self.interpolant_terms),
            "incompatibility_cube": [literal.to_dict() for literal in self.incompatibility_cube],
            "interpolant": self.render_interpolant(),
            "witness": dict(sorted(self.witness.items())) if self.witness else None,
        }


def _domain_key(domain: VariableDomain) -> tuple[str, object]:
    data = domain.to_dict()
    return (str(data.get("type")), json.dumps(data, sort_keys=True))


def combine_contracts(
    source: FiniteContractProblem,
    target_reject: FiniteContractProblem,
    *,
    name: str = "migration-combined",
) -> FiniteContractProblem:
    """Combine two contracts that share an identical variable vocabulary."""

    source_vars = {variable.name: variable for variable in source.variables}
    target_vars = {variable.name: variable for variable in target_reject.variables}
    if set(source_vars) != set(target_vars):
        raise MigrationInterpolantError(
            "source and target contracts must share the same variable vocabulary"
        )
    for variable_name, source_domain in source_vars.items():
        if _domain_key(source_domain) != _domain_key(target_vars[variable_name]):
            raise MigrationInterpolantError(
                f"variable {variable_name!r} has different domains in source and target"
            )
    constraints: list[NamedConstraint] = []
    for constraint in source.constraints:
        constraints.append(NamedConstraint(name=f"source:{constraint.name}", expression=constraint.expression))
    for constraint in target_reject.constraints:
        constraints.append(NamedConstraint(name=f"target:{constraint.name}", expression=constraint.expression))
    return FiniteContractProblem(
        variables=tuple(source.variables),
        constraints=tuple(constraints),
        name=name,
    )


def _satisfies(problem: FiniteContractProblem, assignment: Mapping[str, object]) -> bool:
    return all(bool(constraint.expression.evaluate(assignment)) for constraint in problem.constraints)


def _generalize_cube(
    problem: FiniteContractProblem,
    assignment: Mapping[str, object],
) -> tuple[InterpolantLiteral, ...]:
    """Largest sound cube: keep only variables whose value is necessary."""

    domains = {variable.name: variable.values() for variable in problem.variables}
    necessary: list[InterpolantLiteral] = []
    for variable in problem.variables:
        name = variable.name
        # A variable is a don't-care iff every value (others fixed at the
        # witness) keeps the combined formula satisfiable.
        working = dict(assignment)
        always_sat = True
        for value in domains[name]:
            working[name] = value
            if not _satisfies(problem, working):
                always_sat = False
                break
        working[name] = assignment[name]
        if not always_sat:
            necessary.append(InterpolantLiteral(variable=name, value=assignment[name]))
    return tuple(necessary)


def derive_migration_interpolant(
    source: FiniteContractProblem,
    target_reject: FiniteContractProblem,
    *,
    prefer_z3: bool = True,
) -> MigrationInterpolant:
    """Derive an interpolant (safe) or incompatibility cube (unsafe)."""

    combined = combine_contracts(source, target_reject)
    shared = tuple(variable.name for variable in combined.variables)
    result = combined.solve(prefer_z3=prefer_z3)

    if result.status is SolverStatus.UNSAT:
        core = tuple(name for name in result.unsat_core if name.startswith("source:"))
        if not core:
            # Fall back to all source obligations when the core is target-only
            # (still a valid interpolant: A entails its own constraints).
            core = tuple(f"source:{c.name}" for c in source.constraints)
        return MigrationInterpolant(
            version=MIGRATION_INTERPOLANT_VERSION,
            status=MigrationStatus.SAFE,
            shared_variables=shared,
            interpolant_terms=core,
        )

    if result.status is SolverStatus.SAT and result.assignment is not None:
        assignment = dict(result.assignment)
        cube = _generalize_cube(combined, assignment)
        return MigrationInterpolant(
            version=MIGRATION_INTERPOLANT_VERSION,
            status=MigrationStatus.UNSAFE,
            shared_variables=shared,
            incompatibility_cube=cube,
            witness=assignment,
        )

    raise MigrationInterpolantError(
        "combined migration contract was neither proven safe nor refuted "
        f"(solver returned {result.status.value})"
    )


def render_migration_interpolant_json(interpolant: MigrationInterpolant) -> str:
    return json.dumps(interpolant.to_dict(), indent=2, sort_keys=True) + "\n"


def render_migration_interpolant_text(interpolant: MigrationInterpolant) -> str:
    lines = [
        f"PromptABI migration interpolant ({interpolant.version})",
        f"status: {interpolant.status.value}",
        f"shared variables: {', '.join(interpolant.shared_variables)}",
        f"interpolant: {interpolant.render_interpolant()}",
    ]
    if not interpolant.safe and interpolant.witness:
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(interpolant.witness.items()))
        lines.append(f"witness: {rendered}")
    return "\n".join(lines) + "\n"
