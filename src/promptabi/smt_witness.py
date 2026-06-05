"""Minimize SMT models into human-readable witnesses (step 226).

A raw SMT model is a total assignment to every declared variable -- including
variables that the violated/active constraints never mention, and concrete
values where a whole *range* of values would have served equally well.  When
that model is the counterexample in a bug report, the noise hides the actual
reason the contract failed.

This module turns a satisfying assignment (a counterexample to a negated
guarantee) into a *minimized, human-readable witness*:

* **Structural don't-cares are dropped soundly.**  A variable that appears in no
  constraint cannot affect satisfaction, so it is removed from the witness.  Any
  completion of the partial assignment over the dropped variables still
  satisfies every constraint -- a property the witness asserts and re-checks.
* **Relevant variables are generalized.**  For each kept variable the witness
  records the *flexible set* of alternative values that, holding the other
  variables at their model values, still satisfy every constraint.  This shows a
  reader how much the value matters versus how much was an arbitrary choice.
* **A plain-language narrative is emitted** so the witness reads as sentences,
  not as an opaque dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .formal import (
    FiniteContractProblem,
    SolverResult,
    SolverStatus,
    _expression_variables,
)

SMT_WITNESS_VERSION = "promptabi.smt-witness.v1"


class SmtWitnessError(ValueError):
    """Raised when a model cannot be minimized into a witness."""


@dataclass(frozen=True, slots=True)
class WitnessVariable:
    name: str
    value: object
    constraints: tuple[str, ...]
    flexible_values: tuple[object, ...]

    @property
    def pinned(self) -> bool:
        """True when no other value of this variable satisfies the model."""

        return len(self.flexible_values) <= 1

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "constraints": list(self.constraints),
            "flexible_values": list(self.flexible_values),
            "pinned": self.pinned,
        }


@dataclass(frozen=True, slots=True)
class MinimizedSmtWitness:
    version: str
    problem: str
    relevant: tuple[WitnessVariable, ...]
    omitted: tuple[str, ...]
    narrative: tuple[str, ...]
    full_assignment: Mapping[str, object] = field(default_factory=dict)

    @property
    def variables_dropped(self) -> int:
        return len(self.omitted)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "problem": self.problem,
            "relevant": [variable.to_dict() for variable in self.relevant],
            "omitted": list(self.omitted),
            "variables_dropped": self.variables_dropped,
            "narrative": list(self.narrative),
        }


def _constraint_index(problem: FiniteContractProblem) -> dict[str, frozenset[str]]:
    """Map each variable name to the set of constraints that reference it."""

    index: dict[str, set[str]] = {variable.name: set() for variable in problem.variables}
    for constraint in problem.constraints:
        for name in _expression_variables(constraint.expression):
            index.setdefault(name, set()).add(constraint.name)
    return {name: frozenset(constraints) for name, constraints in index.items()}


def _satisfies(problem: FiniteContractProblem, assignment: Mapping[str, object]) -> bool:
    return all(bool(constraint.expression.evaluate(assignment)) for constraint in problem.constraints)


def _flexible_values(
    problem: FiniteContractProblem,
    domain_values: Sequence[object],
    name: str,
    assignment: Mapping[str, object],
) -> tuple[object, ...]:
    """Alternative values for ``name`` that still satisfy the model."""

    working = dict(assignment)
    flexible: list[object] = []
    for value in domain_values:
        working[name] = value
        if _satisfies(problem, working):
            flexible.append(value)
    working[name] = assignment[name]
    return tuple(flexible)


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def minimize_smt_model(
    problem: FiniteContractProblem,
    result: SolverResult | None = None,
    *,
    prefer_z3: bool = True,
    flexible_value_cap: int = 64,
) -> MinimizedSmtWitness:
    """Minimize a satisfying SMT assignment into a human-readable witness."""

    solver_result = result if result is not None else problem.solve(prefer_z3=prefer_z3)
    if solver_result.status is not SolverStatus.SAT or solver_result.assignment is None:
        raise SmtWitnessError("no SAT model to minimize (the contract had no counterexample)")
    assignment = dict(solver_result.assignment)
    declared = {variable.name for variable in problem.variables}
    if set(assignment) != declared:
        raise SmtWitnessError("model does not assign exactly the declared variables")
    if not _satisfies(problem, assignment):
        raise SmtWitnessError("model does not satisfy all constraints")

    index = _constraint_index(problem)
    domains = {variable.name: variable.values() for variable in problem.variables}

    relevant: list[WitnessVariable] = []
    omitted: list[str] = []
    for variable in problem.variables:
        constraints = index.get(variable.name, frozenset())
        if not constraints:
            omitted.append(variable.name)
            continue
        values = domains[variable.name]
        flexible = (
            _flexible_values(problem, values, variable.name, assignment)
            if len(values) <= flexible_value_cap
            else (assignment[variable.name],)
        )
        relevant.append(
            WitnessVariable(
                name=variable.name,
                value=assignment[variable.name],
                constraints=tuple(sorted(constraints)),
                flexible_values=flexible,
            )
        )

    # Soundness: dropped variables must be absent from every constraint, so any
    # completion of the partial witness still satisfies the contract.
    for name in omitted:
        if index.get(name):
            raise SmtWitnessError(f"variable {name!r} was dropped but appears in a constraint")

    narrative = _build_narrative(problem.name, relevant, omitted)
    return MinimizedSmtWitness(
        version=SMT_WITNESS_VERSION,
        problem=problem.name,
        relevant=tuple(relevant),
        omitted=tuple(sorted(omitted)),
        narrative=narrative,
        full_assignment=assignment,
    )


def _build_narrative(
    problem_name: str,
    relevant: Sequence[WitnessVariable],
    omitted: Sequence[str],
) -> tuple[str, ...]:
    lines = [f"Counterexample for {problem_name}: the contract is satisfiable when"]
    for variable in relevant:
        descriptor = "must be" if variable.pinned else "can be"
        clause = f"  - {variable.name} {descriptor} {_render_value(variable.value)}"
        if not variable.pinned:
            others = [v for v in variable.flexible_values if v != variable.value]
            if 1 <= len(others) <= 4:
                rendered = ", ".join(_render_value(v) for v in others)
                clause += f" (also works: {rendered})"
            else:
                clause += f" ({len(variable.flexible_values)} values work)"
        clause += f"; constrains {', '.join(variable.constraints)}"
        lines.append(clause)
    if omitted:
        lines.append(f"  - irrelevant (any value): {', '.join(sorted(omitted))}")
    return tuple(lines)


def render_smt_witness_json(witness: MinimizedSmtWitness) -> str:
    return json.dumps(witness.to_dict(), indent=2, sort_keys=True) + "\n"


def render_smt_witness_text(witness: MinimizedSmtWitness) -> str:
    lines = [
        f"PromptABI minimized SMT witness ({witness.version})",
        f"problem: {witness.problem}",
        f"relevant variables: {len(witness.relevant)}, dropped: {witness.variables_dropped}",
    ]
    lines.extend(witness.narrative)
    return "\n".join(lines) + "\n"
