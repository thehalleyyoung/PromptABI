"""Executable specifications for PromptABI's finite formal core.

The helpers in this module are intentionally small, bounded, and independent of
the optimized algorithms they check. They are meant for tests, reproducibility
artifacts, and paper claims: a result object is only trusted after a simple
reference predicate confirms the same witness, bounded language law, or solver
contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from .formal import (
    AutomatonWitness,
    BoolDomain,
    BoundedStringDomain,
    DeterministicFiniteAutomaton,
    EnumDomain,
    FiniteContractProblem,
    FiniteStateTransducer,
    IntRangeDomain,
    NamedConstraint,
    SolverConclusion,
    SolverResult,
    SolverStatus,
    TransducerLabel,
    TransducerTransition,
    TransducerWitness,
    VariableDomain,
)


@dataclass(frozen=True, slots=True)
class SpecCheck:
    """One executable-spec assertion."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ExecutableSpecReport:
    """A collection of executable-spec assertions."""

    checks: tuple[SpecCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[SpecCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


ProductOperation = str


def check_dfa_witness(
    automaton: DeterministicFiniteAutomaton,
    witness: AutomatonWitness | None,
    *,
    expected_empty: bool = False,
) -> ExecutableSpecReport:
    """Check a DFA reachability witness by replaying its state path."""

    checks: list[SpecCheck] = []
    if witness is None:
        checks.append(SpecCheck("witness-presence", expected_empty, "no witness returned"))
        return ExecutableSpecReport(tuple(checks))
    checks.append(SpecCheck("witness-presence", not expected_empty, f"text={witness.text!r}"))
    checks.append(SpecCheck("path-starts-at-start", witness.states[:1] == (automaton.start,), f"start={automaton.start!r}"))
    checks.append(
        SpecCheck(
            "path-length-matches-symbols",
            len(witness.states) == len(witness.symbols) + 1,
            f"states={len(witness.states)} symbols={len(witness.symbols)}",
        )
    )
    replay_state = witness.states[0] if witness.states else None
    transitions_valid = replay_state == automaton.start
    for index, symbol in enumerate(witness.symbols):
        if replay_state is None or index + 1 >= len(witness.states):
            transitions_valid = False
            break
        next_state = automaton.step(replay_state, symbol)
        if next_state != witness.states[index + 1]:
            transitions_valid = False
            break
        replay_state = next_state
    checks.append(SpecCheck("path-follows-transitions", transitions_valid, f"states={list(witness.states)}"))
    checks.append(SpecCheck("text-matches-symbols", witness.text == "".join(witness.symbols), f"text={witness.text!r}"))
    checks.append(SpecCheck("path-ends-accepting", replay_state in automaton.accepts, f"end={replay_state!r}"))
    checks.append(SpecCheck("automaton-accepts-witness", automaton.accepts_text(witness.text), f"text={witness.text!r}"))
    return ExecutableSpecReport(tuple(checks))


def check_dfa_product_language(
    product_automaton: DeterministicFiniteAutomaton,
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    *,
    operation: ProductOperation,
    max_depth: int = 4,
) -> ExecutableSpecReport:
    """Check a product DFA against bounded reference language semantics.

    This deliberately does not inspect product state names. It enumerates short
    words over the union alphabet and compares acceptance with the mathematical
    operation on the two operand languages.
    """

    operations: Mapping[str, Callable[[bool, bool], bool]] = {
        "intersection": lambda left_accepts, right_accepts: left_accepts and right_accepts,
        "union": lambda left_accepts, right_accepts: left_accepts or right_accepts,
        "difference": lambda left_accepts, right_accepts: left_accepts and not right_accepts,
    }
    if operation not in operations:
        return ExecutableSpecReport((SpecCheck("known-product-operation", False, operation),))
    alphabet = tuple(sorted(set(left.alphabet).union(right.alphabet).union(product_automaton.alphabet)))
    mismatches: list[str] = []
    for word in _bounded_words(alphabet, max_depth):
        expected = operations[operation](left.accepts_symbols(word), right.accepts_symbols(word))
        actual = product_automaton.accepts_symbols(word)
        if actual != expected:
            mismatches.append(f"{''.join(word)!r}: expected={expected} actual={actual}")
            if len(mismatches) >= 5:
                break
    return ExecutableSpecReport(
        (
            SpecCheck("known-product-operation", True, operation),
            SpecCheck(
                "bounded-product-language",
                not mismatches,
                f"checked_depth={max_depth} alphabet={''.join(alphabet)!r} mismatches={mismatches}",
            ),
        )
    )


def check_transducer_witness(
    transducer: FiniteStateTransducer,
    witness: TransducerWitness | None,
    *,
    expected_empty: bool = False,
) -> ExecutableSpecReport:
    """Check an FST witness by replaying epsilon-aware labels."""

    checks: list[SpecCheck] = []
    if witness is None:
        checks.append(SpecCheck("witness-presence", expected_empty, "no witness returned"))
        return ExecutableSpecReport(tuple(checks))
    checks.append(SpecCheck("witness-presence", not expected_empty, f"input={witness.input_text!r} output={witness.output_text!r}"))
    checks.append(SpecCheck("path-starts-at-start", witness.states[:1] == (transducer.start,), f"start={transducer.start!r}"))
    checks.append(
        SpecCheck(
            "path-length-matches-labels",
            len(witness.states) == len(witness.labels) + 1,
            f"states={len(witness.states)} labels={len(witness.labels)}",
        )
    )
    by_edge = _transducer_edges(transducer.transitions)
    input_symbols: list[str] = []
    output_symbols: list[str] = []
    transitions_valid = True
    for index, label in enumerate(witness.labels):
        if index + 1 >= len(witness.states):
            transitions_valid = False
            break
        source = witness.states[index]
        target = witness.states[index + 1]
        if (source, label.input_symbol, label.output_symbol, target) not in by_edge:
            transitions_valid = False
            break
        if label.input_symbol is not None:
            input_symbols.append(label.input_symbol)
        if label.output_symbol is not None:
            output_symbols.append(label.output_symbol)
    end_state = witness.states[-1] if witness.states else None
    checks.append(SpecCheck("path-follows-transitions", transitions_valid, f"states={list(witness.states)}"))
    checks.append(SpecCheck("input-reconstructed-from-labels", tuple(input_symbols) == witness.input_symbols, witness.input_text))
    checks.append(SpecCheck("output-reconstructed-from-labels", tuple(output_symbols) == witness.output_symbols, witness.output_text))
    checks.append(SpecCheck("path-ends-accepting", end_state in transducer.accepts, f"end={end_state!r}"))
    checks.append(
        SpecCheck(
            "transducer-accepts-witness",
            transducer.accepts_pair(witness.input_symbols, witness.output_symbols),
            f"input={witness.input_text!r} output={witness.output_text!r}",
        )
    )
    return ExecutableSpecReport(tuple(checks))


def check_contract_result(problem: FiniteContractProblem, result: SolverResult) -> ExecutableSpecReport:
    """Check finite-contract solver output against executable proof obligations."""

    checks: list[SpecCheck] = [
        SpecCheck(
            "status-conclusion-boundary",
            _status_matches_conclusion(result.status, result.conclusion),
            f"status={result.status.value} conclusion={result.conclusion.value}",
        )
    ]
    if result.status is SolverStatus.SAT:
        assignment = dict(result.assignment or {})
        checks.append(SpecCheck("sat-assignment-present", result.assignment is not None, repr(assignment)))
        checks.extend(_check_assignment_domains(problem.variables, assignment))
        checks.extend(_check_assignment_constraints(problem.constraints, assignment))
    elif result.status is SolverStatus.UNSAT:
        checks.append(SpecCheck("unsat-has-no-assignment", result.assignment is None, repr(result.assignment)))
        checks.append(SpecCheck("unsat-core-names-known", _unsat_core_names_known(problem, result), repr(result.unsat_core)))
        checks.append(SpecCheck("unsat-core-is-unsatisfiable", _constraints_unsat(problem, result.unsat_core), repr(result.unsat_core)))
        checks.append(SpecCheck("unsat-core-is-deletion-minimal", _unsat_core_minimal(problem, result.unsat_core), repr(result.unsat_core)))
    else:
        checks.append(SpecCheck("abstention-has-no-proof-object", result.assignment is None and not result.unsat_core, result.to_dict().__repr__()))
    return ExecutableSpecReport(tuple(checks))


def assert_spec_report(report: ExecutableSpecReport) -> None:
    """Raise AssertionError with compact details if an executable spec fails."""

    if report.passed:
        return
    details = "; ".join(f"{check.name}: {check.detail}" for check in report.failures)
    raise AssertionError(details)


def _bounded_words(alphabet: Sequence[str], max_depth: int) -> Iterable[tuple[str, ...]]:
    yield ()
    for length in range(1, max_depth + 1):
        yield from product(alphabet, repeat=length)


def _transducer_edges(transitions: Sequence[TransducerTransition]) -> frozenset[tuple[str, str | None, str | None, str]]:
    return frozenset(
        (transition.source, transition.label.input_symbol, transition.label.output_symbol, transition.target)
        for transition in transitions
    )


def _status_matches_conclusion(status: SolverStatus, conclusion: SolverConclusion) -> bool:
    return (
        (status is SolverStatus.SAT and conclusion is SolverConclusion.COUNTEREXAMPLE)
        or (status is SolverStatus.UNSAT and conclusion is SolverConclusion.UNSAT_CORE_PROOF)
        or (status is SolverStatus.UNKNOWN and conclusion is SolverConclusion.ABSTENTION)
    )


def _check_assignment_domains(variables: Sequence[VariableDomain], assignment: Mapping[str, object]) -> tuple[SpecCheck, ...]:
    checks: list[SpecCheck] = []
    assigned_names = set(assignment)
    expected_names = {variable.name for variable in variables}
    checks.append(SpecCheck("assignment-has-all-variables", assigned_names == expected_names, f"expected={sorted(expected_names)} actual={sorted(assigned_names)}"))
    for variable in variables:
        value = assignment.get(variable.name)
        checks.append(SpecCheck(f"domain:{variable.name}", _value_in_domain(variable, value), repr(value)))
    return tuple(checks)


def _value_in_domain(variable: VariableDomain, value: object) -> bool:
    if isinstance(variable, BoolDomain):
        return isinstance(value, bool)
    if isinstance(variable, EnumDomain):
        return isinstance(value, str) and value in variable.members
    if isinstance(variable, IntRangeDomain):
        return isinstance(value, int) and variable.minimum <= value <= variable.maximum
    if isinstance(variable, BoundedStringDomain):
        return (
            isinstance(value, str)
            and variable.min_length <= len(value) <= variable.max_length
            and all(symbol in variable.alphabet for symbol in value)
        )
    return value in variable.values()


def _check_assignment_constraints(constraints: Sequence[NamedConstraint], assignment: Mapping[str, object]) -> tuple[SpecCheck, ...]:
    return tuple(
        SpecCheck(f"constraint:{constraint.name}", bool(constraint.expression.evaluate(assignment)), repr(assignment))
        for constraint in constraints
    )


def _unsat_core_names_known(problem: FiniteContractProblem, result: SolverResult) -> bool:
    known = {constraint.name for constraint in problem.constraints}
    return bool(result.unsat_core) and set(result.unsat_core) <= known


def _constraints_unsat(problem: FiniteContractProblem, names: Sequence[str]) -> bool:
    constraints = tuple(constraint for constraint in problem.constraints if constraint.name in set(names))
    return not _satisfiable(problem.variables, constraints)


def _unsat_core_minimal(problem: FiniteContractProblem, names: Sequence[str]) -> bool:
    if not names:
        return False
    for name in names:
        candidate = tuple(item for item in names if item != name)
        if not _satisfiable(problem.variables, tuple(constraint for constraint in problem.constraints if constraint.name in set(candidate))):
            return False
    return True


def _satisfiable(variables: Sequence[VariableDomain], constraints: Sequence[NamedConstraint]) -> bool:
    for assignment in _bounded_assignments(variables):
        if all(bool(constraint.expression.evaluate(assignment)) for constraint in constraints):
            return True
    return False


def _bounded_assignments(variables: Sequence[VariableDomain]) -> Iterable[dict[str, object]]:
    for values in product(*(variable.values() for variable in variables)):
        yield {variable.name: value for variable, value in zip(variables, values, strict=True)}
