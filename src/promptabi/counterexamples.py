"""Counterexample shrinking for formal PromptABI witnesses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from heapq import heappop, heappush
from typing import Any

from .diagnostics import WitnessStep, WitnessTrace
from .formal import (
    AutomatonWitness,
    DeterministicFiniteAutomaton,
    FiniteContractProblem,
    FiniteStateTransducer,
    SolverResult,
    SolverStatus,
    TransducerLabel,
    TransducerWitness,
    VariableDomain,
)


class CounterexampleShrinkError(ValueError):
    """Raised when a formal counterexample cannot be validated or shrunk."""


class CounterexampleMetric(StrEnum):
    """Supported minimality objectives for structural counterexamples."""

    STRING_LENGTH = "string-length"
    TOKEN_COUNT = "token-count"
    MESSAGE_COUNT = "message-count"
    SCHEMA_SIZE = "schema-size"
    RULE_COMPLEXITY = "rule-complexity"


@dataclass(frozen=True, slots=True)
class CounterexampleShrinkStep:
    """One proof-relevant action in a counterexample shrinking run."""

    action: str
    before_cost: int
    after_cost: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "before_cost": self.before_cost,
            "after_cost": self.after_cost,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CounterexampleShrinkReport:
    """A minimized counterexample with a compact minimality certificate."""

    kind: str
    metric: CounterexampleMetric
    original: Mapping[str, object]
    minimized: Mapping[str, object]
    original_cost: int
    minimized_cost: int
    steps: tuple[CounterexampleShrinkStep, ...]
    certificate: Mapping[str, object]

    @property
    def changed(self) -> bool:
        return dict(self.original) != dict(self.minimized)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "metric": self.metric.value,
            "changed": self.changed,
            "original_cost": self.original_cost,
            "minimized_cost": self.minimized_cost,
            "original": dict(self.original),
            "minimized": dict(self.minimized),
            "steps": [step.to_dict() for step in self.steps],
            "certificate": dict(self.certificate),
        }

    def witness(self) -> WitnessTrace:
        witness_steps = [
            WitnessStep(
                action="validate original counterexample",
                input=self.kind,
                output=f"{self.original_cost} {self.metric.value}",
            )
        ]
        witness_steps.extend(
            WitnessStep(
                action=step.action,
                input=f"{step.before_cost} {self.metric.value}",
                output=f"{step.after_cost} {self.metric.value}",
            )
            for step in self.steps
        )
        witness_steps.append(
            WitnessStep(
                action="certify minimized counterexample",
                input=self.kind,
                output=f"{self.minimized_cost} {self.metric.value}",
            )
        )
        return WitnessTrace(
            summary=(
                f"{self.kind} counterexample minimized from {self.original_cost} "
                f"to {self.minimized_cost} by {self.metric.value}"
            ),
            steps=tuple(witness_steps),
        )


def shrink_automaton_counterexample(
    automaton: DeterministicFiniteAutomaton,
    witness: AutomatonWitness | None = None,
    *,
    metric: CounterexampleMetric | str = CounterexampleMetric.STRING_LENGTH,
) -> CounterexampleShrinkReport:
    """Return the shortest accepted DFA counterexample and prove no shorter word exists."""

    shrink_metric = CounterexampleMetric(metric)
    original = witness or automaton.shortest_witness()
    if original is None or not automaton.accepts_symbols(original.symbols):
        raise CounterexampleShrinkError("DFA counterexample witness is absent or not accepted")

    minimized = _shortest_dfa_by_metric(automaton, shrink_metric)
    if minimized is None:
        raise CounterexampleShrinkError("DFA accepted the original witness but no minimized witness was found")

    original_payload = _automaton_payload(original)
    minimized_payload = _automaton_payload(minimized)
    original_cost = _symbol_cost(original.symbols, metric=shrink_metric)
    minimized_cost = _symbol_cost(minimized.symbols, metric=shrink_metric)
    steps = _steps_if_changed(
        action="replace with globally shortest accepted DFA path",
        before_cost=original_cost,
        after_cost=minimized_cost,
        detail=f"explored all reachable paths below cost {minimized_cost}",
        changed=original_payload != minimized_payload,
    )
    return CounterexampleShrinkReport(
        kind="automaton",
        metric=shrink_metric,
        original=original_payload,
        minimized=minimized_payload,
        original_cost=original_cost,
        minimized_cost=minimized_cost,
        steps=steps,
        certificate={
            "accepted": automaton.accepts_symbols(minimized.symbols),
            "minimality": "uniform-cost reachability over DFA states",
            "states": len(automaton.states),
            "alphabet_symbols": len(automaton.alphabet),
        },
    )


def shrink_transducer_counterexample(
    transducer: FiniteStateTransducer,
    witness: TransducerWitness | None = None,
    *,
    metric: CounterexampleMetric | str = CounterexampleMetric.TOKEN_COUNT,
) -> CounterexampleShrinkReport:
    """Return a minimal accepted FST input/output counterexample pair."""

    shrink_metric = CounterexampleMetric(metric)
    original = witness or transducer.shortest_witness()
    if original is None or not transducer.accepts_pair(original.input_symbols, original.output_symbols):
        raise CounterexampleShrinkError("transducer counterexample witness is absent or not accepted")

    minimized = _shortest_fst_by_metric(transducer, shrink_metric)
    if minimized is None:
        raise CounterexampleShrinkError("transducer accepted the original witness but no minimized witness was found")

    original_payload = _transducer_payload(original)
    minimized_payload = _transducer_payload(minimized)
    original_cost = _transducer_cost(original, metric=shrink_metric)
    minimized_cost = _transducer_cost(minimized, metric=shrink_metric)
    steps = _steps_if_changed(
        action="replace with globally cheapest accepted FST path",
        before_cost=original_cost,
        after_cost=minimized_cost,
        detail=f"uniform-cost search minimized {shrink_metric.value}",
        changed=original_payload != minimized_payload,
    )
    return CounterexampleShrinkReport(
        kind="transducer",
        metric=shrink_metric,
        original=original_payload,
        minimized=minimized_payload,
        original_cost=original_cost,
        minimized_cost=minimized_cost,
        steps=steps,
        certificate={
            "accepted": transducer.accepts_pair(minimized.input_symbols, minimized.output_symbols),
            "minimality": "uniform-cost reachability over FST states",
            "states": len(transducer.states),
            "transitions": len(transducer.transitions),
            "approximation": transducer.approximation,
        },
    )


def shrink_finite_contract_counterexample(
    problem: FiniteContractProblem,
    result: SolverResult | None = None,
    *,
    metric: CounterexampleMetric | str = CounterexampleMetric.STRING_LENGTH,
    prefer_z3: bool = False,
) -> CounterexampleShrinkReport:
    """Minimize a finite-contract SAT assignment across the declared finite domains."""

    shrink_metric = CounterexampleMetric(metric)
    solver_result = result or problem.solve(prefer_z3=prefer_z3)
    if solver_result.status is not SolverStatus.SAT or solver_result.assignment is None:
        raise CounterexampleShrinkError("finite contract did not produce a SAT counterexample assignment")
    original_assignment = dict(solver_result.assignment)
    if not _assignment_satisfies(problem, original_assignment):
        raise CounterexampleShrinkError("finite contract counterexample assignment does not satisfy all constraints")

    minimized_assignment, satisfying_count = _minimal_contract_assignment(problem, metric=shrink_metric)
    minimized_cost = _assignment_cost(minimized_assignment, problem.variables, metric=shrink_metric)
    original_cost = _assignment_cost(original_assignment, problem.variables, metric=shrink_metric)
    steps = _steps_if_changed(
        action="replace with globally cheapest satisfying assignment",
        before_cost=original_cost,
        after_cost=minimized_cost,
        detail=f"enumerated {satisfying_count} satisfying assignments",
        changed=_assignment_payload(original_assignment) != _assignment_payload(minimized_assignment),
    )
    return CounterexampleShrinkReport(
        kind="finite-contract",
        metric=shrink_metric,
        original=_assignment_payload(original_assignment),
        minimized=_assignment_payload(minimized_assignment),
        original_cost=original_cost,
        minimized_cost=minimized_cost,
        steps=steps,
        certificate={
            "satisfies_constraints": _assignment_satisfies(problem, minimized_assignment),
            "minimality": "exhaustive finite-domain enumeration",
            "satisfying_assignments": satisfying_count,
            "variables": len(problem.variables),
            "constraints": len(problem.constraints),
        },
    )


def _shortest_dfa_by_metric(
    automaton: DeterministicFiniteAutomaton,
    metric: CounterexampleMetric,
) -> AutomatonWitness | None:
    heap: list[tuple[int, int, str, tuple[str, ...], tuple[str, ...]]] = []
    heappush(heap, (0, 0, automaton.start, (), (automaton.start,)))
    best: dict[str, int] = {automaton.start: 0}
    while heap:
        cost, depth, state, symbols, states = heappop(heap)
        if cost > best.get(state, cost):
            continue
        if state in automaton.accepts:
            return AutomatonWitness(symbols=symbols, states=states)
        for symbol in automaton.alphabet:
            target = automaton.step(state, symbol)
            if target is None:
                continue
            next_symbols = symbols + (symbol,)
            next_cost = _symbol_cost(next_symbols, metric=metric)
            if next_cost < best.get(target, 10**18):
                best[target] = next_cost
                heappush(heap, (next_cost, depth + 1, target, next_symbols, states + (target,)))
    return None


def _shortest_fst_by_metric(
    transducer: FiniteStateTransducer,
    metric: CounterexampleMetric,
) -> TransducerWitness | None:
    by_source = _transitions_by_source(transducer)
    heap: list[
        tuple[
            int,
            int,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[TransducerLabel, ...],
        ]
    ] = []
    heappush(heap, (0, 0, transducer.start, (), (), (transducer.start,), ()))
    best: dict[str, int] = {transducer.start: 0}
    while heap:
        cost, label_count, state, input_symbols, output_symbols, states, labels = heappop(heap)
        if cost > best.get(state, cost):
            continue
        if state in transducer.accepts:
            return TransducerWitness(
                input_symbols=input_symbols,
                output_symbols=output_symbols,
                states=states,
                labels=labels,
            )
        for transition in by_source.get(state, ()):
            next_input = input_symbols + (() if transition.label.input_symbol is None else (transition.label.input_symbol,))
            next_output = output_symbols + (() if transition.label.output_symbol is None else (transition.label.output_symbol,))
            next_labels = labels + (transition.label,)
            next_cost = _transducer_symbols_cost(next_input, next_output, next_labels, metric=metric)
            if next_cost < best.get(transition.target, 10**18):
                best[transition.target] = next_cost
                heappush(
                    heap,
                    (
                        next_cost,
                        label_count + 1,
                        transition.target,
                        next_input,
                        next_output,
                        states + (transition.target,),
                        next_labels,
                    ),
                )
    return None


def _minimal_contract_assignment(
    problem: FiniteContractProblem,
    *,
    metric: CounterexampleMetric,
) -> tuple[dict[str, object], int]:
    best_assignment: dict[str, object] | None = None
    best_key: tuple[int, str] | None = None
    satisfying_count = 0

    def search(index: int, current: dict[str, object]) -> None:
        nonlocal best_assignment, best_key, satisfying_count
        if index == len(problem.variables):
            if not _assignment_satisfies(problem, current):
                return
            satisfying_count += 1
            candidate = dict(current)
            key = (_assignment_cost(candidate, problem.variables, metric=metric), repr(sorted(candidate.items())))
            if best_key is None or key < best_key:
                best_key = key
                best_assignment = candidate
            return
        variable = problem.variables[index]
        for value in variable.values():
            current[variable.name] = value
            search(index + 1, current)
        current.pop(variable.name, None)

    search(0, {})
    if best_assignment is None:
        raise CounterexampleShrinkError("finite contract has no satisfying assignment to minimize")
    return best_assignment, satisfying_count


def _assignment_satisfies(problem: FiniteContractProblem, assignment: Mapping[str, object]) -> bool:
    expected = {variable.name for variable in problem.variables}
    if set(assignment) != expected:
        return False
    return all(bool(constraint.expression.evaluate(assignment)) for constraint in problem.constraints)


def _symbol_cost(symbols: tuple[str, ...], *, metric: CounterexampleMetric) -> int:
    if metric in (CounterexampleMetric.STRING_LENGTH, CounterexampleMetric.SCHEMA_SIZE):
        return sum(len(symbol) for symbol in symbols)
    if metric in (CounterexampleMetric.TOKEN_COUNT, CounterexampleMetric.MESSAGE_COUNT, CounterexampleMetric.RULE_COMPLEXITY):
        return len(symbols)
    raise CounterexampleShrinkError(f"unsupported counterexample metric: {metric}")


def _transducer_cost(witness: TransducerWitness, *, metric: CounterexampleMetric) -> int:
    return _transducer_symbols_cost(witness.input_symbols, witness.output_symbols, witness.labels, metric=metric)


def _transducer_symbols_cost(
    input_symbols: tuple[str, ...],
    output_symbols: tuple[str, ...],
    labels: tuple[TransducerLabel, ...],
    *,
    metric: CounterexampleMetric,
) -> int:
    if metric is CounterexampleMetric.STRING_LENGTH:
        return sum(len(symbol) for symbol in input_symbols + output_symbols)
    if metric is CounterexampleMetric.TOKEN_COUNT:
        return len(input_symbols) + len(output_symbols)
    if metric is CounterexampleMetric.MESSAGE_COUNT:
        return len(input_symbols)
    if metric is CounterexampleMetric.SCHEMA_SIZE:
        return sum(len(symbol) for symbol in output_symbols)
    if metric is CounterexampleMetric.RULE_COMPLEXITY:
        return len(labels)
    raise CounterexampleShrinkError(f"unsupported counterexample metric: {metric}")


def _assignment_cost(
    assignment: Mapping[str, object],
    variables: tuple[VariableDomain, ...],
    *,
    metric: CounterexampleMetric,
) -> int:
    if metric is CounterexampleMetric.RULE_COMPLEXITY:
        return sum(1 for variable in variables if assignment.get(variable.name) not in (None, False, "", 0))
    cost = 0
    for variable in variables:
        value = assignment[variable.name]
        if isinstance(value, str):
            if metric in (CounterexampleMetric.STRING_LENGTH, CounterexampleMetric.SCHEMA_SIZE):
                cost += len(value)
            else:
                cost += len(value.split()) if " " in value else len(value)
        elif isinstance(value, bool):
            cost += int(value)
        elif isinstance(value, int):
            cost += abs(value)
        else:
            cost += len(repr(value))
    return cost


def _automaton_payload(witness: AutomatonWitness) -> Mapping[str, object]:
    return witness.to_dict()


def _transducer_payload(witness: TransducerWitness) -> Mapping[str, object]:
    return witness.to_dict()


def _assignment_payload(assignment: Mapping[str, object]) -> Mapping[str, object]:
    return {"assignment": dict(sorted(assignment.items()))}


def _steps_if_changed(
    *,
    action: str,
    before_cost: int,
    after_cost: int,
    detail: str,
    changed: bool,
) -> tuple[CounterexampleShrinkStep, ...]:
    if not changed:
        return ()
    return (
        CounterexampleShrinkStep(
            action=action,
            before_cost=before_cost,
            after_cost=after_cost,
            detail=detail,
        ),
    )


def _transitions_by_source(transducer: FiniteStateTransducer) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for transition in transducer.transitions:
        grouped.setdefault(transition.source, []).append(transition)
    return {state: tuple(items) for state, items in grouped.items()}
