"""Counterexample shrinking and slicing for formal PromptABI witnesses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from heapq import heappop, heappush
from typing import Any

from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace
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
class CounterexampleProductArtifact:
    """One artifact participating in a composed counterexample product."""

    ref: ArtifactRef
    facts: tuple[str, ...]
    cost: int = 1

    def __post_init__(self) -> None:
        facts = tuple(dict.fromkeys(str(fact) for fact in self.facts))
        if not facts:
            raise ValueError("product artifact facts must be non-empty")
        if any(not fact for fact in facts):
            raise ValueError("product artifact facts must be non-empty")
        if self.cost < 1:
            raise ValueError("product artifact cost must be positive")
        object.__setattr__(self, "facts", facts)

    @property
    def key(self) -> str:
        return _artifact_key(self.ref)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact": self.ref.to_dict(),
            "facts": list(self.facts),
            "cost": self.cost,
        }


@dataclass(frozen=True, slots=True)
class CounterexampleProduct:
    """A bounded product of artifacts, facts, and failing obligations."""

    name: str
    artifacts: tuple[CounterexampleProductArtifact, ...]
    failing_facts: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("counterexample product name must be non-empty")
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("counterexample product must include at least one artifact")
        failing_facts = tuple(dict.fromkeys(str(fact) for fact in self.failing_facts))
        if not failing_facts:
            raise ValueError("counterexample product failing facts must be non-empty")
        if any(not fact for fact in failing_facts):
            raise ValueError("counterexample product failing facts must be non-empty")

        artifact_keys = [artifact.key for artifact in artifacts]
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("counterexample product artifact keys must be unique")
        known_keys = set(artifact_keys)
        normalized_edges: list[tuple[str, str]] = []
        for source, target in self.edges:
            source_key = str(source)
            target_key = str(target)
            if source_key not in known_keys or target_key not in known_keys:
                raise ValueError("product edges must reference known artifacts")
            normalized_edges.append((source_key, target_key))

        provided_facts = set().union(*(set(artifact.facts) for artifact in artifacts))
        missing = set(failing_facts) - provided_facts
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"failing facts are not produced by any artifact: {missing_list}")

        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "failing_facts", failing_facts)
        object.__setattr__(self, "edges", tuple(dict.fromkeys(normalized_edges)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "failing_facts": list(self.failing_facts),
            "edges": [{"source": source, "target": target} for source, target in self.edges],
        }


@dataclass(frozen=True, slots=True)
class CounterexampleSliceReport:
    """A minimal artifact slice that still explains a composed counterexample."""

    product_name: str
    required_facts: tuple[str, ...]
    artifacts: tuple[CounterexampleProductArtifact, ...]
    omitted_artifacts: tuple[ArtifactRef, ...]
    cut_edges: tuple[tuple[str, str], ...]
    certificate: Mapping[str, object]

    @property
    def artifact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(artifact.ref for artifact in self.artifacts)

    @property
    def artifact_keys(self) -> tuple[str, ...]:
        return tuple(artifact.key for artifact in self.artifacts)

    def to_dict(self) -> dict[str, object]:
        return {
            "product_name": self.product_name,
            "required_facts": list(self.required_facts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "omitted_artifacts": [artifact.to_dict() for artifact in self.omitted_artifacts],
            "cut_edges": [{"source": source, "target": target} for source, target in self.cut_edges],
            "certificate": dict(self.certificate),
        }

    def witness(self) -> WitnessTrace:
        steps = (
            WitnessStep(
                action="identify failing product facts",
                input=self.product_name,
                output=", ".join(self.required_facts),
            ),
            WitnessStep(
                action="solve minimum artifact cover",
                input=f"{len(self.artifacts) + len(self.omitted_artifacts)} artifacts",
                output=f"{len(self.artifacts)} artifacts",
            ),
            WitnessStep(
                action="certify sliced counterexample",
                input=", ".join(self.artifact_keys),
                output=str(self.certificate.get("minimality", "minimal artifact cover")),
            ),
        )
        return WitnessTrace(
            summary=(
                f"{self.product_name} counterexample slice keeps {len(self.artifacts)} "
                f"of {len(self.artifacts) + len(self.omitted_artifacts)} artifacts"
            ),
            steps=steps,
            artifacts=self.artifact_refs,
        )


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


def slice_counterexample_product(
    product: CounterexampleProduct,
    *,
    required_facts: Iterable[str] | None = None,
) -> CounterexampleSliceReport:
    """Find the cheapest artifact subset that still covers the failing product facts.

    The search is exhaustive over the explicit bounded product, making the result
    a real minimality certificate rather than a greedy explanation. Ties are
    resolved deterministically by fewer artifacts and stable artifact keys.
    """

    facts = tuple(dict.fromkeys(str(fact) for fact in (required_facts or product.failing_facts)))
    if not facts:
        raise CounterexampleShrinkError("counterexample slice requires at least one failing fact")
    available = set().union(*(set(artifact.facts) for artifact in product.artifacts))
    missing = set(facts) - available
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise CounterexampleShrinkError(f"required facts are not produced by any artifact: {missing_list}")

    best_indexes: tuple[int, ...] | None = None
    best_key: tuple[int, int, tuple[str, ...]] | None = None
    candidates_examined = 0
    artifact_count = len(product.artifacts)
    for mask in range(1, 1 << artifact_count):
        indexes = tuple(index for index in range(artifact_count) if mask & (1 << index))
        covered = set().union(*(set(product.artifacts[index].facts) for index in indexes))
        candidates_examined += 1
        if not set(facts).issubset(covered):
            continue
        selected = tuple(product.artifacts[index] for index in indexes)
        key = (
            sum(artifact.cost for artifact in selected),
            len(selected),
            tuple(artifact.key for artifact in selected),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_indexes = indexes

    if best_indexes is None:
        raise CounterexampleShrinkError("counterexample product has no slice covering required facts")

    selected = tuple(product.artifacts[index] for index in best_indexes)
    selected_keys = {artifact.key for artifact in selected}
    omitted = tuple(artifact.ref for artifact in product.artifacts if artifact.key not in selected_keys)
    cut_edges = tuple(
        (source, target)
        for source, target in product.edges
        if (source in selected_keys) != (target in selected_keys)
    )
    covered_facts = sorted(set().union(*(set(artifact.facts) for artifact in selected)))
    return CounterexampleSliceReport(
        product_name=product.name,
        required_facts=facts,
        artifacts=selected,
        omitted_artifacts=omitted,
        cut_edges=cut_edges,
        certificate={
            "covers_required_facts": set(facts).issubset(covered_facts),
            "minimality": "exhaustive minimum-cost artifact cover",
            "candidate_slices_examined": candidates_examined,
            "selected_cost": sum(artifact.cost for artifact in selected),
            "covered_facts": covered_facts,
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


def _artifact_key(ref: ArtifactRef) -> str:
    location = ref.location_uri or ref.name
    version = ref.revision or ref.version or ""
    return f"{ref.kind}:{location}:{version}"
