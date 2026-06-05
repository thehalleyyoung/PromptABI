"""Mechanized proof experiments for PromptABI's smallest finite fragments.

These experiments are intentionally tiny and executable. They are not a full
proof-assistant development; they are repository-local mechanizations that
construct canonical automata and finite-contract obligations, run the production
algorithms, and then check the result with independent executable specs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .formal import (
    BoolDomain,
    DeterministicFiniteAutomaton,
    Eq,
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    SolverStatus,
    Value,
    Var,
)
from .specs import ExecutableSpecReport, SpecCheck, check_contract_result, check_dfa_product_language, check_dfa_witness


MECHANIZED_PROOF_EXPERIMENT_VERSION = "2026.06"


@dataclass(frozen=True, slots=True)
class MechanizedProofExperiment:
    """One executable proof experiment over a small finite fragment."""

    experiment_id: str
    title: str
    fragment: str
    theorem: str
    assumptions: tuple[str, ...]
    reports: tuple[ExecutableSpecReport, ...]
    artifacts: tuple[tuple[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    @property
    def checks(self) -> tuple[SpecCheck, ...]:
        return tuple(check for report in self.reports for check in report.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "fragment": self.fragment,
            "theorem": self.theorem,
            "assumptions": list(self.assumptions),
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "artifacts": {name: value for name, value in self.artifacts},
        }


@dataclass(frozen=True, slots=True)
class MechanizedProofExperimentReport:
    """A deterministic suite of mechanized proof experiments."""

    experiments: tuple[MechanizedProofExperiment, ...]

    @property
    def passed(self) -> bool:
        return all(experiment.passed for experiment in self.experiments)

    @property
    def experiment_count(self) -> int:
        return len(self.experiments)

    @property
    def check_count(self) -> int:
        return sum(len(experiment.checks) for experiment in self.experiments)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": MECHANIZED_PROOF_EXPERIMENT_VERSION,
            "passed": self.passed,
            "experiment_count": self.experiment_count,
            "check_count": self.check_count,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }


def run_mechanized_proof_experiments() -> MechanizedProofExperimentReport:
    """Run all built-in mechanized experiments for the finite core."""

    return MechanizedProofExperimentReport(
        experiments=(
            _dfa_de_morgan_experiment(),
            _dfa_minimization_experiment(),
            _finite_contract_sat_experiment(),
            _finite_contract_unsat_core_experiment(),
        )
    )


def render_mechanized_proof_experiments_json(report: MechanizedProofExperimentReport) -> str:
    """Render mechanized proof experiments as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_mechanized_proof_experiments_text(report: MechanizedProofExperimentReport) -> str:
    """Render mechanized proof experiments for CLI logs and artifact review."""

    lines = [
        f"PromptABI mechanized proof experiments ({MECHANIZED_PROOF_EXPERIMENT_VERSION})",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        f"experiments: {report.experiment_count}",
        f"executable checks: {report.check_count}",
    ]
    for experiment in report.experiments:
        lines.append("")
        lines.append(f"{experiment.experiment_id}: {'PASS' if experiment.passed else 'FAIL'}")
        lines.append(f"  title: {experiment.title}")
        lines.append(f"  fragment: {experiment.fragment}")
        lines.append(f"  theorem: {experiment.theorem}")
        if experiment.assumptions:
            lines.append("  assumptions:")
            lines.extend(f"    - {assumption}" for assumption in experiment.assumptions)
        failures = [check for check in experiment.checks if not check.passed]
        lines.append(f"  checks: {len(experiment.checks)} ({'PASS' if not failures else 'FAIL'})")
        for failure in failures:
            lines.append(f"    - {failure.name}: {failure.detail}")
    return "\n".join(lines) + "\n"


def _dfa_de_morgan_experiment() -> MechanizedProofExperiment:
    alphabet = {"a", "b"}
    contains_a = DeterministicFiniteAutomaton.finite_language(("a", "ab", "ba"), alphabet=alphabet, name="contains-a-bounded")
    contains_b = DeterministicFiniteAutomaton.finite_language(("b", "ab", "ba"), alphabet=alphabet, name="contains-b-bounded")
    left = contains_a.intersect(contains_b, name="a-and-b").complement().minimize(name="not-a-and-b")
    right = contains_a.complement().union(contains_b.complement(), name="not-a-or-not-b").minimize(name="de-morgan-right")
    equivalence = _bounded_equivalence(left, right, alphabet=tuple(sorted(alphabet)), max_depth=2)
    return MechanizedProofExperiment(
        experiment_id="dfa-de-morgan-bounded",
        title="DFA De Morgan equivalence",
        fragment="complete minimized DFAs over words of length <= 2",
        theorem="For the bounded control-token alphabet, complement(intersection(A, B)) and union(complement(A), complement(B)) accept the same words.",
        assumptions=("finite alphabet is explicit", "bounded enumeration depth covers the constructed finite language"),
        reports=(
            check_dfa_product_language(contains_a.intersect(contains_b), contains_a, contains_b, operation="intersection", max_depth=2),
            ExecutableSpecReport((equivalence,)),
        ),
        artifacts=(
            ("alphabet", "".join(sorted(alphabet))),
            ("max_depth", 2),
            ("left_states", len(left.states)),
            ("right_states", len(right.states)),
        ),
    )


def _dfa_minimization_experiment() -> MechanizedProofExperiment:
    alphabet = {"x", "y"}
    raw = DeterministicFiniteAutomaton(
        states=frozenset({"q0", "q1", "dead1", "dead2"}),
        alphabet=tuple(alphabet),
        start="q0",
        accepts=frozenset({"q1"}),
        transitions={
            ("q0", "x"): "q1",
            ("q0", "y"): "dead1",
            ("q1", "x"): "q1",
            ("q1", "y"): "dead2",
            ("dead1", "x"): "dead1",
            ("dead1", "y"): "dead1",
            ("dead2", "x"): "dead2",
            ("dead2", "y"): "dead2",
        },
        name="redundant-dead-states",
    )
    minimized = raw.minimize(name="redundant-dead-states-min")
    witness_report = check_dfa_witness(raw, raw.shortest_witness())
    equivalence = _bounded_equivalence(raw, minimized, alphabet=tuple(sorted(alphabet)), max_depth=3)
    merged_dead_states = SpecCheck(
        "minimization-merges-equivalent-dead-states",
        len(minimized.states) < len(raw.states),
        f"raw={len(raw.states)} minimized={len(minimized.states)}",
    )
    return MechanizedProofExperiment(
        experiment_id="dfa-minimization-language-preservation",
        title="DFA minimization preserves bounded language",
        fragment="reachable partition refinement on a total finite automaton",
        theorem="Minimization merges equivalent states without changing accepted words in the checked bounded language.",
        assumptions=("automaton is total over the declared alphabet", "bounded reference enumeration is independent of partition refinement"),
        reports=(witness_report, ExecutableSpecReport((equivalence, merged_dead_states))),
        artifacts=(("raw_states", len(raw.states)), ("minimized_states", len(minimized.states)), ("max_depth", 3)),
    )


def _finite_contract_sat_experiment() -> MechanizedProofExperiment:
    problem = FiniteContractProblem(
        name="small-sat-proof-experiment",
        variables=(BoolDomain("enabled"), IntRangeDomain("budget", 0, 4)),
        constraints=(
            NamedConstraint("enabled", Eq(Var("enabled"), Value(True))),
            NamedConstraint("budget-at-least-two", Le(Value(2), Var("budget"))),
            NamedConstraint("budget-at-most-three", Le(Var("budget"), Value(3))),
        ),
    )
    result = problem.solve(prefer_z3=False)
    status_check = SpecCheck("sat-status", result.status is SolverStatus.SAT, result.status.value)
    return MechanizedProofExperiment(
        experiment_id="finite-contract-sat-assignment",
        title="Finite-contract SAT assignment",
        fragment="Boolean and integer-range finite-domain constraints",
        theorem="A satisfying assignment returned by the finite solver lies in each domain and satisfies every named constraint.",
        assumptions=("finite enumeration backend is used", "all variables have finite explicit domains"),
        reports=(ExecutableSpecReport((status_check,)), check_contract_result(problem, result)),
        artifacts=(("solver_backend", result.backend.value), ("assignment", result.assignment or {})),
    )


def _finite_contract_unsat_core_experiment() -> MechanizedProofExperiment:
    problem = FiniteContractProblem(
        name="small-unsat-core-proof-experiment",
        variables=(BoolDomain("safe"),),
        constraints=(
            NamedConstraint("safe-required", Eq(Var("safe"), Value(True))),
            NamedConstraint("unsafe-required", Eq(Var("safe"), Value(False))),
        ),
    )
    result = problem.solve(prefer_z3=False)
    status_check = SpecCheck("unsat-status", result.status is SolverStatus.UNSAT, result.status.value)
    return MechanizedProofExperiment(
        experiment_id="finite-contract-unsat-core",
        title="Finite-contract UNSAT core",
        fragment="Boolean finite-domain core extraction",
        theorem="An UNSAT finite-contract result carries known constraint names whose deletion-minimal subset remains unsatisfiable.",
        assumptions=("finite enumeration backend is used", "core minimality is checked by independent finite enumeration"),
        reports=(ExecutableSpecReport((status_check,)), check_contract_result(problem, result)),
        artifacts=(("solver_backend", result.backend.value), ("unsat_core", list(result.unsat_core))),
    )


def _bounded_equivalence(
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    *,
    alphabet: tuple[str, ...],
    max_depth: int,
) -> SpecCheck:
    mismatches: list[str] = []
    for word in _bounded_words(alphabet, max_depth):
        left_accepts = left.accepts_symbols(word)
        right_accepts = right.accepts_symbols(word)
        if left_accepts != right_accepts:
            mismatches.append(f"{''.join(word)!r}: left={left_accepts} right={right_accepts}")
            if len(mismatches) >= 5:
                break
    return SpecCheck(
        "bounded-language-equivalence",
        not mismatches,
        f"checked_depth={max_depth} alphabet={''.join(alphabet)!r} mismatches={mismatches}",
    )


def _bounded_words(alphabet: tuple[str, ...], max_depth: int) -> tuple[tuple[str, ...], ...]:
    words: list[tuple[str, ...]] = [()]
    current: list[tuple[str, ...]] = [()]
    for _ in range(max_depth):
        current = [word + (symbol,) for word in current for symbol in alphabet]
        words.extend(current)
    return tuple(words)
