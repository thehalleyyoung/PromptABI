"""Publish solver-fragment conformance suites (step 236).

The value of a sound solver is only as credible as the *definition* of the
fragment it claims to decide.  This module publishes that definition as an
executable **conformance suite**: a labeled corpus of finite-contract problems,
each annotated with the verdict and the in-/out-of-fragment classification a
conforming backend must produce.

:func:`run_fragment_conformance_suite` runs every case, compares the solver's
verdict and the static fragment classification against the published labels, and
reports any divergence.  Backends (a new Z3 release, an alternative enumeration
engine, a downstream re-implementation) can be certified by passing the suite,
and the suite doubles as regression protection for PromptABI itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .abstention_certificate import classify_fragment
from .formal import (
    Assignment,
    Eq,
    FiniteContractProblem,
    IntRangeDomain,
    Le,
    NamedConstraint,
    SolverStatus,
    Value,
    Var,
    _Z3Context,
)

SOLVER_FRAGMENT_CONFORMANCE_VERSION = "promptabi.solver-fragment-conformance.v1"


class ExpectedVerdict(StrEnum):
    SAT = "sat"
    UNSAT = "unsat"
    ABSTAIN = "abstain"


class ConformanceFindingKind(StrEnum):
    VERDICT_MISMATCH = "verdict-mismatch"
    FRAGMENT_MISMATCH = "fragment-mismatch"


@dataclass(frozen=True)
class _UnsupportedNode:
    """An expression node intentionally outside the supported fragment."""

    label: str

    def evaluate(self, assignment: Assignment) -> bool:
        raise TypeError(f"unsupported construct {self.label!r} cannot be evaluated")

    def to_z3(self, context: "_Z3Context") -> object:
        raise TypeError(f"unsupported construct {self.label!r} cannot be encoded")

    def to_dict(self) -> dict[str, object]:
        return {"unsupported": self.label}


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    name: str
    problem: FiniteContractProblem
    expected_verdict: ExpectedVerdict
    expected_in_fragment: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "problem": self.problem.name,
            "expected_verdict": self.expected_verdict.value,
            "expected_in_fragment": self.expected_in_fragment,
        }


@dataclass(frozen=True, slots=True)
class ConformanceFinding:
    case: str
    kind: ConformanceFindingKind
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"case": self.case, "kind": self.kind.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class ConformanceCaseResult:
    name: str
    expected_verdict: str
    actual_verdict: str
    expected_in_fragment: bool
    actual_in_fragment: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_verdict": self.expected_verdict,
            "actual_verdict": self.actual_verdict,
            "expected_in_fragment": self.expected_in_fragment,
            "actual_in_fragment": self.actual_in_fragment,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    version: str
    results: tuple[ConformanceCaseResult, ...] = field(default=())
    findings: tuple[ConformanceFinding, ...] = field(default=())

    @property
    def conformant(self) -> bool:
        return not self.findings

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for result in self.results if result.passed) / len(self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "conformant": self.conformant,
            "pass_rate": round(self.pass_rate, 4),
            "results": [result.to_dict() for result in self.results],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _actual_verdict(problem: FiniteContractProblem, *, prefer_z3: bool) -> ExpectedVerdict:
    result = problem.solve(prefer_z3=prefer_z3)
    if result.status is SolverStatus.SAT:
        return ExpectedVerdict.SAT
    if result.status is SolverStatus.UNSAT:
        return ExpectedVerdict.UNSAT
    return ExpectedVerdict.ABSTAIN


def standard_fragment_conformance_suite() -> tuple[ConformanceCase, ...]:
    """A representative published suite covering sat/unsat/abstain in/out of fragment."""

    sat = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(NamedConstraint(name="x-small", expression=Le(Var("x"), Value(2))),),
        name="sat-int-range",
    )
    unsat = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(
            NamedConstraint(name="x-ge-4", expression=Le(Value(4), Var("x"))),
            NamedConstraint(name="x-le-1", expression=Le(Var("x"), Value(1))),
        ),
        name="unsat-int-range",
    )
    bool_sat = FiniteContractProblem(
        variables=(IntRangeDomain(name="n", minimum=0, maximum=1),),
        constraints=(NamedConstraint(name="n-eq-1", expression=Eq(Var("n"), Value(1))),),
        name="sat-eq",
    )
    unsupported = FiniteContractProblem(
        variables=(IntRangeDomain(name="x", minimum=0, maximum=5),),
        constraints=(
            NamedConstraint(name="weird", expression=_UnsupportedNode(label="opaque")),  # type: ignore[arg-type]
        ),
        name="abstain-unsupported",
    )
    return (
        ConformanceCase("sat-int-range", sat, ExpectedVerdict.SAT, True),
        ConformanceCase("unsat-int-range", unsat, ExpectedVerdict.UNSAT, True),
        ConformanceCase("sat-eq", bool_sat, ExpectedVerdict.SAT, True),
        ConformanceCase("abstain-unsupported", unsupported, ExpectedVerdict.ABSTAIN, False),
    )


def run_fragment_conformance_suite(
    cases: Sequence[ConformanceCase] | None = None,
    *,
    prefer_z3: bool = True,
) -> ConformanceReport:
    """Run a fragment conformance suite and report any divergence from labels."""

    suite = tuple(cases) if cases is not None else standard_fragment_conformance_suite()
    results: list[ConformanceCaseResult] = []
    findings: list[ConformanceFinding] = []
    for case in suite:
        actual_verdict = _actual_verdict(case.problem, prefer_z3=prefer_z3)
        actual_in_fragment = classify_fragment(case.problem).supported
        verdict_ok = actual_verdict is case.expected_verdict
        fragment_ok = actual_in_fragment == case.expected_in_fragment
        if not verdict_ok:
            findings.append(
                ConformanceFinding(
                    case=case.name,
                    kind=ConformanceFindingKind.VERDICT_MISMATCH,
                    message=(
                        f"expected {case.expected_verdict.value!r} but solver returned "
                        f"{actual_verdict.value!r}"
                    ),
                )
            )
        if not fragment_ok:
            findings.append(
                ConformanceFinding(
                    case=case.name,
                    kind=ConformanceFindingKind.FRAGMENT_MISMATCH,
                    message=(
                        f"expected in_fragment={case.expected_in_fragment} but classifier said "
                        f"{actual_in_fragment}"
                    ),
                )
            )
        results.append(
            ConformanceCaseResult(
                name=case.name,
                expected_verdict=case.expected_verdict.value,
                actual_verdict=actual_verdict.value,
                expected_in_fragment=case.expected_in_fragment,
                actual_in_fragment=actual_in_fragment,
                passed=verdict_ok and fragment_ok,
            )
        )
    return ConformanceReport(
        version=SOLVER_FRAGMENT_CONFORMANCE_VERSION,
        results=tuple(results),
        findings=tuple(findings),
    )


def render_conformance_json(report: ConformanceReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_conformance_text(report: ConformanceReport) -> str:
    lines = [
        f"PromptABI solver-fragment conformance ({report.version})",
        f"status: {'CONFORMANT' if report.conformant else 'NON-CONFORMANT'}",
        f"pass rate: {report.pass_rate:.2%}",
    ]
    for result in report.results:
        flag = "pass" if result.passed else "FAIL"
        lines.append(
            f"  [{flag}] {result.name}: {result.actual_verdict}"
            f" (in_fragment={result.actual_in_fragment})"
        )
    for finding in report.findings:
        lines.append(f"  ! {finding.case}: {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
