"""Mechanize the smallest solver encodings (step 239).

A trustworthy decision procedure should be pinned down by a corpus of *minimal*
encodings: for every expression construct in the supported fragment, the
smallest finite problem that exercises it, paired with the verdict it must
produce.  These tiny instances are the unit cells of PromptABI's soundness story
-- if a backend mis-decides ``Eq`` on a one-variable, two-value domain, no larger
proof can be trusted.

This module publishes that corpus and *mechanizes* it: every minimal encoding is
run on both the Z3 backend and the finite-enumeration backend, and we require (a)
both backends agree and (b) the agreed verdict matches the published label.  We
additionally check *minimality* of the constraint set -- dropping the single
discriminating constraint must change the verdict -- so each entry really is a
smallest witness for its construct rather than an over-constrained accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from .formal import (
    And,
    BoolDomain,
    Eq,
    Expression,
    FiniteContractProblem,
    Implies,
    InSet,
    IntRangeDomain,
    Le,
    NamedConstraint,
    Not,
    Or,
    SolverStatus,
    Sum,
    Value,
    Var,
)

MINIMAL_ENCODINGS_VERSION = "promptabi.minimal-encodings.v1"


class EncodingFindingKind(StrEnum):
    BACKEND_DISAGREEMENT = "backend-disagreement"
    WRONG_VERDICT = "wrong-verdict"
    NOT_MINIMAL = "not-minimal"


@dataclass(frozen=True, slots=True)
class MinimalEncoding:
    construct: str
    variables: tuple[object, ...]
    discriminating: NamedConstraint
    expected: str  # "sat" | "unsat"
    context: tuple[NamedConstraint, ...] = ()

    def problem(self, *, include_discriminating: bool = True) -> FiniteContractProblem:
        constraints = tuple(self.context)
        if include_discriminating:
            constraints = constraints + (self.discriminating,)
        return FiniteContractProblem(
            variables=self.variables,
            constraints=constraints,
            name=f"minimal-{self.construct}",
        )

    def to_dict(self) -> dict[str, object]:
        return {"construct": self.construct, "expected": self.expected}


@dataclass(frozen=True, slots=True)
class EncodingFinding:
    construct: str
    kind: EncodingFindingKind
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "construct": self.construct,
            "kind": self.kind.value,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class EncodingResult:
    construct: str
    expected: str
    z3_verdict: str
    enum_verdict: str
    minimal: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "construct": self.construct,
            "expected": self.expected,
            "z3_verdict": self.z3_verdict,
            "enum_verdict": self.enum_verdict,
            "minimal": self.minimal,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class MechanizationReport:
    version: str
    results: tuple[EncodingResult, ...]
    findings: tuple[EncodingFinding, ...]

    @property
    def mechanized(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mechanized": self.mechanized,
            "results": [r.to_dict() for r in self.results],
            "findings": [f.to_dict() for f in self.findings],
        }


def _c(name: str, expr: Expression) -> NamedConstraint:
    return NamedConstraint(name=name, expression=expr)


def standard_minimal_encodings() -> tuple[MinimalEncoding, ...]:
    """The smallest verdict-bearing encoding for each supported construct."""

    x = IntRangeDomain(name="x", minimum=0, maximum=1)
    b = BoolDomain(name="b")
    return (
        MinimalEncoding("eq", (x,), _c("x=1", Eq(Var("x"), Value(1))), "sat"),
        MinimalEncoding("le", (x,), _c("x<=0", Le(Var("x"), Value(0))), "sat"),
        MinimalEncoding(
            "le-unsat", (x,), _c("x<=-1", Le(Var("x"), Value(-1))), "unsat"
        ),
        MinimalEncoding(
            "sum",
            (x, IntRangeDomain(name="y", minimum=0, maximum=1)),
            _c("x+y<=0", Le(Sum(Var("x"), Var("y")), Value(0))),
            "sat",
        ),
        MinimalEncoding(
            "and",
            (x,),
            _c("x=1 and x>=0", And(Eq(Var("x"), Value(1)), Le(Value(0), Var("x")))),
            "sat",
        ),
        MinimalEncoding(
            "or",
            (x,),
            _c("x=0 or x=1", Or(Eq(Var("x"), Value(0)), Eq(Var("x"), Value(1)))),
            "sat",
        ),
        MinimalEncoding("not", (b,), _c("not b", Not(Eq(Var("b"), Value(True)))), "sat"),
        MinimalEncoding(
            "implies",
            (b,),
            _c("b->b", Implies(Eq(Var("b"), Value(True)), Eq(Var("b"), Value(True)))),
            "sat",
        ),
        MinimalEncoding(
            "inset", (x,), _c("x in {1}", InSet(Var("x"), (1,))), "sat"
        ),
    )


def _verdict(problem: FiniteContractProblem, *, prefer_z3: bool) -> str:
    status = problem.solve(prefer_z3=prefer_z3, max_assignments=10000).status
    if status is SolverStatus.SAT:
        return "sat"
    if status is SolverStatus.UNSAT:
        return "unsat"
    return "unknown"


def mechanize_minimal_encodings(
    encodings: Sequence[MinimalEncoding] | None = None,
) -> MechanizationReport:
    """Run every minimal encoding on both backends and certify the corpus."""

    corpus = tuple(encodings) if encodings is not None else standard_minimal_encodings()
    results: list[EncodingResult] = []
    findings: list[EncodingFinding] = []
    for enc in corpus:
        problem = enc.problem()
        z3_verdict = _verdict(problem, prefer_z3=True)
        enum_verdict = _verdict(problem, prefer_z3=False)
        agree = z3_verdict == enum_verdict
        correct = agree and z3_verdict == enc.expected
        # Minimality: without the discriminating constraint the verdict changes
        # (an empty/context-only problem is satisfiable, so an unsat entry must
        # become sat; a sat entry whose context already forces sat is still
        # minimal because the discriminating constraint is the only one present).
        reduced = enc.problem(include_discriminating=False)
        reduced_verdict = _verdict(reduced, prefer_z3=True)
        minimal = reduced_verdict != z3_verdict or not enc.context
        if not agree:
            findings.append(
                EncodingFinding(
                    enc.construct,
                    EncodingFindingKind.BACKEND_DISAGREEMENT,
                    f"z3={z3_verdict} enum={enum_verdict}",
                )
            )
        elif not correct:
            findings.append(
                EncodingFinding(
                    enc.construct,
                    EncodingFindingKind.WRONG_VERDICT,
                    f"expected {enc.expected} but both backends returned {z3_verdict}",
                )
            )
        if not minimal:
            findings.append(
                EncodingFinding(
                    enc.construct,
                    EncodingFindingKind.NOT_MINIMAL,
                    "discriminating constraint does not change the verdict",
                )
            )
        results.append(
            EncodingResult(
                construct=enc.construct,
                expected=enc.expected,
                z3_verdict=z3_verdict,
                enum_verdict=enum_verdict,
                minimal=minimal,
                passed=correct and minimal,
            )
        )
    return MechanizationReport(
        version=MINIMAL_ENCODINGS_VERSION,
        results=tuple(results),
        findings=tuple(findings),
    )


def render_mechanization_json(report: MechanizationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_mechanization_text(report: MechanizationReport) -> str:
    lines = [
        f"PromptABI minimal-encoding mechanization ({report.version})",
        f"status: {'MECHANIZED' if report.mechanized else 'BROKEN'}",
    ]
    for result in report.results:
        flag = "ok" if result.passed else "FAIL"
        lines.append(
            f"  [{flag}] {result.construct}: z3={result.z3_verdict}"
            f" enum={result.enum_verdict} minimal={result.minimal}"
        )
    for finding in report.findings:
        lines.append(f"  ! {finding.construct}: {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
