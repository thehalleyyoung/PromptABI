"""Add timeout-sensitive degradation tests (step 235).

A sound solver must *degrade*, never *lie*.  As the time budget shrinks, the
finite/SMT backend may stop proving (returning a ``TIMED_OUT`` or ``BOUNDED``
outcome), but it must never return a **conclusive** ``sat``/``unsat`` verdict that
contradicts the ground-truth verdict obtained with a generous budget.

This module measures that property.  Given a problem and a ladder of timeouts it

* establishes the ground-truth verdict with a generous budget;
* runs the problem under each timeout (using the finite-enumeration backend,
  whose exploration is interruptible);
* records the outcome at each budget; and
* proves **soundness under degradation**: every conclusive verdict matches the
  ground truth, and tighter budgets are never *more* conclusive in a way that
  flips the answer.

The check is deterministic even though wall-clock timing is not: it asserts a
property over the recorded verdicts rather than over exact timing boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .formal import (
    FiniteContractProblem,
    SolverBudgetOutcome,
    SolverStatus,
)

TIMEOUT_DEGRADATION_VERSION = "promptabi.timeout-degradation.v1"


class DegradationFindingKind(StrEnum):
    UNSOUND_UNDER_TIMEOUT = "unsound-under-timeout"
    NO_GROUND_TRUTH = "no-ground-truth"


@dataclass(frozen=True, slots=True)
class TimeoutObservation:
    timeout_seconds: float
    status: str
    budget_outcome: str
    conclusive: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "status": self.status,
            "budget_outcome": self.budget_outcome,
            "conclusive": self.conclusive,
        }


@dataclass(frozen=True, slots=True)
class DegradationFinding:
    kind: DegradationFindingKind
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class DegradationReport:
    version: str
    problem: str
    ground_truth_status: str
    observations: tuple[TimeoutObservation, ...] = field(default=())
    findings: tuple[DegradationFinding, ...] = field(default=())

    @property
    def sound(self) -> bool:
        return not self.findings

    @property
    def degraded(self) -> bool:
        return any(not obs.conclusive for obs in self.observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "problem": self.problem,
            "sound": self.sound,
            "degraded": self.degraded,
            "ground_truth_status": self.ground_truth_status,
            "observations": [obs.to_dict() for obs in self.observations],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def profile_timeout_degradation(
    problem: FiniteContractProblem,
    timeouts: Sequence[float],
) -> DegradationReport:
    """Run a problem under a ladder of timeouts and prove sound degradation."""

    if not timeouts:
        raise ValueError("at least one timeout must be provided")

    ground_truth = problem.solve(prefer_z3=True)
    findings: list[DegradationFinding] = []
    if ground_truth.status is SolverStatus.UNKNOWN:
        findings.append(
            DegradationFinding(
                kind=DegradationFindingKind.NO_GROUND_TRUTH,
                message="ground-truth solve was inconclusive; degradation cannot be certified",
            )
        )

    observations: list[TimeoutObservation] = []
    for timeout in sorted(timeouts):
        if timeout <= 0:
            raise ValueError("timeouts must be positive")
        result = problem.solve(prefer_z3=False, timeout_seconds=timeout)
        conclusive = result.status is not SolverStatus.UNKNOWN
        observations.append(
            TimeoutObservation(
                timeout_seconds=timeout,
                status=result.status.value,
                budget_outcome=result.budget_outcome.value,
                conclusive=conclusive,
            )
        )
        if (
            conclusive
            and ground_truth.status is not SolverStatus.UNKNOWN
            and result.status is not ground_truth.status
        ):
            findings.append(
                DegradationFinding(
                    kind=DegradationFindingKind.UNSOUND_UNDER_TIMEOUT,
                    message=(
                        f"at timeout {timeout}s the solver returned {result.status.value!r} but the "
                        f"ground truth is {ground_truth.status.value!r}"
                    ),
                )
            )

    return DegradationReport(
        version=TIMEOUT_DEGRADATION_VERSION,
        problem=problem.name,
        ground_truth_status=ground_truth.status.value,
        observations=tuple(observations),
        findings=tuple(findings),
    )


def render_degradation_json(report: DegradationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_degradation_text(report: DegradationReport) -> str:
    lines = [
        f"PromptABI timeout degradation ({report.version})",
        f"problem: {report.problem}",
        f"ground truth: {report.ground_truth_status}",
        f"sound: {report.sound}, degraded: {report.degraded}",
    ]
    for obs in report.observations:
        flag = obs.status if obs.conclusive else f"{obs.status}/{obs.budget_outcome}"
        lines.append(f"  {obs.timeout_seconds}s -> {flag}")
    for finding in report.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
