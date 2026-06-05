"""Provider benchmark drift dashboards (step 294).

Conformance is not a one-time pass/fail: a provider's benchmark pass-rate drifts
over revisions.  This module ingests a time series of conformance runs (per
revision: total vectors, passed, newly-failing vector ids) and computes a drift
dashboard -- pass-rate trend, regressions introduced at each revision, and an
alarm when the pass-rate drops by more than a tolerance versus the best-ever
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

BENCHMARK_DRIFT_VERSION = "promptabi.benchmark-drift.v1"


class DriftFindingKind(StrEnum):
    REGRESSION = "regression"
    PASS_RATE_DROP = "pass-rate-drop"
    RECOVERED = "recovered"


@dataclass(frozen=True, slots=True)
class ConformanceRun:
    revision: str
    total: int
    passed: int
    failing_vector_ids: frozenset[str] = field(default=frozenset())

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class DriftFinding:
    kind: DriftFindingKind
    revision: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "revision": self.revision,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DriftDashboard:
    version: str
    latest_pass_rate: float
    best_pass_rate: float
    alarm: bool
    findings: tuple[DriftFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "latest_pass_rate": self.latest_pass_rate,
            "best_pass_rate": self.best_pass_rate,
            "alarm": self.alarm,
            "findings": [f.to_dict() for f in self.findings],
        }


def build_drift_dashboard(
    runs: tuple[ConformanceRun, ...], *, tolerance: float = 0.02
) -> DriftDashboard:
    if not runs:
        return DriftDashboard(BENCHMARK_DRIFT_VERSION, 0.0, 0.0, False, ())

    findings: list[DriftFinding] = []
    best = 0.0
    prev_failing: frozenset[str] = frozenset()
    for i, run in enumerate(runs):
        new_failures = run.failing_vector_ids - prev_failing
        recovered = prev_failing - run.failing_vector_ids
        if i > 0 and new_failures:
            findings.append(
                DriftFinding(
                    DriftFindingKind.REGRESSION,
                    run.revision,
                    f"newly failing: {sorted(new_failures)}",
                )
            )
        if i > 0 and recovered:
            findings.append(
                DriftFinding(
                    DriftFindingKind.RECOVERED,
                    run.revision,
                    f"recovered: {sorted(recovered)}",
                )
            )
        best = max(best, run.pass_rate)
        prev_failing = run.failing_vector_ids

    latest = runs[-1].pass_rate
    alarm = (best - latest) > tolerance
    if alarm:
        findings.append(
            DriftFinding(
                DriftFindingKind.PASS_RATE_DROP,
                runs[-1].revision,
                f"pass-rate {latest:.3f} is >{tolerance:.3f} below best {best:.3f}",
            )
        )

    return DriftDashboard(
        version=BENCHMARK_DRIFT_VERSION,
        latest_pass_rate=latest,
        best_pass_rate=best,
        alarm=alarm,
        findings=tuple(findings),
    )


def render_drift_dashboard_text(dash: DriftDashboard) -> str:
    lines = [
        f"PromptABI provider benchmark drift ({dash.version})",
        f"latest={dash.latest_pass_rate:.3f} best={dash.best_pass_rate:.3f} "
        f"alarm={'YES' if dash.alarm else 'no'}",
    ]
    for f in dash.findings:
        lines.append(f"  {f.kind.value} @{f.revision}: {f.detail}")
    return "\n".join(lines) + "\n"
