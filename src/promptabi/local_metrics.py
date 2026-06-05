"""Local-only verification metrics exports without prompt or artifact contents."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .diagnostics import CheckMode, Diagnostic
from .session import VerificationResult

METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LocalMetricsReport:
    """Aggregate PromptABI metrics safe for local dashboards and CI archives."""

    generated_at: str
    config_count: int
    ok_count: int
    total_checks_configured: int
    total_artifacts_configured: int
    total_diagnostics: int
    total_runtime_ms: int
    by_severity: dict[str, int] = field(default_factory=dict)
    by_rule_id: dict[str, int] = field(default_factory=dict)
    by_check_mode: dict[str, int] = field(default_factory=dict)
    by_artifact_kind: dict[str, int] = field(default_factory=dict)
    by_diagnostic_artifact_kind: dict[str, int] = field(default_factory=dict)
    by_solver_outcome: dict[str, int] = field(default_factory=dict)
    check_runtimes_ms: dict[str, int] = field(default_factory=dict)
    diagnostics_by_check: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.ok_count == self.config_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "totals": {
                "configs": self.config_count,
                "ok_configs": self.ok_count,
                "checks_configured": self.total_checks_configured,
                "artifacts_configured": self.total_artifacts_configured,
                "diagnostics": self.total_diagnostics,
                "runtime_ms": self.total_runtime_ms,
            },
            "counts": {
                "by_severity": dict(sorted(self.by_severity.items())),
                "by_rule_id": dict(sorted(self.by_rule_id.items())),
                "by_check_mode": dict(sorted(self.by_check_mode.items())),
                "by_artifact_kind": dict(sorted(self.by_artifact_kind.items())),
                "by_diagnostic_artifact_kind": dict(sorted(self.by_diagnostic_artifact_kind.items())),
                "by_solver_outcome": dict(sorted(self.by_solver_outcome.items())),
            },
            "runtimes_ms": dict(sorted(self.check_runtimes_ms.items())),
            "diagnostics_by_check": dict(sorted(self.diagnostics_by_check.items())),
            "privacy": privacy_guarantees(),
        }


def build_local_metrics_report(
    results: tuple[VerificationResult, ...],
    *,
    generated_at: str | None = None,
) -> LocalMetricsReport:
    """Build a sanitized metrics export from real verification results."""

    by_severity: Counter[str] = Counter()
    by_rule_id: Counter[str] = Counter()
    by_check_mode: Counter[str] = Counter()
    by_artifact_kind: Counter[str] = Counter()
    by_diagnostic_artifact_kind: Counter[str] = Counter()
    by_solver_outcome: Counter[str] = Counter()
    runtimes: Counter[str] = Counter()
    diagnostics_by_check: Counter[str] = Counter()

    for result in results:
        by_artifact_kind.update(artifact.kind.value for artifact in result.config.artifact_bundle.artifacts)
        for runtime in result.check_runtimes:
            runtimes[runtime.check] += runtime.duration_ms
            diagnostics_by_check[runtime.check] += runtime.diagnostics
        for diagnostic in result.diagnostics:
            by_severity[diagnostic.severity.value] += 1
            by_rule_id[diagnostic.rule_id] += 1
            by_check_mode.update(mode.value for mode in diagnostic.check_modes)
            if diagnostic.artifact is not None:
                by_diagnostic_artifact_kind[diagnostic.artifact.kind] += 1
            solver_outcome = _solver_outcome(diagnostic)
            if solver_outcome is not None:
                by_solver_outcome[solver_outcome] += 1

    return LocalMetricsReport(
        generated_at=generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        config_count=len(results),
        ok_count=sum(1 for result in results if result.ok),
        total_checks_configured=sum(len(result.config.checks) for result in results),
        total_artifacts_configured=sum(len(result.config.artifact_bundle.artifacts) for result in results),
        total_diagnostics=sum(len(result.diagnostics) for result in results),
        total_runtime_ms=sum(runtimes.values()),
        by_severity=dict(by_severity),
        by_rule_id=dict(by_rule_id),
        by_check_mode=dict(by_check_mode),
        by_artifact_kind=dict(by_artifact_kind),
        by_diagnostic_artifact_kind=dict(by_diagnostic_artifact_kind),
        by_solver_outcome=dict(by_solver_outcome),
        check_runtimes_ms=dict(runtimes),
        diagnostics_by_check=dict(diagnostics_by_check),
    )


def render_local_metrics_json(report: LocalMetricsReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_local_metrics_text(report: LocalMetricsReport) -> str:
    lines = [
        "PromptABI local metrics export",
        f"generated: {report.generated_at}",
        f"status: {'PASS' if report.ok else 'ATTENTION'}",
        (
            "totals: "
            f"configs {report.config_count}, "
            f"checks {report.total_checks_configured}, "
            f"artifacts {report.total_artifacts_configured}, "
            f"diagnostics {report.total_diagnostics}, "
            f"runtime {report.total_runtime_ms} ms"
        ),
    ]
    _extend_counter_lines(lines, "severity", report.by_severity)
    _extend_counter_lines(lines, "rules", report.by_rule_id)
    _extend_counter_lines(lines, "check modes", report.by_check_mode)
    _extend_counter_lines(lines, "artifact kinds", report.by_artifact_kind)
    _extend_counter_lines(lines, "solver outcomes", report.by_solver_outcome)
    _extend_counter_lines(lines, "check runtimes ms", report.check_runtimes_ms)
    lines.append(
        "privacy: local export only; no prompts, schemas, configs, artifact names, paths, "
        "spans, witnesses, or raw properties"
    )
    return "\n".join(lines) + "\n"


def privacy_guarantees() -> tuple[str, ...]:
    return (
        "Metrics are generated locally from in-process verification results.",
        "The export contains aggregate counts, rule IDs, check names, check modes, solver outcome buckets, "
        "artifact categories, and runtimes.",
        "Config names, artifact names, artifact paths, spans, messages, suggestions, witnesses, rendered strings, "
        "token IDs, and raw diagnostic properties are omitted.",
        "No network sends are performed by the metrics export.",
    )


def _solver_outcome(diagnostic: Diagnostic) -> str | None:
    properties = dict(diagnostic.properties)
    status = properties.get("solver_status") or properties.get("status")
    if isinstance(status, str) and status:
        return status
    if CheckMode.Z3_BACKED_SMT not in diagnostic.check_modes and not diagnostic.rule_id.startswith("static-contract"):
        return None
    if "proved" in diagnostic.rule_id:
        return "unsat"
    if "violation" in diagnostic.rule_id:
        return "sat"
    if "abstain" in diagnostic.rule_id or "unknown" in diagnostic.rule_id:
        return "unknown"
    return "not-recorded"


def _extend_counter_lines(lines: list[str], label: str, values: dict[str, int]) -> None:
    if not values:
        return
    rendered = ", ".join(f"{key}={count}" for key, count in sorted(values.items()))
    lines.append(f"{label}: {rendered}")
