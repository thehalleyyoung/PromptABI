"""Team dashboard summaries for structural prompt-interface risk."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corpus_verification import CorpusVerificationReport
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity
from .session import VerificationResult

DASHBOARD_SCHEMA_VERSION = 1


class TeamDashboardError(ValueError):
    """Raised when dashboard history or saved reports are malformed."""


@dataclass(frozen=True, slots=True)
class DashboardSourceSummary:
    """Risk summary for one verified PromptABI config."""

    name: str
    diagnostics: int
    open_risks: int
    accepted_suppressions: int
    solver_abstentions: int
    drift_warnings: int
    by_severity: dict[str, int] = field(default_factory=dict)
    by_rule: dict[str, int] = field(default_factory=dict)
    by_artifact_kind: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "diagnostics": self.diagnostics,
            "open_risks": self.open_risks,
            "accepted_suppressions": self.accepted_suppressions,
            "solver_abstentions": self.solver_abstentions,
            "drift_warnings": self.drift_warnings,
            "by_severity": dict(sorted(self.by_severity.items())),
            "by_rule": dict(sorted(self.by_rule.items())),
            "by_artifact_kind": dict(sorted(self.by_artifact_kind.items())),
        }


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """One timestamped dashboard point suitable for local JSONL history."""

    timestamp: str
    sources: tuple[DashboardSourceSummary, ...]
    corpus_regressions: int = 0
    corpus_checks: int = 0

    @property
    def open_risks(self) -> int:
        return sum(source.open_risks for source in self.sources)

    @property
    def accepted_suppressions(self) -> int:
        return sum(source.accepted_suppressions for source in self.sources)

    @property
    def solver_abstentions(self) -> int:
        return sum(source.solver_abstentions for source in self.sources)

    @property
    def drift_warnings(self) -> int:
        return sum(source.drift_warnings for source in self.sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "totals": {
                "open_risks": self.open_risks,
                "accepted_suppressions": self.accepted_suppressions,
                "solver_abstentions": self.solver_abstentions,
                "drift_warnings": self.drift_warnings,
                "corpus_regressions": self.corpus_regressions,
                "corpus_checks": self.corpus_checks,
            },
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardSnapshot":
        if data.get("schema_version") != DASHBOARD_SCHEMA_VERSION:
            raise TeamDashboardError(f"unsupported dashboard schema_version {data.get('schema_version')!r}")
        totals = data.get("totals")
        if not isinstance(totals, dict):
            raise TeamDashboardError("dashboard snapshot missing totals object")
        sources = data.get("sources", ())
        if not isinstance(sources, list):
            raise TeamDashboardError("dashboard snapshot sources must be a list")
        return cls(
            timestamp=str(data["timestamp"]),
            sources=tuple(_source_from_mapping(source) for source in sources),
            corpus_regressions=int(totals.get("corpus_regressions", 0)),
            corpus_checks=int(totals.get("corpus_checks", 0)),
        )


@dataclass(frozen=True, slots=True)
class TeamDashboardReport:
    """Current dashboard plus optional historical trend context."""

    current: DashboardSnapshot
    history: tuple[DashboardSnapshot, ...] = ()

    @property
    def previous(self) -> DashboardSnapshot | None:
        return self.history[-1] if self.history else None

    @property
    def trend(self) -> dict[str, int]:
        previous = self.previous
        if previous is None:
            return {
                "open_risks": 0,
                "accepted_suppressions": 0,
                "solver_abstentions": 0,
                "drift_warnings": 0,
                "corpus_regressions": 0,
            }
        return {
            "open_risks": self.current.open_risks - previous.open_risks,
            "accepted_suppressions": self.current.accepted_suppressions - previous.accepted_suppressions,
            "solver_abstentions": self.current.solver_abstentions - previous.solver_abstentions,
            "drift_warnings": self.current.drift_warnings - previous.drift_warnings,
            "corpus_regressions": self.current.corpus_regressions - previous.corpus_regressions,
        }

    @property
    def ok(self) -> bool:
        return self.current.open_risks == 0 and self.current.corpus_regressions == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "current": self.current.to_dict(),
            "history_points": len(self.history),
            "trend": self.trend,
        }


def build_team_dashboard(
    results: tuple[VerificationResult, ...],
    *,
    corpus_report: CorpusVerificationReport | dict[str, Any] | None = None,
    history: tuple[DashboardSnapshot, ...] = (),
    timestamp: str | None = None,
) -> TeamDashboardReport:
    """Build a dashboard from real verification results and optional corpus status."""

    resolved_timestamp = timestamp or datetime.now(UTC).isoformat(timespec="seconds")
    corpus_regressions, corpus_checks = _corpus_regression_counts(corpus_report)
    current = DashboardSnapshot(
        timestamp=resolved_timestamp,
        sources=tuple(_summarize_result(result) for result in results),
        corpus_regressions=corpus_regressions,
        corpus_checks=corpus_checks,
    )
    return TeamDashboardReport(current=current, history=history)


def load_dashboard_history(path: str | Path | None) -> tuple[DashboardSnapshot, ...]:
    """Load local dashboard JSONL history without touching network resources."""

    if path is None:
        return ()
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ()
    snapshots: list[DashboardSnapshot] = []
    try:
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeamDashboardError(f"invalid dashboard history {resolved}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise TeamDashboardError(f"dashboard history {resolved}:{line_number} is not an object")
            snapshots.append(DashboardSnapshot.from_dict(payload))
    except OSError as exc:
        raise TeamDashboardError(f"cannot read dashboard history at {resolved}: {exc}") from exc
    return tuple(snapshots)


def append_dashboard_history(path: str | Path, snapshot: DashboardSnapshot) -> Path:
    """Append one sanitized dashboard snapshot to local JSONL history."""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    except OSError as exc:
        raise TeamDashboardError(f"cannot write dashboard history at {resolved}: {exc}") from exc
    return resolved


def load_corpus_report_json(path: str | Path) -> dict[str, Any]:
    """Load a saved ``promptabi corpus verify --format json`` report."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TeamDashboardError(f"corpus report is not valid JSON at {resolved}:{exc.lineno}:{exc.colno}") from exc
    except OSError as exc:
        raise TeamDashboardError(f"cannot read corpus report at {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TeamDashboardError("corpus report root must be a JSON object")
    return payload


def render_team_dashboard_json(report: TeamDashboardReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_team_dashboard_text(report: TeamDashboardReport) -> str:
    current = report.current
    trend = report.trend
    lines = [
        "PromptABI team risk dashboard",
        f"timestamp: {current.timestamp}",
        f"status: {'PASS' if report.ok else 'ATTENTION'}",
        (
            "totals: "
            f"open risks {current.open_risks} ({_signed(trend['open_risks'])}), "
            f"accepted suppressions {current.accepted_suppressions} ({_signed(trend['accepted_suppressions'])}), "
            f"solver abstentions {current.solver_abstentions} ({_signed(trend['solver_abstentions'])}), "
            f"drift warnings {current.drift_warnings} ({_signed(trend['drift_warnings'])}), "
            f"corpus regressions {current.corpus_regressions} ({_signed(trend['corpus_regressions'])})"
        ),
        f"history points: {len(report.history)}",
    ]
    if current.corpus_checks:
        lines.append(f"corpus checks: {current.corpus_checks}")
    for source in current.sources:
        lines.append(
            f"- {source.name}: {source.open_risks} open risk(s), "
            f"{source.accepted_suppressions} accepted suppression(s), "
            f"{source.solver_abstentions} solver abstention(s), "
            f"{source.drift_warnings} drift warning(s)"
        )
        if source.by_rule:
            top_rules = ", ".join(f"{rule}={count}" for rule, count in list(source.by_rule.items())[:5])
            lines.append(f"  top rules: {top_rules}")
        if source.by_artifact_kind:
            kinds = ", ".join(f"{kind}={count}" for kind, count in source.by_artifact_kind.items())
            lines.append(f"  artifact kinds: {kinds}")
    return "\n".join(lines) + "\n"


def _summarize_result(result: VerificationResult) -> DashboardSourceSummary:
    diagnostics = tuple(result.diagnostics)
    policy_suppression_count = len(result.config.policy.suppressions)
    by_severity: Counter[str] = Counter(diagnostic.severity.value for diagnostic in diagnostics)
    by_rule: Counter[str] = Counter(diagnostic.rule_id for diagnostic in diagnostics)
    by_artifact_kind: Counter[str] = Counter(
        diagnostic.artifact.kind for diagnostic in diagnostics if diagnostic.artifact is not None
    )
    accepted_suppressions = policy_suppression_count + sum(1 for diagnostic in diagnostics if _is_suppression(diagnostic))
    return DashboardSourceSummary(
        name=result.config.name,
        diagnostics=len(diagnostics),
        open_risks=sum(1 for diagnostic in diagnostics if _is_open_risk(diagnostic)),
        accepted_suppressions=accepted_suppressions,
        solver_abstentions=sum(1 for diagnostic in diagnostics if _is_solver_abstention(diagnostic)),
        drift_warnings=sum(1 for diagnostic in diagnostics if _is_drift_warning(diagnostic)),
        by_severity=dict(by_severity),
        by_rule=dict(by_rule),
        by_artifact_kind=dict(by_artifact_kind),
    )


def _is_open_risk(diagnostic: Diagnostic) -> bool:
    return diagnostic.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.WARNING} and not _is_suppression(diagnostic)


def _is_suppression(diagnostic: Diagnostic) -> bool:
    return diagnostic.rule_id == "diagnostic-suppressed" or str(dict(diagnostic.properties).get("sarif_suppression")) == "accepted"


def _is_solver_abstention(diagnostic: Diagnostic) -> bool:
    return CheckMode.ABSTAINING in diagnostic.check_modes or "abstain" in diagnostic.rule_id or "unknown" in diagnostic.rule_id


def _is_drift_warning(diagnostic: Diagnostic) -> bool:
    text = diagnostic.rule_id.lower()
    return "drift" in text or "regression" in text or "migration" in text


def _corpus_regression_counts(report: CorpusVerificationReport | dict[str, Any] | None) -> tuple[int, int]:
    if report is None:
        return 0, 0
    if isinstance(report, CorpusVerificationReport):
        return sum(1 for check in report.checks if not check.passed), len(report.checks)
    checks = report.get("checks", ())
    if not isinstance(checks, list):
        raise TeamDashboardError("corpus report checks must be a list")
    regressions = 0
    for check in checks:
        if not isinstance(check, dict):
            raise TeamDashboardError("corpus report check entries must be objects")
        if check.get("passed") is False:
            regressions += 1
    return regressions, len(checks)


def _source_from_mapping(data: Any) -> DashboardSourceSummary:
    if not isinstance(data, dict):
        raise TeamDashboardError("dashboard source entry must be an object")
    return DashboardSourceSummary(
        name=str(data["name"]),
        diagnostics=int(data.get("diagnostics", 0)),
        open_risks=int(data.get("open_risks", 0)),
        accepted_suppressions=int(data.get("accepted_suppressions", 0)),
        solver_abstentions=int(data.get("solver_abstentions", 0)),
        drift_warnings=int(data.get("drift_warnings", 0)),
        by_severity=_str_int_dict(data.get("by_severity", {})),
        by_rule=_str_int_dict(data.get("by_rule", {})),
        by_artifact_kind=_str_int_dict(data.get("by_artifact_kind", {})),
    )


def _str_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TeamDashboardError("dashboard count fields must be objects")
    return {str(key): int(count) for key, count in value.items()}


def _signed(value: int) -> str:
    return f"{value:+d}"
