"""Offline beta-program case-study replay for PromptABI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .diagnostics import CheckMode, Diagnostic
from .session import VerificationSession


DEFAULT_BETA_PROGRAM_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "beta" / "beta_program.json"
BETA_PROGRAM_VERSION = 1


class BetaProgramError(ValueError):
    """Raised when a beta-program manifest is malformed or does not replay."""


@dataclass(frozen=True, slots=True)
class BetaIssue:
    """One issue or upstreamable bug collected during beta replay."""

    title: str
    url: str
    status: str
    rule_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "status": self.status,
            "rule_ids": list(self.rule_ids),
        }


@dataclass(frozen=True, slots=True)
class BetaCaseStudy:
    """Human-readable case-study summary backed by a replayed config."""

    summary: str
    root_cause: str
    production_symptom: str
    fix: str

    def to_dict(self) -> dict[str, str]:
        return {
            "summary": self.summary,
            "root_cause": self.root_cause,
            "production_symptom": self.production_symptom,
            "fix": self.fix,
        }


@dataclass(frozen=True, slots=True)
class BetaCase:
    """One beta project or case-study replay target."""

    case_id: str
    app_name: str
    repository: str
    source_kind: str
    config_path: Path
    expected_rule_ids: tuple[str, ...]
    expected_absent_rule_ids: tuple[str, ...]
    labels: tuple[str, ...]
    issues: tuple[BetaIssue, ...]
    tuning_actions: tuple[str, ...]
    case_study: BetaCaseStudy
    abstention_focus: bool = False


@dataclass(frozen=True, slots=True)
class BetaCaseResult:
    """Replay result for one beta case."""

    case: BetaCase
    observed_rule_ids: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...] = field(repr=False)
    missing_expected_rule_ids: tuple[str, ...] = ()
    false_positive_rule_ids: tuple[str, ...] = ()
    actionable_abstention_rule_ids: tuple[str, ...] = ()
    unactionable_abstention_rule_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            not self.missing_expected_rule_ids
            and not self.false_positive_rule_ids
            and not self.unactionable_abstention_rule_ids
            and (not self.case.abstention_focus or bool(self.actionable_abstention_rule_ids))
        )

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = {"error": 0, "warning": 0, "info": 0}
        for diagnostic in self.diagnostics:
            counts[diagnostic.severity.value] = counts.get(diagnostic.severity.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case.case_id,
            "app_name": self.case.app_name,
            "repository": self.case.repository,
            "source_kind": self.case.source_kind,
            "config": str(self.case.config_path),
            "labels": list(self.case.labels),
            "passed": self.passed,
            "expected_rule_ids": list(self.case.expected_rule_ids),
            "expected_absent_rule_ids": list(self.case.expected_absent_rule_ids),
            "observed_rule_ids": list(self.observed_rule_ids),
            "missing_expected_rule_ids": list(self.missing_expected_rule_ids),
            "false_positive_rule_ids": list(self.false_positive_rule_ids),
            "actionable_abstention_rule_ids": list(self.actionable_abstention_rule_ids),
            "unactionable_abstention_rule_ids": list(self.unactionable_abstention_rule_ids),
            "diagnostic_count": len(self.diagnostics),
            "severity_counts": self.severity_counts,
            "issues": [issue.to_dict() for issue in self.case.issues],
            "tuning_actions": list(self.case.tuning_actions),
            "case_study": self.case.case_study.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BetaProgramReport:
    """Aggregated beta-program replay report."""

    path: Path
    methodology: str
    results: tuple[BetaCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results) and self.upstream_issue_count > 0 and self.tuning_action_count > 0

    @property
    def upstream_issue_count(self) -> int:
        return sum(len(result.case.issues) for result in self.results)

    @property
    def upstreamed_bug_count(self) -> int:
        return sum(
            1
            for result in self.results
            for issue in result.case.issues
            if issue.status in {"filed", "fixed", "triaged"}
        )

    @property
    def tuning_action_count(self) -> int:
        return sum(len(result.case.tuning_actions) for result in self.results)

    @property
    def case_study_count(self) -> int:
        return len(self.results)

    @property
    def abstention_case_count(self) -> int:
        return sum(1 for result in self.results if result.case.abstention_focus)

    @property
    def actionable_abstention_count(self) -> int:
        return sum(len(result.actionable_abstention_rule_ids) for result in self.results)

    @property
    def false_positive_count(self) -> int:
        return sum(len(result.false_positive_rule_ids) for result in self.results)

    @property
    def missing_expected_count(self) -> int:
        return sum(len(result.missing_expected_rule_ids) for result in self.results)

    @property
    def diagnostic_count(self) -> int:
        return sum(len(result.diagnostics) for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": BETA_PROGRAM_VERSION,
            "path": str(self.path),
            "methodology": self.methodology,
            "passed": self.passed,
            "project_count": len(self.results),
            "case_study_count": self.case_study_count,
            "diagnostic_count": self.diagnostic_count,
            "upstream_issue_count": self.upstream_issue_count,
            "upstreamed_bug_count": self.upstreamed_bug_count,
            "tuning_action_count": self.tuning_action_count,
            "abstention_case_count": self.abstention_case_count,
            "actionable_abstention_count": self.actionable_abstention_count,
            "false_positive_count": self.false_positive_count,
            "missing_expected_count": self.missing_expected_count,
            "cases": [result.to_dict() for result in self.results],
        }


def load_beta_cases(path: str | Path | None = None) -> tuple[str, tuple[BetaCase, ...]]:
    """Load and validate the beta-program manifest."""

    manifest_path = Path(path) if path is not None else DEFAULT_BETA_PROGRAM_PATH
    payload = _read_json_object(manifest_path)
    if payload.get("manifest_version") != BETA_PROGRAM_VERSION:
        raise BetaProgramError(f"{manifest_path} has unsupported beta manifest_version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise BetaProgramError(f"{manifest_path} field 'methodology' must be a non-empty string")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BetaProgramError(f"{manifest_path} field 'cases' must be a non-empty list")
    repo_root = manifest_path.parent.parent.parent
    cases = tuple(sorted((_case_from_mapping(manifest_path, repo_root, raw) for raw in raw_cases), key=lambda item: item.case_id))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise BetaProgramError(f"{manifest_path} contains duplicate beta case ids")
    if not any(case.abstention_focus for case in cases):
        raise BetaProgramError(f"{manifest_path} must include at least one abstention-focused beta case")
    if not any(case.issues for case in cases):
        raise BetaProgramError(f"{manifest_path} must record at least one beta issue")
    if not any(case.tuning_actions for case in cases):
        raise BetaProgramError(f"{manifest_path} must record false-positive or abstention tuning actions")
    return methodology, cases


def run_beta_program(path: str | Path | None = None) -> BetaProgramReport:
    """Replay the beta-program configs against real PromptABI checks."""

    manifest_path = Path(path) if path is not None else DEFAULT_BETA_PROGRAM_PATH
    methodology, cases = load_beta_cases(manifest_path)
    return BetaProgramReport(
        path=manifest_path,
        methodology=methodology,
        results=tuple(_run_case(case) for case in cases),
    )


def render_beta_program_json(report: BetaProgramReport) -> str:
    """Render a beta-program report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_beta_program_text(report: BetaProgramReport) -> str:
    """Render a concise beta-program summary."""

    lines = [
        "PromptABI beta program",
        f"corpus: {report.path}",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        f"projects: {len(report.results)}",
        f"diagnostics replayed: {report.diagnostic_count}",
        f"upstream issues: {report.upstream_issue_count} ({report.upstreamed_bug_count} filed/fixed/triaged)",
        f"tuning actions: {report.tuning_action_count}",
        f"false positives from tuned absent rules: {report.false_positive_count}",
        f"actionable abstentions: {report.actionable_abstention_count}/{report.abstention_case_count} case(s)",
    ]
    for result in report.results:
        lines.append(
            f"- {result.case.case_id}: {'PASS' if result.passed else 'FAIL'} "
            f"observed={','.join(result.observed_rule_ids) or 'none'}"
        )
        if result.missing_expected_rule_ids:
            lines.append(f"  missing: {', '.join(result.missing_expected_rule_ids)}")
        if result.false_positive_rule_ids:
            lines.append(f"  false positives: {', '.join(result.false_positive_rule_ids)}")
        if result.unactionable_abstention_rule_ids:
            lines.append(f"  unactionable abstentions: {', '.join(result.unactionable_abstention_rule_ids)}")
    return "\n".join(lines) + "\n"


def _run_case(case: BetaCase) -> BetaCaseResult:
    result = VerificationSession.from_config_file(case.config_path).run()
    observed = tuple(sorted({diagnostic.rule_id for diagnostic in result.diagnostics}))
    expected = set(case.expected_rule_ids)
    absent = set(case.expected_absent_rule_ids)
    actionable_abstentions: list[str] = []
    unactionable_abstentions: list[str] = []
    for diagnostic in result.diagnostics:
        if not _is_abstention(diagnostic):
            continue
        if _is_actionable_abstention(diagnostic):
            actionable_abstentions.append(diagnostic.rule_id)
        else:
            unactionable_abstentions.append(diagnostic.rule_id)
    return BetaCaseResult(
        case=case,
        observed_rule_ids=observed,
        diagnostics=result.diagnostics,
        missing_expected_rule_ids=tuple(sorted(expected.difference(observed))),
        false_positive_rule_ids=tuple(sorted(absent.intersection(observed))),
        actionable_abstention_rule_ids=tuple(sorted(set(actionable_abstentions))),
        unactionable_abstention_rule_ids=tuple(sorted(set(unactionable_abstentions))),
    )


def _case_from_mapping(path: Path, repo_root: Path, raw: object) -> BetaCase:
    if not isinstance(raw, dict):
        raise BetaProgramError(f"{path} cases must be JSON objects")
    case_id = _required_string(path, raw, "id")
    config = _required_string(path, raw, "config")
    case_study_raw = raw.get("case_study")
    if not isinstance(case_study_raw, dict):
        raise BetaProgramError(f"{path} case {case_id!r} field 'case_study' must be an object")
    issues = _issues_from_value(path, case_id, raw.get("issues", ()))
    return BetaCase(
        case_id=case_id,
        app_name=_required_string(path, raw, "app_name"),
        repository=_required_string(path, raw, "repository"),
        source_kind=_required_string(path, raw, "source_kind"),
        config_path=(repo_root / config).resolve(),
        expected_rule_ids=_string_tuple(raw.get("expected_rule_ids"), path, case_id, "expected_rule_ids", allow_empty=False),
        expected_absent_rule_ids=_string_tuple(raw.get("expected_absent_rule_ids", ()), path, case_id, "expected_absent_rule_ids"),
        labels=_string_tuple(raw.get("labels", ()), path, case_id, "labels"),
        issues=issues,
        tuning_actions=_string_tuple(raw.get("tuning_actions", ()), path, case_id, "tuning_actions"),
        case_study=BetaCaseStudy(
            summary=_required_string(path, case_study_raw, "summary", case_id=case_id),
            root_cause=_required_string(path, case_study_raw, "root_cause", case_id=case_id),
            production_symptom=_required_string(path, case_study_raw, "production_symptom", case_id=case_id),
            fix=_required_string(path, case_study_raw, "fix", case_id=case_id),
        ),
        abstention_focus=bool(raw.get("abstention_focus", False)),
    )


def _issues_from_value(path: Path, case_id: str, value: object) -> tuple[BetaIssue, ...]:
    if not isinstance(value, list):
        raise BetaProgramError(f"{path} case {case_id!r} field 'issues' must be a list")
    issues: list[BetaIssue] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BetaProgramError(f"{path} case {case_id!r} issue {index} must be an object")
        issues.append(
            BetaIssue(
                title=_required_string(path, item, "title", case_id=case_id),
                url=_required_string(path, item, "url", case_id=case_id),
                status=_required_string(path, item, "status", case_id=case_id),
                rule_ids=_string_tuple(item.get("rule_ids"), path, case_id, "issue.rule_ids", allow_empty=False),
            )
        )
    return tuple(issues)


def _is_abstention(diagnostic: Diagnostic) -> bool:
    return CheckMode.ABSTAINING in diagnostic.check_modes or "abstain" in diagnostic.rule_id


def _is_actionable_abstention(diagnostic: Diagnostic) -> bool:
    message = diagnostic.message.lower()
    explains_scope = (
        "could not be analyzed" in message
        or "unsupported" in message
        or "outside" in message
        or "exceeded" in message
    )
    return diagnostic.witness is not None and bool(diagnostic.suggestions or explains_scope)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BetaProgramError(f"beta manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BetaProgramError(f"beta manifest is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise BetaProgramError(f"{path} root must be a JSON object")
    return payload


def _required_string(path: Path, raw: dict[str, object], key: str, *, case_id: str | None = None) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        prefix = f"{path} case {case_id!r}" if case_id is not None else str(path)
        raise BetaProgramError(f"{prefix} field '{key}' must be a non-empty string")
    return value


def _string_tuple(
    value: object,
    path: Path,
    case_id: str,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise BetaProgramError(f"{path} case {case_id!r} field '{field_name}' must be a list of non-empty strings")
    normalized = tuple(sorted(dict.fromkeys(value)))
    if not normalized and not allow_empty:
        raise BetaProgramError(f"{path} case {case_id!r} field '{field_name}' must not be empty")
    return normalized
