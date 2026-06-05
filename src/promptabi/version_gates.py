"""Semantic-version gates for PromptABI contract diffs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity
from .diff import diff_config_files
from .policies import PolicyError
from .source import build_json_source_map


VERSION_GATE_POLICY_VERSION = 1


class SemverImpact(StrEnum):
    """Deployment impact levels for contract changes."""

    PATCH_SAFE = "patch-safe"
    MINOR_BREAKING = "minor-breaking"
    MAJOR_BREAKING = "major-breaking"

    @property
    def rank(self) -> int:
        return {
            SemverImpact.PATCH_SAFE: 0,
            SemverImpact.MINOR_BREAKING: 1,
            SemverImpact.MAJOR_BREAKING: 2,
        }[self]

    def exceeds(self, allowed: "SemverImpact") -> bool:
        return self.rank > allowed.rank


class VersionGateFindingStatus(StrEnum):
    """Outcome for one classified diff diagnostic."""

    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class VersionGateRule:
    """One policy override for classifying a diff diagnostic."""

    impact: SemverImpact
    rule_id: str | None = None
    artifact: str | None = None
    artifact_kind: str | None = None
    kind: str | None = None
    field: str | None = None
    rationale: str | None = None

    @property
    def specificity(self) -> int:
        return sum(
            value is not None
            for value in (self.rule_id, self.artifact, self.artifact_kind, self.kind, self.field)
        )

    def matches(self, diagnostic: Diagnostic) -> bool:
        properties = dict(diagnostic.properties)
        if self.rule_id is not None and self.rule_id != diagnostic.rule_id:
            return False
        if self.artifact is not None:
            if diagnostic.artifact is None or self.artifact != diagnostic.artifact.name:
                return False
        if self.artifact_kind is not None:
            if diagnostic.artifact is None or self.artifact_kind != diagnostic.artifact.kind:
                return False
        if self.kind is not None and self.kind != properties.get("kind"):
            return False
        if self.field is not None and self.field != properties.get("field"):
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        match: dict[str, object] = {}
        for key in ("rule_id", "artifact", "artifact_kind", "kind", "field"):
            value = getattr(self, key)
            if value is not None:
                match[key] = value
        data: dict[str, object] = {"impact": self.impact.value, "match": match}
        if self.rationale is not None:
            data["rationale"] = self.rationale
        return data


@dataclass(frozen=True, slots=True)
class VersionGatePolicy:
    """Policy used to classify contract changes by semantic-version impact."""

    rules: tuple[VersionGateRule, ...] = ()
    default_unknown_impact: SemverImpact = SemverImpact.MINOR_BREAKING
    source_path: str | None = None

    def classify(self, diagnostic: Diagnostic) -> tuple[SemverImpact, str, VersionGateRule | None]:
        matching = tuple((index, rule) for index, rule in enumerate(self.rules) if rule.matches(diagnostic))
        if matching:
            index, rule = max(matching, key=lambda item: (item[1].specificity, item[0]))
            return rule.impact, "policy", rule
        default = _default_impact(diagnostic, self.default_unknown_impact)
        return default, "default", None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version_gate_policy_version": VERSION_GATE_POLICY_VERSION,
            "default_unknown_impact": self.default_unknown_impact.value,
            "rules": [rule.to_dict() for rule in self.rules],
        }
        if self.source_path is not None:
            data["source_path"] = self.source_path
        return data


@dataclass(frozen=True, slots=True)
class VersionGateFinding:
    """One semantic-version classification for a diff diagnostic."""

    status: VersionGateFindingStatus
    required_impact: SemverImpact
    allowed_impact: SemverImpact
    diagnostic: Diagnostic
    source: str
    matched_rule: VersionGateRule | None = None

    @property
    def passed(self) -> bool:
        return self.status is VersionGateFindingStatus.PASS

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "status": self.status.value,
            "required_impact": self.required_impact.value,
            "allowed_impact": self.allowed_impact.value,
            "source": self.source,
            "diagnostic": self.diagnostic.to_dict(),
        }
        if self.matched_rule is not None:
            data["matched_rule"] = self.matched_rule.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class VersionGateReport:
    """Complete semantic-version gate report for two PromptABI configs."""

    baseline: str
    current: str
    allowed_impact: SemverImpact
    policy: VersionGatePolicy
    findings: tuple[VersionGateFinding, ...]

    @property
    def ok(self) -> bool:
        return all(finding.passed for finding in self.findings)

    @property
    def max_required_impact(self) -> SemverImpact:
        if not self.findings:
            return SemverImpact.PATCH_SAFE
        return max((finding.required_impact for finding in self.findings), key=lambda impact: impact.rank)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": VERSION_GATE_POLICY_VERSION,
            "ok": self.ok,
            "baseline": self.baseline,
            "current": self.current,
            "allowed_impact": self.allowed_impact.value,
            "max_required_impact": self.max_required_impact.value,
            "policy": self.policy.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def empty_version_gate_policy() -> VersionGatePolicy:
    return VersionGatePolicy()


def load_version_gate_policy(path: str | Path) -> VersionGatePolicy:
    """Load a JSON semantic-version gate policy."""

    policy_path = Path(path).expanduser().resolve()
    try:
        text = policy_path.read_text(encoding="utf-8")
        raw = json.loads(text)
    except FileNotFoundError as exc:
        raise PolicyError(f"version-gate policy file not found: {policy_path}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyError(
            f"version-gate policy file is not valid JSON at {policy_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise PolicyError("version-gate policy root must be a JSON object")
    try:
        source_map = build_json_source_map(text, policy_path)
    except ValueError as exc:
        raise PolicyError(f"version-gate policy source map could not be built: {exc}") from exc
    try:
        return version_gate_policy_from_mapping(raw, source_path=str(policy_path))
    except PolicyError:
        raise
    except ValueError as exc:
        span = source_map.span_for(()) or source_map.key_span_for(())
        location = f" at {span.path}:{span.start_line}:{span.start_column}" if span is not None else ""
        raise PolicyError(f"invalid version-gate policy{location}: {exc}") from exc


def version_gate_policy_from_mapping(
    data: dict[str, Any],
    *,
    source_path: str | None = None,
) -> VersionGatePolicy:
    """Build a semantic-version gate policy from a JSON-like mapping."""

    version = data.get("version_gate_policy_version", data.get("version", VERSION_GATE_POLICY_VERSION))
    if version != VERSION_GATE_POLICY_VERSION:
        raise PolicyError(f"unsupported version-gate policy version: {version!r}")
    default_unknown = _impact(data.get("default_unknown_impact", SemverImpact.MINOR_BREAKING.value))
    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, list):
        raise PolicyError("version-gate policy field 'rules' must be a list")
    return VersionGatePolicy(
        rules=tuple(_rule_from_mapping(item, index) for index, item in enumerate(raw_rules)),
        default_unknown_impact=default_unknown,
        source_path=source_path,
    )


def run_version_gate(
    baseline_path: str | Path,
    current_path: str | Path,
    *,
    allowed_impact: SemverImpact | str = SemverImpact.PATCH_SAFE,
    policy: VersionGatePolicy | None = None,
    policy_path: str | Path | None = None,
) -> VersionGateReport:
    """Diff two configs and classify every contract change by semver impact."""

    if policy is not None and policy_path is not None:
        raise PolicyError("pass either policy or policy_path, not both")
    resolved_policy = policy or (load_version_gate_policy(policy_path) if policy_path is not None else empty_version_gate_policy())
    allowed = _impact(allowed_impact)
    diff_result = diff_config_files(baseline_path, current_path)
    findings = tuple(
        sorted(
            (_finding_for(diagnostic, allowed, resolved_policy) for diagnostic in diff_result.diagnostics),
            key=lambda item: (item.status.value, -item.required_impact.rank, item.diagnostic.sort_key),
        )
    )
    return VersionGateReport(
        baseline=str(Path(baseline_path)),
        current=str(Path(current_path)),
        allowed_impact=allowed,
        policy=resolved_policy,
        findings=findings,
    )


def render_version_gate_json(report: VersionGateReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_version_gate_text(report: VersionGateReport) -> str:
    lines = [
        "PromptABI semantic version gate",
        f"baseline: {report.baseline}",
        f"current: {report.current}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"allowed impact: {report.allowed_impact.value}",
        f"max required impact: {report.max_required_impact.value}",
    ]
    for finding in report.findings:
        diagnostic = finding.diagnostic
        artifact = ""
        if diagnostic.artifact is not None:
            artifact = f" artifact={diagnostic.artifact.kind}:{diagnostic.artifact.name}"
        lines.append(
            f"- {finding.status.value.upper()} {finding.required_impact.value} "
            f"{diagnostic.rule_id}{artifact}: {diagnostic.message}"
        )
        if finding.matched_rule is not None and finding.matched_rule.rationale:
            lines.append(f"  rationale: {finding.matched_rule.rationale}")
    return "\n".join(lines) + "\n"


def _finding_for(diagnostic: Diagnostic, allowed: SemverImpact, policy: VersionGatePolicy) -> VersionGateFinding:
    required, source, matched_rule = policy.classify(diagnostic)
    status = VersionGateFindingStatus.FAIL if required.exceeds(allowed) else VersionGateFindingStatus.PASS
    return VersionGateFinding(
        status=status,
        required_impact=required,
        allowed_impact=allowed,
        diagnostic=diagnostic,
        source=source,
        matched_rule=matched_rule,
    )


def _rule_from_mapping(data: object, index: int) -> VersionGateRule:
    if not isinstance(data, dict):
        raise PolicyError(f"version-gate policy rule {index} must be an object")
    raw_match = data.get("match")
    if not isinstance(raw_match, dict):
        raise PolicyError(f"version-gate policy rule {index} field 'match' must be an object")
    allowed_keys = {"rule_id", "artifact", "artifact_kind", "kind", "field"}
    unknown = sorted(set(raw_match).difference(allowed_keys))
    if unknown:
        raise PolicyError(f"version-gate policy rule {index} has unknown match keys: {', '.join(unknown)}")
    match = {key: _optional_string(raw_match.get(key), f"rules[{index}].match.{key}") for key in allowed_keys}
    if not any(match.values()):
        raise PolicyError(f"version-gate policy rule {index} must set at least one match field")
    rationale = _optional_string(data.get("rationale"), f"rules[{index}].rationale")
    return VersionGateRule(
        impact=_impact(data.get("impact")),
        rule_id=match["rule_id"],
        artifact=match["artifact"],
        artifact_kind=match["artifact_kind"],
        kind=match["kind"],
        field=match["field"],
        rationale=rationale,
    )


def _impact(value: object) -> SemverImpact:
    if isinstance(value, SemverImpact):
        return value
    if not isinstance(value, str):
        raise PolicyError("semantic version impact must be a string")
    try:
        return SemverImpact(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in SemverImpact)
        raise PolicyError(f"unknown semantic version impact {value!r}; expected one of: {choices}") from exc


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PolicyError(f"version-gate policy field '{field_name}' must be a non-empty string")
    return value


def _default_impact(diagnostic: Diagnostic, default_unknown: SemverImpact) -> SemverImpact:
    properties = dict(diagnostic.properties)
    if diagnostic.rule_id == "diff-clean":
        return SemverImpact.PATCH_SAFE
    if diagnostic.rule_id in {"diff-check-added", "diff-artifact-added"}:
        return SemverImpact.PATCH_SAFE
    if diagnostic.rule_id in {"diff-check-removed", "diff-artifact-removed", "diff-artifact-kind-changed", "diff-context-regression"}:
        return SemverImpact.MAJOR_BREAKING
    if "abstain" in diagnostic.rule_id or CheckMode.ABSTAINING in diagnostic.check_modes:
        return SemverImpact.MAJOR_BREAKING
    if diagnostic.rule_id == "diff-artifact-drift":
        return SemverImpact.MINOR_BREAKING
    if diagnostic.rule_id == "diff-tokenizer-drift":
        return _tokenizer_impact(str(properties.get("kind", "")))
    if diagnostic.rule_id == "diff-provider-contract":
        return _provider_impact(str(properties.get("kind", "")), diagnostic.severity)
    if diagnostic.rule_id == "diff-framework-truncation":
        return _framework_impact(str(properties.get("field", "")), diagnostic.severity)
    return max(default_unknown, _severity_impact(diagnostic.severity), key=lambda impact: impact.rank)


def _tokenizer_impact(kind: str) -> SemverImpact:
    if kind in {"special-token-id-change", "bos-eos-change", "normalization-change"}:
        return SemverImpact.MAJOR_BREAKING
    if kind in {"chat-template-change", "stop-policy-change", "added-token-change"}:
        return SemverImpact.MINOR_BREAKING
    return SemverImpact.MAJOR_BREAKING


def _provider_impact(kind: str, severity: DiagnosticSeverity) -> SemverImpact:
    major = {
        "unsupported-provider",
        "request-field-loss",
        "response-field-loss",
        "tool-argument-encoding-mismatch",
        "context-limit-regression",
        "structured-output-mismatch",
        "routing-target-missing",
    }
    if kind in major:
        return SemverImpact.MAJOR_BREAKING
    return _severity_impact(severity)


def _framework_impact(field: str, severity: DiagnosticSeverity) -> SemverImpact:
    if field in {"max_context_tokens", "preserve_system", "preserve_tools"}:
        return SemverImpact.MAJOR_BREAKING
    if field in {"framework", "strategy"}:
        return SemverImpact.MINOR_BREAKING
    return _severity_impact(severity)


def _severity_impact(severity: DiagnosticSeverity) -> SemverImpact:
    if severity is DiagnosticSeverity.ERROR:
        return SemverImpact.MAJOR_BREAKING
    if severity is DiagnosticSeverity.WARNING:
        return SemverImpact.MINOR_BREAKING
    return SemverImpact.PATCH_SAFE
