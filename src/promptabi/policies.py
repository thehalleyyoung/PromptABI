"""Policy and suppression handling for PromptABI diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessStep, WitnessTrace
from .source import JsonSourceMap, build_json_source_map


class PolicyError(ValueError):
    """Raised when a policy or suppression file cannot be loaded soundly."""


@dataclass(frozen=True, slots=True)
class Suppression:
    """One accepted-risk suppression for a stable diagnostic shape."""

    rule_id: str
    justification: str
    fingerprint: str | None = None
    artifact: str | None = None
    artifact_kind: str | None = None
    path: str | None = None
    owner: str | None = None
    accepted_risk: str | None = None
    expires_on: date | None = None
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("suppression rule_id must be non-empty")
        for field_name in ("fingerprint", "artifact", "artifact_kind", "path", "owner", "accepted_risk"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"suppression {field_name} must be non-empty")
        if self.fingerprint is None and self.artifact is None and self.path is None:
            raise ValueError("suppression must set at least one of fingerprint, artifact, or path")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rule_id": self.rule_id,
            "justification": self.justification,
        }
        for key in ("fingerprint", "artifact", "artifact_kind", "path", "owner", "accepted_risk"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.expires_on is not None:
            data["expires_on"] = self.expires_on.isoformat()
        if self.span is not None:
            data["span"] = self.span.to_dict()
        return data

    def matches(self, diagnostic: Diagnostic) -> bool:
        if self.rule_id != diagnostic.rule_id:
            return False
        if self.fingerprint is not None and self.fingerprint != diagnostic.fingerprint:
            return False
        if self.artifact is not None:
            if diagnostic.artifact is None or self.artifact != diagnostic.artifact.name:
                return False
        if self.artifact_kind is not None:
            if diagnostic.artifact is None or self.artifact_kind != diagnostic.artifact.kind:
                return False
        if self.path is not None:
            diagnostic_path = None
            if diagnostic.span is not None:
                diagnostic_path = diagnostic.span.path
            elif diagnostic.artifact is not None:
                diagnostic_path = diagnostic.artifact.location_uri
            if diagnostic_path is None or Path(diagnostic_path).as_posix() != Path(self.path).as_posix():
                return False
        return True


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Repository policy for accepted risks and CI severity thresholds."""

    suppressions: tuple[Suppression, ...] = ()
    severity_threshold: DiagnosticSeverity | None = None
    require_justification: bool = True
    require_expiration: bool = True
    require_owner: bool = False
    source_paths: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(
            self.suppressions
            or self.severity_threshold is not None
            or not self.require_justification
            or not self.require_expiration
            or self.require_owner
            or self.source_paths
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "require_justification": self.require_justification,
            "require_expiration": self.require_expiration,
            "require_owner": self.require_owner,
            "suppressions": [suppression.to_dict() for suppression in self.suppressions],
        }
        if self.severity_threshold is not None:
            data["severity_threshold"] = self.severity_threshold.value
        if self.source_paths:
            data["source_paths"] = list(self.source_paths)
        return data


def empty_policy() -> VerificationPolicy:
    return VerificationPolicy()


def policy_from_config_mapping(
    data: dict[str, Any],
    *,
    base_dir: Path,
    source_map: JsonSourceMap | None = None,
) -> VerificationPolicy:
    """Load inline policy plus referenced policy/suppression files from config JSON."""

    policy = empty_policy()
    for policy_path in _string_list(data.get("policy_files", []), field_name="policy_files"):
        policy = merge_policies(policy, load_policy_file(base_dir / policy_path))
    for suppression_path in _string_list(data.get("suppression_files", []), field_name="suppression_files"):
        policy = merge_policies(policy, load_policy_file(base_dir / suppression_path, suppressions_only=True))

    inline_policy = data.get("policy")
    if inline_policy is not None:
        if not isinstance(inline_policy, dict):
            raise PolicyError("config field 'policy' must be an object")
        policy = merge_policies(
            policy,
            policy_from_mapping(
                inline_policy,
                source_path=str(base_dir / "<inline-policy>"),
                source_map=source_map,
                source_prefix=("policy",),
            ),
        )
    inline_suppressions = data.get("suppressions")
    if inline_suppressions is not None:
        policy = merge_policies(
            policy,
            policy_from_mapping(
                {"suppressions": inline_suppressions},
                source_path=str(base_dir / "<inline-suppressions>"),
                source_map=source_map,
                source_prefix=(),
                suppressions_only=True,
            ),
        )
    return policy


def load_policy_file(path: str | Path, *, suppressions_only: bool = False) -> VerificationPolicy:
    """Load a JSON policy file or suppression list from disk."""

    policy_path = Path(path).expanduser().resolve()
    try:
        text = policy_path.read_text(encoding="utf-8")
        raw = json.loads(text)
    except FileNotFoundError as exc:
        raise PolicyError(f"policy file not found: {policy_path}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyError(
            f"policy file is not valid JSON at {policy_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if isinstance(raw, list):
        raw = {"suppressions": raw}
    if not isinstance(raw, dict):
        raise PolicyError("policy file root must be a JSON object or suppression list")
    try:
        source_map = build_json_source_map(text, policy_path)
    except ValueError as exc:
        raise PolicyError(f"policy source map could not be built: {exc}") from exc
    return policy_from_mapping(
        raw,
        source_path=str(policy_path),
        source_map=source_map,
        suppressions_only=suppressions_only,
    )


def policy_from_mapping(
    data: dict[str, Any],
    *,
    source_path: str,
    source_map: JsonSourceMap | None = None,
    source_prefix: tuple[str, ...] = (),
    suppressions_only: bool = False,
) -> VerificationPolicy:
    if not suppressions_only:
        severity_threshold = _optional_severity(data.get("severity_threshold"))
        require_justification = _optional_bool(data.get("require_justification"), default=True, field_name="require_justification")
        require_expiration = _optional_bool(data.get("require_expiration"), default=True, field_name="require_expiration")
        require_owner = _optional_bool(data.get("require_owner"), default=False, field_name="require_owner")
    else:
        severity_threshold = None
        require_justification = True
        require_expiration = True
        require_owner = False

    raw_suppressions = data.get("suppressions", [])
    if not isinstance(raw_suppressions, list):
        raise PolicyError("policy field 'suppressions' must be a list")
    suppressions = tuple(
        _suppression_from_mapping(
            item,
            source_map=source_map,
            source_prefix=(*source_prefix, "suppressions", str(index)),
        )
        for index, item in enumerate(raw_suppressions)
    )
    return VerificationPolicy(
        suppressions=suppressions,
        severity_threshold=severity_threshold,
        require_justification=require_justification,
        require_expiration=require_expiration,
        require_owner=require_owner,
        source_paths=(source_path,),
    )


def merge_policies(*policies: VerificationPolicy) -> VerificationPolicy:
    """Merge policies in config order; later scalar settings override earlier ones."""

    if not policies:
        return empty_policy()
    severity_threshold = next((policy.severity_threshold for policy in reversed(policies) if policy.severity_threshold is not None), None)
    return VerificationPolicy(
        suppressions=tuple(suppression for policy in policies for suppression in policy.suppressions),
        severity_threshold=severity_threshold,
        require_justification=policies[-1].require_justification,
        require_expiration=policies[-1].require_expiration,
        require_owner=policies[-1].require_owner,
        source_paths=tuple(path for policy in policies for path in policy.source_paths),
    )


def apply_policy_diagnostics(
    diagnostics: tuple[Diagnostic, ...],
    policy: VerificationPolicy,
    *,
    today: date | None = None,
) -> tuple[Diagnostic, ...]:
    """Return diagnostics after applying valid suppressions and policy thresholds."""

    if not policy.active:
        return diagnostics
    today = today or date.today()
    invalid_suppression_diagnostics = tuple(_invalid_suppression_diagnostics(policy, today=today))
    remaining: list[Diagnostic] = []
    policy_diagnostics: list[Diagnostic] = list(invalid_suppression_diagnostics)
    valid_suppressions = tuple(
        suppression
        for suppression in policy.suppressions
        if _suppression_errors(suppression, policy, today=today) == ()
    )
    for diagnostic in diagnostics:
        suppression = next((item for item in valid_suppressions if item.matches(diagnostic)), None)
        if suppression is None:
            remaining.append(diagnostic)
        else:
            policy_diagnostics.append(_suppressed_diagnostic(diagnostic, suppression))

    if policy.severity_threshold is not None:
        policy_diagnostics.extend(_threshold_diagnostics(tuple(remaining), policy.severity_threshold))
    return tuple(sorted((*remaining, *policy_diagnostics), key=lambda item: item.sort_key))


def _suppression_from_mapping(
    data: Any,
    *,
    source_map: JsonSourceMap | None,
    source_prefix: tuple[str, ...],
) -> Suppression:
    if not isinstance(data, dict):
        raise PolicyError("each suppression must be an object")
    rule_id = _required_string(data, "rule_id")
    justification = _string_or_empty(data.get("justification"))
    expires_on = _optional_date(data.get("expires_on") or data.get("expires"))
    try:
        return Suppression(
            rule_id=rule_id,
            fingerprint=_optional_string(data.get("fingerprint"), field_name="fingerprint"),
            artifact=_optional_string(data.get("artifact"), field_name="artifact"),
            artifact_kind=_optional_string(data.get("artifact_kind"), field_name="artifact_kind"),
            path=_optional_string(data.get("path"), field_name="path"),
            justification=justification,
            owner=_optional_string(data.get("owner"), field_name="owner"),
            accepted_risk=_optional_string(data.get("accepted_risk"), field_name="accepted_risk"),
            expires_on=expires_on,
            span=_suppression_span(source_map, source_prefix),
        )
    except ValueError as exc:
        raise PolicyError(str(exc)) from exc


def _invalid_suppression_diagnostics(policy: VerificationPolicy, *, today: date) -> tuple[Diagnostic, ...]:
    diagnostics = []
    for suppression in policy.suppressions:
        errors = _suppression_errors(suppression, policy, today=today)
        if errors:
            diagnostics.append(
                Diagnostic(
                    rule_id="policy-suppression-invalid",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"suppression for rule '{suppression.rule_id}' is invalid: {'; '.join(errors)}",
                    span=suppression.span,
                    check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                    suggestions=("Add owner/justification/expiration metadata or remove the stale accepted risk.",),
                    properties=(
                        ("suppressed_rule_id", suppression.rule_id),
                        ("suppression", suppression.to_dict()),
                    ),
                    witness=WitnessTrace(
                        summary="PromptABI rejected a suppression before applying it.",
                        steps=tuple(WitnessStep(action="validate suppression metadata", output=error) for error in errors),
                    ),
                )
            )
    return tuple(diagnostics)


def _suppression_errors(suppression: Suppression, policy: VerificationPolicy, *, today: date) -> tuple[str, ...]:
    errors = []
    if policy.require_justification and not suppression.justification.strip():
        errors.append("justification is required")
    if policy.require_expiration and suppression.expires_on is None:
        errors.append("expires_on is required")
    if policy.require_owner and not (suppression.owner or "").strip():
        errors.append("owner is required")
    if suppression.expires_on is not None and suppression.expires_on < today:
        errors.append(f"expires_on {suppression.expires_on.isoformat()} is in the past")
    return tuple(errors)


def _suppressed_diagnostic(diagnostic: Diagnostic, suppression: Suppression) -> Diagnostic:
    properties: tuple[tuple[str, Any], ...] = (
        ("accepted_risk", suppression.accepted_risk or suppression.justification),
        ("original_fingerprint", diagnostic.fingerprint),
        ("original_rule_id", diagnostic.rule_id),
        ("original_severity", diagnostic.severity.value),
        ("sarif_suppression", "external"),
        ("sarif_suppression_justification", suppression.justification),
        ("suppression", suppression.to_dict()),
    )
    return Diagnostic(
        rule_id="diagnostic-suppressed",
        severity=DiagnosticSeverity.INFO,
        message=f"suppressed {diagnostic.severity.value} diagnostic '{diagnostic.rule_id}' as accepted risk",
        artifact=diagnostic.artifact,
        span=suppression.span or diagnostic.span,
        witness=WitnessTrace(
            summary="A policy suppression matched this diagnostic, so it no longer contributes to CI failure.",
            steps=(
                WitnessStep(action="match rule", input=suppression.rule_id, output=diagnostic.rule_id),
                WitnessStep(action="match fingerprint", input=suppression.fingerprint or "*", output=diagnostic.fingerprint),
                WitnessStep(action="record accepted risk", output=suppression.justification),
            ),
        ),
        suggestions=("Review or remove the suppression before its expiration.",),
        check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        properties=properties,
    )


def _threshold_diagnostics(diagnostics: tuple[Diagnostic, ...], threshold: DiagnosticSeverity) -> tuple[Diagnostic, ...]:
    gated = tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity.rank <= threshold.rank
        and diagnostic.rule_id not in {"policy-threshold-violation", "diagnostic-suppressed"}
    )
    if not gated:
        return ()
    return (
        Diagnostic(
            rule_id="policy-threshold-violation",
            severity=DiagnosticSeverity.ERROR,
            message=f"policy threshold '{threshold.value}' matched {len(gated)} unsuppressed diagnostic(s)",
            check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
            suggestions=("Fix the diagnostics or add reviewed, unexpired suppressions with accepted-risk justifications.",),
            properties=(
                ("matched_count", len(gated)),
                ("severity_threshold", threshold.value),
                ("matched_fingerprints", [diagnostic.fingerprint for diagnostic in gated]),
            ),
            witness=WitnessTrace(
                summary="Repository policy requires all diagnostics at or above the threshold to be resolved or explicitly suppressed.",
                steps=tuple(
                    WitnessStep(
                        action="classify unsuppressed diagnostic",
                        input=diagnostic.rule_id,
                        output=f"{diagnostic.severity.value}:{diagnostic.fingerprint}",
                    )
                    for diagnostic in gated
                ),
            ),
        ),
    )


def _required_string(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"suppression field '{field_name}' must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"suppression field '{field_name}' must be a non-empty string")
    return value.strip()


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PolicyError("suppression field 'justification' must be a string")
    return value.strip()


def _optional_bool(value: Any, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise PolicyError(f"policy field '{field_name}' must be a boolean")
    return value


def _optional_severity(value: Any) -> DiagnosticSeverity | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyError("policy field 'severity_threshold' must be a string")
    try:
        return DiagnosticSeverity(value)
    except ValueError as exc:
        choices = ", ".join(severity.value for severity in DiagnosticSeverity)
        raise PolicyError(f"policy field 'severity_threshold' must be one of: {choices}") from exc


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyError("suppression field 'expires_on' must be an ISO date string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise PolicyError("suppression field 'expires_on' must use YYYY-MM-DD") from exc
    return parsed.date()


def _string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise PolicyError(f"config field '{field_name}' must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _suppression_span(source_map: JsonSourceMap | None, source_prefix: tuple[str, ...]) -> SourceSpan | None:
    if source_map is None:
        return None
    return source_map.span_for(source_prefix) or source_map.key_span_for(source_prefix)
