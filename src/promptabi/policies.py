"""Policy and suppression handling for PromptABI diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
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
    witness_digest: str | None = None
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
        for field_name in ("fingerprint", "witness_digest", "artifact", "artifact_kind", "path", "owner", "accepted_risk"):
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
        for key in ("fingerprint", "witness_digest", "artifact", "artifact_kind", "path", "owner", "accepted_risk"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.expires_on is not None:
            data["expires_on"] = self.expires_on.isoformat()
        if self.span is not None:
            data["span"] = self.span.to_dict()
        return data

    def matches(self, diagnostic: Diagnostic) -> bool:
        if not self.matches_shape(diagnostic):
            return False
        if self.witness_digest is not None and self.witness_digest != diagnostic.witness_digest:
            return False
        return True

    def matches_shape(self, diagnostic: Diagnostic) -> bool:
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
class OrgPolicyPack:
    """Organization-wide constraints layered on top of repository policy."""

    required_checks: tuple[str, ...] = ()
    supported_fragments: tuple[tuple[str, tuple[CheckMode, ...]], ...] = ()
    max_solver_timeout_ms: int | None = None
    require_strict_no_network: bool = False
    forbid_local_usage_summary: bool = False
    approved_provider_fixture_sha256: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(
            self.required_checks
            or self.supported_fragments
            or self.max_solver_timeout_ms is not None
            or self.require_strict_no_network
            or self.forbid_local_usage_summary
            or self.approved_provider_fixture_sha256
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.required_checks:
            data["required_checks"] = list(self.required_checks)
        if self.supported_fragments:
            data["supported_fragments"] = {
                check: [mode.value for mode in modes] for check, modes in self.supported_fragments
            }
        if self.max_solver_timeout_ms is not None:
            data["max_solver_timeout_ms"] = self.max_solver_timeout_ms
        if self.require_strict_no_network:
            data["require_strict_no_network"] = True
        if self.forbid_local_usage_summary:
            data["forbid_local_usage_summary"] = True
        if self.approved_provider_fixture_sha256:
            data["approved_provider_fixture_sha256"] = list(self.approved_provider_fixture_sha256)
        return data


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Repository policy for accepted risks and CI severity thresholds."""

    suppressions: tuple[Suppression, ...] = ()
    severity_threshold: DiagnosticSeverity | None = None
    require_justification: bool = True
    require_expiration: bool = True
    require_owner: bool = True
    require_accepted_risk: bool = True
    require_witness_digest: bool = True
    severity_overrides: tuple[tuple[str, DiagnosticSeverity], ...] = ()
    org_policy: OrgPolicyPack = field(default_factory=OrgPolicyPack)
    source_paths: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(
            self.suppressions
            or self.severity_threshold is not None
            or not self.require_justification
            or not self.require_expiration
            or not self.require_owner
            or not self.require_accepted_risk
            or not self.require_witness_digest
            or self.severity_overrides
            or self.org_policy.active
            or self.source_paths
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "require_justification": self.require_justification,
            "require_expiration": self.require_expiration,
            "require_owner": self.require_owner,
            "require_accepted_risk": self.require_accepted_risk,
            "require_witness_digest": self.require_witness_digest,
            "suppressions": [suppression.to_dict() for suppression in self.suppressions],
        }
        if self.severity_overrides:
            data["severity_overrides"] = {
                rule_id: severity.value for rule_id, severity in self.severity_overrides
            }
        if self.severity_threshold is not None:
            data["severity_threshold"] = self.severity_threshold.value
        if self.org_policy.active:
            data["org_policy"] = self.org_policy.to_dict()
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
        require_owner = _optional_bool(data.get("require_owner"), default=True, field_name="require_owner")
        require_accepted_risk = _optional_bool(data.get("require_accepted_risk"), default=True, field_name="require_accepted_risk")
        require_witness_digest = _optional_bool(data.get("require_witness_digest"), default=True, field_name="require_witness_digest")
        severity_overrides = _severity_overrides(data.get("severity_overrides", {}))
        org_policy = _org_policy_pack_from_mapping(data)
    else:
        severity_threshold = None
        require_justification = True
        require_expiration = True
        require_owner = True
        require_accepted_risk = True
        require_witness_digest = True
        severity_overrides = ()
        org_policy = OrgPolicyPack()

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
        require_accepted_risk=require_accepted_risk,
        require_witness_digest=require_witness_digest,
        severity_overrides=severity_overrides,
        org_policy=org_policy,
        source_paths=(source_path,),
    )


def merge_policies(*policies: VerificationPolicy) -> VerificationPolicy:
    """Merge policies in config order while preserving restrictive org-pack constraints."""

    if not policies:
        return empty_policy()
    severity_threshold = next((policy.severity_threshold for policy in reversed(policies) if policy.severity_threshold is not None), None)
    return VerificationPolicy(
        suppressions=tuple(suppression for policy in policies for suppression in policy.suppressions),
        severity_threshold=severity_threshold,
        require_justification=policies[-1].require_justification,
        require_expiration=policies[-1].require_expiration,
        require_owner=policies[-1].require_owner,
        require_accepted_risk=policies[-1].require_accepted_risk,
        require_witness_digest=policies[-1].require_witness_digest,
        severity_overrides=_merged_severity_overrides(policies),
        org_policy=_merged_org_policy(policies),
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
    diagnostics = _apply_severity_overrides(diagnostics, policy.severity_overrides)
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
            stale = tuple(
                item
                for item in valid_suppressions
                if item.witness_digest is not None
                and item.matches_shape(diagnostic)
                and item.witness_digest != diagnostic.witness_digest
            )
            policy_diagnostics.extend(_stale_witness_suppression_diagnostics(diagnostic, stale))
            remaining.append(diagnostic)
        else:
            policy_diagnostics.append(_suppressed_diagnostic(diagnostic, suppression))

    if policy.severity_threshold is not None:
        policy_diagnostics.extend(_threshold_diagnostics(tuple(remaining), policy.severity_threshold))
    return tuple(sorted((*remaining, *policy_diagnostics), key=lambda item: item.sort_key))


def apply_org_policy_diagnostics(
    config: Any,
    policy: VerificationPolicy,
    *,
    selected_checks: tuple[str, ...],
    check_modes: dict[str, tuple[CheckMode, ...]],
) -> tuple[Diagnostic, ...]:
    """Evaluate organization policy-pack constraints against a verification run."""

    org = policy.org_policy
    if not org.active:
        return ()
    diagnostics: list[Diagnostic] = []
    configured_checks = set(selected_checks)
    missing_checks = tuple(check for check in org.required_checks if check not in configured_checks)
    if missing_checks:
        diagnostics.append(
            _org_policy_diagnostic(
                "policy-pack-required-check-missing",
                DiagnosticSeverity.ERROR,
                f"organization policy pack requires {len(missing_checks)} check(s) not selected for this run",
                "required_checks",
                ", ".join(missing_checks),
                "Add the required checks to the PromptABI config or do not apply this organization policy pack.",
                extra=(("missing_checks", list(missing_checks)),),
            )
        )

    supported = dict(org.supported_fragments)
    for check_name in sorted(configured_checks & set(supported)):
        allowed = set(supported[check_name])
        actual = set(check_modes.get(check_name, ()))
        if not actual:
            diagnostics.append(
                _org_policy_diagnostic(
                    "policy-pack-supported-fragment-unknown",
                    DiagnosticSeverity.ERROR,
                    f"organization policy pack constrains check '{check_name}', but PromptABI has no mode metadata for it",
                    check_name,
                    "unknown",
                    "Use a registered check with declared guarantee modes or update the policy pack.",
                    extra=(("allowed_modes", sorted(mode.value for mode in allowed)),),
                )
            )
        elif actual.isdisjoint(allowed):
            diagnostics.append(
                _org_policy_diagnostic(
                    "policy-pack-supported-fragment-violation",
                    DiagnosticSeverity.ERROR,
                    f"check '{check_name}' runs outside the organization-supported guarantee fragment",
                    check_name,
                    ", ".join(sorted(mode.value for mode in actual)),
                    "Select a check/configuration whose guarantee modes are allowed by the organization policy pack.",
                    extra=(
                        ("allowed_modes", sorted(mode.value for mode in allowed)),
                        ("actual_modes", sorted(mode.value for mode in actual)),
                    ),
                )
            )

    enterprise = getattr(config, "enterprise", None)
    if org.require_strict_no_network and not bool(getattr(enterprise, "strict_no_network", False)):
        diagnostics.append(
            _org_policy_diagnostic(
                "policy-pack-strict-no-network-required",
                DiagnosticSeverity.ERROR,
                "organization policy pack requires enterprise.strict_no_network=true",
                "strict_no_network",
                "false",
                "Enable enterprise.strict_no_network and use local mirrored artifacts.",
            )
        )

    sandbox = getattr(enterprise, "solver_sandbox", None)
    if org.max_solver_timeout_ms is not None:
        timeout_ms = getattr(sandbox, "timeout_ms", None)
        if timeout_ms is None:
            diagnostics.append(
                _org_policy_diagnostic(
                    "policy-pack-solver-timeout-missing",
                    DiagnosticSeverity.ERROR,
                    "organization policy pack requires a finite solver sandbox timeout",
                    "solver_sandbox.timeout_ms",
                    "missing",
                    "Set enterprise.solver_sandbox.timeout_ms at or below the organization cap.",
                    extra=(("max_solver_timeout_ms", org.max_solver_timeout_ms),),
                )
            )
        elif timeout_ms > org.max_solver_timeout_ms:
            diagnostics.append(
                _org_policy_diagnostic(
                    "policy-pack-solver-timeout-exceeded",
                    DiagnosticSeverity.ERROR,
                    f"solver timeout {timeout_ms}ms exceeds organization cap {org.max_solver_timeout_ms}ms",
                    "solver_sandbox.timeout_ms",
                    str(timeout_ms),
                    "Lower enterprise.solver_sandbox.timeout_ms or use a different approved policy pack.",
                    extra=(("max_solver_timeout_ms", org.max_solver_timeout_ms),),
                )
            )

    approved = set(org.approved_provider_fixture_sha256)
    if approved:
        for fixture in getattr(enterprise, "internal_provider_fixtures", ()):
            sha256 = getattr(fixture, "sha256", None)
            name = getattr(fixture, "name", "fixture")
            if sha256 is None:
                diagnostics.append(
                    _org_policy_diagnostic(
                        "policy-pack-provider-fixture-unpinned",
                        DiagnosticSeverity.ERROR,
                        f"internal provider fixture '{name}' cannot be approved without a sha256 pin",
                        name,
                        "missing-sha256",
                        "Pin the fixture sha256 and list that digest in approved_provider_fixture_sha256.",
                    )
                )
            elif sha256 not in approved:
                diagnostics.append(
                    _org_policy_diagnostic(
                        "policy-pack-provider-fixture-unapproved",
                        DiagnosticSeverity.ERROR,
                        f"internal provider fixture '{name}' is not approved by the organization policy pack",
                        name,
                        sha256,
                        "Use an approved fixture digest or update the organization policy pack after review.",
                        extra=(("fixture_sha256", sha256),),
                    )
                )

    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                rule_id="policy-pack-verified",
                severity=DiagnosticSeverity.INFO,
                message="organization policy pack constraints are satisfied for this verification run",
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                properties=(
                    ("required_checks", list(org.required_checks)),
                    ("source_paths", list(policy.source_paths)),
                ),
                witness=WitnessTrace(
                    summary="PromptABI evaluated organization policy-pack requirements before applying suppressions or severity gates.",
                    steps=(
                        WitnessStep(action="verify required checks", output=str(len(org.required_checks))),
                        WitnessStep(action="verify supported fragments", output=str(len(org.supported_fragments))),
                        WitnessStep(action="verify solver timeout", output=str(org.max_solver_timeout_ms or "not-required")),
                        WitnessStep(action="verify provider fixture approvals", output=str(len(org.approved_provider_fixture_sha256))),
                    ),
                ),
            )
        )
    return tuple(diagnostics)


def policy_forbids_local_summary(policy: VerificationPolicy) -> bool:
    """Return whether org privacy policy forbids local usage-summary writes."""

    return policy.org_policy.forbid_local_usage_summary


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
            witness_digest=_optional_hex_digest(data.get("witness_digest"), field_name="witness_digest"),
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
    if policy.require_accepted_risk and not (suppression.accepted_risk or "").strip():
        errors.append("accepted_risk is required")
    if policy.require_witness_digest and suppression.witness_digest is None:
        errors.append("witness_digest is required")
    if suppression.expires_on is not None and suppression.expires_on < today:
        errors.append(f"expires_on {suppression.expires_on.isoformat()} is in the past")
    return tuple(errors)


def _stale_witness_suppression_diagnostics(
    diagnostic: Diagnostic,
    suppressions: tuple[Suppression, ...],
) -> tuple[Diagnostic, ...]:
    diagnostics = []
    for suppression in suppressions:
        diagnostics.append(
            Diagnostic(
                rule_id="policy-suppression-invalid",
                severity=DiagnosticSeverity.ERROR,
                message=(
                    f"suppression for rule '{suppression.rule_id}' is invalid: "
                    f"witness_digest {suppression.witness_digest} does not match current witness {diagnostic.witness_digest}"
                ),
                artifact=diagnostic.artifact,
                span=suppression.span or diagnostic.span,
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                suggestions=("Review the changed witness, then update or remove the accepted-risk suppression.",),
                properties=(
                    ("current_witness_digest", diagnostic.witness_digest),
                    ("original_fingerprint", diagnostic.fingerprint),
                    ("suppressed_rule_id", suppression.rule_id),
                    ("suppression", suppression.to_dict()),
                ),
                witness=WitnessTrace(
                    summary="PromptABI rejected a suppression because the suppressed witness changed.",
                    steps=(
                        WitnessStep(action="match diagnostic shape", input=suppression.rule_id, output=diagnostic.fingerprint),
                        WitnessStep(
                            action="compare witness digest",
                            input=suppression.witness_digest or "",
                            output=diagnostic.witness_digest,
                        ),
                    ),
                ),
            )
        )
    return tuple(diagnostics)


def _suppressed_diagnostic(diagnostic: Diagnostic, suppression: Suppression) -> Diagnostic:
    properties: tuple[tuple[str, Any], ...] = (
        ("accepted_risk", suppression.accepted_risk or suppression.justification),
        ("original_fingerprint", diagnostic.fingerprint),
        ("original_rule_id", diagnostic.rule_id),
        ("original_severity", diagnostic.severity.value),
        ("original_witness_digest", diagnostic.witness_digest),
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
                WitnessStep(action="match witness digest", input=suppression.witness_digest or "*", output=diagnostic.witness_digest),
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


def _apply_severity_overrides(
    diagnostics: tuple[Diagnostic, ...],
    overrides: tuple[tuple[str, DiagnosticSeverity], ...],
) -> tuple[Diagnostic, ...]:
    if not overrides:
        return diagnostics
    override_map = dict(overrides)
    return tuple(
        replace(
            diagnostic,
            severity=override_map[diagnostic.rule_id],
            properties=(
                *diagnostic.properties,
                ("original_severity", diagnostic.severity.value),
                ("severity_override", override_map[diagnostic.rule_id].value),
            ),
        )
        if diagnostic.rule_id in override_map and diagnostic.severity is not override_map[diagnostic.rule_id]
        else diagnostic
        for diagnostic in diagnostics
    )


def _merged_severity_overrides(policies: tuple[VerificationPolicy, ...]) -> tuple[tuple[str, DiagnosticSeverity], ...]:
    merged: dict[str, DiagnosticSeverity] = {}
    for policy in policies:
        merged.update(policy.severity_overrides)
    return tuple(sorted(merged.items()))


def _merged_org_policy(policies: tuple[VerificationPolicy, ...]) -> OrgPolicyPack:
    packs = tuple(policy.org_policy for policy in policies if policy.org_policy.active)
    if not packs:
        return OrgPolicyPack()
    return OrgPolicyPack(
        required_checks=tuple(sorted({check for pack in packs for check in pack.required_checks})),
        supported_fragments=_merge_supported_fragments(packs),
        max_solver_timeout_ms=_min_optional(pack.max_solver_timeout_ms for pack in packs),
        require_strict_no_network=any(pack.require_strict_no_network for pack in packs),
        forbid_local_usage_summary=any(pack.forbid_local_usage_summary for pack in packs),
        approved_provider_fixture_sha256=tuple(
            sorted({digest for pack in packs for digest in pack.approved_provider_fixture_sha256})
        ),
    )


def _merge_supported_fragments(packs: tuple[OrgPolicyPack, ...]) -> tuple[tuple[str, tuple[CheckMode, ...]], ...]:
    merged: dict[str, set[CheckMode]] = {}
    for pack in packs:
        for check_name, modes in pack.supported_fragments:
            if check_name in merged:
                merged[check_name] &= set(modes)
            else:
                merged[check_name] = set(modes)
    return tuple((check, tuple(sorted(modes, key=lambda mode: mode.value))) for check, modes in sorted(merged.items()))


def _min_optional(values: Any) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _org_policy_pack_from_mapping(data: dict[str, Any]) -> OrgPolicyPack:
    raw_fragments = data.get("supported_fragments", {})
    if not isinstance(raw_fragments, dict):
        raise PolicyError("policy field 'supported_fragments' must be an object")
    supported_fragments = []
    for check_name, raw_modes in sorted(raw_fragments.items()):
        if not isinstance(check_name, str) or not check_name.strip():
            raise PolicyError("policy supported_fragments keys must be non-empty check names")
        supported_fragments.append((check_name.strip(), _check_mode_list(raw_modes, field_name=f"supported_fragments.{check_name}")))
    return OrgPolicyPack(
        required_checks=tuple(sorted(set(_policy_string_list(data.get("required_checks", []), field_name="required_checks")))),
        supported_fragments=tuple(supported_fragments),
        max_solver_timeout_ms=_optional_positive_int(data.get("max_solver_timeout_ms"), field_name="max_solver_timeout_ms"),
        require_strict_no_network=_optional_bool(data.get("require_strict_no_network"), default=False, field_name="require_strict_no_network"),
        forbid_local_usage_summary=_optional_bool(data.get("forbid_local_usage_summary"), default=False, field_name="forbid_local_usage_summary"),
        approved_provider_fixture_sha256=_approved_fixture_sha256(data.get("approved_provider_fixtures", data.get("approved_provider_fixture_sha256", []))),
    )


def _org_policy_diagnostic(
    rule_id: str,
    severity: DiagnosticSeverity,
    message: str,
    subject: str,
    observed: str,
    suggestion: str,
    *,
    extra: tuple[tuple[str, object], ...] = (),
) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        suggestions=(suggestion,),
        properties=(("policy_subject", subject), ("observed", observed), *extra),
        witness=WitnessTrace(
            summary="PromptABI evaluated an organization-wide policy-pack constraint.",
            steps=(WitnessStep(action="check organization policy", input=subject, output=observed),),
        ),
    )


def _severity_overrides(value: Any) -> tuple[tuple[str, DiagnosticSeverity], ...]:
    if not isinstance(value, dict):
        raise PolicyError("policy field 'severity_overrides' must be an object")
    overrides: dict[str, DiagnosticSeverity] = {}
    for rule_id, severity in value.items():
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise PolicyError("policy severity_overrides keys must be non-empty rule ids")
        overrides[rule_id.strip()] = _required_severity(severity, field_name=f"severity_overrides.{rule_id}")
    return tuple(sorted(overrides.items()))


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


def _optional_hex_digest(value: Any, *, field_name: str) -> str | None:
    digest = _optional_string(value, field_name=field_name)
    if digest is None:
        return None
    if len(digest) not in {16, 64} or any(c not in "0123456789abcdef" for c in digest):
        raise PolicyError(f"suppression field '{field_name}' must be a lowercase 16- or 64-character hex digest")
    return digest


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


def _required_severity(value: Any, *, field_name: str) -> DiagnosticSeverity:
    severity = _optional_severity(value)
    if severity is None:
        raise PolicyError(f"policy field '{field_name}' must be a string")
    return severity


def _check_mode_list(value: Any, *, field_name: str) -> tuple[CheckMode, ...]:
    modes = _policy_string_list(value, field_name=field_name)
    parsed = []
    for mode in modes:
        try:
            parsed.append(CheckMode(mode))
        except ValueError as exc:
            choices = ", ".join(item.value for item in CheckMode)
            raise PolicyError(f"policy field '{field_name}' must contain only check modes: {choices}") from exc
    return tuple(sorted(set(parsed), key=lambda mode: mode.value))


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise PolicyError(f"policy field '{field_name}' must be a positive integer")
    return value


def _approved_fixture_sha256(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PolicyError("policy field 'approved_provider_fixtures' must be a list")
    digests = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            digest = item.strip()
        elif isinstance(item, dict):
            digest = _required_policy_string(item, "sha256", prefix=f"approved_provider_fixtures.{index}")
        else:
            raise PolicyError("policy approved_provider_fixtures entries must be sha256 strings or objects")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PolicyError("policy approved provider fixture sha256 values must be lowercase 64-character hex digests")
        digests.append(digest)
    return tuple(sorted(set(digests)))


def _policy_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise PolicyError(f"policy field '{field_name}' must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _required_policy_string(data: dict[str, Any], field_name: str, *, prefix: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"policy field '{prefix}.{field_name}' must be a non-empty string")
    return value.strip()


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
