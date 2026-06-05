"""Enterprise readiness settings for offline PromptABI deployments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .provider_fixture_packs import ProviderFixturePackError, reject_secret_like_values


class EnterpriseConfigError(ValueError):
    """Raised when enterprise readiness settings are malformed."""


@dataclass(frozen=True, slots=True)
class EnterprisePath:
    """A named local enterprise resource such as a mirror, index, or fixture."""

    name: str
    path: str
    sha256: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("enterprise resource name must be non-empty")
        if not self.path:
            raise ValueError("enterprise resource path must be non-empty")
        if self.sha256 is not None and (len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("enterprise resource sha256 must be a lowercase 64-character hex digest")
        if self.source is not None and not self.source:
            raise ValueError("enterprise resource source must be non-empty")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "path": self.path}
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.source is not None:
            data["source"] = self.source
        return data


@dataclass(frozen=True, slots=True)
class PrivateArtifactIndex:
    """A private artifact index whose metadata is checked without network access."""

    name: str
    path: str
    trusted_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("private artifact index name must be non-empty")
        if not self.path:
            raise ValueError("private artifact index path must be non-empty")
        if any(not source for source in self.trusted_sources):
            raise ValueError("private artifact index trusted sources must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": self.path, "trusted_sources": list(self.trusted_sources)}


@dataclass(frozen=True, slots=True)
class EnterpriseApproval:
    """One access-control approval for an enterprise-local resource."""

    name: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("enterprise access-control approval name must be non-empty")
        if self.sha256 is not None and (len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("enterprise access-control approval sha256 must be a lowercase 64-character hex digest")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name}
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        return data


@dataclass(frozen=True, slots=True)
class EnterpriseAccessControl:
    """Local access-control policy for private indexes, prompt packs, policy packs, and audit bundles."""

    principals: tuple[str, ...] = ()
    approved_private_artifact_indexes: tuple[EnterpriseApproval, ...] = ()
    approved_prompt_packs: tuple[EnterpriseApproval, ...] = ()
    approved_policy_packs: tuple[EnterpriseApproval, ...] = ()
    audit_bundle_retention_days: int | None = None
    audit_bundle_min_replicas: int | None = None

    @property
    def configured(self) -> bool:
        return bool(
            self.principals
            or self.approved_private_artifact_indexes
            or self.approved_prompt_packs
            or self.approved_policy_packs
            or self.audit_bundle_retention_days is not None
            or self.audit_bundle_min_replicas is not None
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.principals:
            data["principals"] = list(self.principals)
        if self.approved_private_artifact_indexes:
            data["approved_private_artifact_indexes"] = [item.to_dict() for item in self.approved_private_artifact_indexes]
        if self.approved_prompt_packs:
            data["approved_prompt_packs"] = [item.to_dict() for item in self.approved_prompt_packs]
        if self.approved_policy_packs:
            data["approved_policy_packs"] = [item.to_dict() for item in self.approved_policy_packs]
        if self.audit_bundle_retention_days is not None:
            data["audit_bundle_retention_days"] = self.audit_bundle_retention_days
        if self.audit_bundle_min_replicas is not None:
            data["audit_bundle_min_replicas"] = self.audit_bundle_min_replicas
        return data


@dataclass(frozen=True, slots=True)
class SolverSandbox:
    """Declarative solver sandbox policy for CI runners and enterprise wrappers."""

    enabled: bool = False
    timeout_ms: int | None = None
    max_memory_mb: int | None = None
    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("solver sandbox timeout_ms must be positive")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("solver sandbox max_memory_mb must be positive")

    @property
    def configured(self) -> bool:
        return self.enabled or self.timeout_ms is not None or self.max_memory_mb is not None or self.allow_network

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"enabled": self.enabled, "allow_network": self.allow_network}
        if self.timeout_ms is not None:
            data["timeout_ms"] = self.timeout_ms
        if self.max_memory_mb is not None:
            data["max_memory_mb"] = self.max_memory_mb
        return data


@dataclass(frozen=True, slots=True)
class EnterpriseSettings:
    """Enterprise controls for strict local verification and internal fixtures."""

    strict_no_network: bool = False
    offline_mirrors: tuple[EnterprisePath, ...] = ()
    private_artifact_indexes: tuple[PrivateArtifactIndex, ...] = ()
    internal_prompt_packs: tuple[EnterprisePath, ...] = ()
    internal_provider_fixtures: tuple[EnterprisePath, ...] = ()
    policy_packs: tuple[EnterprisePath, ...] = ()
    access_control: EnterpriseAccessControl = field(default_factory=EnterpriseAccessControl)
    solver_sandbox: SolverSandbox = field(default_factory=SolverSandbox)

    @property
    def active(self) -> bool:
        return bool(
            self.strict_no_network
            or self.offline_mirrors
            or self.private_artifact_indexes
            or self.internal_prompt_packs
            or self.internal_provider_fixtures
            or self.policy_packs
            or self.access_control.configured
            or self.solver_sandbox.configured
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "strict_no_network": self.strict_no_network,
            "offline_mirrors": [item.to_dict() for item in self.offline_mirrors],
            "private_artifact_indexes": [item.to_dict() for item in self.private_artifact_indexes],
            "internal_prompt_packs": [item.to_dict() for item in self.internal_prompt_packs],
            "internal_provider_fixtures": [item.to_dict() for item in self.internal_provider_fixtures],
            "policy_packs": [item.to_dict() for item in self.policy_packs],
            "access_control": self.access_control.to_dict(),
            "solver_sandbox": self.solver_sandbox.to_dict(),
        }


def empty_enterprise_settings() -> EnterpriseSettings:
    return EnterpriseSettings()


def enterprise_from_config_mapping(data: dict[str, Any], *, base_dir: Path) -> EnterpriseSettings:
    """Parse enterprise settings and resolve local paths against the config directory."""

    raw = data.get("enterprise")
    if raw is None:
        return empty_enterprise_settings()
    if not isinstance(raw, dict):
        raise EnterpriseConfigError("config field 'enterprise' must be an object")
    try:
        return EnterpriseSettings(
            strict_no_network=_optional_bool(raw.get("strict_no_network"), default=False, field_name="strict_no_network"),
            offline_mirrors=_enterprise_paths(raw.get("offline_mirrors", []), base_dir=base_dir, field_name="offline_mirrors"),
            private_artifact_indexes=_private_indexes(raw.get("private_artifact_indexes", []), base_dir=base_dir),
            internal_prompt_packs=_enterprise_paths(raw.get("internal_prompt_packs", []), base_dir=base_dir, field_name="internal_prompt_packs"),
            internal_provider_fixtures=_enterprise_paths(
                raw.get("internal_provider_fixtures", []),
                base_dir=base_dir,
                field_name="internal_provider_fixtures",
            ),
            policy_packs=_enterprise_paths(raw.get("policy_packs", []), base_dir=base_dir, field_name="policy_packs"),
            access_control=_access_control(raw.get("access_control", {})),
            solver_sandbox=_solver_sandbox(raw.get("solver_sandbox", {})),
        )
    except ValueError as exc:
        raise EnterpriseConfigError(str(exc)) from exc


def enterprise_readiness_diagnostics(settings: EnterpriseSettings, *, artifact_locations: tuple[str, ...] = ()) -> tuple[Diagnostic, ...]:
    """Return static enterprise readiness diagnostics without attempting network IO."""

    if not settings.active:
        return ()
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_resource_diagnostics("offline mirror", settings.offline_mirrors))
    diagnostics.extend(_index_diagnostics(settings.private_artifact_indexes))
    diagnostics.extend(_resource_diagnostics("internal prompt pack", settings.internal_prompt_packs))
    diagnostics.extend(_resource_diagnostics("policy pack", settings.policy_packs))
    diagnostics.extend(_fixture_diagnostics(settings.internal_provider_fixtures))
    diagnostics.extend(_access_control_diagnostics(settings))
    diagnostics.extend(_no_network_diagnostics(settings, artifact_locations))
    diagnostics.extend(_solver_sandbox_diagnostics(settings.solver_sandbox, strict_no_network=settings.strict_no_network))
    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                rule_id="enterprise-readiness-verified",
                severity=DiagnosticSeverity.INFO,
                message="enterprise offline mirrors, private indexes, policy packs, fixtures, and solver sandbox declarations are locally consistent",
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                witness=WitnessTrace(
                    summary="PromptABI checked enterprise declarations using only local files and config metadata.",
                    steps=(
                        WitnessStep(action="validate strict no-network posture", output=str(settings.strict_no_network).lower()),
                        WitnessStep(action="verify offline mirrors", output=str(len(settings.offline_mirrors))),
                        WitnessStep(action="verify private indexes", output=str(len(settings.private_artifact_indexes))),
                        WitnessStep(action="verify internal prompt packs", output=str(len(settings.internal_prompt_packs))),
                        WitnessStep(action="verify internal provider fixtures", output=str(len(settings.internal_provider_fixtures))),
                        WitnessStep(action="verify policy packs", output=str(len(settings.policy_packs))),
                        WitnessStep(action="verify access-control approvals", output=str(_access_control_approval_count(settings.access_control))),
                        WitnessStep(action="verify audit-bundle retention", output=_retention_summary(settings.access_control)),
                        WitnessStep(action="classify solver sandbox", output="declared-local"),
                    ),
                ),
            )
        )
    return tuple(diagnostics)


def render_enterprise_readiness_text(diagnostics: tuple[Diagnostic, ...]) -> str:
    """Render a compact standalone enterprise readiness report."""

    lines = ["PromptABI enterprise readiness"]
    for diagnostic in diagnostics:
        lines.append(f"{diagnostic.severity.value.upper()} {diagnostic.rule_id}: {diagnostic.message}")
        if diagnostic.suggestions:
            lines.append(f"  suggestion: {diagnostic.suggestions[0]}")
    return "\n".join(lines) + "\n"


def render_enterprise_readiness_json(diagnostics: tuple[Diagnostic, ...]) -> str:
    return json.dumps({"diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics]}, indent=2, sort_keys=True) + "\n"


def _enterprise_paths(raw: Any, *, base_dir: Path, field_name: str) -> tuple[EnterprisePath, ...]:
    if not isinstance(raw, list):
        raise EnterpriseConfigError(f"enterprise field '{field_name}' must be a list")
    return tuple(_enterprise_path(item, base_dir=base_dir, field_name=field_name) for item in raw)


def _enterprise_path(raw: Any, *, base_dir: Path, field_name: str) -> EnterprisePath:
    if isinstance(raw, str):
        path = raw
        name = Path(raw).name
        sha256 = None
        source = None
    elif isinstance(raw, dict):
        name = _required_string(raw, "name", prefix=field_name)
        path = _required_string(raw, "path", prefix=field_name)
        sha256 = _optional_string(raw.get("sha256"), field_name="sha256", prefix=field_name)
        source = _optional_string(raw.get("source"), field_name="source", prefix=field_name)
    else:
        raise EnterpriseConfigError(f"enterprise field '{field_name}' entries must be objects or strings")
    return EnterprisePath(name=name, path=_resolve_local_path(path, base_dir=base_dir), sha256=sha256, source=source)


def _private_indexes(raw: Any, *, base_dir: Path) -> tuple[PrivateArtifactIndex, ...]:
    if not isinstance(raw, list):
        raise EnterpriseConfigError("enterprise field 'private_artifact_indexes' must be a list")
    indexes = []
    for item in raw:
        if not isinstance(item, dict):
            raise EnterpriseConfigError("enterprise private_artifact_indexes entries must be objects")
        indexes.append(
            PrivateArtifactIndex(
                name=_required_string(item, "name", prefix="private_artifact_indexes"),
                path=_resolve_local_path(_required_string(item, "path", prefix="private_artifact_indexes"), base_dir=base_dir),
                trusted_sources=tuple(_string_list(item.get("trusted_sources", []), field_name="trusted_sources")),
            )
        )
    return tuple(indexes)


def _access_control(raw: Any) -> EnterpriseAccessControl:
    if raw is None:
        return EnterpriseAccessControl()
    if not isinstance(raw, dict):
        raise EnterpriseConfigError("enterprise field 'access_control' must be an object")
    return EnterpriseAccessControl(
        principals=_string_tuple(raw.get("principals", []), field_name="access_control.principals"),
        approved_private_artifact_indexes=_approval_list(
            raw.get("approved_private_artifact_indexes", []),
            field_name="access_control.approved_private_artifact_indexes",
        ),
        approved_prompt_packs=_approval_list(
            raw.get("approved_prompt_packs", []),
            field_name="access_control.approved_prompt_packs",
        ),
        approved_policy_packs=_approval_list(
            raw.get("approved_policy_packs", []),
            field_name="access_control.approved_policy_packs",
        ),
        audit_bundle_retention_days=_optional_positive_int(
            raw.get("audit_bundle_retention_days"),
            field_name="access_control.audit_bundle_retention_days",
        ),
        audit_bundle_min_replicas=_optional_positive_int(
            raw.get("audit_bundle_min_replicas"),
            field_name="access_control.audit_bundle_min_replicas",
        ),
    )


def _solver_sandbox(raw: Any) -> SolverSandbox:
    if raw is None:
        return SolverSandbox()
    if not isinstance(raw, dict):
        raise EnterpriseConfigError("enterprise field 'solver_sandbox' must be an object")
    return SolverSandbox(
        enabled=_optional_bool(raw.get("enabled"), default=False, field_name="solver_sandbox.enabled"),
        timeout_ms=_optional_positive_int(raw.get("timeout_ms"), field_name="solver_sandbox.timeout_ms"),
        max_memory_mb=_optional_positive_int(raw.get("max_memory_mb"), field_name="solver_sandbox.max_memory_mb"),
        allow_network=_optional_bool(raw.get("allow_network"), default=False, field_name="solver_sandbox.allow_network"),
    )


def _resource_diagnostics(label: str, resources: tuple[EnterprisePath, ...]) -> tuple[Diagnostic, ...]:
    diagnostics = []
    for resource in resources:
        path = Path(resource.path)
        if not path.exists():
            diagnostics.append(_enterprise_diagnostic("enterprise-local-resource-missing", DiagnosticSeverity.ERROR, f"{label} '{resource.name}' is missing at {path}", resource.name, str(path), "Create the local mirror/index/pack or update the enterprise path."))
            continue
        if resource.sha256 is not None and path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != resource.sha256:
                diagnostics.append(_enterprise_diagnostic("enterprise-local-resource-hash-mismatch", DiagnosticSeverity.ERROR, f"{label} '{resource.name}' sha256 does not match its local bytes", resource.name, str(path), "Review the local bytes and update the expected digest only after approval.", expected=resource.sha256, actual=actual))
        elif resource.sha256 is not None and path.is_dir():
            diagnostics.append(_enterprise_diagnostic("enterprise-local-resource-hash-abstained", DiagnosticSeverity.WARNING, f"{label} '{resource.name}' declares a file sha256 but points at a directory", resource.name, str(path), "Pin a manifest file digest for directory mirrors.", check_modes=(CheckMode.ABSTAINING, CheckMode.COMPLETE)))
    return tuple(diagnostics)


def _index_diagnostics(indexes: tuple[PrivateArtifactIndex, ...]) -> tuple[Diagnostic, ...]:
    diagnostics = []
    for index in indexes:
        path = Path(index.path)
        if not path.exists():
            diagnostics.append(_enterprise_diagnostic("enterprise-local-resource-missing", DiagnosticSeverity.ERROR, f"private artifact index '{index.name}' is missing at {path}", index.name, str(path), "Create the private index file or update the enterprise path."))
            continue
        if not index.trusted_sources:
            diagnostics.append(_enterprise_diagnostic("enterprise-private-index-untrusted", DiagnosticSeverity.WARNING, f"private artifact index '{index.name}' has no trusted_sources allowlist", index.name, str(path), "Declare the internal mirror/source prefixes approved for this index."))
    return tuple(diagnostics)


def _fixture_diagnostics(fixtures: tuple[EnterprisePath, ...]) -> tuple[Diagnostic, ...]:
    diagnostics = list(_resource_diagnostics("internal provider fixture", fixtures))
    for fixture in fixtures:
        path = Path(fixture.path)
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            reject_secret_like_values(fixture.name, payload)
        except (json.JSONDecodeError, ProviderFixturePackError) as exc:
            diagnostics.append(_enterprise_diagnostic("enterprise-internal-fixture-unsafe", DiagnosticSeverity.ERROR, f"internal provider fixture '{fixture.name}' is not safe to replay offline: {exc}", fixture.name, str(path), "Store only redacted, valid JSON provider fixtures in enterprise packs."))
    return tuple(diagnostics)


def _access_control_diagnostics(settings: EnterpriseSettings) -> tuple[Diagnostic, ...]:
    access = settings.access_control
    if not access.configured:
        return ()
    diagnostics: list[Diagnostic] = []
    if not access.principals:
        diagnostics.append(
            _enterprise_diagnostic(
                "enterprise-access-control-incomplete",
                DiagnosticSeverity.WARNING,
                "enterprise access-control policy has no principals for audit ownership",
                "access_control",
                "principals",
                "Declare at least one team, service account, or release role responsible for approved enterprise resources.",
            )
        )
    diagnostics.extend(
        _approval_diagnostics(
            "private artifact index",
            tuple((index.name, index.path) for index in settings.private_artifact_indexes),
            access.approved_private_artifact_indexes,
        )
    )
    diagnostics.extend(
        _approval_diagnostics(
            "internal prompt pack",
            tuple((pack.name, pack.path) for pack in settings.internal_prompt_packs),
            access.approved_prompt_packs,
        )
    )
    diagnostics.extend(
        _approval_diagnostics(
            "policy pack",
            tuple((pack.name, pack.path) for pack in settings.policy_packs),
            access.approved_policy_packs,
        )
    )
    retention_days = access.audit_bundle_retention_days
    min_replicas = access.audit_bundle_min_replicas
    weak_reasons = []
    if retention_days is not None and retention_days < 90:
        weak_reasons.append(f"retention is {retention_days} days")
    if min_replicas is not None and min_replicas < 2:
        weak_reasons.append(f"replicas is {min_replicas}")
    if weak_reasons:
        diagnostics.append(
            _enterprise_diagnostic(
                "enterprise-access-control-retention-weak",
                DiagnosticSeverity.WARNING,
                f"audit-bundle retention policy is weak: {'; '.join(weak_reasons)}",
                "access_control",
                _retention_summary(access),
                "Keep signed audit bundles for at least 90 days and at least two local replicas for release evidence.",
            )
        )
    return tuple(diagnostics)


def _approval_diagnostics(
    label: str,
    resources: tuple[tuple[str, str], ...],
    approvals: tuple[EnterpriseApproval, ...],
) -> tuple[Diagnostic, ...]:
    if not approvals:
        return ()
    by_name = {approval.name: approval for approval in approvals}
    diagnostics: list[Diagnostic] = []
    for name, path_value in resources:
        approval = by_name.get(name)
        if approval is None:
            diagnostics.append(
                _enterprise_diagnostic(
                    "enterprise-access-control-unapproved",
                    DiagnosticSeverity.ERROR,
                    f"{label} '{name}' is not approved by enterprise access-control policy",
                    name,
                    path_value,
                    f"Add '{name}' to the approved {label} list only after internal review.",
                )
            )
            continue
        if approval.sha256 is not None:
            path = Path(path_value)
            if not path.is_file():
                diagnostics.append(
                    _enterprise_diagnostic(
                        "enterprise-access-control-hash-abstained",
                        DiagnosticSeverity.WARNING,
                        f"{label} '{name}' has an approval digest but is not a local file",
                        name,
                        path_value,
                        "Approve a manifest file digest for directories or generated resources.",
                        check_modes=(CheckMode.ABSTAINING, CheckMode.COMPLETE),
                    )
                )
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != approval.sha256:
                diagnostics.append(
                    _enterprise_diagnostic(
                        "enterprise-access-control-hash-mismatch",
                        DiagnosticSeverity.ERROR,
                        f"{label} '{name}' bytes do not match the access-control approval digest",
                        name,
                        path_value,
                        "Re-review the resource and update the approval digest only after authorization.",
                        expected=approval.sha256,
                        actual=actual,
                    )
                )
    return tuple(diagnostics)


def _no_network_diagnostics(settings: EnterpriseSettings, artifact_locations: tuple[str, ...]) -> tuple[Diagnostic, ...]:
    if not settings.strict_no_network:
        return ()
    diagnostics = []
    for location in artifact_locations:
        if _is_remote_reference(location):
            diagnostics.append(_enterprise_diagnostic("enterprise-no-network-violation", DiagnosticSeverity.ERROR, f"strict no-network mode forbids remote artifact location {location}", "artifact", location, "Use a local offline mirror path and pinned sha256 provenance."))
    for resource in (*settings.offline_mirrors, *settings.internal_prompt_packs, *settings.internal_provider_fixtures, *settings.policy_packs):
        if resource.source and _is_remote_reference(resource.source):
            diagnostics.append(_enterprise_diagnostic("enterprise-no-network-violation", DiagnosticSeverity.WARNING, f"strict no-network mode records remote source metadata for '{resource.name}'", resource.name, resource.source, "Keep source metadata as an audit reference only if all runtime locations are local mirrors."))
    return tuple(diagnostics)


def _solver_sandbox_diagnostics(sandbox: SolverSandbox, *, strict_no_network: bool) -> tuple[Diagnostic, ...]:
    if not sandbox.configured:
        return ()
    diagnostics = []
    if sandbox.allow_network:
        diagnostics.append(_enterprise_diagnostic("enterprise-solver-sandbox-unsafe", DiagnosticSeverity.ERROR if strict_no_network else DiagnosticSeverity.WARNING, "solver sandbox declaration allows network access", "solver_sandbox", "allow_network=true", "Set enterprise.solver_sandbox.allow_network to false for offline CI runners."))
    if sandbox.enabled and (sandbox.timeout_ms is None or sandbox.max_memory_mb is None):
        diagnostics.append(_enterprise_diagnostic("enterprise-solver-sandbox-incomplete", DiagnosticSeverity.WARNING, "solver sandbox is enabled without both timeout_ms and max_memory_mb", "solver_sandbox", "resource-limits", "Declare finite solver timeout and memory ceilings for reproducible enterprise runs."))
    return tuple(diagnostics)


def _enterprise_diagnostic(
    rule_id: str,
    severity: DiagnosticSeverity,
    message: str,
    name: str,
    subject: str,
    suggestion: str,
    *,
    expected: str | None = None,
    actual: str | None = None,
    check_modes: tuple[CheckMode, ...] = (CheckMode.SOUND, CheckMode.COMPLETE),
) -> Diagnostic:
    properties: list[tuple[str, object]] = [("enterprise_resource", name), ("subject", subject)]
    if expected is not None:
        properties.append(("expected_sha256", expected))
    if actual is not None:
        properties.append(("actual_sha256", actual))
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        check_modes=check_modes,
        suggestions=(suggestion,),
        properties=tuple(properties),
        witness=WitnessTrace(
            summary="PromptABI evaluated an enterprise readiness declaration without contacting a network.",
            steps=(WitnessStep(action="inspect enterprise declaration", input=name, output=subject),),
        ),
    )


def _resolve_local_path(path: str, *, base_dir: Path) -> str:
    parsed = urlparse(path)
    if parsed.scheme and parsed.scheme != "file":
        raise EnterpriseConfigError(f"enterprise paths must be local paths or file:// URIs, got {path}")
    raw = Path(parsed.path if parsed.scheme == "file" else path).expanduser()
    if not raw.is_absolute():
        raw = base_dir / raw
    return str(raw)


def _is_remote_reference(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.scheme not in {"", "file"})


def _required_string(data: dict[str, Any], field_name: str, *, prefix: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EnterpriseConfigError(f"enterprise {prefix}.{field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, field_name: str, prefix: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EnterpriseConfigError(f"enterprise {prefix}.{field_name} must be a non-empty string")
    return value.strip()


def _optional_bool(value: Any, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise EnterpriseConfigError(f"enterprise field '{field_name}' must be a boolean")
    return value


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise EnterpriseConfigError(f"enterprise field '{field_name}' must be a positive integer")
    return value


def _string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise EnterpriseConfigError(f"enterprise field '{field_name}' must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(_string_list(value, field_name=field_name))))


def _approval_list(value: Any, *, field_name: str) -> tuple[EnterpriseApproval, ...]:
    if not isinstance(value, list):
        raise EnterpriseConfigError(f"enterprise field '{field_name}' must be a list")
    approvals: list[EnterpriseApproval] = []
    for item in value:
        if isinstance(item, str):
            approvals.append(EnterpriseApproval(name=item.strip()))
        elif isinstance(item, dict):
            approvals.append(
                EnterpriseApproval(
                    name=_required_string(item, "name", prefix=field_name),
                    sha256=_optional_string(item.get("sha256"), field_name="sha256", prefix=field_name),
                )
            )
        else:
            raise EnterpriseConfigError(f"enterprise field '{field_name}' entries must be objects or strings")
    return tuple(sorted(approvals, key=lambda item: item.name))


def _access_control_approval_count(access: EnterpriseAccessControl) -> int:
    return (
        len(access.approved_private_artifact_indexes)
        + len(access.approved_prompt_packs)
        + len(access.approved_policy_packs)
    )


def _retention_summary(access: EnterpriseAccessControl) -> str:
    days = access.audit_bundle_retention_days
    replicas = access.audit_bundle_min_replicas
    return f"{days if days is not None else 'not-declared'} days/{replicas if replicas is not None else 'not-declared'} replicas"
