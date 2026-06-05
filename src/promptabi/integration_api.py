"""Stable downstream integration API for platforms embedding PromptABI."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .bundles import build_verification_bundle_payload, sign_verification_bundle_payload
from .config import VerificationConfig, load_config
from .diagnostics import Diagnostic, DiagnosticSeverity
from .loaders import ArtifactLoader, LoadedArtifact
from .plugins import PluginRegistry
from .render import SarifRenderOptions, render_github_annotations, render_sarif
from .session import CheckCallable, VerificationResult, VerificationSession
from .usage_analytics import privacy_guarantees


INTEGRATION_API_VERSION = "promptabi.integration.v1"


class IntegrationSurface(StrEnum):
    """Downstream platform surfaces with stable PromptABI payloads."""

    CI_PROVIDER = "ci-provider"
    IDE_EXTENSION = "ide-extension"
    DATASET_PLATFORM = "dataset-platform"
    MODEL_REGISTRY = "model-registry"
    INTERNAL_PLATFORM = "internal-platform"


class IntegrationGate(StrEnum):
    """Normalized gate status for downstream systems."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class IntegrationCapability:
    """One platform capability exposed by the stable integration API."""

    surface: IntegrationSurface
    name: str
    description: str
    payload_schema: str = INTEGRATION_API_VERSION

    def to_dict(self) -> dict[str, str]:
        return {
            "surface": self.surface.value,
            "name": self.name,
            "description": self.description,
            "payload_schema": self.payload_schema,
        }


@dataclass(frozen=True, slots=True)
class IntegrationRequest:
    """Normalized integration request metadata safe to persist in platform logs."""

    config_name: str
    surfaces: tuple[IntegrationSurface, ...]
    fail_on: str = "error"
    config_path: str | None = None
    workspace_root: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_name": self.config_name,
            "surfaces": [surface.value for surface in self.surfaces],
            "fail_on": self.fail_on,
        }
        if self.config_path is not None:
            payload["config_path"] = self.config_path
        if self.workspace_root is not None:
            payload["workspace_root"] = self.workspace_root
        return payload


@dataclass(frozen=True, slots=True)
class IntegrationArtifactSummary:
    """Stable, non-content artifact summary for external platforms."""

    name: str
    kind: str
    location_type: str
    location: str | None = None
    sha256: str | None = None
    version: str | None = None
    revision: str | None = None
    license: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "location_type": self.location_type,
        }
        for key in ("location", "sha256", "version", "revision", "license", "source"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class IntegrationReport:
    """Stable platform-facing report assembled from one PromptABI verification result."""

    request: IntegrationRequest
    gate: IntegrationGate
    ok: bool
    diagnostic_counts: Mapping[str, int]
    artifacts: tuple[IntegrationArtifactSummary, ...]
    capabilities: tuple[IntegrationCapability, ...]
    surfaces: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": INTEGRATION_API_VERSION,
            "request": self.request.to_dict(),
            "gate": self.gate.value,
            "ok": self.ok,
            "diagnostic_counts": dict(sorted(self.diagnostic_counts.items())),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "surfaces": dict(sorted(self.surfaces.items())),
        }


DEFAULT_INTEGRATION_SURFACES: tuple[IntegrationSurface, ...] = tuple(IntegrationSurface)

INTEGRATION_CAPABILITIES: tuple[IntegrationCapability, ...] = (
    IntegrationCapability(
        IntegrationSurface.CI_PROVIDER,
        "verification-gate",
        "Exit-code, SARIF, annotation, diagnostic-count, and check-runtime payloads for CI systems.",
    ),
    IntegrationCapability(
        IntegrationSurface.IDE_EXTENSION,
        "inline-diagnostics",
        "Stable source-span, fingerprint, suggestion, and document grouping payloads for editor extensions.",
    ),
    IntegrationCapability(
        IntegrationSurface.DATASET_PLATFORM,
        "training-eval-contracts",
        "Training and evaluation artifact summaries plus relevant diagnostics for dataset platforms.",
    ),
    IntegrationCapability(
        IntegrationSurface.MODEL_REGISTRY,
        "registry-evidence",
        "Artifact provenance, reproducibility hashes, and optional signed verification-bundle evidence.",
    ),
    IntegrationCapability(
        IntegrationSurface.INTERNAL_PLATFORM,
        "platform-posture",
        "Policy, enterprise, privacy, guarantee-mode, and risk rollup metadata for internal AI platforms.",
    ),
)


def build_integration_report(
    config: str | Path | VerificationConfig,
    surfaces: Sequence[IntegrationSurface | str] = DEFAULT_INTEGRATION_SURFACES,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    fail_on: str = "error",
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "local",
    checks: Mapping[str, CheckCallable] | None = None,
    selected_checks: Sequence[str | CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> IntegrationReport:
    """Build a stable report for CI, editor, dataset, registry, and platform integrations."""

    selected_surfaces = _normalize_surfaces(surfaces)
    config_path = Path(config).expanduser().resolve() if isinstance(config, (str, Path)) else None
    resolved_config = load_config(config_path) if config_path is not None else config
    if artifact_overrides:
        base_dir = Path(override_base_dir) if override_base_dir is not None else Path.cwd()
        resolved_config = resolved_config.with_artifact_overrides(dict(artifact_overrides), base_dir=base_dir)
    root = _workspace_root(workspace_root, config_path)
    session = VerificationSession(resolved_config, checks=checks, loader=loader, plugin_registry=plugin_registry)
    result = session.run(checks=selected_checks)
    loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
    artifact_summaries = tuple(
        _artifact_summary(loaded, workspace_root=root) for loaded in loaded_artifacts
    )
    if not loaded_artifacts:
        artifact_summaries = tuple(
            _configured_artifact_summary(artifact, workspace_root=root)
            for artifact in result.config.artifact_bundle
        )
    request = IntegrationRequest(
        config_name=result.config.name,
        surfaces=selected_surfaces,
        fail_on=fail_on,
        config_path=_relative_path(config_path, root) if config_path is not None else None,
        workspace_root=str(root) if workspace_root is not None else None,
    )
    surface_payloads: dict[str, object] = {}
    for surface in selected_surfaces:
        if surface is IntegrationSurface.CI_PROVIDER:
            surface_payloads[surface.value] = _ci_payload(result, fail_on=fail_on, workspace_root=root)
        elif surface is IntegrationSurface.IDE_EXTENSION:
            surface_payloads[surface.value] = _ide_payload(result, workspace_root=root)
        elif surface is IntegrationSurface.DATASET_PLATFORM:
            surface_payloads[surface.value] = _dataset_payload(result, artifact_summaries)
        elif surface is IntegrationSurface.MODEL_REGISTRY:
            surface_payloads[surface.value] = _model_registry_payload(
                result,
                loaded_artifacts=loaded_artifacts,
                load_diagnostics=load_diagnostics,
                config_path=config_path,
                artifacts=artifact_summaries,
                bundle_key=bundle_key,
                bundle_key_id=bundle_key_id,
            )
        elif surface is IntegrationSurface.INTERNAL_PLATFORM:
            surface_payloads[surface.value] = _internal_platform_payload(result)
    return IntegrationReport(
        request=request,
        gate=_gate(result, fail_on=fail_on),
        ok=result.ok,
        diagnostic_counts=_diagnostic_counts(result),
        artifacts=artifact_summaries,
        capabilities=tuple(
            capability for capability in INTEGRATION_CAPABILITIES if capability.surface in selected_surfaces
        ),
        surfaces=surface_payloads,
    )


def render_integration_report_json(report: IntegrationReport) -> str:
    """Render an integration report as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_integration_report_text(report: IntegrationReport) -> str:
    """Render a compact text summary for human platform owners."""

    lines = [
        "PromptABI integration report:",
        f"protocol: {INTEGRATION_API_VERSION}",
        f"config: {report.request.config_name}",
        f"gate: {report.gate.value}",
        "diagnostics: "
        + ", ".join(f"{key}={value}" for key, value in sorted(report.diagnostic_counts.items())),
        "surfaces: " + ", ".join(report.surfaces),
        f"artifacts: {len(report.artifacts)}",
    ]
    return "\n".join(lines) + "\n"


def _normalize_surfaces(surfaces: Sequence[IntegrationSurface | str]) -> tuple[IntegrationSurface, ...]:
    normalized = tuple(dict.fromkeys(IntegrationSurface(surface) for surface in surfaces))
    if not normalized:
        raise ValueError("at least one integration surface must be requested")
    return normalized


def _workspace_root(workspace_root: str | Path | None, config_path: Path | None) -> Path:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve()
    if config_path is not None:
        return config_path.parent
    return Path.cwd().resolve()


def _artifact_summary(loaded: LoadedArtifact, *, workspace_root: Path) -> IntegrationArtifactSummary:
    return _configured_artifact_summary(loaded.artifact, workspace_root=workspace_root)


def _configured_artifact_summary(artifact: Any, *, workspace_root: Path) -> IntegrationArtifactSummary:
    location = artifact.location
    raw_location = location.path or location.uri
    location_type = "path" if location.path is not None else "uri" if location.uri is not None else "inline"
    provenance = getattr(artifact, "provenance", None)
    return IntegrationArtifactSummary(
        name=artifact.name,
        kind=artifact.kind.value,
        location_type=location_type,
        location=_relative_path(Path(raw_location), workspace_root) if location.path is not None else raw_location,
        sha256=getattr(provenance, "sha256", None),
        version=getattr(provenance, "version", None),
        revision=getattr(provenance, "revision", None),
        license=getattr(provenance, "license", None),
        source=getattr(provenance, "source", None),
    )


def _ci_payload(result: VerificationResult, *, fail_on: str, workspace_root: Path) -> dict[str, object]:
    sarif_options = SarifRenderOptions(checkout_uri_base=workspace_root, include_invocation=False)
    annotations = tuple(
        line
        for line in render_github_annotations(result, checkout_uri_base=workspace_root).splitlines()
        if line
    )
    return {
        "exit_code": _exit_code(result, fail_on=fail_on),
        "fail_on": fail_on,
        "sarif": json.loads(render_sarif(result, options=sarif_options)),
        "github_annotations": list(annotations),
        "diagnostic_counts": _diagnostic_counts(result),
        "check_runtimes": [runtime.to_dict() for runtime in result.check_runtimes],
    }


def _ide_payload(result: VerificationResult, *, workspace_root: Path) -> dict[str, object]:
    documents: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for diagnostic in result.diagnostics:
        document = _diagnostic_document(diagnostic, workspace_root=workspace_root)
        documents.setdefault(document, []).append(_ide_diagnostic(diagnostic, workspace_root=workspace_root))
    return {
        "protocol": "promptabi.inlineDiagnostics.v1",
        "documents": [
            {"uri": uri, "diagnostics": diagnostics}
            for uri, diagnostics in sorted(documents.items(), key=lambda item: item[0])
        ],
        "diagnostic_count": len(result.diagnostics),
    }


def _dataset_payload(
    result: VerificationResult,
    artifacts: tuple[IntegrationArtifactSummary, ...],
) -> dict[str, object]:
    dataset_kinds = {"training-manifest", "evaluation-harness"}
    relevant_prefixes = ("training-", "evaluation-", "synthetic-generator-")
    relevant_diagnostics = tuple(
        diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith(relevant_prefixes)
    )
    return {
        "artifacts": [
            artifact.to_dict()
            for artifact in artifacts
            if artifact.kind in dataset_kinds
        ],
        "diagnostics": [_diagnostic_summary(diagnostic) for diagnostic in relevant_diagnostics],
        "diagnostic_count": len(relevant_diagnostics),
        "privacy": [
            "Dataset payloads expose artifact metadata, fingerprints, and structural diagnostics, not dataset rows.",
            "Use witness privacy modes for any separately rendered verification result that may include examples.",
        ],
    }


def _model_registry_payload(
    result: VerificationResult,
    *,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    load_diagnostics: tuple[Diagnostic, ...],
    config_path: Path | None,
    artifacts: tuple[IntegrationArtifactSummary, ...],
    bundle_key: str | bytes | None,
    bundle_key_id: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "diagnostic_fingerprints": [diagnostic.fingerprint for diagnostic in result.diagnostics],
        "load_diagnostic_fingerprints": [diagnostic.fingerprint for diagnostic in load_diagnostics],
    }
    if config_path is None:
        payload["signed_bundle"] = {
            "available": False,
            "reason": "config_path is required to bind registry evidence to a persisted config file",
        }
        return payload
    bundle_payload = build_verification_bundle_payload(
        result.config,
        config_path=config_path,
        result=result,
        loaded_artifacts=loaded_artifacts,
        excerpt_bytes=0,
    )
    payload["reproducibility_hash"] = bundle_payload["reproducibility_hash"]
    if bundle_key is not None:
        bundle = sign_verification_bundle_payload(bundle_payload, key=bundle_key, key_id=bundle_key_id)
        payload["signed_bundle"] = {
            "available": True,
            "algorithm": bundle.algorithm,
            "bundle_hash": bundle.bundle_hash,
            "signature": bundle.signature,
            "signing_key_id": bundle.signing_key_id,
        }
    else:
        payload["signed_bundle"] = {
            "available": False,
            "reason": "bundle_key was not provided",
        }
    return payload


def _internal_platform_payload(result: VerificationResult) -> dict[str, object]:
    modes = Counter(mode.value for diagnostic in result.diagnostics for mode in diagnostic.check_modes)
    rules = Counter(diagnostic.rule_id for diagnostic in result.diagnostics)
    return {
        "policy_active": result.config.policy.active,
        "enterprise_active": result.config.enterprise.active,
        "checks": list(result.config.checks),
        "guarantee_modes": dict(sorted(modes.items())),
        "rule_counts": dict(sorted(rules.items())),
        "privacy_guarantees": list(privacy_guarantees()),
    }


def _diagnostic_counts(result: VerificationResult) -> dict[str, int]:
    severities = [diagnostic.severity for diagnostic in result.diagnostics]
    return {
        "error": severities.count(DiagnosticSeverity.ERROR),
        "warning": severities.count(DiagnosticSeverity.WARNING),
        "info": severities.count(DiagnosticSeverity.INFO),
        "total": len(severities),
    }


def _gate(result: VerificationResult, *, fail_on: str) -> IntegrationGate:
    if _exit_code(result, fail_on=fail_on):
        return IntegrationGate.FAIL
    if any(diagnostic.severity is DiagnosticSeverity.WARNING for diagnostic in result.diagnostics):
        return IntegrationGate.WARN
    return IntegrationGate.PASS


def _exit_code(result: VerificationResult, *, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    severities = {diagnostic.severity.value for diagnostic in result.diagnostics}
    if fail_on == "any":
        return 1 if severities else 0
    if fail_on == "warning":
        return 1 if severities.intersection({"error", "warning"}) else 0
    if fail_on != "error":
        raise ValueError("fail_on must be one of: error, warning, any, never")
    return 0 if result.ok else 1


def _diagnostic_document(diagnostic: Diagnostic, *, workspace_root: Path) -> str:
    if diagnostic.span is not None:
        return _document_uri(diagnostic.span.path, workspace_root=workspace_root)
    if diagnostic.artifact is not None and diagnostic.artifact.location_uri is not None:
        return _document_uri(diagnostic.artifact.location_uri, workspace_root=workspace_root)
    return "promptabi://config"


def _ide_diagnostic(diagnostic: Diagnostic, *, workspace_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": diagnostic.rule_id,
        "severity": diagnostic.severity.value,
        "message": diagnostic.message,
        "fingerprint": diagnostic.fingerprint,
        "check_modes": [mode.value for mode in diagnostic.check_modes],
        "suggestions": list(diagnostic.suggestions),
    }
    if diagnostic.span is not None:
        payload["span"] = {
            **diagnostic.span.to_dict(),
            "path": _relative_path(Path(diagnostic.span.path), workspace_root),
        }
    if diagnostic.artifact is not None:
        payload["artifact"] = diagnostic.artifact.to_dict()
    return payload


def _diagnostic_summary(diagnostic: Diagnostic) -> dict[str, object]:
    return {
        "rule_id": diagnostic.rule_id,
        "severity": diagnostic.severity.value,
        "message": diagnostic.message,
        "fingerprint": diagnostic.fingerprint,
        "check_modes": [mode.value for mode in diagnostic.check_modes],
    }


def _document_uri(value: str, *, workspace_root: Path) -> str:
    if "://" in value:
        return value
    return f"file://{_relative_path(Path(value), workspace_root)}"


def _relative_path(path: Path | None, workspace_root: Path) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(workspace_root).as_posix()
    except ValueError:
        return resolved.as_posix()
