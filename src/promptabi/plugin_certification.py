"""Certification checks for third-party PromptABI plugins."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .artifacts import ArtifactKind, ArtifactLocation, ArtifactProvenance, BaseArtifact
from .config import VerificationConfig
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .loaders import LoadedArtifact
from .plugins import PluginCapabilityKind, PluginRegistry
from .session import CheckContext, VerificationResult
from .witness_privacy import WitnessPrivacyMode, apply_witness_privacy


PLUGIN_CERTIFICATION_VERSION = "1.0"
PLUGIN_CERTIFICATION_SECRET = "promptabi-certification-secret-do-not-leak"
_SECRET_PATTERNS = (
    re.compile(re.escape(PLUGIN_CERTIFICATION_SECRET)),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
)


class PluginCertificationStatus(StrEnum):
    """Outcome for one plugin certification case."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class PluginCertificationCase:
    """One concrete certification observation for a plugin surface."""

    name: str
    status: PluginCertificationStatus
    surface: str
    message: str
    plugin: str | None = None
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("certification case name must be non-empty")
        if not self.surface:
            raise ValueError("certification case surface must be non-empty")
        if not self.message:
            raise ValueError("certification case message must be non-empty")
        object.__setattr__(self, "status", PluginCertificationStatus(self.status))
        object.__setattr__(self, "details", tuple(sorted(self.details, key=lambda item: item[0])))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "surface": self.surface,
            "message": self.message,
        }
        if self.plugin is not None:
            data["plugin"] = self.plugin
        if self.details:
            data["details"] = dict(self.details)
        return data


@dataclass(frozen=True, slots=True)
class PluginCertificationReport:
    """Certification report proving plugin extension points obey PromptABI contracts."""

    version: str
    cases: tuple[PluginCertificationCase, ...]

    @property
    def ok(self) -> bool:
        return not any(case.status is PluginCertificationStatus.FAIL for case in self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.status is PluginCertificationStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for case in self.cases if case.status is PluginCertificationStatus.FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for case in self.cases if case.status is PluginCertificationStatus.WARN)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ok": self.ok,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "warned": self.warned,
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def certify_plugin_registry(registry: PluginRegistry) -> PluginCertificationReport:
    """Run deterministic certification cases for all registered plugin surfaces."""

    cases: list[PluginCertificationCase] = []
    cases.extend(_capability_cases(registry))
    cases.extend(_loader_cases(registry))
    cases.extend(_check_cases(registry))
    cases.extend(_renderer_cases(registry))
    return PluginCertificationReport(
        version=PLUGIN_CERTIFICATION_VERSION,
        cases=tuple(sorted(cases, key=lambda case: (case.status.value, case.surface, case.name, case.message))),
    )


def render_plugin_certification_json(report: PluginCertificationReport) -> str:
    """Render plugin certification as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_plugin_certification_text(report: PluginCertificationReport) -> str:
    """Render plugin certification as concise CLI text."""

    headline = (
        f"PromptABI plugin certification: {'PASS' if report.ok else 'FAIL'} "
        f"({report.passed} passed, {report.warned} warned, {report.failed} failed)"
    )
    lines = [headline]
    for case in report.cases:
        plugin = f" [{case.plugin}]" if case.plugin else ""
        lines.append(f"{case.status.value.upper()} {case.surface}:{case.name}{plugin} - {case.message}")
    return "\n".join(lines) + "\n"


def _capability_cases(registry: PluginRegistry) -> tuple[PluginCertificationCase, ...]:
    cases: list[PluginCertificationCase] = []
    seen = set()
    for capability in registry.capabilities:
        key = (capability.kind, capability.plugin, capability.name)
        if key in seen:
            cases.append(
                _case(
                    capability.name,
                    PluginCertificationStatus.FAIL,
                    "capability",
                    "duplicate capability identity",
                    plugin=capability.plugin,
                    kind=capability.kind.value,
                )
            )
        else:
            seen.add(key)
        if not _json_safe(capability.to_dict()):
            cases.append(
                _case(
                    capability.name,
                    PluginCertificationStatus.FAIL,
                    "capability",
                    "capability metadata is not JSON-serializable",
                    plugin=capability.plugin,
                )
            )
        if capability.kind in {
            PluginCapabilityKind.CHECK,
            PluginCapabilityKind.PROVIDER_ADAPTER,
            PluginCapabilityKind.GRAMMAR_BACKEND,
            PluginCapabilityKind.SOLVER_ENCODING,
        } and not capability.modes:
            cases.append(
                _case(
                    capability.name,
                    PluginCertificationStatus.WARN,
                    "capability",
                    "capability declares no guarantee modes",
                    plugin=capability.plugin,
                    kind=capability.kind.value,
                )
            )
    cases.append(
        _case(
            "capability-manifest",
            PluginCertificationStatus.PASS if not any(case.status is PluginCertificationStatus.FAIL for case in cases) else PluginCertificationStatus.FAIL,
            "capability",
            f"{len(registry.capabilities)} capabilities have stable machine-readable metadata",
        )
    )
    return tuple(cases)


def _loader_cases(registry: PluginRegistry) -> tuple[PluginCertificationCase, ...]:
    cases: list[PluginCertificationCase] = []
    capability_names = {
        capability.name
        for capability in registry.capabilities
        if capability.kind is PluginCapabilityKind.ARTIFACT_LOADER
    }
    for loader in registry.artifact_loaders:
        if loader.name not in capability_names:
            cases.append(_case(loader.name, PluginCertificationStatus.FAIL, "loader", "loader has no artifact-loader capability"))
        artifact = _sample_artifact(
            (loader.artifact_kinds or (ArtifactKind.SCHEMA,))[0],
            _sample_uri_scheme(loader.uri_schemes),
            f"loader-{loader.name}",
        )
        try:
            loaded = loader.hook(artifact)
        except Exception as exc:
            cases.append(
                _case(loader.name, PluginCertificationStatus.FAIL, "loader", f"loader raised during synthetic load: {exc}")
            )
            continue
        if loaded is None:
            cases.append(
                _case(loader.name, PluginCertificationStatus.WARN, "loader", "loader declined its registered synthetic artifact")
            )
            continue
        if not isinstance(loaded, LoadedArtifact):
            cases.append(_case(loader.name, PluginCertificationStatus.FAIL, "loader", "loader did not return LoadedArtifact"))
            continue
        if loaded.artifact is not artifact:
            cases.append(_case(loader.name, PluginCertificationStatus.FAIL, "loader", "loader returned a different artifact object"))
        if not _json_safe(loaded.to_dict()):
            cases.append(_case(loader.name, PluginCertificationStatus.FAIL, "loader", "loader output is not JSON-serializable"))
        if _contains_secret(loaded.to_dict()):
            cases.append(_case(loader.name, PluginCertificationStatus.FAIL, "loader", "loader output leaked certification secret"))
        if not any(case.name == loader.name and case.surface == "loader" and case.status is PluginCertificationStatus.FAIL for case in cases):
            cases.append(_case(loader.name, PluginCertificationStatus.PASS, "loader", "loader returned certified metadata"))
    return tuple(cases)


def _check_cases(registry: PluginRegistry) -> tuple[PluginCertificationCase, ...]:
    cases: list[PluginCertificationCase] = []
    capability_names = {
        capability.name
        for capability in registry.capabilities
        if capability.kind is PluginCapabilityKind.CHECK
    }
    context = CheckContext(
        config=VerificationConfig(name=f"plugin-certification-{PLUGIN_CERTIFICATION_SECRET}", artifacts={}, checks=()),
        loaded_artifacts=(),
    )
    for registration in registry.checks.values():
        if registration.name not in capability_names:
            cases.append(_case(registration.name, PluginCertificationStatus.FAIL, "check", "check has no check capability"))
        if not registration.modes:
            cases.append(_case(registration.name, PluginCertificationStatus.WARN, "check", "check registered without guarantee modes"))
        try:
            diagnostics = tuple(registration.callable(context))
        except Exception as exc:
            cases.append(
                _case(registration.name, PluginCertificationStatus.FAIL, "check", f"check raised on synthetic context: {exc}")
            )
            continue
        if not all(isinstance(diagnostic, Diagnostic) for diagnostic in diagnostics):
            cases.append(_case(registration.name, PluginCertificationStatus.FAIL, "check", "check emitted non-Diagnostic values"))
            continue
        result = VerificationResult(config=context.config, diagnostics=diagnostics)
        private = apply_witness_privacy(result, WitnessPrivacyMode.HASH_ONLY)
        private_diagnostics = [diagnostic.to_dict() for diagnostic in private.diagnostics]
        if _contains_secret(private_diagnostics):
            cases.append(_case(registration.name, PluginCertificationStatus.FAIL, "check", "hash-only privacy output leaked secret"))
        if not _json_safe(private_diagnostics):
            cases.append(_case(registration.name, PluginCertificationStatus.FAIL, "check", "diagnostics are not JSON-serializable"))
        if not any(case.name == registration.name and case.surface == "check" and case.status is PluginCertificationStatus.FAIL for case in cases):
            cases.append(
                _case(
                    registration.name,
                    PluginCertificationStatus.PASS,
                    "check",
                    f"check emitted {len(diagnostics)} privacy-safe diagnostics",
                    modes=tuple(mode.value for mode in registration.modes),
                )
            )
    return tuple(cases)


def _renderer_cases(registry: PluginRegistry) -> tuple[PluginCertificationCase, ...]:
    cases: list[PluginCertificationCase] = []
    capability_names = {
        capability.name
        for capability in registry.capabilities
        if capability.kind is PluginCapabilityKind.DIAGNOSTIC_RENDERER
    }
    private_result = apply_witness_privacy(_sample_result(), WitnessPrivacyMode.HASH_ONLY)
    for renderer in registry.renderers.values():
        if renderer.format_name not in capability_names:
            cases.append(_case(renderer.format_name, PluginCertificationStatus.FAIL, "renderer", "renderer has no diagnostic-renderer capability"))
        try:
            rendered = renderer.renderer(private_result)
        except Exception as exc:
            cases.append(
                _case(renderer.format_name, PluginCertificationStatus.FAIL, "renderer", f"renderer raised on private result: {exc}")
            )
            continue
        if not isinstance(rendered, str):
            cases.append(_case(renderer.format_name, PluginCertificationStatus.FAIL, "renderer", "renderer did not return text"))
            continue
        if _contains_secret(rendered):
            cases.append(_case(renderer.format_name, PluginCertificationStatus.FAIL, "renderer", "renderer leaked hash-only witness payload"))
        if not rendered:
            cases.append(_case(renderer.format_name, PluginCertificationStatus.WARN, "renderer", "renderer returned empty text"))
        if not any(case.name == renderer.format_name and case.surface == "renderer" and case.status is PluginCertificationStatus.FAIL for case in cases):
            cases.append(_case(renderer.format_name, PluginCertificationStatus.PASS, "renderer", "renderer preserved witness privacy"))
    return tuple(cases)


def _sample_artifact(kind: ArtifactKind, uri_scheme: str | None, name: str) -> BaseArtifact:
    uri = f"{uri_scheme or 'plugin-cert'}://certification/{name}?version=certified"
    return BaseArtifact(
        kind=kind,
        name=name,
        location=ArtifactLocation(uri=uri),
        provenance=ArtifactProvenance(version="certified"),
    )


def _sample_result() -> VerificationResult:
    return VerificationResult(
        config=VerificationConfig(name="plugin-certification-renderer", artifacts={}, checks=()),
        diagnostics=(
            Diagnostic(
                rule_id="plugin-certification-privacy",
                severity=DiagnosticSeverity.INFO,
                message="synthetic privacy probe",
                check_modes=(CheckMode.SOUND,),
                witness=WitnessTrace(
                    summary="renderer privacy probe",
                    steps=(WitnessStep(action="render secret", output=PLUGIN_CERTIFICATION_SECRET),),
                    rendered_strings=(f"system\n{PLUGIN_CERTIFICATION_SECRET}\nassistant",),
                    solver_assignments=({"secret": PLUGIN_CERTIFICATION_SECRET},),
                ),
            ),
        ),
    )


def _sample_uri_scheme(schemes: tuple[str, ...]) -> str | None:
    if not schemes:
        return None
    return schemes[0]


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _contains_secret(value: Any) -> bool:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _case(
    name: str,
    status: PluginCertificationStatus,
    surface: str,
    message: str,
    plugin: str | None = None,
    **details: object,
) -> PluginCertificationCase:
    return PluginCertificationCase(
        name=name,
        status=status,
        surface=surface,
        message=message,
        plugin=plugin,
        details=tuple(details.items()),
    )
