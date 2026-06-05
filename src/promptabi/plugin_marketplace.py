"""Marketplace-style plugin index for PromptABI extension packages."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from ._version import __version__
from .diagnostics import CheckMode
from .plugin_certification import (
    PluginCertificationCase,
    PluginCertificationReport,
    PluginCertificationStatus,
    certify_plugin_registry,
)
from .plugins import PluginCapability, PluginCapabilityKind, PluginRegistry


PLUGIN_MARKETPLACE_INDEX_VERSION = "1.0"
PROMPTABI_MARKETPLACE_GLOBAL_PACKAGE = "promptabi.marketplace.global"
PROMPTABI_COMPATIBILITY_MIN_VERSION = "1.0.0"
PROMPTABI_COMPATIBILITY_MAX_VERSION = "1.x"
PYTHON_COMPATIBILITY_RANGE = ">=3.11"


class PluginMarketplaceVerificationStatus(StrEnum):
    """Verification status published for a plugin index entry."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class PluginMarketplaceCompatibility:
    """Static compatibility declaration for one plugin package."""

    promptabi_min_version: str
    promptabi_max_version: str
    python: str
    plugin_versions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "promptabi": {
                "min_version": self.promptabi_min_version,
                "max_version": self.promptabi_max_version,
            },
            "python": self.python,
            "plugin_versions": list(self.plugin_versions),
        }


@dataclass(frozen=True, slots=True)
class PluginMarketplacePrivacyProfile:
    """Non-sensitive privacy summary for marketplace readers."""

    network_access: str
    witness_privacy: str
    sensitive_metadata: str
    certification_findings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "network_access": self.network_access,
            "witness_privacy": self.witness_privacy,
            "sensitive_metadata": self.sensitive_metadata,
            "certification_findings": list(self.certification_findings),
        }


@dataclass(frozen=True, slots=True)
class PluginMarketplacePackage:
    """One plugin package entry in the marketplace index."""

    name: str
    capabilities: tuple[dict[str, object], ...]
    supported_fragments: tuple[str, ...]
    guarantee_modes: tuple[CheckMode, ...]
    compatibility: PluginMarketplaceCompatibility
    privacy: PluginMarketplacePrivacyProfile
    verification_status: PluginMarketplaceVerificationStatus
    certification_summary: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "supported_fragments": list(self.supported_fragments),
            "guarantee_modes": [mode.value for mode in self.guarantee_modes],
            "compatibility": self.compatibility.to_dict(),
            "privacy": self.privacy.to_dict(),
            "verification": {
                "status": self.verification_status.value,
                "summary": dict(self.certification_summary),
            },
        }


@dataclass(frozen=True, slots=True)
class PluginMarketplaceIndex:
    """Deterministic marketplace index for PromptABI plugins."""

    version: str
    promptabi_version: str
    packages: tuple[PluginMarketplacePackage, ...]
    certification_ok: bool

    @property
    def ok(self) -> bool:
        return self.certification_ok and not any(
            package.verification_status is PluginMarketplaceVerificationStatus.FAIL
            for package in self.packages
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "promptabi_version": self.promptabi_version,
            "ok": self.ok,
            "certification_ok": self.certification_ok,
            "packages": [package.to_dict() for package in self.packages],
        }


def build_plugin_marketplace_index(
    registry: PluginRegistry,
    *,
    certification_report: PluginCertificationReport | None = None,
) -> PluginMarketplaceIndex:
    """Build a deterministic marketplace index from registered plugin metadata."""

    report = certification_report or certify_plugin_registry(registry)
    capabilities_by_plugin: dict[str, list[PluginCapability]] = defaultdict(list)
    for capability in registry.capabilities:
        capabilities_by_plugin[capability.plugin].append(capability)

    cases_by_plugin = _attribute_certification_cases(registry, report)
    package_names = sorted(set(capabilities_by_plugin) | set(cases_by_plugin))
    packages = tuple(
        _package_from_metadata(
            name,
            tuple(sorted(capabilities_by_plugin.get(name, ()), key=_capability_sort_key)),
            tuple(sorted(cases_by_plugin.get(name, ()), key=_case_sort_key)),
        )
        for name in package_names
    )
    return PluginMarketplaceIndex(
        version=PLUGIN_MARKETPLACE_INDEX_VERSION,
        promptabi_version=__version__,
        packages=packages,
        certification_ok=report.ok,
    )


def render_plugin_marketplace_json(index: PluginMarketplaceIndex) -> str:
    """Render the marketplace index as stable JSON."""

    return json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n"


def render_plugin_marketplace_text(index: PluginMarketplaceIndex) -> str:
    """Render the marketplace index as concise CLI text."""

    status = "PASS" if index.ok else "FAIL"
    lines = [
        f"PromptABI plugin marketplace index: {status} ({len(index.packages)} packages)",
        f"index version: {index.version}; promptabi: {index.promptabi_version}",
    ]
    for package in index.packages:
        modes = ", ".join(mode.value for mode in package.guarantee_modes) or "metadata"
        fragments = ", ".join(package.supported_fragments) or "unspecified"
        lines.append(
            f"{package.verification_status.value.upper()} {package.name}: "
            f"{len(package.capabilities)} capabilities; modes={modes}; fragments={fragments}"
        )
        lines.append(
            "  "
            f"privacy network={package.privacy.network_access}; "
            f"witness={package.privacy.witness_privacy}; "
            f"compat=PromptABI {package.compatibility.promptabi_min_version}-{package.compatibility.promptabi_max_version}"
        )
    return "\n".join(lines) + "\n"


def _package_from_metadata(
    name: str,
    capabilities: tuple[PluginCapability, ...],
    cases: tuple[PluginCertificationCase, ...],
) -> PluginMarketplacePackage:
    status = _verification_status(cases)
    return PluginMarketplacePackage(
        name=name,
        capabilities=tuple(capability.to_dict() for capability in capabilities),
        supported_fragments=_supported_fragments(capabilities),
        guarantee_modes=_guarantee_modes(capabilities),
        compatibility=_compatibility(capabilities),
        privacy=_privacy_profile(capabilities, cases),
        verification_status=status,
        certification_summary=_certification_summary(cases),
    )


def _attribute_certification_cases(
    registry: PluginRegistry,
    report: PluginCertificationReport,
) -> dict[str, list[PluginCertificationCase]]:
    capability_owner = {
        (capability.kind, capability.name): capability.plugin
        for capability in registry.capabilities
    }
    cases_by_plugin: dict[str, list[PluginCertificationCase]] = defaultdict(list)
    for case in report.cases:
        plugin = case.plugin or _case_owner(case, capability_owner)
        cases_by_plugin[plugin].append(case)
    return cases_by_plugin


def _case_owner(
    case: PluginCertificationCase,
    capability_owner: dict[tuple[PluginCapabilityKind, str], str],
) -> str:
    if case.surface == "loader":
        return capability_owner.get((PluginCapabilityKind.ARTIFACT_LOADER, case.name), PROMPTABI_MARKETPLACE_GLOBAL_PACKAGE)
    if case.surface == "check":
        return capability_owner.get((PluginCapabilityKind.CHECK, case.name), PROMPTABI_MARKETPLACE_GLOBAL_PACKAGE)
    if case.surface == "renderer":
        return capability_owner.get((PluginCapabilityKind.DIAGNOSTIC_RENDERER, case.name), PROMPTABI_MARKETPLACE_GLOBAL_PACKAGE)
    if case.surface == "capability":
        for kind in PluginCapabilityKind:
            owner = capability_owner.get((kind, case.name))
            if owner is not None:
                return owner
    return PROMPTABI_MARKETPLACE_GLOBAL_PACKAGE


def _supported_fragments(capabilities: tuple[PluginCapability, ...]) -> tuple[str, ...]:
    fragments: set[str] = set()
    for capability in capabilities:
        fragments.add(f"capability:{capability.kind.value}")
        for key, value in capability.properties:
            if key in {"artifact_kinds", "checks", "providers", "uri_schemes"}:
                fragments.update(f"{key}:{item}" for item in _string_values(value))
            elif key in {"grammar_type", "template_format", "framework", "tool_provider", "api_family", "schema_source", "check"}:
                fragments.add(f"{key}:{value}")
            elif key in {"supported_fragment", "supported_fragments"}:
                fragments.update(f"supported:{item}" for item in _string_values(value))
    return tuple(sorted(fragments))


def _guarantee_modes(capabilities: tuple[PluginCapability, ...]) -> tuple[CheckMode, ...]:
    return tuple(sorted({mode for capability in capabilities for mode in capability.modes}, key=lambda mode: mode.value))


def _compatibility(capabilities: tuple[PluginCapability, ...]) -> PluginMarketplaceCompatibility:
    versions = tuple(sorted({capability.version for capability in capabilities if capability.version is not None}))
    return PluginMarketplaceCompatibility(
        promptabi_min_version=PROMPTABI_COMPATIBILITY_MIN_VERSION,
        promptabi_max_version=PROMPTABI_COMPATIBILITY_MAX_VERSION,
        python=PYTHON_COMPATIBILITY_RANGE,
        plugin_versions=versions,
    )


def _privacy_profile(
    capabilities: tuple[PluginCapability, ...],
    cases: tuple[PluginCertificationCase, ...],
) -> PluginMarketplacePrivacyProfile:
    network_values = sorted(
        str(value)
        for capability in capabilities
        for key, value in capability.properties
        if key in {"network", "network_access"}
    )
    network_access = "unknown"
    if network_values and all(value == "never" for value in network_values):
        network_access = "never"
    elif network_values:
        network_access = ",".join(network_values)

    privacy_findings = tuple(
        sorted(
            {
                f"{case.status.value}:{case.surface}:{case.name}:{case.message}"
                for case in cases
                if _is_privacy_case(case)
            }
        )
    )
    failed_privacy = any(
        case.status is PluginCertificationStatus.FAIL and _is_privacy_case(case)
        for case in cases
    )
    warned_privacy = any(
        case.status is PluginCertificationStatus.WARN and _is_privacy_case(case)
        for case in cases
    )
    if failed_privacy:
        witness_privacy = "failed"
        sensitive_metadata = "unsafe"
    elif warned_privacy:
        witness_privacy = "warn"
        sensitive_metadata = "review"
    else:
        witness_privacy = "hash-only-certified"
        sensitive_metadata = "metadata-only-or-unknown"

    return PluginMarketplacePrivacyProfile(
        network_access=network_access,
        witness_privacy=witness_privacy,
        sensitive_metadata=sensitive_metadata,
        certification_findings=privacy_findings,
    )


def _verification_status(cases: tuple[PluginCertificationCase, ...]) -> PluginMarketplaceVerificationStatus:
    if any(case.status is PluginCertificationStatus.FAIL for case in cases):
        return PluginMarketplaceVerificationStatus.FAIL
    if any(case.status is PluginCertificationStatus.WARN for case in cases):
        return PluginMarketplaceVerificationStatus.WARN
    return PluginMarketplaceVerificationStatus.PASS


def _certification_summary(cases: tuple[PluginCertificationCase, ...]) -> tuple[tuple[str, int], ...]:
    counts = {
        "passed": sum(1 for case in cases if case.status is PluginCertificationStatus.PASS),
        "warned": sum(1 for case in cases if case.status is PluginCertificationStatus.WARN),
        "failed": sum(1 for case in cases if case.status is PluginCertificationStatus.FAIL),
    }
    return tuple(counts.items())


def _is_privacy_case(case: PluginCertificationCase) -> bool:
    message = case.message.lower()
    return "secret" in message or "privacy" in message or "hash-only" in message


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, int, float, bool)):
        return (str(value),)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _capability_sort_key(capability: PluginCapability) -> tuple[str, str, str]:
    return (capability.kind.value, capability.plugin, capability.name)


def _case_sort_key(case: PluginCertificationCase) -> tuple[str, str, str, str]:
    return (case.status.value, case.surface, case.name, case.message)
