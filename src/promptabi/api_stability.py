"""Public API stability metadata, docs generation, and compatibility checks."""

from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import wraps
from importlib import import_module
from typing import Any, TypeVar


PUBLIC_API_POLICY_VERSION = "2026.06"
STABLE_API_MODULE = "promptabi"

F = TypeVar("F", bound=Callable[..., Any])


class ApiStability(StrEnum):
    """Compatibility promise attached to a public PromptABI symbol."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    DEPRECATED = "deprecated"


class ApiCompatibilityIssueKind(StrEnum):
    """Kinds of public API compatibility failures."""

    REMOVED_STABLE_SYMBOL = "removed-stable-symbol"
    CHANGED_STABLE_SIGNATURE = "changed-stable-signature"
    REMOVED_DEPRECATION_METADATA = "removed-deprecation-metadata"


@dataclass(frozen=True, slots=True)
class DeprecatedApi:
    """Machine-readable deprecation metadata for a public API symbol."""

    since: str
    replacement: str | None = None
    remove_in: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.since:
            raise ValueError("deprecated API metadata requires a non-empty 'since' version")
        if self.replacement is not None and not self.replacement:
            raise ValueError("deprecated API replacement must be non-empty when provided")
        if self.remove_in is not None and not self.remove_in:
            raise ValueError("deprecated API removal version must be non-empty when provided")
        if self.reason is not None and not self.reason:
            raise ValueError("deprecated API reason must be non-empty when provided")

    def to_dict(self) -> dict[str, str]:
        payload = {"since": self.since}
        if self.replacement is not None:
            payload["replacement"] = self.replacement
        if self.remove_in is not None:
            payload["remove_in"] = self.remove_in
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class ApiSymbol:
    """One importable public symbol and its compatibility metadata."""

    name: str
    module: str
    kind: str
    stability: ApiStability
    signature: str | None = None
    summary: str | None = None
    deprecated: DeprecatedApi | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("API symbol name must be non-empty")
        if not self.module:
            raise ValueError("API symbol module must be non-empty")
        if not self.kind:
            raise ValueError("API symbol kind must be non-empty")
        if self.stability is ApiStability.DEPRECATED and self.deprecated is None:
            raise ValueError("deprecated API symbols must include deprecation metadata")

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "module": self.module,
            "kind": self.kind,
            "stability": self.stability.value,
        }
        if self.signature is not None:
            payload["signature"] = self.signature
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.deprecated is not None:
            payload["deprecated"] = self.deprecated.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class PublicApiManifest:
    """Generated public API manifest used by docs and compatibility tests."""

    policy_version: str
    module: str
    symbols: tuple[ApiSymbol, ...] = field(default_factory=tuple)
    stability_policy: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise ValueError("public API manifest policy version must be non-empty")
        if not self.module:
            raise ValueError("public API manifest module must be non-empty")
        object.__setattr__(self, "symbols", tuple(sorted(self.symbols, key=lambda symbol: symbol.name)))
        names = [symbol.name for symbol in self.symbols]
        if len(set(names)) != len(names):
            raise ValueError("public API manifest symbol names must be unique")

    def symbol_map(self) -> dict[str, ApiSymbol]:
        return {symbol.name: symbol for symbol in self.symbols}

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "module": self.module,
            "stability_policy": list(self.stability_policy),
            "symbols": [symbol.to_dict() for symbol in self.symbols],
        }


@dataclass(frozen=True, slots=True)
class PublicApiCompatibilityIssue:
    """A stable public API compatibility issue detected against a baseline."""

    kind: ApiCompatibilityIssueKind
    symbol: str
    message: str
    baseline: str | None = None
    current: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "kind": self.kind.value,
            "symbol": self.symbol,
            "message": self.message,
        }
        if self.baseline is not None:
            payload["baseline"] = self.baseline
        if self.current is not None:
            payload["current"] = self.current
        return payload


STABILITY_POLICY: tuple[str, ...] = (
    "Stable symbols keep import paths, callability, dataclass fields, enum values, and documented constructor arguments "
    "compatible for the entire 1.x line.",
    "Stable callable signatures may add keyword-only parameters with defaults, but existing required parameters are not "
    "removed or renamed without a deprecation cycle.",
    "Provisional symbols remain importable for experimentation but may change before they are promoted into the stable "
    "plugin and embedding contract.",
    "Deprecated symbols emit DeprecationWarning through the decorator utility and carry replacement/removal metadata in "
    "the generated manifest before removal.",
)

STABLE_PUBLIC_API: frozenset[str] = frozenset(
    {
        "ApiCompatibilityIssueKind",
        "ApiStability",
        "ApiSymbol",
        "ArtifactBundle",
        "ArtifactKind",
        "ArtifactLoadHook",
        "ArtifactLocation",
        "ArtifactLoader",
        "ArtifactRef",
        "CheckCallable",
        "CheckContext",
        "CheckMode",
        "Diagnostic",
        "DiagnosticCluster",
        "DiagnosticClusterMember",
        "DiagnosticClusterReport",
        "DiagnosticClusterStrategy",
        "DiagnosticRenderer",
        "DiagnosticSeverity",
        "ContributionWorkflow",
        "FixBlastRadius",
        "FixCompatibility",
        "FixSafety",
        "INTEGRATION_API_VERSION",
        "INTEGRATION_CAPABILITIES",
        "IntegrationArtifactSummary",
        "IntegrationCapability",
        "IntegrationGate",
        "IntegrationReport",
        "IntegrationRequest",
        "IntegrationSurface",
        "LoadedArtifact",
        "LocalMetricsReport",
        "PUBLIC_API_POLICY_VERSION",
        "PluginArtifactLoaderRegistration",
        "PluginCapability",
        "PluginCapabilityKind",
        "PluginCheckRegistration",
        "PluginError",
        "PluginMarketplaceCompatibility",
        "PluginMarketplaceIndex",
        "PluginMarketplacePackage",
        "PluginMarketplacePrivacyProfile",
        "PluginMarketplaceVerificationStatus",
        "PluginRegistrar",
        "PluginRegistry",
        "PluginRendererRegistration",
        "ProviderConfigArtifact",
        "PublicApiCompatibilityIssue",
        "PublicApiManifest",
        "RankedFixSuggestion",
        "STABILITY_POLICY",
        "STABLE_PUBLIC_API",
        "SchemaArtifact",
        "SourceSpan",
        "VerificationConfig",
        "VerificationResult",
        "VerificationSession",
        "WitnessStep",
        "WitnessPrivacyMode",
        "WitnessTrace",
        "apply_witness_privacy",
        "collect_diagnostics",
        "compare_public_api_manifests",
        "create_session",
        "diagnostic_clusters",
        "contribution_workflows",
        "build_diagnostic_clusters",
        "build_integration_report",
        "build_local_metrics_report",
        "build_plugin_marketplace_index",
        "build_contribution_workflows",
        "deprecated_api",
        "load_artifacts",
        "local_metrics",
        "load_entry_point_plugins",
        "load_plugin_modules",
        "plugin_marketplace_index",
        "public_api_manifest_from_mapping",
        "public_api_reference",
        "private_witness",
        "rank_fix_suggestions",
        "render_local_metrics_json",
        "render_local_metrics_text",
        "render_plugin_marketplace_json",
        "render_plugin_marketplace_text",
        "render_result",
        "render_diagnostic_clusters_json",
        "render_diagnostic_clusters_text",
        "render_integration_report_json",
        "render_integration_report_text",
        "render_public_api_manifest_json",
        "render_public_api_manifest_markdown",
        "run_verification",
    }
)


def deprecated_api(
    *,
    since: str,
    replacement: str | None = None,
    remove_in: str | None = None,
    reason: str | None = None,
) -> Callable[[F], F]:
    """Mark a public callable as deprecated while preserving its type signature."""

    metadata = DeprecatedApi(since=since, replacement=replacement, remove_in=remove_in, reason=reason)

    def decorate(func: F) -> F:
        message = f"{func.__module__}.{func.__name__} is deprecated since {metadata.since}"
        if metadata.replacement is not None:
            message += f"; use {metadata.replacement}"
        if metadata.remove_in is not None:
            message += f"; removal is planned for {metadata.remove_in}"
        if metadata.reason is not None:
            message += f": {metadata.reason}"

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        setattr(wrapper, "__promptabi_deprecated__", metadata)
        return wrapper  # type: ignore[return-value]

    return decorate


def build_public_api_manifest(
    *,
    module_name: str = STABLE_API_MODULE,
    stable_symbols: Iterable[str] = STABLE_PUBLIC_API,
) -> PublicApiManifest:
    """Inspect the public package namespace and return a generated API manifest."""

    module = import_module(module_name)
    exported_names = tuple(getattr(module, "__all__", ()))
    stable = frozenset(stable_symbols)
    symbols = tuple(_symbol_from_object(module_name, name, getattr(module, name), stable) for name in exported_names)
    return PublicApiManifest(
        policy_version=PUBLIC_API_POLICY_VERSION,
        module=module_name,
        symbols=symbols,
        stability_policy=STABILITY_POLICY,
    )


def compare_public_api_manifests(
    baseline: PublicApiManifest,
    current: PublicApiManifest,
) -> tuple[PublicApiCompatibilityIssue, ...]:
    """Return compatibility issues that would break stable downstream users."""

    current_symbols = current.symbol_map()
    issues: list[PublicApiCompatibilityIssue] = []
    for baseline_symbol in baseline.symbols:
        if baseline_symbol.stability not in (ApiStability.STABLE, ApiStability.DEPRECATED):
            continue
        current_symbol = current_symbols.get(baseline_symbol.name)
        if current_symbol is None:
            issues.append(
                PublicApiCompatibilityIssue(
                    kind=ApiCompatibilityIssueKind.REMOVED_STABLE_SYMBOL,
                    symbol=baseline_symbol.name,
                    message=f"stable public API symbol was removed: {baseline_symbol.name}",
                    baseline=baseline_symbol.qualified_name,
                )
            )
            continue
        if baseline_symbol.signature and current_symbol.signature:
            baseline_required = _required_signature_prefix(baseline_symbol.signature)
            current_required = _required_signature_prefix(current_symbol.signature)
            if baseline_required != current_required:
                issues.append(
                    PublicApiCompatibilityIssue(
                        kind=ApiCompatibilityIssueKind.CHANGED_STABLE_SIGNATURE,
                        symbol=baseline_symbol.name,
                        message=f"stable public API signature changed: {baseline_symbol.name}",
                        baseline=baseline_symbol.signature,
                        current=current_symbol.signature,
                    )
                )
        if baseline_symbol.deprecated is not None and current_symbol.deprecated is None:
            issues.append(
                PublicApiCompatibilityIssue(
                    kind=ApiCompatibilityIssueKind.REMOVED_DEPRECATION_METADATA,
                    symbol=baseline_symbol.name,
                    message=f"deprecated symbol lost deprecation metadata: {baseline_symbol.name}",
                )
            )
    return tuple(issues)


def public_api_manifest_from_mapping(data: Mapping[str, object]) -> PublicApiManifest:
    """Load a public API manifest from JSON-compatible mapping data."""

    return PublicApiManifest(
        policy_version=str(data["policy_version"]),
        module=str(data["module"]),
        stability_policy=tuple(str(item) for item in data.get("stability_policy", ())),  # type: ignore[arg-type]
        symbols=tuple(_api_symbol_from_mapping(item) for item in data["symbols"]),  # type: ignore[index,arg-type]
    )


def render_public_api_manifest_json(manifest: PublicApiManifest) -> str:
    """Render a public API manifest as deterministic JSON."""

    return json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"


def render_public_api_manifest_markdown(manifest: PublicApiManifest) -> str:
    """Render a concise generated Markdown API reference."""

    counts = {
        stability.value: sum(1 for symbol in manifest.symbols if symbol.stability is stability)
        for stability in ApiStability
    }
    lines = [
        "# PromptABI public API",
        "",
        f"Generated from `{manifest.module}.__all__` under policy `{manifest.policy_version}`.",
        "",
        "## Stability policy",
        "",
    ]
    lines.extend(f"- {rule}" for rule in manifest.stability_policy)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Stability | Symbols |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {stability} | {count} |" for stability, count in counts.items())
    lines.extend(
        [
            "",
            "## Stable embedding and plugin surface",
            "",
            "| Symbol | Kind | Signature | Summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for symbol in manifest.symbols:
        if symbol.stability is not ApiStability.STABLE:
            continue
        lines.append(
            "| "
            f"`{symbol.name}` | {symbol.kind} | {_markdown_code(symbol.signature or '')} | "
            f"{_escape_markdown(symbol.summary or '')} |"
        )
    deprecated = [symbol for symbol in manifest.symbols if symbol.stability is ApiStability.DEPRECATED]
    if deprecated:
        lines.extend(["", "## Deprecated symbols", "", "| Symbol | Replacement | Removal | Reason |", "| --- | --- | --- | --- |"])
        for symbol in deprecated:
            metadata = symbol.deprecated
            assert metadata is not None
            lines.append(
                "| "
                f"`{symbol.name}` | {metadata.replacement or ''} | {metadata.remove_in or ''} | "
                f"{_escape_markdown(metadata.reason or '')} |"
            )
    lines.extend(["", "## Provisional surface", ""])
    provisional_names = [symbol.name for symbol in manifest.symbols if symbol.stability is ApiStability.PROVISIONAL]
    lines.append(
        "The remaining importable names are intentionally provisional until they are promoted: "
        + ", ".join(f"`{name}`" for name in provisional_names)
        + "."
    )
    return "\n".join(lines) + "\n"


def _symbol_from_object(module_name: str, name: str, obj: object, stable_symbols: frozenset[str]) -> ApiSymbol:
    deprecated = getattr(obj, "__promptabi_deprecated__", None)
    if deprecated is not None and not isinstance(deprecated, DeprecatedApi):
        deprecated = None
    if isinstance(deprecated, DeprecatedApi):
        stability = ApiStability.DEPRECATED
    elif name in stable_symbols:
        stability = ApiStability.STABLE
    else:
        stability = ApiStability.PROVISIONAL
    return ApiSymbol(
        name=name,
        module=module_name,
        kind=_symbol_kind(obj),
        stability=stability,
        signature=_safe_signature(obj),
        summary=_summary(obj),
        deprecated=deprecated,
    )


def _api_symbol_from_mapping(data: Mapping[str, object]) -> ApiSymbol:
    deprecated = data.get("deprecated")
    return ApiSymbol(
        name=str(data["name"]),
        module=str(data["module"]),
        kind=str(data["kind"]),
        stability=ApiStability(str(data["stability"])),
        signature=str(data["signature"]) if data.get("signature") is not None else None,
        summary=str(data["summary"]) if data.get("summary") is not None else None,
        deprecated=(
            DeprecatedApi(
                since=str(deprecated["since"]),  # type: ignore[index]
                replacement=str(deprecated["replacement"]) if deprecated.get("replacement") is not None else None,  # type: ignore[union-attr,index]
                remove_in=str(deprecated["remove_in"]) if deprecated.get("remove_in") is not None else None,  # type: ignore[union-attr,index]
                reason=str(deprecated["reason"]) if deprecated.get("reason") is not None else None,  # type: ignore[union-attr,index]
            )
            if isinstance(deprecated, Mapping)
            else None
        ),
    )


def _symbol_kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if callable(obj):
        return "callable"
    return type(obj).__name__


def _safe_signature(obj: object) -> str | None:
    if not callable(obj):
        return None
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _summary(obj: object) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    return doc.splitlines()[0]


def _required_signature_prefix(signature: str) -> str:
    required: list[str] = []
    for part in signature.strip("()").split(", "):
        if not part or part in {"/", "*"}:
            continue
        if part.startswith("*") or "=" in part:
            continue
        required.append(part)
    return ", ".join(required)


def _markdown_code(value: str) -> str:
    return f"`{_escape_markdown(value)}`" if value else ""


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
