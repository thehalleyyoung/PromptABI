"""Typed plugin registry for PromptABI extension points."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import metadata
from typing import TYPE_CHECKING, Protocol

from .artifacts import Artifact, ArtifactKind
from .diagnostics import CheckMode

if TYPE_CHECKING:
    from .diagnostics import Diagnostic
    from .loaders import LoadedArtifact
    from .session import CheckContext, VerificationResult


PROMPTABI_PLUGIN_ENTRY_POINT = "promptabi.plugins"


class PluginError(ValueError):
    """Raised when a plugin cannot be registered safely."""


class PluginCapabilityKind(StrEnum):
    """Supported first-class PromptABI plugin extension families."""

    ARTIFACT_LOADER = "artifact-loader"
    CHECK = "check"
    PROVIDER_ADAPTER = "provider-adapter"
    GRAMMAR_BACKEND = "grammar-backend"
    TEMPLATE_DIALECT = "template-dialect"
    TRUNCATION_POLICY = "truncation-policy"
    SOLVER_ENCODING = "solver-encoding"
    DIAGNOSTIC_RENDERER = "diagnostic-renderer"


class ArtifactLoadHook(Protocol):
    """Return a loaded artifact when the plugin handles it, otherwise ``None``."""

    def __call__(self, artifact: Artifact) -> "LoadedArtifact | None": ...


class DiagnosticRenderer(Protocol):
    """Render a complete verification result to text."""

    def __call__(self, result: "VerificationResult") -> str: ...


class PluginRegistrar(Protocol):
    """Protocol implemented by module-level or object plugins."""

    def register_promptabi_plugin(self, registry: "PluginRegistry") -> None: ...


PluginCheckCallable = Callable[["CheckContext"], Iterable["Diagnostic"]]


@dataclass(frozen=True, slots=True)
class PluginCapability:
    """Machine-readable metadata for a registered extension surface."""

    kind: PluginCapabilityKind
    name: str
    plugin: str
    version: str | None = None
    modes: tuple[CheckMode, ...] = ()
    properties: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise PluginError("plugin capability name must be non-empty")
        if not self.plugin:
            raise PluginError("plugin capability plugin must be non-empty")
        object.__setattr__(self, "modes", tuple(sorted(self.modes, key=lambda mode: mode.value)))
        object.__setattr__(self, "properties", tuple(sorted(self.properties, key=lambda item: item[0])))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "name": self.name,
            "plugin": self.plugin,
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.modes:
            payload["modes"] = [mode.value for mode in self.modes]
        if self.properties:
            payload["properties"] = dict(self.properties)
        return payload


@dataclass(frozen=True, slots=True)
class PluginCheckRegistration:
    """A check registered by a plugin with scheduler metadata."""

    name: str
    callable: PluginCheckCallable
    artifact_kinds: tuple[ArtifactKind, ...] = ()
    after: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    modes: tuple[CheckMode, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise PluginError("plugin check name must be non-empty")
        object.__setattr__(self, "artifact_kinds", tuple(self.artifact_kinds))
        object.__setattr__(self, "after", tuple(dict.fromkeys(self.after)))
        object.__setattr__(self, "resources", tuple(dict.fromkeys(self.resources)))
        object.__setattr__(self, "modes", tuple(sorted(self.modes, key=lambda mode: mode.value)))


@dataclass(frozen=True, slots=True)
class PluginArtifactLoaderRegistration:
    """An artifact loader hook registered by a plugin."""

    name: str
    hook: ArtifactLoadHook
    artifact_kinds: tuple[ArtifactKind, ...] = ()
    uri_schemes: tuple[str, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise PluginError("plugin artifact loader name must be non-empty")
        object.__setattr__(self, "artifact_kinds", tuple(self.artifact_kinds))
        object.__setattr__(self, "uri_schemes", tuple(dict.fromkeys(self.uri_schemes)))

    def matches(self, artifact: Artifact) -> bool:
        if self.artifact_kinds and artifact.kind not in self.artifact_kinds:
            return False
        if self.uri_schemes:
            uri = artifact.location.uri
            if uri is None or ":" not in uri:
                return False
            if uri.split(":", 1)[0] not in self.uri_schemes:
                return False
        return True


@dataclass(frozen=True, slots=True)
class PluginRendererRegistration:
    """A diagnostic renderer registered by a plugin."""

    format_name: str
    renderer: DiagnosticRenderer
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not self.format_name:
            raise PluginError("plugin renderer format must be non-empty")


@dataclass(slots=True)
class PluginRegistry:
    """Mutable registry used to assemble PromptABI extensions explicitly."""

    capabilities: list[PluginCapability] = field(default_factory=list)
    checks: dict[str, PluginCheckRegistration] = field(default_factory=dict)
    artifact_loaders: list[PluginArtifactLoaderRegistration] = field(default_factory=list)
    renderers: dict[str, PluginRendererRegistration] = field(default_factory=dict)

    def register_capability(
        self,
        kind: PluginCapabilityKind | str,
        name: str,
        *,
        plugin: str,
        version: str | None = None,
        modes: Sequence[CheckMode | str] = (),
        properties: Mapping[str, object] | None = None,
    ) -> PluginCapability:
        capability = PluginCapability(
            kind=PluginCapabilityKind(kind),
            name=name,
            plugin=plugin,
            version=version,
            modes=tuple(CheckMode(mode) for mode in modes),
            properties=tuple((properties or {}).items()),
        )
        self.capabilities.append(capability)
        return capability

    def register_check(
        self,
        name: str,
        check: PluginCheckCallable,
        *,
        artifact_kinds: Sequence[ArtifactKind | str] = (),
        after: Sequence[str] = (),
        resources: Sequence[str] = (),
        modes: Sequence[CheckMode | str] = (),
        plugin: str = "local",
        version: str | None = None,
    ) -> PluginCheckRegistration:
        if name in self.checks:
            raise PluginError(f"plugin check already registered: {name}")
        registration = PluginCheckRegistration(
            name=name,
            callable=check,
            artifact_kinds=tuple(ArtifactKind(kind) for kind in artifact_kinds),
            after=tuple(after),
            resources=tuple(resources),
            modes=tuple(CheckMode(mode) for mode in modes),
        )
        self.checks[name] = registration
        self.register_capability(
            PluginCapabilityKind.CHECK,
            name,
            plugin=plugin,
            version=version,
            modes=registration.modes,
        )
        return registration

    def register_artifact_loader(
        self,
        name: str,
        hook: ArtifactLoadHook,
        *,
        artifact_kinds: Sequence[ArtifactKind | str] = (),
        uri_schemes: Sequence[str] = (),
        priority: int = 0,
        plugin: str = "local",
        version: str | None = None,
    ) -> PluginArtifactLoaderRegistration:
        registration = PluginArtifactLoaderRegistration(
            name=name,
            hook=hook,
            artifact_kinds=tuple(ArtifactKind(kind) for kind in artifact_kinds),
            uri_schemes=tuple(uri_schemes),
            priority=priority,
        )
        self.artifact_loaders.append(registration)
        self.artifact_loaders.sort(key=lambda item: (-item.priority, item.name))
        properties: dict[str, object] = {}
        if registration.artifact_kinds:
            properties["artifact_kinds"] = [kind.value for kind in registration.artifact_kinds]
        if registration.uri_schemes:
            properties["uri_schemes"] = list(registration.uri_schemes)
        self.register_capability(
            PluginCapabilityKind.ARTIFACT_LOADER,
            name,
            plugin=plugin,
            version=version,
            properties=properties,
        )
        return registration

    def register_renderer(
        self,
        format_name: str,
        renderer: DiagnosticRenderer,
        *,
        media_type: str | None = None,
        plugin: str = "local",
        version: str | None = None,
    ) -> PluginRendererRegistration:
        if format_name in self.renderers:
            raise PluginError(f"plugin renderer already registered: {format_name}")
        registration = PluginRendererRegistration(
            format_name=format_name,
            renderer=renderer,
            media_type=media_type,
        )
        self.renderers[format_name] = registration
        properties = {"media_type": media_type} if media_type is not None else None
        self.register_capability(
            PluginCapabilityKind.DIAGNOSTIC_RENDERER,
            format_name,
            plugin=plugin,
            version=version,
            properties=properties,
        )
        return registration

    def resolve_artifact_loader(self, artifact: Artifact) -> ArtifactLoadHook | None:
        for registration in self.artifact_loaders:
            if registration.matches(artifact):
                return registration.hook
        return None

    def render(self, format_name: str, result: "VerificationResult") -> str:
        try:
            renderer = self.renderers[format_name].renderer
        except KeyError as exc:
            raise PluginError(f"unknown plugin renderer format: {format_name}") from exc
        return renderer(result)

    def merge(self, other: "PluginRegistry") -> None:
        for registration in other.checks.values():
            self.register_check(
                registration.name,
                registration.callable,
                artifact_kinds=registration.artifact_kinds,
                after=registration.after,
                resources=registration.resources,
                modes=registration.modes,
                plugin="merged",
            )
        for registration in other.artifact_loaders:
            self.register_artifact_loader(
                registration.name,
                registration.hook,
                artifact_kinds=registration.artifact_kinds,
                uri_schemes=registration.uri_schemes,
                priority=registration.priority,
                plugin="merged",
            )
        for registration in other.renderers.values():
            self.register_renderer(
                registration.format_name,
                registration.renderer,
                media_type=registration.media_type,
                plugin="merged",
            )
        self.capabilities.extend(other.capabilities)


def load_plugin_modules(module_specs: Sequence[str], *, registry: PluginRegistry | None = None) -> PluginRegistry:
    """Import module plugins and let them register with a registry.

    A module can expose ``register_promptabi_plugin(registry)`` or a specific
    ``module:object`` whose object implements that method or is directly callable
    with the registry.
    """

    target = registry or PluginRegistry()
    for spec in module_specs:
        module_name, separator, object_name = spec.partition(":")
        if not module_name:
            raise PluginError("plugin module spec must be non-empty")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PluginError(f"could not import PromptABI plugin '{spec}': {exc}") from exc
        if separator:
            try:
                plugin_object = getattr(module, object_name)
            except AttributeError as exc:
                raise PluginError(f"PromptABI plugin '{spec}' does not define '{object_name}'") from exc
        else:
            plugin_object = module
        _register_plugin_object(plugin_object, target, spec)
    return target


def load_entry_point_plugins(
    *,
    group: str = PROMPTABI_PLUGIN_ENTRY_POINT,
    registry: PluginRegistry | None = None,
) -> PluginRegistry:
    """Load installed plugins from ``promptabi.plugins`` entry points."""

    target = registry or PluginRegistry()
    try:
        entry_points = metadata.entry_points(group=group)
    except TypeError:
        entry_points = metadata.entry_points().select(group=group)
    for entry_point in entry_points:
        try:
            plugin_object = entry_point.load()
        except Exception as exc:
            raise PluginError(f"could not load PromptABI plugin entry point '{entry_point.name}': {exc}") from exc
        _register_plugin_object(plugin_object, target, entry_point.name)
    return target


def _register_plugin_object(plugin_object: object, registry: PluginRegistry, label: str) -> None:
    register = getattr(plugin_object, "register_promptabi_plugin", None)
    if callable(register):
        register(registry)
        return
    if callable(plugin_object):
        plugin_object(registry)
        return
    raise PluginError(
        f"PromptABI plugin '{label}' must expose register_promptabi_plugin(registry) or be callable"
    )
