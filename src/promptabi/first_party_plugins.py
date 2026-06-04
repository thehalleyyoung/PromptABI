"""Bundled PromptABI plugin pack for common LLM interface stacks."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from ._version import __version__
from .artifacts import Artifact, ArtifactKind
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .loaders import ArtifactLoadWarning, LoadedArtifact
from .plugins import (
    PluginCapabilityKind,
    PluginRegistry,
)


FIRST_PARTY_PLUGIN_NAME = "promptabi.first_party"


@dataclass(frozen=True, slots=True)
class FirstPartyPluginSpec:
    """A bundled adapter surface that can be inspected by CLI and API users."""

    family: str
    kind: PluginCapabilityKind
    name: str
    modes: tuple[CheckMode, ...]
    properties: tuple[tuple[str, object], ...] = ()


FIRST_PARTY_PLUGIN_SPECS: tuple[FirstPartyPluginSpec, ...] = (
    FirstPartyPluginSpec(
        "huggingface",
        PluginCapabilityKind.ARTIFACT_LOADER,
        "huggingface-artifact-reference",
        (CheckMode.SOUND, CheckMode.COMPLETE),
        (
            ("artifact_kinds", ("tokenizer", "chat-template", "special-token-map")),
            ("uri_schemes", ("hf",)),
            ("network", "never"),
        ),
    ),
    FirstPartyPluginSpec(
        "huggingface",
        PluginCapabilityKind.TEMPLATE_DIALECT,
        "huggingface-tokenizer-config-chat-template",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("template_format", "jinja"), ("artifact", "tokenizer_config.json")),
    ),
    FirstPartyPluginSpec(
        "openai-compatible",
        PluginCapabilityKind.PROVIDER_ADAPTER,
        "openai-compatible-provider-contract",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("providers", ("openai", "azure-openai", "litellm")), ("tool_envelope", "request.tools")),
    ),
    FirstPartyPluginSpec(
        "vllm",
        PluginCapabilityKind.PROVIDER_ADAPTER,
        "vllm-openai-server-contract",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("api_family", "openai-compatible"), ("stop_policy_source", "vllm_sampling_params")),
    ),
    FirstPartyPluginSpec(
        "llama.cpp",
        PluginCapabilityKind.PROVIDER_ADAPTER,
        "llamacpp-server-contract",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("api_family", "openai-compatible"), ("local_artifacts", ("gguf", "server-options"))),
    ),
    FirstPartyPluginSpec(
        "langchain",
        PluginCapabilityKind.TRUNCATION_POLICY,
        "langchain-rag-truncation-policy",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("framework", "langchain"), ("checks", ("rag-chunking-compatibility", "token-budget-model"))),
    ),
    FirstPartyPluginSpec(
        "llamaindex",
        PluginCapabilityKind.TRUNCATION_POLICY,
        "llamaindex-agent-truncation-policy",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("framework", "llamaindex"), ("checks", ("rag-chunking-compatibility", "token-budget-model"))),
    ),
    FirstPartyPluginSpec(
        "outlines",
        PluginCapabilityKind.GRAMMAR_BACKEND,
        "outlines-grammar-backend",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("grammar_type", "outlines"), ("checks", ("grammar-differential", "parser-compatibility"))),
    ),
    FirstPartyPluginSpec(
        "xgrammar",
        PluginCapabilityKind.GRAMMAR_BACKEND,
        "xgrammar-backend",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("grammar_type", "xgrammar"), ("checks", ("grammar-tokenizer-emptiness", "grammar-tokenizer-ambiguity"))),
    ),
    FirstPartyPluginSpec(
        "llguidance",
        PluginCapabilityKind.GRAMMAR_BACKEND,
        "llguidance-backend",
        (CheckMode.BOUNDED, CheckMode.HEURISTIC),
        (("grammar_type", "llguidance"), ("checks", ("grammar-differential", "parser-compatibility"))),
    ),
    FirstPartyPluginSpec(
        "pydantic",
        PluginCapabilityKind.GRAMMAR_BACKEND,
        "pydantic-json-schema-backend",
        (CheckMode.SOUND, CheckMode.BOUNDED),
        (("schema_source", "pydantic"), ("normalizes_to", "json-schema")),
    ),
    FirstPartyPluginSpec(
        "pydantic",
        PluginCapabilityKind.CHECK,
        "pydantic-tool-schema-ingestion",
        (CheckMode.SOUND, CheckMode.COMPLETE),
        (("tool_provider", "pydantic"), ("check", "tool-schema-ingestion")),
    ),
    FirstPartyPluginSpec(
        "mcp",
        PluginCapabilityKind.PROVIDER_ADAPTER,
        "mcp-tool-contract",
        (CheckMode.SOUND, CheckMode.BOUNDED),
        (("tool_provider", "mcp"), ("checks", ("tool-schema-ingestion", "tool-serialization"))),
    ),
    FirstPartyPluginSpec(
        "z3",
        PluginCapabilityKind.SOLVER_ENCODING,
        "z3-finite-contract-encoding",
        (CheckMode.SOUND, CheckMode.BOUNDED, CheckMode.Z3_BACKED_SMT),
        (("check", "static-contracts"), ("fallback", "finite-enumeration")),
    ),
)

FIRST_PARTY_REFERENCE_SCHEMES: tuple[str, ...] = (
    "openai",
    "vllm",
    "llamacpp",
    "langchain",
    "llamaindex",
    "outlines",
    "xgrammar",
    "llguidance",
    "pydantic",
    "mcp",
    "z3",
)

_SCHEME_TO_FAMILY = {
    "openai": "openai-compatible",
    "vllm": "vllm",
    "llamacpp": "llama.cpp",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "outlines": "outlines",
    "xgrammar": "xgrammar",
    "llguidance": "llguidance",
    "pydantic": "pydantic",
    "mcp": "mcp",
    "z3": "z3",
}


def create_first_party_plugin_registry() -> PluginRegistry:
    """Return a registry containing PromptABI's bundled web-free adapters."""

    registry = PluginRegistry()
    register_promptabi_plugin(registry)
    return registry


def register_promptabi_plugin(registry: PluginRegistry) -> None:
    """Register bundled adapters for common provider, grammar, framework, and solver families."""

    for spec in FIRST_PARTY_PLUGIN_SPECS:
        _register_capability_once(registry, spec)

    if not any(loader.name == "first-party-reference-loader" for loader in registry.artifact_loaders):
        registry.register_artifact_loader(
            "first-party-reference-loader",
            _load_first_party_reference,
            artifact_kinds=tuple(ArtifactKind),
            uri_schemes=FIRST_PARTY_REFERENCE_SCHEMES,
            priority=10,
            plugin=FIRST_PARTY_PLUGIN_NAME,
            version=__version__,
        )

    if "first-party-plugin-coverage" not in registry.checks:
        registry.register_check(
            "first-party-plugin-coverage",
            _coverage_check(registry),
            artifact_kinds=tuple(ArtifactKind),
            resources=("first-party-plugin-registry",),
            modes=(CheckMode.HEURISTIC,),
            plugin=FIRST_PARTY_PLUGIN_NAME,
            version=__version__,
        )

    if "plugins-json" not in registry.renderers:
        registry.register_renderer(
            "plugins-json",
            lambda result: json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            media_type="application/json",
            plugin=FIRST_PARTY_PLUGIN_NAME,
            version=__version__,
        )


def render_plugin_capabilities(registry: PluginRegistry, *, output_format: str = "text") -> str:
    """Render registered plugin capabilities for CLI inspection."""

    capabilities = sorted(
        (capability.to_dict() for capability in registry.capabilities),
        key=lambda item: (str(item["kind"]), str(item["plugin"]), str(item["name"])),
    )
    if output_format == "json":
        return json.dumps({"capabilities": capabilities}, indent=2, sort_keys=True) + "\n"
    if output_format != "text":
        raise ValueError("output_format must be one of: text, json")

    lines = ["PromptABI plugin capabilities:"]
    for kind in sorted({str(capability["kind"]) for capability in capabilities}):
        lines.append(f"{kind}:")
        for capability in [item for item in capabilities if item["kind"] == kind]:
            modes = ", ".join(capability.get("modes", ())) or "metadata"
            lines.append(f"  {capability['name']} ({capability['plugin']}; {modes})")
    return "\n".join(lines) + "\n"


def _register_capability_once(registry: PluginRegistry, spec: FirstPartyPluginSpec) -> None:
    plugin = f"{FIRST_PARTY_PLUGIN_NAME}.{spec.family}"
    if any(
        capability.kind == spec.kind
        and capability.name == spec.name
        and capability.plugin == plugin
        for capability in registry.capabilities
    ):
        return
    registry.register_capability(
        spec.kind,
        spec.name,
        plugin=plugin,
        version=__version__,
        modes=spec.modes,
        properties=dict(spec.properties),
    )


def _load_first_party_reference(artifact: Artifact) -> LoadedArtifact | None:
    uri = artifact.location.uri
    if uri is None:
        return None
    parsed = urlparse(uri)
    family = _SCHEME_TO_FAMILY.get(parsed.scheme)
    if family is None:
        return None

    query = parse_qs(parsed.query)
    pin = (
        artifact.provenance.sha256
        or artifact.provenance.revision
        or artifact.provenance.version
        or _first_query_value(query, "sha256")
        or _first_query_value(query, "revision")
        or _first_query_value(query, "version")
    )
    metadata: list[tuple[str, object]] = [
        ("first_party_plugin", family),
        ("network", "never"),
        ("uri_scheme", parsed.scheme),
    ]
    if parsed.netloc:
        metadata.append(("authority", parsed.netloc))
    if parsed.path and parsed.path != "/":
        metadata.append(("reference_path", parsed.path.lstrip("/")))
    if pin is not None:
        metadata.append(("pin", pin))

    warnings: tuple[ArtifactLoadWarning, ...] = ()
    if pin is None:
        warnings = (
            ArtifactLoadWarning(
                rule_id="artifact-unpinned",
                message=f"first-party {family} artifact '{artifact.name}' is not pinned",
                suggestion="Add ?version=, ?revision=, ?sha256=, or artifact provenance before using this reference in CI.",
                steps=(("parse first-party URI pin", uri, "missing"),),
            ),
        )

    return LoadedArtifact(
        artifact=artifact,
        source_type=f"first-party-{family}-reference",
        pinned=pin is not None,
        resolved=False,
        metadata=tuple(metadata),
        warnings=warnings,
    )


def _coverage_check(registry: PluginRegistry):
    def run_first_party_coverage(context) -> tuple[Diagnostic, ...]:
        by_kind = Counter(capability.kind.value for capability in registry.capabilities)
        by_family = Counter(
            capability.plugin.removeprefix(f"{FIRST_PARTY_PLUGIN_NAME}.")
            for capability in registry.capabilities
            if capability.plugin.startswith(FIRST_PARTY_PLUGIN_NAME)
        )
        loaded_families = tuple(
            sorted(
                str(value)
                for artifact in context.loaded_artifacts
                for key, value in artifact.metadata
                if key == "first_party_plugin"
            )
        )
        steps = [
            WitnessStep(
                action="register first-party capability families",
                output=", ".join(sorted(by_family)) or "none",
            ),
            WitnessStep(
                action="register extension kinds",
                output=", ".join(f"{kind}={count}" for kind, count in sorted(by_kind.items())),
            ),
        ]
        if loaded_families:
            steps.append(
                WitnessStep(
                    action="load first-party reference artifacts",
                    output=", ".join(loaded_families),
                )
            )
        return (
            Diagnostic(
                rule_id="first-party-plugin-coverage",
                severity=DiagnosticSeverity.INFO,
                message=(
                    f"PromptABI registered {sum(by_family.values())} bundled capabilities "
                    f"across {len(by_family)} first-party plugin families."
                ),
                check_modes=(CheckMode.HEURISTIC,),
                witness=WitnessTrace(
                    summary="The bundled plugin pack is available without network access or heavyweight backend imports.",
                    steps=tuple(steps),
                ),
                properties=(
                    ("capability_count", sum(by_family.values())),
                    ("families", tuple(sorted(by_family))),
                    ("loaded_reference_families", loaded_families),
                ),
            ),
        )

    return run_first_party_coverage


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0]
    return value or None
