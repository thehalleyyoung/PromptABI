"""Compatibility matrix for PromptABI check guarantees and artifact surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .artifacts import ArtifactKind
from .diagnostics import CHECK_MODE_DESCRIPTIONS, CheckMode
from .first_party_plugins import FIRST_PARTY_PLUGIN_SPECS
from .plugins import PluginCapabilityKind, PluginRegistry
from .session import CHECK_DEPENDENCIES, CHECK_MODE_CATALOG, VerificationSession


CHECK_RULE_IDS: dict[str, tuple[str, ...]] = {
    "repository-skeleton": ("repository-skeleton",),
    "artifact-provenance": (
        "artifact-provenance-missing-hash",
        "artifact-provenance-missing-license",
        "artifact-provenance-missing-source",
        "artifact-provenance-nonreproducible-remote",
        "artifact-provenance-untrusted-source",
        "artifact-provenance-verified",
    ),
    "enterprise-readiness": (
        "enterprise-internal-fixture-unsafe",
        "enterprise-local-resource-hash-abstained",
        "enterprise-local-resource-hash-mismatch",
        "enterprise-local-resource-missing",
        "enterprise-no-network-violation",
        "enterprise-private-index-untrusted",
        "enterprise-readiness-verified",
        "enterprise-solver-sandbox-incomplete",
        "enterprise-solver-sandbox-unsafe",
    ),
    "role-boundary-nonforgeability": ("role-boundary-abstained", "role-boundary-nonforgeability"),
    "stop-differential": (
        "stop-differential-abstained",
        "stop-differential-agreement",
        "stop-differential-mismatch",
    ),
    "stop-overreachability": ("stop-overreach-abstained", "stop-overreach-content", "stop-overreach-structural"),
    "stop-tokenizer-analysis": (
        "stop-tokenizer-abstained",
        "stop-tokenizer-alignment",
        "stop-tokenizer-ambiguous",
        "stop-tokenizer-collision",
        "stop-tokenizer-special-interaction",
        "stop-tokenizer-unreachable",
    ),
    "grammar-differential": (
        "grammar-differential-abstained",
        "grammar-differential-agreement",
        "grammar-differential-mismatch",
    ),
    "grammar-tokenizer-ambiguity": (
        "grammar-tokenizer-ambiguity",
        "grammar-tokenizer-ambiguity-abstained",
    ),
    "grammar-tokenizer-emptiness": (
        "grammar-tokenizer-abstained",
        "grammar-tokenizer-empty",
        "grammar-tokenizer-satisfiable",
    ),
    "parser-compatibility": (
        "parser-compatibility-abstained",
        "parser-compatibility-agreement",
        "parser-compatibility-mismatch",
    ),
    "provider-fixture-replay": ("provider-fixture-replay",),
    "provider-migration": ("provider-migration",),
    "rag-chunking-compatibility": (
        "rag-chunk-boundary-drift",
        "rag-citation-loss",
        "rag-metadata-inflation",
        "rag-overlap-accounting",
        "rag-payload-truncation",
        "rag-template-overhead",
        "rag-tokenizer-mismatch",
    ),
    "static-contracts": (
        "static-contract-abstained",
        "static-contract-proved",
        "static-contract-unknown",
        "static-contract-violation",
    ),
    "token-budget-model": (
        "token-budget-abstained",
        "token-budget-context-conflict",
        "token-budget-framework-truncation",
        "token-budget-invalid",
        "token-budget-model",
        "token-budget-must-survive",
        "token-budget-policy-overflow",
        "token-budget-required-overflow",
        "token-budget-required-truncated",
        "token-budget-segment-overflow",
        "token-budget-total-overflow",
        "token-budget-truncation-abstained",
    ),
    "tool-schema-ingestion": ("tool-schema-ingestion",),
    "tool-serialization": ("tool-serialization",),
    "training-packing": ("training-packing-boundary", "training-packing-mask", "training-packing-verified"),
    "training-redaction": (
        "training-redaction-hash-missing",
        "training-redaction-policy-missing",
        "training-redaction-raw-witness-field",
        "training-redaction-secret-material",
        "training-redaction-verified",
    ),
    "tokenizer-config-drift": ("tokenizer-drift", "tokenizer-drift-abstained", "tokenizer-drift-clean"),
    "tokenizer-drift": ("tokenizer-drift", "tokenizer-drift-abstained", "tokenizer-drift-clean"),
}


@dataclass(frozen=True, slots=True)
class CompatibilitySurface:
    """One documented artifact family, dialect, backend, provider, or framework surface."""

    axis: str
    name: str
    artifact_kind: ArtifactKind | None = None
    status: str = "covered"
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.axis}:{self.name}"

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "axis": self.axis,
            "name": self.name,
            "status": self.status,
        }
        if self.artifact_kind is not None:
            data["artifact_kind"] = self.artifact_kind.value
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True, slots=True)
class CompatibilityMatrixEntry:
    """A check row with aggregate guarantee modes and the surfaces it applies to."""

    check: str
    rule_ids: tuple[str, ...]
    modes: tuple[CheckMode, ...]
    artifact_kinds: tuple[ArtifactKind, ...]
    surfaces: tuple[CompatibilitySurface, ...]
    after: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    source: str = "built-in"
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "check": self.check,
            "source": self.source,
            "rule_ids": list(self.rule_ids),
            "modes": [mode.value for mode in self.modes],
            "artifact_kinds": [kind.value for kind in self.artifact_kinds],
            "surfaces": [surface.to_dict() for surface in self.surfaces],
        }
        if self.after:
            data["after"] = list(self.after)
        if self.resources:
            data["resources"] = list(self.resources)
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True, slots=True)
class CompatibilityMatrix:
    """Deterministic compatibility matrix consumable by CLI, docs, and CI."""

    entries: tuple[CompatibilityMatrixEntry, ...]
    surfaces: tuple[CompatibilitySurface, ...]

    @property
    def uncovered_surfaces(self) -> tuple[CompatibilitySurface, ...]:
        covered_keys = {surface.key for entry in self.entries for surface in entry.surfaces}
        return tuple(surface for surface in self.surfaces if surface.key not in covered_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "mode_descriptions": {mode.value: CHECK_MODE_DESCRIPTIONS[mode] for mode in CheckMode},
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "uncovered_surfaces": [surface.to_dict() for surface in self.uncovered_surfaces],
        }


def build_compatibility_matrix(
    plugin_registry: PluginRegistry | None = None,
    *,
    include_plugins: bool = True,
) -> CompatibilityMatrix:
    """Build a matrix from real scheduler metadata plus curated surface semantics."""

    registry = plugin_registry if include_plugins else None
    configured = VerificationSession(
        config=_empty_config(),
        plugin_registry=registry,
    )
    entries: list[CompatibilityMatrixEntry] = []
    for check_name in sorted(configured.checks):
        if not include_plugins and check_name not in CHECK_RULE_IDS:
            continue
        dependency = configured.check_dependencies.get(check_name, CHECK_DEPENDENCIES.get(check_name))
        artifact_kinds = dependency.artifact_kinds if dependency is not None else ()
        rule_ids = CHECK_RULE_IDS.get(check_name, (check_name,))
        modes = _aggregate_modes(check_name, rule_ids, configured.check_modes)
        entries.append(
            CompatibilityMatrixEntry(
                check=check_name,
                rule_ids=rule_ids,
                modes=modes,
                artifact_kinds=artifact_kinds,
                surfaces=_surfaces_for_check(check_name, artifact_kinds),
                after=dependency.after if dependency is not None else (),
                resources=dependency.resources if dependency is not None else (),
                source="built-in" if check_name in CHECK_RULE_IDS else "plugin",
                notes=_notes_for_check(check_name),
            )
        )
    surfaces = _surface_catalog()
    entries.sort(key=lambda entry: (entry.source, entry.check))
    return CompatibilityMatrix(entries=tuple(entries), surfaces=surfaces)


def render_compatibility_matrix_text(matrix: CompatibilityMatrix) -> str:
    """Render a compact but complete human-readable compatibility matrix."""

    lines = ["PromptABI compatibility matrix", "modes: " + ", ".join(mode.value for mode in CheckMode), ""]
    current_source = ""
    for entry in matrix.entries:
        if entry.source != current_source:
            current_source = entry.source
            lines.append(f"{current_source}:")
        mode_text = ",".join(mode.value for mode in entry.modes) or "unspecified"
        surface_text = ", ".join(surface.key for surface in entry.surfaces) or "custom/plugin surface"
        rules = ", ".join(entry.rule_ids)
        lines.append(f"  {entry.check} [{mode_text}]")
        lines.append(f"    rules: {rules}")
        lines.append(f"    surfaces: {surface_text}")
        if entry.notes:
            lines.append(f"    notes: {entry.notes}")
    if matrix.uncovered_surfaces:
        lines.append("")
        lines.append("documented surfaces without a dedicated check row:")
        for surface in matrix.uncovered_surfaces:
            note = f" — {surface.notes}" if surface.notes else ""
            lines.append(f"  {surface.key} [{surface.status}]{note}")
    return "\n".join(lines) + "\n"


def render_compatibility_matrix_json(matrix: CompatibilityMatrix) -> str:
    """Render the compatibility matrix as deterministic JSON."""

    return json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n"


def _aggregate_modes(
    check_name: str,
    rule_ids: Iterable[str],
    check_modes: dict[str, tuple[CheckMode, ...]],
) -> tuple[CheckMode, ...]:
    modes: set[CheckMode] = set()
    for rule_id in rule_ids:
        modes.update(check_modes.get(rule_id, CHECK_MODE_CATALOG.get(rule_id, ())))
    modes.update(check_modes.get(check_name, ()))
    return tuple(sorted(modes, key=lambda mode: mode.value))


def _empty_config():
    from .config import VerificationConfig

    return VerificationConfig(name="compatibility-matrix", checks=())


def _surface_catalog() -> tuple[CompatibilitySurface, ...]:
    surfaces = {
        surface.key: surface
        for surface in (
            *_tokenizer_surfaces(),
            *_template_surfaces(),
            *_grammar_surfaces(),
            *_provider_surfaces(),
            *_framework_surfaces(),
            *_training_surfaces(),
            *_enterprise_surfaces(),
        )
    }
    for spec in FIRST_PARTY_PLUGIN_SPECS:
        for surface in _surfaces_from_plugin_spec(spec):
            surfaces.setdefault(surface.key, surface)
    return tuple(sorted(surfaces.values(), key=lambda surface: (surface.axis, surface.name)))


def _surfaces_for_check(check_name: str, artifact_kinds: tuple[ArtifactKind, ...]) -> tuple[CompatibilitySurface, ...]:
    explicit: dict[str, tuple[CompatibilitySurface, ...]] = {
        "enterprise-readiness": (
            _surface("enterprise", "offline-mirrors", None, "covered", "local mirror paths and optional manifest/file digests"),
            _surface("enterprise", "private-artifact-indexes", None, "covered", "local private indexes with trusted-source allowlists"),
            _surface("enterprise", "internal-provider-fixtures", ArtifactKind.PROVIDER_CONFIG, "covered", "redacted local fixture JSON checked with the shared secret scanner"),
            _surface("enterprise", "policy-packs", None, "covered", "JSON policy packs merged into existing severity/suppression policy"),
            _surface("enterprise", "strict-no-network", None, "covered", "remote artifact locations are rejected when enabled"),
            _surface("solver", "sandbox-declaration", None, "covered", "timeout, memory, and network posture are checked declaratively"),
        ),
        "role-boundary-nonforgeability": _template_surfaces(),
        "stop-differential": _provider_surfaces(),
        "stop-overreachability": (*_grammar_surfaces(), *_provider_surfaces()),
        "stop-tokenizer-analysis": _tokenizer_surfaces(),
        "grammar-differential": _grammar_surfaces(),
        "grammar-tokenizer-ambiguity": (*_tokenizer_surfaces(), *_grammar_surfaces()),
        "grammar-tokenizer-emptiness": (*_tokenizer_surfaces(), *_grammar_surfaces()),
        "parser-compatibility": _grammar_surfaces(),
        "provider-fixture-replay": _provider_surfaces(),
        "provider-migration": _provider_surfaces(),
        "rag-chunking-compatibility": (*_tokenizer_surfaces(), *_framework_surfaces()),
        "static-contracts": (
            *_tokenizer_surfaces(),
            *_template_surfaces(),
            *_grammar_surfaces(),
            *_provider_surfaces(),
            *_framework_surfaces(),
            _surface("training", "supervised-jsonl", ArtifactKind.TRAINING_MANIFEST, "covered", "target-role alignment and supervised-span alignment over finite training manifests"),
            _surface("training", "loss-masks", ArtifactKind.TRAINING_MANIFEST, "covered", "observed supervised spans must be selected by the declared loss-mask contract"),
            _surface("training", "packed-datasets", ArtifactKind.TRAINING_MANIFEST, "bounded", "observed supervised spans are checked against declared preserved packing boundaries"),
            _surface("training", "source-leakage", ArtifactKind.TRAINING_MANIFEST, "bounded", "declared transform source ranges cannot place user, tool, retrieval, or preference text into supervised targets"),
            _surface("training", "tokenizer-template-stage-consistency", ArtifactKind.TRAINING_MANIFEST, "covered", "fine-tuning preparation, training, evaluation, and serving tokenizer/template pins are compared by static-contracts"),
            _surface("training", "preference-pairs", ArtifactKind.TRAINING_MANIFEST, "bounded", "chosen/rejected pairs must share prompt prefix, role layout, tokenizer version, masking policy, and packing/truncation invariants"),
        ),
        "training-redaction": (
            _surface("training", "redaction-policy", ArtifactKind.TRAINING_MANIFEST, "covered", "stored witnesses and reports must be structural, hashed, and free of restricted metadata"),
            _surface("training", "source-leakage", ArtifactKind.TRAINING_MANIFEST, "bounded", "source contributions are checked for hash-only evidence and provider-key-like values"),
            _surface("training", "preference-pairs", ArtifactKind.TRAINING_MANIFEST, "bounded", "preference prompt/chosen/rejected evidence is required to be sha256-referenced"),
        ),
        "token-budget-model": (*_tokenizer_surfaces(), *_framework_surfaces()),
        "tool-schema-ingestion": (_surface("provider", "mcp", ArtifactKind.TOOL_DEFINITION),),
        "tool-serialization": (*_provider_surfaces(), *_template_surfaces()),
        "tokenizer-config-drift": _tokenizer_surfaces(),
        "tokenizer-drift": _tokenizer_surfaces(),
    }
    if check_name in explicit:
        return _dedupe_surfaces(explicit[check_name])
    return _dedupe_surfaces(_generic_surfaces_for_artifact_kinds(artifact_kinds))


def _generic_surfaces_for_artifact_kinds(artifact_kinds: tuple[ArtifactKind, ...]) -> tuple[CompatibilitySurface, ...]:
    surfaces: list[CompatibilitySurface] = []
    for kind in artifact_kinds:
        surfaces.append(_surface("artifact", kind.value, kind))
    return tuple(surfaces)


def _surfaces_from_plugin_spec(spec) -> tuple[CompatibilitySurface, ...]:
    props = dict(spec.properties)
    surfaces: list[CompatibilitySurface] = []
    if spec.kind is PluginCapabilityKind.TEMPLATE_DIALECT:
        surfaces.append(_surface("template", str(props.get("template_format", spec.family)), ArtifactKind.CHAT_TEMPLATE))
    if spec.kind is PluginCapabilityKind.GRAMMAR_BACKEND:
        grammar_type = props.get("grammar_type") or props.get("schema_source") or spec.family
        surfaces.append(_surface("grammar", str(grammar_type), ArtifactKind.GRAMMAR))
    if spec.kind is PluginCapabilityKind.PROVIDER_ADAPTER:
        providers = props.get("providers", (spec.family,))
        if isinstance(providers, str):
            providers = (providers,)
        for provider in providers:
            surfaces.append(_surface("provider", str(provider), ArtifactKind.PROVIDER_CONFIG))
    if spec.kind is PluginCapabilityKind.TRUNCATION_POLICY:
        surfaces.append(_surface("framework", str(props.get("framework", spec.family)), ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG))
    if spec.kind is PluginCapabilityKind.SOLVER_ENCODING:
        surfaces.append(_surface("solver", spec.family, None))
    return tuple(surfaces)


def _notes_for_check(check_name: str) -> str:
    return {
        "enterprise-readiness": "declarative enterprise posture check for offline mirrors, private indexes, internal fixtures, policy packs, severity overrides, solver limits, and strict no-network operation",
        "static-contracts": "includes finite SMT obligations for supervised target/message role alignment, observed rendered/tokenized/loss-masked span contracts, source leakage, stage consistency, and preference-pair prefix/layout/tokenizer/mask invariants",
        "training-redaction": "statically ensures training-manifest witnesses and reports use structural or hashed evidence instead of raw secrets, provider keys, restricted metadata, or dataset text",
        "tokenizer-config-drift": "alias retained for configs that select tokenizer drift under the older check name",
        "tokenizer-drift": "alias of tokenizer-config-drift for user-facing compatibility",
    }.get(check_name, "")


def _surface(axis: str, name: str, artifact_kind: ArtifactKind | None = None, status: str = "covered", notes: str = "") -> CompatibilitySurface:
    return CompatibilitySurface(axis=axis, name=name, artifact_kind=artifact_kind, status=status, notes=notes)


def _tokenizer_surfaces() -> tuple[CompatibilitySurface, ...]:
    return (
        _surface("tokenizer", "byte-level", ArtifactKind.TOKENIZER),
        _surface("tokenizer", "huggingface-tokenizers", ArtifactKind.TOKENIZER),
        _surface("tokenizer", "sentencepiece", ArtifactKind.TOKENIZER),
        _surface("tokenizer", "tiktoken", ArtifactKind.TOKENIZER),
    )


def _template_surfaces() -> tuple[CompatibilitySurface, ...]:
    return (
        _surface("template", "huggingface-tokenizer-config", ArtifactKind.CHAT_TEMPLATE),
        _surface("template", "jinja-supported-fragment", ArtifactKind.CHAT_TEMPLATE, "bounded", "unsupported Jinja constructs produce abstentions"),
    )


def _grammar_surfaces() -> tuple[CompatibilitySurface, ...]:
    return (
        _surface("grammar", "ebnf", ArtifactKind.GRAMMAR),
        _surface("grammar", "json-schema", ArtifactKind.SCHEMA),
        _surface("grammar", "llguidance", ArtifactKind.GRAMMAR),
        _surface("grammar", "outlines", ArtifactKind.GRAMMAR),
        _surface("grammar", "promptabi", ArtifactKind.GRAMMAR),
        _surface("grammar", "pydantic-json-schema", ArtifactKind.SCHEMA),
        _surface("grammar", "regex", ArtifactKind.GRAMMAR),
        _surface("grammar", "xgrammar", ArtifactKind.GRAMMAR),
    )


def _provider_surfaces() -> tuple[CompatibilitySurface, ...]:
    return tuple(
        _surface("provider", provider, ArtifactKind.PROVIDER_CONFIG)
        for provider in (
            "anthropic",
            "azure-openai",
            "bedrock",
            "gemini",
            "groq",
            "litellm",
            "llama.cpp-server",
            "mcp",
            "ollama",
            "openai",
            "openai-compatible",
            "together",
            "vllm-openai-server",
        )
    )


def _framework_surfaces() -> tuple[CompatibilitySurface, ...]:
    return tuple(
        _surface("framework", framework, ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG)
        for framework in (
            "custom-rag",
            "langchain",
            "litellm",
            "llama.cpp",
            "llamaindex",
            "openai-compatible",
            "transformers",
            "vllm",
        )
    )


def _training_surfaces() -> tuple[CompatibilitySurface, ...]:
    return (
        _surface("training", "supervised-jsonl", ArtifactKind.TRAINING_MANIFEST, "covered", "finite target-role and supervised-span alignment are checked by static-contracts"),
        _surface("training", "loss-masks", ArtifactKind.TRAINING_MANIFEST, "covered", "observed supervised spans must be selected by the declared loss-mask contract"),
        _surface("training", "packed-datasets", ArtifactKind.TRAINING_MANIFEST, "bounded", "observed supervised spans are checked against declared preserved packing boundaries"),
        _surface("training", "redaction-policy", ArtifactKind.TRAINING_MANIFEST, "covered", "stored witnesses and reports are constrained to structural or hashed training evidence"),
        _surface("training", "tokenizer-template-stage-consistency", ArtifactKind.TRAINING_MANIFEST, "covered", "dataset preparation, training, evaluation, and serving tokenizer/template pins are compared by static-contracts"),
        _surface("training", "preference-pairs", ArtifactKind.TRAINING_MANIFEST, "bounded", "preference-pair prefix equivalence and hash-only report evidence are checked"),
    )


def _enterprise_surfaces() -> tuple[CompatibilitySurface, ...]:
    return (
        _surface("enterprise", "offline-mirrors", None, "covered", "local mirror paths and optional manifest/file digests"),
        _surface("enterprise", "private-artifact-indexes", None, "covered", "local private indexes with trusted-source allowlists"),
        _surface("enterprise", "internal-provider-fixtures", ArtifactKind.PROVIDER_CONFIG, "covered", "redacted local fixture JSON"),
        _surface("enterprise", "policy-packs", None, "covered", "JSON policy packs and severity customization"),
        _surface("enterprise", "strict-no-network", None, "covered", "remote runtime artifact locations rejected when enabled"),
        _surface("solver", "sandbox-declaration", None, "covered", "declarative solver timeout, memory, and network posture"),
    )


def _dedupe_surfaces(surfaces: Iterable[CompatibilitySurface]) -> tuple[CompatibilitySurface, ...]:
    by_key = {surface.key: surface for surface in surfaces}
    return tuple(sorted(by_key.values(), key=lambda surface: (surface.axis, surface.name)))
