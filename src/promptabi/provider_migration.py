"""Provider migration compatibility checks over recorded API fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, ProviderConfigArtifact
from .diagnostics import SourceSpan
from .loaders import LoadedArtifact
from .source import build_json_source_map


class ProviderMigrationFindingKind(StrEnum):
    """Concrete migration incompatibilities found in recorded provider fixtures."""

    UNSUPPORTED_PROVIDER = "unsupported-provider"
    REQUEST_FIELD_LOSS = "request-field-loss"
    RESPONSE_FIELD_LOSS = "response-field-loss"
    TOOL_ARGUMENT_ENCODING_MISMATCH = "tool-argument-encoding-mismatch"
    TOOL_ID_MISMATCH = "tool-id-mismatch"
    PARALLEL_TOOL_CALL_MISMATCH = "parallel-tool-call-mismatch"
    STREAMING_CHUNK_MISMATCH = "streaming-chunk-mismatch"
    STOP_BEHAVIOR_MISMATCH = "stop-behavior-mismatch"
    CONTEXT_LIMIT_REGRESSION = "context-limit-regression"
    STRUCTURED_OUTPUT_MISMATCH = "structured-output-mismatch"
    ERROR_SHAPE_MISMATCH = "error-shape-mismatch"
    ROUTING_TARGET_MISSING = "routing-target-missing"


@dataclass(frozen=True, slots=True)
class ProviderMigrationFinding:
    """One bounded source-provider to target-provider migration incompatibility."""

    kind: ProviderMigrationFindingKind
    message: str
    severity: str
    source_provider: str
    target_provider: str
    source_artifact_name: str
    target_artifact_name: str | None = None
    span: SourceSpan | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    suggestion: str = "Record an explicit compatibility shim or update the migrated provider contract before deployment."


@dataclass(frozen=True, slots=True)
class ProviderMigrationReport:
    """Bounded analysis result for recorded provider migration pairs."""

    findings: tuple[ProviderMigrationFinding, ...]
    migrations_checked: int
    providers_checked: tuple[str, ...]
    supported_targets_seen: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProviderSnapshot:
    artifact_name: str
    provider: str
    canonical_family: str | None
    request_fields: tuple[str, ...]
    response_fields: tuple[str, ...]
    tool_argument_encoding: str | None
    tool_id_path: str | None
    supports_parallel_tools: bool | None
    streams_argument_fragments: bool | None
    stop_sequences: tuple[str, ...]
    max_input_tokens: int | None
    max_output_tokens: int | None
    structured_output_modes: tuple[str, ...]
    error_code_path: str | None
    rate_limit_path: str | None
    routes_to: tuple[str, ...]
    migration_targets: tuple[str, ...]
    span_by_field: dict[str, SourceSpan]


SUPPORTED_PROVIDER_FAMILIES: tuple[str, ...] = (
    "anthropic",
    "azure-openai",
    "bedrock",
    "gemini",
    "groq",
    "litellm",
    "llama.cpp-server",
    "ollama",
    "openai",
    "together",
    "vllm-openai-server",
)

_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "azure": "azure-openai",
    "azure-openai": "azure-openai",
    "azure openai": "azure-openai",
    "azure_openai": "azure-openai",
    "bedrock": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "gemini": "gemini",
    "google-gemini": "gemini",
    "google": "gemini",
    "groq": "groq",
    "litellm": "litellm",
    "lite-llm": "litellm",
    "llama.cpp": "llama.cpp-server",
    "llama.cpp-server": "llama.cpp-server",
    "llama.cpp server": "llama.cpp-server",
    "llamacpp": "llama.cpp-server",
    "ollama": "ollama",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai compatible": "openai",
    "together": "together",
    "together-ai": "together",
    "vllm": "vllm-openai-server",
    "vllm-openai": "vllm-openai-server",
    "vllm-openai-server": "vllm-openai-server",
    "vllm openai server": "vllm-openai-server",
}


def analyze_provider_migration(
    loaded_artifacts: tuple[LoadedArtifact, ...],
) -> ProviderMigrationReport:
    """Compare recorded source and target provider fixtures for migration safety.

    The check is deliberately offline and bounded: each provider artifact records
    concrete request, response, tool-call, streaming, stop, context, structured
    output, routing, and error-shape facts. PromptABI compares only declared
    source->target migration pairs and emits exact fixture disagreements.
    """

    providers = tuple(
        sorted(
            (_provider_snapshot(loaded) for loaded in loaded_artifacts if _is_provider_snapshot(loaded)),
            key=lambda item: item.artifact_name,
        )
    )
    by_name = {provider.artifact_name: provider for provider in providers}
    findings: list[ProviderMigrationFinding] = []
    migrations_checked = 0

    for source in providers:
        for target_name in source.migration_targets:
            migrations_checked += 1
            target = by_name.get(target_name)
            if target is None:
                findings.append(
                    ProviderMigrationFinding(
                        kind=ProviderMigrationFindingKind.ROUTING_TARGET_MISSING,
                        severity="error",
                        source_provider=source.provider,
                        target_provider=target_name,
                        source_artifact_name=source.artifact_name,
                        span=source.span_by_field.get("migration targets"),
                        message=(
                            f"provider fixture '{source.artifact_name}' declares migration target "
                            f"'{target_name}', but no matching provider-config artifact is loaded"
                        ),
                        evidence=(("declared target", target_name), ("loaded providers", ", ".join(sorted(by_name)))),
                        suggestion="Add the target provider fixture to the PromptABI config or remove the migration pair.",
                    )
                )
                continue
            findings.extend(_compare_provider_pair(source, target))

    supported_seen = tuple(
        sorted({provider.canonical_family for provider in providers if provider.canonical_family is not None})
    )
    return ProviderMigrationReport(
        findings=tuple(sorted(findings, key=lambda item: (item.severity, item.kind.value, item.message))),
        migrations_checked=migrations_checked,
        providers_checked=tuple(provider.artifact_name for provider in providers),
        supported_targets_seen=supported_seen,
    )


def compare_provider_config_artifacts(
    baseline: LoadedArtifact,
    current: LoadedArtifact,
) -> tuple[ProviderMigrationFinding, ...]:
    """Compare two loaded provider-config snapshots as a baseline->current diff."""

    if not _is_provider_snapshot(baseline) or not _is_provider_snapshot(current):
        return ()
    findings = _compare_provider_pair(_provider_snapshot(baseline), _provider_snapshot(current))
    return tuple(
        finding
        for finding in findings
        if finding.kind is not ProviderMigrationFindingKind.ROUTING_TARGET_MISSING
    )


def _compare_provider_pair(
    source: _ProviderSnapshot,
    target: _ProviderSnapshot,
) -> tuple[ProviderMigrationFinding, ...]:
    findings: list[ProviderMigrationFinding] = []
    if target.canonical_family is None:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.UNSUPPORTED_PROVIDER,
                "error",
                source,
                target,
                f"target provider '{target.provider}' is not in PromptABI's supported migration catalog",
                (("target provider", target.provider), ("supported families", ", ".join(SUPPORTED_PROVIDER_FAMILIES))),
                "Use one of the supported provider fixture families or add an explicit adapter before migrating.",
                "provider",
            )
        )

    request_loss = _missing(source.request_fields, target.request_fields)
    if request_loss:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.REQUEST_FIELD_LOSS,
                "error",
                source,
                target,
                f"migration from '{source.artifact_name}' to '{target.artifact_name}' drops request fields",
                (
                    ("source request fields", ", ".join(source.request_fields)),
                    ("target request fields", ", ".join(target.request_fields)),
                    ("missing on target", ", ".join(request_loss)),
                ),
                "Add a request adapter for the missing fields or block this migration path.",
                "request fields",
            )
        )

    response_loss = _missing(source.response_fields, target.response_fields)
    if response_loss:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.RESPONSE_FIELD_LOSS,
                "error",
                source,
                target,
                f"migration from '{source.artifact_name}' to '{target.artifact_name}' drops response fields",
                (
                    ("source response fields", ", ".join(source.response_fields)),
                    ("target response fields", ", ".join(target.response_fields)),
                    ("missing on target", ", ".join(response_loss)),
                ),
                "Update the response parser or record a provider-specific response translation layer.",
                "response fields",
            )
        )

    if _both(source.tool_argument_encoding, target.tool_argument_encoding) and (
        source.tool_argument_encoding != target.tool_argument_encoding
    ):
        findings.append(
            _finding(
                ProviderMigrationFindingKind.TOOL_ARGUMENT_ENCODING_MISMATCH,
                "error",
                source,
                target,
                f"tool arguments migrate from {source.tool_argument_encoding} to {target.tool_argument_encoding}",
                (
                    ("source tool argument encoding", source.tool_argument_encoding or "<missing>"),
                    ("target tool argument encoding", target.tool_argument_encoding or "<missing>"),
                ),
                "Normalize tool arguments at the provider boundary before application parsing.",
                "tool argument encoding",
            )
        )

    if bool(source.tool_id_path) != bool(target.tool_id_path):
        findings.append(
            _finding(
                ProviderMigrationFindingKind.TOOL_ID_MISMATCH,
                "error",
                source,
                target,
                "tool-call ID availability changes across the provider migration",
                (
                    ("source tool ID path", source.tool_id_path or "<missing>"),
                    ("target tool ID path", target.tool_id_path or "<missing>"),
                ),
                "Preserve or synthesize stable tool-call IDs before handing calls to downstream parsers.",
                "tool id",
            )
        )

    if source.supports_parallel_tools is True and target.supports_parallel_tools is False:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.PARALLEL_TOOL_CALL_MISMATCH,
                "warning",
                source,
                target,
                "target provider fixture disables parallel tool calls accepted by the source",
                (("source parallel tools", "true"), ("target parallel tools", "false")),
                "Disable source parallel calls before migration or make the target route explicitly single-call.",
                "parallel tool calls",
            )
        )

    if source.streams_argument_fragments != target.streams_argument_fragments and (
        source.streams_argument_fragments is not None and target.streams_argument_fragments is not None
    ):
        findings.append(
            _finding(
                ProviderMigrationFindingKind.STREAMING_CHUNK_MISMATCH,
                "error",
                source,
                target,
                "streaming tool-call argument chunking changes across the provider migration",
                (
                    ("source streams argument fragments", str(source.streams_argument_fragments).lower()),
                    ("target streams argument fragments", str(target.streams_argument_fragments).lower()),
                ),
                "Record a streaming assembler/parser contract for the migrated provider path.",
                "streaming",
            )
        )

    if set(source.stop_sequences) != set(target.stop_sequences):
        findings.append(
            _finding(
                ProviderMigrationFindingKind.STOP_BEHAVIOR_MISMATCH,
                "warning",
                source,
                target,
                "provider stop-sequence behavior differs across the migration",
                (
                    ("source stop sequences", ", ".join(source.stop_sequences) or "<none>"),
                    ("target stop sequences", ", ".join(target.stop_sequences) or "<none>"),
                ),
                "Pin and replay stop fixtures for the migrated provider before enabling traffic.",
                "stop sequences",
            )
        )

    if _regresses_limit(source.max_input_tokens, target.max_input_tokens) or _regresses_limit(
        source.max_output_tokens, target.max_output_tokens
    ):
        findings.append(
            _finding(
                ProviderMigrationFindingKind.CONTEXT_LIMIT_REGRESSION,
                "error",
                source,
                target,
                "target provider fixture has a smaller recorded context or output limit",
                (
                    ("source max input tokens", _int_text(source.max_input_tokens)),
                    ("target max input tokens", _int_text(target.max_input_tokens)),
                    ("source max output tokens", _int_text(source.max_output_tokens)),
                    ("target max output tokens", _int_text(target.max_output_tokens)),
                ),
                "Run budget survival checks with the target limits or reject the migration for prompts near the boundary.",
                "context limits",
            )
        )

    structured_loss = _missing(source.structured_output_modes, target.structured_output_modes)
    if structured_loss:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.STRUCTURED_OUTPUT_MISMATCH,
                "error",
                source,
                target,
                "target provider fixture lacks structured-output modes used by the source",
                (
                    ("source structured modes", ", ".join(source.structured_output_modes)),
                    ("target structured modes", ", ".join(target.structured_output_modes)),
                    ("missing modes", ", ".join(structured_loss)),
                ),
                "Use a target provider structured-output mode with equivalent parser semantics, or add a compatibility parser.",
                "structured output modes",
            )
        )

    if _both(source.error_code_path, target.error_code_path) and source.error_code_path != target.error_code_path:
        findings.append(
            _finding(
                ProviderMigrationFindingKind.ERROR_SHAPE_MISMATCH,
                "warning",
                source,
                target,
                "provider error-code paths differ across the migration",
                (
                    ("source error code path", source.error_code_path or "<missing>"),
                    ("target error code path", target.error_code_path or "<missing>"),
                    ("source rate-limit path", source.rate_limit_path or "<missing>"),
                    ("target rate-limit path", target.rate_limit_path or "<missing>"),
                ),
                "Update retry, rate-limit, and observability code to parse the target provider error envelope.",
                "error shape",
            )
        )

    if target.canonical_family == "litellm":
        missing_routes = tuple(
            family
            for family in sorted({source.canonical_family or source.provider})
            if family not in target.routes_to
        )
        if missing_routes:
            findings.append(
                _finding(
                    ProviderMigrationFindingKind.ROUTING_TARGET_MISSING,
                    "error",
                    source,
                    target,
                    "LiteLLM target fixture does not route to the source provider family",
                    (
                        ("required route", ", ".join(missing_routes)),
                        ("LiteLLM routes", ", ".join(target.routes_to) or "<none>"),
                    ),
                    "Record the LiteLLM model route for this migration before relying on proxy compatibility.",
                    "routes",
                )
            )
    return tuple(findings)


def _finding(
    kind: ProviderMigrationFindingKind,
    severity: str,
    source: _ProviderSnapshot,
    target: _ProviderSnapshot,
    message: str,
    evidence: tuple[tuple[str, str], ...],
    suggestion: str,
    span_key: str,
) -> ProviderMigrationFinding:
    return ProviderMigrationFinding(
        kind=kind,
        severity=severity,
        source_provider=source.provider,
        target_provider=target.provider,
        source_artifact_name=source.artifact_name,
        target_artifact_name=target.artifact_name,
        span=target.span_by_field.get(span_key) or source.span_by_field.get("migration targets"),
        message=message,
        evidence=evidence,
        suggestion=suggestion,
    )


def _provider_snapshot(loaded: LoadedArtifact) -> _ProviderSnapshot:
    artifact = loaded.artifact
    assert isinstance(artifact, ProviderConfigArtifact)
    raw, spans = _read_provider_snapshot(Path(artifact.location.path)) if artifact.location.path else ({}, {})
    compatibility = _mapping(raw.get("migration_compatibility"))
    request = _mapping(compatibility.get("request")) or _mapping(raw.get("request_shape"))
    response = _mapping(compatibility.get("response")) or _mapping(raw.get("response_shape"))
    tools = _mapping(compatibility.get("tools")) or _mapping(raw.get("tool_serialization"))
    streaming = _mapping(compatibility.get("streaming")) or _mapping(raw.get("streaming_deltas"))
    stops = _mapping(compatibility.get("stops"))
    limits = _mapping(compatibility.get("limits"))
    structured = _mapping(compatibility.get("structured_outputs"))
    errors = _mapping(compatibility.get("errors"))
    routing = _mapping(compatibility.get("routing"))
    migration = _mapping(raw.get("provider_migration"))
    provider = _string(raw.get("provider")) or artifact.provider

    canonical = canonical_provider_family(
        _string(compatibility.get("provider_family"))
        or _string(raw.get("api_family"))
        or artifact.api_family
        or provider
    )
    return _ProviderSnapshot(
        artifact_name=artifact.name,
        provider=provider,
        canonical_family=canonical,
        request_fields=_string_tuple(request.get("required_fields") or request.get("fields")),
        response_fields=_string_tuple(response.get("required_fields") or response.get("fields")),
        tool_argument_encoding=_string(tools.get("argument_encoding")),
        tool_id_path=_string(tools.get("id_path") or tools.get("tool_call_id_path")),
        supports_parallel_tools=_optional_bool(tools.get("supports_parallel_tool_calls")),
        streams_argument_fragments=_optional_bool(streaming.get("emits_argument_fragments")),
        stop_sequences=_string_tuple(stops.get("sequences") or raw.get("stop")),
        max_input_tokens=_int(limits.get("max_input_tokens")),
        max_output_tokens=_int(limits.get("max_output_tokens")),
        structured_output_modes=_string_tuple(structured.get("modes")),
        error_code_path=_string(errors.get("code_path")),
        rate_limit_path=_string(errors.get("rate_limit_path")),
        routes_to=tuple(
            sorted(
                canonical
                for value in _string_tuple(routing.get("routes_to"))
                if (canonical := canonical_provider_family(value)) is not None
            )
        ),
        migration_targets=_string_tuple(migration.get("targets")),
        span_by_field=spans,
    )


def canonical_provider_family(value: str | None) -> str | None:
    """Return PromptABI's canonical provider family name for aliases."""

    if value is None:
        return None
    normalized = " ".join(value.replace("_", "-").strip().lower().split())
    normalized = normalized.replace(" ", "-") if normalized in _PROVIDER_ALIASES else normalized
    return _PROVIDER_ALIASES.get(normalized)


def _read_provider_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, SourceSpan]]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        return {}, {}
    source_map = build_json_source_map(text, path)
    field_paths = {
        "provider": ("provider",),
        "migration targets": ("provider_migration", "targets"),
        "request fields": ("migration_compatibility", "request", "required_fields"),
        "response fields": ("migration_compatibility", "response", "required_fields"),
        "tool argument encoding": ("migration_compatibility", "tools", "argument_encoding"),
        "tool id": ("migration_compatibility", "tools", "id_path"),
        "parallel tool calls": ("migration_compatibility", "tools", "supports_parallel_tool_calls"),
        "streaming": ("migration_compatibility", "streaming", "emits_argument_fragments"),
        "stop sequences": ("migration_compatibility", "stops", "sequences"),
        "context limits": ("migration_compatibility", "limits"),
        "structured output modes": ("migration_compatibility", "structured_outputs", "modes"),
        "error shape": ("migration_compatibility", "errors", "code_path"),
        "routes": ("migration_compatibility", "routing", "routes_to"),
    }
    spans = {
        field: span
        for field, field_path in field_paths.items()
        if (span := source_map.span_for(field_path) or source_map.key_span_for(field_path)) is not None
    }
    return raw, spans


def _is_provider_snapshot(loaded: LoadedArtifact) -> bool:
    return loaded.artifact.kind is ArtifactKind.PROVIDER_CONFIG and loaded.source_type == "provider-config-snapshot"


def _missing(source_values: tuple[str, ...], target_values: tuple[str, ...]) -> tuple[str, ...]:
    if not source_values or not target_values:
        return ()
    return tuple(sorted(set(source_values).difference(target_values)))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(sorted(dict.fromkeys(value)))
    if isinstance(value, tuple) and all(isinstance(item, str) and item for item in value):
        return tuple(sorted(dict.fromkeys(value)))
    return ()


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _both(left: str | None, right: str | None) -> bool:
    return bool(left) and bool(right)


def _regresses_limit(source: int | None, target: int | None) -> bool:
    return source is not None and target is not None and target < source


def _int_text(value: int | None) -> str:
    return str(value) if value is not None else "<missing>"
