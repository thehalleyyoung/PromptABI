"""Tool-call serialization compatibility checks over recorded provider contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, ProviderConfigArtifact, StopPolicyArtifact
from .diagnostics import SourceSpan
from .loaders import LoadedArtifact
from .source import build_json_source_map


class ToolSerializationFindingKind(StrEnum):
    """Concrete disagreements in a bounded tool-call serialization contract."""

    TOOL_NAME_MISMATCH = "tool-name-mismatch"
    ARGUMENT_ENCODING_MISMATCH = "argument-encoding-mismatch"
    ARGUMENT_ESCAPING_RISK = "argument-escaping-risk"
    TOOL_ID_MISMATCH = "tool-id-mismatch"
    PARALLEL_CALL_MISMATCH = "parallel-call-mismatch"
    STREAMING_CHUNK_MISMATCH = "streaming-chunk-mismatch"
    TEMPLATE_TOOL_MISMATCH = "template-tool-mismatch"
    STOP_SERIALIZATION_MISMATCH = "stop-serialization-mismatch"


@dataclass(frozen=True, slots=True)
class ToolSerializationFinding:
    """One provider/template/parser/tool serialization incompatibility."""

    kind: ToolSerializationFindingKind
    message: str
    severity: str
    provider_name: str | None = None
    tool_artifact_name: str | None = None
    template_name: str | None = None
    stop_policy_name: str | None = None
    span: SourceSpan | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    suggestion: str = "Align the provider fixture, tool schema, template, and parser contract before deployment."


@dataclass(frozen=True, slots=True)
class ToolSerializationReport:
    """Bounded analysis result for the selected tool-call serialization stack."""

    findings: tuple[ToolSerializationFinding, ...]
    checked_pairs: int
    providers_checked: tuple[str, ...]
    tool_artifacts_checked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ToolBundle:
    name: str
    tool_names: tuple[str, ...]
    argument_encodings: tuple[str, ...]
    envelope_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProviderContract:
    artifact_name: str
    provider: str
    request_tool_names: tuple[str, ...]
    response_tool_names: tuple[str, ...]
    parser_tool_names: tuple[str, ...]
    request_argument_encoding: str | None
    response_argument_encoding: str | None
    parser_argument_encoding: str | None
    argument_escaping: str | None
    response_id_path: str | None
    parser_requires_id: bool
    supports_parallel: bool | None
    observed_parallel: bool
    parser_allows_parallel: bool | None
    streaming_argument_fragments: bool
    parser_streaming_mode: str | None
    tool_call_stop_sequence: str | None
    span_by_field: dict[str, SourceSpan]


def analyze_tool_call_serialization(
    loaded_artifacts: tuple[LoadedArtifact, ...],
) -> ToolSerializationReport:
    """Compare tool definitions with recorded provider, template, stop, and parser shapes.

    The check is deliberately bounded: provider snapshots must record concrete
    request/response/parser serialization facts. PromptABI reports exact
    disagreements in that fixture rather than inferring live API behavior.
    """

    tools = tuple(_tool_bundle(loaded) for loaded in loaded_artifacts if _is_tool_schema(loaded))
    providers = tuple(_provider_contract(loaded) for loaded in loaded_artifacts if _is_provider_snapshot(loaded))
    templates = tuple(loaded for loaded in loaded_artifacts if loaded.artifact.kind is ArtifactKind.CHAT_TEMPLATE)
    stop_policies = tuple(
        loaded.artifact for loaded in loaded_artifacts if isinstance(loaded.artifact, StopPolicyArtifact)
    )
    findings: list[ToolSerializationFinding] = []

    for tool in tools:
        for provider in providers:
            findings.extend(_compare_tool_provider(tool, provider, stop_policies))

    if tools:
        findings.extend(_compare_templates(tools, templates))

    return ToolSerializationReport(
        findings=tuple(sorted(findings, key=lambda item: (item.severity, item.kind.value, item.message))),
        checked_pairs=len(tools) * len(providers),
        providers_checked=tuple(sorted(provider.artifact_name for provider in providers)),
        tool_artifacts_checked=tuple(sorted(tool.name for tool in tools)),
    )


def _compare_tool_provider(
    tool: _ToolBundle,
    provider: _ProviderContract,
    stop_policies: tuple[StopPolicyArtifact, ...],
) -> list[ToolSerializationFinding]:
    findings: list[ToolSerializationFinding] = []
    declared_names = set(tool.tool_names)
    for field, names in (
        ("request tool names", provider.request_tool_names),
        ("response tool names", provider.response_tool_names),
        ("parser accepted tool names", provider.parser_tool_names),
    ):
        if not names:
            continue
        missing = tuple(sorted(declared_names.difference(names)))
        unknown = tuple(sorted(set(names).difference(declared_names)))
        if missing or unknown:
            findings.append(
                ToolSerializationFinding(
                    kind=ToolSerializationFindingKind.TOOL_NAME_MISMATCH,
                    severity="error",
                    provider_name=provider.artifact_name,
                    tool_artifact_name=tool.name,
                    span=provider.span_by_field.get(field),
                    message=(
                        f"provider '{provider.artifact_name}' {field} disagree with tool-definition "
                        f"'{tool.name}'"
                    ),
                    evidence=(
                        ("tool-definition names", ", ".join(tool.tool_names) or "<none>"),
                        (field, ", ".join(names) or "<none>"),
                        ("missing from provider/parser", ", ".join(missing) or "<none>"),
                        ("unknown to tool schema", ", ".join(unknown) or "<none>"),
                    ),
                    suggestion="Use the same canonical tool names in request tools, response tool calls, and the downstream parser allowlist.",
                )
            )

    expected_encodings = tuple(
        encoding
        for encoding in (
            provider.request_argument_encoding,
            provider.response_argument_encoding,
            provider.parser_argument_encoding,
        )
        if encoding
    )
    for encoding in expected_encodings:
        if tool.argument_encodings and encoding not in tool.argument_encodings:
            findings.append(
                ToolSerializationFinding(
                    kind=ToolSerializationFindingKind.ARGUMENT_ENCODING_MISMATCH,
                    severity="error",
                    provider_name=provider.artifact_name,
                    tool_artifact_name=tool.name,
                    span=provider.span_by_field.get("argument encoding"),
                    message=(
                        f"provider '{provider.artifact_name}' expects {encoding} tool arguments, "
                        f"but '{tool.name}' is normalized as {', '.join(tool.argument_encodings)}"
                    ),
                    evidence=(
                        ("tool-definition argument encodings", ", ".join(tool.argument_encodings)),
                        ("provider/parser argument encoding", encoding),
                    ),
                    suggestion="Normalize whether tool arguments are embedded JSON objects or JSON-encoded strings at every boundary.",
                )
            )
            break

    if provider.response_argument_encoding == "json-string" and provider.parser_argument_encoding == "json-object":
        findings.append(
            ToolSerializationFinding(
                kind=ToolSerializationFindingKind.ARGUMENT_ENCODING_MISMATCH,
                severity="error",
                provider_name=provider.artifact_name,
                tool_artifact_name=tool.name,
                span=provider.span_by_field.get("parser argument encoding"),
                message=(
                    f"provider '{provider.artifact_name}' records JSON-string tool-call arguments "
                    "but the parser contract expects JSON objects"
                ),
                evidence=(
                    ("response arguments", "json-string"),
                    ("parser arguments", "json-object"),
                ),
                suggestion="Parse provider argument strings before application parsing, or configure the parser contract to accept JSON strings.",
            )
        )

    if provider.response_argument_encoding == "json-string" and provider.argument_escaping == "raw":
        findings.append(
            ToolSerializationFinding(
                kind=ToolSerializationFindingKind.ARGUMENT_ESCAPING_RISK,
                severity="warning",
                provider_name=provider.artifact_name,
                tool_artifact_name=tool.name,
                span=provider.span_by_field.get("argument escaping"),
                message=f"provider '{provider.artifact_name}' records raw JSON-string tool arguments without an escaping guarantee",
                evidence=(("argument escaping", "raw"), ("response arguments", "json-string")),
                suggestion="Record or add a JSON escaping layer before embedding argument strings in provider/tool envelopes.",
            )
        )

    if provider.parser_requires_id and provider.response_id_path is None:
        findings.append(
            ToolSerializationFinding(
                kind=ToolSerializationFindingKind.TOOL_ID_MISMATCH,
                severity="error",
                provider_name=provider.artifact_name,
                tool_artifact_name=tool.name,
                span=provider.span_by_field.get("tool call id"),
                message=f"parser for provider '{provider.artifact_name}' requires tool-call IDs, but the recorded response shape has no ID path",
                evidence=(("parser requires ID", "true"), ("response ID path", "<missing>")),
                suggestion="Preserve provider tool-call IDs through streaming assembly and parser handoff, or make the parser not require IDs.",
            )
        )

    if (provider.supports_parallel or provider.observed_parallel) and provider.parser_allows_parallel is False:
        findings.append(
            ToolSerializationFinding(
                kind=ToolSerializationFindingKind.PARALLEL_CALL_MISMATCH,
                severity="error",
                provider_name=provider.artifact_name,
                tool_artifact_name=tool.name,
                span=provider.span_by_field.get("parallel tool calls"),
                message=f"provider '{provider.artifact_name}' can emit parallel tool calls, but the parser contract is single-call",
                evidence=(
                    ("provider supports parallel", str(bool(provider.supports_parallel)).lower()),
                    ("observed parallel fixture", str(provider.observed_parallel).lower()),
                    ("parser allows parallel", "false"),
                ),
                suggestion="Disable provider parallel tool calls or update the parser to accept arrays of tool calls deterministically.",
            )
        )

    if provider.streaming_argument_fragments and provider.parser_streaming_mode != "assemble-chunks":
        findings.append(
            ToolSerializationFinding(
                kind=ToolSerializationFindingKind.STREAMING_CHUNK_MISMATCH,
                severity="error",
                provider_name=provider.artifact_name,
                tool_artifact_name=tool.name,
                span=provider.span_by_field.get("streaming mode"),
                message=f"provider '{provider.artifact_name}' streams argument fragments, but the parser contract does not assemble chunks",
                evidence=(
                    ("streaming argument fragments", "true"),
                    ("parser streaming mode", provider.parser_streaming_mode or "<missing>"),
                ),
                suggestion="Buffer streaming deltas by tool-call index and ID before parsing arguments.",
            )
        )

    if provider.tool_call_stop_sequence:
        for stop_policy in stop_policies:
            if provider.tool_call_stop_sequence not in stop_policy.stop_sequences:
                findings.append(
                    ToolSerializationFinding(
                        kind=ToolSerializationFindingKind.STOP_SERIALIZATION_MISMATCH,
                        severity="warning",
                        provider_name=provider.artifact_name,
                        tool_artifact_name=tool.name,
                        stop_policy_name=stop_policy.name,
                        span=provider.span_by_field.get("tool call stop sequence"),
                        message=(
                            f"provider '{provider.artifact_name}' declares tool-call delimiter "
                            f"{provider.tool_call_stop_sequence!r}, but stop policy '{stop_policy.name}' does not include it"
                        ),
                        evidence=(
                            ("provider tool-call delimiter", provider.tool_call_stop_sequence),
                            ("stop policy sequences", ", ".join(stop_policy.stop_sequences) or "<none>"),
                        ),
                        suggestion="Keep provider tool-call delimiters and configured stop sequences synchronized, or document why truncation is parser-controlled.",
                    )
                )
    return findings


def _compare_templates(
    tools: tuple[_ToolBundle, ...],
    templates: tuple[LoadedArtifact, ...],
) -> list[ToolSerializationFinding]:
    findings: list[ToolSerializationFinding] = []
    for template in templates:
        metadata = dict(template.metadata)
        if metadata.get("uses_tools") is False:
            for tool in tools:
                findings.append(
                    ToolSerializationFinding(
                        kind=ToolSerializationFindingKind.TEMPLATE_TOOL_MISMATCH,
                        severity="warning",
                        template_name=template.artifact.name,
                        tool_artifact_name=tool.name,
                        span=template.artifact.source_span,
                        message=(
                            f"chat template '{template.artifact.name}' does not render tools, "
                            f"but tool-definition '{tool.name}' is selected"
                        ),
                        evidence=(
                            ("template uses_tools", "false"),
                            ("tool-definition names", ", ".join(tool.tool_names) or "<none>"),
                        ),
                        suggestion="Use a tool-aware chat template or avoid advertising tools for this provider/model path.",
                    )
                )
    return findings


def _tool_bundle(loaded: LoadedArtifact) -> _ToolBundle:
    metadata = dict(loaded.metadata)
    return _ToolBundle(
        name=loaded.artifact.name,
        tool_names=_tuple_value(metadata.get("tool_names")),
        argument_encodings=_tuple_value(metadata.get("argument_encodings")),
        envelope_kinds=_tuple_value(metadata.get("envelope_kinds")),
    )


def _provider_contract(loaded: LoadedArtifact) -> _ProviderContract:
    artifact = loaded.artifact
    assert isinstance(artifact, ProviderConfigArtifact)
    raw, span_by_field = _read_provider_snapshot(Path(artifact.location.path)) if artifact.location.path else ({}, {})
    contract = raw.get("tool_serialization", raw)
    if not isinstance(contract, dict):
        contract = {}
    request = _mapping(contract.get("request")) or _mapping(raw.get("request_shape"))
    response = _mapping(contract.get("response")) or _mapping(raw.get("response_shape"))
    parser = _mapping(contract.get("parser")) or _mapping(raw.get("parser_shape"))
    streaming = _mapping(contract.get("streaming")) or _mapping(raw.get("streaming_deltas"))

    return _ProviderContract(
        artifact_name=artifact.name,
        provider=artifact.provider,
        request_tool_names=_string_tuple(request.get("tool_names")),
        response_tool_names=_string_tuple(response.get("tool_names")),
        parser_tool_names=_string_tuple(parser.get("accepted_tool_names") or parser.get("tool_names")),
        request_argument_encoding=_string(request.get("argument_encoding")),
        response_argument_encoding=_string(response.get("argument_encoding")),
        parser_argument_encoding=_string(parser.get("argument_encoding")),
        argument_escaping=_string(response.get("argument_escaping") or contract.get("argument_escaping")),
        response_id_path=_string(response.get("id_path") or response.get("tool_call_id_path")),
        parser_requires_id=_bool(parser.get("require_tool_call_id") or parser.get("requires_tool_call_id")),
        supports_parallel=_optional_bool(
            _first_present(request.get("supports_parallel_tool_calls"), response.get("supports_parallel_tool_calls"))
        ),
        observed_parallel=_bool(response.get("observed_parallel_tool_calls") or contract.get("observed_parallel_tool_calls")),
        parser_allows_parallel=_optional_bool(
            _first_present(parser.get("allow_parallel_tool_calls"), parser.get("allows_parallel_tool_calls"))
        ),
        streaming_argument_fragments=_bool(
            streaming.get("emits_argument_fragments") or streaming.get("argument_fragments")
        ),
        parser_streaming_mode=_string(parser.get("streaming_mode")),
        tool_call_stop_sequence=_string(response.get("tool_call_stop_sequence") or contract.get("tool_call_stop_sequence")),
        span_by_field=span_by_field,
    )


def _read_provider_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, SourceSpan]]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        return {}, {}
    source_map = build_json_source_map(text, path)
    field_paths = {
        "request tool names": ("tool_serialization", "request", "tool_names"),
        "response tool names": ("tool_serialization", "response", "tool_names"),
        "parser accepted tool names": ("tool_serialization", "parser", "accepted_tool_names"),
        "argument encoding": ("tool_serialization", "response", "argument_encoding"),
        "parser argument encoding": ("tool_serialization", "parser", "argument_encoding"),
        "argument escaping": ("tool_serialization", "response", "argument_escaping"),
        "tool call id": ("tool_serialization", "response", "id_path"),
        "parallel tool calls": ("tool_serialization", "parser", "allow_parallel_tool_calls"),
        "streaming mode": ("tool_serialization", "parser", "streaming_mode"),
        "tool call stop sequence": ("tool_serialization", "response", "tool_call_stop_sequence"),
    }
    spans = {
        field: span
        for field, field_path in field_paths.items()
        if (span := source_map.span_for(field_path) or source_map.key_span_for(field_path)) is not None
    }
    return raw, spans


def _is_tool_schema(loaded: LoadedArtifact) -> bool:
    return loaded.artifact.kind is ArtifactKind.TOOL_DEFINITION and loaded.source_type == "tool-definition-schema"


def _is_provider_snapshot(loaded: LoadedArtifact) -> bool:
    return loaded.artifact.kind is ArtifactKind.PROVIDER_CONFIG and loaded.source_type == "provider-config-snapshot"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tuple_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


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


def _bool(value: Any) -> bool:
    return value is True


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
