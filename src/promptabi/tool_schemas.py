"""Tool/function schema ingestion for provider and framework tool definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .diagnostics import SourceSpan
from .source import JsonSourceMap


class ToolSchemaProvider(StrEnum):
    """Normalized families for tool/function schema envelopes."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LANGCHAIN = "langchain"
    PYDANTIC = "pydantic"
    TYPESCRIPT = "typescript"
    MCP = "mcp"
    PROVIDER_ENVELOPE = "provider-envelope"
    GENERIC = "generic"


class ToolSchemaIssueKind(StrEnum):
    """Non-fatal facts discovered while ingesting a tool schema bundle."""

    MISSING_DESCRIPTION = "missing-description"
    NON_OBJECT_PARAMETERS = "non-object-parameters"
    OPEN_PARAMETERS = "open-parameters"
    ARGUMENT_STRING_ENVELOPE = "argument-string-envelope"
    STREAMING_CHUNK_ENVELOPE = "streaming-chunk-envelope"


@dataclass(frozen=True, slots=True)
class ToolSchemaIssue:
    """A deterministic ingestion issue with an optional source span."""

    kind: ToolSchemaIssueKind
    message: str
    path: tuple[str, ...] = ()
    span: SourceSpan | None = None

    def to_metadata(self, index: int) -> tuple[tuple[str, object], ...]:
        return (
            (f"issue_{index}_kind", self.kind.value),
            (f"issue_{index}_message", self.message),
            (f"issue_{index}_path", ".".join(self.path)),
        )


@dataclass(frozen=True, slots=True)
class ToolParameterSummary:
    """Summary of one JSON-schema-like tool parameter object."""

    property_names: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    schema_type: str | None = None
    additional_properties: bool | None = None
    enum_paths: tuple[str, ...] = ()
    constraint_paths: tuple[str, ...] = ()

    @property
    def closed(self) -> bool:
        return self.additional_properties is False

    def to_metadata(self, prefix: str) -> tuple[tuple[str, object], ...]:
        data: list[tuple[str, object]] = [
            (f"{prefix}_property_count", len(self.property_names)),
            (f"{prefix}_properties", self.property_names),
            (f"{prefix}_required", self.required),
        ]
        if self.schema_type is not None:
            data.append((f"{prefix}_schema_type", self.schema_type))
        if self.additional_properties is not None:
            data.append((f"{prefix}_additional_properties", self.additional_properties))
        if self.enum_paths:
            data.append((f"{prefix}_enum_paths", self.enum_paths))
        if self.constraint_paths:
            data.append((f"{prefix}_constraint_paths", self.constraint_paths))
        return tuple(data)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One normalized tool/function definition."""

    name: str
    provider: ToolSchemaProvider
    parameter_schema: ToolParameterSummary
    source_path: tuple[str, ...]
    description: str | None = None
    envelope: str | None = None
    argument_encoding: str = "json-object"
    tool_id_path: tuple[str, ...] = ()
    span: SourceSpan | None = None

    def to_metadata(self, index: int) -> tuple[tuple[str, object], ...]:
        prefix = f"tool_{index}"
        data: list[tuple[str, object]] = [
            (f"{prefix}_name", self.name),
            (f"{prefix}_provider", self.provider.value),
            (f"{prefix}_source_path", ".".join(self.source_path)),
            (f"{prefix}_argument_encoding", self.argument_encoding),
        ]
        if self.description is not None:
            data.append((f"{prefix}_has_description", True))
        if self.envelope is not None:
            data.append((f"{prefix}_envelope", self.envelope))
        if self.tool_id_path:
            data.append((f"{prefix}_tool_id_path", ".".join(self.tool_id_path)))
        data.extend(self.parameter_schema.to_metadata(prefix))
        return tuple(data)


@dataclass(frozen=True, slots=True)
class ToolSchemaIngestionResult:
    """Normalized summary of a tool schema bundle."""

    provider_family: ToolSchemaProvider
    tools: tuple[ToolDefinition, ...]
    issues: tuple[ToolSchemaIssue, ...] = ()
    source_spans: tuple[tuple[str, SourceSpan], ...] = ()

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(tool.name for tool in self.tools))

    @property
    def closed_tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(tool.name for tool in self.tools if tool.parameter_schema.closed))

    @property
    def argument_encodings(self) -> tuple[str, ...]:
        return tuple(sorted({tool.argument_encoding for tool in self.tools}))

    @property
    def envelope_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({tool.envelope for tool in self.tools if tool.envelope is not None}))

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        data: list[tuple[str, object]] = [
            ("provider_family", self.provider_family.value),
            ("tool_count", len(self.tools)),
            ("tool_names", self.tool_names),
            ("closed_tool_names", self.closed_tool_names),
            ("argument_encodings", self.argument_encodings),
            ("envelope_kinds", self.envelope_kinds),
            ("issue_count", len(self.issues)),
        ]
        for index, tool in enumerate(sorted(self.tools, key=lambda item: item.name)):
            data.extend(tool.to_metadata(index))
        for index, issue in enumerate(self.issues):
            data.extend(issue.to_metadata(index))
        return tuple(data)


class ToolSchemaIngestionError(ValueError):
    """Raised when a tool/function schema bundle cannot be normalized."""

    def __init__(self, message: str, *, path: tuple[str, ...] = (), span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.span = span


def ingest_tool_schema_mapping(
    raw: Any,
    *,
    declared_provider: str | None = None,
    source_map: JsonSourceMap | None = None,
) -> ToolSchemaIngestionResult:
    """Normalize common tool/function schema containers into typed definitions."""

    provider_hint = _provider_hint(declared_provider)
    tools: list[ToolDefinition] = []
    issues: list[ToolSchemaIssue] = []

    if isinstance(raw, list):
        provider = provider_hint or _guess_list_provider(raw)
        for index, item in enumerate(raw):
            tools.append(_ingest_tool_entry(item, provider, (str(index),), source_map, issues))
        return _result(provider, tools, issues, source_map)

    if not isinstance(raw, dict):
        raise ToolSchemaIngestionError("tool-definition root must be a JSON object or array")

    provider = provider_hint or _provider_hint(_optional_string(raw.get("provider"))) or _guess_provider(raw)
    if isinstance(raw.get("tools"), list):
        tool_items = raw["tools"]
        assert isinstance(tool_items, list)
        for index, item in enumerate(tool_items):
            tools.append(_ingest_tool_entry(item, provider, ("tools", str(index)), source_map, issues))
    elif isinstance(raw.get("functions"), list):
        functions = raw["functions"]
        assert isinstance(functions, list)
        for index, item in enumerate(functions):
            tools.append(_ingest_generic_function(item, provider, ("functions", str(index)), source_map, issues))
    elif isinstance(raw.get("request"), dict):
        request = raw["request"]
        nested = ingest_tool_schema_mapping(
            request,
            declared_provider=provider.value,
            source_map=source_map,
        )
        tools.extend(
            ToolDefinition(
                name=tool.name,
                provider=provider,
                parameter_schema=tool.parameter_schema,
                source_path=("request", *tool.source_path),
                description=tool.description,
                envelope="request.tools",
                argument_encoding=tool.argument_encoding,
                tool_id_path=tool.tool_id_path,
                span=tool.span,
            )
            for tool in nested.tools
        )
        issues.extend(nested.issues)
    elif _looks_like_single_tool(raw):
        tools.append(_ingest_tool_entry(raw, provider, (), source_map, issues))
    elif _looks_like_pydantic_schema(raw):
        tools.append(_ingest_pydantic_model(raw, ("<root>",), source_map, issues))
        provider = ToolSchemaProvider.PYDANTIC
    else:
        response_tools = _ingest_response_envelope(raw, provider, source_map, issues)
        if response_tools:
            tools.extend(response_tools)

    if not tools:
        raise ToolSchemaIngestionError(
            "tool-definition artifact does not contain a supported tool schema envelope",
            span=_span(source_map, ()),
        )
    return _result(provider, tools, issues, source_map)


def _result(
    provider: ToolSchemaProvider,
    tools: list[ToolDefinition],
    issues: list[ToolSchemaIssue],
    source_map: JsonSourceMap | None,
) -> ToolSchemaIngestionResult:
    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ToolSchemaIngestionError(f"duplicate tool names: {', '.join(duplicates)}")
    spans = _tool_source_spans(source_map, tools)
    return ToolSchemaIngestionResult(
        provider_family=provider,
        tools=tuple(sorted(tools, key=lambda item: item.name)),
        issues=tuple(issues),
        source_spans=spans,
    )


def _ingest_tool_entry(
    item: Any,
    provider: ToolSchemaProvider,
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolDefinition:
    if not isinstance(item, dict):
        raise ToolSchemaIngestionError("tool entries must be JSON objects", path=path, span=_span(source_map, path))
    if provider is ToolSchemaProvider.OPENAI or "function" in item:
        return _ingest_openai_tool(item, path, source_map, issues)
    if provider is ToolSchemaProvider.ANTHROPIC or "input_schema" in item:
        return _ingest_named_schema_tool(item, provider, "input_schema", path, source_map, issues)
    if provider is ToolSchemaProvider.MCP or "inputSchema" in item:
        return _ingest_named_schema_tool(item, ToolSchemaProvider.MCP, "inputSchema", path, source_map, issues)
    if provider is ToolSchemaProvider.LANGCHAIN or "args_schema" in item:
        return _ingest_named_schema_tool(item, ToolSchemaProvider.LANGCHAIN, "args_schema", path, source_map, issues)
    if "parameters" in item or "schema" in item:
        return _ingest_generic_function(item, provider, path, source_map, issues)
    raise ToolSchemaIngestionError(
        "tool entry lacks a supported parameter schema field",
        path=path,
        span=_span(source_map, path),
    )


def _ingest_openai_tool(
    item: dict[str, Any],
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolDefinition:
    function = item.get("function", item)
    if not isinstance(function, dict):
        raise ToolSchemaIngestionError("OpenAI tool field 'function' must be an object", path=path)
    name = _required_name(function.get("name"), (*path, "function", "name") if "function" in item else (*path, "name"), source_map)
    parameters = function.get("parameters", {"type": "object", "properties": {}})
    parameter_path = (*path, "function", "parameters") if "function" in item else (*path, "parameters")
    summary = _summarize_parameter_schema(parameters, parameter_path, source_map, issues)
    description = _optional_string(function.get("description"))
    if description is None:
        issues.append(_issue(ToolSchemaIssueKind.MISSING_DESCRIPTION, "tool lacks a description", (*path, "function"), source_map))
    return ToolDefinition(
        name=name,
        provider=ToolSchemaProvider.OPENAI,
        parameter_schema=summary,
        source_path=path,
        description=description,
        envelope="openai.function",
        span=_span(source_map, path),
    )


def _ingest_named_schema_tool(
    item: dict[str, Any],
    provider: ToolSchemaProvider,
    schema_key: str,
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolDefinition:
    name = _required_name(item.get("name"), (*path, "name"), source_map)
    summary = _summarize_parameter_schema(item.get(schema_key), (*path, schema_key), source_map, issues)
    description = _optional_string(item.get("description"))
    if description is None:
        issues.append(_issue(ToolSchemaIssueKind.MISSING_DESCRIPTION, "tool lacks a description", path, source_map))
    envelope = {
        ToolSchemaProvider.ANTHROPIC: "anthropic.tool",
        ToolSchemaProvider.LANGCHAIN: "langchain.tool",
        ToolSchemaProvider.MCP: "mcp.tool",
        ToolSchemaProvider.TYPESCRIPT: "typescript.function",
    }.get(provider, "generic.tool")
    return ToolDefinition(
        name=name,
        provider=provider,
        parameter_schema=summary,
        source_path=path,
        description=description,
        envelope=envelope,
        span=_span(source_map, path),
    )


def _ingest_generic_function(
    item: Any,
    provider: ToolSchemaProvider,
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolDefinition:
    if not isinstance(item, dict):
        raise ToolSchemaIngestionError("function entries must be objects", path=path, span=_span(source_map, path))
    name = _required_name(item.get("name"), (*path, "name"), source_map)
    schema_key = "parameters" if "parameters" in item else "schema"
    summary = _summarize_parameter_schema(item.get(schema_key), (*path, schema_key), source_map, issues)
    description = _optional_string(item.get("description"))
    if description is None:
        issues.append(_issue(ToolSchemaIssueKind.MISSING_DESCRIPTION, "function lacks a description", path, source_map))
    if provider is ToolSchemaProvider.GENERIC and "typescript" in str(item.get("source", "")).lower():
        provider = ToolSchemaProvider.TYPESCRIPT
    return ToolDefinition(
        name=name,
        provider=provider,
        parameter_schema=summary,
        source_path=path,
        description=description,
        envelope="function.parameters",
        span=_span(source_map, path),
    )


def _ingest_pydantic_model(
    raw: dict[str, Any],
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolDefinition:
    name = _optional_string(raw.get("title")) or _optional_string(raw.get("model")) or "PydanticModel"
    summary = _summarize_parameter_schema(raw, (), source_map, issues)
    return ToolDefinition(
        name=name,
        provider=ToolSchemaProvider.PYDANTIC,
        parameter_schema=summary,
        source_path=path,
        description=_optional_string(raw.get("description")),
        envelope="pydantic.model-json-schema",
        span=_span(source_map, ()),
    )


def _ingest_response_envelope(
    raw: dict[str, Any],
    provider: ToolSchemaProvider,
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> tuple[ToolDefinition, ...]:
    tool_calls = raw.get("tool_calls") or raw.get("toolCalls")
    if not isinstance(tool_calls, list):
        return ()
    tools: list[ToolDefinition] = []
    for index, call in enumerate(tool_calls):
        path = ("tool_calls", str(index)) if "tool_calls" in raw else ("toolCalls", str(index))
        if not isinstance(call, dict):
            raise ToolSchemaIngestionError("tool-call envelope entries must be objects", path=path)
        function = call.get("function", call)
        if not isinstance(function, dict):
            raise ToolSchemaIngestionError("tool-call function envelope must be an object", path=path)
        name = _required_name(function.get("name"), (*path, "function", "name") if "function" in call else (*path, "name"), source_map)
        arguments = function.get("arguments") or call.get("args") or {}
        argument_encoding = "json-string" if isinstance(arguments, str) else "json-object"
        if argument_encoding == "json-string":
            issues.append(
                _issue(
                    ToolSchemaIssueKind.ARGUMENT_STRING_ENVELOPE,
                    "tool-call arguments are serialized as a JSON string",
                    (*path, "function", "arguments") if "function" in call else (*path, "args"),
                    source_map,
                )
            )
        if "delta" in call or "index" in call:
            issues.append(
                _issue(
                    ToolSchemaIssueKind.STREAMING_CHUNK_ENVELOPE,
                    "tool call appears inside a streaming chunk envelope",
                    path,
                    source_map,
                )
            )
        schema = _schema_from_arguments(arguments)
        summary = _summarize_parameter_schema(schema, path, source_map, issues)
        tools.append(
            ToolDefinition(
                name=name,
                provider=provider if provider is not ToolSchemaProvider.GENERIC else ToolSchemaProvider.PROVIDER_ENVELOPE,
                parameter_schema=summary,
                source_path=path,
                envelope="provider.tool-call",
                argument_encoding=argument_encoding,
                tool_id_path=(*path, "id") if "id" in call else (),
                span=_span(source_map, path),
            )
        )
    return tuple(tools)


def _summarize_parameter_schema(
    schema: Any,
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
    issues: list[ToolSchemaIssue],
) -> ToolParameterSummary:
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        raise ToolSchemaIngestionError("tool parameter schema must be a JSON object", path=path, span=_span(source_map, path))
    schema_type = _schema_type(schema.get("type"))
    if schema_type is not None and schema_type != "object":
        issues.append(_issue(ToolSchemaIssueKind.NON_OBJECT_PARAMETERS, "parameter schema is not an object", path, source_map))
    properties = schema.get("properties", {})
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        raise ToolSchemaIngestionError("tool parameter schema 'properties' must be an object", path=(*path, "properties"))
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
        raise ToolSchemaIngestionError("tool parameter schema 'required' must be a list of strings", path=(*path, "required"))
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        additional = True
    if additional is not False:
        issues.append(_issue(ToolSchemaIssueKind.OPEN_PARAMETERS, "parameter schema permits undeclared properties", path, source_map))
    enum_paths, constraint_paths = _schema_feature_paths(schema)
    return ToolParameterSummary(
        property_names=tuple(sorted(properties)),
        required=tuple(sorted(required)),
        schema_type=schema_type,
        additional_properties=additional if isinstance(additional, bool) else None,
        enum_paths=enum_paths,
        constraint_paths=constraint_paths,
    )


def _schema_feature_paths(schema: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    enum_paths: list[str] = []
    constraint_paths: list[str] = []
    constraints = {
        "const",
        "format",
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "pattern",
    }

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = (*path, key)
                if key == "enum":
                    enum_paths.append(".".join(child))
                if key in constraints:
                    constraint_paths.append(".".join(child))
                visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))

    visit(schema, ())
    return tuple(sorted(enum_paths)), tuple(sorted(constraint_paths))


def _schema_from_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"type": "object", "properties": {}, "additionalProperties": True}
    if isinstance(arguments, dict):
        properties = {key: {"type": _json_type(value)} for key, value in arguments.items()}
        return {"type": "object", "properties": properties, "required": sorted(arguments), "additionalProperties": False}
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _tool_source_spans(
    source_map: JsonSourceMap | None,
    tools: list[ToolDefinition],
) -> tuple[tuple[str, SourceSpan], ...]:
    if source_map is None:
        return ()
    spans: list[tuple[str, SourceSpan]] = []
    for tool in tools:
        if tool.span is not None:
            spans.append((f"tool.{tool.name}", tool.span))
        name_span = _span(source_map, (*tool.source_path, "name"))
        if name_span is None and tool.provider is ToolSchemaProvider.OPENAI:
            name_span = _span(source_map, (*tool.source_path, "function", "name"))
        if name_span is not None:
            spans.append((f"tool.{tool.name}.name", name_span))
    return tuple(sorted(spans, key=lambda item: item[0]))


def _guess_provider(raw: dict[str, Any]) -> ToolSchemaProvider:
    provider = str(raw.get("provider", "")).lower()
    if provider:
        return _provider_hint(provider) or ToolSchemaProvider.GENERIC
    if "mcp" in raw or raw.get("protocol") == "mcp":
        return ToolSchemaProvider.MCP
    if isinstance(raw.get("tools"), list) and raw["tools"]:
        first = raw["tools"][0]
        if isinstance(first, dict):
            if "function" in first:
                return ToolSchemaProvider.OPENAI
            if "input_schema" in first:
                return ToolSchemaProvider.ANTHROPIC
            if "inputSchema" in first:
                return ToolSchemaProvider.MCP
            if "args_schema" in first:
                return ToolSchemaProvider.LANGCHAIN
    if isinstance(raw.get("functions"), list):
        return ToolSchemaProvider.TYPESCRIPT
    if _looks_like_pydantic_schema(raw):
        return ToolSchemaProvider.PYDANTIC
    if isinstance(raw.get("tool_calls"), list) or isinstance(raw.get("toolCalls"), list):
        return ToolSchemaProvider.PROVIDER_ENVELOPE
    return ToolSchemaProvider.GENERIC


def _guess_list_provider(raw: list[Any]) -> ToolSchemaProvider:
    first = raw[0] if raw else {}
    if isinstance(first, dict):
        if "function" in first:
            return ToolSchemaProvider.OPENAI
        if "input_schema" in first:
            return ToolSchemaProvider.ANTHROPIC
        if "inputSchema" in first:
            return ToolSchemaProvider.MCP
        if "args_schema" in first:
            return ToolSchemaProvider.LANGCHAIN
    return ToolSchemaProvider.GENERIC


def _provider_hint(value: str | None) -> ToolSchemaProvider | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "openai-compatible": ToolSchemaProvider.OPENAI,
        "openai": ToolSchemaProvider.OPENAI,
        "anthropic": ToolSchemaProvider.ANTHROPIC,
        "claude": ToolSchemaProvider.ANTHROPIC,
        "langchain": ToolSchemaProvider.LANGCHAIN,
        "pydantic": ToolSchemaProvider.PYDANTIC,
        "typescript": ToolSchemaProvider.TYPESCRIPT,
        "ts": ToolSchemaProvider.TYPESCRIPT,
        "mcp": ToolSchemaProvider.MCP,
        "model-context-protocol": ToolSchemaProvider.MCP,
        "provider-envelope": ToolSchemaProvider.PROVIDER_ENVELOPE,
        "generic": ToolSchemaProvider.GENERIC,
    }
    return aliases.get(normalized)


def _schema_type(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "|".join(sorted(value))
    return None


def _looks_like_single_tool(raw: dict[str, Any]) -> bool:
    return "name" in raw and any(key in raw for key in ("function", "parameters", "schema", "input_schema", "inputSchema", "args_schema"))


def _looks_like_pydantic_schema(raw: dict[str, Any]) -> bool:
    return (
        raw.get("type") == "object"
        and isinstance(raw.get("properties"), dict)
        and ("title" in raw or raw.get("$schema") is not None)
    )


def _required_name(value: Any, path: tuple[str, ...], source_map: JsonSourceMap | None) -> str:
    if not isinstance(value, str) or not value:
        raise ToolSchemaIngestionError("tool name must be a non-empty string", path=path, span=_span(source_map, path))
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _issue(
    kind: ToolSchemaIssueKind,
    message: str,
    path: tuple[str, ...],
    source_map: JsonSourceMap | None,
) -> ToolSchemaIssue:
    return ToolSchemaIssue(kind=kind, message=message, path=path, span=_span(source_map, path))


def _span(source_map: JsonSourceMap | None, path: tuple[str, ...]) -> SourceSpan | None:
    if source_map is None:
        return None
    return source_map.span_for(path) or source_map.key_span_for(path)
