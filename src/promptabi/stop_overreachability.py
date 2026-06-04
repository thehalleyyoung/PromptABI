"""Bounded stop-overreachability analysis for structured-output regions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import Artifact, ArtifactKind, StopPolicyArtifact

_STOP_MARKER = "__PROMPTABI_STOP__"


@dataclass(frozen=True, slots=True)
class StructuredOutputRegion:
    """One serialized structured-output witness considered by the bounded check."""

    kind: str
    name: str
    path: str
    witness_text: str
    parser_state: str
    description: str
    structural_stops: tuple[str, ...] = ()
    artifact_name: str | None = None


@dataclass(frozen=True, slots=True)
class StopOverreachabilityFinding:
    """A configured stop can fire before a valid structured output is complete."""

    stop_sequence: str
    category: str
    region: StructuredOutputRegion
    firing_offset: int
    valid_output: str
    valid_output_prefix: str
    truncated_prefix: str
    resulting_state: str
    firing_point: str
    resulting_structure: str


@dataclass(frozen=True, slots=True)
class StopOverreachabilityAbstention:
    """A structured artifact could not be modeled by the bounded checker."""

    artifact_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class StopOverreachabilityReport:
    """Bounded stop-overreachability result for one stop policy."""

    stop_policy_name: str
    bound: str
    findings: tuple[StopOverreachabilityFinding, ...]
    abstentions: tuple[StopOverreachabilityAbstention, ...] = ()

    @property
    def structural_findings(self) -> tuple[StopOverreachabilityFinding, ...]:
        return tuple(finding for finding in self.findings if finding.category == "structural")

    @property
    def content_findings(self) -> tuple[StopOverreachabilityFinding, ...]:
        return tuple(finding for finding in self.findings if finding.category == "content")


def analyze_stop_overreachability(
    stop_policy: StopPolicyArtifact,
    structured_artifacts: Sequence[Artifact] = (),
) -> StopOverreachabilityReport:
    """Check whether stop strings can truncate valid structured outputs.

    The bounded model contains fixed structural regions for JSON, markdown code
    fences, XML-like tool-call envelopes, and common provider tool-call envelopes.
    Content findings are only emitted for supported JSON Schema/tool-parameter
    string fields whose constraints permit the configured stop text.
    """

    regions, abstentions = _regions_from_artifacts(structured_artifacts)
    all_regions = (*_builtin_structural_regions(), *regions)
    findings: list[StopOverreachabilityFinding] = []
    for stop_sequence in stop_policy.stop_sequences:
        if not stop_sequence:
            continue
        for region in all_regions:
            category = _category_for_stop(stop_sequence, region)
            if category is None:
                continue
            finding = _finding_for(stop_sequence, category, region)
            if finding is not None:
                findings.append(finding)
    return StopOverreachabilityReport(
        stop_policy_name=stop_policy.name,
        bound=(
            f"{len(_builtin_structural_regions())} built-in structural regions plus "
            f"{len(regions)} artifact-derived regions; schema/tool recursion depth <= 3"
        ),
        findings=tuple(sorted(findings, key=_finding_sort_key)),
        abstentions=tuple(sorted(abstentions, key=lambda item: (item.artifact_name, item.reason))),
    )


def _category_for_stop(stop_sequence: str, region: StructuredOutputRegion) -> str | None:
    if stop_sequence in region.structural_stops:
        return "structural"
    if not region.structural_stops and _STOP_MARKER in region.witness_text:
        return "content"
    return None


def _finding_for(
    stop_sequence: str,
    category: str,
    region: StructuredOutputRegion,
) -> StopOverreachabilityFinding | None:
    witness_text = region.witness_text
    if category == "content":
        marker_offset = witness_text.find(_STOP_MARKER)
        if marker_offset < 0:
            return None
        escaped_stop = json.dumps(stop_sequence, ensure_ascii=False)[1:-1]
        in_value_offset = escaped_stop.find(stop_sequence)
        if in_value_offset < 0:
            return None
        witness_text = witness_text.replace(_STOP_MARKER, escaped_stop, 1)
        offset = marker_offset + in_value_offset
    else:
        offset = witness_text.find(stop_sequence)
    if offset < 0:
        return None
    truncated_prefix = witness_text[:offset]
    valid_output_prefix = witness_text[: offset + len(stop_sequence)]
    return StopOverreachabilityFinding(
        stop_sequence=stop_sequence,
        category=category,
        region=region,
        firing_offset=offset,
        valid_output=witness_text,
        valid_output_prefix=valid_output_prefix,
        truncated_prefix=truncated_prefix,
        resulting_state=region.parser_state,
        firing_point=_firing_point(witness_text, offset, stop_sequence),
        resulting_structure=_resulting_structure(region, truncated_prefix, witness_text),
    )


def _finding_sort_key(finding: StopOverreachabilityFinding) -> tuple[str, str, str, int]:
    return (
        finding.category,
        finding.stop_sequence,
        finding.region.kind,
        finding.firing_offset,
    )


def _firing_point(valid_output: str, offset: int, stop_sequence: str) -> str:
    line = valid_output.count("\n", 0, offset) + 1
    line_start = valid_output.rfind("\n", 0, offset) + 1
    column = offset - line_start + 1
    context_start = max(0, offset - 24)
    context_end = min(len(valid_output), offset + len(stop_sequence) + 24)
    context = valid_output[context_start:context_end]
    return (
        f"offset {offset}, line {line}, column {column}, matched "
        f"{stop_sequence!r} in context {_visible(context)!r}"
    )


def _resulting_structure(
    region: StructuredOutputRegion,
    truncated_prefix: str,
    valid_output: str,
) -> str:
    if region.kind in {
        "json",
        "json-schema-string",
        "tool-arguments-json",
        "openai-tool-envelope",
        "anthropic-tool-envelope",
        "openai-tool-call-content",
    }:
        return _json_truncation_effect(truncated_prefix)
    if region.kind in {"xml-tool-call", "xml-tool-call-content"}:
        return _xml_truncation_effect(truncated_prefix)
    if region.kind == "markdown-code-block":
        return _markdown_truncation_effect(truncated_prefix)
    remaining = len(valid_output) - len(truncated_prefix)
    return f"prematurely accepted prefix with {remaining} required trailing character(s) removed"


def _json_truncation_effect(truncated_prefix: str) -> str:
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(truncated_prefix)
    except json.JSONDecodeError as exc:
        return f"malformed JSON prefix: {exc.msg} at line {exc.lineno}, column {exc.colno}"
    trailing = truncated_prefix[end:].strip()
    if trailing:
        return f"malformed JSON prefix: valid JSON value followed by trailing text {trailing!r}"
    return f"prematurely accepted JSON prefix: {type(value).__name__} value parses before the full witness"


def _xml_truncation_effect(truncated_prefix: str) -> str:
    if not truncated_prefix:
        return "prematurely accepted empty prefix before XML-like tool-call envelope"
    missing: list[str] = []
    for tag in ("tool_call", "arguments"):
        opens = truncated_prefix.count(f"<{tag}")
        closes = truncated_prefix.count(f"</{tag}>")
        if opens > closes:
            missing.append(f"</{tag}>")
    if missing:
        return f"malformed XML-like prefix: missing closing tag(s) {', '.join(missing)}"
    return "prematurely accepted XML-like prefix before the complete tool-call envelope"


def _markdown_truncation_effect(truncated_prefix: str) -> str:
    fence_count = truncated_prefix.count("```")
    if fence_count % 2 == 1:
        return "malformed markdown prefix: code fence is open at truncation"
    if not truncated_prefix:
        return "prematurely accepted empty prefix before fenced structured output"
    return "prematurely accepted markdown prefix before fenced structured output is complete"


def _visible(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r")


def _builtin_structural_regions() -> tuple[StructuredOutputRegion, ...]:
    nested_json = _compact_json({"outer": {"inner": ["value"]}, "tail": True})
    markdown = "```json\n" + _compact_json({"ok": True}) + "\n```"
    xml = '<tool_call><arguments>{"ok":true}</arguments></tool_call>'
    openai = _compact_json(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"ok":true}'},
                }
            ],
            "finish_reason": "tool_calls",
        }
    )
    anthropic = _compact_json({"type": "tool_use", "name": "lookup", "input": {"ok": True}})
    return (
        StructuredOutputRegion(
            kind="json",
            name="nested-json-structure",
            path="$",
            witness_text=nested_json,
            parser_state="inside nested JSON document before the complete root value has been consumed",
            description="nested JSON object/array delimiters can occur before the document is complete",
            structural_stops=('"', "}", "]", ",", ":"),
        ),
        StructuredOutputRegion(
            kind="markdown-code-block",
            name="json-code-fence",
            path="markdown.fence",
            witness_text=markdown,
            parser_state="inside markdown code block before the closing fence has been emitted",
            description="a valid markdown code block closes with a fence that may be configured as a stop",
            structural_stops=("```",),
        ),
        StructuredOutputRegion(
            kind="xml-tool-call",
            name="xml-tool-call-envelope",
            path="tool_call.arguments",
            witness_text=xml,
            parser_state="inside XML-like tool-call envelope before the closing tag has been emitted",
            description="XML-like tool-call tags are structural delimiters, not parser-complete output by themselves",
            structural_stops=("<tool_call>", "</tool_call>", "<arguments>", "</arguments>"),
        ),
        StructuredOutputRegion(
            kind="openai-tool-envelope",
            name="openai-tool-call-envelope",
            path="$.tool_calls[0].function.arguments",
            witness_text=openai,
            parser_state="inside provider tool-call envelope before the JSON response object is complete",
            description="OpenAI-compatible tool-call envelopes contain nested structural JSON delimiters",
            structural_stops=('"tool_calls"', '"function"', '"arguments"', "}", "]"),
        ),
        StructuredOutputRegion(
            kind="anthropic-tool-envelope",
            name="anthropic-tool-use-envelope",
            path="$.input",
            witness_text=anthropic,
            parser_state="inside provider tool-use envelope before the JSON response object is complete",
            description="Anthropic-style tool-use envelopes contain nested structural JSON delimiters",
            structural_stops=('"tool_use"', '"input"', "}", "]"),
        ),
    )


def _regions_from_artifacts(
    artifacts: Sequence[Artifact],
) -> tuple[tuple[StructuredOutputRegion, ...], tuple[StopOverreachabilityAbstention, ...]]:
    regions: list[StructuredOutputRegion] = []
    abstentions: list[StopOverreachabilityAbstention] = []
    for artifact in artifacts:
        if artifact.kind is ArtifactKind.SCHEMA:
            loaded, reason = _load_json_artifact(artifact)
            if reason is not None:
                abstentions.append(StopOverreachabilityAbstention(artifact.name, reason))
                continue
            schema_regions, schema_abstentions = _schema_regions(artifact, loaded)
            regions.extend(schema_regions)
            abstentions.extend(schema_abstentions)
        elif artifact.kind is ArtifactKind.TOOL_DEFINITION:
            loaded, reason = _load_json_artifact(artifact)
            if reason is not None:
                abstentions.append(StopOverreachabilityAbstention(artifact.name, reason))
                continue
            tool_regions, tool_abstentions = _tool_regions(artifact, loaded)
            regions.extend(tool_regions)
            abstentions.extend(tool_abstentions)
    return tuple(regions), tuple(abstentions)


def _load_json_artifact(artifact: Artifact) -> tuple[Any, str | None]:
    path = artifact.location.path
    if path is None:
        return None, "artifact is not a local JSON file"
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text), None
    except OSError as exc:
        return None, f"artifact could not be read: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"artifact is not valid JSON at {exc.lineno}:{exc.colno}: {exc.msg}"


def _schema_regions(
    artifact: Artifact,
    schema: Any,
) -> tuple[tuple[StructuredOutputRegion, ...], tuple[StopOverreachabilityAbstention, ...]]:
    if not isinstance(schema, Mapping):
        return (), (StopOverreachabilityAbstention(artifact.name, "schema root is not an object"),)
    witnesses = _schema_string_witnesses(schema, depth=0, path="$")
    if not witnesses:
        return (), (
            StopOverreachabilityAbstention(
                artifact.name,
                "no unconstrained string field was found in the supported object/string schema fragment",
            ),
        )
    regions = tuple(
        StructuredOutputRegion(
            kind="json-schema-string",
            name=f"{artifact.name}:{path}",
            path=path,
            witness_text=_compact_json(witness),
            parser_state=f"inside JSON string value at {path}",
            description="supported JSON Schema fragment admits this stop string inside a valid string field",
            artifact_name=artifact.name,
        )
        for path, witness in witnesses
    )
    return regions, ()


def _schema_string_witnesses(
    schema: Mapping[str, Any],
    *,
    depth: int,
    path: str,
) -> tuple[tuple[str, Any], ...]:
    if depth > 3:
        return ()
    schema_type = schema.get("type")
    if schema_type == "string" or (schema_type is None and "properties" not in schema):
        if _is_free_string_schema(schema):
            return ((path, _STOP_MARKER),)
        return ()
    if schema_type not in (None, "object") or not isinstance(schema.get("properties"), Mapping):
        return ()
    required = schema.get("required", ())
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        required = ()
    properties = schema["properties"]
    witnesses: list[tuple[str, Any]] = []
    for property_name, property_schema in sorted(properties.items()):
        if not isinstance(property_name, str) or not isinstance(property_schema, Mapping):
            continue
        for string_path, marker_object in _schema_string_witnesses(
            property_schema,
            depth=depth + 1,
            path=f"{path}.{property_name}",
        ):
            obj: dict[str, Any] = {}
            for required_name in required:
                if required_name == property_name:
                    continue
                required_schema = properties.get(required_name)
                if isinstance(required_schema, Mapping):
                    obj[required_name] = _minimal_value(required_schema)
            obj[property_name] = marker_object
            witnesses.append((string_path, obj))
    return tuple(witnesses)


def _tool_regions(
    artifact: Artifact,
    tools: Any,
) -> tuple[tuple[StructuredOutputRegion, ...], tuple[StopOverreachabilityAbstention, ...]]:
    tool_items = tools if isinstance(tools, list) else tools.get("tools") if isinstance(tools, Mapping) else None
    if not isinstance(tool_items, list):
        return (), (StopOverreachabilityAbstention(artifact.name, "tool definitions are not a list"),)
    regions: list[StructuredOutputRegion] = []
    abstentions: list[StopOverreachabilityAbstention] = []
    for index, tool in enumerate(tool_items):
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            continue
        schema_regions, schema_abstentions = _schema_regions(artifact, parameters)
        if schema_abstentions:
            abstentions.append(
                StopOverreachabilityAbstention(
                    artifact.name,
                    f"tool {name!r} parameters do not expose a supported free string field",
                )
            )
            continue
        for region in schema_regions:
            argument_json = region.witness_text
            regions.extend(
                (
                    StructuredOutputRegion(
                        kind="tool-arguments-json",
                        name=f"{artifact.name}:{name}:{region.path}",
                        path=f"tools[{index}].function.parameters{region.path[1:]}",
                        witness_text=argument_json,
                        parser_state=region.parser_state,
                        description=f"tool {name!r} arguments schema admits this stop inside a valid JSON string",
                        artifact_name=artifact.name,
                    ),
                    StructuredOutputRegion(
                        kind="xml-tool-call-content",
                        name=f"{artifact.name}:{name}:xml:{region.path}",
                        path=f"tool_call[{name}].arguments{region.path[1:]}",
                        witness_text=f'<tool_call name="{name}"><arguments>{argument_json}</arguments></tool_call>',
                        parser_state=f"inside XML-like tool-call argument string at {region.path}",
                        description=f"tool {name!r} XML-style envelope contains valid JSON arguments",
                        artifact_name=artifact.name,
                    ),
                    StructuredOutputRegion(
                        kind="openai-tool-call-content",
                        name=f"{artifact.name}:{name}:openai:{region.path}",
                        path=f"$.tool_calls[0].function.arguments{region.path[1:]}",
                        witness_text=_compact_json(
                            {
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {"name": name, "arguments": argument_json},
                                    }
                                ]
                            }
                        ),
                        parser_state=f"inside OpenAI-compatible function.arguments string at {region.path}",
                        description=f"tool {name!r} OpenAI-compatible envelope serializes arguments as a JSON string",
                        artifact_name=artifact.name,
                    ),
                )
            )
    if not regions and not abstentions:
        abstentions.append(StopOverreachabilityAbstention(artifact.name, "no supported function tools found"))
    return tuple(regions), tuple(abstentions)


def _is_free_string_schema(schema: Mapping[str, Any]) -> bool:
    if any(key in schema for key in ("enum", "const", "pattern")):
        return False
    return "maxLength" not in schema


def _minimal_value(schema: Mapping[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    schema_type = schema.get("type")
    if schema_type == "string":
        return "x"
    if schema_type in ("integer", "number"):
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object" or isinstance(schema.get("properties"), Mapping):
        return {}
    return None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
