import json
from hashlib import sha256
from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ToolDefinitionArtifact,
    ToolSchemaIngestionError,
    ToolSchemaProvider,
    ingest_tool_schema_mapping,
)
from promptabi.cli import main
from promptabi.loaders import ArtifactLoadError, ArtifactLoader
from promptabi.source import build_json_source_map


def test_ingests_openai_tool_bundle_with_closed_parameters() -> None:
    raw = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "required": ["order_id"],
                        "additionalProperties": False,
                        "properties": {"order_id": {"type": "string", "pattern": "^ORD-[0-9]+$"}},
                    },
                },
            }
        ]
    }

    result = ingest_tool_schema_mapping(raw, declared_provider="openai")

    assert result.provider_family is ToolSchemaProvider.OPENAI
    assert result.tool_names == ("lookup_order",)
    assert result.closed_tool_names == ("lookup_order",)
    assert result.tools[0].parameter_schema.constraint_paths == ("properties.order_id.pattern",)
    assert result.issues == ()


def test_ingests_anthropic_langchain_pydantic_typescript_mcp_and_envelopes() -> None:
    cases = [
        (
            "anthropic",
            {"tools": [{"name": "search", "description": "Search.", "input_schema": _schema("query")}]},
            ToolSchemaProvider.ANTHROPIC,
            ("search",),
        ),
        (
            "langchain",
            {"tools": [{"name": "refund", "description": "Refund.", "args_schema": _schema("user_id")}]},
            ToolSchemaProvider.LANGCHAIN,
            ("refund",),
        ),
        (
            "pydantic",
            {"title": "RefundArgs", "type": "object", "properties": {"reason": {"enum": ["duplicate"]}}},
            ToolSchemaProvider.PYDANTIC,
            ("RefundArgs",),
        ),
        (
            "typescript",
            {"functions": [{"name": "routeTicket", "description": "Route.", "parameters": _schema("queue")}]},
            ToolSchemaProvider.TYPESCRIPT,
            ("routeTicket",),
        ),
        (
            "mcp",
            {"protocol": "mcp", "tools": [{"name": "read_file", "description": "Read.", "inputSchema": _schema("path")}]},
            ToolSchemaProvider.MCP,
            ("read_file",),
        ),
        (
            "provider-envelope",
            {"tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": '{"order_id":"ORD-1"}'}}]},
            ToolSchemaProvider.PROVIDER_ENVELOPE,
            ("lookup",),
        ),
    ]

    for declared_provider, raw, expected_provider, expected_names in cases:
        result = ingest_tool_schema_mapping(raw, declared_provider=declared_provider)
        assert result.provider_family is expected_provider
        assert result.tool_names == expected_names
        assert result.tools[0].parameter_schema.property_names

    envelope = ingest_tool_schema_mapping(cases[-1][1], declared_provider="provider-envelope")
    assert envelope.argument_encodings == ("json-string",)
    assert [issue.kind.value for issue in envelope.issues] == [
        "argument-string-envelope",
    ]


def test_tool_schema_ingestion_preserves_source_spans() -> None:
    text = json.dumps(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup_order", "parameters": _schema("order_id")},
                }
            ]
        },
        indent=2,
    )
    source_map = build_json_source_map(text, "tools.json")

    result = ingest_tool_schema_mapping(json.loads(text), declared_provider="openai", source_map=source_map)

    spans = dict(result.source_spans)
    assert "tool.lookup_order.name" in spans
    assert spans["tool.lookup_order.name"].path == "tools.json"


def test_tool_schema_ingestion_rejects_invalid_and_duplicate_tools() -> None:
    with pytest.raises(ToolSchemaIngestionError, match="duplicate tool names"):
        ingest_tool_schema_mapping(
            {
                "tools": [
                    {"name": "dup", "inputSchema": _schema("x")},
                    {"name": "dup", "inputSchema": _schema("y")},
                ]
            },
            declared_provider="mcp",
        )

    with pytest.raises(ToolSchemaIngestionError, match="tool name"):
        ingest_tool_schema_mapping({"tools": [{"input_schema": _schema("query")}]}, declared_provider="anthropic")


def test_loader_parses_tool_definition_and_updates_artifact_names(tmp_path: Path) -> None:
    tools = tmp_path / "tools.json"
    tools.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "description": "Lookup.",
                            "parameters": _schema("order_id", closed=True),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "refund_order",
                            "description": "Refund.",
                            "parameters": _schema("reason", closed=True),
                        },
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = ToolDefinitionArtifact(
        kind=ArtifactKind.TOOL_DEFINITION,
        name="tools",
        location=ArtifactLocation(path=str(tools)),
        provenance=ArtifactProvenance(sha256=sha256(tools.read_bytes()).hexdigest()),
        provider="openai",
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "tool-definition-schema"
    assert isinstance(loaded.artifact, ToolDefinitionArtifact)
    assert loaded.artifact.tool_names == ("lookup_order", "refund_order")
    metadata = dict(loaded.metadata)
    assert metadata["provider_family"] == "openai"
    assert metadata["tool_count"] == 2
    assert metadata["closed_tool_names"] == ("lookup_order", "refund_order")
    assert any(name == "tool.lookup_order.name" for name, _span in loaded.source_spans)


def test_loader_rejects_malformed_tool_definition(tmp_path: Path) -> None:
    tools = tmp_path / "bad-tools.json"
    tools.write_text('{"tools":[{"name":"broken","input_schema":[]}]}', encoding="utf-8")
    artifact = ToolDefinitionArtifact(
        kind=ArtifactKind.TOOL_DEFINITION,
        name="bad-tools",
        location=ArtifactLocation(path=str(tools)),
        provider="anthropic",
    )

    with pytest.raises(ArtifactLoadError, match="could not be ingested"):
        ArtifactLoader().load(artifact)


def test_cli_reports_tool_schema_ingestion_for_structured_fixture(capsys) -> None:
    exit_code = main(
        [
            "verify",
            "--config",
            "fixtures/structured_schemas/openai-tool-definition/promptabi.json",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [item for item in payload["diagnostics"] if item["rule_id"] == "tool-schema-ingestion"]

    assert exit_code == 0
    assert diagnostics
    assert diagnostics[0]["message"] == "tool-definition artifact 'openai-tool-bundle' ingested 2 openai tool schema(s)"
    assert diagnostics[0]["check_modes"] == ["complete", "sound"]
    assert diagnostics[0]["witness"]["steps"][1]["output"] == "lookup_order, refund_order"


def _schema(field: str, *, closed: bool = False) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "required": [field],
        "properties": {field: {"type": "string"}},
    }
    if closed:
        schema["additionalProperties"] = False
    return schema
