import json
from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    GrammarArtifact,
    GrammarDialect,
    GrammarIngestionError,
    JsonSchemaNodeKind,
    SchemaArtifact,
    ingest_grammar_text,
    normalize_json_schema_mapping,
)
from promptabi.loaders import ArtifactLoadError, ArtifactLoader


def test_json_schema_ingestion_records_supported_features_and_abstentions(tmp_path: Path) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok", "error"]},
                    "count": {"type": "integer", "minimum": 0},
                },
                "required": ["status"],
                "dependentRequired": {"count": ["status"]},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer",
        location=ArtifactLocation(path=str(schema_path)),
        dialect="json-schema-2020-12",
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "json-schema"
    metadata = dict(loaded.metadata)
    assert metadata["dialect"] == GrammarDialect.JSON_SCHEMA.value
    assert metadata["rule_names"] == ("property:count", "property:status", "schema")
    assert metadata["terminals"] == ("ok", "error")
    assert metadata["supported_fragment"] is False
    assert metadata["issue_codes"] == ("json-schema-unsupported-keyword",)
    assert metadata["root_kind"] == "object"
    assert metadata["node_count"] == 3
    assert metadata["property_paths"] == ("properties.count", "properties.status")
    assert any(name == "properties.status.enum.0" for name, _span in loaded.source_spans)


def test_json_schema_normalization_covers_nested_supported_fragment() -> None:
    result = normalize_json_schema_mapping(
        {
            "$defs": {
                "tag": {"type": "string", "minLength": 2, "maxLength": 12, "pattern": "^[a-z]+$"},
            },
            "type": "object",
            "required": ["answer", "scores"],
            "additionalProperties": False,
            "properties": {
                "answer": {"const": "yes"},
                "scores": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.5},
                },
                "tag": {"$ref": "#/$defs/tag"},
                "nullable_note": {"type": ["string", "null"]},
                "state": {"oneOf": [{"enum": ["open", "closed"]}, {"const": "unknown"}]},
            },
        }
    )

    assert result.supported_fragment is True
    assert result.root.kind is JsonSchemaNodeKind.OBJECT
    assert result.root.required == ("answer", "scores")
    assert result.root.additional_properties is False
    assert result.node_count == 11
    assert result.max_depth == 3
    assert set(result.features) >= {
        "$ref",
        "additionalProperties",
        "array",
        "array-constraints",
        "const",
        "enum",
        "numeric-constraints",
        "object",
        "oneOf",
        "properties",
        "required",
        "string-constraints",
        "union",
    }
    tag = next(property.schema for property in result.root.properties if property.name == "tag")
    assert tag.ref == "#/$defs/tag"
    assert tag.min_length == 2
    state = next(property.schema for property in result.root.properties if property.name == "state")
    assert state.kind is JsonSchemaNodeKind.UNION
    assert state.union_kind == "oneOf"
    assert state.variants[0].enum_values == ('"open"', '"closed"')


def test_json_schema_normalization_abstains_on_recursion_and_tuple_items() -> None:
    result = normalize_json_schema_mapping(
        {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/node"},
                    },
                }
            },
            "anyOf": [
                {"$ref": "#/$defs/node"},
                {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]},
            ],
        },
        max_ref_depth=3,
    )

    assert result.supported_fragment is False
    assert [issue.code for issue in result.issues] == [
        "json-schema-recursion-limit",
        "json-schema-tuple-items",
    ]
    assert result.root.kind is JsonSchemaNodeKind.UNION


def test_regex_ingestion_accepts_supported_subset_and_flags_lookaround() -> None:
    result = ingest_grammar_text(r"^(yes|no)-[0-9]+$", declared_type="regex")

    assert result.dialect is GrammarDialect.REGEX
    assert result.start_symbol == "regex"
    assert result.supported_fragment is True
    assert "alternation" in result.features
    assert "quantifier" in result.features

    lookaround = ingest_grammar_text(r"foo(?=bar)", declared_type="regex")
    assert lookaround.supported_fragment is False
    assert [issue.code for issue in lookaround.issues] == ["regex-lookaround"]


def test_ebnf_ingestion_extracts_rules_references_terminals_and_line_spans(tmp_path: Path) -> None:
    grammar_path = tmp_path / "grammar.ebnf"
    grammar_path.write_text(
        """
        root ::= greeting subject ;
        greeting = "hello" | "hi" ;
        subject ::= /[a-z]+/ ;
        """,
        encoding="utf-8",
    )
    artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="ebnf",
        location=ArtifactLocation(path=str(grammar_path)),
        grammar_type="ebnf",
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "grammar-ebnf"
    assert isinstance(loaded.artifact, GrammarArtifact)
    assert loaded.artifact.start_symbol == "root"
    assert loaded.artifact.rule_names == ("greeting", "root", "subject")
    metadata = dict(loaded.metadata)
    assert metadata["references"] == ("greeting", "subject")
    assert metadata["terminals"] == ("hello", "hi", "[a-z]+")
    assert any(name == "rules.root" for name, _span in loaded.source_spans)


def test_outlines_ingestion_supports_json_schema_regex_and_choices() -> None:
    schema = ingest_grammar_text(
        json.dumps({"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"const": "yes"}}}}),
        declared_type="outlines",
    )
    regex = ingest_grammar_text(json.dumps({"regex": r"tool_[a-z]+"}), declared_type="outlines")
    choices = ingest_grammar_text(json.dumps({"choices": ["red", "blue"]}), declared_type="outlines")

    assert schema.dialect is GrammarDialect.OUTLINES
    assert schema.features == ("json-schema",)
    assert schema.terminal_texts == ("yes",)
    assert regex.features == ("regex",)
    assert regex.rules[0].expression == r"tool_[a-z]+"
    assert choices.start_symbol == "choice"
    assert choices.terminal_texts == ("red", "blue")


def test_xgrammar_ingestion_supports_bnf_text_and_rules_object(tmp_path: Path) -> None:
    text_path = tmp_path / "xgrammar.json"
    text_path.write_text(
        json.dumps({"root_rule": "expr", "grammar": "expr ::= term ;\nterm ::= 'x' ;"}),
        encoding="utf-8",
    )
    text_artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="xgrammar-text",
        location=ArtifactLocation(path=str(text_path)),
        grammar_type="xgrammar",
    )

    loaded = ArtifactLoader().load(text_artifact)
    assert loaded.source_type == "grammar-xgrammar"
    assert loaded.artifact.start_symbol == "expr"
    assert dict(loaded.metadata)["rule_names"] == ("expr", "term")

    mapping = ingest_grammar_text(json.dumps({"root": "value", "rules": {"value": '"ok"'}}), declared_type="xgrammar")
    assert mapping.dialect is GrammarDialect.XGRAMMAR
    assert mapping.start_symbol == "value"


def test_llguidance_ingestion_supports_schema_regex_and_grammar_forms() -> None:
    schema = ingest_grammar_text(
        json.dumps({"llguidance": True, "json_schema": {"type": "string", "enum": ["safe"]}}),
        declared_type="llguidance",
    )
    regex = ingest_grammar_text(json.dumps({"regex": r"safe|unsafe"}), declared_type="llguidance")
    grammar = ingest_grammar_text(json.dumps({"lark_grammar": "start: 'ok'"}), declared_type="llguidance")

    assert schema.dialect is GrammarDialect.LLGUIDANCE
    assert schema.terminal_texts == ("safe",)
    assert regex.features == ("regex",)
    assert grammar.rule_names == ("start",)


def test_promptabi_ingestion_validates_start_rule_and_preserves_terminals(tmp_path: Path) -> None:
    grammar_path = tmp_path / "answer.promptabi.grammar.json"
    grammar_path.write_text(
        json.dumps(
            {
                "version": 1,
                "start": "answer",
                "rules": {"answer": '"yes" | maybe'},
                "terminals": {"literal": "yes"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="promptabi",
        location=ArtifactLocation(path=str(grammar_path)),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "grammar-promptabi"
    assert loaded.artifact.start_symbol == "answer"
    assert dict(loaded.metadata)["issue_codes"] == ("grammar-undefined-reference",)
    assert dict(loaded.metadata)["terminals"] == ("yes", "yes")


def test_malformed_grammars_fail_with_loader_diagnostics(tmp_path: Path) -> None:
    bad_regex = tmp_path / "bad.regex"
    bad_regex.write_text("(", encoding="utf-8")
    regex_artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="bad-regex",
        location=ArtifactLocation(path=str(bad_regex)),
        grammar_type="regex",
    )

    with pytest.raises(ArtifactLoadError, match="could not be ingested"):
        ArtifactLoader().load(regex_artifact)

    with pytest.raises(GrammarIngestionError, match="line 1"):
        ingest_grammar_text("root -> unsupported", declared_type="ebnf", path="inline.ebnf")
