import json
from hashlib import sha256
from pathlib import Path

from promptabi.artifacts import ArtifactKind, ArtifactLocation, ArtifactProvenance, SchemaArtifact, artifact_from_config
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoadError, ArtifactLoader
from promptabi.session import VerificationSession
from promptabi.source import build_json_source_map


def test_json_source_map_tracks_nested_values_and_unicode_columns(tmp_path: Path) -> None:
    source = '{\n  "emoji": "☃",\n  "tools": [{"name": "lookup_order"}]\n}\n'
    source_map = build_json_source_map(source, tmp_path / "tools.json")

    emoji = source_map.span_for(("emoji",))
    tool_name = source_map.span_for(("tools", "0", "name"))

    assert emoji is not None
    assert emoji.start_line == 2
    assert emoji.start_column == 12
    assert tool_name is not None
    assert tool_name.start_line == 3
    assert tool_name.start_column == 22


def test_config_loader_attaches_promptabi_config_spans_to_every_artifact(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer_config.json"
    tokenizer.write_text('{"chat_template":"{{ messages[0].content }}"}', encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "spans",
                "artifacts": {
                    "schema": "schema.json",
                    "tok": {"kind": "tokenizer", "path": "tokenizer_config.json"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)

    schema_span = loaded.artifact_bundle.by_name("schema").source_span
    tokenizer_span = loaded.artifact_bundle.by_name("tok").source_span
    assert schema_span is not None
    assert schema_span.path == str(config)
    assert schema_span.start_line == 4
    assert tokenizer_span is not None
    assert tokenizer_span.path == str(config)
    assert tokenizer_span.start_line == 5


def test_missing_artifact_diagnostic_points_to_config_declaration(tmp_path: Path) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{\n  "name": "missing",\n  "artifacts": {\n    "schema": "missing.schema.json"\n  }\n}\n',
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config).run()

    assert result.ok is False
    diagnostic = result.diagnostics[0]
    assert diagnostic.rule_id == "artifact-missing"
    assert diagnostic.span is not None
    assert diagnostic.span.path == str(config)
    assert diagnostic.span.start_line == 4


def test_loader_records_field_spans_for_tokenizer_schema_and_tool_json(tmp_path: Path) -> None:
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        '{\n  "model_max_length": 4096,\n  "chat_template": "{{ bos_token }}{{ messages[0].content }}"\n}\n',
        encoding="utf-8",
    )
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{\n  "type": "object",\n  "required": ["answer"]\n}\n', encoding="utf-8")
    tools = tmp_path / "tools.json"
    tools.write_text('{\n  "tools": [{"name": "lookup_order"}]\n}\n', encoding="utf-8")

    tokenizer = ArtifactLoader().load(
        artifact_from_config(
            "tok",
            {
                "kind": "tokenizer",
                "path": tokenizer_config.name,
                "sha256": sha256(tokenizer_config.read_bytes()).hexdigest(),
            },
            base_dir=tmp_path,
        )
    )
    loaded_schema = ArtifactLoader().load(
        SchemaArtifact(
            kind=ArtifactKind.SCHEMA,
            name="schema",
            location=ArtifactLocation(path=str(schema)),
            provenance=ArtifactProvenance(sha256=sha256(schema.read_bytes()).hexdigest()),
        )
    )
    loaded_tools = ArtifactLoader().load(
        artifact_from_config(
            "tools",
            {
                "kind": "tool-definition",
                "path": tools.name,
                "sha256": sha256(tools.read_bytes()).hexdigest(),
                "tool_names": ["lookup_order"],
            },
            base_dir=tmp_path,
        )
    )

    tokenizer_spans = dict(tokenizer.source_spans)
    schema_spans = dict(loaded_schema.source_spans)
    tool_spans = dict(loaded_tools.source_spans)
    assert tokenizer_spans["chat_template"].start_line == 3
    assert schema_spans["required.0"].start_line == 3
    assert tool_spans["tools.0.name"].start_line == 2
    assert tokenizer.to_dict()["source_spans"]["chat_template"]["path"] == str(tokenizer_config)


def test_invalid_json_loader_error_carries_exact_parse_span(tmp_path: Path) -> None:
    schema = tmp_path / "broken.schema.json"
    schema.write_text('{\n  "type": "object",\n  "required": [\n}\n', encoding="utf-8")
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="schema",
        location=ArtifactLocation(path=str(schema)),
        provenance=ArtifactProvenance(version="local-test"),
    )

    try:
        ArtifactLoader().load(artifact)
    except ArtifactLoadError as exc:
        assert exc.span is not None
        assert exc.span.path == str(schema)
        assert exc.span.start_line == 4
        assert exc.span.start_column == 1
    else:  # pragma: no cover
        raise AssertionError("expected malformed JSON to fail")
