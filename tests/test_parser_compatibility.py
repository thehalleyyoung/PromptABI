import json
import re
from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    GrammarArtifact,
    ParserCompatibilityDirection,
    ParserCompatibilityStatus,
    SchemaArtifact,
    analyze_parser_compatibility,
)
from promptabi.cli import main


def test_json_schema_parser_compatibility_detects_broad_json_parser(tmp_path: Path) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string", "const": "ok"}},
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer",
        location=ArtifactLocation(path=str(schema_path)),
        metadata=(("parser_format", "json"),),
    )

    report = analyze_parser_compatibility(artifact)

    assert report.status is ParserCompatibilityStatus.MISMATCH
    assert any(
        observation.direction is ParserCompatibilityDirection.PARSER_BROADER
        and observation.sample.text == "{}"
        for observation in report.mismatches
    )
    assert report.assumptions[-1] == "heuristic-not-language-equivalence"


def test_json_schema_parser_compatibility_agrees_with_schema_validator(tmp_path: Path) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string", "const": "ok"}},
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer",
        location=ArtifactLocation(path=str(schema_path)),
    )

    report = analyze_parser_compatibility(artifact)

    assert report.status is ParserCompatibilityStatus.AGREEMENT
    assert report.parser_format == "json-schema"


def test_parser_compatibility_can_replay_grading_parser_override(tmp_path: Path) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer",
        location=ArtifactLocation(path=str(schema_path)),
    )

    report = analyze_parser_compatibility(artifact, parser_format_override="json")

    assert report.status is ParserCompatibilityStatus.MISMATCH
    assert report.parser_format == "json"
    assert any(
        observation.direction is ParserCompatibilityDirection.PARSER_BROADER
        and observation.sample.text == "{}"
        for observation in report.mismatches
    )


def test_xml_tool_call_parser_compatibility_detects_parser_broader_fixture(tmp_path: Path) -> None:
    grammar_path = tmp_path / "tool.regex"
    grammar_path.write_text(r"<tool_call name=\"refund\">\{\"id\":1\}</tool_call>", encoding="utf-8")
    artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="tool-call",
        location=ArtifactLocation(path=str(grammar_path)),
        grammar_type="regex",
        metadata=(
            ("parser_format", "xml-tool-call"),
            (
                "parser_compatibility",
                {
                    "samples": [
                        '<tool_call name="refund">{"id":1}</tool_call>',
                        '<tool_call name="refund">{"id":2}</tool_call>',
                    ]
                },
            ),
        ),
    )

    report = analyze_parser_compatibility(artifact)

    assert report.status is ParserCompatibilityStatus.MISMATCH
    assert any(
        observation.direction is ParserCompatibilityDirection.PARSER_BROADER
        and '"id":2' in observation.sample.text
        for observation in report.mismatches
    )


def test_markdown_fence_parser_compatibility_accepts_exact_fence_fixture(tmp_path: Path) -> None:
    grammar_path = tmp_path / "fence.regex"
    text = '```json\n{"answer":"ok"}\n```'
    grammar_path.write_text(re.escape(text), encoding="utf-8")
    artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="json-fence",
        location=ArtifactLocation(path=str(grammar_path)),
        grammar_type="regex",
        metadata=(
            ("parser_format", "markdown-fence"),
            ("fence_language", "json"),
            ("payload_format", "json"),
            ("parser_compatibility", {"samples": [text]}),
        ),
    )

    report = analyze_parser_compatibility(artifact)

    assert report.status is ParserCompatibilityStatus.AGREEMENT


def test_custom_delimited_parser_compatibility_requires_declared_delimiters(tmp_path: Path) -> None:
    grammar_path = tmp_path / "custom.regex"
    grammar_path.write_text(r"@@safe@@", encoding="utf-8")
    artifact = GrammarArtifact(
        kind=ArtifactKind.GRAMMAR,
        name="custom-delimited",
        location=ArtifactLocation(path=str(grammar_path)),
        grammar_type="regex",
        metadata=(
            ("parser_format", "custom-delimited"),
            ("start_delimiter", "@@"),
            ("end_delimiter", "@@"),
            ("parser_compatibility", {"samples": ["@@safe@@", "@@unsafe@@"]}),
        ),
    )

    report = analyze_parser_compatibility(artifact)

    assert report.status is ParserCompatibilityStatus.MISMATCH
    assert any(observation.sample.text == "@@unsafe@@" for observation in report.mismatches)


def test_cli_reports_parser_compatibility_mismatch_without_sound_mode(tmp_path: Path, capsys) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string", "const": "ok"}},
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "parser-compat",
                "checks": ["parser-compatibility"],
                "artifacts": {
                    "answer": {
                        "kind": "schema",
                        "path": str(schema_path),
                        "metadata": {"parser_format": "json"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostic = payload["diagnostics"][0]
    assert exit_code == 1
    assert diagnostic["rule_id"] == "parser-compatibility-mismatch"
    assert diagnostic["check_modes"] == ["heuristic"]
    assert "sound" not in diagnostic["check_modes"]
    assert diagnostic["witness"]["steps"][-1]["output"] == "parser-broader"
