import json
from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    GrammarTokenizerEmptinessStatus,
    SchemaArtifact,
    TokenizerArtifact,
    analyze_tokenizer_grammar_emptiness,
)
from promptabi.cli import main
from promptabi.tokenizers import ByteLevelTokenizer


def test_tokenizer_grammar_emptiness_proves_json_schema_witness_survives(tmp_path: Path) -> None:
    schema_path = tmp_path / "object.schema.json"
    schema_path.write_text(json.dumps({"type": "object", "additionalProperties": False}), encoding="utf-8")
    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
    )
    schema_artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="object",
        location=ArtifactLocation(path=str(schema_path)),
    )

    report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        schema_artifact,
        ByteLevelTokenizer(),
    )

    assert report.status is GrammarTokenizerEmptinessStatus.SATISFIABLE
    assert report.witness is not None
    assert report.witness.grammar_text == "{}"
    assert report.witness.decoded_text == "{}"
    assert report.witness.token_ids == (123, 125)
    assert report.checked_candidates == 1
    assert "bounded-compiled-witness-dfa" in report.assumptions


def test_tokenizer_grammar_emptiness_detects_normalization_empty_product(tmp_path: Path) -> None:
    schema_path = tmp_path / "literal.schema.json"
    schema_path.write_text(json.dumps({"const": "OK"}), encoding="utf-8")
    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="lower-byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
        metadata=(("normalization", ("lowercase",)),),
    )
    schema_artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="literal",
        location=ArtifactLocation(path=str(schema_path)),
    )

    report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        schema_artifact,
        ByteLevelTokenizer(normalization=("lowercase",)),
    )

    assert report.status is GrammarTokenizerEmptinessStatus.EMPTY
    assert report.reason == "no bounded grammar witness survived the tokenizer encode/decode product"
    assert report.attempts[0].grammar_text == '"OK"'
    assert report.attempts[0].decoded_text == '"ok"'
    assert report.attempts[0].reason == "tokenizer normalization changed the grammar witness outside the accepted language"


def test_verify_reports_tokenizer_grammar_empty_product(tmp_path: Path, capsys) -> None:
    schema_path = tmp_path / "literal.schema.json"
    schema_path.write_text(json.dumps({"const": "OK"}), encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "empty-product",
                "checks": ["grammar-tokenizer-emptiness"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": str(schema_path),
                    },
                    "tokenizer": {
                        "kind": "tokenizer",
                        "uri": "memory://byte",
                        "family": "byte-level",
                        "metadata": {"normalization": ["lowercase"]},
                    },
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
    assert diagnostic["rule_id"] == "grammar-tokenizer-empty"
    assert diagnostic["check_modes"] == ["bounded", "sound"]
    assert "empty under tokenizer" in diagnostic["message"]
    assert any(step["action"] == "reject decoded text" for step in diagnostic["witness"]["steps"])
