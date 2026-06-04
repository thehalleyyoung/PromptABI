import json
from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    GrammarTokenizerAmbiguityKind,
    SchemaArtifact,
    TokenizerArtifact,
    analyze_tokenizer_grammar_ambiguity,
)
from promptabi.cli import main
from promptabi.tokenizers import ByteLevelTokenizer


def _schema_artifact(path: Path) -> SchemaArtifact:
    return SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="schema",
        location=ArtifactLocation(path=str(path)),
    )


def _tokenizer_artifact(**metadata) -> TokenizerArtifact:
    return TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
        metadata=tuple(metadata.items()),
    )


def test_tokenizer_grammar_ambiguity_detects_normalization_conflicting_values(tmp_path: Path) -> None:
    schema_path = tmp_path / "enum.schema.json"
    schema_path.write_text(json.dumps({"enum": ["OK", "ok"]}), encoding="utf-8")

    report = analyze_tokenizer_grammar_ambiguity(
        _tokenizer_artifact(normalization=("lowercase",)),
        _schema_artifact(schema_path),
        ByteLevelTokenizer(normalization=("lowercase",)),
    )

    conflict = next(
        finding
        for finding in report.findings
        if finding.kind is GrammarTokenizerAmbiguityKind.TOKEN_PATH_CONFLICT
    )
    assert report.checked_candidates >= 2
    assert conflict.structured_value == '"OK"'
    assert conflict.other_structured_value == '"ok"'
    assert conflict.token_ids == conflict.other_token_ids
    assert json.loads(conflict.decoded_text) == "ok"


def test_tokenizer_grammar_ambiguity_detects_unicode_and_whitespace_byte_aliases(tmp_path: Path) -> None:
    schema_path = tmp_path / "unicode.schema.json"
    schema_path.write_text(json.dumps({"const": "é"}), encoding="utf-8")

    report = analyze_tokenizer_grammar_ambiguity(
        _tokenizer_artifact(),
        _schema_artifact(schema_path),
        ByteLevelTokenizer(),
    )

    alias = next(
        finding
        for finding in report.findings
        if finding.kind is GrammarTokenizerAmbiguityKind.BYTE_ALIAS
    )
    assert alias.structured_value == '"é"'
    assert alias.other_structured_value == '"é"'
    assert alias.grammar_text != alias.other_grammar_text
    assert alias.token_ids != alias.other_token_ids
    assert json.loads(alias.grammar_text) == json.loads(alias.other_grammar_text) == "é"


def test_tokenizer_grammar_ambiguity_detects_added_token_aliases(tmp_path: Path) -> None:
    schema_path = tmp_path / "tool.schema.json"
    schema_path.write_text(json.dumps({"const": "tool"}), encoding="utf-8")

    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
        added_tokens=('"tool"',),
    )
    report = analyze_tokenizer_grammar_ambiguity(
        tokenizer_artifact,
        _schema_artifact(schema_path),
        ByteLevelTokenizer(added_tokens=('"tool"',)),
    )

    added_alias = next(
        finding
        for finding in report.findings
        if finding.kind is GrammarTokenizerAmbiguityKind.ADDED_TOKEN_ALIAS
    )
    assert 256 in added_alias.token_ids
    assert added_alias.other_token_ids == tuple(added_alias.grammar_text.encode("utf-8"))
    assert added_alias.decoded_text == added_alias.other_decoded_text == added_alias.grammar_text


def test_verify_reports_tokenizer_grammar_ambiguity(tmp_path: Path, capsys) -> None:
    schema_path = tmp_path / "enum.schema.json"
    schema_path.write_text(json.dumps({"enum": ["OK", "ok"]}), encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "ambiguous-product",
                "checks": ["grammar-tokenizer-ambiguity"],
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
    diagnostics = payload["diagnostics"]
    assert exit_code == 1
    assert any(
        diagnostic["rule_id"] == "grammar-tokenizer-ambiguity"
        and diagnostic["check_modes"] == ["bounded", "sound"]
        and "conflict" in diagnostic["message"]
        for diagnostic in diagnostics
    )
    assert any(
        step["action"] == "compare structured values"
        for diagnostic in diagnostics
        for step in diagnostic["witness"]["steps"]
    )
