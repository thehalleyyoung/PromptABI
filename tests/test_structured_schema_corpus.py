import json
from pathlib import Path

import pytest

from promptabi import (
    ArtifactLoader,
    REQUIRED_STRUCTURED_SCHEMA_SOURCES,
    StructuredSchemaCorpusError,
    build_structured_schema_corpus_manifest,
    load_structured_schema_corpus,
    validate_structured_schema_entry,
    write_structured_schema_corpus_manifest,
)
from promptabi.cli import main


def test_structured_schema_corpus_covers_required_sources_and_labels() -> None:
    corpus = load_structured_schema_corpus()

    assert set(corpus.source_categories) == set(REQUIRED_STRUCTURED_SCHEMA_SOURCES)
    assert {"schema", "grammar", "tool-definition"}.issubset(corpus.entry_types)
    assert corpus.by_id("open-source-agent-ticket").expected_status == "mismatch"
    assert corpus.by_id("anonymized-markdown-json").expected_status == "agreement"
    assert all(entry.labels for entry in corpus.entries)
    assert all(len(entry.artifact_sha256) == 64 for entry in corpus.entries)
    assert all(entry.metadata["download_required"] is False for entry in corpus.entries)
    assert all(entry.expected_rule_ids for entry in corpus.entries)


def test_structured_schema_corpus_entries_materialize_loadable_artifacts() -> None:
    corpus = load_structured_schema_corpus()
    loader = ArtifactLoader()

    loaded = [loader.load(artifact) for artifact in corpus.artifact_bundle()]

    assert len(loaded) == len(corpus.entries)
    assert {item.artifact.kind.value for item in loaded} == {"grammar", "schema", "tool-definition"}
    assert {item.source_type for item in loaded} >= {"json-schema", "tool-definition-schema"}
    assert all(item.resolved for item in loaded)


def test_structured_schema_corpus_replays_labeled_parser_compatibility() -> None:
    corpus = load_structured_schema_corpus()

    statuses = {
        entry.entry_id: validate_structured_schema_entry(entry).value
        for entry in corpus.entries
        if entry.entry_type in {"schema", "grammar"}
    }

    assert statuses == {
        "anonymized-markdown-json": "agreement",
        "open-source-agent-ticket": "mismatch",
        "synthetic-xml-tool-call": "mismatch",
    }


def test_structured_schema_corpus_cli_configs_match_expected_rule_ids(capsys) -> None:
    corpus = load_structured_schema_corpus()

    for entry in corpus.entries:
        exit_code = main(["verify", "--config", str(entry.promptabi_config_path), "--format", "json"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        rule_ids = {diagnostic["rule_id"] for diagnostic in payload["diagnostics"]}

        assert captured.err == ""
        assert set(entry.expected_rule_ids).issubset(rule_ids)
        assert exit_code == (1 if "parser-compatibility-mismatch" in rule_ids else 0)


def test_structured_schema_manifest_records_hashes_and_writes(tmp_path: Path) -> None:
    manifest = build_structured_schema_corpus_manifest()
    output = tmp_path / "structured-schema.manifest.json"
    written = write_structured_schema_corpus_manifest(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    ticket = next(entry for entry in manifest["entries"] if entry["id"] == "open-source-agent-ticket")

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == 1
    assert manifest["entry_count"] >= 4
    assert len(manifest["manifest_sha256"]) == 64
    assert ticket["source_category"] == "open-source-agent-reduction"
    assert len(ticket["metadata_sha256"]) == 64
    assert len(ticket["artifact_sha256"]) == 64
    assert len(ticket["fixture_sha256"]) == 64


def test_structured_schema_manifest_cli_prints_and_writes(tmp_path: Path, capsys) -> None:
    exit_code = main(["corpus", "structured-schema-manifest"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["entry_count"] >= 4

    output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "structured-schema-manifest", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.is_file()
    assert "wrote structured schema corpus manifest" in captured.out


def test_structured_schema_corpus_validation_rejects_missing_labels(tmp_path: Path) -> None:
    entry = tmp_path / "broken"
    entry.mkdir()
    (entry / "schema.json").write_text('{"type":"object"}', encoding="utf-8")
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "id": "broken",
                "entry_type": "schema",
                "source_category": "synthetic-stress",
                "display_name": "Broken",
                "source": "test",
                "license": "fixture-only",
                "fixture_revision": "test-v1",
                "upstream_reference": "test",
                "upstream_revision": "test",
                "download_required": False,
                "anonymized": False,
                "reproducibility_notes": "test",
                "artifact": "schema.json",
                "parser_format": "json",
                "parser_compatibility": {},
                "labels": [],
                "expected_parser_compatibility_status": "agreement",
                "expected_rule_ids": ["parser-compatibility-agreement"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StructuredSchemaCorpusError, match="labels"):
        load_structured_schema_corpus(tmp_path)
