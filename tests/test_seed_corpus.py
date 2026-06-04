from pathlib import Path

import json

import pytest

from promptabi import (
    ArtifactLoader,
    REQUIRED_FAMILIES,
    SeedCorpusError,
    build_seed_corpus_manifest,
    load_seed_corpus,
    write_seed_corpus_manifest,
)
from promptabi.cli import main


def test_seed_corpus_covers_required_instruct_template_families() -> None:
    corpus = load_seed_corpus()

    assert set(corpus.families) == set(REQUIRED_FAMILIES)
    assert len(corpus.entries) >= 10
    assert corpus.by_id("llama").family == "llama"
    assert corpus.by_family("chatml")[0].chat_template.startswith("{% for message in messages %}")
    assert all(entry.roles for entry in corpus.entries)
    assert all(entry.sentinels for entry in corpus.entries)
    assert all(entry.expected_behaviors for entry in corpus.entries)
    assert all(len(entry.tokenizer_config_sha256) == 64 for entry in corpus.entries)
    assert all(entry.metadata["download_required"] is False for entry in corpus.entries)


def test_seed_corpus_entries_materialize_loadable_promptabi_artifacts() -> None:
    corpus = load_seed_corpus()
    bundle = corpus.artifact_bundle()
    loader = ArtifactLoader()

    loaded = [loader.load(artifact) for artifact in bundle]

    assert len(bundle.artifacts) == len(corpus.entries) * 2
    assert {artifact.kind.value for artifact in bundle.artifacts} == {"chat-template", "tokenizer"}
    assert {item.source_type for item in loaded} == {"local-file", "tokenizer-directory"}
    assert all(item.resolved for item in loaded)
    assert all(item.warnings == () for item in loaded)
    assert {
        artifact.name
        for artifact in bundle.artifacts
        if artifact.name.endswith("-chat-template")
    } == {f"{entry.entry_id}-chat-template" for entry in corpus.entries}


def test_seed_corpus_validation_rejects_inconsistent_sentinel(tmp_path: Path) -> None:
    entry = tmp_path / "chatml"
    entry.mkdir()
    (entry / "metadata.json").write_text(
        """{
  "id": "chatml",
  "family": "chatml",
  "display_name": "Broken",
  "source": "synthetic-minimized",
  "license": "fixture-only",
  "fixture_revision": "seed-v1",
  "upstream_reference": "test",
  "upstream_revision": "test-seed-v1",
  "download_required": false,
  "reproducibility_notes": "test fixture",
  "roles": ["user", "assistant"],
  "sentinels": ["<missing>"],
  "supports_generation_prompt": true,
  "expected_behaviors": ["role_headers"]
}""",
        encoding="utf-8",
    )
    (entry / "tokenizer_config.json").write_text(
        """{
  "chat_template": "{% for message in messages %}<s>{{ message['content'] }}</s>{% endfor %}{% if add_generation_prompt %}<s>{% endif %}",
  "bos_token": "<s>",
  "eos_token": "</s>"
}""",
        encoding="utf-8",
    )

    with pytest.raises(SeedCorpusError, match="sentinel '<missing>'"):
        load_seed_corpus(tmp_path)


def test_seed_corpus_manifest_records_hashes_provenance_and_expected_behaviors(tmp_path: Path) -> None:
    manifest = build_seed_corpus_manifest()
    output = tmp_path / "seed-corpus.manifest.json"
    written = write_seed_corpus_manifest(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    llama = next(entry for entry in manifest["entries"] if entry["id"] == "llama")

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == 1
    assert manifest["entry_count"] >= 10
    assert len(manifest["manifest_sha256"]) == 64
    assert llama["upstream_revision"] == "minimized-llama-seed-v1"
    assert llama["download_required"] is False
    assert "header_role_boundaries" in llama["expected_behaviors"]
    assert len(llama["metadata_sha256"]) == 64
    assert len(llama["tokenizer_config_sha256"]) == 64
    assert len(llama["fixture_sha256"]) == 64


def test_corpus_manifest_cli_prints_and_writes_manifest(tmp_path: Path, capsys) -> None:
    exit_code = main(["corpus", "manifest"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["entry_count"] >= 10

    output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "manifest", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.is_file()
    assert "wrote seed corpus manifest" in captured.out
