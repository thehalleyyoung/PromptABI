import json
from pathlib import Path

import pytest

from promptabi import (
    ArtifactLoader,
    PROVIDER_FIXTURE_PACK_MANIFEST_VERSION,
    REQUIRED_PROVIDER_FIXTURE_FAMILIES,
    ProviderFixturePackError,
    build_provider_fixture_pack_manifest,
    load_provider_fixture_pack_corpus,
    write_provider_fixture_pack_manifest,
)
from promptabi.cli import main


def test_provider_fixture_pack_corpus_captures_required_surfaces() -> None:
    corpus = load_provider_fixture_pack_corpus()

    assert set(corpus.provider_families) >= set(REQUIRED_PROVIDER_FIXTURE_FAMILIES)
    assert corpus.by_id("openai-chat-completions").provider_family == "openai"
    assert corpus.by_id("litellm-router").edge_case_ids == (
        "provider-dependent-argument-encoding",
        "target-limit-varies",
        "normalized-error-shape",
    )
    assert all(entry.metadata["secrets_included"] is False for entry in corpus.entries)
    assert all(entry.metadata["download_required"] is False for entry in corpus.entries)
    assert all(len(entry.pack_sha256) == 64 for entry in corpus.entries)
    assert all(set(entry.captured_surfaces) == {"request", "response", "tool_calls", "stops", "streaming", "errors", "limits"} for entry in corpus.entries)


def test_provider_fixture_pack_entries_materialize_loadable_artifacts() -> None:
    corpus = load_provider_fixture_pack_corpus()
    loader = ArtifactLoader()

    loaded = [loader.load(artifact) for artifact in corpus.artifact_bundle()]

    assert len(loaded) == len(corpus.entries)
    assert {item.artifact.kind.value for item in loaded} == {"provider-config"}
    assert {item.source_type for item in loaded} == {"provider-config-snapshot"}
    assert all(item.resolved for item in loaded)
    assert {item.artifact.api_family for item in loaded} >= set(REQUIRED_PROVIDER_FIXTURE_FAMILIES)


def test_provider_fixture_pack_manifest_records_hashes_and_cli_writes(tmp_path: Path, capsys) -> None:
    manifest = build_provider_fixture_pack_manifest()
    output = tmp_path / "provider-fixtures.manifest.json"
    written = write_provider_fixture_pack_manifest(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    openai = next(entry for entry in manifest["entries"] if entry["id"] == "openai-chat-completions")

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == PROVIDER_FIXTURE_PACK_MANIFEST_VERSION
    assert manifest["entry_count"] >= 6
    assert len(manifest["manifest_sha256"]) == 64
    assert "streamed-json-string-fragments" in openai["edge_cases"]
    assert openai["secrets_included"] is False
    assert len(openai["metadata_sha256"]) == 64
    assert len(openai["pack_sha256"]) == 64
    assert len(openai["fixture_sha256"]) == 64

    exit_code = main(["corpus", "provider-fixture-manifest"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["manifest_sha256"] == manifest["manifest_sha256"]

    cli_output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "provider-fixture-manifest", "--output", str(cli_output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert cli_output.is_file()
    assert "wrote provider fixture pack manifest" in captured.out


def test_provider_fixture_pack_validation_rejects_secret_like_values(tmp_path: Path) -> None:
    entry = tmp_path / "openai-chat-completions"
    entry.mkdir()
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "id": "openai-chat-completions",
                "provider_family": "openai",
                "display_name": "Broken",
                "source": "test",
                "license": "fixture-only",
                "fixture_revision": "test-v1",
                "upstream_reference": "test",
                "upstream_revision": "test",
                "download_required": False,
                "secrets_included": False,
                "anonymized": True,
                "reproducibility_notes": "test",
            }
        ),
        encoding="utf-8",
    )
    (entry / "pack.json").write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_family": "openai",
                "request": {
                    "fields": ["messages"],
                    "authorization": "Bearer sk-testsecretvalue1234567890",
                },
                "response": {
                    "fields": ["choices"],
                    "tool_calls": {
                        "name_path": "choices[].message.tool_calls[].function.name",
                        "arguments_path": "choices[].message.tool_calls[].function.arguments",
                        "argument_encoding": "json-string",
                    },
                },
                "stops": {"sequences": ["</tool_call>"]},
                "streaming": {"emits_argument_fragments": True},
                "errors": {
                    "code_path": "error.code",
                    "message_path": "error.message",
                    "rate_limit_path": "error.type",
                },
                "limits": {"max_input_tokens": 1, "max_output_tokens": 1},
                "edge_cases": [{"id": "x", "surface": "request", "expected_behavior": "redacted"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderFixturePackError, match="secret-like field"):
        load_provider_fixture_pack_corpus(tmp_path)
