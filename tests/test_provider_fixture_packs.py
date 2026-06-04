import json
from pathlib import Path

import pytest

from promptabi import (
    ArtifactLoader,
    PROVIDER_FIXTURE_PACK_MANIFEST_VERSION,
    REQUIRED_PROVIDER_FIXTURE_FAMILIES,
    ProviderFixturePackError,
    analyze_provider_fixture_replay,
    build_provider_fixture_pack_manifest,
    load_provider_fixture_pack_corpus,
    write_provider_fixture_pack_manifest,
)
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.session import VerificationSession


FIXTURE_REPLAY_CONFIG = Path("fixtures/provider_fixture_packs/promptabi.json")


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


def test_provider_fixture_replay_analyzer_replays_all_recorded_packs() -> None:
    corpus = load_provider_fixture_pack_corpus()
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in corpus.artifact_bundle())

    report = analyze_provider_fixture_replay(loaded)
    cases_by_name = {case.artifact_name: case for case in report.cases}

    assert report.fixtures_checked == len(corpus.entries)
    assert report.findings == ()
    assert set(report.provider_families) >= set(REQUIRED_PROVIDER_FIXTURE_FAMILIES)
    assert len(report.replay_hash) == 64
    assert all(len(case.replay_hash) == 64 for case in report.cases)
    assert all(
        set(case.surfaces) == {"request", "response", "stops", "streaming", "errors", "limits", "edge_cases"}
        for case in report.cases
    )
    assert (
        corpus.by_id("openai-chat-completions").edge_case_ids
        == cases_by_name["openai-chat-completions-provider-fixture"].edge_cases
    )


def test_provider_fixture_replay_session_diagnostics_are_stable() -> None:
    result = VerificationSession.from_config_file(FIXTURE_REPLAY_CONFIG).run()

    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "provider-fixture-replay"]

    assert result.ok is True
    assert len(diagnostics) == 6
    assert all(diagnostic.severity.value == "info" for diagnostic in diagnostics)
    assert all(diagnostic.check_modes[0].value == "bounded" for diagnostic in diagnostics)
    assert any("openai-chat-completions" in diagnostic.message for diagnostic in diagnostics)
    assert all(
        any(step.action == "replay provider fixture pack" for step in diagnostic.witness.steps)
        for diagnostic in diagnostics
        if diagnostic.witness is not None
    )


def test_provider_fixture_replay_cli_reports_offline_replay(capsys) -> None:
    exit_code = main(["verify", "--config", str(FIXTURE_REPLAY_CONFIG), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [item for item in payload["diagnostics"] if item["rule_id"] == "provider-fixture-replay"]

    assert exit_code == 0
    assert captured.err == ""
    assert len(diagnostics) == 6
    assert any("provider fixture 'vllm-openai-server' replayed" in item["message"] for item in diagnostics)
    assert any(
        step["action"] == "record corpus replay hash"
        for item in diagnostics
        for step in item["witness"]["steps"]
    )


def test_provider_fixture_replay_flags_unresolved_edge_cases(tmp_path: Path) -> None:
    fixture = tmp_path / "pack.json"
    fixture.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_family": "openai",
                "request": {"method": "POST", "endpoint": "/v1/chat/completions", "fields": ["messages"]},
                "response": {
                    "fields": ["choices"],
                    "finish_reasons": ["stop"],
                    "tool_calls": {
                        "name_path": "choices[].message.tool_calls[].function.name",
                        "arguments_path": "choices[].message.tool_calls[].function.arguments",
                        "argument_encoding": "json-string",
                        "supports_parallel_tool_calls": False
                    }
                },
                "stops": {
                    "sequences": ["END"],
                    "finish_reason_path": "choices[].finish_reason",
                    "truncates_before_parser": True
                },
                "streaming": {
                    "delta_path": "choices[].delta",
                    "emits_argument_fragments": True,
                    "assembly_key": "tool_calls[].index"
                },
                "errors": {
                    "code_path": "error.code",
                    "message_path": "error.message",
                    "rate_limit_path": "error.type"
                },
                "limits": {"max_input_tokens": 100, "max_output_tokens": 10},
                "edge_cases": [
                    {
                        "id": "missing-surface",
                        "surface": "response.not_recorded",
                        "expected_behavior": "this should be rejected"
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "bad-provider-fixture",
                "checks": ["provider-fixture-replay"],
                "artifacts": {
                    "bad-provider": {
                        "kind": "provider-config",
                        "path": "pack.json",
                        "provider": "openai",
                        "api_family": "openai"
                    }
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession(load_config(config)).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "provider-fixture-replay"]

    assert result.ok is False
    assert any("edge-case-unresolved" in diagnostic.message for diagnostic in diagnostics)
    assert any(diagnostic.span and diagnostic.span.path.endswith("pack.json") for diagnostic in diagnostics)
