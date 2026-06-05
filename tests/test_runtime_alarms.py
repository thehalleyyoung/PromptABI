import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.lockfiles import build_lockfile, write_lockfile
from promptabi.runtime_alarms import (
    build_runtime_alarm_report,
    render_runtime_alarm_json,
    render_runtime_alarm_text,
)
from promptabi.runtime_attestation import build_runtime_attestation_report, render_runtime_attestation_json
from promptabi.session import VerificationSession


def _runtime_alarm_config(tmp_path: Path) -> Path:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object","properties":{"answer":{"type":"string"}}}', encoding="utf-8")
    config = {
        "name": "runtime-alarm-service",
        "checks": ["repository-skeleton"],
        "artifacts": {
            "system-prompt": {
                "kind": "prompt-segment",
                "uri": "memory://runtime/system-prompt",
                "segments": [{"name": "system", "role": "system", "required": True}],
            },
            "runtime-tokenizer": {
                "kind": "tokenizer",
                "uri": "memory://runtime/tokenizer",
                "family": "byte-bpe",
            },
            "chat-template": {
                "kind": "chat-template",
                "uri": "memory://runtime/template",
                "roles": ["system", "user", "assistant"],
            },
            "answer-schema": {
                "kind": "schema",
                "path": "answer.schema.json",
                "version": "schema-safe-v1",
            },
            "provider-config": {
                "kind": "provider-config",
                "uri": "memory://runtime/provider",
                "provider": "openai-compatible",
            },
        },
    }
    path = tmp_path / "promptabi.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path


def _write_lockfile(config_path: Path, lockfile_path: Path) -> None:
    config = load_config(config_path)
    session = VerificationSession(config)
    result = session.run()
    loaded, load_diagnostics = session.load_artifacts_with_diagnostics()
    assert load_diagnostics == ()
    write_lockfile(lockfile_path, build_lockfile(config, loaded, result.diagnostics, base_dir=lockfile_path.parent))


def _set_schema_version(config_path: Path, version: str) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["artifacts"]["answer-schema"]["version"] = version
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _attestation(config_path: Path, *, environment: str = "prod") -> dict[str, object]:
    report = build_runtime_attestation_report(
        config_path,
        bundle_key="runtime-secret",
        bundle_key_id="runtime-key",
        service="checkout-agent",
        environment=environment,
        runtime_contract_refs={"answer-schema": "schema-registry://answer@schema-safe-v1"},
    )
    return json.loads(render_runtime_attestation_json(report))


def test_runtime_alarm_report_passes_against_matching_real_evidence(tmp_path: Path) -> None:
    config = _runtime_alarm_config(tmp_path)
    lockfile = tmp_path / "promptabi.lock.json"
    _write_lockfile(config, lockfile)
    attestation = _attestation(config)

    report = build_runtime_alarm_report(
        attestation,
        lockfile=lockfile,
        policy_pack={
            "runtime_alarms": {
                "required_contract_families": ["prompt", "tokenizer", "template", "schema", "provider"],
                "allowed_environments": ["prod"],
                "required_signing_key_ids": ["runtime-key"],
            }
        },
        corpus_baseline={
            "artifact_baseline": [
                {"name": "answer-schema", "kind": "schema", "version": "schema-safe-v1"}
            ]
        },
        known_bad={"known_bad_artifacts": [{"name": "answer-schema", "kind": "schema", "version": "schema-bad-v0"}]},
    )

    assert report.ok is True
    assert json.loads(render_runtime_alarm_json(report))["alarm_counts"]["error"] == 0
    assert "PromptABI runtime alarms" in render_runtime_alarm_text(report)


def test_runtime_alarm_report_detects_lock_policy_corpus_and_known_bad_drift(tmp_path: Path) -> None:
    config = _runtime_alarm_config(tmp_path)
    lockfile = tmp_path / "promptabi.lock.json"
    _write_lockfile(config, lockfile)

    (tmp_path / "answer.schema.json").write_text(
        '{"type":"object","properties":{"answer":{"type":"string"},"debug":{"type":"string"}}}',
        encoding="utf-8",
    )
    _set_schema_version(config, "schema-bad-v2")
    attestation = _attestation(config, environment="staging")

    report = build_runtime_alarm_report(
        attestation,
        lockfile=lockfile,
        policy_pack={"runtime_alarms": {"allowed_environments": ["prod"], "required_signing_key_ids": ["runtime-key"]}},
        corpus_baseline={
            "artifact_baseline": [{"name": "answer-schema", "kind": "schema", "version": "schema-safe-v1"}]
        },
        known_bad={
            "known_bad_artifacts": [
                {
                    "name": "answer-schema",
                    "kind": "schema",
                    "version": "schema-bad-v2",
                    "reason": "schema revision admits debug leakage",
                }
            ]
        },
    )
    payload = json.loads(render_runtime_alarm_json(report))
    rules = {alarm["rule_id"] for alarm in payload["alarms"]}

    assert report.ok is False
    assert "runtime-lockfile-artifact-drift" in rules
    assert "runtime-policy-environment-denied" in rules
    assert "runtime-corpus-baseline-drift" in rules
    assert "runtime-known-bad-artifact" in rules
    assert payload["alarm_counts"]["error"] >= 3


def test_runtime_alarm_cli_and_public_api(tmp_path: Path, capsys) -> None:
    config = _runtime_alarm_config(tmp_path)
    lockfile = tmp_path / "promptabi.lock.json"
    attestation_path = tmp_path / "runtime-attestation.json"
    _write_lockfile(config, lockfile)
    attestation_path.write_text(json.dumps(_attestation(config), sort_keys=True), encoding="utf-8")

    exit_code = main(["runtime-alarms", str(attestation_path), "--lockfile", str(lockfile), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True

    api_payload = json.loads(
        promptabi.runtime_alarms(
            attestation_path,
            lockfile=lockfile,
            output_format="json",
        )
    )
    assert api_payload["report_version"] == "promptabi.runtime-alarms.v1"
