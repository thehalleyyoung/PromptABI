import json
from hashlib import sha256
from pathlib import Path

from promptabi.cli import main
from promptabi.config import load_config
from promptabi.lockfiles import (
    LOCKFILE_VERSION,
    build_lockfile,
    compare_lockfile,
    load_lockfile,
    lockfile_to_json,
)
from promptabi.session import VerificationSession


def test_lockfile_captures_artifact_hashes_versions_and_diagnostic_baseline(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "locked",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "answer.schema.json",
                        "sha256": sha256(schema.read_bytes()).hexdigest(),
                        "version": "fixture-v1",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    session = VerificationSession(config)
    result = session.run()
    loaded, load_diagnostics = session.load_artifacts_with_diagnostics()

    lockfile = build_lockfile(config, loaded, result.diagnostics)
    payload = json.loads(lockfile_to_json(lockfile))

    assert load_diagnostics == ()
    assert payload["lockfile_version"] == LOCKFILE_VERSION
    assert payload["config_name"] == "locked"
    assert payload["artifacts"][0]["sha256"] == sha256(schema.read_bytes()).hexdigest()
    assert payload["artifacts"][0]["version"] == "fixture-v1"
    assert payload["diagnostic_baseline"][0]["rule_id"] == "repository-skeleton"
    assert payload["library_versions"]["python"]


def test_compare_lockfile_reports_real_artifact_drift(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    original_hash = sha256(schema.read_bytes()).hexdigest()
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "drift",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "answer.schema.json",
                        "version": "reviewed-v1",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    session = VerificationSession(config)
    result = session.run()
    loaded, _ = session.load_artifacts_with_diagnostics()
    lockfile = build_lockfile(config, loaded, result.diagnostics)

    schema.write_text('{"type":"object","required":["answer"]}', encoding="utf-8")
    new_session = VerificationSession(load_config(config_path))
    new_result = new_session.run()
    new_loaded, _ = new_session.load_artifacts_with_diagnostics()
    diagnostics = compare_lockfile(lockfile, new_session.config, new_loaded, new_result.diagnostics)

    artifact_drift = [diagnostic for diagnostic in diagnostics if diagnostic.rule_id == "lockfile-artifact-drift"]
    assert artifact_drift
    assert artifact_drift[0].severity.value == "error"
    assert dict(artifact_drift[0].properties)["expected"] == original_hash
    assert dict(artifact_drift[0].properties)["actual"] == sha256(schema.read_bytes()).hexdigest()


def test_cli_writes_and_requires_lockfile_against_real_config(tmp_path: Path, capsys) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "cli-lock",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "answer.schema.json",
                        "sha256": sha256(schema.read_bytes()).hexdigest(),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lockfile_path = tmp_path / "promptabi.lock.json"

    write_exit = main(
        [
            "verify",
            "--config",
            str(config_path),
            "--lockfile",
            str(lockfile_path),
            "--write-lockfile",
            "--format",
            "json",
        ]
    )
    written = capsys.readouterr()
    assert write_exit == 0
    assert written.err == ""
    assert lockfile_path.is_file()
    assert load_lockfile(lockfile_path).config_name == "cli-lock"

    require_exit = main(
        [
            "verify",
            "--config",
            str(config_path),
            "--lockfile",
            str(lockfile_path),
            "--require-lockfile",
            "--format",
            "json",
        ]
    )
    required = capsys.readouterr()
    payload = json.loads(required.out)
    assert require_exit == 0
    assert required.err == ""
    assert any(diagnostic["rule_id"] == "lockfile-verified" for diagnostic in payload["diagnostics"])


def test_cli_lockfile_enforcement_fails_on_drift(tmp_path: Path, capsys) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "cli-lock-drift",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "answer.schema.json",
                        "version": "reviewed-v1",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lockfile_path = tmp_path / "promptabi.lock.json"

    assert main(["verify", "--config", str(config_path), "--lockfile", str(lockfile_path), "--write-lockfile"]) == 0
    capsys.readouterr()
    schema.write_text('{"type":"object","properties":{"answer":{"type":"string"}}}', encoding="utf-8")

    exit_code = main(
        [
            "verify",
            "--config",
            str(config_path),
            "--lockfile",
            str(lockfile_path),
            "--require-lockfile",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(diagnostic["rule_id"] == "lockfile-artifact-drift" for diagnostic in payload["diagnostics"])
