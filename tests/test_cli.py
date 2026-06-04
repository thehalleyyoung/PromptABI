import json
from pathlib import Path

from promptabi.cli import main


def test_verify_text_output_passes_for_example_config(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI verification: minimal-chat-template" in captured.out
    assert "status: PASS" in captured.out
    assert captured.err == ""


def test_verify_discovers_config_from_nested_directory(tmp_path, monkeypatch, capsys) -> None:
    config_dir = tmp_path / "project"
    nested = config_dir / "app" / "prompts"
    nested.mkdir(parents=True)
    config = config_dir / "promptabi.json"
    schema = config_dir / "schema.json"
    cache_dir = tmp_path / "cache"
    schema.write_text("{}", encoding="utf-8")
    config.write_text(
        '{"name": "discovered", "artifacts": {"schema": "schema.json"}}',
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)

    exit_code = main(["verify", "--cache-dir", str(cache_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI verification: discovered" in captured.out
    assert cache_dir.is_dir()
    assert captured.err == ""


def test_verify_json_output_is_stable(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["diagnostics"][0]["rule_id"] == "repository-skeleton"
    assert payload["diagnostics"][0]["check_modes"] == ["heuristic"]
    assert "fingerprint" in payload["diagnostics"][0]
    assert payload["diagnostics"][0]["witness"]["steps"][0]["action"] == "load JSON config"
    assert list(payload) == ["config", "diagnostics", "ok"]


def test_verify_artifact_override_replaces_configured_location(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    existing = tmp_path / "schema.json"
    missing = tmp_path / "missing.schema.json"
    existing.write_text("{}", encoding="utf-8")
    config.write_text(
        f'{{"name": "override", "artifacts": {{"schema": "{missing.name}"}}}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--artifact",
            f"schema={existing}",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["config"]["artifacts"] == {"schema": str(existing)}


def test_verify_missing_artifact_fails_with_error(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{"name": "bad", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["rule_id"] == "artifact-missing"
    assert payload["diagnostics"][0]["witness"]["steps"][1]["output"] == "missing"


def test_verify_exit_code_policy_can_fail_on_any_diagnostic(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--fail-on", "any"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "INFO repository-skeleton [heuristic]" in captured.out
    assert captured.err == ""


def test_verify_quiet_suppresses_info_diagnostics(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: PASS" in captured.out
    assert "INFO repository-skeleton" not in captured.out


def test_verify_verbose_prints_workflow_metadata(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "promptabi-cache"

    exit_code = main(
        [
            "verify",
            "--config",
            "examples/minimal/promptabi.json",
            "--cache-dir",
            str(cache_dir),
            "--verbose",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"cache: {cache_dir}" in captured.out
    assert "artifacts: 3" in captured.out


def test_verify_sarif_output_is_code_scanning_compatible(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--format", "sarif"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    result = payload["runs"][0]["results"][0]
    assert exit_code == 0
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["name"] == "PromptABI"
    assert result["ruleId"] == "repository-skeleton"
    assert result["level"] == "note"
    assert result["properties"]["checkModes"] == ["heuristic"]
    assert payload["runs"][0]["tool"]["driver"]["rules"][0]["properties"]["checkModes"] == ["heuristic"]
    assert "promptabiFingerprint" in result["partialFingerprints"]


def test_verify_role_boundary_nonforgeability_reports_real_fixture(tmp_path, capsys) -> None:
    fixture = Path("fixtures/seed_corpus/llama/tokenizer_config.json").resolve()
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "llama-role-boundary",
                "checks": ["role-boundary-nonforgeability"],
                "artifacts": {
                    "llama": {
                        "kind": "chat-template",
                        "path": str(fixture),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    role_diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ]
    assert exit_code == 1
    assert role_diagnostics
    assert any("role-header 'assistant'" in diagnostic["message"] for diagnostic in role_diagnostics)
    assert any("<|start_header_id|>" in diagnostic["message"] for diagnostic in role_diagnostics)
    assert role_diagnostics[0]["check_modes"] == ["bounded", "sound"]
    witness_steps = role_diagnostics[0]["witness"]["steps"]
    assert any(step["action"] == "tokenize forged excerpt" for step in witness_steps)
    assert any(step["action"] == "locate forged boundary" for step in witness_steps)
    assert any("byte-level" in step.get("output", "") for step in witness_steps)


def test_verify_role_boundary_example_reports_structural_not_semantic_boundary(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/role-boundary/unsafe.promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ]

    assert exit_code == 1
    assert diagnostics
    assert any("role-header 'assistant'" in diagnostic["message"] for diagnostic in diagnostics)
    assert any("assistant-prefix '<|im_start|>'" in diagnostic["message"] for diagnostic in diagnostics)
    assert all("model will" not in diagnostic["message"] for diagnostic in diagnostics)
    witness_steps = diagnostics[0]["witness"]["steps"]
    assert any(step["action"] == "render forged boundary excerpt" for step in witness_steps)
    assert any(step["action"] == "tokenize forged excerpt" for step in witness_steps)
    assert any(step["action"] == "locate forged boundary" for step in witness_steps)


def test_verify_role_boundary_example_accepts_sanitized_template(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/role-boundary/safe.promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "role-boundary-nonforgeability"
    ] == []
