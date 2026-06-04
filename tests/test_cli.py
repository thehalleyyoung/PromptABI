import json

from promptabi.cli import main


def test_verify_text_output_passes_for_example_config(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI verification: minimal-chat-template" in captured.out
    assert "status: PASS" in captured.out
    assert captured.err == ""


def test_verify_json_output_is_stable(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/minimal/promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["diagnostics"][0]["rule_id"] == "repository-skeleton"
    assert "fingerprint" in payload["diagnostics"][0]
    assert payload["diagnostics"][0]["witness"]["steps"][0]["action"] == "load JSON config"
    assert list(payload) == ["config", "diagnostics", "ok"]


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
    assert "promptabiFingerprint" in result["partialFingerprints"]
