import json

from promptabi.cli import main
from promptabi.doctor import render_doctor_json, render_doctor_text, run_doctor


def test_doctor_reports_ready_example_environment(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "cache"

    exit_code = main(
        [
            "doctor",
            "--config",
            "examples/minimal/promptabi.json",
            "--cache-dir",
            str(cache_dir),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert exit_code == 0
    assert payload["ok"] is True
    assert checks["environment"]["status"] == "ok"
    assert checks["cache-health"]["details"]["path"] == str(cache_dir.resolve())
    assert checks["config-validity"]["details"]["name"] == "minimal-chat-template"
    assert checks["artifact-paths"]["status"] == "ok"
    assert checks["supported-backends"]["details"]["capability_counts"]["provider-adapter"] >= 1
    assert captured.err == ""


def test_doctor_fails_when_config_artifact_is_missing(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{"name": "doctor-missing", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(["doctor", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert exit_code == 1
    assert payload["ok"] is False
    assert checks["config-validity"]["status"] == "error"
    assert checks["artifact-paths"]["status"] == "error"
    assert checks["artifact-paths"]["details"]["missing_paths"] == [str(tmp_path / "missing.schema.json")]
    assert checks["setup-summary"]["details"]["errors"] == ["config-validity", "artifact-paths"]
    assert captured.err == ""


def test_doctor_reports_bad_plugin_without_losing_first_party_backends(capsys) -> None:
    exit_code = main(["doctor", "--config", "examples/minimal/promptabi.json", "--plugin", "does_not_exist", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert exit_code == 1
    assert checks["plugins"]["status"] == "error"
    assert "does_not_exist" in checks["plugins"]["details"]["error"]
    assert checks["supported-backends"]["details"]["artifact_loaders"] == ["first-party-reference-loader"]
    assert captured.err == ""


def test_doctor_renderers_are_stable(tmp_path) -> None:
    report = run_doctor(config_path="examples/minimal/promptabi.json", cache_dir=tmp_path / "cache")

    text = render_doctor_text(report)
    payload = json.loads(render_doctor_json(report))
    assert text.startswith("PromptABI doctor:\nstatus: PASS\n")
    assert "OK config-validity: Config loaded and artifact preflight completed." in text
    assert list(payload) == [
        "cache_dir",
        "checks",
        "config_path",
        "cwd",
        "ok",
        "platform",
        "promptabi_version",
        "python_executable",
        "python_version",
    ]
