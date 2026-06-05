import json
from pathlib import Path

from promptabi.api import editor_diagnostics
from promptabi.cli import main
from promptabi.editor_protocol import (
    EDITOR_PROTOCOL_VERSION,
    build_editor_diagnostic_report,
    render_editor_diagnostic_json,
)


def test_editor_protocol_groups_lsp_diagnostics_by_config_and_artifacts(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "editor-pass",
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": schema.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_editor_diagnostic_report(config_path=config, workspace_root=tmp_path)
    payload = json.loads(render_editor_diagnostic_json(report))

    assert payload["protocol"] == EDITOR_PROTOCOL_VERSION
    assert payload["ok"] is True
    assert {document["method"] for document in payload["documents"]} == {
        "textDocument/publishDiagnostics"
    }
    by_uri = {document["params"]["uri"]: document for document in payload["documents"]}
    assert config.resolve().as_uri() in by_uri
    assert schema.resolve().as_uri() in by_uri
    diagnostics = [
        diagnostic
        for document in payload["documents"]
        for diagnostic in document["params"]["diagnostics"]
    ]
    by_code = {diagnostic["code"]: diagnostic for diagnostic in diagnostics}
    assert all(diagnostic["source"] == "PromptABI" for diagnostic in diagnostics)
    assert by_code["repository-skeleton"]["severity"] == 3
    assert by_code["repository-skeleton"]["data"]["protocol"] == EDITOR_PROTOCOL_VERSION
    assert by_code["repository-skeleton"]["data"]["fingerprint"]
    assert by_code["repository-skeleton"]["range"]["start"] == {"line": 0, "character": 0}


def test_editor_protocol_reports_invalid_json_at_lsp_range(tmp_path: Path) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text('{"name": "bad",\n "artifacts": }', encoding="utf-8")

    report = build_editor_diagnostic_report(config_path=config, workspace_root=tmp_path)
    payload = report.to_dict()
    diagnostics = payload["documents"][0]["params"]["diagnostics"]

    assert payload["ok"] is False
    assert diagnostics[0]["code"] == "config-load-failed"
    assert diagnostics[0]["severity"] == 1
    assert diagnostics[0]["range"]["start"]["line"] == 1
    assert diagnostics[0]["range"]["start"]["character"] >= 13


def test_editor_diagnostics_api_can_render_json() -> None:
    rendered = editor_diagnostics(config_path="examples/minimal/promptabi.json", output_format="json")
    payload = json.loads(rendered)

    assert payload["protocol"] == EDITOR_PROTOCOL_VERSION
    assert payload["ok"] is True
    assert any(
        diagnostic["code"] == "repository-skeleton"
        for document in payload["documents"]
        for diagnostic in document["params"]["diagnostics"]
    )


def test_diagnostics_lsp_cli_emits_publish_diagnostics(capsys) -> None:
    exit_code = main(
        [
            "diagnostics",
            "lsp",
            "--config",
            "examples/minimal/promptabi.json",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["protocol"] == EDITOR_PROTOCOL_VERSION
    assert payload["documents"]
    assert captured.err == ""


def test_diagnostics_lsp_cli_returns_failure_for_real_missing_artifact(tmp_path: Path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{"name": "editor-missing", "artifacts": {"schema": "missing.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(["diagnostics", "lsp", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [
        diagnostic
        for document in payload["documents"]
        for diagnostic in document["params"]["diagnostics"]
    ]
    assert exit_code == 1
    assert any(diagnostic["code"] == "artifact-missing" for diagnostic in diagnostics)
    assert captured.err == ""
