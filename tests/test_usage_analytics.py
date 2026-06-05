import json

from promptabi.cli import main
from promptabi.usage_analytics import append_local_command_summary, summarize_local_command_usage


def test_verify_local_summary_records_only_sanitized_aggregate_metadata(tmp_path, capsys) -> None:
    summary_path = tmp_path / "usage.jsonl"

    exit_code = main(
        [
            "verify",
            "--config",
            "examples/minimal/promptabi.json",
            "--local-summary",
            str(summary_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert exit_code == 0
    assert "PromptABI verification: minimal-chat-template" in captured.out
    assert captured.err == ""
    assert payload["command"] == "verify"
    assert payload["exit_code"] == 0
    assert payload["metadata"] == {
        "artifact_count": 3,
        "check_count": 1,
        "diagnostics_total": 1,
        "errors": 0,
        "fail_on": "error",
        "format": "text",
        "info": 1,
        "ok": True,
        "warnings": 0,
    }
    assert "minimal-chat-template" not in serialized
    assert "examples/minimal" not in serialized
    assert "fingerprint" not in serialized
    assert "witness" not in serialized
    assert "answer.schema.json" not in serialized


def test_usage_summary_cli_aggregates_local_jsonl_without_raw_records(tmp_path, capsys) -> None:
    summary_path = tmp_path / "usage.jsonl"
    append_local_command_summary(path=summary_path, command="verify", exit_code=0, duration_ms=12, metadata={"ok": True})
    append_local_command_summary(path=summary_path, command="diff", exit_code=1, duration_ms=34, metadata={"ok": False})

    exit_code = main(["usage", "summary", "--path", str(summary_path), "--format", "json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["command_count"] == 2
    assert report["commands"] == {"diff": 1, "verify": 1}
    assert report["exit_codes"] == {"0": 1, "1": 1}
    assert report["total_duration_ms"] == 46
    assert "No telemetry is sent" in " ".join(report["privacy"])
    assert "metadata" not in report
    assert captured.err == ""


def test_usage_privacy_cli_states_no_prompt_or_network_transmission(capsys) -> None:
    exit_code = main(["usage", "privacy"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No telemetry is sent" in captured.out
    assert "Prompts, schemas, configs, constraints, witnesses" in captured.out
    assert "local JSONL" in captured.out
    assert captured.err == ""


def test_usage_summary_api_counts_missing_file_as_empty(tmp_path) -> None:
    report = summarize_local_command_usage(tmp_path / "missing.jsonl")

    assert report.command_count == 0
    assert report.commands == {}
    assert report.exit_codes == {}
    assert report.latest_timestamp is None
