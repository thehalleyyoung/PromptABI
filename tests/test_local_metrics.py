import json

import promptabi
from promptabi.cli import main
from promptabi.local_metrics import build_local_metrics_report, render_local_metrics_json, render_local_metrics_text


def test_local_metrics_report_counts_real_diagnostics_without_raw_artifact_content() -> None:
    unsafe = promptabi.run_verification("examples/role-boundary/unsafe.promptabi.json")
    minimal = promptabi.run_verification("examples/minimal/promptabi.json")

    report = build_local_metrics_report((unsafe, minimal), generated_at="2026-06-05T00:00:00+00:00")
    payload = json.loads(render_local_metrics_json(report))
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["totals"]["configs"] == 2
    assert payload["totals"]["checks_configured"] >= 2
    assert payload["totals"]["artifacts_configured"] >= 4
    assert payload["counts"]["by_rule_id"]["role-boundary-nonforgeability"] >= 1
    assert payload["counts"]["by_artifact_kind"]["chat-template"] >= 1
    assert payload["runtimes_ms"]["role-boundary-nonforgeability"] >= 0
    assert "role-boundary-unsafe-chatml" not in serialized
    assert "examples/role-boundary" not in serialized
    assert "unsafe-tokenizer_config.json" not in serialized
    assert "<|im_start|>" not in serialized
    assert "witness_digest" not in serialized
    assert "rendered_strings" not in serialized
    assert "start_line" not in serialized
    assert "artifact names" in " ".join(payload["privacy"])


def test_local_metrics_text_is_dashboard_friendly_and_privacy_preserving() -> None:
    result = promptabi.run_verification("examples/end-to-end/provider-migration/buggy.promptabi.json")
    report = build_local_metrics_report((result,), generated_at="2026-06-05T00:00:00+00:00")

    text = render_local_metrics_text(report)

    assert "PromptABI local metrics export" in text
    assert "provider-migration" in text
    assert "artifact kinds:" in text
    assert "runtime" in text
    assert "buggy-target-anthropic" not in text
    assert "examples/end-to-end/provider-migration" not in text


def test_usage_metrics_cli_writes_json_export_from_real_configs(tmp_path, capsys) -> None:
    output_path = tmp_path / "metrics.json"

    exit_code = main(
        [
            "usage",
            "metrics",
            "--config",
            "examples/minimal/promptabi.json",
            "--config",
            "examples/role-boundary/unsafe.promptabi.json",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert exit_code == 0
    assert "wrote local metrics" in captured.out
    assert captured.err == ""
    assert payload["totals"]["configs"] == 2
    assert payload["counts"]["by_severity"]["error"] >= 1
    assert payload["counts"]["by_rule_id"]["role-boundary-nonforgeability"] >= 1
    assert "minimal-chat-template" not in serialized
    assert "answer.schema.json" not in serialized
    assert "witness_digest" not in serialized
    assert "rendered_strings" not in serialized


def test_local_metrics_public_api_renders_json() -> None:
    payload = json.loads(promptabi.local_metrics("examples/minimal/promptabi.json"))

    assert payload["ok"] is True
    assert payload["totals"]["configs"] == 1
    assert payload["counts"]["by_artifact_kind"]["schema"] == 1
    assert "privacy" in payload
