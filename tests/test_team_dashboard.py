import json

import promptabi
from promptabi.cli import main
from promptabi.team_dashboard import (
    append_dashboard_history,
    build_team_dashboard,
    load_dashboard_history,
    render_team_dashboard_json,
    render_team_dashboard_text,
)


def test_team_dashboard_tracks_real_risk_categories_and_history(tmp_path) -> None:
    risky = promptabi.run_verification("examples/role-boundary/unsafe.promptabi.json")
    suppressed = promptabi.run_verification("examples/policy/promptabi.json")
    drift = promptabi.run_verification("examples/end-to-end/provider-migration/buggy.promptabi.json")
    history_path = tmp_path / "dashboard.jsonl"
    older = build_team_dashboard((suppressed,), timestamp="2026-06-01T00:00:00+00:00").current
    append_dashboard_history(history_path, older)

    report = build_team_dashboard(
        (risky, suppressed, drift),
        corpus_report={"checks": [{"name": "seed-corpus", "passed": True}, {"name": "eval", "passed": False}]},
        history=load_dashboard_history(history_path),
        timestamp="2026-06-05T00:00:00+00:00",
    )

    payload = json.loads(render_team_dashboard_json(report))
    text = render_team_dashboard_text(report)
    by_source = {source["name"]: source for source in payload["current"]["sources"]}
    assert not report.ok
    assert payload["history_points"] == 1
    assert payload["current"]["totals"]["corpus_regressions"] == 1
    assert by_source["role-boundary-unsafe-chatml"]["open_risks"] >= 1
    assert by_source["policy-accepted-risk-demo"]["accepted_suppressions"] >= 1
    assert by_source["end-to-end-provider-migration-buggy"]["drift_warnings"] >= 1
    assert "open risks" in text
    assert "corpus regressions 1" in text


def test_team_dashboard_cli_records_jsonl_history_from_live_configs(tmp_path, capsys) -> None:
    history_path = tmp_path / "dashboard-history.jsonl"
    output_path = tmp_path / "dashboard.json"

    exit_code = main(
        [
            "dashboard",
            "--config",
            "examples/role-boundary/unsafe.promptabi.json",
            "--config",
            "examples/policy/promptabi.json",
            "--history",
            str(history_path),
            "--record",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert "wrote team dashboard" in captured.out
    assert captured.err == ""
    assert payload["current"]["totals"]["open_risks"] >= 1
    assert history_payload["schema_version"] == 1
    assert history_payload["totals"]["accepted_suppressions"] >= 1


def test_team_dashboard_api_renders_from_public_surface() -> None:
    rendered = promptabi.team_dashboard("examples/minimal/promptabi.json", output_format="json")

    payload = json.loads(rendered)
    assert payload["ok"] is True
    assert payload["current"]["sources"][0]["name"] == "minimal-chat-template"
    assert payload["current"]["totals"]["open_risks"] == 0
