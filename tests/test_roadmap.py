import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.roadmap import (
    ROADMAP_REPORT_VERSION,
    build_annual_corpus_refresh_report,
    build_award_submission_report,
    build_historical_trend_report,
    build_research_agenda_report,
    build_teaching_materials_report,
    next_research_agenda_steps,
    render_roadmap_json,
    render_roadmap_markdown,
    render_roadmap_text,
)


def test_historical_trends_use_real_dashboard_history_and_corpus_coverage() -> None:
    report = build_historical_trend_report()
    payload = report.payload

    assert report.ok is True
    assert payload["source"] == "repository-fixture-history"
    assert payload["history_points"] >= 2
    assert payload["trend"]["open_risks"] < 0
    assert payload["coverage"]["model_families"] >= 10
    assert payload["coverage"]["provider_fixture_packs"] >= 6
    assert "provider-migration" in payload["coverage"]["pipelines"]


def test_annual_corpus_refresh_is_read_only_and_preserves_benchmarks() -> None:
    report = build_annual_corpus_refresh_report()
    payload = report.payload

    assert report.ok is True
    assert payload["read_only"] is True
    assert payload["corpora"]["real_bug_cases"] > 0
    assert {action["id"] for action in payload["annual_actions"]} == {
        "retire-obsolete-artifacts",
        "add-new-model-families",
        "update-provider-semantics",
        "preserve-old-benchmarks",
    }
    assert any("leaderboard" in command for command in payload["release_gates"])


def test_award_and_teaching_materials_are_evidence_backed() -> None:
    award = build_award_submission_report()
    teaching = build_teaching_materials_report()

    assert award.ok is True
    evidence = award.payload["impact_evidence"]
    assert evidence["comparative_case_count"] == evidence["promptabi_detected_cases"]
    assert evidence["baseline_classes_with_misses"] >= 1
    assert "does not prove model intent or sampled behavior" in award.payload["limitations"]

    assert teaching.ok is True
    assert teaching.payload["course_length_weeks"] == 5
    assert len(teaching.payload["proof_notebooks"]) >= 5
    assert any(lab["id"] == "smt-contracts" for lab in teaching.payload["labs"])


def test_research_agenda_has_next_100_steps_and_renderers_are_stable() -> None:
    steps = next_research_agenda_steps()
    report = build_research_agenda_report()
    payload = json.loads(render_roadmap_json(report))

    assert len(steps) == 100
    assert steps[0]["number"] == 200
    assert steps[-1]["number"] == 299
    assert payload["manifest_version"] == ROADMAP_REPORT_VERSION
    assert payload["payload"]["categories"]["compositional verification"] == 20
    assert "Long-term research agenda" in render_roadmap_markdown(report)
    assert "PromptABI Long-term research agenda" in render_roadmap_text(report)


def test_roadmap_cli_and_public_api_write_markdown(tmp_path: Path, capsys) -> None:
    output = tmp_path / "research-agenda.md"
    exit_code = main(["roadmap", "research-agenda", "--format", "markdown", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "wrote roadmap report" in captured.out
    assert "299." in output.read_text(encoding="utf-8")

    api_payload = json.loads(promptabi.historical_trends(output_format="json"))
    assert api_payload["kind"] == "historical-trends"
    assert api_payload["ok"] is True

