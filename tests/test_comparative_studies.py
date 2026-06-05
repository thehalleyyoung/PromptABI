import json
from pathlib import Path

from promptabi.cli import main
from promptabi.comparative_studies import (
    BASELINE_CLASSES,
    build_comparative_study_report,
    render_comparative_study_json,
    render_comparative_study_markdown,
    render_comparative_study_text,
)


def test_comparative_study_runs_live_evidence_against_all_baseline_classes() -> None:
    report = build_comparative_study_report()
    payload = report.to_dict()

    assert payload["passed"] is True
    assert payload["case_count"] >= 16
    assert payload["promptabi_detected_cases"] == payload["case_count"]
    assert payload["evaluation_score"]["precision"] == 1.0
    assert payload["evaluation_score"]["recall"] == 1.0
    assert payload["real_bug_cases"] >= 7
    assert {row["baseline"]["id"] for row in payload["baselines"]} == {
        baseline.baseline_id for baseline in BASELINE_CLASSES
    }
    assert all(row["missed_case_count"] > 0 for row in payload["baselines"])

    by_baseline = {row["baseline"]["id"]: row for row in payload["baselines"]}
    assert "role-boundary-nonforgeability" not in by_baseline["prompt-linter"]["promptabi_only_rule_ids"]
    assert "provider-migration" in by_baseline["prompt-linter"]["promptabi_only_rule_ids"]
    assert "real-bug:byte-tokenizer-added-special-drift" in by_baseline["tokenizer-diff"]["covered_case_ids"]
    assert by_baseline["generic-static-analyzer"]["covered_case_count"] == 0


def test_comparative_study_renderers_are_stable_and_paper_ready() -> None:
    report = build_comparative_study_report()
    json_payload = json.loads(render_comparative_study_json(report))
    markdown = render_comparative_study_markdown(report)
    text = render_comparative_study_text(report)

    assert json_payload["manifest_version"] == 1
    assert "PromptABI comparative study" in markdown
    assert "| Baseline class |" in markdown
    assert "PromptABI-only rules" in markdown
    assert "status: PASS" in text
    assert "Prompt linters" in text


def test_comparative_study_cli_prints_and_writes_report(tmp_path: Path, capsys) -> None:
    exit_code = main(["corpus", "comparative-study", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert payload["baselines"]

    output = tmp_path / "comparative-study.md"
    exit_code = main(["corpus", "comparative-study", "--format", "markdown", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.read_text(encoding="utf-8").startswith("# PromptABI comparative study")
    assert "wrote comparative study report" in captured.out
