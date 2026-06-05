import json
from pathlib import Path

import promptabi
from promptabi.beta import BetaProgramError, load_beta_cases, render_beta_program_json, render_beta_program_text, run_beta_program
from promptabi.cli import main


def test_beta_manifest_records_case_studies_issues_and_tuning_actions() -> None:
    methodology, cases = load_beta_cases()

    assert "Offline beta replay" in methodology
    assert len(cases) >= 7
    assert any(case.abstention_focus for case in cases)
    assert sum(len(case.issues) for case in cases) >= 6
    assert all(case.case_study.summary for case in cases)
    assert all(case.tuning_actions for case in cases)
    assert any(case.source_kind == "synthetic-abstention-message-regression" for case in cases)


def test_beta_program_replays_real_configs_and_actionable_abstentions() -> None:
    report = run_beta_program()
    payload = report.to_dict()

    assert payload["passed"] is True
    assert payload["project_count"] >= 7
    assert payload["upstream_issue_count"] >= 6
    assert payload["upstreamed_bug_count"] >= 1
    assert payload["false_positive_count"] == 0
    assert payload["missing_expected_count"] == 0
    assert payload["actionable_abstention_count"] >= 1

    by_id = {case["id"]: case for case in payload["cases"]}
    assert "role-boundary-nonforgeability" in by_id["chatml-role-boundary-beta"]["observed_rule_ids"]
    assert "rag-citation-loss" in by_id["rag-truncation-beta"]["observed_rule_ids"]
    assert "provider-migration" in by_id["provider-migration-beta"]["observed_rule_ids"]
    assert by_id["abstention-message-beta"]["actionable_abstention_rule_ids"] == ["stop-tokenizer-abstained"]
    assert by_id["abstention-message-beta"]["unactionable_abstention_rule_ids"] == []


def test_beta_program_renderers_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = run_beta_program()
    json_payload = json.loads(render_beta_program_json(report))
    text_payload = render_beta_program_text(report)

    assert json_payload["passed"] is True
    assert "PromptABI beta program" in text_payload
    assert "actionable abstentions" in text_payload

    exit_code = main(["corpus", "beta-report", "--format", "json"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["passed"] is True

    output = tmp_path / "beta-report.json"
    exit_code = main(["corpus", "beta-report", "--format", "json", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.is_file()
    assert "wrote beta report" in captured.out

    api_report = promptabi.beta_program()
    api_rendered = promptabi.beta_program(output_format="json")
    assert isinstance(api_report, promptabi.BetaProgramReport)
    assert json.loads(api_rendered)["passed"] is True


def test_beta_validation_rejects_missing_abstention_case(tmp_path: Path) -> None:
    manifest = tmp_path / "beta_program.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "methodology": "test",
                "cases": [
                    {
                        "id": "one-case",
                        "app_name": "One case",
                        "repository": "https://github.com/thehalleyyoung/PromptABI",
                        "source_kind": "synthetic-test",
                        "config": "examples/minimal/promptabi.json",
                        "labels": ["test"],
                        "expected_rule_ids": ["repository-skeleton"],
                        "expected_absent_rule_ids": ["check-failed"],
                        "issues": [
                            {
                                "title": "test",
                                "url": "https://github.com/thehalleyyoung/PromptABI",
                                "status": "local-fixture",
                                "rule_ids": ["repository-skeleton"],
                            }
                        ],
                        "tuning_actions": ["test tuning"],
                        "case_study": {
                            "summary": "summary",
                            "root_cause": "root",
                            "production_symptom": "symptom",
                            "fix": "fix",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_beta_cases(manifest)
    except BetaProgramError as exc:
        assert "abstention-focused" in str(exc)
    else:
        raise AssertionError("expected beta manifest validation failure")
