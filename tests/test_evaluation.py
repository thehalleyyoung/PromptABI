import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.evaluation import (
    EvaluationError,
    load_evaluation_cases,
    render_evaluation_json,
    render_evaluation_text,
    run_evaluation,
)


def test_evaluation_corpus_expands_real_bug_suite_and_declares_metrics() -> None:
    methodology, metadata, cases = load_evaluation_cases()

    assert "labeled evaluation" in methodology
    assert "precision" in metadata["metrics"]
    assert "solver_result_quality" in metadata["metrics"]
    assert len(cases) >= 11
    assert any(case.case_id.startswith("real-bug:") for case in cases)
    assert all(case.expected_rule_ids for case in cases)


def test_default_evaluation_runs_real_code_paths_and_reports_scoped_metrics() -> None:
    report = run_evaluation()
    payload = report.to_dict()

    assert promptabi.__version__
    assert payload["passed"] is True
    assert payload["case_count"] >= 11
    assert payload["score"]["precision"] == 1.0
    assert payload["score"]["recall"] == 1.0
    assert payload["runtime_seconds"] >= 0
    assert payload["peak_memory_bytes"] > 0
    assert payload["witness_quality_score"] > 0
    assert payload["solver_result_quality"]["z3_backed_results"] >= 1
    assert payload["differential_agreement_rate"] > 0

    by_id = {case["id"]: case for case in payload["cases"]}
    assert by_id["stop-differential-vllm-agreement"]["differential_agreements"] == 1
    assert "stop-differential-mismatch" not in by_id["stop-differential-vllm-agreement"]["observed_rule_ids"]
    assert "rag-citation-loss" in by_id["rag-truncation-contracts"]["observed_rule_ids"]


def test_evaluation_renderers_and_cli_shape(tmp_path: Path, capsys) -> None:
    report = run_evaluation()
    json_payload = json.loads(render_evaluation_json(report))
    text_payload = render_evaluation_text(report)

    assert json_payload["score"]["f1"] == 1.0
    assert "PromptABI evaluation" in text_payload
    assert "precision/recall/f1" in text_payload

    exit_code = main(["corpus", "evaluation", "--format", "json"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["passed"] is True

    output = tmp_path / "evaluation.json"
    exit_code = main(["corpus", "evaluation", "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.is_file()
    assert "wrote evaluation report" in captured.out


def test_public_api_evaluate_corpus_returns_report_and_rendered_forms() -> None:
    report = promptabi.evaluate_corpus()
    rendered = promptabi.evaluate_corpus(output_format="json")

    assert isinstance(report, promptabi.EvaluationReport)
    assert json.loads(rendered)["passed"] is True


def test_evaluation_validation_rejects_unlabeled_cases(tmp_path: Path) -> None:
    corpus = tmp_path / "labeled_corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "methodology": "test",
                "cases": [
                    {
                        "id": "missing-labels",
                        "source": "config",
                        "config": "examples/minimal/promptabi.json",
                        "expected_rule_ids": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_evaluation_cases(corpus)
    except EvaluationError as exc:
        assert "expected_rule_ids" in str(exc)
    else:
        raise AssertionError("expected evaluation corpus validation failure")
