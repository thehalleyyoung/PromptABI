import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.corpus_verification import (
    CorpusVerificationThresholds,
    render_corpus_verification_json,
    render_corpus_verification_text,
    run_corpus_verification,
)


def test_corpus_verification_release_gate_runs_full_real_corpora() -> None:
    report = run_corpus_verification()
    payload = report.to_dict()
    checks = {check["name"]: check for check in payload["checks"]}

    assert report.ok is True
    assert payload["coverage_count"] >= 35
    assert checks["seed-corpus"]["coverage_count"] >= 10
    assert checks["structured-schema-corpus"]["metrics"]["config_replays"] >= 4
    assert checks["provider-fixture-corpus"]["coverage_count"] >= 6
    assert checks["real-bug-benchmark"]["coverage_count"] >= 7
    assert checks["evaluation-fixture-pack"]["coverage_count"] == 5
    assert set(checks["evaluation-fixture-pack"]["metrics"]["bug_classes"]) == {
        "parser",
        "role-boundary",
        "stop-string",
        "tokenizer-mismatch",
        "truncation",
    }
    assert checks["labeled-evaluation"]["metrics"]["f1"] == 1.0
    assert checks["labeled-evaluation"]["metrics"]["differential_cases"] > 0
    assert checks["labeled-evaluation"]["metrics"]["z3_backed_results"] > 0
    assert checks["smt-benchmark"]["coverage_count"] == 4
    assert set(checks["smt-benchmark"]["metrics"]["categories"]) == {
        "satisfiable",
        "timeout-prone",
        "unsatisfiable",
        "unsupported",
    }
    assert checks["performance-thresholds"]["passed"] is True


def test_corpus_verification_renderers_and_cli_shape(tmp_path: Path, capsys) -> None:
    report = run_corpus_verification()
    text = render_corpus_verification_text(report)
    payload = json.loads(render_corpus_verification_json(report))

    assert "PromptABI corpus verification" in text
    assert "seed-corpus" in text
    assert payload["ok"] is True
    assert payload["check_count"] == 8

    exit_code = main(["corpus", "verify", "--format", "json"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["ok"] is True

    output = tmp_path / "corpus-verification.json"
    exit_code = main(["corpus", "verify", "--format", "json", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.is_file()
    assert "wrote corpus verification report" in captured.out


def test_corpus_verification_fails_release_blocking_threshold(capsys) -> None:
    exit_code = main(["corpus", "verify", "--max-runtime-seconds", "0.000001"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "performance-thresholds: FAIL" in captured.out
    assert captured.err == ""


def test_public_api_verifies_corpora_and_renders() -> None:
    report = promptabi.verify_corpora()
    rendered = promptabi.verify_corpora(
        thresholds=CorpusVerificationThresholds(min_witness_quality=0.5),
        output_format="json",
    )

    assert isinstance(report, promptabi.CorpusVerificationReport)
    assert report.ok is True
    assert json.loads(rendered)["ok"] is True
