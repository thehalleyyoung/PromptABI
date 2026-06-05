import json
from pathlib import Path

import promptabi
from promptabi import (
    REQUIRED_FRAMEWORK_TRUNCATION_FAMILIES,
    FrameworkTruncationConformanceReport,
    build_framework_truncation_conformance_report,
    render_framework_truncation_conformance_json,
    render_framework_truncation_conformance_text,
    write_framework_truncation_conformance_manifest,
)
from promptabi.cli import main


SUITE = Path("fixtures/framework_truncation_conformance/suite.json")


def test_framework_truncation_conformance_covers_required_frameworks_and_real_budget_replay() -> None:
    report = build_framework_truncation_conformance_report(SUITE)
    cases = {case.case_id: case for case in report.cases}

    assert report.all_cases_passed is True
    assert report.case_count == 8
    assert report.missing_frameworks == ()
    assert {coverage.framework for coverage in report.framework_coverage if coverage.case_ids} == set(
        REQUIRED_FRAMEWORK_TRUNCATION_FAMILIES
    )
    assert len(report.manifest_sha256) == 64
    assert all(case.passed for case in report.cases)

    langchain = cases["langchain-conversation-buffer-window"]
    assert langchain.strategy == "oldest-message"
    assert langchain.kept_segments == ("system-policy", "latest-user-task")
    assert langchain.dropped_segments == ("early-user-task",)
    assert langchain.must_survive_status == "violated"

    transformers = cases["transformers-no-truncation"]
    assert transformers.strategy == "left"
    assert transformers.dropped_segments == ("system-policy",)
    assert transformers.must_survive_status == "violated"


def test_framework_truncation_conformance_renderers_writer_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = build_framework_truncation_conformance_report(SUITE)
    text = render_framework_truncation_conformance_text(report)
    payload = json.loads(render_framework_truncation_conformance_json(report))

    assert "PromptABI framework truncation conformance" in text
    assert "langchain: PASS" in text
    assert payload["all_cases_passed"] is True
    assert payload["manifest_sha256"] == report.manifest_sha256

    output = tmp_path / "framework-truncation-conformance.json"
    written = write_framework_truncation_conformance_manifest(output, suite_path=SUITE)
    assert json.loads(output.read_text(encoding="utf-8")) == written

    api_report = promptabi.framework_truncation_conformance_suite(SUITE)
    api_rendered = promptabi.framework_truncation_conformance_suite(SUITE, output_format="json")
    assert isinstance(api_report, FrameworkTruncationConformanceReport)
    assert json.loads(api_rendered)["all_cases_passed"] is True

    exit_code = main(["corpus", "framework-truncation-conformance", "--suite", str(SUITE), "--format", "text"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "custom-rag: PASS" in captured.out

    cli_output = tmp_path / "cli-framework-truncation-conformance.json"
    exit_code = main(
        [
            "corpus",
            "framework-truncation-conformance",
            "--suite",
            str(SUITE),
            "--output",
            str(cli_output),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote framework truncation conformance manifest" in captured.out
    assert json.loads(cli_output.read_text(encoding="utf-8"))["all_cases_passed"] is True


def test_framework_truncation_conformance_release_gate_reports_malformed_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    report = promptabi.verify_corpora(framework_truncation_conformance_suite_path=suite)
    check = next(check for check in report.checks if check.name == "framework-truncation-conformance")

    assert report.ok is False
    assert check.passed is False
    assert check.coverage_count == 0
    assert "framework truncation conformance suite could not be replayed" in check.summary
