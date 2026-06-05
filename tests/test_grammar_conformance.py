import json
from pathlib import Path

import promptabi
from promptabi import (
    REQUIRED_GRAMMAR_BACKENDS,
    GrammarConformanceReport,
    build_grammar_conformance_report,
    render_grammar_conformance_json,
    render_grammar_conformance_text,
    write_grammar_conformance_manifest,
)
from promptabi.cli import main


SUITE_PATH = Path("fixtures/grammar_conformance/suite.json")


def test_grammar_conformance_suite_covers_required_backends() -> None:
    report = build_grammar_conformance_report(SUITE_PATH)
    observed_backends = {coverage.backend_family for coverage in report.backend_coverage if coverage.case_ids}

    assert report.all_cases_passed is True
    assert report.case_count >= 10
    assert report.sample_count >= 50
    assert observed_backends == set(REQUIRED_GRAMMAR_BACKENDS)
    assert report.missing_backends == ()
    assert all(coverage.passed for coverage in report.backend_coverage)
    assert len(report.manifest_sha256) == 64


def test_grammar_conformance_replays_provider_native_structured_outputs() -> None:
    report = build_grammar_conformance_report(SUITE_PATH)
    provider = next(coverage for coverage in report.backend_coverage if coverage.backend_family == "provider-native")
    provider_cases = {
        case.case_id: case
        for case in report.differential_report.cases
        if case.backend_family == "provider-native"
    }

    assert provider.passed is True
    assert provider.declared_types == ("json-schema",)
    assert provider.accepted_samples == 3
    assert provider.rejected_samples == 6
    assert {
        "provider-native-openai-json-schema-response-format",
        "provider-native-tool-arguments-schema",
    } == set(provider.case_ids)
    assert all(case.status.value == "agreement" for case in provider_cases.values())


def test_grammar_conformance_renderers_writer_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = build_grammar_conformance_report(SUITE_PATH)
    text = render_grammar_conformance_text(report)
    payload = json.loads(render_grammar_conformance_json(report))

    assert "PromptABI grammar backend conformance" in text
    assert payload["all_cases_passed"] is True
    assert payload["manifest_sha256"] == report.manifest_sha256

    output = tmp_path / "grammar-conformance.json"
    written = write_grammar_conformance_manifest(output, suite_path=SUITE_PATH)
    assert json.loads(output.read_text(encoding="utf-8")) == written

    api_report = promptabi.grammar_conformance_suite(SUITE_PATH)
    api_rendered = promptabi.grammar_conformance_suite(SUITE_PATH, output_format="json")
    assert isinstance(api_report, GrammarConformanceReport)
    assert json.loads(api_rendered)["all_cases_passed"] is True

    exit_code = main(["corpus", "grammar-conformance", "--suite", str(SUITE_PATH), "--format", "text"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "provider-native: PASS" in captured.out

    cli_output = tmp_path / "cli-grammar-conformance.json"
    exit_code = main(["corpus", "grammar-conformance", "--suite", str(SUITE_PATH), "--output", str(cli_output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote grammar conformance manifest" in captured.out
    assert json.loads(cli_output.read_text(encoding="utf-8"))["all_cases_passed"] is True


def test_grammar_conformance_release_gate_reports_malformed_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    report = promptabi.verify_corpora(grammar_conformance_suite_path=suite)
    check = next(check for check in report.checks if check.name == "grammar-conformance")

    assert report.ok is False
    assert check.passed is False
    assert check.coverage_count == 0
    assert "grammar differential corpus requires a non-empty cases array" in check.summary
