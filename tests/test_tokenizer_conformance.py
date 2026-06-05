import json
from pathlib import Path

import promptabi
from promptabi import (
    REQUIRED_TOKENIZER_FAMILIES,
    REQUIRED_TOKENIZER_FEATURES,
    TokenizerConformanceReport,
    build_tokenizer_conformance_report,
    render_tokenizer_conformance_json,
    render_tokenizer_conformance_text,
    write_tokenizer_conformance_manifest,
)
from promptabi.cli import main


SUITE_PATH = Path("fixtures/tokenizer_conformance/suite.json")


def test_tokenizer_conformance_suite_covers_required_families_and_features() -> None:
    report = build_tokenizer_conformance_report(SUITE_PATH)
    observed_families = {coverage.family for coverage in report.family_coverage if coverage.case_ids}
    observed_features = {feature for case in report.cases for feature in case.features}

    assert report.all_cases_passed is True
    assert report.case_count >= 4
    assert report.sample_count >= 10
    assert observed_families == set(REQUIRED_TOKENIZER_FAMILIES)
    assert set(REQUIRED_TOKENIZER_FEATURES).issubset(observed_features)
    assert report.missing_families == ()
    assert report.missing_features == ()
    assert all(coverage.passed for coverage in report.family_coverage)
    assert len(report.manifest_sha256) == 64


def test_tokenizer_conformance_replays_real_bpe_and_unigram_code() -> None:
    report = build_tokenizer_conformance_report(SUITE_PATH)
    cases = {case.case_id: case for case in report.cases}

    bpe = cases["huggingface-byte-bpe-added-special-normalized"]
    unigram = cases["sentencepiece-unigram-normalized-space-detokenization"]

    assert bpe.differential_report.backend == "huggingface-tokenizers"
    assert bpe.passed is True
    assert bpe.sample_count == 3
    assert "added-tokens" in bpe.features
    assert unigram.differential_report.backend == "sentencepiece"
    assert unigram.passed is True
    assert unigram.sample_count == 2


def test_tokenizer_conformance_renderers_writer_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = build_tokenizer_conformance_report(SUITE_PATH)
    text = render_tokenizer_conformance_text(report)
    payload = json.loads(render_tokenizer_conformance_json(report))

    assert "PromptABI tokenizer family conformance" in text
    assert "bpe: PASS" in text
    assert payload["all_cases_passed"] is True
    assert payload["manifest_sha256"] == report.manifest_sha256

    output = tmp_path / "tokenizer-conformance.json"
    written = write_tokenizer_conformance_manifest(output, suite_path=SUITE_PATH)
    assert json.loads(output.read_text(encoding="utf-8")) == written

    api_report = promptabi.tokenizer_conformance_suite(SUITE_PATH)
    api_rendered = promptabi.tokenizer_conformance_suite(SUITE_PATH, output_format="json")
    assert isinstance(api_report, TokenizerConformanceReport)
    assert json.loads(api_rendered)["all_cases_passed"] is True

    exit_code = main(["corpus", "tokenizer-conformance", "--suite", str(SUITE_PATH), "--format", "text"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "unigram: PASS" in captured.out

    cli_output = tmp_path / "cli-tokenizer-conformance.json"
    exit_code = main(["corpus", "tokenizer-conformance", "--suite", str(SUITE_PATH), "--output", str(cli_output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote tokenizer conformance manifest" in captured.out
    assert json.loads(cli_output.read_text(encoding="utf-8"))["all_cases_passed"] is True


def test_tokenizer_conformance_release_gate_reports_malformed_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    report = promptabi.verify_corpora(tokenizer_conformance_suite_path=suite)
    check = next(check for check in report.checks if check.name == "tokenizer-conformance")

    assert report.ok is False
    assert check.passed is False
    assert check.coverage_count == 0
    assert "tokenizer conformance suite requires a non-empty cases array" in check.summary
