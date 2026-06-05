import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.real_bug_benchmarks import (
    REQUIRED_REAL_BUG_CATEGORIES,
    RealBugBenchmarkError,
    build_real_bug_benchmark_manifest,
    load_real_bug_benchmark_suite,
    replay_real_bug_benchmarks,
    write_real_bug_benchmark_manifest,
)


def test_real_bug_benchmark_suite_covers_required_categories_and_provenance() -> None:
    suite = load_real_bug_benchmark_suite()

    assert promptabi.__version__
    assert set(suite.categories) == set(REQUIRED_REAL_BUG_CATEGORIES)
    assert len(suite.cases) >= 7
    for case in suite.cases:
        assert case.public_reference.startswith("https://github.com/")
        assert case.expected_rule_ids
        assert case.labels
        assert "synthetic" in case.source_kind or "public-github" in case.source_kind


def test_real_bug_benchmark_replays_all_labeled_failures_against_real_analyzers() -> None:
    results = replay_real_bug_benchmarks()

    assert len(results) >= 7
    assert all(result.passed for result in results)
    observed_by_category = {result.category: set(result.observed_rule_ids) for result in results}
    assert "role-boundary-nonforgeability" in observed_by_category["popular-template"]
    assert "tokenizer-differential-mismatch" in observed_by_category["tokenizer"]
    assert "provider-migration" in observed_by_category["provider-migration"]
    assert "rag-citation-loss" in observed_by_category["rag-truncation"]
    assert "training-target-role-alignment" in observed_by_category["training-pipeline"]


def test_real_bug_benchmark_manifest_records_replay_hashes_and_results(tmp_path: Path) -> None:
    manifest = build_real_bug_benchmark_manifest()
    output = tmp_path / "real-bug-benchmark.manifest.json"
    written = write_real_bug_benchmark_manifest(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert written == manifest
    assert payload == manifest
    assert manifest["manifest_version"] == 1
    assert manifest["all_cases_passed"] is True
    assert manifest["case_count"] >= 7
    assert len(manifest["manifest_sha256"]) == 64
    assert {entry["category"] for entry in manifest["entries"]} == set(REQUIRED_REAL_BUG_CATEGORIES)
    assert all(entry["observed_rule_ids"] for entry in manifest["entries"])


def test_real_bug_benchmark_cli_prints_and_writes_manifest(tmp_path: Path, capsys) -> None:
    exit_code = main(["corpus", "real-bug-benchmark"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["all_cases_passed"] is True

    output = tmp_path / "manifest.json"
    exit_code = main(["corpus", "real-bug-benchmark", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output.is_file()
    assert "wrote real-bug benchmark manifest" in captured.out


def test_real_bug_benchmark_validation_rejects_missing_category(tmp_path: Path) -> None:
    fixture = tmp_path / "benchmark.json"
    fixture.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "methodology": "test",
                "cases": [
                    {
                        "id": "only-one",
                        "category": "tokenizer",
                        "display_name": "Only one",
                        "public_reference": "https://github.com/thehalleyyoung/PromptABI",
                        "source_kind": "synthetic-test",
                        "bug_class": "test",
                        "labels": ["test"],
                        "expected_rule_ids": ["tokenizer-differential-mismatch"],
                        "replay": {
                            "method": "tokenizer-differential",
                            "text": "A",
                            "expected_token_ids": [66]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_real_bug_benchmark_suite(fixture)
    except RealBugBenchmarkError as exc:
        assert "missing required categories" in str(exc)
    else:
        raise AssertionError("expected missing benchmark category validation failure")
