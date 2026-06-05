import json

from promptabi import (
    CompositionalProofVerdict,
    compositional_proof_benchmark_cases,
    publish_compositional_proof_benchmarks,
    render_compositional_proof_benchmark_text,
    run_compositional_proof_benchmarks,
)
from promptabi.cli import main


def test_compositional_proof_benchmarks_all_pass_and_are_deterministic() -> None:
    first = run_compositional_proof_benchmarks()
    second = run_compositional_proof_benchmarks()

    assert first.ok
    assert first.total == len(compositional_proof_benchmark_cases())
    assert first.passed == first.total
    # The content-addressed manifest is stable across runs.
    assert first.manifest_sha256 == second.manifest_sha256
    assert len(first.families) >= 5


def test_each_case_observed_matches_expected_verdict() -> None:
    report = run_compositional_proof_benchmarks()
    for result in report.results:
        assert result.observed == result.expected, result.case_id
        if result.expected is CompositionalProofVerdict.REFUTED:
            assert result.witness_count >= 1


def test_benchmark_text_lists_every_case() -> None:
    report = run_compositional_proof_benchmarks()
    text = render_compositional_proof_benchmark_text(report)
    assert "compositional proof benchmarks" in text
    for case in compositional_proof_benchmark_cases():
        assert case.case_id in text


def test_publish_writes_manifest(tmp_path) -> None:
    manifest_path = publish_compositional_proof_benchmarks(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["version"] == "promptabi.compositional-proof-benchmarks.v1"
    assert payload["manifest_sha256"]
    assert len(payload["results"]) == len(compositional_proof_benchmark_cases())


def test_compositional_bench_cli(tmp_path, capsys) -> None:
    exit_code = main(
        ["compositional-bench", "--publish-dir", str(tmp_path), "--format", "json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert (tmp_path / "compositional-proof-benchmarks.json").exists()
