import json
from pathlib import Path

import pytest

from promptabi.cli import main
from promptabi.formal import SolverConclusion, SolverStatus
from promptabi.smt_benchmarks import (
    REQUIRED_SMT_BENCHMARK_CATEGORIES,
    SmtBenchmarkError,
    build_smt_benchmark_manifest,
    load_smt_benchmark_suite,
    render_smt_benchmark_text,
)


def test_smt_benchmark_replays_all_required_solver_outcomes() -> None:
    suite = load_smt_benchmark_suite()
    results = suite.replay()
    by_category = {result.case.category: result for result in results}

    assert set(by_category) == REQUIRED_SMT_BENCHMARK_CATEGORIES
    assert all(result.passed for result in results)
    assert by_category["satisfiable"].status is SolverStatus.SAT
    assert by_category["satisfiable"].conclusion is SolverConclusion.COUNTEREXAMPLE
    assert by_category["unsatisfiable"].status is SolverStatus.UNSAT
    assert by_category["unsatisfiable"].conclusion is SolverConclusion.UNSAT_CORE_PROOF
    assert by_category["timeout-prone"].status is SolverStatus.UNKNOWN
    assert by_category["timeout-prone"].checked_assignments >= 5
    assert by_category["unsupported"].status is SolverStatus.UNKNOWN
    assert by_category["unsupported"].backend == "z3"


def test_smt_benchmark_manifest_is_deterministic_and_redacted() -> None:
    manifest = build_smt_benchmark_manifest()
    text = render_smt_benchmark_text(manifest)
    second = build_smt_benchmark_manifest()

    assert manifest == second
    assert manifest["all_cases_passed"] is True
    assert manifest["case_count"] == 4
    assert set(manifest["categories"]) == REQUIRED_SMT_BENCHMARK_CATEGORIES
    assert len(str(manifest["manifest_sha256"])) == 64
    assert "PromptABI SMT benchmark corpus" in text
    assert "provider_credentials" not in json.dumps(manifest)


def test_smt_benchmark_cli_outputs_json_and_text(capsys) -> None:
    exit_code = main(["corpus", "smt-benchmark", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["all_cases_passed"] is True
    assert payload["case_count"] == 4
    assert captured.err == ""

    exit_code = main(["corpus", "smt-benchmark", "--format", "text"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "role-boundary-sat-forgery: sat/concrete-counterexample" in captured.out
    assert captured.err == ""


def test_smt_benchmark_rejects_missing_required_category(tmp_path: Path) -> None:
    source = json.loads(Path("fixtures/smt_benchmarks/benchmark.json").read_text(encoding="utf-8"))
    source["cases"] = [case for case in source["cases"] if case["category"] != "unsupported"]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(SmtBenchmarkError, match="missing required categories: unsupported"):
        load_smt_benchmark_suite(path)

