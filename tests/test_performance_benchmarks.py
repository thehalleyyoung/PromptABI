import json
from pathlib import Path

import pytest

from promptabi.benchmarks import (
    benchmark_cases,
    main,
    render_benchmark_json,
    repo_root,
    run_benchmarks,
)


def test_benchmark_registry_covers_required_performance_surfaces() -> None:
    names = {case.name for case in benchmark_cases()}

    assert names == {
        "tokenizer-analysis",
        "template-symbolic-execution",
        "grammar-emptiness",
        "stop-checks",
        "z3-static-contracts",
        "budget-checks",
        "corpus-wide-verification",
        "cache-cold-warm",
    }
    assert all(case.default_iterations > 0 for case in benchmark_cases())


def test_selected_benchmarks_execute_real_fixture_backed_code_paths() -> None:
    results = run_benchmarks(
        ("tokenizer-analysis", "grammar-emptiness", "stop-checks", "budget-checks", "cache-cold-warm"),
        iterations=1,
        root=repo_root(),
    )

    by_name = {result.name: result for result in results}
    assert by_name["tokenizer-analysis"].metrics["samples"] == 50
    assert by_name["grammar-emptiness"].metrics["status"] == "satisfiable"
    assert by_name["stop-checks"].metrics["overreach_findings"] > 0
    assert by_name["budget-checks"].metrics["segments"] == 4
    assert by_name["cache-cold-warm"].metrics["cache_reused"] is True
    assert all(result.iterations == 1 and result.seconds >= 0 for result in results)


def test_corpus_and_template_benchmarks_use_repository_fixtures() -> None:
    results = run_benchmarks(("template-symbolic-execution", "corpus-wide-verification"), iterations=1)
    by_name = {result.name: result for result in results}

    assert by_name["template-symbolic-execution"].metrics["templates"] >= 10
    assert by_name["template-symbolic-execution"].metrics["symbolic_paths"] > 0
    assert by_name["corpus-wide-verification"].metrics["configs"] == 6
    assert by_name["corpus-wide-verification"].metrics["diagnostics"] > 0


def test_benchmark_json_renderer_and_cli_shape(capsys) -> None:
    results = run_benchmarks(("z3-static-contracts",), iterations=1)
    payload = json.loads(render_benchmark_json(results))

    assert payload[0]["benchmark"] == "z3-static-contracts"
    assert payload[0]["metrics"]["findings"] > 0

    exit_code = main(["tokenizer-analysis", "--iterations", "1", "--repo-root", str(Path.cwd())])
    cli_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_payload[0]["benchmark"] == "tokenizer-analysis"
    assert cli_payload[0]["iterations"] == 1


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        run_benchmarks(("not-a-benchmark",), iterations=1)
