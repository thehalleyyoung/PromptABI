import json
from pathlib import Path

import promptabi
from promptabi.benchmark_leaderboards import (
    BENCHMARK_LEADERBOARD_VERSION,
    build_benchmark_leaderboard,
    render_benchmark_leaderboard_json,
    render_benchmark_leaderboard_text,
)
from promptabi.cli import main


def test_benchmark_leaderboard_scores_real_evaluation_performance_and_solver_assets() -> None:
    report = build_benchmark_leaderboard(
        release="promptabi-test",
        performance_cases=("tokenizer-analysis", "stop-checks"),
        benchmark_iterations=1,
    )
    payload = report.to_dict()
    entry = payload["entries"][0]

    assert payload["manifest_version"] == BENCHMARK_LEADERBOARD_VERSION
    assert payload["ok"] is True
    assert payload["metrics"] == [
        "precision",
        "recall",
        "abstention_rate",
        "runtime_seconds",
        "peak_memory_bytes",
        "mean_witness_size_bytes",
        "solver_reliability",
    ]
    assert entry["release"] == "promptabi-test"
    assert entry["quality"]["precision"] == 1.0
    assert entry["quality"]["recall"] == 1.0
    assert entry["evaluation_case_count"] >= 10
    assert entry["peak_memory_bytes"] > 0
    assert entry["witness"]["total_size_bytes"] > 0
    assert entry["witness"]["mean_case_size_bytes"] > 0
    assert entry["solver"]["reliability"] == 1.0
    assert entry["solver"]["case_count"] == 4
    assert [case["benchmark"] for case in entry["performance"]] == ["tokenizer-analysis", "stop-checks"]


def test_benchmark_leaderboard_renderers_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = build_benchmark_leaderboard(
        release="promptabi-test",
        performance_cases=("tokenizer-analysis",),
        benchmark_iterations=1,
    )
    text = render_benchmark_leaderboard_text(report)
    payload = json.loads(render_benchmark_leaderboard_json(report))

    assert "PromptABI benchmark leaderboard" in text
    assert "solver=1.000" in text
    assert payload["entries"][0]["quality"]["f1"] == 1.0

    output = tmp_path / "leaderboard.json"
    exit_code = main(
        [
            "corpus",
            "leaderboard",
            "--release",
            "promptabi-cli",
            "--benchmark-case",
            "tokenizer-analysis",
            "--benchmark-iterations",
            "1",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    cli_payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "wrote benchmark leaderboard" in captured.out
    assert cli_payload["entries"][0]["release"] == "promptabi-cli"
    assert cli_payload["entries"][0]["solver"]["reliability"] == 1.0

    api_payload = json.loads(
        promptabi.benchmark_leaderboard(
            release="promptabi-api",
            benchmark_iterations=1,
            output_format="json",
        )
    )
    assert api_payload["entries"][0]["release"] == "promptabi-api"
