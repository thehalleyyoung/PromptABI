"""Versioned benchmark leaderboards for PromptABI checker quality and cost."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._version import __version__
from .benchmarks import BenchmarkResult, run_benchmarks
from .evaluation import EvaluationReport, run_evaluation
from .smt_benchmarks import build_smt_benchmark_manifest


BENCHMARK_LEADERBOARD_VERSION = 1
DEFAULT_LEADERBOARD_RELEASE = f"promptabi-{__version__}"
DEFAULT_PERFORMANCE_CASES = (
    "tokenizer-analysis",
    "grammar-emptiness",
    "stop-checks",
    "z3-static-contracts",
    "budget-checks",
    "corpus-wide-verification",
)


@dataclass(frozen=True, slots=True)
class BenchmarkLeaderboardEntry:
    """One released PromptABI version scored against real benchmark assets."""

    release: str
    evaluation: EvaluationReport
    performance: tuple[BenchmarkResult, ...]
    smt_manifest: dict[str, object]

    @property
    def precision(self) -> float:
        return self.evaluation.score.precision

    @property
    def recall(self) -> float:
        return self.evaluation.score.recall

    @property
    def f1(self) -> float:
        return self.evaluation.score.f1

    @property
    def abstention_rate(self) -> float:
        return self.evaluation.abstention_rate

    @property
    def runtime_seconds(self) -> float:
        return float(self.evaluation.to_dict()["runtime_seconds"]) + sum(result.seconds for result in self.performance)

    @property
    def peak_memory_bytes(self) -> int:
        evaluation_peak = int(self.evaluation.to_dict()["peak_memory_bytes"])
        benchmark_peaks = [
            int(value)
            for result in self.performance
            if isinstance((value := result.metrics.get("peak_memory_bytes")), int)
        ]
        return max((evaluation_peak, *benchmark_peaks), default=0)

    @property
    def mean_witness_size(self) -> float:
        witness_bearing_cases = [result for result in self.evaluation.results if result.witness_size_bytes > 0]
        if not witness_bearing_cases:
            return 0.0
        return sum(result.witness_size_bytes for result in witness_bearing_cases) / len(witness_bearing_cases)

    @property
    def solver_reliability(self) -> float:
        entries = self.smt_manifest.get("entries", ())
        if not isinstance(entries, list):
            return 0.0
        if not entries:
            return 1.0
        passed = sum(1 for entry in entries if isinstance(entry, dict) and entry.get("passed") is True)
        return passed / len(entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "release": self.release,
            "quality": {
                "precision": _round(self.precision),
                "recall": _round(self.recall),
                "f1": _round(self.f1),
                "true_positives": self.evaluation.score.true_positives,
                "false_positives": self.evaluation.score.false_positives,
                "false_negatives": self.evaluation.score.false_negatives,
            },
            "abstention_rate": _round(self.abstention_rate),
            "runtime_seconds": round(self.runtime_seconds, 6),
            "peak_memory_bytes": self.peak_memory_bytes,
            "witness": {
                "quality_score": _round(self.evaluation.witness_quality_score),
                "total_size_bytes": int(self.evaluation.to_dict()["witness_size_bytes"]),
                "mean_case_size_bytes": _round(self.mean_witness_size),
            },
            "solver": {
                "reliability": _round(self.solver_reliability),
                "case_count": int(self.smt_manifest.get("case_count", 0)),
                "categories": list(self.smt_manifest.get("categories", ())),
                "manifest_sha256": self.smt_manifest.get("manifest_sha256"),
            },
            "performance": [result.to_dict() for result in self.performance],
            "evaluation_case_count": len(self.evaluation.results),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkLeaderboardReport:
    """Deterministic leaderboard report suitable for releases and papers."""

    entries: tuple[BenchmarkLeaderboardEntry, ...]
    methodology: str

    @property
    def ok(self) -> bool:
        return all(
            entry.precision == 1.0
            and entry.recall == 1.0
            and entry.solver_reliability == 1.0
            and all(result.seconds >= 0 for result in entry.performance)
            for entry in self.entries
        )

    def to_dict(self) -> dict[str, object]:
        ranked = sorted(
            self.entries,
            key=lambda entry: (
                -entry.f1,
                -entry.solver_reliability,
                entry.abstention_rate,
                entry.runtime_seconds,
                entry.peak_memory_bytes,
                entry.release,
            ),
        )
        return {
            "manifest_version": BENCHMARK_LEADERBOARD_VERSION,
            "methodology": self.methodology,
            "ok": self.ok,
            "entry_count": len(ranked),
            "metrics": [
                "precision",
                "recall",
                "abstention_rate",
                "runtime_seconds",
                "peak_memory_bytes",
                "mean_witness_size_bytes",
                "solver_reliability",
            ],
            "entries": [entry.to_dict() for entry in ranked],
        }


def build_benchmark_leaderboard(
    *,
    release: str = DEFAULT_LEADERBOARD_RELEASE,
    evaluation_corpus_path: str | Path | None = None,
    smt_benchmark_path: str | Path | None = None,
    performance_cases: Sequence[str] = DEFAULT_PERFORMANCE_CASES,
    benchmark_iterations: int = 1,
    repo_root: str | Path | None = None,
) -> BenchmarkLeaderboardReport:
    """Run leaderboard inputs against real repository code and fixture corpora."""

    if not release:
        raise ValueError("release must be a non-empty string")
    if benchmark_iterations <= 0:
        raise ValueError("benchmark_iterations must be positive")
    root = Path(repo_root) if repo_root is not None else None
    evaluation = run_evaluation(evaluation_corpus_path)
    performance = run_benchmarks(tuple(performance_cases), iterations=benchmark_iterations, root=root)
    smt_manifest = build_smt_benchmark_manifest(smt_benchmark_path)
    methodology = (
        "Scores are computed by replaying the labeled evaluation corpus, deterministic CPU-only "
        "performance benchmarks, and minimized SMT obligations in this checkout; no provider calls "
        "or model inference are used."
    )
    return BenchmarkLeaderboardReport(
        entries=(
            BenchmarkLeaderboardEntry(
                release=release,
                evaluation=evaluation,
                performance=performance,
                smt_manifest=smt_manifest,
            ),
        ),
        methodology=methodology,
    )


def render_benchmark_leaderboard_json(report: BenchmarkLeaderboardReport) -> str:
    """Render a stable machine-readable leaderboard."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_benchmark_leaderboard_text(report: BenchmarkLeaderboardReport) -> str:
    """Render a concise terminal leaderboard."""

    lines = [
        "PromptABI benchmark leaderboard",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"entries: {len(report.entries)}",
    ]
    for index, entry in enumerate(report.to_dict()["entries"], start=1):
        if not isinstance(entry, dict):
            continue
        quality = entry.get("quality", {})
        witness = entry.get("witness", {})
        solver = entry.get("solver", {})
        lines.append(
            "#{rank} {release}: precision={precision:.3f} recall={recall:.3f} "
            "abstention={abstention:.3f} runtime={runtime:.3f}s memory={memory} "
            "witness={witness_size:.3f} solver={solver_reliability:.3f}".format(
                rank=index,
                release=entry.get("release"),
                precision=float(quality.get("precision", 0.0)) if isinstance(quality, dict) else 0.0,
                recall=float(quality.get("recall", 0.0)) if isinstance(quality, dict) else 0.0,
                abstention=float(entry.get("abstention_rate", 0.0)),
                runtime=float(entry.get("runtime_seconds", 0.0)),
                memory=int(entry.get("peak_memory_bytes", 0)),
                witness_size=float(witness.get("mean_case_size_bytes", 0.0)) if isinstance(witness, dict) else 0.0,
                solver_reliability=float(solver.get("reliability", 0.0)) if isinstance(solver, dict) else 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def _round(value: float) -> float:
    return round(value, 6)


if sys.version_info < (3, 11):  # pragma: no cover - package metadata enforces this.
    raise RuntimeError("PromptABI requires Python 3.11+")
