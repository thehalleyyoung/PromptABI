#!/usr/bin/env python3
"""One-command reproduction harness for the scaled empirical evaluation.

Roadmap step 330: ``python benchmarks/scaled_eval_repro.py`` rebuilds the
>=10,000-case labeled corpus, replays the production analyzers, and prints the
full report.  Pass ``--check-golden`` to assert the deterministic summary still
matches ``benchmarks/scaled_eval_golden.json`` (a regression gate), or
``--update-golden`` to refresh it.

Everything is CPU-only, network-free, and seedless-deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from promptabi.scaled_evaluation import (
    render_scaled_evaluation_text,
    run_scaled_evaluation,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "scaled_eval_golden.json"


def _golden_summary(report) -> dict[str, object]:
    """A small, stable projection of the report (excludes wall-clock timings)."""

    role = next(s for s in report.analyzer_scores if s.analyzer == "role-boundary")
    return {
        "version": report.version,
        "corpus_size": report.corpus_size,
        "passed": report.passed,
        "ground_truth_prevalence": round(report.prevalence.prevalence, 4),
        "predicted_prevalence": round(report.prevalence.predicted_prevalence, 4),
        "analyzer_invocations": report.prevalence.analyzer_invocations,
        "role_boundary": role.matrix.to_dict(),
        "ablation_precision_gain": round(report.ablation.precision_gain, 4),
        "cohen_kappa": round(report.inter_rater.cohen_kappa, 4),
        "schema_violation_rate": round(report.schema.overall_violation_rate, 4),
        "max_drift": round(report.drift.max_month_over_month_drift, 4),
        "leaderboard": [e.family for e in report.leaderboard],
        "cve_detected": [c.cve_id for c in report.cve_regressions if c.detected],
        "false_discovery_rate": round(
            report.false_positive_cost.false_discovery_rate, 4
        ),
        "cross_tokenizer_error_rate": round(
            report.cross_tokenizer.alignment_error_rate, 4
        ),
        "training_violation_rate": round(
            report.training_contracts.violation_rate, 4
        ),
        "tokenizer_round_trip_exact": report.throughput.round_trip_exact,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-golden", action="store_true")
    parser.add_argument("--update-golden", action="store_true")
    parser.add_argument("--corpus-limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = run_scaled_evaluation(corpus_limit=args.corpus_limit)
    if not args.quiet:
        print(render_scaled_evaluation_text(report), end="")

    summary = _golden_summary(report)

    if args.update_golden:
        GOLDEN_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"updated golden -> {GOLDEN_PATH}")
        return 0

    if args.check_golden:
        if not GOLDEN_PATH.exists():
            print("golden file missing; run with --update-golden", file=sys.stderr)
            return 2
        expected = json.loads(GOLDEN_PATH.read_text())
        if expected != summary:
            print("GOLDEN MISMATCH", file=sys.stderr)
            print(json.dumps({"expected": expected, "actual": summary}, indent=2),
                  file=sys.stderr)
            return 1
        print("golden OK")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
