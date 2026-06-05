"""Evaluation scripts for labeled PromptABI corpora."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .diagnostics import CheckMode, Diagnostic
from .real_bug_benchmarks import load_real_bug_benchmark_suite
from .session import CHECK_MODE_CATALOG, VerificationSession


DEFAULT_EVALUATION_CORPUS_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "evaluation" / "labeled_corpus.json"
EVALUATION_CORPUS_VERSION = 1
DIFFERENTIAL_AGREEMENT_RULES = frozenset(
    {
        "grammar-differential-agreement",
        "parser-compatibility-agreement",
        "stop-differential-agreement",
    }
)
DIFFERENTIAL_MISMATCH_RULES = frozenset(
    {
        "grammar-differential-mismatch",
        "parser-compatibility-mismatch",
        "stop-differential-mismatch",
        "tokenizer-differential-mismatch",
    }
)
DIFFERENTIAL_ABSTENTION_RULES = frozenset(
    {
        "grammar-differential-abstained",
        "parser-compatibility-abstained",
        "stop-differential-abstained",
    }
)


class EvaluationError(ValueError):
    """Raised when an evaluation corpus is malformed or cannot be replayed."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One labeled evaluation case."""

    case_id: str
    source: str
    expected_rule_ids: tuple[str, ...]
    expected_absent_rule_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    config_path: Path | None = None
    benchmark_path: Path | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class RuleScore:
    """Scoped precision/recall accounting for labeled rule IDs."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return (2 * self.precision * self.recall / denominator) if denominator else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": _round_metric(self.precision),
            "recall": _round_metric(self.recall),
            "f1": _round_metric(self.f1),
        }


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """Replay result for one labeled case."""

    case: EvaluationCase
    observed_rule_ids: tuple[str, ...]
    score: RuleScore
    runtime_seconds: float
    peak_memory_bytes: int
    diagnostic_count: int
    abstaining_diagnostic_count: int
    witness_quality_score: float
    z3_backed_rule_ids: tuple[str, ...] = ()
    differential_agreements: int = 0
    differential_mismatches: int = 0
    differential_abstentions: int = 0

    @property
    def passed(self) -> bool:
        return self.score.false_positives == 0 and self.score.false_negatives == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case.case_id,
            "source": self.case.source,
            "labels": list(self.case.labels),
            "description": self.case.description,
            "expected_rule_ids": list(self.case.expected_rule_ids),
            "expected_absent_rule_ids": list(self.case.expected_absent_rule_ids),
            "observed_rule_ids": list(self.observed_rule_ids),
            "passed": self.passed,
            "score": self.score.to_dict(),
            "diagnostic_count": self.diagnostic_count,
            "abstaining_diagnostic_count": self.abstaining_diagnostic_count,
            "runtime_seconds": round(self.runtime_seconds, 6),
            "peak_memory_bytes": self.peak_memory_bytes,
            "witness_quality_score": _round_metric(self.witness_quality_score),
            "z3_backed_rule_ids": list(self.z3_backed_rule_ids),
            "differential_agreements": self.differential_agreements,
            "differential_mismatches": self.differential_mismatches,
            "differential_abstentions": self.differential_abstentions,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregated metrics for a labeled PromptABI corpus."""

    corpus_path: Path
    methodology: str
    results: tuple[EvaluationCaseResult, ...]
    z3_available: bool
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> RuleScore:
        return RuleScore(
            true_positives=sum(result.score.true_positives for result in self.results),
            false_positives=sum(result.score.false_positives for result in self.results),
            false_negatives=sum(result.score.false_negatives for result in self.results),
        )

    @property
    def abstention_rate(self) -> float:
        diagnostics = sum(result.diagnostic_count for result in self.results)
        abstentions = sum(result.abstaining_diagnostic_count for result in self.results)
        return abstentions / diagnostics if diagnostics else 0.0

    @property
    def witness_quality_score(self) -> float:
        if not self.results:
            return 1.0
        return sum(result.witness_quality_score for result in self.results) / len(self.results)

    @property
    def differential_agreement_rate(self) -> float:
        agreements = sum(result.differential_agreements for result in self.results)
        mismatches = sum(result.differential_mismatches for result in self.results)
        abstentions = sum(result.differential_abstentions for result in self.results)
        denominator = agreements + mismatches + abstentions
        return agreements / denominator if denominator else 1.0

    @property
    def solver_result_quality(self) -> dict[str, object]:
        z3_results = [rule for result in self.results for rule in result.z3_backed_rule_ids]
        decisive = [rule for rule in z3_results if rule not in {"static-contract-abstained", "static-contract-unknown"}]
        return {
            "z3_available": self.z3_available,
            "z3_backed_results": len(z3_results),
            "decisive_results": len(decisive),
            "decisive_rate": _round_metric(len(decisive) / len(z3_results) if z3_results else 1.0),
            "observed_rule_ids": sorted(set(z3_results)),
        }

    def to_dict(self) -> dict[str, object]:
        total_runtime = sum(result.runtime_seconds for result in self.results)
        peak_memory = max((result.peak_memory_bytes for result in self.results), default=0)
        return {
            "manifest_version": EVALUATION_CORPUS_VERSION,
            "corpus_path": str(self.corpus_path),
            "methodology": self.methodology,
            "metadata": dict(sorted(self.metadata.items())),
            "case_count": len(self.results),
            "passed": all(result.passed for result in self.results),
            "score": self.score.to_dict(),
            "abstention_rate": _round_metric(self.abstention_rate),
            "witness_quality_score": _round_metric(self.witness_quality_score),
            "solver_result_quality": self.solver_result_quality,
            "differential_agreement_rate": _round_metric(self.differential_agreement_rate),
            "runtime_seconds": round(total_runtime, 6),
            "peak_memory_bytes": peak_memory,
            "cases": [result.to_dict() for result in self.results],
        }


def load_evaluation_cases(path: str | Path | None = None) -> tuple[str, dict[str, object], tuple[EvaluationCase, ...]]:
    """Load and expand a labeled evaluation corpus."""

    corpus_path = Path(path) if path is not None else DEFAULT_EVALUATION_CORPUS_PATH
    payload = _read_json_object(corpus_path)
    if payload.get("manifest_version") != EVALUATION_CORPUS_VERSION:
        raise EvaluationError(f"{corpus_path} has unsupported evaluation manifest_version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise EvaluationError(f"{corpus_path} field 'methodology' must be a non-empty string")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise EvaluationError(f"{corpus_path} field 'metadata' must be an object when present")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationError(f"{corpus_path} field 'cases' must be a non-empty list")

    cases: list[EvaluationCase] = []
    repo_root = corpus_path.parent.parent.parent
    for raw in raw_cases:
        cases.extend(_cases_from_mapping(corpus_path, repo_root, raw))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError(f"{corpus_path} expands to duplicate evaluation case ids")
    return methodology, dict(metadata), tuple(sorted(cases, key=lambda item: item.case_id))


def run_evaluation(path: str | Path | None = None) -> EvaluationReport:
    """Run a labeled evaluation corpus against real PromptABI code paths."""

    corpus_path = Path(path) if path is not None else DEFAULT_EVALUATION_CORPUS_PATH
    methodology, metadata, cases = load_evaluation_cases(corpus_path)
    results = tuple(_run_case(case) for case in cases)
    return EvaluationReport(
        corpus_path=corpus_path,
        methodology=methodology,
        metadata=metadata,
        results=results,
        z3_available=_z3_available(),
    )


def render_evaluation_json(report: EvaluationReport) -> str:
    """Render an evaluation report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_evaluation_text(report: EvaluationReport) -> str:
    """Render a concise terminal evaluation summary."""

    payload = report.to_dict()
    score = report.score
    lines = [
        "PromptABI evaluation",
        f"corpus: {report.corpus_path}",
        f"cases: {len(report.results)}",
        f"status: {'PASS' if payload['passed'] else 'FAIL'}",
        (
            "precision/recall/f1: "
            f"{score.precision:.3f}/{score.recall:.3f}/{score.f1:.3f} "
            f"(tp={score.true_positives}, fp={score.false_positives}, fn={score.false_negatives})"
        ),
        f"abstention rate: {report.abstention_rate:.3f}",
        f"witness quality: {report.witness_quality_score:.3f}",
        f"differential agreement: {report.differential_agreement_rate:.3f}",
        (
            "solver quality: "
            f"{report.solver_result_quality['decisive_results']}/"
            f"{report.solver_result_quality['z3_backed_results']} decisive "
            f"(z3={'available' if report.z3_available else 'unavailable'})"
        ),
    ]
    for result in report.results:
        lines.append(
            f"- {result.case.case_id}: {'PASS' if result.passed else 'FAIL'} "
            f"observed={','.join(result.observed_rule_ids) or 'none'}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PromptABI labeled-corpus evaluation.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_EVALUATION_CORPUS_PATH)
    parser.add_argument("--format", choices=("text", "json"), default="json")
    args = parser.parse_args(argv)
    report = run_evaluation(args.corpus)
    output = render_evaluation_text(report) if args.format == "text" else render_evaluation_json(report)
    print(output, end="")
    return 0 if all(result.passed for result in report.results) else 1


def _cases_from_mapping(corpus_path: Path, repo_root: Path, raw: object) -> tuple[EvaluationCase, ...]:
    if not isinstance(raw, dict):
        raise EvaluationError(f"{corpus_path} cases must be JSON objects")
    source = _required_string(corpus_path, raw, "source")
    if source == "config":
        case_id = _required_string(corpus_path, raw, "id")
        config = _required_string(corpus_path, raw, "config")
        return (
            EvaluationCase(
                case_id=case_id,
                source=source,
                config_path=(repo_root / config).resolve(),
                expected_rule_ids=_string_tuple(
                    raw.get("expected_rule_ids"),
                    corpus_path,
                    case_id,
                    "expected_rule_ids",
                    allow_empty=False,
                ),
                expected_absent_rule_ids=_string_tuple(raw.get("expected_absent_rule_ids", ()), corpus_path, case_id, "expected_absent_rule_ids"),
                labels=_string_tuple(raw.get("labels", ()), corpus_path, case_id, "labels"),
                description=_optional_string(corpus_path, raw, "description"),
            ),
        )
    if source == "real-bug-benchmark":
        benchmark = raw.get("benchmark")
        benchmark_path = (repo_root / benchmark).resolve() if isinstance(benchmark, str) else None
        suite = load_real_bug_benchmark_suite(benchmark_path)
        prefix = _optional_string(corpus_path, raw, "id_prefix") or "real-bug"
        expected_absent = _string_tuple(raw.get("expected_absent_rule_ids", ()), corpus_path, prefix, "expected_absent_rule_ids")
        cases = []
        for benchmark_case in suite.cases:
            cases.append(
                EvaluationCase(
                    case_id=f"{prefix}:{benchmark_case.case_id}",
                    source=source,
                    benchmark_path=benchmark_path,
                    expected_rule_ids=benchmark_case.expected_rule_ids,
                    expected_absent_rule_ids=expected_absent,
                    labels=benchmark_case.labels,
                    description=benchmark_case.display_name,
                )
            )
        return tuple(cases)
    raise EvaluationError(f"{corpus_path} has unsupported evaluation source {source!r}")


def _run_case(case: EvaluationCase) -> EvaluationCaseResult:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        diagnostics, observed_rule_ids = _collect_case_observations(case)
        _current, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    runtime = time.perf_counter() - start
    observed = set(observed_rule_ids)
    expected = set(case.expected_rule_ids)
    expected_absent = set(case.expected_absent_rule_ids)
    score = RuleScore(
        true_positives=len(observed.intersection(expected)),
        false_positives=len(observed.intersection(expected_absent)),
        false_negatives=len(expected.difference(observed)),
    )
    abstaining = sum(1 for diagnostic in diagnostics if _diagnostic_has_mode(diagnostic, CheckMode.ABSTAINING))
    z3_rules = tuple(sorted(rule for rule in observed_rule_ids if CheckMode.Z3_BACKED_SMT in CHECK_MODE_CATALOG.get(rule, ())))
    return EvaluationCaseResult(
        case=case,
        observed_rule_ids=tuple(sorted(observed)),
        score=score,
        runtime_seconds=runtime,
        peak_memory_bytes=peak_memory,
        diagnostic_count=len(diagnostics) if diagnostics else len(observed_rule_ids),
        abstaining_diagnostic_count=abstaining,
        witness_quality_score=_witness_quality(diagnostics),
        z3_backed_rule_ids=z3_rules,
        differential_agreements=len(observed.intersection(DIFFERENTIAL_AGREEMENT_RULES)),
        differential_mismatches=len(observed.intersection(DIFFERENTIAL_MISMATCH_RULES)),
        differential_abstentions=len(observed.intersection(DIFFERENTIAL_ABSTENTION_RULES)),
    )


def _collect_case_observations(case: EvaluationCase) -> tuple[tuple[Diagnostic, ...], tuple[str, ...]]:
    if case.source == "config":
        if case.config_path is None:
            raise EvaluationError(f"{case.case_id} config case is missing config_path")
        result = VerificationSession.from_config_file(case.config_path).run()
        return result.diagnostics, tuple(diagnostic.rule_id for diagnostic in result.diagnostics)
    if case.source == "real-bug-benchmark":
        suite = load_real_bug_benchmark_suite(case.benchmark_path)
        case_id = case.case_id.split(":", 1)[1] if ":" in case.case_id else case.case_id
        benchmark_result = next((result for result in suite.replay() if result.case_id == case_id), None)
        if benchmark_result is None:
            raise EvaluationError(f"real-bug benchmark case not found: {case.case_id}")
        return (), benchmark_result.observed_rule_ids
    raise EvaluationError(f"{case.case_id} uses unsupported source {case.source!r}")


def _witness_quality(diagnostics: tuple[Diagnostic, ...]) -> float:
    eligible = [diagnostic for diagnostic in diagnostics if diagnostic.severity.value in {"error", "warning"}]
    if not eligible:
        return 1.0
    scores = []
    for diagnostic in eligible:
        score = 0.0
        if diagnostic.witness is not None:
            score += 0.4
            if diagnostic.witness.steps:
                score += 0.2
            if diagnostic.witness.artifacts:
                score += 0.1
        if diagnostic.suggestions:
            score += 0.2
        if diagnostic.artifact is not None or diagnostic.span is not None:
            score += 0.1
        scores.append(score)
    return sum(scores) / len(scores)


def _diagnostic_has_mode(diagnostic: Diagnostic, mode: CheckMode) -> bool:
    return mode in diagnostic.check_modes or mode in CHECK_MODE_CATALOG.get(diagnostic.rule_id, ())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"evaluation corpus not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"evaluation corpus is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{path} root must be a JSON object")
    return payload


def _required_string(path: Path, raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{path} case field '{key}' must be a non-empty string")
    return value


def _optional_string(path: Path, raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{path} case field '{key}' must be a non-empty string when present")
    return value


def _string_tuple(
    value: object,
    path: Path,
    case_id: str,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EvaluationError(f"{path} case {case_id!r} field '{field_name}' must be a list of non-empty strings")
    normalized = tuple(sorted(dict.fromkeys(value)))
    if not normalized and not allow_empty:
        raise EvaluationError(f"{path} case {case_id!r} field '{field_name}' must not be empty")
    return normalized


def _z3_available() -> bool:
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def _round_metric(value: float) -> float:
    return round(value, 6)


if __name__ == "__main__":
    raise SystemExit(main())
