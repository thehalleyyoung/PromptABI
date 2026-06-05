"""Release-blocking verification across PromptABI's maintained corpora."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adversarial_corpus import build_adversarial_corpus_manifest
from .evaluation import EvaluationError, EvaluationReport, run_evaluation
from .evaluation_fixture_packs import build_evaluation_fixture_pack_manifest
from .grammar_conformance import build_grammar_conformance_report
from .loaders import ArtifactLoader
from .provider_fixture_packs import ProviderFixturePackError, load_provider_fixture_pack_corpus
from .provider_fixture_replay import analyze_provider_fixture_replay
from .real_bug_benchmarks import RealBugBenchmarkError, build_real_bug_benchmark_manifest
from .seed_corpus import SeedCorpusError, build_seed_corpus_manifest
from .smt_benchmarks import SmtBenchmarkError, build_smt_benchmark_manifest
from .session import VerificationSession
from .structured_schema_corpus import (
    StructuredSchemaCorpusError,
    load_structured_schema_corpus,
    validate_structured_schema_entry,
)
from .tokenizer_conformance import build_tokenizer_conformance_report


class CorpusVerificationError(ValueError):
    """Raised when a maintained corpus cannot be verified."""


@dataclass(frozen=True, slots=True)
class CorpusVerificationThresholds:
    """Release-gate thresholds for corpus verification."""

    min_witness_quality: float = 0.75
    min_differential_agreement: float = 0.30
    max_runtime_seconds: float | None = None
    max_peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("min_witness_quality", "min_differential_agreement"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise CorpusVerificationError(f"{field_name} must be between 0.0 and 1.0")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise CorpusVerificationError("max_runtime_seconds must be positive when set")
        if self.max_peak_memory_bytes is not None and self.max_peak_memory_bytes <= 0:
            raise CorpusVerificationError("max_peak_memory_bytes must be positive when set")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_witness_quality": self.min_witness_quality,
            "min_differential_agreement": self.min_differential_agreement,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_peak_memory_bytes": self.max_peak_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class CorpusVerificationCheck:
    """One release-blocking corpus verification component."""

    name: str
    passed: bool
    summary: str
    coverage_count: int
    expected_count: int
    failures: tuple[str, ...] = ()
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "coverage_count": self.coverage_count,
            "expected_count": self.expected_count,
            "failures": list(self.failures),
            "metrics": dict(sorted(self.metrics.items())),
        }


@dataclass(frozen=True, slots=True)
class CorpusVerificationReport:
    """Aggregated release-blocking verification report for maintained corpora."""

    checks: tuple[CorpusVerificationCheck, ...]
    thresholds: CorpusVerificationThresholds
    runtime_seconds: float
    peak_memory_bytes: int

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def coverage_count(self) -> int:
        return sum(check.coverage_count for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "check_count": len(self.checks),
            "coverage_count": self.coverage_count,
            "runtime_seconds": round(self.runtime_seconds, 6),
            "peak_memory_bytes": self.peak_memory_bytes,
            "thresholds": self.thresholds.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
        }


def run_corpus_verification(
    *,
    seed_root: str | Path | None = None,
    structured_schema_root: str | Path | None = None,
    provider_fixture_root: str | Path | None = None,
    grammar_conformance_suite_path: str | Path | None = None,
    tokenizer_conformance_suite_path: str | Path | None = None,
    real_bug_benchmark_path: str | Path | None = None,
    evaluation_corpus_path: str | Path | None = None,
    evaluation_fixture_pack_path: str | Path | None = None,
    smt_benchmark_path: str | Path | None = None,
    thresholds: CorpusVerificationThresholds | None = None,
) -> CorpusVerificationReport:
    """Verify all maintained corpora as a release-blocking gate."""

    resolved_thresholds = thresholds or CorpusVerificationThresholds()
    started_at = time.perf_counter()
    tracemalloc.start()
    try:
        checks = [
            _verify_seed_corpus(seed_root),
            _verify_structured_schema_corpus(structured_schema_root),
            _verify_provider_fixture_corpus(provider_fixture_root),
            _verify_grammar_conformance(grammar_conformance_suite_path),
            _verify_tokenizer_conformance(tokenizer_conformance_suite_path),
            _verify_real_bug_benchmark(real_bug_benchmark_path),
            _verify_evaluation_fixture_pack(evaluation_fixture_pack_path),
            _verify_labeled_evaluation(evaluation_corpus_path, resolved_thresholds),
            _verify_smt_benchmark(smt_benchmark_path),
            _verify_adversarial_corpus(),
        ]
        if tracemalloc.is_tracing():
            _current, peak_memory_bytes = tracemalloc.get_traced_memory()
        else:
            peak_memory_bytes = 0
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    runtime_seconds = time.perf_counter() - started_at
    peak_memory_bytes = max(peak_memory_bytes, _component_peak_memory_bytes(checks))
    checks.append(_verify_performance(runtime_seconds, peak_memory_bytes, resolved_thresholds))
    return CorpusVerificationReport(
        checks=tuple(checks),
        thresholds=resolved_thresholds,
        runtime_seconds=runtime_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )


def render_corpus_verification_json(report: CorpusVerificationReport) -> str:
    """Render corpus verification as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_corpus_verification_text(report: CorpusVerificationReport) -> str:
    """Render corpus verification as a concise maintainer summary."""

    lines = [
        "PromptABI corpus verification",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"checks: {len(report.checks)}",
        f"coverage: {report.coverage_count} replayed corpus item(s)",
        f"runtime: {report.runtime_seconds:.3f}s",
        f"peak Python heap: {report.peak_memory_bytes} bytes",
    ]
    for check in report.checks:
        lines.append(
            f"- {check.name}: {'PASS' if check.passed else 'FAIL'} "
            f"({check.coverage_count}/{check.expected_count}) {check.summary}"
        )
        for failure in check.failures:
            lines.append(f"  failure: {failure}")
    return "\n".join(lines) + "\n"


def _verify_seed_corpus(root: str | Path | None) -> CorpusVerificationCheck:
    manifest = build_seed_corpus_manifest(root)
    entry_count = _int_metric(manifest, "entry_count")
    families = tuple(manifest.get("families", ()))
    required = tuple(manifest.get("required_families", ()))
    failures = []
    if entry_count <= 0:
        failures.append("seed corpus has no entries")
    if set(required).difference(families):
        failures.append("seed corpus is missing required families after manifest validation")
    return CorpusVerificationCheck(
        name="seed-corpus",
        passed=not failures,
        summary=f"{entry_count} tokenizer/template fixtures across {len(families)} families",
        coverage_count=entry_count,
        expected_count=max(entry_count, len(required)),
        failures=tuple(failures),
        metrics={
            "manifest_sha256": manifest["manifest_sha256"],
            "families": list(families),
        },
    )


def _verify_structured_schema_corpus(root: str | Path | None) -> CorpusVerificationCheck:
    corpus = load_structured_schema_corpus(root)
    failures = []
    parser_replays = 0
    config_replays = 0
    for entry in corpus.entries:
        if entry.entry_type in {"schema", "grammar"}:
            validate_structured_schema_entry(entry)
            parser_replays += 1
        if entry.promptabi_config_path.is_file():
            result = VerificationSession.from_config_file(entry.promptabi_config_path).run()
            observed = {diagnostic.rule_id for diagnostic in result.diagnostics}
            missing = sorted(set(entry.expected_rule_ids).difference(observed))
            if missing:
                failures.append(f"{entry.entry_id} missing expected rule(s): {', '.join(missing)}")
            config_replays += 1
        else:
            failures.append(f"{entry.entry_id} is missing promptabi.json replay config")
    if not corpus.entries:
        failures.append("structured schema corpus has no entries")
    if parser_replays == 0:
        failures.append("structured schema corpus has no parser-compatibility replays")
    if config_replays != len(corpus.entries):
        failures.append(f"structured schema config replay coverage was {config_replays}/{len(corpus.entries)}")
    return CorpusVerificationCheck(
        name="structured-schema-corpus",
        passed=not failures,
        summary=f"{len(corpus.entries)} schema/grammar/tool fixtures with {config_replays} config replays",
        coverage_count=parser_replays + config_replays,
        expected_count=len(corpus.entries) * 2,
        failures=tuple(failures),
        metrics={
            "entry_types": list(corpus.entry_types),
            "source_categories": list(corpus.source_categories),
            "parser_replays": parser_replays,
            "config_replays": config_replays,
        },
    )


def _verify_provider_fixture_corpus(root: str | Path | None) -> CorpusVerificationCheck:
    corpus = load_provider_fixture_pack_corpus(root)
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in corpus.artifact_bundle())
    replay = analyze_provider_fixture_replay(loaded)
    failures = []
    if not corpus.entries:
        failures.append("provider fixture corpus has no entries")
    if replay.fixtures_checked != len(corpus.entries):
        failures.append(f"provider replay coverage was {replay.fixtures_checked}/{len(corpus.entries)}")
    if replay.findings:
        failures.extend(f"{finding.artifact_name}: {finding.message}" for finding in replay.findings)
    return CorpusVerificationCheck(
        name="provider-fixture-corpus",
        passed=not failures,
        summary=f"{replay.fixtures_checked} provider fixture packs replayed",
        coverage_count=replay.fixtures_checked,
        expected_count=len(corpus.entries),
        failures=tuple(failures),
        metrics={
            "provider_families": list(replay.provider_families),
            "replay_hash": replay.replay_hash,
        },
    )


def _verify_grammar_conformance(path: str | Path | None) -> CorpusVerificationCheck:
    try:
        report = build_grammar_conformance_report(path)
    except ValueError as exc:
        return CorpusVerificationCheck(
            name="grammar-conformance",
            passed=False,
            summary=f"grammar backend conformance suite could not be replayed: {exc}",
            coverage_count=0,
            expected_count=1,
            failures=(str(exc),),
        )
    failures = []
    if not report.all_cases_passed:
        failures.extend(f"missing required backend: {backend}" for backend in report.missing_backends)
        failures.extend(f"{case.case_id}: {case.reason}" for case in report.differential_report.mismatches)
        failures.extend(f"{case.case_id}: {case.reason}" for case in report.differential_report.abstentions)
        failures.extend(
            f"{coverage.backend_family}: missing accepted/rejected samples or clean replay"
            for coverage in report.backend_coverage
            if not coverage.passed
        )
    return CorpusVerificationCheck(
        name="grammar-conformance",
        passed=not failures,
        summary=(
            f"{report.case_count} backend conformance cases, {report.sample_count} recorded samples, "
            f"{len(report.backend_coverage)} backend families"
        ),
        coverage_count=report.case_count,
        expected_count=max(report.case_count, len(report.required_backends)),
        failures=tuple(failures),
        metrics={
            "manifest_sha256": report.manifest_sha256,
            "sample_count": report.sample_count,
            "required_backends": list(report.required_backends),
            "backend_families": [coverage.backend_family for coverage in report.backend_coverage if coverage.case_ids],
        },
    )


def _verify_tokenizer_conformance(path: str | Path | None) -> CorpusVerificationCheck:
    try:
        report = build_tokenizer_conformance_report(path)
    except ValueError as exc:
        return CorpusVerificationCheck(
            name="tokenizer-conformance",
            passed=False,
            summary=f"tokenizer family conformance suite could not be replayed: {exc}",
            coverage_count=0,
            expected_count=1,
            failures=(str(exc),),
        )
    failures = []
    if not report.all_cases_passed:
        failures.extend(f"missing required tokenizer family: {family}" for family in report.missing_families)
        failures.extend(f"missing required tokenizer feature: {feature}" for feature in report.missing_features)
        failures.extend(
            f"{case.case_id}: {mismatch.field} expected {mismatch.expected!r}, got {mismatch.actual!r}"
            for case in report.cases
            for mismatch in case.differential_report.mismatches
        )
        failures.extend(
            f"{coverage.family}: missing samples or clean replay"
            for coverage in report.family_coverage
            if not coverage.passed
        )
    return CorpusVerificationCheck(
        name="tokenizer-conformance",
        passed=not failures,
        summary=(
            f"{report.case_count} tokenizer conformance cases, {report.sample_count} replayed samples, "
            f"{len(report.family_coverage)} tokenizer families"
        ),
        coverage_count=report.case_count,
        expected_count=max(report.case_count, len(report.required_families)),
        failures=tuple(failures),
        metrics={
            "manifest_sha256": report.manifest_sha256,
            "sample_count": report.sample_count,
            "required_families": list(report.required_families),
            "required_features": list(report.required_features),
            "families": [coverage.family for coverage in report.family_coverage if coverage.case_ids],
        },
    )


def _verify_real_bug_benchmark(path: str | Path | None) -> CorpusVerificationCheck:
    manifest = build_real_bug_benchmark_manifest(path)
    case_count = _int_metric(manifest, "case_count")
    failures = []
    if case_count <= 0:
        failures.append("real-bug benchmark has no cases")
    if not manifest.get("all_cases_passed"):
        failures.append("one or more real-bug benchmark cases failed")
    return CorpusVerificationCheck(
        name="real-bug-benchmark",
        passed=not failures,
        summary=f"{case_count} reduced public-bug cases replayed",
        coverage_count=case_count,
        expected_count=case_count,
        failures=tuple(failures),
        metrics={
            "manifest_sha256": manifest["manifest_sha256"],
            "categories": list(manifest["categories"]),
        },
    )


def _verify_evaluation_fixture_pack(path: str | Path | None) -> CorpusVerificationCheck:
    manifest = build_evaluation_fixture_pack_manifest(path)
    case_count = _int_metric(manifest, "case_count")
    failures = []
    if case_count <= 0:
        failures.append("evaluation fixture pack has no cases")
    if not manifest.get("all_cases_passed"):
        failures.append("one or more evaluation fixture cases failed")
    return CorpusVerificationCheck(
        name="evaluation-fixture-pack",
        passed=not failures,
        summary=f"{case_count} benchmark-interface bug fixtures replayed",
        coverage_count=case_count,
        expected_count=case_count,
        failures=tuple(failures),
        metrics={
            "manifest_sha256": manifest["manifest_sha256"],
            "bug_classes": list(manifest["bug_classes"]),
        },
    )


def _verify_labeled_evaluation(
    path: str | Path | None,
    thresholds: CorpusVerificationThresholds,
) -> CorpusVerificationCheck:
    report = run_evaluation(path)
    payload = report.to_dict()
    case_count = len(report.results)
    score = report.score
    differential_denominator = _differential_denominator(report)
    solver_quality = report.solver_result_quality
    failures = []
    if case_count <= 0:
        failures.append("labeled evaluation has no cases")
    if score.true_positives <= 0:
        failures.append("labeled evaluation produced no true positives")
    if not all(result.passed for result in report.results):
        failures.append("one or more labeled evaluation cases failed")
    if score.precision != 1.0 or score.recall != 1.0 or score.f1 != 1.0:
        failures.append(
            f"expected precision/recall/f1 of 1.0, got {score.precision:.3f}/{score.recall:.3f}/{score.f1:.3f}"
        )
    if report.witness_quality_score < thresholds.min_witness_quality:
        failures.append(
            f"witness quality {report.witness_quality_score:.3f} below {thresholds.min_witness_quality:.3f}"
        )
    if differential_denominator <= 0:
        failures.append("labeled evaluation has no differential agreement/mismatch/abstention coverage")
    elif report.differential_agreement_rate < thresholds.min_differential_agreement:
        failures.append(
            "differential agreement "
            f"{report.differential_agreement_rate:.3f} below {thresholds.min_differential_agreement:.3f}"
        )
    if int(solver_quality["z3_backed_results"]) <= 0:
        failures.append("labeled evaluation has no Z3-backed rule coverage")
    return CorpusVerificationCheck(
        name="labeled-evaluation",
        passed=not failures,
        summary=(
            f"{case_count} cases, f1={score.f1:.3f}, "
            f"witness={report.witness_quality_score:.3f}, differential={report.differential_agreement_rate:.3f}"
        ),
        coverage_count=case_count,
        expected_count=case_count,
        failures=tuple(failures),
        metrics={
            "passed": payload["passed"],
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "witness_quality_score": report.witness_quality_score,
            "differential_agreement_rate": report.differential_agreement_rate,
            "differential_cases": differential_denominator,
            "z3_backed_results": solver_quality["z3_backed_results"],
            "peak_memory_bytes": payload["peak_memory_bytes"],
        },
    )


def _verify_smt_benchmark(path: str | Path | None) -> CorpusVerificationCheck:
    manifest = build_smt_benchmark_manifest(path)
    case_count = _int_metric(manifest, "case_count")
    failures = []
    if case_count <= 0:
        failures.append("SMT benchmark has no cases")
    if not manifest.get("all_cases_passed"):
        failures.append("one or more SMT benchmark cases failed")
    categories = tuple(manifest.get("categories", ()))
    return CorpusVerificationCheck(
        name="smt-benchmark",
        passed=not failures,
        summary=f"{case_count} minimized SMT obligations across {len(categories)} categories",
        coverage_count=case_count,
        expected_count=case_count,
        failures=tuple(failures),
        metrics={
            "manifest_sha256": manifest["manifest_sha256"],
            "categories": list(categories),
        },
    )


def _verify_adversarial_corpus() -> CorpusVerificationCheck:
    manifest = build_adversarial_corpus_manifest()
    case_count = _int_metric(manifest, "case_count")
    failures = []
    if case_count <= 0:
        failures.append("adversarial corpus has no generated cases")
    if not manifest.get("all_cases_passed"):
        failures.append("one or more adversarial corpus cases failed")
    surfaces = tuple(manifest.get("surfaces", ()))
    return CorpusVerificationCheck(
        name="adversarial-corpus",
        passed=not failures,
        summary=f"{case_count} generated adversarial cases across {len(surfaces)} surfaces",
        coverage_count=case_count,
        expected_count=case_count,
        failures=tuple(failures),
        metrics={
            "manifest_sha256": manifest["manifest_sha256"],
            "surfaces": list(surfaces),
        },
    )


def _verify_performance(
    runtime_seconds: float,
    peak_memory_bytes: int,
    thresholds: CorpusVerificationThresholds,
) -> CorpusVerificationCheck:
    failures = []
    if thresholds.max_runtime_seconds is not None and runtime_seconds > thresholds.max_runtime_seconds:
        failures.append(
            f"runtime {runtime_seconds:.3f}s exceeded {thresholds.max_runtime_seconds:.3f}s"
        )
    if thresholds.max_peak_memory_bytes is not None and peak_memory_bytes > thresholds.max_peak_memory_bytes:
        failures.append(
            f"peak Python heap {peak_memory_bytes} bytes exceeded {thresholds.max_peak_memory_bytes} bytes"
        )
    return CorpusVerificationCheck(
        name="performance-thresholds",
        passed=not failures,
        summary="optional wall-clock and tracemalloc Python-heap thresholds",
        coverage_count=1,
        expected_count=1,
        failures=tuple(failures),
        metrics={
            "runtime_seconds": runtime_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        },
    )


def _differential_denominator(report: EvaluationReport) -> int:
    return sum(
        result.differential_agreements + result.differential_mismatches + result.differential_abstentions
        for result in report.results
    )


def _component_peak_memory_bytes(checks: list[CorpusVerificationCheck]) -> int:
    peaks = []
    for check in checks:
        value = check.metrics.get("peak_memory_bytes")
        if isinstance(value, int):
            peaks.append(value)
    return max(peaks, default=0)


def _int_metric(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise CorpusVerificationError(f"corpus manifest field {key!r} must be an integer")
    return value
