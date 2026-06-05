"""Comparative studies against adjacent prompt-interface tooling classes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import EvaluationError, EvaluationReport, run_evaluation
from .real_bug_benchmarks import RealBugBenchmarkError, build_real_bug_benchmark_manifest


COMPARATIVE_STUDY_VERSION = 1


class ComparativeStudyError(ValueError):
    """Raised when a comparative study cannot be built from repository evidence."""


@dataclass(frozen=True, slots=True)
class BaselineClass:
    """One adjacent class of tools used as a non-overlap baseline."""

    baseline_id: str
    name: str
    typical_scope: str
    modeled_strengths: tuple[str, ...]
    unsupported_surfaces: tuple[str, ...]
    covered_rule_ids: tuple[str, ...]
    covered_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.baseline_id,
            "name": self.name,
            "typical_scope": self.typical_scope,
            "modeled_strengths": list(self.modeled_strengths),
            "unsupported_surfaces": list(self.unsupported_surfaces),
            "covered_rule_ids": list(self.covered_rule_ids),
            "covered_labels": list(self.covered_labels),
        }


@dataclass(frozen=True, slots=True)
class ComparativeCase:
    """One labeled case projected into the baseline-comparison study."""

    case_id: str
    labels: tuple[str, ...]
    expected_rule_ids: tuple[str, ...]
    observed_rule_ids: tuple[str, ...]
    promptabi_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "labels": list(self.labels),
            "expected_rule_ids": list(self.expected_rule_ids),
            "observed_rule_ids": list(self.observed_rule_ids),
            "promptabi_passed": self.promptabi_passed,
        }


@dataclass(frozen=True, slots=True)
class BaselineComparisonResult:
    """Coverage projection for one baseline class."""

    baseline: BaselineClass
    covered_case_ids: tuple[str, ...]
    missed_case_ids: tuple[str, ...]
    promptabi_only_rule_ids: tuple[str, ...]

    @property
    def case_count(self) -> int:
        return len(self.covered_case_ids) + len(self.missed_case_ids)

    @property
    def covered_case_count(self) -> int:
        return len(self.covered_case_ids)

    @property
    def missed_case_count(self) -> int:
        return len(self.missed_case_ids)

    @property
    def coverage_rate(self) -> float:
        return self.covered_case_count / self.case_count if self.case_count else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "case_count": self.case_count,
            "covered_case_count": self.covered_case_count,
            "missed_case_count": self.missed_case_count,
            "coverage_rate": _round_metric(self.coverage_rate),
            "covered_case_ids": list(self.covered_case_ids),
            "missed_case_ids": list(self.missed_case_ids),
            "promptabi_only_rule_ids": list(self.promptabi_only_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class ComparativeStudyReport:
    """Evidence-backed comparison between PromptABI and baseline tool classes."""

    evaluation: EvaluationReport
    real_bug_manifest: dict[str, object]
    cases: tuple[ComparativeCase, ...]
    baselines: tuple[BaselineComparisonResult, ...]

    @property
    def promptabi_detected_cases(self) -> int:
        return sum(1 for case in self.cases if case.promptabi_passed)

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> bool:
        return self.promptabi_detected_cases == self.case_count and all(
            baseline.missed_case_count > 0 for baseline in self.baselines
        )

    def to_dict(self) -> dict[str, object]:
        evaluation_payload = self.evaluation.to_dict()
        return {
            "manifest_version": COMPARATIVE_STUDY_VERSION,
            "methodology": (
                "PromptABI is run on the repository's labeled evaluation corpus and replayable real-bug "
                "reductions. Adjacent baseline classes are modeled by their documented artifact scope: a "
                "case is credited only when its labels or expected diagnostic rules fall inside that scope. "
                "The study intentionally compares capability classes, not specific proprietary products."
            ),
            "passed": self.passed,
            "case_count": self.case_count,
            "promptabi_detected_cases": self.promptabi_detected_cases,
            "evaluation_score": evaluation_payload["score"],
            "abstention_rate": evaluation_payload["abstention_rate"],
            "differential_agreement_rate": evaluation_payload["differential_agreement_rate"],
            "real_bug_cases": self.real_bug_manifest["case_count"],
            "real_bug_categories": self.real_bug_manifest["categories"],
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "cases": [case.to_dict() for case in self.cases],
        }


BASELINE_CLASSES: tuple[BaselineClass, ...] = (
    BaselineClass(
        baseline_id="prompt-linter",
        name="Prompt linters",
        typical_scope="Textual heuristics over prompt strings and policy wording.",
        modeled_strengths=("keyword/style warnings", "basic prompt hygiene", "human-readable suggestions"),
        unsupported_surfaces=(
            "tokenizer special-token behavior",
            "provider tool envelopes",
            "JSON parser compatibility",
            "RAG truncation",
            "training loss-mask alignment",
        ),
        covered_rule_ids=("role-boundary-nonforgeability",),
        covered_labels=("chat-template", "role-boundary"),
    ),
    BaselineClass(
        baseline_id="schema-validator",
        name="Schema validators",
        typical_scope="Validate final JSON-like objects after generation.",
        modeled_strengths=("JSON shape validation", "required fields", "enum/type constraints"),
        unsupported_surfaces=(
            "tokenizer x grammar emptiness",
            "stop strings inside valid fields",
            "provider streaming chunks",
            "prompt budgets",
            "training manifests",
        ),
        covered_rule_ids=("parser-compatibility-mismatch", "tool-schema-ingestion"),
        covered_labels=("json-schema", "structured-output"),
    ),
    BaselineClass(
        baseline_id="constrained-decoder",
        name="Constrained-decoding libraries",
        typical_scope="Runtime token masking for a declared grammar or schema.",
        modeled_strengths=("grammar-constrained generation", "schema-guided decoding", "runtime parser hooks"),
        unsupported_surfaces=(
            "offline provider migration",
            "tool serialization drift",
            "framework truncation",
            "training/evaluation contract drift",
            "lockfile provenance",
        ),
        covered_rule_ids=(
            "grammar-differential-agreement",
            "grammar-differential-mismatch",
            "parser-compatibility-mismatch",
            "stop-overreach-content",
        ),
        covered_labels=("structured-output", "stop-policy", "differential"),
    ),
    BaselineClass(
        baseline_id="tokenizer-diff",
        name="Tokenizer diff tools",
        typical_scope="Compare tokenizer files, token IDs, normalization, and special-token revisions.",
        modeled_strengths=("token ID drift", "normalizer changes", "special-token deltas"),
        unsupported_surfaces=(
            "chat-template role regions",
            "tool schemas",
            "provider migrations",
            "RAG citation survival",
            "dataset packing and loss masks",
        ),
        covered_rule_ids=("tokenizer-differential-mismatch", "rag-tokenizer-mismatch"),
        covered_labels=("tokenizer", "special-token", "normalization"),
    ),
    BaselineClass(
        baseline_id="generic-static-analyzer",
        name="Generic static analyzers",
        typical_scope="Language-specific code smells, taint rules, and dependency checks.",
        modeled_strengths=("application code linting", "dependency vulnerabilities", "generic taint patterns"),
        unsupported_surfaces=(
            "Jinja chat-template semantics",
            "finite transducer products",
            "Z3-backed prompt contracts",
            "provider fixture replay",
            "LLM dataset manifests",
        ),
        covered_rule_ids=(),
        covered_labels=(),
    ),
)


def build_comparative_study_report(
    *,
    evaluation_corpus_path: str | Path | None = None,
    real_bug_benchmark_path: str | Path | None = None,
) -> ComparativeStudyReport:
    """Run live PromptABI evidence and compare coverage against adjacent tool classes."""

    try:
        evaluation = run_evaluation(evaluation_corpus_path)
        real_bug_manifest = build_real_bug_benchmark_manifest(real_bug_benchmark_path)
    except (EvaluationError, RealBugBenchmarkError) as exc:
        raise ComparativeStudyError(str(exc)) from exc

    cases = tuple(
        ComparativeCase(
            case_id=result.case.case_id,
            labels=result.case.labels,
            expected_rule_ids=result.case.expected_rule_ids,
            observed_rule_ids=result.observed_rule_ids,
            promptabi_passed=result.passed,
        )
        for result in evaluation.results
    )
    baselines = tuple(_compare_baseline(baseline, cases) for baseline in BASELINE_CLASSES)
    return ComparativeStudyReport(
        evaluation=evaluation,
        real_bug_manifest=real_bug_manifest,
        cases=cases,
        baselines=baselines,
    )


def render_comparative_study_json(report: ComparativeStudyReport) -> str:
    """Render the comparative study as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_comparative_study_markdown(report: ComparativeStudyReport) -> str:
    """Render the comparative study as a concise paper-ready Markdown table."""

    payload = report.to_dict()
    lines = [
        "# PromptABI comparative study",
        "",
        (
            f"PromptABI replayed {payload['case_count']} labeled cases, detected "
            f"{payload['promptabi_detected_cases']}, and replayed "
            f"{payload['real_bug_cases']} real-bug reductions across "
            f"{len(payload['real_bug_categories'])} categories."
        ),
        "",
        "| Baseline class | Modeled coverage | Missed cases | PromptABI-only rules |",
        "| --- | ---: | ---: | --- |",
    ]
    for baseline in report.baselines:
        rules = ", ".join(baseline.promptabi_only_rule_ids[:6])
        if len(baseline.promptabi_only_rule_ids) > 6:
            rules += ", ..."
        lines.append(
            f"| {baseline.baseline.name} | {_round_metric(baseline.coverage_rate):.2f} | "
            f"{baseline.missed_case_count} | {rules or 'all labeled rules'} |"
        )
    lines.extend(
        [
            "",
            "The baseline rows are capability-class projections, not claims about a named product. "
            "PromptABI's advantage comes from composing tokenizers, templates, stops, grammars, "
            "tool/provider contracts, RAG budgets, training/eval manifests, witnesses, and proof modes "
            "in one offline verifier.",
            "",
        ]
    )
    return "\n".join(lines)


def render_comparative_study_text(report: ComparativeStudyReport) -> str:
    """Render the comparative study for terminals."""

    payload = report.to_dict()
    lines = [
        "PromptABI comparative study",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        f"labeled cases: {payload['case_count']}",
        f"promptabi detected: {payload['promptabi_detected_cases']}",
        f"real-bug reductions: {payload['real_bug_cases']}",
        "baselines:",
    ]
    for baseline in report.baselines:
        lines.append(
            f"  - {baseline.baseline.name}: "
            f"{baseline.covered_case_count}/{baseline.case_count} modeled, "
            f"{baseline.missed_case_count} missed"
        )
    return "\n".join(lines) + "\n"


def _compare_baseline(baseline: BaselineClass, cases: tuple[ComparativeCase, ...]) -> BaselineComparisonResult:
    covered: list[str] = []
    missed: list[str] = []
    promptabi_only_rules: set[str] = set()
    baseline_rules = set(baseline.covered_rule_ids)
    baseline_labels = set(baseline.covered_labels)
    for case in cases:
        expected_rules = set(case.expected_rule_ids)
        labels = set(case.labels)
        baseline_can_model = bool(expected_rules.intersection(baseline_rules) or labels.intersection(baseline_labels))
        if baseline_can_model:
            covered.append(case.case_id)
        else:
            missed.append(case.case_id)
            promptabi_only_rules.update(expected_rules)
    return BaselineComparisonResult(
        baseline=baseline,
        covered_case_ids=tuple(sorted(covered)),
        missed_case_ids=tuple(sorted(missed)),
        promptabi_only_rule_ids=tuple(sorted(promptabi_only_rules)),
    )


def _round_metric(value: float) -> float:
    return round(value, 6)
