"""Maintained grammar-backend conformance suite replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grammar_differential import (
    GrammarDifferentialCaseReport,
    GrammarDifferentialReport,
    GrammarDifferentialStatus,
    analyze_grammar_differential_mapping,
)
from .grammars import GrammarIngestionError


GRAMMAR_CONFORMANCE_VERSION = 1
DEFAULT_GRAMMAR_CONFORMANCE_SUITE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "grammar_conformance" / "suite.json"
)
REQUIRED_GRAMMAR_BACKENDS = (
    "outlines",
    "xgrammar",
    "llguidance",
    "lm-format-enforcer",
    "guidance",
    "instructor",
    "provider-native",
)


class GrammarConformanceError(ValueError):
    """Raised when a grammar-backend conformance suite is malformed."""


@dataclass(frozen=True, slots=True)
class GrammarBackendCoverage:
    """Coverage and replay status for one structured-output grammar backend."""

    backend_family: str
    case_ids: tuple[str, ...]
    declared_types: tuple[str, ...]
    features: tuple[str, ...]
    accepted_samples: int
    rejected_samples: int
    agreements: int
    mismatches: int
    abstentions: int

    @property
    def passed(self) -> bool:
        return (
            bool(self.case_ids)
            and self.accepted_samples > 0
            and self.rejected_samples > 0
            and self.mismatches == 0
            and self.abstentions == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_family": self.backend_family,
            "passed": self.passed,
            "case_ids": list(self.case_ids),
            "declared_types": list(self.declared_types),
            "features": list(self.features),
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "agreements": self.agreements,
            "mismatches": self.mismatches,
            "abstentions": self.abstentions,
        }


@dataclass(frozen=True, slots=True)
class GrammarConformanceReport:
    """Release-grade replay report for grammar backend conformance suites."""

    suite_version: int
    differential_report: GrammarDifferentialReport
    backend_coverage: tuple[GrammarBackendCoverage, ...]
    required_backends: tuple[str, ...] = REQUIRED_GRAMMAR_BACKENDS

    @property
    def case_count(self) -> int:
        return len(self.differential_report.cases)

    @property
    def sample_count(self) -> int:
        return sum(len(case.observations) for case in self.differential_report.cases)

    @property
    def all_cases_passed(self) -> bool:
        return (
            self.case_count > 0
            and not self.differential_report.mismatches
            and not self.differential_report.abstentions
            and all(coverage.passed for coverage in self.backend_coverage)
        )

    @property
    def missing_backends(self) -> tuple[str, ...]:
        observed = {coverage.backend_family for coverage in self.backend_coverage if coverage.case_ids}
        return tuple(backend for backend in self.required_backends if backend not in observed)

    @property
    def manifest_sha256(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_version": GRAMMAR_CONFORMANCE_VERSION,
            "suite_version": self.suite_version,
            "case_count": self.case_count,
            "sample_count": self.sample_count,
            "all_cases_passed": self.all_cases_passed,
            "required_backends": list(self.required_backends),
            "missing_backends": list(self.missing_backends),
            "backend_coverage": [coverage.to_dict() for coverage in self.backend_coverage],
            "cases": [case.to_dict() for case in self.differential_report.cases],
        }
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def build_grammar_conformance_report(path: str | Path | None = None) -> GrammarConformanceReport:
    """Replay the maintained grammar-backend suite against PromptABI semantics."""

    suite_path = Path(path) if path is not None else DEFAULT_GRAMMAR_CONFORMANCE_SUITE
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GrammarConformanceError(f"cannot read grammar conformance suite {suite_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GrammarConformanceError(f"grammar conformance suite is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise GrammarConformanceError("grammar conformance suite root must be an object")
    try:
        differential_report = analyze_grammar_differential_mapping(raw)
    except GrammarIngestionError as exc:
        raise GrammarConformanceError(str(exc)) from exc
    version = raw.get("version")
    if not isinstance(version, int) or version <= 0:
        raise GrammarConformanceError("grammar conformance suite requires a positive integer version")
    return GrammarConformanceReport(
        suite_version=version,
        differential_report=differential_report,
        backend_coverage=_backend_coverage(differential_report.cases),
    )


def write_grammar_conformance_manifest(path: str | Path, *, suite_path: str | Path | None = None) -> dict[str, object]:
    """Write the conformance report manifest as deterministic JSON."""

    report = build_grammar_conformance_report(suite_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_grammar_conformance_json(report), encoding="utf-8")
    return report.to_dict()


def render_grammar_conformance_json(report: GrammarConformanceReport | None = None) -> str:
    """Render grammar conformance as deterministic JSON."""

    resolved = report or build_grammar_conformance_report()
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"


def render_grammar_conformance_text(report: GrammarConformanceReport | None = None) -> str:
    """Render a concise conformance replay summary."""

    resolved = report or build_grammar_conformance_report()
    lines = [
        "PromptABI grammar backend conformance",
        f"status: {'PASS' if resolved.all_cases_passed else 'FAIL'}",
        f"cases: {resolved.case_count}",
        f"samples: {resolved.sample_count}",
        f"required backends: {', '.join(resolved.required_backends)}",
        f"manifest_sha256: {resolved.manifest_sha256}",
    ]
    for coverage in resolved.backend_coverage:
        status = "PASS" if coverage.passed else "FAIL"
        lines.append(
            f"- {coverage.backend_family}: {status} "
            f"({len(coverage.case_ids)} case(s), {coverage.accepted_samples}+/{coverage.rejected_samples}- samples, "
            f"{coverage.agreements} agreement(s), {coverage.mismatches} mismatch(es), {coverage.abstentions} abstention(s))"
        )
    if resolved.missing_backends:
        lines.append(f"missing: {', '.join(resolved.missing_backends)}")
    return "\n".join(lines) + "\n"


def _backend_coverage(cases: tuple[GrammarDifferentialCaseReport, ...]) -> tuple[GrammarBackendCoverage, ...]:
    by_backend: dict[str, list[GrammarDifferentialCaseReport]] = {backend: [] for backend in REQUIRED_GRAMMAR_BACKENDS}
    for case in cases:
        by_backend.setdefault(case.backend_family, []).append(case)
    return tuple(
        _coverage_for_backend(backend, tuple(by_backend.get(backend, ())))
        for backend in sorted(by_backend, key=_backend_sort_key)
    )


def _coverage_for_backend(
    backend: str,
    cases: tuple[GrammarDifferentialCaseReport, ...],
) -> GrammarBackendCoverage:
    observations = tuple(observation for case in cases for observation in case.observations)
    return GrammarBackendCoverage(
        backend_family=backend,
        case_ids=tuple(case.case_id for case in cases),
        declared_types=tuple(sorted({case.declared_type for case in cases})),
        features=tuple(sorted({feature for case in cases for feature in case.features})),
        accepted_samples=sum(1 for observation in observations if observation.sample.expected_accepts),
        rejected_samples=sum(1 for observation in observations if not observation.sample.expected_accepts),
        agreements=sum(1 for case in cases if case.status is GrammarDifferentialStatus.AGREEMENT),
        mismatches=sum(1 for case in cases if case.status is GrammarDifferentialStatus.MISMATCH),
        abstentions=sum(1 for case in cases if case.status is GrammarDifferentialStatus.ABSTAINED),
    )


def _backend_sort_key(backend: str) -> tuple[int, str]:
    try:
        return REQUIRED_GRAMMAR_BACKENDS.index(backend), backend
    except ValueError:
        return len(REQUIRED_GRAMMAR_BACKENDS), backend
