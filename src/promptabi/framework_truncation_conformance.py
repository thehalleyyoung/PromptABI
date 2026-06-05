"""Maintained framework-truncation conformance suite replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactKind,
    ArtifactLocation,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    TruncationStrategy,
)
from .budgets import TokenBudgetReport, analyze_token_budget
from .config import VerificationConfig
from .loaders import ArtifactLoader


FRAMEWORK_TRUNCATION_CONFORMANCE_VERSION = 1
DEFAULT_FRAMEWORK_TRUNCATION_CONFORMANCE_SUITE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "framework_truncation_conformance" / "suite.json"
)
REQUIRED_FRAMEWORK_TRUNCATION_FAMILIES = (
    "langchain",
    "llamaindex",
    "vllm",
    "transformers",
    "llama.cpp",
    "litellm",
    "openai-compatible",
    "custom-rag",
)


class FrameworkTruncationConformanceError(ValueError):
    """Raised when framework-truncation conformance fixtures are malformed."""


@dataclass(frozen=True, slots=True)
class FrameworkTruncationConformanceCaseReport:
    """Replay result for one framework truncation behavior case."""

    case_id: str
    framework: str
    strategy: str
    expected_strategy: str
    kept_segments: tuple[str, ...]
    expected_kept_segments: tuple[str, ...]
    dropped_segments: tuple[str, ...]
    expected_dropped_segments: tuple[str, ...]
    must_survive_status: str
    expected_must_survive_status: str
    rule_ids: tuple[str, ...]
    expected_rule_ids: tuple[str, ...]
    input_budget_tokens: int
    total_prompt_tokens: int | None
    evidence: str

    @property
    def passed(self) -> bool:
        return (
            self.strategy == self.expected_strategy
            and self.kept_segments == self.expected_kept_segments
            and self.dropped_segments == self.expected_dropped_segments
            and self.must_survive_status == self.expected_must_survive_status
            and self.rule_ids == self.expected_rule_ids
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "case_id": self.case_id,
            "framework": self.framework,
            "strategy": self.strategy,
            "expected_strategy": self.expected_strategy,
            "kept_segments": list(self.kept_segments),
            "expected_kept_segments": list(self.expected_kept_segments),
            "dropped_segments": list(self.dropped_segments),
            "expected_dropped_segments": list(self.expected_dropped_segments),
            "must_survive_status": self.must_survive_status,
            "expected_must_survive_status": self.expected_must_survive_status,
            "rule_ids": list(self.rule_ids),
            "expected_rule_ids": list(self.expected_rule_ids),
            "input_budget_tokens": self.input_budget_tokens,
            "passed": self.passed,
            "evidence": self.evidence,
        }
        if self.total_prompt_tokens is not None:
            data["total_prompt_tokens"] = self.total_prompt_tokens
        return data


@dataclass(frozen=True, slots=True)
class FrameworkTruncationCoverage:
    """Coverage and replay status for one framework family."""

    framework: str
    case_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    passed_cases: int

    @property
    def passed(self) -> bool:
        return bool(self.case_ids) and self.passed_cases == len(self.case_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "passed": self.passed,
            "case_ids": list(self.case_ids),
            "strategies": list(self.strategies),
            "passed_cases": self.passed_cases,
        }


@dataclass(frozen=True, slots=True)
class FrameworkTruncationConformanceReport:
    """Release-grade replay report for framework truncation policy conformance."""

    suite_version: int
    cases: tuple[FrameworkTruncationConformanceCaseReport, ...]
    framework_coverage: tuple[FrameworkTruncationCoverage, ...]
    required_frameworks: tuple[str, ...] = REQUIRED_FRAMEWORK_TRUNCATION_FAMILIES

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def all_cases_passed(self) -> bool:
        return (
            self.case_count > 0
            and all(case.passed for case in self.cases)
            and not self.missing_frameworks
            and all(coverage.passed for coverage in self.framework_coverage)
        )

    @property
    def missing_frameworks(self) -> tuple[str, ...]:
        observed = {coverage.framework for coverage in self.framework_coverage if coverage.case_ids}
        return tuple(framework for framework in self.required_frameworks if framework not in observed)

    @property
    def manifest_sha256(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_version": FRAMEWORK_TRUNCATION_CONFORMANCE_VERSION,
            "suite_version": self.suite_version,
            "case_count": self.case_count,
            "all_cases_passed": self.all_cases_passed,
            "required_frameworks": list(self.required_frameworks),
            "missing_frameworks": list(self.missing_frameworks),
            "framework_coverage": [coverage.to_dict() for coverage in self.framework_coverage],
            "cases": [case.to_dict() for case in self.cases],
        }
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def build_framework_truncation_conformance_report(
    path: str | Path | None = None,
) -> FrameworkTruncationConformanceReport:
    """Replay maintained truncation fixtures against the real budget analyzer."""

    suite_path = Path(path) if path is not None else DEFAULT_FRAMEWORK_TRUNCATION_CONFORMANCE_SUITE
    raw = _load_suite(suite_path)
    version = raw.get("version")
    if not isinstance(version, int) or version <= 0:
        raise FrameworkTruncationConformanceError("framework truncation suite requires a positive integer version")
    cases_data = raw.get("cases")
    if not isinstance(cases_data, list) or not cases_data:
        raise FrameworkTruncationConformanceError("framework truncation suite requires a non-empty cases array")
    cases = tuple(_replay_case(_mapping(case, "case")) for case in cases_data)
    return FrameworkTruncationConformanceReport(
        suite_version=version,
        cases=cases,
        framework_coverage=_framework_coverage(cases),
    )


def write_framework_truncation_conformance_manifest(
    path: str | Path,
    *,
    suite_path: str | Path | None = None,
) -> dict[str, object]:
    """Write a deterministic framework-truncation conformance manifest."""

    report = build_framework_truncation_conformance_report(suite_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_framework_truncation_conformance_json(report), encoding="utf-8")
    return report.to_dict()


def render_framework_truncation_conformance_json(
    report: FrameworkTruncationConformanceReport | None = None,
) -> str:
    """Render framework-truncation conformance as deterministic JSON."""

    resolved = report or build_framework_truncation_conformance_report()
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"


def render_framework_truncation_conformance_text(
    report: FrameworkTruncationConformanceReport | None = None,
) -> str:
    """Render a concise framework-truncation replay summary."""

    resolved = report or build_framework_truncation_conformance_report()
    lines = [
        "PromptABI framework truncation conformance",
        f"status: {'PASS' if resolved.all_cases_passed else 'FAIL'}",
        f"cases: {resolved.case_count}",
        f"required frameworks: {', '.join(resolved.required_frameworks)}",
        f"manifest_sha256: {resolved.manifest_sha256}",
    ]
    for coverage in resolved.framework_coverage:
        status = "PASS" if coverage.passed else "FAIL"
        lines.append(
            f"- {coverage.framework}: {status} "
            f"({len(coverage.case_ids)} case(s), strategies={', '.join(coverage.strategies) or 'none'})"
        )
    if resolved.missing_frameworks:
        lines.append(f"missing frameworks: {', '.join(resolved.missing_frameworks)}")
    failing = tuple(case.case_id for case in resolved.cases if not case.passed)
    if failing:
        lines.append(f"failing cases: {', '.join(failing)}")
    return "\n".join(lines) + "\n"


def _replay_case(data: dict[str, Any]) -> FrameworkTruncationConformanceCaseReport:
    case_id = _required_string(data, "id")
    framework = _required_string(data, "framework")
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name=f"{case_id}-segments",
        location=ArtifactLocation(uri=f"memory://framework-truncation/{case_id}/segments"),
        segments=tuple(_segment_from_mapping(_mapping(item, "segment")) for item in _required_list(data, "segments")),
    )
    policy_data = _mapping(data.get("policy"), "policy")
    policy = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name=f"{case_id}-policy",
        location=ArtifactLocation(uri=f"memory://framework-truncation/{case_id}/policy"),
        framework=_required_string(policy_data, "framework"),
        strategy=_strategy(policy_data.get("strategy")),
        max_context_tokens=_optional_positive_int(policy_data.get("max_context_tokens"), "policy.max_context_tokens"),
        reserve_output_tokens=_optional_non_negative_int(policy_data.get("reserve_output_tokens"), "policy.reserve_output_tokens") or 0,
        reserved_tool_tokens=_optional_non_negative_int(policy_data.get("reserved_tool_tokens"), "policy.reserved_tool_tokens") or 0,
        generation_prompt_tokens=_optional_non_negative_int(policy_data.get("generation_prompt_tokens"), "policy.generation_prompt_tokens") or 0,
        special_token_overhead=_optional_non_negative_int(policy_data.get("special_token_overhead"), "policy.special_token_overhead") or 0,
        preserve_system=bool(policy_data.get("preserve_system", False)),
        preserve_tools=bool(policy_data.get("preserve_tools", False)),
        drop_roles=_string_tuple(policy_data.get("drop_roles"), "policy.drop_roles"),
    )
    report = analyze_token_budget(
        VerificationConfig(name=case_id, artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(policy)),
    )
    expected = _mapping(data.get("expected"), "expected")
    if report.policy is None or report.truncation is None or report.reservation is None:
        raise FrameworkTruncationConformanceError(f"{case_id} did not produce a truncation report")
    return FrameworkTruncationConformanceCaseReport(
        case_id=case_id,
        framework=framework,
        strategy=report.policy.strategy,
        expected_strategy=_required_string(expected, "strategy"),
        kept_segments=tuple(segment.name for segment in report.truncation.kept_segments),
        expected_kept_segments=_string_tuple(expected.get("kept"), "expected.kept"),
        dropped_segments=tuple(segment.name for segment in report.truncation.dropped_segments),
        expected_dropped_segments=_string_tuple(expected.get("dropped"), "expected.dropped"),
        must_survive_status=(
            report.must_survive_proof.status
            if report.must_survive_proof is not None
            else "not-modeled"
        ),
        expected_must_survive_status=_required_string(expected, "must_survive_status"),
        rule_ids=tuple(finding.rule_id for finding in report.findings),
        expected_rule_ids=_string_tuple(expected.get("rule_ids"), "expected.rule_ids"),
        input_budget_tokens=report.reservation.input_budget_tokens,
        total_prompt_tokens=report.total_prompt_tokens,
        evidence=_required_string(data, "evidence"),
    )


def _framework_coverage(
    cases: tuple[FrameworkTruncationConformanceCaseReport, ...],
) -> tuple[FrameworkTruncationCoverage, ...]:
    coverage = []
    frameworks = sorted({*REQUIRED_FRAMEWORK_TRUNCATION_FAMILIES, *(case.framework for case in cases)})
    for framework in frameworks:
        framework_cases = tuple(case for case in cases if case.framework == framework)
        coverage.append(
            FrameworkTruncationCoverage(
                framework=framework,
                case_ids=tuple(case.case_id for case in framework_cases),
                strategies=tuple(sorted({case.strategy for case in framework_cases})),
                passed_cases=sum(1 for case in framework_cases if case.passed),
            )
        )
    return tuple(coverage)


def _segment_from_mapping(data: dict[str, Any]) -> PromptSegment:
    return PromptSegment(
        name=_required_string(data, "name"),
        role=_optional_string(data.get("role"), "segment.role"),
        required=bool(data.get("required", False)),
        max_tokens=_optional_positive_int(data.get("max_tokens"), "segment.max_tokens"),
        token_count=_optional_non_negative_int(data.get("token_count"), "segment.token_count"),
        overhead_tokens=_optional_non_negative_int(data.get("overhead_tokens"), "segment.overhead_tokens") or 0,
        metadata_tokens=_optional_non_negative_int(data.get("metadata_tokens"), "segment.metadata_tokens") or 0,
        template_overhead_tokens=_optional_non_negative_int(data.get("template_overhead_tokens"), "segment.template_overhead_tokens") or 0,
    )


def _load_suite(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FrameworkTruncationConformanceError(f"framework truncation suite not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FrameworkTruncationConformanceError(
            f"framework truncation suite is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    return _mapping(raw, "suite")


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrameworkTruncationConformanceError(f"{field_name} must be an object")
    return value


def _required_list(data: dict[str, Any], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise FrameworkTruncationConformanceError(f"{key} must be a non-empty array")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FrameworkTruncationConformanceError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise FrameworkTruncationConformanceError(f"{field_name} must be a non-empty string when set")
    return value.strip()


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise FrameworkTruncationConformanceError(f"{field_name} must be a positive integer when set")
    return value


def _optional_non_negative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise FrameworkTruncationConformanceError(f"{field_name} must be a non-negative integer when set")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise FrameworkTruncationConformanceError(f"{field_name} must be an array of non-empty strings")
    return tuple(item.strip() for item in value)


def _strategy(value: object) -> TruncationStrategy:
    if value is None:
        return TruncationStrategy.NONE
    if not isinstance(value, str):
        raise FrameworkTruncationConformanceError("policy.strategy must be a string when set")
    try:
        return TruncationStrategy(value)
    except ValueError as exc:
        choices = ", ".join(strategy.value for strategy in TruncationStrategy)
        raise FrameworkTruncationConformanceError(f"policy.strategy must be one of: {choices}") from exc
