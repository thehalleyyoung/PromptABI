"""Provider-fixture conformance suites over recorded offline API contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loaders import ArtifactLoader
from .provider_fixture_packs import (
    REQUIRED_PROVIDER_FIXTURE_FAMILIES,
    ProviderFixturePackEntry,
    load_provider_fixture_pack_corpus,
)
from .provider_fixture_replay import ProviderFixtureReplayFinding, analyze_provider_fixture_replay


PROVIDER_CONFORMANCE_VERSION = 1
REQUIRED_PROVIDER_CONFORMANCE_SURFACES = (
    "tool-call-streaming",
    "parallel-tool-calls",
    "json-mode",
    "response-formats",
    "stop-handling",
    "error-shapes",
    "context-window-limits",
)


class ProviderConformanceError(ValueError):
    """Raised when provider conformance fixtures cannot be replayed."""


@dataclass(frozen=True, slots=True)
class ProviderSurfaceCoverage:
    """Coverage for one provider behavior surface across recorded fixtures."""

    surface: str
    provider_ids: tuple[str, ...]
    provider_families: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]

    @property
    def passed(self) -> bool:
        return bool(self.provider_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "passed": self.passed,
            "provider_ids": list(self.provider_ids),
            "provider_families": list(self.provider_families),
            "evidence": [{"provider_id": provider_id, "detail": detail} for provider_id, detail in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ProviderConformanceReport:
    """Release-grade provider conformance report from local fixture packs."""

    manifest_version: int
    provider_count: int
    provider_families: tuple[str, ...]
    required_provider_families: tuple[str, ...]
    surface_coverage: tuple[ProviderSurfaceCoverage, ...]
    replay_findings: tuple[ProviderFixtureReplayFinding, ...]
    replay_hash: str

    @property
    def missing_provider_families(self) -> tuple[str, ...]:
        observed = set(self.provider_families)
        return tuple(family for family in self.required_provider_families if family not in observed)

    @property
    def missing_surfaces(self) -> tuple[str, ...]:
        return tuple(coverage.surface for coverage in self.surface_coverage if not coverage.passed)

    @property
    def all_cases_passed(self) -> bool:
        return self.provider_count > 0 and not self.missing_provider_families and not self.missing_surfaces and not self.replay_findings

    @property
    def manifest_sha256(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_version": self.manifest_version,
            "provider_count": self.provider_count,
            "provider_families": list(self.provider_families),
            "required_provider_families": list(self.required_provider_families),
            "missing_provider_families": list(self.missing_provider_families),
            "required_surfaces": list(REQUIRED_PROVIDER_CONFORMANCE_SURFACES),
            "missing_surfaces": list(self.missing_surfaces),
            "all_cases_passed": self.all_cases_passed,
            "replay_hash": self.replay_hash,
            "surface_coverage": [coverage.to_dict() for coverage in self.surface_coverage],
            "replay_findings": [
                {
                    "artifact_name": finding.artifact_name,
                    "kind": finding.kind.value,
                    "severity": finding.severity,
                    "message": finding.message,
                    "provider": finding.provider,
                }
                for finding in self.replay_findings
            ],
        }
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def build_provider_conformance_report(root: str | Path | None = None) -> ProviderConformanceReport:
    """Replay provider fixtures and summarize required provider behavior coverage."""

    corpus = load_provider_fixture_pack_corpus(root)
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in corpus.artifact_bundle())
    replay = analyze_provider_fixture_replay(loaded)
    entries = tuple(corpus.entries)
    return ProviderConformanceReport(
        manifest_version=PROVIDER_CONFORMANCE_VERSION,
        provider_count=len(entries),
        provider_families=tuple(sorted({entry.provider_family for entry in entries})),
        required_provider_families=tuple(sorted(REQUIRED_PROVIDER_FIXTURE_FAMILIES)),
        surface_coverage=tuple(_surface_coverage(surface, entries) for surface in REQUIRED_PROVIDER_CONFORMANCE_SURFACES),
        replay_findings=replay.findings,
        replay_hash=replay.replay_hash,
    )


def write_provider_conformance_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Write a deterministic provider conformance manifest."""

    report = build_provider_conformance_report(root)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_provider_conformance_json(report), encoding="utf-8")
    return report.to_dict()


def render_provider_conformance_json(report: ProviderConformanceReport | None = None) -> str:
    """Render provider conformance as deterministic JSON."""

    resolved = report or build_provider_conformance_report()
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"


def render_provider_conformance_text(report: ProviderConformanceReport | None = None) -> str:
    """Render a concise provider conformance replay summary."""

    resolved = report or build_provider_conformance_report()
    lines = [
        "PromptABI provider fixture conformance",
        f"status: {'PASS' if resolved.all_cases_passed else 'FAIL'}",
        f"providers: {resolved.provider_count}",
        f"provider families: {', '.join(resolved.provider_families)}",
        f"required surfaces: {', '.join(REQUIRED_PROVIDER_CONFORMANCE_SURFACES)}",
        f"manifest_sha256: {resolved.manifest_sha256}",
    ]
    for coverage in resolved.surface_coverage:
        status = "PASS" if coverage.passed else "FAIL"
        lines.append(
            f"- {coverage.surface}: {status} "
            f"({len(coverage.provider_ids)} provider(s): {', '.join(coverage.provider_ids) or 'none'})"
        )
    if resolved.missing_provider_families:
        lines.append(f"missing provider families: {', '.join(resolved.missing_provider_families)}")
    if resolved.replay_findings:
        lines.append(f"replay findings: {len(resolved.replay_findings)}")
    return "\n".join(lines) + "\n"


def _surface_coverage(
    surface: str,
    entries: tuple[ProviderFixturePackEntry, ...],
) -> ProviderSurfaceCoverage:
    covered: list[ProviderFixturePackEntry] = []
    evidence: list[tuple[str, str]] = []
    for entry in entries:
        detail = _surface_evidence(surface, entry.pack)
        if detail is not None:
            covered.append(entry)
            evidence.append((entry.entry_id, detail))
    return ProviderSurfaceCoverage(
        surface=surface,
        provider_ids=tuple(entry.entry_id for entry in covered),
        provider_families=tuple(sorted({entry.provider_family for entry in covered})),
        evidence=tuple(evidence),
    )


def _surface_evidence(surface: str, pack: dict[str, object]) -> str | None:
    request = _mapping(pack.get("request"))
    response = _mapping(pack.get("response"))
    tool_calls = _mapping(response.get("tool_calls"))
    stops = _mapping(pack.get("stops"))
    streaming = _mapping(pack.get("streaming"))
    errors = _mapping(pack.get("errors"))
    limits = _mapping(pack.get("limits"))
    request_fields = set(_string_tuple(request.get("fields")))
    response_fields = set(_string_tuple(response.get("fields")))
    finish_reasons = _string_tuple(response.get("finish_reasons"))

    if surface == "tool-call-streaming":
        if streaming.get("emits_argument_fragments") is True and _string(streaming.get("assembly_key")):
            return f"streaming fragments assembled by {streaming['assembly_key']}"
        return None
    if surface == "parallel-tool-calls":
        parallel_limit = limits.get("parallel_tool_call_limit")
        supports_parallel = tool_calls.get("supports_parallel_tool_calls") is True
        if supports_parallel or (_positive_int(parallel_limit) and int(parallel_limit) > 1):
            return f"parallel={supports_parallel}, limit={parallel_limit}"
        return None
    if surface == "json-mode":
        if (
            "response_format" in request_fields
            or "guided_json" in request_fields
            or "responseSchema" in request_fields
            or "responseMimeType" in request_fields
        ):
            return f"request fields: {', '.join(sorted(request_fields))}"
        return None
    if surface == "response-formats":
        if response_fields and finish_reasons and all(_string(tool_calls.get(field)) for field in ("name_path", "arguments_path", "argument_encoding")):
            return f"{len(response_fields)} response fields, {len(finish_reasons)} finish reasons"
        return None
    if surface == "stop-handling":
        if _string_tuple(stops.get("sequences")) and _string(stops.get("finish_reason_path")) and isinstance(stops.get("truncates_before_parser"), bool):
            return f"{len(_string_tuple(stops.get('sequences')))} stops, finish path {stops['finish_reason_path']}"
        return None
    if surface == "error-shapes":
        if all(_string(errors.get(field)) for field in ("code_path", "message_path", "rate_limit_path")) and isinstance(errors.get("sample"), dict):
            return f"code={errors['code_path']}, rate_limit={errors['rate_limit_path']}"
        return None
    if surface == "context-window-limits":
        if _positive_int(limits.get("max_input_tokens")) and _positive_int(limits.get("max_output_tokens")):
            return f"input={limits['max_input_tokens']}, output={limits['max_output_tokens']}"
        return None
    raise ProviderConformanceError(f"unknown provider conformance surface: {surface}")


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
