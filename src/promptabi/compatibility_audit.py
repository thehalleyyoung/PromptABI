"""Post-1.0 compatibility audit for minor-release candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .corpus_verification import CorpusVerificationThresholds, run_corpus_verification
from .grammar_differential import analyze_grammar_differential_corpus
from .loaders import ArtifactLoader
from .provider_fixture_packs import load_provider_fixture_pack_corpus
from .provider_fixture_replay import analyze_provider_fixture_replay
from .seed_corpus import load_seed_corpus
from .structured_schema_corpus import load_structured_schema_corpus, validate_structured_schema_entry


DEFAULT_GRAMMAR_DIFFERENTIAL_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "grammar_differential" / "corpus.json"
)
COMPATIBILITY_AUDIT_VERSION = 1
COMPATIBILITY_AUDIT_SURFACES = ("tokenizer", "template", "provider", "grammar", "framework")


class CompatibilityAuditStatus(StrEnum):
    """Outcome for one pre-release compatibility-audit surface."""

    PASS = "pass"
    FAIL = "fail"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class CompatibilityAuditTarget:
    """One surface/version pair evaluated by the compatibility audit."""

    surface: str
    requested_version: str
    observed_versions: tuple[str, ...]
    status: CompatibilityAuditStatus
    summary: str
    coverage_count: int
    failures: tuple[str, ...] = ()
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CompatibilityAuditStatus.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "requested_version": self.requested_version,
            "observed_versions": list(self.observed_versions),
            "status": self.status.value,
            "summary": self.summary,
            "coverage_count": self.coverage_count,
            "failures": list(self.failures),
            "evidence": dict(sorted(self.evidence.items())),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityAuditReport:
    """Fixture-backed compatibility audit for a candidate minor release."""

    targets: tuple[CompatibilityAuditTarget, ...]
    corpus_gate_ok: bool

    @property
    def ok(self) -> bool:
        return self.corpus_gate_ok and all(target.passed for target in self.targets)

    @property
    def coverage_count(self) -> int:
        return sum(target.coverage_count for target in self.targets)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": COMPATIBILITY_AUDIT_VERSION,
            "ok": self.ok,
            "corpus_gate_ok": self.corpus_gate_ok,
            "coverage_count": self.coverage_count,
            "targets": [target.to_dict() for target in self.targets],
        }


class CompatibilityAuditError(ValueError):
    """Raised when a compatibility-audit request is malformed."""


def run_compatibility_audit(
    candidate_versions: Mapping[str, str],
    *,
    seed_root: str | Path | None = None,
    provider_fixture_root: str | Path | None = None,
    structured_schema_root: str | Path | None = None,
    grammar_differential_corpus: str | Path | None = None,
    thresholds: CorpusVerificationThresholds | None = None,
) -> CompatibilityAuditReport:
    """Run the post-1.0 compatibility audit against local, pinned fixtures.

    The audit deliberately refuses to certify arbitrary upstream releases. A
    target only passes when the requested version matches the local fixture
    revision that was actually replayed; otherwise the target abstains with the
    observed fixture versions needed to add coverage.
    """

    resolved_versions = normalize_candidate_versions(candidate_versions)
    corpus_gate = run_corpus_verification(
        seed_root=seed_root,
        structured_schema_root=structured_schema_root,
        provider_fixture_root=provider_fixture_root,
        thresholds=thresholds,
    )
    seed = load_seed_corpus(seed_root)
    provider = load_provider_fixture_pack_corpus(provider_fixture_root)
    structured = load_structured_schema_corpus(structured_schema_root)
    grammar_path = Path(grammar_differential_corpus) if grammar_differential_corpus is not None else DEFAULT_GRAMMAR_DIFFERENTIAL_CORPUS_PATH

    targets = (
        _seed_surface_target("tokenizer", resolved_versions["tokenizer"], seed, corpus_gate.ok),
        _seed_surface_target("template", resolved_versions["template"], seed, corpus_gate.ok),
        _provider_target(resolved_versions["provider"], provider),
        _grammar_target(resolved_versions["grammar"], grammar_path),
        _framework_target(resolved_versions["framework"], structured),
    )
    return CompatibilityAuditReport(targets=targets, corpus_gate_ok=corpus_gate.ok)


def normalize_candidate_versions(candidate_versions: Mapping[str, str]) -> dict[str, str]:
    """Normalize CLI/API candidate-version declarations to every audit surface."""

    if not candidate_versions:
        raise CompatibilityAuditError("at least one candidate version is required")
    normalized = {key.lower(): value for key, value in candidate_versions.items()}
    unknown = sorted(set(normalized).difference((*COMPATIBILITY_AUDIT_SURFACES, "all")))
    if unknown:
        raise CompatibilityAuditError(f"unknown compatibility-audit surface(s): {', '.join(unknown)}")
    default = normalized.get("all")
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for surface in COMPATIBILITY_AUDIT_SURFACES:
        value = normalized.get(surface, default)
        if not value:
            missing.append(surface)
        else:
            resolved[surface] = value
    if missing:
        raise CompatibilityAuditError(
            "candidate versions are missing for: "
            + ", ".join(missing)
            + " (use SURFACE=VERSION or all=VERSION)"
        )
    return resolved


def render_compatibility_audit_json(report: CompatibilityAuditReport) -> str:
    """Render compatibility audit output as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_compatibility_audit_text(report: CompatibilityAuditReport) -> str:
    """Render compatibility audit output for release maintainers."""

    lines = [
        "PromptABI compatibility audit",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"corpus gate: {'PASS' if report.corpus_gate_ok else 'FAIL'}",
        f"coverage: {report.coverage_count} fixture-backed replay(s)",
    ]
    for target in report.targets:
        lines.append(
            f"- {target.surface}: {target.status.value.upper()} "
            f"requested={target.requested_version} observed={', '.join(target.observed_versions)} "
            f"({target.coverage_count}) {target.summary}"
        )
        for failure in target.failures:
            lines.append(f"  failure: {failure}")
    return "\n".join(lines) + "\n"


def _seed_surface_target(surface: str, requested_version: str, seed, corpus_gate_ok: bool) -> CompatibilityAuditTarget:
    observed = _versions(entry.metadata["fixture_revision"] for entry in seed.entries)
    if requested_version not in observed:
        return _abstain(
            surface,
            requested_version,
            observed,
            f"no pinned seed-corpus fixture for {surface} version {requested_version}",
            len(seed.entries),
            {
                "families": list(seed.families),
                "entry_count": len(seed.entries),
                "upstream_revisions": sorted(str(entry.metadata["upstream_revision"]) for entry in seed.entries),
            },
        )
    status = CompatibilityAuditStatus.PASS if corpus_gate_ok else CompatibilityAuditStatus.FAIL
    failures = () if corpus_gate_ok else ("maintained corpus verification failed while replaying seed corpus",)
    return CompatibilityAuditTarget(
        surface=surface,
        requested_version=requested_version,
        observed_versions=observed,
        status=status,
        summary=f"replayed {len(seed.entries)} seed corpus entries across {len(seed.families)} families",
        coverage_count=len(seed.entries),
        failures=failures,
        evidence={"families": list(seed.families), "entry_count": len(seed.entries)},
    )


def _provider_target(requested_version: str, provider) -> CompatibilityAuditTarget:
    observed = _versions(entry.metadata["fixture_revision"] for entry in provider.entries)
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in provider.artifact_bundle())
    replay = analyze_provider_fixture_replay(loaded)
    if requested_version not in observed:
        return _abstain(
            "provider",
            requested_version,
            observed,
            f"no pinned provider fixture pack for version {requested_version}",
            replay.fixtures_checked,
            {"provider_families": list(provider.provider_families), "replay_hash": replay.replay_hash},
        )
    failures = tuple(f"{finding.artifact_name}: {finding.message}" for finding in replay.findings)
    return CompatibilityAuditTarget(
        surface="provider",
        requested_version=requested_version,
        observed_versions=observed,
        status=CompatibilityAuditStatus.FAIL if failures else CompatibilityAuditStatus.PASS,
        summary=f"replayed {replay.fixtures_checked} provider fixture packs",
        coverage_count=replay.fixtures_checked,
        failures=failures,
        evidence={"provider_families": list(replay.provider_families), "replay_hash": replay.replay_hash},
    )


def _grammar_target(requested_version: str, corpus_path: Path) -> CompatibilityAuditTarget:
    report = analyze_grammar_differential_corpus(corpus_path)
    observed = (f"grammar-differential-v{report.version}",)
    if requested_version not in observed:
        return _abstain(
            "grammar",
            requested_version,
            observed,
            f"no pinned grammar differential corpus for version {requested_version}",
            len(report.cases),
            {"corpus": str(corpus_path)},
        )
    failures = tuple(
        [f"{case.case_id}: {case.reason or 'backend-label mismatch'}" for case in report.mismatches]
        + [f"{case.case_id}: abstained: {case.reason or 'unsupported grammar fragment'}" for case in report.abstentions]
    )
    return CompatibilityAuditTarget(
        surface="grammar",
        requested_version=requested_version,
        observed_versions=observed,
        status=CompatibilityAuditStatus.FAIL if failures else CompatibilityAuditStatus.PASS,
        summary=f"replayed {len(report.cases)} grammar backend differential cases",
        coverage_count=len(report.cases),
        failures=failures,
        evidence={
            "agreements": len(report.agreements),
            "mismatches": len(report.mismatches),
            "abstentions": len(report.abstentions),
        },
    )


def _framework_target(requested_version: str, structured) -> CompatibilityAuditTarget:
    observed = _versions(entry.metadata["fixture_revision"] for entry in structured.entries)
    parser_replays = 0
    statuses: dict[str, int] = {}
    for entry in structured.entries:
        status = validate_structured_schema_entry(entry)
        if status is not None:
            parser_replays += 1
            statuses[status.value] = statuses.get(status.value, 0) + 1
    if requested_version not in observed:
        return _abstain(
            "framework",
            requested_version,
            observed,
            f"no pinned parser/framework compatibility fixture for version {requested_version}",
            parser_replays,
            {"entry_types": list(structured.entry_types), "parser_statuses": statuses},
        )
    return CompatibilityAuditTarget(
        surface="framework",
        requested_version=requested_version,
        observed_versions=observed,
        status=CompatibilityAuditStatus.PASS,
        summary=f"replayed {parser_replays} parser/framework compatibility fixtures",
        coverage_count=parser_replays,
        evidence={"entry_types": list(structured.entry_types), "parser_statuses": statuses},
    )


def _abstain(
    surface: str,
    requested_version: str,
    observed_versions: tuple[str, ...],
    reason: str,
    coverage_count: int,
    evidence: dict[str, object],
) -> CompatibilityAuditTarget:
    return CompatibilityAuditTarget(
        surface=surface,
        requested_version=requested_version,
        observed_versions=observed_versions,
        status=CompatibilityAuditStatus.ABSTAINED,
        summary="candidate version is not covered by local pinned fixtures",
        coverage_count=coverage_count,
        failures=(reason,),
        evidence=evidence,
    )


def _versions(values) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))
