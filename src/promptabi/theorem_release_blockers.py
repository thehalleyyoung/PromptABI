"""Connect theorem traceability to concrete release blockers.

Theorem traceability (:mod:`promptabi.theorem_traceability`) proves that every
core theorem maps to executable specs, property tests, corpus fixtures, and a
release gate.  This module turns any gap in that mapping into a *first-class
release blocker*: a structured, replayable record that release automation can
enforce.  A candidate release is allowed only when every required theorem is
present in the proof catalog and fully traced; otherwise PromptABI emits a
blocker per unproven or untraced theorem, with the missing evidence kinds and a
concrete remediation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace
from .theorem_traceability import (
    THEOREM_TRACEABILITY_VERSION,
    TheoremTrace,
    build_theorem_traceability_report,
)


THEOREM_RELEASE_BLOCKER_VERSION = "promptabi.theorem-release-blockers.v1"

#: Theorems that must be proven and fully traced before any PromptABI release.
REQUIRED_RELEASE_THEOREMS: tuple[str, ...] = (
    "role-boundary-nonforgeability",
    "stop-overreachability",
    "grammar-tokenizer-emptiness",
    "must-survive-budget",
    "z3-backed-finite-contract",
)


class TheoremReleaseBlockerKind(StrEnum):
    """Concrete reasons a theorem blocks a release."""

    MISSING_REQUIRED_THEOREM = "missing-required-theorem"
    INCOMPLETE_EVIDENCE = "incomplete-evidence"
    STALE_OR_BROKEN_EVIDENCE = "stale-or-broken-evidence"


@dataclass(frozen=True, slots=True)
class TheoremReleaseBlocker:
    """One theorem-derived release blocker with a replayable witness."""

    property_id: str
    kind: TheoremReleaseBlockerKind
    reason: str
    missing_evidence_kinds: tuple[str, ...]
    failures: tuple[str, ...]
    remediation: str
    witness: WitnessTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "failures": list(self.failures),
            "kind": self.kind.value,
            "missing_evidence_kinds": list(self.missing_evidence_kinds),
            "property_id": self.property_id,
            "reason": self.reason,
            "remediation": self.remediation,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TheoremReleaseGateReport:
    """Release gate derived from theorem traceability."""

    version: str
    required_theorems: tuple[str, ...]
    traced_count: int
    proven_count: int
    blockers: tuple[TheoremReleaseBlocker, ...]

    @property
    def release_allowed(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "proven_count": self.proven_count,
            "release_allowed": self.release_allowed,
            "required_theorems": list(self.required_theorems),
            "traceability_version": THEOREM_TRACEABILITY_VERSION,
            "traced_count": self.traced_count,
            "version": self.version,
            "manifest_version": THEOREM_RELEASE_BLOCKER_VERSION,
        }


def derive_theorem_release_blockers(
    repo_root: str | Path | None = None,
    *,
    required_theorems: tuple[str, ...] = REQUIRED_RELEASE_THEOREMS,
    release_version: str = "1.0.0",
) -> TheoremReleaseGateReport:
    """Derive release blockers from the live theorem-to-test traceability report."""

    report = build_theorem_traceability_report(repo_root)
    traces_by_id = {trace.property_id: trace for trace in report.traces}
    proven_count = sum(1 for trace in report.traces if trace.passed)
    blockers: list[TheoremReleaseBlocker] = []

    for property_id in required_theorems:
        trace = traces_by_id.get(property_id)
        if trace is None:
            blockers.append(
                _missing_required_blocker(property_id, release_version)
            )
            continue
        if not trace.passed:
            blockers.append(_incomplete_blocker(trace, release_version))

    return TheoremReleaseGateReport(
        version=release_version,
        required_theorems=required_theorems,
        traced_count=len(report.traces),
        proven_count=proven_count,
        blockers=tuple(blockers),
    )


def render_theorem_release_blockers_json(report: TheoremReleaseGateReport) -> str:
    """Render the theorem release gate as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_theorem_release_blockers_text(report: TheoremReleaseGateReport) -> str:
    """Render the theorem release gate for CI logs and release automation."""

    status = "RELEASE-ALLOWED" if report.release_allowed else "RELEASE-BLOCKED"
    lines = [
        "PromptABI theorem-derived release gate",
        f"release_version: {report.version}",
        f"status: {status}",
        f"proven: {report.proven_count}/{report.traced_count}",
        f"required_theorems: {len(report.required_theorems)}",
    ]
    if report.release_allowed:
        lines.append("blockers: none")
        return "\n".join(lines) + "\n"
    lines.append(f"blockers: {len(report.blockers)}")
    for blocker in report.blockers:
        lines.append(f"BLOCK {blocker.kind.value} [{blocker.property_id}]: {blocker.reason}")
        if blocker.missing_evidence_kinds:
            lines.append("  missing evidence: " + ", ".join(blocker.missing_evidence_kinds))
        for failure in blocker.failures:
            lines.append(f"  failure: {failure}")
        lines.append(f"  remediation: {blocker.remediation}")
    return "\n".join(lines) + "\n"


def _missing_required_blocker(property_id: str, release_version: str) -> TheoremReleaseBlocker:
    reason = f"required theorem '{property_id}' is not present in the proof catalog"
    remediation = "Add the theorem to the proof catalog and register its traceability evidence before release."
    return TheoremReleaseBlocker(
        property_id=property_id,
        kind=TheoremReleaseBlockerKind.MISSING_REQUIRED_THEOREM,
        reason=reason,
        missing_evidence_kinds=(),
        failures=(),
        remediation=remediation,
        witness=_witness(property_id, TheoremReleaseBlockerKind.MISSING_REQUIRED_THEOREM, reason, remediation, release_version),
    )


def _incomplete_blocker(trace: TheoremTrace, release_version: str) -> TheoremReleaseBlocker:
    missing = tuple(kind.value for kind in trace.missing_kinds)
    if missing:
        kind = TheoremReleaseBlockerKind.INCOMPLETE_EVIDENCE
        reason = f"theorem '{trace.property_id}' is missing required evidence kinds"
        remediation = "Add the missing executable-spec/property-test/corpus/release-gate evidence link."
    else:
        kind = TheoremReleaseBlockerKind.STALE_OR_BROKEN_EVIDENCE
        reason = f"theorem '{trace.property_id}' has stale or broken evidence links"
        remediation = "Repair the broken evidence paths/symbols so the theorem stays traceable."
    return TheoremReleaseBlocker(
        property_id=trace.property_id,
        kind=kind,
        reason=reason,
        missing_evidence_kinds=missing,
        failures=trace.failures,
        remediation=remediation,
        witness=_witness(trace.property_id, kind, reason, remediation, release_version),
    )


def _witness(
    property_id: str,
    kind: TheoremReleaseBlockerKind,
    reason: str,
    remediation: str,
    release_version: str,
) -> WitnessTrace:
    return WitnessTrace(
        summary=f"theorem '{property_id}' blocks release {release_version}: {kind.value}",
        steps=(
            WitnessStep(action="read theorem traceability report", input=property_id, output=reason),
            WitnessStep(action="evaluate release gate", input=release_version, output=kind.value),
            WitnessStep(action="emit minimal release fix", input=kind.value, output=remediation),
        ),
        artifacts=(ArtifactRef(kind="theorem", name=property_id, path="src/promptabi/theorem_traceability.py"),),
        minimal_fixes=(remediation,),
    )
