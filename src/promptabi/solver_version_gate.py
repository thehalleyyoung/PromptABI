"""Prove solver-version compatibility gates (step 230).

A PromptABI verdict is only reproducible against the *solver stack that produced
it*.  When that stack changes -- a new Z3 release, a bumped finite-enumeration
engine version -- a cached "proven safe" verdict may no longer be trustworthy.
This module makes that boundary explicit and checkable.

A :class:`SolverVersionGate` records the solver-version fingerprints captured
when a contract was verified, plus a :class:`SolverVersionPolicy` declaring which
fingerprint components are *pinned* (any change invalidates the verdict and
forces re-verification) and which are *flexible* (changes are tolerated).
:func:`evaluate_solver_version_gate` compares a fresh fingerprint set against the
recorded baseline and proves whether the cached verdict may still be relied on,
naming every component that drifted and whether the drift is allowed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence

from .formal import _solver_version_fingerprints

SOLVER_VERSION_GATE_VERSION = "promptabi.solver-version-gate.v1"


def capture_solver_fingerprints(*, prefer_z3: bool = True) -> dict[str, str]:
    """Capture the current solver-version fingerprints as a component map."""

    return {str(name): str(version) for name, version in _solver_version_fingerprints(prefer_z3=prefer_z3)}


class SolverVersionDriftKind(StrEnum):
    PINNED_CHANGED = "pinned-changed"
    PINNED_MISSING = "pinned-missing"
    FLEXIBLE_CHANGED = "flexible-changed"
    NEW_COMPONENT = "new-component"


@dataclass(frozen=True, slots=True)
class SolverVersionPolicy:
    """Which fingerprint components are pinned (verdict-invalidating) vs flexible."""

    pinned: frozenset[str]
    flexible: frozenset[str] = field(default_factory=frozenset)
    # How to treat a component that appears in neither set.
    unknown_is_pinned: bool = True

    def classification(self, component: str) -> str:
        if component in self.pinned:
            return "pinned"
        if component in self.flexible:
            return "flexible"
        return "pinned" if self.unknown_is_pinned else "flexible"

    def to_dict(self) -> dict[str, object]:
        return {
            "pinned": sorted(self.pinned),
            "flexible": sorted(self.flexible),
            "unknown_is_pinned": self.unknown_is_pinned,
        }


@dataclass(frozen=True, slots=True)
class SolverVersionDrift:
    component: str
    kind: SolverVersionDriftKind
    baseline: str | None
    current: str | None

    @property
    def blocking(self) -> bool:
        return self.kind in (
            SolverVersionDriftKind.PINNED_CHANGED,
            SolverVersionDriftKind.PINNED_MISSING,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "kind": self.kind.value,
            "baseline": self.baseline,
            "current": self.current,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class SolverVersionGateReport:
    version: str
    compatible: bool
    drifts: tuple[SolverVersionDrift, ...] = field(default=())

    @property
    def blocking_drifts(self) -> tuple[SolverVersionDrift, ...]:
        return tuple(drift for drift in self.drifts if drift.blocking)

    @property
    def requires_reverification(self) -> bool:
        return not self.compatible

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "compatible": self.compatible,
            "requires_reverification": self.requires_reverification,
            "drifts": [drift.to_dict() for drift in self.drifts],
        }


@dataclass(frozen=True, slots=True)
class SolverVersionGate:
    """A recorded baseline fingerprint set plus the policy that governs drift."""

    baseline: Mapping[str, str]
    policy: SolverVersionPolicy

    def evaluate(self, current: Mapping[str, str]) -> SolverVersionGateReport:
        return evaluate_solver_version_gate(self.baseline, current, self.policy)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": dict(sorted(self.baseline.items())),
            "policy": self.policy.to_dict(),
        }


def evaluate_solver_version_gate(
    baseline: Mapping[str, str],
    current: Mapping[str, str],
    policy: SolverVersionPolicy,
) -> SolverVersionGateReport:
    """Prove whether a verdict recorded under ``baseline`` survives ``current``."""

    drifts: list[SolverVersionDrift] = []
    for component, baseline_version in sorted(baseline.items()):
        classification = policy.classification(component)
        if component not in current:
            if classification == "pinned":
                drifts.append(
                    SolverVersionDrift(
                        component=component,
                        kind=SolverVersionDriftKind.PINNED_MISSING,
                        baseline=baseline_version,
                        current=None,
                    )
                )
            continue
        current_version = current[component]
        if current_version == baseline_version:
            continue
        kind = (
            SolverVersionDriftKind.PINNED_CHANGED
            if classification == "pinned"
            else SolverVersionDriftKind.FLEXIBLE_CHANGED
        )
        drifts.append(
            SolverVersionDrift(
                component=component,
                kind=kind,
                baseline=baseline_version,
                current=current_version,
            )
        )
    for component, current_version in sorted(current.items()):
        if component in baseline:
            continue
        # A brand-new pinned component the baseline never saw is treated as drift.
        if policy.classification(component) == "pinned":
            drifts.append(
                SolverVersionDrift(
                    component=component,
                    kind=SolverVersionDriftKind.NEW_COMPONENT,
                    baseline=None,
                    current=current_version,
                )
            )

    compatible = not any(drift.blocking for drift in drifts) and not any(
        drift.kind is SolverVersionDriftKind.NEW_COMPONENT for drift in drifts
    )
    return SolverVersionGateReport(
        version=SOLVER_VERSION_GATE_VERSION,
        compatible=compatible,
        drifts=tuple(drifts),
    )


def render_solver_version_gate_json(report: SolverVersionGateReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_solver_version_gate_text(report: SolverVersionGateReport) -> str:
    lines = [
        f"PromptABI solver-version gate ({report.version})",
        f"status: {'COMPATIBLE' if report.compatible else 'REQUIRES RE-VERIFICATION'}",
        f"drifts: {len(report.drifts)} ({len(report.blocking_drifts)} blocking)",
    ]
    for drift in report.drifts:
        marker = "BLOCK" if drift.blocking else "ok"
        lines.append(
            f"  [{marker}] {drift.component}: {drift.baseline} -> {drift.current} ({drift.kind.value})"
        )
    return "\n".join(lines) + "\n"
