"""Verify that provider *downgrade* migrations have explicit, complete mitigation paths.

A downgrade migration moves a recorded source provider contract onto a target
provider that drops or weakens a capability (a request/response field, a tool
encoding, parallel tool calls, streaming fragments, a stop behavior, a context
window, a structured-output mode, or an error envelope).  Such a migration is
only *safe* if the source artifact declares a downgrade plan that mitigates
**every** real capability loss with a documented fallback.

This module does not re-derive provider semantics.  It runs the real
:func:`promptabi.provider_migration.analyze_provider_migration` analyzer over
recorded fixtures, then proves two soundness properties against the declared
plans, mirroring PromptABI's policy-pack discipline:

* **completeness** -- no actual capability loss is left unmitigated (no dropped
  loss);
* **honesty** -- no declared mitigation references a loss that does not actually
  occur (no fabricated mitigation).

A downgrade path is ``verified`` only when it is complete; spurious mitigations
are surfaced as warnings so plans stay tied to real fixture evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ProviderConfigArtifact
from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace
from .loaders import LoadedArtifact
from .provider_migration import (
    ProviderMigrationFinding,
    ProviderMigrationFindingKind,
    analyze_provider_migration,
)


PROVIDER_DOWNGRADE_PATH_VERSION = "promptabi.provider-downgrade-paths.v1"

#: Migration finding kinds that represent a recoverable *capability loss* and so
#: require an explicit downgrade mitigation. Structural defects (an unsupported
#: provider family, a missing routing target, an invalid adapter chain) are not
#: capability losses and cannot be waved away by a downgrade plan.
DOWNGRADE_LOSS_KINDS: tuple[ProviderMigrationFindingKind, ...] = (
    ProviderMigrationFindingKind.REQUEST_FIELD_LOSS,
    ProviderMigrationFindingKind.RESPONSE_FIELD_LOSS,
    ProviderMigrationFindingKind.TOOL_ARGUMENT_ENCODING_MISMATCH,
    ProviderMigrationFindingKind.TOOL_ID_MISMATCH,
    ProviderMigrationFindingKind.PARALLEL_TOOL_CALL_MISMATCH,
    ProviderMigrationFindingKind.STREAMING_CHUNK_MISMATCH,
    ProviderMigrationFindingKind.STOP_BEHAVIOR_MISMATCH,
    ProviderMigrationFindingKind.CONTEXT_LIMIT_REGRESSION,
    ProviderMigrationFindingKind.STRUCTURED_OUTPUT_MISMATCH,
    ProviderMigrationFindingKind.ERROR_SHAPE_MISMATCH,
)


class DowngradePathStatus(StrEnum):
    """Outcome for a single source->target downgrade migration."""

    NO_DOWNGRADE = "no-downgrade"
    VERIFIED = "verified"
    INCOMPLETE = "incomplete"
    UNDECLARED = "undeclared"


@dataclass(frozen=True, slots=True)
class DowngradeMitigation:
    """One declared mitigation for a capability loss in a downgrade plan."""

    loss: str
    fallback: str
    fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"loss": self.loss, "fallback": self.fallback, "fields": list(self.fields)}


@dataclass(frozen=True, slots=True)
class DowngradePathVerification:
    """Verification record for one recorded source->target downgrade migration."""

    source_artifact: str
    target_artifact: str
    status: DowngradePathStatus
    covered_losses: tuple[str, ...]
    uncovered_losses: tuple[str, ...]
    spurious_mitigations: tuple[str, ...]
    witness: WitnessTrace

    @property
    def verified(self) -> bool:
        return self.status is DowngradePathStatus.VERIFIED

    def to_dict(self) -> dict[str, object]:
        return {
            "covered_losses": list(self.covered_losses),
            "source_artifact": self.source_artifact,
            "spurious_mitigations": list(self.spurious_mitigations),
            "status": self.status.value,
            "target_artifact": self.target_artifact,
            "uncovered_losses": list(self.uncovered_losses),
            "verified": self.verified,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProviderDowngradePathReport:
    """Whole-config verification of every declared downgrade migration path."""

    version: str
    verifications: tuple[DowngradePathVerification, ...]

    @property
    def downgrades_checked(self) -> int:
        return sum(
            1
            for verification in self.verifications
            if verification.status is not DowngradePathStatus.NO_DOWNGRADE
        )

    @property
    def blocking(self) -> tuple[DowngradePathVerification, ...]:
        return tuple(
            verification
            for verification in self.verifications
            if verification.status in (DowngradePathStatus.INCOMPLETE, DowngradePathStatus.UNDECLARED)
        )

    @property
    def ok(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, object]:
        return {
            "blocking": len(self.blocking),
            "downgrades_checked": self.downgrades_checked,
            "ok": self.ok,
            "verifications": [verification.to_dict() for verification in self.verifications],
            "version": self.version,
        }


def verify_provider_downgrade_paths(
    loaded_artifacts: tuple[LoadedArtifact, ...],
) -> ProviderDowngradePathReport:
    """Verify that every declared downgrade migration mitigates all real losses.

    The analysis is offline and bounded: capability losses come from the real
    provider-migration analyzer over recorded fixtures, and mitigations come from
    the source artifact's declared ``provider_migration.downgrade_plans`` table.
    """

    migration = analyze_provider_migration(loaded_artifacts)
    plans_by_source = {
        loaded.artifact.name: _read_downgrade_plans(loaded)
        for loaded in loaded_artifacts
        if isinstance(loaded.artifact, ProviderConfigArtifact)
    }

    losses_by_pair: dict[tuple[str, str], dict[str, ProviderMigrationFinding]] = {}
    for finding in migration.findings:
        if finding.kind not in DOWNGRADE_LOSS_KINDS:
            continue
        if finding.target_artifact_name is None:
            continue
        pair = (finding.source_artifact_name, finding.target_artifact_name)
        losses_by_pair.setdefault(pair, {})[finding.kind.value] = finding

    pairs: set[tuple[str, str]] = set(losses_by_pair)
    for source, plans in plans_by_source.items():
        for target in plans:
            pairs.add((source, target))

    verifications: list[DowngradePathVerification] = []
    for source, target in sorted(pairs):
        losses = losses_by_pair.get((source, target), {})
        plans = plans_by_source.get(source, {})
        verifications.append(
            _verify_pair(source, target, losses, plans.get(target)),
        )

    return ProviderDowngradePathReport(
        version=PROVIDER_DOWNGRADE_PATH_VERSION,
        verifications=tuple(verifications),
    )


def render_provider_downgrade_paths_json(report: ProviderDowngradePathReport) -> str:
    """Render the downgrade-path report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_provider_downgrade_paths_text(report: ProviderDowngradePathReport) -> str:
    """Render the downgrade-path report for CI logs and reviewers."""

    lines = [
        f"PromptABI provider downgrade paths ({report.version})",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"downgrades_checked: {report.downgrades_checked}",
        f"blocking: {len(report.blocking)}",
    ]
    for verification in report.verifications:
        lines.append("")
        lines.append(
            f"{verification.source_artifact} -> {verification.target_artifact}: {verification.status.value}"
        )
        if verification.covered_losses:
            lines.append("  mitigated: " + ", ".join(verification.covered_losses))
        if verification.uncovered_losses:
            lines.append("  UNMITIGATED: " + ", ".join(verification.uncovered_losses))
        if verification.spurious_mitigations:
            lines.append("  spurious mitigations: " + ", ".join(verification.spurious_mitigations))
    return "\n".join(lines) + "\n"


def _verify_pair(
    source: str,
    target: str,
    losses: dict[str, ProviderMigrationFinding],
    mitigations: tuple[DowngradeMitigation, ...] | None,
) -> DowngradePathVerification:
    loss_kinds = tuple(sorted(losses))
    if not loss_kinds:
        return DowngradePathVerification(
            source_artifact=source,
            target_artifact=target,
            status=DowngradePathStatus.NO_DOWNGRADE,
            covered_losses=(),
            uncovered_losses=(),
            spurious_mitigations=(),
            witness=_witness(source, target, DowngradePathStatus.NO_DOWNGRADE, (), (), ()),
        )

    declared = {mitigation.loss for mitigation in mitigations} if mitigations is not None else set()
    covered = tuple(loss for loss in loss_kinds if loss in declared)
    uncovered = tuple(loss for loss in loss_kinds if loss not in declared)
    spurious = tuple(sorted(declared - set(loss_kinds)))

    if mitigations is None:
        status = DowngradePathStatus.UNDECLARED
    elif uncovered:
        status = DowngradePathStatus.INCOMPLETE
    else:
        status = DowngradePathStatus.VERIFIED

    return DowngradePathVerification(
        source_artifact=source,
        target_artifact=target,
        status=status,
        covered_losses=covered,
        uncovered_losses=uncovered,
        spurious_mitigations=spurious,
        witness=_witness(source, target, status, covered, uncovered, spurious),
    )


def _witness(
    source: str,
    target: str,
    status: DowngradePathStatus,
    covered: tuple[str, ...],
    uncovered: tuple[str, ...],
    spurious: tuple[str, ...],
) -> WitnessTrace:
    if status is DowngradePathStatus.NO_DOWNGRADE:
        summary = f"{source} -> {target}: no capability loss, no downgrade plan required"
        fixes: tuple[str, ...] = ()
    elif status is DowngradePathStatus.VERIFIED:
        summary = f"{source} -> {target}: every capability loss is mitigated by a declared fallback"
        fixes = ()
    elif status is DowngradePathStatus.UNDECLARED:
        summary = f"{source} -> {target}: downgrade drops capabilities but declares no downgrade plan"
        fixes = (
            "Declare a provider_migration.downgrade_plans entry mitigating: " + ", ".join(uncovered),
        )
    else:
        summary = f"{source} -> {target}: downgrade plan leaves capability losses unmitigated"
        fixes = (
            "Add downgrade mitigations for: " + ", ".join(uncovered),
        )

    steps = [
        WitnessStep(
            action="analyze provider migration losses",
            input=f"{source} -> {target}",
            output=", ".join(covered + uncovered) or "no-loss",
        ),
        WitnessStep(
            action="match declared downgrade mitigations",
            input=", ".join(covered) or "none",
            output=", ".join(uncovered) or "all-mitigated",
        ),
    ]
    if spurious:
        steps.append(
            WitnessStep(
                action="flag spurious mitigations",
                input=", ".join(spurious),
                output="mitigation references a loss that does not occur",
            )
        )
    return WitnessTrace(
        summary=summary,
        steps=tuple(steps),
        artifacts=(ArtifactRef(kind="provider-config", name=source, path=None),),
        minimal_fixes=fixes,
    )


def _read_downgrade_plans(
    loaded: LoadedArtifact,
) -> dict[str, tuple[DowngradeMitigation, ...]]:
    artifact = loaded.artifact
    assert isinstance(artifact, ProviderConfigArtifact)
    path = artifact.location.path
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    migration = raw.get("provider_migration")
    if not isinstance(migration, dict):
        return {}
    plans = migration.get("downgrade_plans")
    if not isinstance(plans, dict):
        return {}
    result: dict[str, tuple[DowngradeMitigation, ...]] = {}
    for target, plan in plans.items():
        if not isinstance(target, str):
            continue
        result[target] = _mitigations_from_plan(plan)
    return result


def _mitigations_from_plan(plan: Any) -> tuple[DowngradeMitigation, ...]:
    if not isinstance(plan, dict):
        return ()
    entries = plan.get("mitigations")
    if not isinstance(entries, list):
        return ()
    mitigations: list[DowngradeMitigation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        loss = entry.get("loss")
        if not isinstance(loss, str):
            continue
        fallback = entry.get("fallback")
        fields = entry.get("fields")
        mitigations.append(
            DowngradeMitigation(
                loss=loss,
                fallback=fallback if isinstance(fallback, str) else "",
                fields=tuple(value for value in fields if isinstance(value, str)) if isinstance(fields, list) else (),
            )
        )
    return tuple(mitigations)
