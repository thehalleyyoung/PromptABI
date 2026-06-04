"""Configuration-to-configuration contract diffing for PromptABI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import Artifact, ArtifactKind, FrameworkTruncationConfigArtifact
from .config import VerificationConfig
from .diagnostics import ArtifactRef, CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace, diagnostic_sort_key
from .loaders import LoadedArtifact
from .provider_migration import ProviderMigrationFinding, compare_provider_config_artifacts
from .session import VerificationResult, VerificationSession
from .tokenizer_drift import (
    TokenizerDriftFinding,
    compare_tokenizer_config_snapshots,
    load_tokenizer_config_snapshot,
)


DIFF_CHECK_MODES = (CheckMode.BOUNDED, CheckMode.HEURISTIC)
DIFF_ABSTAIN_MODES = (CheckMode.ABSTAINING, CheckMode.BOUNDED)
TOKENIZER_DIFF_MODES = (CheckMode.SOUND, CheckMode.COMPLETE)
PROVIDER_DIFF_MODES = (CheckMode.BOUNDED, CheckMode.HEURISTIC)
FRAMEWORK_DIFF_MODES = (CheckMode.SOUND, CheckMode.BOUNDED)


@dataclass(frozen=True, slots=True)
class ConfigDiffInputs:
    """Loaded inputs for one side of a PromptABI config diff."""

    label: str
    config: VerificationConfig
    loaded_artifacts: tuple[LoadedArtifact, ...]
    load_diagnostics: tuple[Diagnostic, ...]

    @property
    def artifacts_by_name(self) -> dict[str, Artifact]:
        return {artifact.name: artifact for artifact in self.config.artifact_bundle}

    @property
    def loaded_by_name(self) -> dict[str, LoadedArtifact]:
        return {loaded.artifact.name: loaded for loaded in self.loaded_artifacts}


def diff_config_files(baseline_path: str | Path, current_path: str | Path) -> VerificationResult:
    """Load two config files and compare contract-relevant surfaces."""

    baseline = _load_side("baseline", VerificationSession.from_config_file(baseline_path))
    current = _load_side("current", VerificationSession.from_config_file(current_path))
    return diff_configs(baseline, current)


def diff_configs(baseline: ConfigDiffInputs, current: ConfigDiffInputs) -> VerificationResult:
    """Compare two already-loaded PromptABI configs."""

    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_load_abstention_diagnostics(baseline))
    diagnostics.extend(_load_abstention_diagnostics(current))
    diagnostics.extend(_config_diagnostics(baseline.config, current.config))
    diagnostics.extend(_artifact_set_diagnostics(baseline, current))
    diagnostics.extend(_paired_artifact_diagnostics(baseline, current))
    if not diagnostics:
        diagnostics.append(_clean_diagnostic(baseline.config, current.config))
    diagnostics.sort(key=diagnostic_sort_key)
    return VerificationResult(
        config=VerificationConfig(
            name=f"{baseline.config.name} -> {current.config.name}",
            checks=("configuration-diff",),
            max_context_tokens=current.config.max_context_tokens,
        ),
        diagnostics=tuple(diagnostics),
    )


def _load_side(label: str, session: VerificationSession) -> ConfigDiffInputs:
    loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
    return ConfigDiffInputs(
        label=label,
        config=session.config,
        loaded_artifacts=loaded_artifacts,
        load_diagnostics=load_diagnostics,
    )


def _load_abstention_diagnostics(side: ConfigDiffInputs) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    for diagnostic in side.load_diagnostics:
        if diagnostic.severity is not DiagnosticSeverity.ERROR:
            continue
        diagnostics.append(
            Diagnostic(
                rule_id="diff-abstained",
                severity=DiagnosticSeverity.ERROR,
                message=f"{side.label} config artifact loading did not complete cleanly: {diagnostic.message}",
                artifact=diagnostic.artifact,
                span=diagnostic.span,
                check_modes=DIFF_ABSTAIN_MODES,
                suggestions=("Fix artifact loading before trusting configuration diff results.",),
                witness=WitnessTrace(
                    summary="PromptABI refuses to treat an unloaded artifact as an unchanged contract.",
                    steps=(
                        WitnessStep(action="load diff side", input=side.label, output=diagnostic.rule_id),
                        WitnessStep(action="preserve original loader finding", output=diagnostic.message),
                    ),
                    artifacts=(diagnostic.artifact,) if diagnostic.artifact is not None else (),
                ),
                properties=(("side", side.label), ("source_rule_id", diagnostic.rule_id)),
            )
        )
    return tuple(diagnostics)


def _config_diagnostics(baseline: VerificationConfig, current: VerificationConfig) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    removed_checks = tuple(sorted(set(baseline.checks).difference(current.checks)))
    added_checks = tuple(sorted(set(current.checks).difference(baseline.checks)))
    if removed_checks:
        diagnostics.append(
            Diagnostic(
                rule_id="diff-check-removed",
                severity=DiagnosticSeverity.ERROR,
                message="current config removes verification checks that protected the baseline contract",
                check_modes=DIFF_CHECK_MODES,
                suggestions=("Keep the baseline checks enabled or document a policy exception before migration.",),
                witness=WitnessTrace(
                    summary="A removed check shrinks the set of contracts PromptABI is asked to enforce.",
                    steps=(
                        WitnessStep(action="compare checks", input="baseline", output=", ".join(baseline.checks)),
                        WitnessStep(action="compare checks", input="current", output=", ".join(current.checks) or "<none>"),
                        WitnessStep(action="find removed checks", output=", ".join(removed_checks)),
                    ),
                ),
                properties=(("removed_checks", removed_checks),),
            )
        )
    if added_checks:
        diagnostics.append(
            Diagnostic(
                rule_id="diff-check-added",
                severity=DiagnosticSeverity.INFO,
                message="current config adds verification checks beyond the baseline contract",
                check_modes=DIFF_CHECK_MODES,
                suggestions=("Review new diagnostics before making the added check release-blocking.",),
                witness=WitnessTrace(
                    summary="The current config asks PromptABI to enforce additional contracts.",
                    steps=(WitnessStep(action="find added checks", output=", ".join(added_checks)),),
                ),
                properties=(("added_checks", added_checks),),
            )
        )
    if _regresses_limit(baseline.max_context_tokens, current.max_context_tokens):
        diagnostics.append(
            Diagnostic(
                rule_id="diff-context-regression",
                severity=DiagnosticSeverity.ERROR,
                message="current config lowers the declared maximum context budget",
                check_modes=FRAMEWORK_DIFF_MODES,
                suggestions=("Re-run must-survive budget checks with the lower limit or keep the baseline context budget.",),
                witness=WitnessTrace(
                    summary="A smaller top-level context budget can make previously surviving prompt regions droppable.",
                    steps=(
                        WitnessStep(action="read baseline max_context_tokens", output=str(baseline.max_context_tokens)),
                        WitnessStep(action="read current max_context_tokens", output=str(current.max_context_tokens)),
                    ),
                ),
                properties=(
                    ("baseline_max_context_tokens", baseline.max_context_tokens),
                    ("current_max_context_tokens", current.max_context_tokens),
                ),
            )
        )
    return tuple(diagnostics)


def _artifact_set_diagnostics(baseline: ConfigDiffInputs, current: ConfigDiffInputs) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    baseline_artifacts = baseline.artifacts_by_name
    current_artifacts = current.artifacts_by_name
    for name in sorted(set(baseline_artifacts).difference(current_artifacts)):
        artifact = baseline_artifacts[name]
        diagnostics.append(
            Diagnostic(
                rule_id="diff-artifact-removed",
                severity=DiagnosticSeverity.ERROR,
                message=f"current config removes baseline {artifact.kind.value} artifact '{name}'",
                artifact=artifact.to_ref(),
                span=artifact.source_span,
                check_modes=DIFF_CHECK_MODES,
                suggestions=("Keep the artifact or remove dependent checks only with an explicit migration review.",),
                witness=WitnessTrace(
                    summary="A baseline artifact disappeared from the current configuration.",
                    steps=(WitnessStep(action="pair artifacts by name", input=name, output="missing in current"),),
                    artifacts=(artifact.to_ref(),),
                ),
                properties=(("artifact_name", name), ("baseline_kind", artifact.kind.value)),
            )
        )
    for name in sorted(set(current_artifacts).difference(baseline_artifacts)):
        artifact = current_artifacts[name]
        diagnostics.append(
            Diagnostic(
                rule_id="diff-artifact-added",
                severity=DiagnosticSeverity.INFO,
                message=f"current config adds {artifact.kind.value} artifact '{name}'",
                artifact=artifact.to_ref(),
                span=artifact.source_span,
                check_modes=DIFF_CHECK_MODES,
                suggestions=("Ensure release gating includes checks that exercise the new artifact.",),
                witness=WitnessTrace(
                    summary="A current artifact has no baseline counterpart.",
                    steps=(WitnessStep(action="pair artifacts by name", input=name, output="added in current"),),
                    artifacts=(artifact.to_ref(),),
                ),
                properties=(("artifact_name", name), ("current_kind", artifact.kind.value)),
            )
        )
    for name in sorted(set(baseline_artifacts).intersection(current_artifacts)):
        baseline_artifact = baseline_artifacts[name]
        current_artifact = current_artifacts[name]
        if baseline_artifact.kind is not current_artifact.kind:
            diagnostics.append(
                Diagnostic(
                    rule_id="diff-artifact-kind-changed",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"artifact '{name}' changes kind from {baseline_artifact.kind.value} "
                        f"to {current_artifact.kind.value}"
                    ),
                    artifact=current_artifact.to_ref(),
                    span=current_artifact.source_span,
                    check_modes=DIFF_CHECK_MODES,
                    suggestions=("Use a new artifact name when replacing one contract kind with another.",),
                    witness=WitnessTrace(
                        summary="PromptABI cannot compare different artifact kinds as the same contract.",
                        steps=(
                            WitnessStep(action="read baseline artifact kind", input=name, output=baseline_artifact.kind.value),
                            WitnessStep(action="read current artifact kind", input=name, output=current_artifact.kind.value),
                        ),
                        artifacts=(baseline_artifact.to_ref(), current_artifact.to_ref()),
                    ),
                    properties=(
                        ("artifact_name", name),
                        ("baseline_kind", baseline_artifact.kind.value),
                        ("current_kind", current_artifact.kind.value),
                    ),
                )
            )
    return tuple(diagnostics)


def _paired_artifact_diagnostics(baseline: ConfigDiffInputs, current: ConfigDiffInputs) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    baseline_loaded = baseline.loaded_by_name
    current_loaded = current.loaded_by_name
    for name in sorted(set(baseline_loaded).intersection(current_loaded)):
        baseline_artifact = baseline_loaded[name].artifact
        current_artifact = current_loaded[name].artifact
        if baseline_artifact.kind is not current_artifact.kind:
            continue
        if baseline_artifact.kind is ArtifactKind.TOKENIZER:
            diagnostics.extend(_tokenizer_diagnostics(name, baseline_loaded[name], current_loaded[name]))
        elif baseline_artifact.kind is ArtifactKind.PROVIDER_CONFIG:
            diagnostics.extend(_provider_diagnostics(compare_provider_config_artifacts(baseline_loaded[name], current_loaded[name])))
        elif baseline_artifact.kind is ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG:
            diagnostics.extend(_framework_diagnostics(name, baseline_artifact, current_artifact))
        else:
            diagnostic = _generic_artifact_diagnostic(name, baseline_loaded[name], current_loaded[name])
            if diagnostic is not None:
                diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _tokenizer_diagnostics(name: str, baseline: LoadedArtifact, current: LoadedArtifact) -> tuple[Diagnostic, ...]:
    baseline_path = baseline.artifact.location.path
    current_path = current.artifact.location.path
    if baseline_path is None or current_path is None:
        return (
            Diagnostic(
                rule_id="diff-tokenizer-abstained",
                severity=DiagnosticSeverity.WARNING,
                message=f"tokenizer artifact '{name}' cannot be diffed without local tokenizer files",
                artifact=current.artifact.to_ref(),
                check_modes=DIFF_ABSTAIN_MODES,
                suggestions=("Use local tokenizer snapshots or lockfiles for deterministic tokenizer diffs.",),
                witness=WitnessTrace(
                    summary="Tokenizer diffing requires loading tokenizer_config/tokenizer/generation files.",
                    steps=(
                        WitnessStep(action="inspect baseline tokenizer location", output=baseline.artifact.to_ref().location_uri or "<none>"),
                        WitnessStep(action="inspect current tokenizer location", output=current.artifact.to_ref().location_uri or "<none>"),
                    ),
                    artifacts=(baseline.artifact.to_ref(), current.artifact.to_ref()),
                ),
            ),
        )
    try:
        baseline_snapshot = load_tokenizer_config_snapshot(
            baseline_path,
            revision=baseline.artifact.provenance.revision or baseline.artifact.provenance.version,
        )
        current_snapshot = load_tokenizer_config_snapshot(
            current_path,
            revision=current.artifact.provenance.revision or current.artifact.provenance.version,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            Diagnostic(
                rule_id="diff-tokenizer-abstained",
                severity=DiagnosticSeverity.WARNING,
                message=f"tokenizer artifact '{name}' could not be snapshot-diffed: {exc}",
                artifact=current.artifact.to_ref(),
                check_modes=DIFF_ABSTAIN_MODES,
                suggestions=("Use complete tokenizer snapshots containing tokenizer_config.json or tokenizer.json.",),
                witness=WitnessTrace(
                    summary="PromptABI could load the artifact but could not build tokenizer drift snapshots.",
                    steps=(WitnessStep(action="load tokenizer snapshots", input=name, output=str(exc)),),
                    artifacts=(baseline.artifact.to_ref(), current.artifact.to_ref()),
                ),
            ),
        )
    return tuple(
        _tokenizer_finding_diagnostic(current.artifact.to_ref(), finding)
        for finding in compare_tokenizer_config_snapshots(baseline_snapshot, current_snapshot)
    )


def _tokenizer_finding_diagnostic(artifact: ArtifactRef, finding: TokenizerDriftFinding) -> Diagnostic:
    return Diagnostic(
        rule_id="diff-tokenizer-drift",
        severity=DiagnosticSeverity.ERROR if finding.breaking else DiagnosticSeverity.WARNING,
        message=f"tokenizer {finding.field} changed between baseline and current configs",
        artifact=artifact,
        check_modes=TOKENIZER_DIFF_MODES,
        suggestions=(_tokenizer_suggestion(finding),),
        witness=WitnessTrace(
            summary=f"{finding.kind.value} detected while comparing tokenizer snapshots.",
            steps=(
                WitnessStep(action="compare artifact revisions", input=finding.baseline_revision or "<unknown>", output=finding.current_revision or "<unknown>"),
                WitnessStep(action=f"read baseline {finding.field}", output=_stable_text(finding.baseline)),
                WitnessStep(action=f"read current {finding.field}", output=_stable_text(finding.current)),
            ),
            artifacts=(artifact,),
        ),
        properties=(
            ("kind", finding.kind.value),
            ("field", finding.field),
            ("baseline", finding.baseline),
            ("current", finding.current),
            ("baseline_path", finding.baseline_path),
            ("current_path", finding.current_path),
        ),
    )


def _provider_diagnostics(findings: tuple[ProviderMigrationFinding, ...]) -> tuple[Diagnostic, ...]:
    return tuple(_provider_finding_diagnostic(finding) for finding in findings)


def _provider_finding_diagnostic(finding: ProviderMigrationFinding) -> Diagnostic:
    artifact = ArtifactRef(kind=ArtifactKind.PROVIDER_CONFIG.value, name=finding.target_artifact_name or finding.target_provider)
    return Diagnostic(
        rule_id="diff-provider-contract",
        severity=DiagnosticSeverity.ERROR if finding.severity == "error" else DiagnosticSeverity.WARNING,
        message=f"provider contract changed incompatibly: {finding.kind.value}: {finding.message}",
        artifact=artifact,
        span=finding.span,
        check_modes=PROVIDER_DIFF_MODES,
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(
            summary="Recorded provider fixtures disagree on a migration-sensitive contract field.",
            steps=(
                WitnessStep(
                    action="compare provider pair",
                    input=finding.source_artifact_name,
                    output=finding.target_artifact_name or finding.target_provider,
                ),
                *(
                    WitnessStep(action="compare provider contract field", input=key, output=value)
                    for key, value in finding.evidence
                ),
            ),
            artifacts=(artifact,),
        ),
        properties=(
            ("kind", finding.kind.value),
            ("source_provider", finding.source_provider),
            ("target_provider", finding.target_provider),
            ("source_artifact", finding.source_artifact_name),
            ("target_artifact", finding.target_artifact_name),
        ),
    )


def _framework_diagnostics(name: str, baseline: Artifact, current: Artifact) -> tuple[Diagnostic, ...]:
    if not isinstance(baseline, FrameworkTruncationConfigArtifact) or not isinstance(current, FrameworkTruncationConfigArtifact):
        return ()
    diagnostics: list[Diagnostic] = []
    if baseline.framework != current.framework:
        diagnostics.append(_framework_field_diagnostic(name, current, "framework", baseline.framework, current.framework, DiagnosticSeverity.WARNING))
    if baseline.strategy != current.strategy:
        diagnostics.append(
            _framework_field_diagnostic(name, current, "strategy", baseline.strategy.value, current.strategy.value, DiagnosticSeverity.WARNING)
        )
    if _regresses_limit(baseline.max_context_tokens, current.max_context_tokens):
        diagnostics.append(
            _framework_field_diagnostic(
                name,
                current,
                "max_context_tokens",
                baseline.max_context_tokens,
                current.max_context_tokens,
                DiagnosticSeverity.ERROR,
                suggestion="Run must-survive checks against the lower framework limit or keep the baseline limit.",
            )
        )
    for field in ("preserve_system", "preserve_tools"):
        if getattr(baseline, field) is True and getattr(current, field) is False:
            diagnostics.append(
                _framework_field_diagnostic(
                    name,
                    current,
                    field,
                    True,
                    False,
                    DiagnosticSeverity.ERROR,
                    suggestion=f"Keep {field} enabled or prove the required prompt region survives another way.",
                )
            )
    return tuple(diagnostics)


def _framework_field_diagnostic(
    name: str,
    current: FrameworkTruncationConfigArtifact,
    field: str,
    baseline_value: object,
    current_value: object,
    severity: DiagnosticSeverity,
    *,
    suggestion: str = "Re-run token-budget verification before deploying this framework policy change.",
) -> Diagnostic:
    return Diagnostic(
        rule_id="diff-framework-truncation",
        severity=severity,
        message=f"framework truncation artifact '{name}' changes {field}",
        artifact=current.to_ref(),
        span=current.source_span,
        check_modes=FRAMEWORK_DIFF_MODES,
        suggestions=(suggestion,),
        witness=WitnessTrace(
            summary="A framework truncation policy field changed across the configuration diff.",
            steps=(
                WitnessStep(action=f"read baseline {field}", output=_stable_text(baseline_value)),
                WitnessStep(action=f"read current {field}", output=_stable_text(current_value)),
            ),
            artifacts=(current.to_ref(),),
        ),
        properties=(("field", field), ("baseline", baseline_value), ("current", current_value)),
    )


def _generic_artifact_diagnostic(name: str, baseline: LoadedArtifact, current: LoadedArtifact) -> Diagnostic | None:
    baseline_identity = _artifact_identity(baseline)
    current_identity = _artifact_identity(current)
    if baseline_identity == current_identity:
        return None
    return Diagnostic(
        rule_id="diff-artifact-drift",
        severity=DiagnosticSeverity.WARNING,
        message=f"artifact '{name}' changes reproducible content or provenance",
        artifact=current.artifact.to_ref(),
        span=current.artifact.source_span,
        check_modes=DIFF_CHECK_MODES,
        suggestions=("Review downstream checks for this artifact before accepting the migration.",),
        witness=WitnessTrace(
            summary="PromptABI compared artifact content hashes or provenance pins rather than machine-local paths.",
            steps=(
                WitnessStep(action="read baseline artifact identity", output=_stable_text(baseline_identity)),
                WitnessStep(action="read current artifact identity", output=_stable_text(current_identity)),
            ),
            artifacts=(baseline.artifact.to_ref(), current.artifact.to_ref()),
        ),
        properties=(("artifact_name", name), ("baseline_identity", baseline_identity), ("current_identity", current_identity)),
    )


def _clean_diagnostic(baseline: VerificationConfig, current: VerificationConfig) -> Diagnostic:
    return Diagnostic(
        rule_id="diff-clean",
        severity=DiagnosticSeverity.INFO,
        message="no contract-breaking configuration changes were detected",
        check_modes=DIFF_CHECK_MODES,
        suggestions=("Keep both configs under lockfile enforcement for reproducible CI gating.",),
        witness=WitnessTrace(
            summary="PromptABI compared config fields, artifact sets, tokenizer snapshots, provider fixtures, and framework policies.",
            steps=(
                WitnessStep(action="load baseline config", input=baseline.name, output=f"{len(baseline.artifact_bundle.artifacts)} artifacts"),
                WitnessStep(action="load current config", input=current.name, output=f"{len(current.artifact_bundle.artifacts)} artifacts"),
                WitnessStep(action="compare contract surfaces", output="no breaking differences"),
            ),
        ),
    )


def _artifact_identity(loaded: LoadedArtifact) -> tuple[tuple[str, Any], ...]:
    artifact = loaded.artifact
    values: dict[str, Any] = {
        "kind": artifact.kind.value,
        "source_type": loaded.source_type,
        "actual_sha256": loaded.actual_sha256,
        "manifest_sha256": loaded.manifest_sha256,
        "provenance_sha256": artifact.provenance.sha256,
        "revision": artifact.provenance.revision,
        "version": artifact.provenance.version,
        "uri": artifact.location.uri,
        "members": loaded.members,
    }
    return tuple(sorted((key, value) for key, value in values.items() if value not in (None, (), "")))


def _regresses_limit(baseline: int | None, current: int | None) -> bool:
    return baseline is not None and (current is None or current < baseline)


def _stable_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _tokenizer_suggestion(finding: TokenizerDriftFinding) -> str:
    if finding.breaking:
        return "Pin the baseline tokenizer/config or update templates, stops, parsers, and budget checks for the new tokenizer contract."
    return "Review tokenizer normalization and added-token changes against differential tokenizer tests before migration."
