"""Dedicated training-manifest verification workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateArtifact,
    TokenizerArtifact,
    TrainingManifestArtifact,
)
from .config import VerificationConfig
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .loaders import LoadedArtifact
from .plugins import PluginRegistry
from .session import CheckContext, VerificationResult, VerificationSession

TRAINING_WORKFLOW_CHECKS = (
    "training-workflow",
    "training-packing",
    "training-redaction",
    "training-invalid-interface",
    "training-bridge",
    "training-drift",
)


def build_training_config(
    manifest_path: str | Path,
    *,
    name: str | None = None,
    tokenizers: Iterable[tuple[str, str | Path]] = (),
    chat_templates: Iterable[tuple[str, str | Path]] = (),
    checks: tuple[str, ...] = TRAINING_WORKFLOW_CHECKS,
) -> VerificationConfig:
    """Create a verification config centered on a local training manifest."""

    manifest = TrainingManifestArtifact(
        kind=ArtifactKind.TRAINING_MANIFEST,
        name="training-manifest",
        location=ArtifactLocation(path=str(Path(manifest_path).expanduser().resolve())),
    )
    artifacts = [manifest]
    for tokenizer_name, tokenizer_path in tokenizers:
        artifacts.append(
            TokenizerArtifact(
                kind=ArtifactKind.TOKENIZER,
                name=tokenizer_name,
                location=ArtifactLocation(path=str(Path(tokenizer_path).expanduser().resolve())),
            )
        )
    for template_name, template_path in chat_templates:
        artifacts.append(
            ChatTemplateArtifact(
                kind=ArtifactKind.CHAT_TEMPLATE,
                name=template_name,
                location=ArtifactLocation(path=str(Path(template_path).expanduser().resolve())),
            )
        )
    bundle = ArtifactBundle(tuple(artifacts))
    return VerificationConfig(
        name=name or f"training-{Path(manifest_path).stem}",
        artifacts={
            artifact.name: artifact.location.path
            for artifact in bundle
            if artifact.location.path is not None
        },
        artifact_bundle=bundle,
        checks=checks,
    )


def run_training_verification(
    config: VerificationConfig,
    *,
    plugin_registry: PluginRegistry | None = None,
    checks: tuple[str, ...] = TRAINING_WORKFLOW_CHECKS,
) -> VerificationResult:
    """Run the training workflow with workflow coverage plus real training checks."""

    selected = tuple(dict.fromkeys(checks))
    config = replace(config, checks=selected)
    session = VerificationSession(
        config,
        checks={"training-workflow": training_workflow_check},
        plugin_registry=plugin_registry,
    )
    session.check_modes["training-workflow"] = (CheckMode.SOUND, CheckMode.BOUNDED)
    return session.run(checks=selected)


def training_workflow_check(context: CheckContext) -> tuple[Diagnostic, ...]:
    """Check that the training workflow has enough finite facts to verify."""

    manifests = tuple(
        loaded for loaded in context.loaded_artifacts if isinstance(loaded.artifact, TrainingManifestArtifact)
    )
    tokenizers = tuple(loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.TOKENIZER)
    templates = tuple(loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.CHAT_TEMPLATE)
    if not manifests:
        return (
            Diagnostic(
                rule_id="training-workflow-manifest-missing",
                severity=DiagnosticSeverity.ERROR,
                message="verify-training requires at least one training-manifest artifact",
                check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
                suggestions=("Pass --manifest or include a training-manifest artifact in the config.",),
                witness=WitnessTrace(
                    summary="The dedicated training workflow could not select a manifest.",
                    steps=(WitnessStep(action="select training manifests", output="0"),),
                ),
            ),
        )

    diagnostics: list[Diagnostic] = []
    for loaded in manifests:
        manifest = loaded.artifact
        assert isinstance(manifest, TrainingManifestArtifact)
        diagnostics.extend(_manifest_coverage_diagnostics(loaded, tokenizers=tokenizers, templates=templates))
    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                rule_id="training-workflow-verified",
                severity=DiagnosticSeverity.INFO,
                message=(
                    f"verify-training selected {len(manifests)} manifest(s), {len(tokenizers)} tokenizer(s), "
                    f"and {len(templates)} chat template(s) for finite training-contract checks"
                ),
                check_modes=(CheckMode.SOUND, CheckMode.BOUNDED),
                properties=(
                    ("manifest_count", len(manifests)),
                    ("tokenizer_count", len(tokenizers)),
                    ("chat_template_count", len(templates)),
                ),
                witness=WitnessTrace(
                    summary="PromptABI assembled the dedicated training verification workflow.",
                    steps=(
                        WitnessStep(action="select training manifests", output=str(len(manifests))),
                        WitnessStep(action="select dataset declarations", output=_dataset_summary(manifests)),
                        WitnessStep(action="select tokenizer artifacts", output=_artifact_names(tokenizers)),
                        WitnessStep(action="select chat-template artifacts", output=_artifact_names(templates)),
                        WitnessStep(action="enable training checks", output=", ".join(TRAINING_WORKFLOW_CHECKS[1:])),
                    ),
                    artifacts=tuple(loaded.artifact.to_ref() for loaded in manifests),
                ),
            )
        )
    return tuple(diagnostics)


def _manifest_coverage_diagnostics(
    loaded: LoadedArtifact,
    *,
    tokenizers: tuple[LoadedArtifact, ...],
    templates: tuple[LoadedArtifact, ...],
) -> tuple[Diagnostic, ...]:
    manifest = loaded.artifact
    assert isinstance(manifest, TrainingManifestArtifact)
    diagnostics: list[Diagnostic] = []
    supervised = any(dataset.kind.value == "supervised" for dataset in manifest.datasets) or bool(manifest.supervised_spans)
    if not manifest.datasets:
        diagnostics.append(
            _coverage_diagnostic(
                loaded,
                "training-workflow-datasets-missing",
                DiagnosticSeverity.ERROR,
                "training manifest declares no supervised or preference datasets",
                "datasets",
                "0",
                "Declare finite dataset metadata so verify-training can connect examples, roles, packing, and masks.",
            )
        )
    if supervised and manifest.loss_mask_policy is None:
        diagnostics.append(
            _coverage_diagnostic(
                loaded,
                "training-workflow-loss-mask-missing",
                DiagnosticSeverity.ERROR,
                "supervised training manifest has no loss_mask_policy",
                "loss_mask_policy",
                "missing",
                "Declare assistant-only, completion-only, all-tokens, or explicit loss-mask policy before fine-tuning.",
            )
        )
    if supervised and manifest.packing_window is None:
        diagnostics.append(
            _coverage_diagnostic(
                loaded,
                "training-workflow-packing-missing",
                DiagnosticSeverity.WARNING,
                "supervised training manifest has no packing_window",
                "packing_window",
                "missing",
                "Declare the sequence-packing window even when packing is disabled so truncation bounds are explicit.",
            )
        )
    if manifest.chat_template_version is None:
        diagnostics.append(
            _coverage_diagnostic(
                loaded,
                "training-workflow-template-unpinned",
                DiagnosticSeverity.WARNING,
                "training manifest has no chat_template_version pin",
                "chat_template_version",
                "missing",
                "Pin the chat template used to render training data and serving prompts.",
            )
        )
    if not manifest.pipeline_stages:
        diagnostics.append(
            _coverage_diagnostic(
                loaded,
                "training-workflow-pipeline-unpinned",
                DiagnosticSeverity.WARNING,
                "training manifest has no tokenizer/template pipeline stage pins",
                "pipeline_stages",
                "0",
                "Record dataset-preparation, training, evaluation, and serving tokenizer/template pins.",
            )
        )
    diagnostics.extend(_configured_artifact_alignment(loaded, tokenizers=tokenizers, templates=templates))
    return tuple(diagnostics)


def _configured_artifact_alignment(
    loaded: LoadedArtifact,
    *,
    tokenizers: tuple[LoadedArtifact, ...],
    templates: tuple[LoadedArtifact, ...],
) -> tuple[Diagnostic, ...]:
    manifest = loaded.artifact
    assert isinstance(manifest, TrainingManifestArtifact)
    stage_tokenizers = {stage.tokenizer_name for stage in manifest.pipeline_stages if stage.tokenizer_name is not None}
    if manifest.chat_template_version is not None and manifest.chat_template_version.tokenizer_name is not None:
        stage_tokenizers.add(manifest.chat_template_version.tokenizer_name)
    stage_templates = {stage.chat_template_name for stage in manifest.pipeline_stages if stage.chat_template_name is not None}
    if manifest.chat_template_version is not None:
        stage_templates.add(manifest.chat_template_version.name)

    diagnostics: list[Diagnostic] = []
    for tokenizer in tokenizers:
        if stage_tokenizers and tokenizer.artifact.name not in stage_tokenizers:
            diagnostics.append(
                _alignment_diagnostic(
                    loaded,
                    "training-workflow-tokenizer-mismatch",
                    "configured tokenizer artifact is not referenced by the training manifest",
                    tokenizer.artifact.name,
                    sorted(stage_tokenizers),
                    "Use artifact names that match manifest tokenizer_name pins, or update the manifest pins.",
                )
            )
    for template in templates:
        if stage_templates and template.artifact.name not in stage_templates:
            diagnostics.append(
                _alignment_diagnostic(
                    loaded,
                    "training-workflow-template-mismatch",
                    "configured chat-template artifact is not referenced by the training manifest",
                    template.artifact.name,
                    sorted(stage_templates),
                    "Use artifact names that match manifest chat_template_name pins, or update the manifest pins.",
                )
            )
    return tuple(diagnostics)


def _coverage_diagnostic(
    loaded: LoadedArtifact,
    rule_id: str,
    severity: DiagnosticSeverity,
    message: str,
    field: str,
    observed: str,
    suggestion: str,
) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=loaded.artifact.to_ref(),
        span=loaded.artifact.source_span,
        check_modes=(CheckMode.SOUND, CheckMode.BOUNDED),
        suggestions=(suggestion,),
        properties=(("field", field),),
        witness=WitnessTrace(
            summary="verify-training inspected the manifest coverage needed for finite training checks.",
            steps=(
                WitnessStep(action="select training manifest", input=loaded.artifact.name),
                WitnessStep(action="inspect manifest field", input=field, output=observed),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _alignment_diagnostic(
    loaded: LoadedArtifact,
    rule_id: str,
    message: str,
    artifact_name: str,
    manifest_names: list[str],
    suggestion: str,
) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=DiagnosticSeverity.WARNING,
        message=f"{message}: {artifact_name}",
        artifact=loaded.artifact.to_ref(),
        span=loaded.artifact.source_span,
        check_modes=(CheckMode.SOUND, CheckMode.BOUNDED),
        suggestions=(suggestion,),
        properties=(("artifact_name", artifact_name), ("manifest_names", manifest_names)),
        witness=WitnessTrace(
            summary="verify-training compared configured artifacts against manifest stage pins.",
            steps=(
                WitnessStep(action="select configured artifact", input=artifact_name),
                WitnessStep(action="compare manifest names", output=", ".join(manifest_names) or "<none>"),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _artifact_names(loaded_artifacts: tuple[LoadedArtifact, ...]) -> str:
    return ", ".join(loaded.artifact.name for loaded in loaded_artifacts) or "<none>"


def _dataset_summary(manifests: tuple[LoadedArtifact, ...]) -> str:
    count = 0
    for loaded in manifests:
        manifest = loaded.artifact
        if isinstance(manifest, TrainingManifestArtifact):
            count += len(manifest.datasets)
    return str(count)
