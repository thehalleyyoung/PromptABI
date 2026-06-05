"""Training-corpus metadata drift checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .artifacts import ChatTemplateVersion, TrainingManifestArtifact, TrainingPipelineStageVersion


class TrainingDriftFindingKind(StrEnum):
    """Bounded drift outcomes for training-corpus metadata."""

    METADATA_MISSING = "metadata-missing"
    MODEL_CARD_MISMATCH = "model-card-mismatch"
    TOKENIZER_MISMATCH = "tokenizer-mismatch"
    TEMPLATE_MISMATCH = "template-mismatch"
    SERVING_LOCKFILE_MISMATCH = "serving-lockfile-mismatch"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class TrainingDriftFinding:
    """One offline training-corpus drift finding."""

    kind: TrainingDriftFindingKind
    manifest_name: str
    message: str
    severity: str
    subject: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingDriftReport:
    """Bounded drift report for a training manifest."""

    manifest_name: str
    findings: tuple[TrainingDriftFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(finding.kind is TrainingDriftFindingKind.VERIFIED for finding in self.findings)


_BASELINE_METADATA_KEYS = ("training_corpus_contract", "training_corpus_metadata", "corpus_metadata")
_MODEL_CARD_METADATA_KEYS = ("model_card", "model_card_metadata")
_LOCKFILE_METADATA_KEYS = ("serving_lockfile", "serving_lockfile_metadata", "lockfile")
_TOKENIZER_FIELDS = ("name", "version", "revision", "sha256")
_TEMPLATE_FIELDS = ("name", "version", "revision", "sha256", "add_generation_prompt")
_LOCKFILE_FIELDS = ("name", "version", "revision", "sha256", "config_sha256")
_TRAINING_STAGE_NAMES = ("dataset-preparation", "data-preparation", "training", "fine-tuning", "sft", "dpo")
_SERVING_STAGE_NAMES = ("serving", "inference", "deployment", "production")


def analyze_training_metadata_drift(manifest: TrainingManifestArtifact) -> TrainingDriftReport:
    """Compare training-corpus metadata against model-card and serving pins.

    This checker is intentionally local and finite. It compares only metadata
    declared in the training manifest: corpus baseline facts, model-card facts,
    pipeline stage tokenizer/template pins, chat-template pins, and serving
    lockfile summaries. It never opens private dataset rows or calls providers.
    """

    baseline = _baseline_metadata(manifest)
    if not baseline:
        return TrainingDriftReport(
            manifest_name=manifest.name,
            findings=(
                _finding(
                    manifest,
                    TrainingDriftFindingKind.METADATA_MISSING,
                    f"training manifest '{manifest.name}' has no training_corpus_contract metadata for drift checks",
                    "info",
                    subject="metadata.training_corpus_contract",
                    witness=(
                        ("select training manifest", manifest.name, _dataset_summary(manifest)),
                        ("inspect corpus baseline metadata", None, "missing"),
                    ),
                ),
            ),
        )

    findings: list[TrainingDriftFinding] = []
    findings.extend(_model_card_findings(manifest, baseline))
    findings.extend(_tokenizer_findings(manifest, baseline))
    findings.extend(_template_findings(manifest, baseline))
    findings.extend(_serving_lockfile_findings(manifest, baseline))

    if not findings:
        findings.append(
            _finding(
                manifest,
                TrainingDriftFindingKind.VERIFIED,
                f"training manifest '{manifest.name}' corpus metadata matches model-card, tokenizer, template, and serving lockfile pins",
                "info",
                subject="metadata.training_corpus_contract",
                witness=(
                    ("select corpus baseline", None, _baseline_summary(baseline)),
                    ("compare model-card metadata", None, "match"),
                    ("compare tokenizer pins", None, "match"),
                    ("compare chat-template pins", None, "match"),
                    ("compare serving lockfile metadata", None, "match"),
                ),
            )
        )
    return TrainingDriftReport(manifest_name=manifest.name, findings=tuple(findings))


def _model_card_findings(
    manifest: TrainingManifestArtifact,
    baseline: Mapping[str, object],
) -> tuple[TrainingDriftFinding, ...]:
    expected = _mapping(baseline.get("model_card"))
    actual = _first_metadata_mapping(manifest, _MODEL_CARD_METADATA_KEYS)
    return _mapping_findings(
        manifest,
        expected=expected,
        actual=actual,
        kind=TrainingDriftFindingKind.MODEL_CARD_MISMATCH,
        subject="metadata.model_card",
        message="training corpus model-card metadata differs from declared model-card facts",
    )


def _tokenizer_findings(
    manifest: TrainingManifestArtifact,
    baseline: Mapping[str, object],
) -> tuple[TrainingDriftFinding, ...]:
    expected = _pin_mapping(_mapping(baseline.get("tokenizer")), _TOKENIZER_FIELDS)
    findings: list[TrainingDriftFinding] = []
    for stage in _matching_stages(manifest.pipeline_stages, (*_TRAINING_STAGE_NAMES, *_SERVING_STAGE_NAMES)):
        actual = _pin_mapping(_stage_tokenizer_mapping(stage), _TOKENIZER_FIELDS)
        findings.extend(
            _mapping_findings(
                manifest,
                expected=expected,
                actual=actual,
                kind=TrainingDriftFindingKind.TOKENIZER_MISMATCH,
                subject=f"pipeline_stages.{stage.stage}.tokenizer",
                message=f"training corpus tokenizer metadata differs from {stage.stage} tokenizer pin",
            )
        )
    lockfile = _first_metadata_mapping(manifest, _LOCKFILE_METADATA_KEYS)
    findings.extend(
        _mapping_findings(
            manifest,
            expected=expected,
            actual=_pin_mapping(_mapping(lockfile.get("tokenizer")), _TOKENIZER_FIELDS),
            kind=TrainingDriftFindingKind.TOKENIZER_MISMATCH,
            subject="metadata.serving_lockfile.tokenizer",
            message="training corpus tokenizer metadata differs from serving lockfile tokenizer pin",
        )
    )
    return tuple(findings)


def _template_findings(
    manifest: TrainingManifestArtifact,
    baseline: Mapping[str, object],
) -> tuple[TrainingDriftFinding, ...]:
    expected = _pin_mapping(_mapping(baseline.get("chat_template")), _TEMPLATE_FIELDS)
    findings: list[TrainingDriftFinding] = []
    if manifest.chat_template_version is not None:
        findings.extend(
            _mapping_findings(
                manifest,
                expected=expected,
                actual=_pin_mapping(_chat_template_version_mapping(manifest.chat_template_version), _TEMPLATE_FIELDS),
                kind=TrainingDriftFindingKind.TEMPLATE_MISMATCH,
                subject="chat_template_version",
                message="training corpus chat-template metadata differs from manifest chat_template_version",
                require_all_expected_fields=False,
            )
        )
    for stage in _matching_stages(manifest.pipeline_stages, (*_TRAINING_STAGE_NAMES, *_SERVING_STAGE_NAMES)):
        findings.extend(
            _mapping_findings(
                manifest,
                expected=expected,
                actual=_pin_mapping(_stage_template_mapping(stage), _TEMPLATE_FIELDS),
                kind=TrainingDriftFindingKind.TEMPLATE_MISMATCH,
                subject=f"pipeline_stages.{stage.stage}.chat_template",
                message=f"training corpus chat-template metadata differs from {stage.stage} chat-template pin",
            )
        )
    lockfile = _first_metadata_mapping(manifest, _LOCKFILE_METADATA_KEYS)
    findings.extend(
        _mapping_findings(
            manifest,
            expected=expected,
            actual=_pin_mapping(_mapping(lockfile.get("chat_template")), _TEMPLATE_FIELDS),
            kind=TrainingDriftFindingKind.TEMPLATE_MISMATCH,
            subject="metadata.serving_lockfile.chat_template",
            message="training corpus chat-template metadata differs from serving lockfile chat-template pin",
        )
    )
    return tuple(findings)


def _serving_lockfile_findings(
    manifest: TrainingManifestArtifact,
    baseline: Mapping[str, object],
) -> tuple[TrainingDriftFinding, ...]:
    expected = _pin_mapping(_mapping(baseline.get("serving_lockfile")), _LOCKFILE_FIELDS)
    actual = _pin_mapping(_first_metadata_mapping(manifest, _LOCKFILE_METADATA_KEYS), _LOCKFILE_FIELDS)
    return _mapping_findings(
        manifest,
        expected=expected,
        actual=actual,
        kind=TrainingDriftFindingKind.SERVING_LOCKFILE_MISMATCH,
        subject="metadata.serving_lockfile",
        message="training corpus serving-lockfile metadata differs from serving lockfile facts",
    )


def _mapping_findings(
    manifest: TrainingManifestArtifact,
    *,
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    kind: TrainingDriftFindingKind,
    subject: str,
    message: str,
    require_all_expected_fields: bool = True,
) -> tuple[TrainingDriftFinding, ...]:
    if not expected:
        return ()
    findings: list[TrainingDriftFinding] = []
    for field in sorted(expected):
        expected_value = expected[field]
        actual_value = actual.get(field)
        if actual_value is None and not require_all_expected_fields:
            continue
        if expected_value == actual_value:
            continue
        findings.append(
            _finding(
                manifest,
                kind,
                f"{message}: {field}",
                "error",
                subject=f"{subject}.{field}",
                witness=(
                    ("select corpus baseline field", field, expected_value),
                    ("select comparison field", subject, _display(actual_value)),
                    ("compare drift field", None, "mismatch"),
                ),
            )
        )
    return tuple(findings)


def _baseline_metadata(manifest: TrainingManifestArtifact) -> Mapping[str, object]:
    return _first_metadata_mapping(manifest, _BASELINE_METADATA_KEYS)


def _first_metadata_mapping(manifest: TrainingManifestArtifact, keys: tuple[str, ...]) -> Mapping[str, object]:
    metadata = dict(manifest.metadata)
    for key in keys:
        value = _mapping(metadata.get(key))
        if value:
            return value
    return {}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _pin_mapping(value: Mapping[str, object], fields: tuple[str, ...]) -> Mapping[str, str]:
    return {field: _string_value(value[field]) for field in fields if field in value and _string_value(value[field]) is not None}


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return None


def _matching_stages(
    stages: tuple[TrainingPipelineStageVersion, ...],
    names: tuple[str, ...],
) -> tuple[TrainingPipelineStageVersion, ...]:
    normalized = {_normalize_name(name) for name in names}
    return tuple(stage for stage in stages if _normalize_name(stage.stage) in normalized)


def _stage_tokenizer_mapping(stage: TrainingPipelineStageVersion) -> Mapping[str, object]:
    return {
        "name": stage.tokenizer_name,
        "version": stage.tokenizer_version,
        "revision": stage.tokenizer_revision,
        "sha256": stage.tokenizer_sha256,
    }


def _stage_template_mapping(stage: TrainingPipelineStageVersion) -> Mapping[str, object]:
    return {
        "name": stage.chat_template_name,
        "version": stage.chat_template_version,
        "revision": stage.chat_template_revision,
        "sha256": stage.chat_template_sha256,
        "add_generation_prompt": stage.add_generation_prompt,
    }


def _chat_template_version_mapping(version: ChatTemplateVersion) -> Mapping[str, object]:
    return {
        "name": version.name,
        "version": version.version,
        "revision": version.revision,
        "sha256": version.sha256,
        "add_generation_prompt": version.add_generation_prompt,
    }


def _normalize_name(value: str) -> str:
    return value.lower().replace("_", "-")


def _dataset_summary(manifest: TrainingManifestArtifact) -> str:
    if not manifest.datasets:
        return "0 dataset declarations"
    return ", ".join(f"{dataset.name}:{dataset.kind.value}" for dataset in manifest.datasets)


def _baseline_summary(baseline: Mapping[str, object]) -> str:
    sections = sorted(key for key in baseline if key in {"model_card", "tokenizer", "chat_template", "serving_lockfile"})
    return ", ".join(sections) or "declared"


def _display(value: str | None) -> str:
    return value if value is not None else "<missing>"


def _finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingDriftFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingDriftFinding:
    return TrainingDriftFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        subject=subject,
        witness=witness,
    )
