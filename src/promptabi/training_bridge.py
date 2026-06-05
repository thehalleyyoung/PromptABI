"""Training-to-serving bridge checks for prompt-interface contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .artifacts import TrainingManifestArtifact, TrainingPipelineStageVersion


class TrainingBridgeFindingKind(StrEnum):
    """Finite bridge invariants between training-time and serving-time contracts."""

    STAGE_MISSING = "stage-missing"
    TOKENIZER_MISMATCH = "tokenizer-mismatch"
    TEMPLATE_MISMATCH = "template-mismatch"
    ROLE_DELIMITER_MISMATCH = "role-delimiter-mismatch"
    SPECIAL_TOKEN_MISMATCH = "special-token-mismatch"
    TOOL_FORMAT_MISMATCH = "tool-format-mismatch"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class TrainingBridgeFinding:
    """One bounded training-to-serving bridge finding."""

    kind: TrainingBridgeFindingKind
    manifest_name: str
    message: str
    severity: str
    subject: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingBridgeReport:
    """Bounded bridge report for a single training manifest."""

    manifest_name: str
    findings: tuple[TrainingBridgeFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is TrainingBridgeFindingKind.VERIFIED for finding in self.findings
        )


_TRAINING_STAGE_NAMES = ("training", "fine-tuning", "sft", "dpo")
_SERVING_STAGE_NAMES = ("serving", "inference", "deployment", "production")


def analyze_training_inference_bridge(manifest: TrainingManifestArtifact) -> TrainingBridgeReport:
    """Compare finite training-time interface facts against serving assumptions.

    The check is intentionally structural and offline. It compares manifest stage
    pins plus optional ``metadata.bridge_contract`` facts for role delimiters,
    special tokens, and tool-call formats. It does not inspect private dataset
    rows or call a serving endpoint.
    """

    findings: list[TrainingBridgeFinding] = []
    training_stage = _select_stage(manifest.pipeline_stages, _TRAINING_STAGE_NAMES)
    serving_stage = _select_stage(manifest.pipeline_stages, _SERVING_STAGE_NAMES)
    if training_stage is None:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.STAGE_MISSING,
                "training manifest has no training-stage tokenizer/template pin",
                "error",
                subject="pipeline_stages.training",
                witness=(("select training stage", None, "missing"),),
            )
        )
    if serving_stage is None:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.STAGE_MISSING,
                "training manifest has no serving-stage tokenizer/template pin",
                "error",
                subject="pipeline_stages.serving",
                witness=(("select serving stage", None, "missing"),),
            )
        )
    if training_stage is not None and serving_stage is not None:
        findings.extend(_stage_pin_findings(manifest, training_stage, serving_stage))

    bridge_contract = _bridge_contract(manifest)
    if bridge_contract:
        findings.extend(_bridge_contract_findings(manifest, bridge_contract))

    if not findings and training_stage is not None and serving_stage is not None:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.VERIFIED,
                f"training manifest '{manifest.name}' matches serving prompt-interface assumptions",
                "info",
                subject="training_serving_bridge",
                witness=(
                    ("select training stage", training_stage.stage, _stage_summary(training_stage)),
                    ("select serving stage", serving_stage.stage, _stage_summary(serving_stage)),
                    ("compare tokenizer pins", None, "match"),
                    ("compare chat-template pins", None, "match"),
                    ("compare bridge contract facts", None, _bridge_contract_summary(bridge_contract)),
                ),
            )
        )
    return TrainingBridgeReport(manifest_name=manifest.name, findings=tuple(findings))


def _stage_pin_findings(
    manifest: TrainingManifestArtifact,
    training_stage: TrainingPipelineStageVersion,
    serving_stage: TrainingPipelineStageVersion,
) -> tuple[TrainingBridgeFinding, ...]:
    findings: list[TrainingBridgeFinding] = []
    tokenizer_training = _tokenizer_pin(training_stage)
    tokenizer_serving = _tokenizer_pin(serving_stage)
    if tokenizer_training != tokenizer_serving:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.TOKENIZER_MISMATCH,
                "training tokenizer pin differs from serving tokenizer pin",
                "error",
                subject="pipeline_stages.tokenizer",
                witness=(
                    ("select training tokenizer pin", training_stage.stage, _pin_summary(tokenizer_training)),
                    ("select serving tokenizer pin", serving_stage.stage, _pin_summary(tokenizer_serving)),
                    ("compare tokenizer pins", None, "mismatch"),
                ),
            )
        )
    template_training = _template_pin(training_stage)
    template_serving = _template_pin(serving_stage)
    if template_training != template_serving:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.TEMPLATE_MISMATCH,
                "training chat-template pin differs from serving chat-template pin",
                "error",
                subject="pipeline_stages.chat_template",
                witness=(
                    ("select training template pin", training_stage.stage, _pin_summary(template_training)),
                    ("select serving template pin", serving_stage.stage, _pin_summary(template_serving)),
                    ("compare chat-template pins", None, "mismatch"),
                ),
            )
        )
    if training_stage.add_generation_prompt != serving_stage.add_generation_prompt:
        findings.append(
            _finding(
                manifest,
                TrainingBridgeFindingKind.TEMPLATE_MISMATCH,
                "training add_generation_prompt setting differs from serving",
                "error",
                subject="pipeline_stages.add_generation_prompt",
                witness=(
                    ("select training generation-prompt flag", training_stage.stage, str(training_stage.add_generation_prompt)),
                    ("select serving generation-prompt flag", serving_stage.stage, str(serving_stage.add_generation_prompt)),
                    ("compare generation-prompt behavior", None, "mismatch"),
                ),
            )
        )
    return tuple(findings)


def _bridge_contract_findings(
    manifest: TrainingManifestArtifact,
    bridge_contract: Mapping[str, object],
) -> tuple[TrainingBridgeFinding, ...]:
    training = _mapping(bridge_contract.get("training"))
    serving = _mapping(bridge_contract.get("serving"))
    if not training or not serving:
        return ()
    findings: list[TrainingBridgeFinding] = []
    comparisons = (
        (
            "role_delimiters",
            TrainingBridgeFindingKind.ROLE_DELIMITER_MISMATCH,
            "trained role delimiter differs from serving role delimiter",
        ),
        (
            "special_tokens",
            TrainingBridgeFindingKind.SPECIAL_TOKEN_MISMATCH,
            "trained special token differs from serving special token",
        ),
        (
            "tool_format",
            TrainingBridgeFindingKind.TOOL_FORMAT_MISMATCH,
            "trained tool-call format differs from serving tool-call format",
        ),
    )
    for field, kind, message in comparisons:
        findings.extend(
            _fact_map_findings(
                manifest,
                kind=kind,
                message=message,
                subject=f"metadata.bridge_contract.{field}",
                training_facts=_string_mapping(training.get(field)),
                serving_facts=_string_mapping(serving.get(field)),
            )
        )
    return tuple(findings)


def _fact_map_findings(
    manifest: TrainingManifestArtifact,
    *,
    kind: TrainingBridgeFindingKind,
    message: str,
    subject: str,
    training_facts: Mapping[str, str],
    serving_facts: Mapping[str, str],
) -> tuple[TrainingBridgeFinding, ...]:
    findings: list[TrainingBridgeFinding] = []
    for key in sorted(set(training_facts).union(serving_facts)):
        training_value = training_facts.get(key)
        serving_value = serving_facts.get(key)
        if training_value == serving_value:
            continue
        findings.append(
            _finding(
                manifest,
                kind,
                f"{message}: {key}",
                "error",
                subject=f"{subject}.{key}",
                witness=(
                    ("select trained bridge fact", key, _display_fact(training_value)),
                    ("select serving bridge fact", key, _display_fact(serving_value)),
                    ("compare bridge fact", subject, "mismatch"),
                ),
            )
        )
    return tuple(findings)


def _select_stage(
    stages: tuple[TrainingPipelineStageVersion, ...],
    names: tuple[str, ...],
) -> TrainingPipelineStageVersion | None:
    normalized_names = {_normalize_name(name) for name in names}
    for stage in stages:
        if _normalize_name(stage.stage) in normalized_names:
            return stage
    return None


def _tokenizer_pin(stage: TrainingPipelineStageVersion) -> tuple[str | None, str | None, str | None, str | None]:
    return (stage.tokenizer_name, stage.tokenizer_version, stage.tokenizer_revision, stage.tokenizer_sha256)


def _template_pin(stage: TrainingPipelineStageVersion) -> tuple[str | None, str | None, str | None, str | None]:
    return (
        stage.chat_template_name,
        stage.chat_template_version,
        stage.chat_template_revision,
        stage.chat_template_sha256,
    )


def _bridge_contract(manifest: TrainingManifestArtifact) -> Mapping[str, object]:
    metadata = dict(manifest.metadata)
    return _mapping(metadata.get("bridge_contract"))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(key, str)}


def _string_mapping(value: object) -> Mapping[str, str]:
    raw = _mapping(value)
    return {key: str(item) for key, item in raw.items() if isinstance(item, (str, int, bool))}


def _normalize_name(value: str) -> str:
    return value.lower().replace("_", "-")


def _pin_summary(pin: tuple[str | None, str | None, str | None, str | None]) -> str:
    labels = ("name", "version", "revision", "sha256")
    parts = [f"{label}={value}" for label, value in zip(labels, pin, strict=True) if value is not None]
    return ", ".join(parts) or "<unpinned>"


def _stage_summary(stage: TrainingPipelineStageVersion) -> str:
    return f"{_pin_summary(_tokenizer_pin(stage))}; {_pin_summary(_template_pin(stage))}; add_generation_prompt={stage.add_generation_prompt}"


def _bridge_contract_summary(bridge_contract: Mapping[str, object]) -> str:
    if not bridge_contract:
        return "stage pins only"
    training = _mapping(bridge_contract.get("training"))
    serving = _mapping(bridge_contract.get("serving"))
    fields = sorted(set(training).intersection(serving))
    return ", ".join(fields) or "declared"


def _display_fact(value: str | None) -> str:
    return value if value is not None else "<missing>"


def _finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingBridgeFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingBridgeFinding:
    return TrainingBridgeFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        subject=subject,
        witness=witness,
    )
