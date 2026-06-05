import json
from hashlib import sha256
from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ChatTemplateVersion,
    LossMaskStrategy,
    PackingStrategy,
    TrainingDatasetKind,
    TrainingManifestArtifact,
    TrainingPipelineStageVersion,
    TrainingSourceContribution,
    TrainingTextSourceKind,
)
from promptabi.artifacts import artifact_from_config
from promptabi.loaders import ArtifactLoadError, ArtifactLoader


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_format": "chat-jsonl",
                "datasets": [
                    {
                        "name": "sft-train",
                        "kind": "supervised",
                        "path": "train.jsonl",
                        "split": "train",
                        "format": "chat-jsonl",
                        "example_count": 128,
                        "content_fields": ["messages"],
                    },
                    {
                        "name": "prefs",
                        "kind": "preference",
                        "path": "prefs.jsonl",
                        "format": "preference-jsonl",
                        "example_count": 64,
                        "content_fields": ["prompt"],
                        "preference_fields": ["chosen", "rejected"],
                    },
                ],
                "system_message_policy": {
                    "required": True,
                    "allow_override": False,
                    "default": "Follow the policy.",
                    "allowed_hashes": ["sha256:policy"],
                    "max_tokens": 96,
                },
                "role_labels": [
                    {
                        "source_role": "assistant",
                        "canonical_role": "assistant",
                        "supervised_target": True,
                        "trainable": True,
                    },
                    {
                        "source_role": "human",
                        "canonical_role": "user",
                        "trainable": False,
                    },
                    {
                        "source_role": "system",
                        "canonical_role": "system",
                        "required": True,
                        "trainable": False,
                    },
                ],
                "loss_mask_policy": {
                    "strategy": "assistant-only",
                    "target_roles": ["assistant"],
                    "ignored_roles": ["system", "user"],
                    "label_pad_token_id": -100,
                },
                "supervised_spans": [
                    {
                        "span_id": "train-0001.assistant-0",
                        "target_role": "assistant",
                        "rendered_region_role": "assistant",
                        "start_token": 18,
                        "end_token": 36,
                        "region_start_token": 16,
                        "region_end_token": 38,
                        "supervised_target": True,
                        "loss_masked": True,
                        "packed_example_id": "train-0001",
                        "source_contributions": [
                            {
                                "source_id": "assistant-answer",
                                "source_kind": "assistant",
                                "source_field": "messages.content",
                                "start_token": 18,
                                "end_token": 36,
                                "transform": "chat-template-render",
                                "text_sha256": "sha256:assistant-answer",
                            }
                        ],
                    }
                ],
                "packing_window": {
                    "strategy": "sample-packing",
                    "max_tokens": 4096,
                    "stride_tokens": 128,
                    "boundary_token": "<|endoftext|>",
                    "preserve_example_boundaries": True,
                    "reset_position_ids": True,
                },
                "chat_template_version": {
                    "name": "llama-3.1-instruct",
                    "version": "2026-06-01",
                    "revision": "abc123",
                    "tokenizer_name": "llama-tokenizer",
                    "add_generation_prompt": False,
                },
                "pipeline_stages": [
                    {
                        "stage": "dataset-preparation",
                        "tokenizer_name": "llama-tokenizer",
                        "tokenizer_sha256": "sha256:tok",
                        "chat_template_name": "llama-3.1-instruct",
                        "chat_template_sha256": "sha256:tmpl",
                        "add_generation_prompt": False,
                    },
                    {
                        "stage": "training",
                        "tokenizer_name": "llama-tokenizer",
                        "tokenizer_sha256": "sha256:tok",
                        "chat_template_name": "llama-3.1-instruct",
                        "chat_template_sha256": "sha256:tmpl",
                        "add_generation_prompt": False,
                    },
                    {
                        "stage": "evaluation",
                        "tokenizer_name": "llama-tokenizer",
                        "tokenizer_sha256": "sha256:tok",
                        "chat_template_name": "llama-3.1-instruct",
                        "chat_template_sha256": "sha256:tmpl",
                        "add_generation_prompt": False,
                    },
                    {
                        "stage": "serving",
                        "tokenizer_name": "llama-tokenizer",
                        "tokenizer_sha256": "sha256:tok",
                        "chat_template_name": "llama-3.1-instruct",
                        "chat_template_sha256": "sha256:tmpl",
                        "add_generation_prompt": False,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_training_manifest_config_model_covers_training_pipeline_contracts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    artifact = artifact_from_config(
        "train",
        {
            "kind": "training-manifest",
            "path": manifest_path.name,
            "dataset_format": "chat-jsonl",
            "datasets": [
                {
                    "name": "prefs",
                    "kind": "preference",
                    "format": "preference-jsonl",
                    "preference_fields": ["chosen", "rejected"],
                }
            ],
            "role_labels": [
                {"source_role": "assistant", "canonical_role": "assistant", "supervised_target": True},
                {"source_role": "human", "canonical_role": "user", "trainable": False},
            ],
            "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["assistant"]},
            "supervised_spans": [
                {
                    "span_id": "target",
                    "target_role": "assistant",
                    "rendered_region_role": "assistant",
                    "start_token": 2,
                    "end_token": 4,
                    "region_start_token": 1,
                    "region_end_token": 5,
                }
            ],
            "packing_window": {"strategy": "sample-packing", "max_tokens": 2048},
            "chat_template_version": {"name": "chatml", "sha256": "a" * 64},
            "pipeline_stages": [
                {
                    "stage": "dataset-preparation",
                    "tokenizer_name": "chatml-tokenizer",
                    "tokenizer_revision": "rev1",
                    "chat_template_name": "chatml",
                    "chat_template_revision": "tmpl1",
                },
                {
                    "stage": "training",
                    "tokenizer_name": "chatml-tokenizer",
                    "tokenizer_revision": "rev1",
                    "chat_template_name": "chatml",
                    "chat_template_revision": "tmpl1",
                },
            ],
        },
        base_dir=tmp_path,
    )

    assert isinstance(artifact, TrainingManifestArtifact)
    assert artifact.datasets[0].kind is TrainingDatasetKind.PREFERENCE
    assert artifact.message_roles == ("assistant", "user")
    assert artifact.target_roles == ("assistant",)
    assert artifact.loss_mask_policy is not None
    assert artifact.loss_mask_policy.strategy is LossMaskStrategy.ASSISTANT_ONLY
    assert artifact.supervised_spans[0].span_id == "target"
    assert artifact.supervised_spans[0].rendered_region_role == "assistant"
    assert artifact.supervised_spans[0].source_contributions == ()
    assert artifact.packing_window is not None
    assert artifact.packing_window.strategy is PackingStrategy.SAMPLE_PACKING
    assert artifact.chat_template_version == ChatTemplateVersion(name="chatml", sha256="a" * 64)
    assert artifact.pipeline_stages == (
        TrainingPipelineStageVersion(
            stage="dataset-preparation",
            tokenizer_name="chatml-tokenizer",
            tokenizer_revision="rev1",
            chat_template_name="chatml",
            chat_template_revision="tmpl1",
        ),
        TrainingPipelineStageVersion(
            stage="training",
            tokenizer_name="chatml-tokenizer",
            tokenizer_revision="rev1",
            chat_template_name="chatml",
            chat_template_revision="tmpl1",
        ),
    )
    assert artifact.to_dict()["datasets"][0]["preference_fields"] == ["chosen", "rejected"]
    assert artifact.to_dict()["pipeline_stages"][0]["tokenizer_revision"] == "rev1"


def test_loader_parses_training_manifest_json_and_reports_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training-manifest.json"
    _write_manifest(manifest_path)
    artifact = TrainingManifestArtifact(
        kind=ArtifactKind.TRAINING_MANIFEST,
        name="train",
        location=ArtifactLocation(path=str(manifest_path)),
        provenance=ArtifactProvenance(sha256=sha256(manifest_path.read_bytes()).hexdigest()),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "training-manifest"
    assert isinstance(loaded.artifact, TrainingManifestArtifact)
    assert loaded.artifact.message_roles == ("assistant", "system", "user")
    assert loaded.artifact.target_roles == ("assistant",)
    assert loaded.artifact.supervised_spans[0].span_id == "train-0001.assistant-0"
    assert loaded.artifact.supervised_spans[0].loss_masked is True
    assert loaded.artifact.supervised_spans[0].source_contributions == (
        TrainingSourceContribution(
            source_id="assistant-answer",
            source_kind=TrainingTextSourceKind.ASSISTANT,
            source_field="messages.content",
            start_token=18,
            end_token=36,
            transform="chat-template-render",
            text_sha256="sha256:assistant-answer",
        ),
    )
    assert loaded.artifact.datasets[1].kind is TrainingDatasetKind.PREFERENCE
    assert loaded.artifact.packing_window is not None
    assert loaded.artifact.packing_window.max_tokens == 4096
    serving_stage = next(stage for stage in loaded.artifact.pipeline_stages if stage.stage == "serving")
    assert serving_stage.chat_template_sha256 == "sha256:tmpl"
    metadata = dict(loaded.metadata)
    assert metadata["dataset_count"] == 2
    assert metadata["supervised_dataset_count"] == 1
    assert metadata["preference_dataset_count"] == 1
    assert metadata["example_count"] == 192
    assert metadata["supervised_span_count"] == 1
    assert metadata["source_contribution_count"] == 1
    assert metadata["loss_mask_strategy"] == "assistant-only"
    assert metadata["packing_max_tokens"] == 4096
    assert metadata["chat_template_pinned"] is True
    assert metadata["pipeline_stage_count"] == 4
    assert metadata["pipeline_stages"] == ("dataset-preparation", "evaluation", "serving", "training")
    assert metadata["pipeline_tokenizer_pinned_count"] == 4
    assert metadata["pipeline_chat_template_pinned_count"] == 4
    assert any(name == "datasets.1.preference_fields.0" for name, _span in loaded.source_spans)
    assert loaded.warnings == ()


def test_loader_rejects_malformed_training_manifests_with_guidance(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad-training-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [{"name": "prefs", "kind": "preference"}],
                "loss_mask_policy": {"strategy": "explicit"},
            }
        ),
        encoding="utf-8",
    )
    artifact = TrainingManifestArtifact(
        kind=ArtifactKind.TRAINING_MANIFEST,
        name="bad-train",
        location=ArtifactLocation(path=str(manifest_path)),
    )

    with pytest.raises(ArtifactLoadError, match="could not be parsed") as exc_info:
        ArtifactLoader().load(artifact)

    assert exc_info.value.rule_id == "artifact-load-failed"
    assert "supervised/preference datasets" in exc_info.value.suggestion
