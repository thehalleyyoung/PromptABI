"""Typed artifact model for PromptABI verification inputs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .diagnostics import ArtifactRef, SourceSpan


class ArtifactKind(StrEnum):
    """First-class artifact categories understood by PromptABI."""

    TOKENIZER = "tokenizer"
    CHAT_TEMPLATE = "chat-template"
    SPECIAL_TOKEN_MAP = "special-token-map"
    STOP_POLICY = "stop-policy"
    SCHEMA = "schema"
    GRAMMAR = "grammar"
    TOOL_DEFINITION = "tool-definition"
    PROMPT_SEGMENT = "prompt-segment"
    PROVIDER_CONFIG = "provider-config"
    FRAMEWORK_TRUNCATION_CONFIG = "framework-truncation-config"
    TRAINING_MANIFEST = "training-manifest"
    EVALUATION_HARNESS = "evaluation-harness"


class TruncationStrategy(StrEnum):
    """Framework prompt-budget strategies represented before semantic checks exist."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    OLDEST_MESSAGE = "oldest-message"
    MIDDLE = "middle"
    SLIDING_WINDOW = "sliding-window"
    PRIORITY = "priority"
    CUSTOM = "custom"


class TrainingDatasetKind(StrEnum):
    """Training data families represented by training manifests."""

    SUPERVISED = "supervised"
    PREFERENCE = "preference"


class LossMaskStrategy(StrEnum):
    """How a training pipeline constructs labels/loss masks."""

    ASSISTANT_ONLY = "assistant-only"
    COMPLETION_ONLY = "completion-only"
    ALL_TOKENS = "all-tokens"
    EXPLICIT = "explicit"


class PackingStrategy(StrEnum):
    """How training examples are packed into finite windows."""

    NONE = "none"
    SAMPLE_PACKING = "sample-packing"
    CONCATENATE = "concatenate"
    PACKED_SEQUENCE = "packed-sequence"


class TrainingRedactionMode(StrEnum):
    """How training checks may materialize evidence in diagnostics."""

    HASH_ONLY = "hash-only"
    METADATA_ONLY = "metadata-only"
    STRUCTURAL = "structural"


class TrainingTextSourceKind(StrEnum):
    """Source categories that can contribute text to supervised target spans."""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    PREFERENCE = "preference"
    SYSTEM = "system"
    DEVELOPER = "developer"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvaluationFewShotExample:
    """A finite benchmark example rendered by an evaluation harness."""

    example_id: str
    role: str
    content: str = ""
    token_count: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty("few-shot example id", self.example_id)
        _require_non_empty("few-shot example role", self.role)
        _optional_non_negative("few-shot example token_count", self.token_count)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.example_id, "role": self.role}
        if self.content:
            data["content"] = self.content
        if self.token_count is not None:
            data["token_count"] = self.token_count
        return data


@dataclass(frozen=True, slots=True)
class ArtifactLocation:
    """Where an artifact came from.

    Exactly one of ``path`` or ``uri`` is required. Inline artifacts should be
    materialized through a config file path before semantic loaders consume them;
    the config's artifact payload remains available on the typed artifact.
    """

    path: str | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.uri is None):
            raise ValueError("artifact location must set exactly one of path or uri")
        if self.path is not None and not self.path:
            raise ValueError("artifact path must be non-empty")
        if self.uri is not None and not self.uri:
            raise ValueError("artifact uri must be non-empty")

    @property
    def ref_path(self) -> str | None:
        return self.path if self.path is not None else self.uri

    def to_dict(self) -> dict[str, str]:
        if self.path is not None:
            return {"path": self.path}
        if self.uri is not None:
            return {"uri": self.uri}
        raise AssertionError("ArtifactLocation invariant violated")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Version and supply-chain metadata attached to an artifact."""

    version: str | None = None
    revision: str | None = None
    sha256: str | None = None
    license: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("version", "revision", "sha256", "license", "source"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"artifact provenance field '{field_name}' must be non-empty")

    @property
    def ref_version(self) -> str | None:
        return self.version or self.revision or self.sha256

    def to_dict(self) -> dict[str, str]:
        data: dict[str, str] = {}
        for key in ("version", "revision", "sha256", "license", "source"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class SpecialToken:
    """A named tokenizer special token and optional numeric ID."""

    name: str
    text: str
    token_id: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("special token name must be non-empty")
        if not self.text:
            raise ValueError("special token text must be non-empty")
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("special token id must be non-negative")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "text": self.text}
        if self.token_id is not None:
            data["token_id"] = self.token_id
        return data


@dataclass(frozen=True, slots=True)
class PromptSegment:
    """A named prompt region that future budget checks can require to survive."""

    name: str
    role: str | None = None
    required: bool = False
    max_tokens: int | None = None
    token_count: int | None = None
    content: str | None = None
    overhead_tokens: int = 0
    chunk_id: str | None = None
    document_id: str | None = None
    chunk_tokenizer: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    chunk_start: int | None = None
    chunk_end: int | None = None
    expected_overlap_tokens: int | None = None
    actual_overlap_tokens: int | None = None
    citation: str | None = None
    citation_required: bool = False
    metadata_tokens: int = 0
    template_overhead_tokens: int = 0
    retrieval_payload_limit_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("prompt segment name must be non-empty")
        if self.role is not None and not self.role:
            raise ValueError("prompt segment role must be non-empty")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("prompt segment max_tokens must be positive")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("prompt segment token_count must be non-negative")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("prompt segment content must be a string")
        if self.overhead_tokens < 0:
            raise ValueError("prompt segment overhead_tokens must be non-negative")
        for field_name in ("chunk_id", "document_id", "chunk_tokenizer", "citation"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"prompt segment {field_name} must be non-empty")
        for field_name in (
            "source_start",
            "source_end",
            "chunk_start",
            "chunk_end",
            "expected_overlap_tokens",
            "actual_overlap_tokens",
            "metadata_tokens",
            "template_overhead_tokens",
            "retrieval_payload_limit_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"prompt segment {field_name} must be non-negative")
        if self.source_start is not None and self.source_end is not None and self.source_end < self.source_start:
            raise ValueError("prompt segment source_end must be greater than or equal to source_start")
        if self.chunk_start is not None and self.chunk_end is not None and self.chunk_end < self.chunk_start:
            raise ValueError("prompt segment chunk_end must be greater than or equal to chunk_start")
        if self.metadata_tokens < 0:
            raise ValueError("prompt segment metadata_tokens must be non-negative")
        if self.template_overhead_tokens < 0:
            raise ValueError("prompt segment template_overhead_tokens must be non-negative")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "required": self.required}
        if self.role is not None:
            data["role"] = self.role
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.token_count is not None:
            data["token_count"] = self.token_count
        if self.content is not None:
            data["content"] = self.content
        if self.overhead_tokens:
            data["overhead_tokens"] = self.overhead_tokens
        for key in (
            "chunk_id",
            "document_id",
            "chunk_tokenizer",
            "source_start",
            "source_end",
            "chunk_start",
            "chunk_end",
            "expected_overlap_tokens",
            "actual_overlap_tokens",
            "citation",
            "retrieval_payload_limit_tokens",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.citation_required:
            data["citation_required"] = self.citation_required
        if self.metadata_tokens:
            data["metadata_tokens"] = self.metadata_tokens
        if self.template_overhead_tokens:
            data["template_overhead_tokens"] = self.template_overhead_tokens
        return data


@dataclass(frozen=True, slots=True)
class TrainingDatasetSpec:
    """One supervised or preference dataset declared by a training manifest."""

    name: str
    kind: TrainingDatasetKind = TrainingDatasetKind.SUPERVISED
    path: str | None = None
    split: str | None = None
    format: str = "chat-jsonl"
    example_count: int | None = None
    content_fields: tuple[str, ...] = ()
    preference_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", TrainingDatasetKind(self.kind))
        _require_non_empty("training dataset name", self.name)
        _optional_non_empty("training dataset path", self.path)
        _optional_non_empty("training dataset split", self.split)
        _require_non_empty("training dataset format", self.format)
        _optional_non_negative("training dataset example_count", self.example_count)
        object.__setattr__(self, "content_fields", _unique_strings(self.content_fields, field_name="training dataset content_fields"))
        object.__setattr__(
            self,
            "preference_fields",
            _unique_strings(self.preference_fields, field_name="training dataset preference_fields"),
        )
        if self.kind is TrainingDatasetKind.PREFERENCE and not self.preference_fields:
            raise ValueError("preference training datasets must declare preference_fields")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "kind": self.kind.value,
            "format": self.format,
        }
        if self.path is not None:
            data["path"] = self.path
        if self.split is not None:
            data["split"] = self.split
        if self.example_count is not None:
            data["example_count"] = self.example_count
        if self.content_fields:
            data["content_fields"] = list(self.content_fields)
        if self.preference_fields:
            data["preference_fields"] = list(self.preference_fields)
        return data


@dataclass(frozen=True, slots=True)
class SystemMessagePolicy:
    """Policy for system/developer messages in training examples."""

    required: bool = False
    allow_override: bool = False
    default: str | None = None
    allowed_hashes: tuple[str, ...] = ()
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        _optional_non_empty("system message default", self.default)
        _optional_non_negative("system message max_tokens", self.max_tokens)
        object.__setattr__(self, "allowed_hashes", _unique_strings(self.allowed_hashes, field_name="system message allowed_hashes"))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "required": self.required,
            "allow_override": self.allow_override,
        }
        if self.default is not None:
            data["default"] = self.default
        if self.allowed_hashes:
            data["allowed_hashes"] = list(self.allowed_hashes)
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        return data


@dataclass(frozen=True, slots=True)
class RoleLabel:
    """Mapping from dataset role labels to canonical chat-template roles."""

    source_role: str
    canonical_role: str
    supervised_target: bool = False
    trainable: bool = True
    required: bool = False

    def __post_init__(self) -> None:
        _require_non_empty("role label source_role", self.source_role)
        _require_non_empty("role label canonical_role", self.canonical_role)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "source_role": self.source_role,
            "canonical_role": self.canonical_role,
            "supervised_target": self.supervised_target,
            "trainable": self.trainable,
        }
        if self.required:
            data["required"] = self.required
        return data


@dataclass(frozen=True, slots=True)
class TrainingSourceContribution:
    """A token range inside a supervised span attributed to an upstream text source."""

    source_id: str
    source_kind: TrainingTextSourceKind
    start_token: int
    end_token: int
    transform: str
    source_field: str | None = None
    text_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source_kind, str):
            object.__setattr__(self, "source_kind", TrainingTextSourceKind(self.source_kind))
        _require_non_empty("training source contribution source_id", self.source_id)
        _require_non_empty("training source contribution transform", self.transform)
        _optional_non_empty("training source contribution source_field", self.source_field)
        _optional_non_empty("training source contribution text_sha256", self.text_sha256)
        for field_name in ("start_token", "end_token"):
            _optional_non_negative(f"training source contribution {field_name}", getattr(self, field_name))
        if self.end_token < self.start_token:
            raise ValueError("training source contribution end_token must be greater than or equal to start_token")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "transform": self.transform,
        }
        if self.source_field is not None:
            data["source_field"] = self.source_field
        if self.text_sha256 is not None:
            data["text_sha256"] = self.text_sha256
        return data


@dataclass(frozen=True, slots=True)
class TrainingSpanContract:
    """Observed finite span facts from a rendered/tokenized supervised example."""

    span_id: str
    target_role: str
    rendered_region_role: str
    start_token: int
    end_token: int
    region_start_token: int
    region_end_token: int
    supervised_target: bool = True
    loss_masked: bool = True
    packed_example_id: str | None = None
    crosses_packing_boundary: bool = False
    source_contributions: tuple[TrainingSourceContribution, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("training span id", self.span_id)
        _require_non_empty("training span target_role", self.target_role)
        _require_non_empty("training span rendered_region_role", self.rendered_region_role)
        for field_name in ("start_token", "end_token", "region_start_token", "region_end_token"):
            _optional_non_negative(f"training span {field_name}", getattr(self, field_name))
        if self.end_token < self.start_token:
            raise ValueError("training span end_token must be greater than or equal to start_token")
        if self.region_end_token < self.region_start_token:
            raise ValueError("training span region_end_token must be greater than or equal to region_start_token")
        _optional_non_empty("training span packed_example_id", self.packed_example_id)
        object.__setattr__(self, "source_contributions", tuple(sorted(self.source_contributions, key=lambda item: item.source_id)))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "span_id": self.span_id,
            "target_role": self.target_role,
            "rendered_region_role": self.rendered_region_role,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "region_start_token": self.region_start_token,
            "region_end_token": self.region_end_token,
            "supervised_target": self.supervised_target,
            "loss_masked": self.loss_masked,
        }
        if self.packed_example_id is not None:
            data["packed_example_id"] = self.packed_example_id
        if self.crosses_packing_boundary:
            data["crosses_packing_boundary"] = self.crosses_packing_boundary
        if self.source_contributions:
            data["source_contributions"] = [contribution.to_dict() for contribution in self.source_contributions]
        return data


@dataclass(frozen=True, slots=True)
class PreferencePairContract:
    """Finite DPO/RLHF preference-pair facts emitted by a data-preparation job."""

    pair_id: str
    prompt_sha256: str
    chosen_sha256: str
    rejected_sha256: str
    chosen_role_layout: tuple[str, ...]
    rejected_role_layout: tuple[str, ...]
    chosen_tokenizer: str
    rejected_tokenizer: str
    chosen_mask_policy: str
    rejected_mask_policy: str
    chosen_prompt_tokens: int
    rejected_prompt_tokens: int
    chosen_response_start_token: int
    rejected_response_start_token: int
    chosen_response_end_token: int
    rejected_response_end_token: int
    chosen_prompt_sha256: str | None = None
    rejected_prompt_sha256: str | None = None
    chosen_truncated: bool = False
    rejected_truncated: bool = False
    chosen_packed_example_id: str | None = None
    rejected_packed_example_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "pair_id",
            "prompt_sha256",
            "chosen_sha256",
            "rejected_sha256",
            "chosen_tokenizer",
            "rejected_tokenizer",
            "chosen_mask_policy",
            "rejected_mask_policy",
        ):
            _require_non_empty(f"preference pair {field_name}", getattr(self, field_name))
        object.__setattr__(
            self,
            "chosen_role_layout",
            _strings_preserve_order(self.chosen_role_layout, field_name="preference pair chosen_role_layout"),
        )
        object.__setattr__(
            self,
            "rejected_role_layout",
            _strings_preserve_order(self.rejected_role_layout, field_name="preference pair rejected_role_layout"),
        )
        if not self.chosen_role_layout or not self.rejected_role_layout:
            raise ValueError("preference pair role layouts must be non-empty")
        for field_name in (
            "chosen_prompt_tokens",
            "rejected_prompt_tokens",
            "chosen_response_start_token",
            "rejected_response_start_token",
            "chosen_response_end_token",
            "rejected_response_end_token",
        ):
            _optional_non_negative(f"preference pair {field_name}", getattr(self, field_name))
        if self.chosen_response_end_token < self.chosen_response_start_token:
            raise ValueError("preference pair chosen_response_end_token must be greater than or equal to chosen_response_start_token")
        if self.rejected_response_end_token < self.rejected_response_start_token:
            raise ValueError("preference pair rejected_response_end_token must be greater than or equal to rejected_response_start_token")
        for field_name in ("chosen_prompt_sha256", "rejected_prompt_sha256"):
            _optional_non_empty(f"preference pair {field_name}", getattr(self, field_name))
        for field_name in ("chosen_packed_example_id", "rejected_packed_example_id"):
            _optional_non_empty(f"preference pair {field_name}", getattr(self, field_name))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "pair_id": self.pair_id,
            "prompt_sha256": self.prompt_sha256,
            "chosen_sha256": self.chosen_sha256,
            "rejected_sha256": self.rejected_sha256,
            "chosen_role_layout": list(self.chosen_role_layout),
            "rejected_role_layout": list(self.rejected_role_layout),
            "chosen_tokenizer": self.chosen_tokenizer,
            "rejected_tokenizer": self.rejected_tokenizer,
            "chosen_mask_policy": self.chosen_mask_policy,
            "rejected_mask_policy": self.rejected_mask_policy,
            "chosen_prompt_tokens": self.chosen_prompt_tokens,
            "rejected_prompt_tokens": self.rejected_prompt_tokens,
            "chosen_response_start_token": self.chosen_response_start_token,
            "rejected_response_start_token": self.rejected_response_start_token,
            "chosen_response_end_token": self.chosen_response_end_token,
            "rejected_response_end_token": self.rejected_response_end_token,
            "chosen_truncated": self.chosen_truncated,
            "rejected_truncated": self.rejected_truncated,
        }
        if self.chosen_prompt_sha256 is not None:
            data["chosen_prompt_sha256"] = self.chosen_prompt_sha256
        if self.rejected_prompt_sha256 is not None:
            data["rejected_prompt_sha256"] = self.rejected_prompt_sha256
        if self.chosen_packed_example_id is not None:
            data["chosen_packed_example_id"] = self.chosen_packed_example_id
        if self.rejected_packed_example_id is not None:
            data["rejected_packed_example_id"] = self.rejected_packed_example_id
        return data


@dataclass(frozen=True, slots=True)
class LossMaskPolicy:
    """Finite loss-mask contract for supervised target construction."""

    strategy: LossMaskStrategy = LossMaskStrategy.ASSISTANT_ONLY
    target_roles: tuple[str, ...] = ()
    ignored_roles: tuple[str, ...] = ()
    explicit_mask_field: str | None = None
    label_pad_token_id: int = -100

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", LossMaskStrategy(self.strategy))
        _optional_non_empty("loss mask explicit_mask_field", self.explicit_mask_field)
        object.__setattr__(self, "target_roles", _unique_strings(self.target_roles, field_name="loss mask target_roles"))
        object.__setattr__(self, "ignored_roles", _unique_strings(self.ignored_roles, field_name="loss mask ignored_roles"))
        if self.strategy is LossMaskStrategy.EXPLICIT and self.explicit_mask_field is None:
            raise ValueError("explicit loss-mask policies must declare explicit_mask_field")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "strategy": self.strategy.value,
            "label_pad_token_id": self.label_pad_token_id,
        }
        if self.target_roles:
            data["target_roles"] = list(self.target_roles)
        if self.ignored_roles:
            data["ignored_roles"] = list(self.ignored_roles)
        if self.explicit_mask_field is not None:
            data["explicit_mask_field"] = self.explicit_mask_field
        return data


@dataclass(frozen=True, slots=True)
class PackingWindow:
    """Finite sequence-packing window for training examples."""

    strategy: PackingStrategy = PackingStrategy.NONE
    max_tokens: int | None = None
    stride_tokens: int = 0
    boundary_token: str | None = None
    preserve_example_boundaries: bool = True
    reset_position_ids: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", PackingStrategy(self.strategy))
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("packing max_tokens must be positive")
        _optional_non_negative("packing stride_tokens", self.stride_tokens)
        _optional_non_empty("packing boundary_token", self.boundary_token)
        if self.strategy is not PackingStrategy.NONE and self.max_tokens is None:
            raise ValueError("non-empty packing strategies must declare max_tokens")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "strategy": self.strategy.value,
            "preserve_example_boundaries": self.preserve_example_boundaries,
            "reset_position_ids": self.reset_position_ids,
        }
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.stride_tokens:
            data["stride_tokens"] = self.stride_tokens
        if self.boundary_token is not None:
            data["boundary_token"] = self.boundary_token
        return data


@dataclass(frozen=True, slots=True)
class ChatTemplateVersion:
    """Pinned chat-template contract used when materializing training data."""

    name: str
    version: str | None = None
    revision: str | None = None
    sha256: str | None = None
    tokenizer_name: str | None = None
    add_generation_prompt: bool | None = None

    def __post_init__(self) -> None:
        _require_non_empty("chat-template version name", self.name)
        for field_name in ("version", "revision", "sha256", "tokenizer_name"):
            _optional_non_empty(f"chat-template {field_name}", getattr(self, field_name))

    @property
    def pinned(self) -> bool:
        return self.version is not None or self.revision is not None or self.sha256 is not None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name}
        for key in ("version", "revision", "sha256", "tokenizer_name"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.add_generation_prompt is not None:
            data["add_generation_prompt"] = self.add_generation_prompt
        return data


@dataclass(frozen=True, slots=True)
class TrainingPipelineStageVersion:
    """Tokenizer and template pins observed at one fine-tuning pipeline stage."""

    stage: str
    tokenizer_name: str | None = None
    tokenizer_version: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_sha256: str | None = None
    chat_template_name: str | None = None
    chat_template_version: str | None = None
    chat_template_revision: str | None = None
    chat_template_sha256: str | None = None
    add_generation_prompt: bool | None = None

    def __post_init__(self) -> None:
        _require_non_empty("training pipeline stage", self.stage)
        for field_name in (
            "tokenizer_name",
            "tokenizer_version",
            "tokenizer_revision",
            "tokenizer_sha256",
            "chat_template_name",
            "chat_template_version",
            "chat_template_revision",
            "chat_template_sha256",
        ):
            _optional_non_empty(f"training pipeline stage {field_name}", getattr(self, field_name))

    @property
    def tokenizer_pinned(self) -> bool:
        return any(
            value is not None
            for value in (
                self.tokenizer_name,
                self.tokenizer_version,
                self.tokenizer_revision,
                self.tokenizer_sha256,
            )
        )

    @property
    def chat_template_pinned(self) -> bool:
        return any(
            value is not None
            for value in (
                self.chat_template_name,
                self.chat_template_version,
                self.chat_template_revision,
                self.chat_template_sha256,
                self.add_generation_prompt,
            )
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"stage": self.stage}
        for key in (
            "tokenizer_name",
            "tokenizer_version",
            "tokenizer_revision",
            "tokenizer_sha256",
            "chat_template_name",
            "chat_template_version",
            "chat_template_revision",
            "chat_template_sha256",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.add_generation_prompt is not None:
            data["add_generation_prompt"] = self.add_generation_prompt
        return data


@dataclass(frozen=True, slots=True)
class TrainingRedactionPolicy:
    """Privacy contract for persisted training witnesses and reports."""

    mode: TrainingRedactionMode = TrainingRedactionMode.HASH_ONLY
    require_text_hashes: bool = True
    allow_raw_text_in_witnesses: bool = False
    allowed_report_fields: tuple[str, ...] = ()
    forbidden_report_fields: tuple[str, ...] = ()
    restricted_metadata_keys: tuple[str, ...] = ()
    secret_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", TrainingRedactionMode(self.mode))
        object.__setattr__(
            self,
            "allowed_report_fields",
            _unique_strings(self.allowed_report_fields, field_name="training redaction allowed_report_fields"),
        )
        object.__setattr__(
            self,
            "forbidden_report_fields",
            _unique_strings(self.forbidden_report_fields, field_name="training redaction forbidden_report_fields"),
        )
        object.__setattr__(
            self,
            "restricted_metadata_keys",
            _unique_strings(self.restricted_metadata_keys, field_name="training redaction restricted_metadata_keys"),
        )
        object.__setattr__(
            self,
            "secret_patterns",
            _unique_strings(self.secret_patterns, field_name="training redaction secret_patterns"),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "mode": self.mode.value,
            "require_text_hashes": self.require_text_hashes,
            "allow_raw_text_in_witnesses": self.allow_raw_text_in_witnesses,
        }
        if self.allowed_report_fields:
            data["allowed_report_fields"] = list(self.allowed_report_fields)
        if self.forbidden_report_fields:
            data["forbidden_report_fields"] = list(self.forbidden_report_fields)
        if self.restricted_metadata_keys:
            data["restricted_metadata_keys"] = list(self.restricted_metadata_keys)
        if self.secret_patterns:
            data["secret_patterns"] = list(self.secret_patterns)
        return data


@dataclass(frozen=True, slots=True)
class SyntheticSchemaOutputContract:
    """Finite schema/parser facts promised by a synthetic-data generator."""

    case_id: str
    valid: bool | None = None
    parses: bool | None = None
    schema_valid: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("synthetic schema output case_id", self.case_id)
        _optional_non_empty("synthetic schema output reason", self.reason)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.case_id}
        for key in ("valid", "parses", "schema_valid", "reason"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class SyntheticToolCallContract:
    """Finite tool-call envelope facts promised by a synthetic-data generator."""

    case_id: str
    valid: bool | None = None
    malformed: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("synthetic tool-call case_id", self.case_id)
        _optional_non_empty("synthetic tool-call reason", self.reason)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.case_id}
        for key in ("valid", "malformed", "reason"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class SyntheticTruncationContract:
    """Finite truncation facts promised by a synthetic-data generator."""

    case_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    max_context_tokens: int | None = None
    preserved_required_roles: tuple[str, ...] = ()
    truncated_required_roles: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("synthetic truncation case_id", self.case_id)
        for field_name in ("input_tokens", "output_tokens", "max_context_tokens"):
            _optional_non_negative(f"synthetic truncation {field_name}", getattr(self, field_name))
        object.__setattr__(
            self,
            "preserved_required_roles",
            _unique_strings(self.preserved_required_roles, field_name="synthetic truncation preserved_required_roles"),
        )
        object.__setattr__(
            self,
            "truncated_required_roles",
            _unique_strings(self.truncated_required_roles, field_name="synthetic truncation truncated_required_roles"),
        )
        _optional_non_empty("synthetic truncation reason", self.reason)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.case_id}
        for key in ("input_tokens", "output_tokens", "max_context_tokens"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.preserved_required_roles:
            data["preserved_required_roles"] = list(self.preserved_required_roles)
        if self.truncated_required_roles:
            data["truncated_required_roles"] = list(self.truncated_required_roles)
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class SyntheticGeneratorSpec:
    """Static contract for a synthetic-data generator before examples are materialized."""

    name: str
    generator_type: str = "synthetic-chat"
    output_roles: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    forbidden_roles: tuple[str, ...] = ()
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    schema_outputs: tuple[SyntheticSchemaOutputContract, ...] = ()
    tool_calls: tuple[SyntheticToolCallContract, ...] = ()
    truncation_cases: tuple[SyntheticTruncationContract, ...] = ()
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("synthetic generator name", self.name)
        _require_non_empty("synthetic generator type", self.generator_type)
        object.__setattr__(self, "output_roles", _unique_strings(self.output_roles, field_name="synthetic generator output_roles"))
        object.__setattr__(self, "required_roles", _unique_strings(self.required_roles, field_name="synthetic generator required_roles"))
        object.__setattr__(self, "forbidden_roles", _unique_strings(self.forbidden_roles, field_name="synthetic generator forbidden_roles"))
        for field_name in ("max_prompt_tokens", "max_completion_tokens"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"synthetic generator {field_name} must be positive")
        object.__setattr__(self, "schema_outputs", tuple(sorted(self.schema_outputs, key=lambda item: item.case_id)))
        object.__setattr__(self, "tool_calls", tuple(sorted(self.tool_calls, key=lambda item: item.case_id)))
        object.__setattr__(self, "truncation_cases", tuple(sorted(self.truncation_cases, key=lambda item: item.case_id)))
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata, key=lambda item: item[0])))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "generator_type": self.generator_type,
        }
        if self.output_roles:
            data["output_roles"] = list(self.output_roles)
        if self.required_roles:
            data["required_roles"] = list(self.required_roles)
        if self.forbidden_roles:
            data["forbidden_roles"] = list(self.forbidden_roles)
        if self.max_prompt_tokens is not None:
            data["max_prompt_tokens"] = self.max_prompt_tokens
        if self.max_completion_tokens is not None:
            data["max_completion_tokens"] = self.max_completion_tokens
        if self.schema_outputs:
            data["schema_outputs"] = [case.to_dict() for case in self.schema_outputs]
        if self.tool_calls:
            data["tool_calls"] = [case.to_dict() for case in self.tool_calls]
        if self.truncation_cases:
            data["truncation_cases"] = [case.to_dict() for case in self.truncation_cases]
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True, slots=True)
class BaseArtifact:
    """Common artifact identity, location, provenance, and payload."""

    kind: ArtifactKind
    name: str
    location: ArtifactLocation
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)
    metadata: tuple[tuple[str, object], ...] = ()
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact name must be non-empty")
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata, key=lambda item: item[0])))

    def to_ref(self) -> ArtifactRef:
        return ArtifactRef(
            kind=self.kind.value,
            name=self.name,
            path=self.location.path,
            uri=self.location.uri,
            version=self.provenance.version,
            revision=self.provenance.revision,
            sha256=self.provenance.sha256,
            license=self.provenance.license,
            source=self.provenance.source,
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind.value,
            "name": self.name,
            "location": self.location.to_dict(),
        }
        provenance = self.provenance.to_dict()
        if provenance:
            data["provenance"] = provenance
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.source_span is not None:
            data["source_span"] = self.source_span.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class TokenizerArtifact(BaseArtifact):
    family: str | None = None
    added_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.TOKENIZER)
        object.__setattr__(self, "added_tokens", tuple(sorted(dict.fromkeys(self.added_tokens))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        if self.family is not None:
            data["family"] = self.family
        if self.added_tokens:
            data["added_tokens"] = list(self.added_tokens)
        return data


@dataclass(frozen=True, slots=True)
class ChatTemplateArtifact(BaseArtifact):
    template_format: str = "jinja"
    roles: tuple[str, ...] = ()
    add_generation_prompt: bool | None = None

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.CHAT_TEMPLATE)
        if not self.template_format:
            raise ValueError("chat template format must be non-empty")
        object.__setattr__(self, "roles", tuple(sorted(dict.fromkeys(self.roles))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["template_format"] = self.template_format
        if self.roles:
            data["roles"] = list(self.roles)
        if self.add_generation_prompt is not None:
            data["add_generation_prompt"] = self.add_generation_prompt
        return data


@dataclass(frozen=True, slots=True)
class SpecialTokenMapArtifact(BaseArtifact):
    tokens: tuple[SpecialToken, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.SPECIAL_TOKEN_MAP)
        object.__setattr__(self, "tokens", tuple(sorted(self.tokens, key=lambda token: token.name)))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["tokens"] = [token.to_dict() for token in self.tokens]
        return data


@dataclass(frozen=True, slots=True)
class StopPolicyArtifact(BaseArtifact):
    stop_sequences: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    include_eos: bool = True
    source_family: str | None = None

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.STOP_POLICY)
        if any(sequence == "" for sequence in self.stop_sequences):
            raise ValueError("stop sequences must be non-empty")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop token ids must be non-negative")
        object.__setattr__(self, "stop_sequences", tuple(sorted(dict.fromkeys(self.stop_sequences))))
        object.__setattr__(self, "stop_token_ids", tuple(sorted(dict.fromkeys(self.stop_token_ids))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["stop_sequences"] = list(self.stop_sequences)
        if self.stop_token_ids:
            data["stop_token_ids"] = list(self.stop_token_ids)
        data["include_eos"] = self.include_eos
        if self.source_family is not None:
            data["source_family"] = self.source_family
        return data


@dataclass(frozen=True, slots=True)
class SchemaArtifact(BaseArtifact):
    dialect: str = "json-schema"

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.SCHEMA)
        if not self.dialect:
            raise ValueError("schema dialect must be non-empty")

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["dialect"] = self.dialect
        return data


@dataclass(frozen=True, slots=True)
class GrammarArtifact(BaseArtifact):
    grammar_type: str = "promptabi"
    start_symbol: str | None = None
    rule_names: tuple[str, ...] = ()
    supported_fragment: bool | None = None

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.GRAMMAR)
        if not self.grammar_type:
            raise ValueError("grammar type must be non-empty")
        if self.start_symbol is not None and not self.start_symbol:
            raise ValueError("grammar start_symbol must be non-empty")
        object.__setattr__(self, "rule_names", tuple(sorted(dict.fromkeys(self.rule_names))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["grammar_type"] = self.grammar_type
        if self.start_symbol is not None:
            data["start_symbol"] = self.start_symbol
        if self.rule_names:
            data["rule_names"] = list(self.rule_names)
        if self.supported_fragment is not None:
            data["supported_fragment"] = self.supported_fragment
        return data


@dataclass(frozen=True, slots=True)
class ToolDefinitionArtifact(BaseArtifact):
    provider: str | None = None
    tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.TOOL_DEFINITION)
        object.__setattr__(self, "tool_names", tuple(sorted(dict.fromkeys(self.tool_names))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        if self.provider is not None:
            data["provider"] = self.provider
        if self.tool_names:
            data["tool_names"] = list(self.tool_names)
        return data


@dataclass(frozen=True, slots=True)
class PromptSegmentArtifact(BaseArtifact):
    segments: tuple[PromptSegment, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.PROMPT_SEGMENT)
        if not self.segments:
            raise ValueError("prompt-segment artifacts must define at least one segment")
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def required_segments(self) -> tuple[PromptSegment, ...]:
        return tuple(segment for segment in self.segments if segment.required)

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["segments"] = [segment.to_dict() for segment in self.segments]
        return data


@dataclass(frozen=True, slots=True)
class ProviderConfigArtifact(BaseArtifact):
    provider: str = ""
    api_family: str | None = None

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.PROVIDER_CONFIG)
        if not self.provider:
            raise ValueError("provider name must be non-empty")

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["provider"] = self.provider
        if self.api_family is not None:
            data["api_family"] = self.api_family
        return data


@dataclass(frozen=True, slots=True)
class FrameworkTruncationConfigArtifact(BaseArtifact):
    framework: str = ""
    strategy: TruncationStrategy = TruncationStrategy.NONE
    max_context_tokens: int | None = None
    reserve_output_tokens: int = 0
    reserved_tool_tokens: int = 0
    generation_prompt_tokens: int = 0
    special_token_overhead: int = 0
    model: str | None = None
    preserve_system: bool = False
    preserve_tools: bool = False
    drop_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG)
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", TruncationStrategy(self.strategy))
        if not self.framework:
            raise ValueError("framework name must be non-empty")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.reserve_output_tokens < 0:
            raise ValueError("reserve_output_tokens must be non-negative")
        if self.reserved_tool_tokens < 0:
            raise ValueError("reserved_tool_tokens must be non-negative")
        if self.generation_prompt_tokens < 0:
            raise ValueError("generation_prompt_tokens must be non-negative")
        if self.special_token_overhead < 0:
            raise ValueError("special_token_overhead must be non-negative")
        if self.model is not None and not self.model:
            raise ValueError("model must be non-empty")
        object.__setattr__(self, "drop_roles", tuple(sorted(dict.fromkeys(self.drop_roles))))

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["framework"] = self.framework
        data["strategy"] = self.strategy.value
        if self.max_context_tokens is not None:
            data["max_context_tokens"] = self.max_context_tokens
        if self.reserve_output_tokens:
            data["reserve_output_tokens"] = self.reserve_output_tokens
        if self.reserved_tool_tokens:
            data["reserved_tool_tokens"] = self.reserved_tool_tokens
        if self.generation_prompt_tokens:
            data["generation_prompt_tokens"] = self.generation_prompt_tokens
        if self.special_token_overhead:
            data["special_token_overhead"] = self.special_token_overhead
        if self.model is not None:
            data["model"] = self.model
        if self.preserve_system:
            data["preserve_system"] = self.preserve_system
        if self.preserve_tools:
            data["preserve_tools"] = self.preserve_tools
        if self.drop_roles:
            data["drop_roles"] = list(self.drop_roles)
        return data


@dataclass(frozen=True, slots=True)
class TrainingManifestArtifact(BaseArtifact):
    """A finite summary of supervised/preference data interface contracts."""

    dataset_format: str = "jsonl"
    message_roles: tuple[str, ...] = ()
    target_roles: tuple[str, ...] = ()
    example_count: int | None = None
    packed: bool = False
    datasets: tuple[TrainingDatasetSpec, ...] = ()
    system_message_policy: SystemMessagePolicy | None = None
    role_labels: tuple[RoleLabel, ...] = ()
    supervised_spans: tuple[TrainingSpanContract, ...] = ()
    preference_pairs: tuple[PreferencePairContract, ...] = ()
    loss_mask_policy: LossMaskPolicy | None = None
    packing_window: PackingWindow | None = None
    chat_template_version: ChatTemplateVersion | None = None
    pipeline_stages: tuple[TrainingPipelineStageVersion, ...] = ()
    redaction_policy: TrainingRedactionPolicy | None = None
    synthetic_generators: tuple[SyntheticGeneratorSpec, ...] = ()

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.TRAINING_MANIFEST)
        if not self.dataset_format:
            raise ValueError("training manifest dataset_format must be non-empty")
        if self.example_count is not None and self.example_count < 0:
            raise ValueError("training manifest example_count must be non-negative")
        object.__setattr__(self, "message_roles", _unique_strings(self.message_roles, field_name="training manifest message_roles"))
        object.__setattr__(self, "target_roles", _unique_strings(self.target_roles, field_name="training manifest target_roles"))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "role_labels", tuple(sorted(self.role_labels, key=lambda label: (label.source_role, label.canonical_role))))
        supervised_spans = tuple(sorted(self.supervised_spans, key=lambda span: span.span_id))
        if len({span.span_id for span in supervised_spans}) != len(supervised_spans):
            raise ValueError("training manifest supervised span IDs must be unique")
        object.__setattr__(self, "supervised_spans", supervised_spans)
        preference_pairs = tuple(sorted(self.preference_pairs, key=lambda pair: pair.pair_id))
        if len({pair.pair_id for pair in preference_pairs}) != len(preference_pairs):
            raise ValueError("training manifest preference pair IDs must be unique")
        object.__setattr__(self, "preference_pairs", preference_pairs)
        pipeline_stages = tuple(sorted(self.pipeline_stages, key=lambda stage: stage.stage))
        if len({stage.stage for stage in pipeline_stages}) != len(pipeline_stages):
            raise ValueError("training manifest pipeline stage names must be unique")
        object.__setattr__(self, "pipeline_stages", pipeline_stages)
        synthetic_generators = tuple(sorted(self.synthetic_generators, key=lambda generator: generator.name))
        if len({generator.name for generator in synthetic_generators}) != len(synthetic_generators):
            raise ValueError("training manifest synthetic generator names must be unique")
        object.__setattr__(self, "synthetic_generators", synthetic_generators)
        if not self.target_roles and self.loss_mask_policy is not None and self.loss_mask_policy.target_roles:
            object.__setattr__(self, "target_roles", self.loss_mask_policy.target_roles)
        if not self.message_roles and self.role_labels:
            object.__setattr__(
                self,
                "message_roles",
                _unique_strings(
                    (label.canonical_role for label in self.role_labels),
                    field_name="training manifest message_roles",
                ),
            )

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["dataset_format"] = self.dataset_format
        if self.message_roles:
            data["message_roles"] = list(self.message_roles)
        if self.target_roles:
            data["target_roles"] = list(self.target_roles)
        if self.example_count is not None:
            data["example_count"] = self.example_count
        if self.packed:
            data["packed"] = self.packed
        if self.datasets:
            data["datasets"] = [dataset.to_dict() for dataset in self.datasets]
        if self.system_message_policy is not None:
            data["system_message_policy"] = self.system_message_policy.to_dict()
        if self.role_labels:
            data["role_labels"] = [label.to_dict() for label in self.role_labels]
        if self.supervised_spans:
            data["supervised_spans"] = [span.to_dict() for span in self.supervised_spans]
        if self.preference_pairs:
            data["preference_pairs"] = [pair.to_dict() for pair in self.preference_pairs]
        if self.loss_mask_policy is not None:
            data["loss_mask_policy"] = self.loss_mask_policy.to_dict()
        if self.packing_window is not None:
            data["packing_window"] = self.packing_window.to_dict()
        if self.chat_template_version is not None:
            data["chat_template_version"] = self.chat_template_version.to_dict()
        if self.pipeline_stages:
            data["pipeline_stages"] = [stage.to_dict() for stage in self.pipeline_stages]
        if self.redaction_policy is not None:
            data["redaction_policy"] = self.redaction_policy.to_dict()
        if self.synthetic_generators:
            data["synthetic_generators"] = [generator.to_dict() for generator in self.synthetic_generators]
        return data


@dataclass(frozen=True, slots=True)
class EvaluationHarnessArtifact(BaseArtifact):
    """A finite summary of benchmark prompt/parser/provider assumptions."""

    benchmark_name: str = "evaluation"
    model: str | None = None
    provider: str | None = None
    tokenizer: str | None = None
    prompt_template: str | None = None
    answer_parser: str | None = None
    answer_schema: str | None = None
    stop_sequences: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ()
    required_prompt_variables: tuple[str, ...] = ()
    prompt_variables: tuple[str, ...] = ()
    few_shot_examples: tuple[EvaluationFewShotExample, ...] = ()
    max_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.EVALUATION_HARNESS)
        _require_non_empty("evaluation harness benchmark_name", self.benchmark_name)
        for field_name in ("model", "provider", "tokenizer", "prompt_template", "answer_parser", "answer_schema"):
            _optional_non_empty(f"evaluation harness {field_name}", getattr(self, field_name))
        _optional_non_negative("evaluation harness max_prompt_tokens", self.max_prompt_tokens)
        object.__setattr__(self, "stop_sequences", _unique_strings(self.stop_sequences, field_name="evaluation harness stop_sequences"))
        object.__setattr__(self, "allowed_roles", _unique_strings(self.allowed_roles, field_name="evaluation harness allowed_roles"))
        object.__setattr__(
            self,
            "required_prompt_variables",
            _unique_strings(self.required_prompt_variables, field_name="evaluation harness required_prompt_variables"),
        )
        object.__setattr__(self, "prompt_variables", _unique_strings(self.prompt_variables, field_name="evaluation harness prompt_variables"))
        examples = tuple(sorted(self.few_shot_examples, key=lambda example: example.example_id))
        if len({example.example_id for example in examples}) != len(examples):
            raise ValueError("evaluation harness few-shot example IDs must be unique")
        object.__setattr__(self, "few_shot_examples", examples)

    def to_dict(self) -> dict[str, object]:
        data = BaseArtifact.to_dict(self)
        data["benchmark_name"] = self.benchmark_name
        for key in ("model", "provider", "tokenizer", "prompt_template", "answer_parser", "answer_schema"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.stop_sequences:
            data["stop_sequences"] = list(self.stop_sequences)
        if self.allowed_roles:
            data["allowed_roles"] = list(self.allowed_roles)
        if self.required_prompt_variables:
            data["required_prompt_variables"] = list(self.required_prompt_variables)
        if self.prompt_variables:
            data["prompt_variables"] = list(self.prompt_variables)
        if self.few_shot_examples:
            data["few_shot_examples"] = [example.to_dict() for example in self.few_shot_examples]
        if self.max_prompt_tokens is not None:
            data["max_prompt_tokens"] = self.max_prompt_tokens
        return data


Artifact = (
    TokenizerArtifact
    | ChatTemplateArtifact
    | SpecialTokenMapArtifact
    | StopPolicyArtifact
    | SchemaArtifact
    | GrammarArtifact
    | ToolDefinitionArtifact
    | PromptSegmentArtifact
    | ProviderConfigArtifact
    | FrameworkTruncationConfigArtifact
    | TrainingManifestArtifact
    | EvaluationHarnessArtifact
)


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """A deterministic collection of all artifacts in one verification run."""

    artifacts: tuple[Artifact, ...] = ()

    def __post_init__(self) -> None:
        names: set[str] = set()
        for artifact in self.artifacts:
            if artifact.name in names:
                raise ValueError(f"duplicate artifact name: {artifact.name}")
            names.add(artifact.name)
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=lambda artifact: (artifact.kind.value, artifact.name))),
        )

    def __iter__(self):
        return iter(self.artifacts)

    def by_name(self, name: str) -> Artifact:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {"artifacts": [artifact.to_dict() for artifact in self.artifacts]}


def artifact_from_config(
    name: str,
    spec: str | dict[str, Any],
    *,
    base_dir: Path,
    source_span: SourceSpan | None = None,
) -> Artifact:
    """Build a typed artifact from legacy or typed config syntax."""

    if isinstance(spec, str):
        location = _location_from_path(spec, base_dir)
        return SchemaArtifact(
            kind=ArtifactKind.SCHEMA,
            name=name,
            location=location,
            source_span=source_span,
        )

    if not isinstance(spec, dict):
        raise ValueError(f"artifact '{name}' must be a path string or object")

    raw_kind = spec.get("kind")
    if not isinstance(raw_kind, str) or not raw_kind:
        raise ValueError(f"artifact '{name}' field 'kind' must be a non-empty string")
    try:
        kind = ArtifactKind(raw_kind)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ArtifactKind)
        raise ValueError(f"artifact '{name}' has unsupported kind '{raw_kind}' (expected one of {allowed})") from exc

    location = _location_from_spec(name, spec, base_dir)
    provenance = ArtifactProvenance(
        version=_optional_str(spec, "version"),
        revision=_optional_str(spec, "revision"),
        sha256=_optional_str(spec, "sha256"),
        license=_optional_str(spec, "license"),
        source=_optional_str(spec, "source"),
    )
    metadata = _metadata(spec)

    common = {
        "kind": kind,
        "name": name,
        "location": location,
        "provenance": provenance,
        "metadata": metadata,
        "source_span": source_span,
    }
    if kind is ArtifactKind.TOKENIZER:
        return TokenizerArtifact(
            **common,
            family=_optional_str(spec, "family"),
            added_tokens=_tuple_of_str(spec, "added_tokens"),
        )
    if kind is ArtifactKind.CHAT_TEMPLATE:
        return ChatTemplateArtifact(
            **common,
            template_format=_str(spec, "template_format", default="jinja"),
            roles=_tuple_of_str(spec, "roles"),
            add_generation_prompt=_optional_bool(spec, "add_generation_prompt"),
        )
    if kind is ArtifactKind.SPECIAL_TOKEN_MAP:
        return SpecialTokenMapArtifact(**common, tokens=_special_tokens(spec))
    if kind is ArtifactKind.STOP_POLICY:
        return StopPolicyArtifact(
            **common,
            stop_sequences=_tuple_of_str(spec, "stop_sequences"),
            stop_token_ids=_tuple_of_int(spec, "stop_token_ids"),
            include_eos=_bool(spec, "include_eos", default=True),
            source_family=_optional_str(spec, "source_family"),
        )
    if kind is ArtifactKind.SCHEMA:
        return SchemaArtifact(**common, dialect=_str(spec, "dialect", default="json-schema"))
    if kind is ArtifactKind.GRAMMAR:
        return GrammarArtifact(
            **common,
            grammar_type=_str(spec, "grammar_type", default="promptabi"),
            start_symbol=_optional_str(spec, "start_symbol"),
            rule_names=_tuple_of_str(spec, "rule_names"),
            supported_fragment=_optional_bool(spec, "supported_fragment"),
        )
    if kind is ArtifactKind.TOOL_DEFINITION:
        return ToolDefinitionArtifact(
            **common,
            provider=_optional_str(spec, "provider"),
            tool_names=_tuple_of_str(spec, "tool_names"),
        )
    if kind is ArtifactKind.PROMPT_SEGMENT:
        return PromptSegmentArtifact(**common, segments=_prompt_segments(spec))
    if kind is ArtifactKind.PROVIDER_CONFIG:
        return ProviderConfigArtifact(
            **common,
            provider=_str(spec, "provider"),
            api_family=_optional_str(spec, "api_family"),
        )
    if kind is ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG:
        return FrameworkTruncationConfigArtifact(
            **common,
            framework=_str(spec, "framework"),
            strategy=TruncationStrategy(_str(spec, "strategy", default=TruncationStrategy.NONE.value)),
            max_context_tokens=_optional_int(spec, "max_context_tokens"),
            reserve_output_tokens=_int(spec, "reserve_output_tokens", default=0),
            reserved_tool_tokens=_int(spec, "reserved_tool_tokens", default=0),
            generation_prompt_tokens=_int(spec, "generation_prompt_tokens", default=0),
            special_token_overhead=_int(spec, "special_token_overhead", default=0),
            model=_optional_str(spec, "model"),
            preserve_system=_bool(spec, "preserve_system", default=False),
            preserve_tools=_bool(spec, "preserve_tools", default=False),
            drop_roles=_tuple_of_str(spec, "drop_roles"),
        )
    if kind is ArtifactKind.TRAINING_MANIFEST:
        return TrainingManifestArtifact(
            **common,
            dataset_format=_str(spec, "dataset_format", default="jsonl"),
            message_roles=_tuple_of_str(spec, "message_roles"),
            target_roles=_tuple_of_str(spec, "target_roles"),
            example_count=_optional_int(spec, "example_count"),
            packed=_bool(spec, "packed", default=False),
            datasets=_training_datasets(spec),
            system_message_policy=_system_message_policy(spec),
            role_labels=_role_labels(spec),
            supervised_spans=_training_span_contracts(spec),
            preference_pairs=_preference_pair_contracts(spec),
            loss_mask_policy=_loss_mask_policy(spec),
            packing_window=_packing_window(spec),
            chat_template_version=_chat_template_version(spec),
            pipeline_stages=_training_pipeline_stages(spec),
            redaction_policy=_training_redaction_policy(spec),
            synthetic_generators=_synthetic_generators(spec),
        )
    if kind is ArtifactKind.EVALUATION_HARNESS:
        return EvaluationHarnessArtifact(
            **common,
            benchmark_name=_str(spec, "benchmark_name", default="evaluation"),
            model=_optional_str(spec, "model"),
            provider=_optional_str(spec, "provider"),
            tokenizer=_optional_str(spec, "tokenizer"),
            prompt_template=_optional_str(spec, "prompt_template"),
            answer_parser=_optional_str(spec, "answer_parser"),
            answer_schema=_optional_str(spec, "answer_schema"),
            stop_sequences=_tuple_of_str(spec, "stop_sequences"),
            allowed_roles=_tuple_of_str(spec, "allowed_roles"),
            required_prompt_variables=_tuple_of_str(spec, "required_prompt_variables"),
            prompt_variables=_tuple_of_str(spec, "prompt_variables"),
            few_shot_examples=_evaluation_few_shot_examples(spec),
            max_prompt_tokens=_optional_int(spec, "max_prompt_tokens"),
        )
    raise AssertionError(f"unhandled artifact kind: {kind}")


def artifact_from_cli_override(
    name: str,
    value: str,
    *,
    base_dir: Path,
    existing: Artifact | None = None,
) -> Artifact:
    """Build or relocate an artifact from a ``NAME=PATH_OR_URI`` CLI override."""

    if not name:
        raise ValueError("artifact override names must be non-empty")
    if not value:
        raise ValueError(f"artifact override '{name}' must point to a path or URI")
    location = _location_from_uri_or_path(value, base_dir)
    if existing is None:
        return SchemaArtifact(kind=ArtifactKind.SCHEMA, name=name, location=location)
    return replace(existing, location=location)


def local_artifact_paths(bundle: ArtifactBundle) -> dict[str, str]:
    return {
        artifact.name: artifact.location.path
        for artifact in bundle
        if artifact.location.path is not None
    }


def _require_kind(actual: ArtifactKind, expected: ArtifactKind) -> None:
    if actual != expected:
        raise ValueError(f"expected artifact kind {expected.value}, got {actual.value}")


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _optional_non_empty(field_name: str, value: str | None) -> None:
    if value is not None and not value:
        raise ValueError(f"{field_name} must be non-empty")


def _optional_non_negative(field_name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _unique_strings(values, *, field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} values must be non-empty strings")
        result.append(value)
    return tuple(sorted(dict.fromkeys(result)))


def _strings_preserve_order(values, *, field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} values must be non-empty strings")
        result.append(value)
    return tuple(result)


def _location_from_spec(name: str, spec: dict[str, Any], base_dir: Path) -> ArtifactLocation:
    path = spec.get("path")
    uri = spec.get("uri")
    if path is not None and not isinstance(path, str):
        raise ValueError(f"artifact '{name}' field 'path' must be a string")
    if uri is not None and not isinstance(uri, str):
        raise ValueError(f"artifact '{name}' field 'uri' must be a string")
    if path is not None:
        return _location_from_path(path, base_dir)
    if uri is not None:
        return ArtifactLocation(uri=uri)
    raise ValueError(f"artifact '{name}' must define either 'path' or 'uri'")


def _location_from_path(path: str, base_dir: Path) -> ArtifactLocation:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    return ArtifactLocation(path=str(raw_path.resolve()))


def _location_from_uri_or_path(value: str, base_dir: Path) -> ArtifactLocation:
    if "://" in value:
        return ArtifactLocation(uri=value)
    return _location_from_path(value, base_dir)


def _optional_str(spec: dict[str, Any], key: str) -> str | None:
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact field '{key}' must be a non-empty string")
    return value


def _str(spec: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = spec.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact field '{key}' must be a non-empty string")
    return value


def _optional_bool(spec: dict[str, Any], key: str) -> bool | None:
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"artifact field '{key}' must be a boolean")
    return value


def _bool(spec: dict[str, Any], key: str, *, default: bool) -> bool:
    value = spec.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"artifact field '{key}' must be a boolean")
    return value


def _optional_int(spec: dict[str, Any], key: str) -> int | None:
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"artifact field '{key}' must be an integer")
    return value


def _int(spec: dict[str, Any], key: str, *, default: int) -> int:
    value = spec.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"artifact field '{key}' must be an integer")
    return value


def _tuple_of_str(spec: dict[str, Any], key: str) -> tuple[str, ...]:
    value = spec.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"artifact field '{key}' must be a list of non-empty strings")
    return tuple(value)


def _tuple_of_int(spec: dict[str, Any], key: str) -> tuple[int, ...]:
    value = spec.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"artifact field '{key}' must be a list of integers")
    return tuple(value)


def _metadata(spec: dict[str, Any]) -> tuple[tuple[str, object], ...]:
    value = spec.get("metadata", {})
    if not isinstance(value, dict) or not all(isinstance(key, str) and key for key in value):
        raise ValueError("artifact field 'metadata' must be an object with string keys")
    return tuple(sorted(value.items()))


def _special_tokens(spec: dict[str, Any]) -> tuple[SpecialToken, ...]:
    raw_tokens = spec.get("tokens", [])
    if isinstance(raw_tokens, dict):
        tokens: list[SpecialToken] = []
        for name, text in sorted(raw_tokens.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("special token map names must be non-empty strings")
            if not isinstance(text, str) or not text:
                raise ValueError("special token map values must be non-empty strings")
            tokens.append(SpecialToken(name=name, text=text))
        return tuple(tokens)
    if not isinstance(raw_tokens, list):
        raise ValueError("artifact field 'tokens' must be an object or list")
    tokens: list[SpecialToken] = []
    for item in raw_tokens:
        if not isinstance(item, dict):
            raise ValueError("special token entries must be objects")
        tokens.append(
            SpecialToken(
                name=_str(item, "name"),
                text=_str(item, "text"),
                token_id=_optional_int(item, "token_id"),
            )
        )
    return tuple(tokens)


def _prompt_segments(spec: dict[str, Any]) -> tuple[PromptSegment, ...]:
    raw_segments = spec.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("artifact field 'segments' must be a list")
    segments: list[PromptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise ValueError("prompt segment entries must be objects")
        required = _bool(item, "required", default=False)
        must_survive = _optional_bool(item, "must_survive")
        if must_survive is not None:
            if "required" in item and required != must_survive:
                raise ValueError("prompt segment fields 'required' and 'must_survive' must agree")
            required = must_survive
        segments.append(
            PromptSegment(
                name=_str(item, "name"),
                role=_optional_str(item, "role"),
                required=required,
                max_tokens=_optional_int(item, "max_tokens"),
                token_count=_optional_int(item, "token_count"),
                content=_optional_text(item, "content"),
                overhead_tokens=_int(item, "overhead_tokens", default=0),
                chunk_id=_optional_str(item, "chunk_id"),
                document_id=_optional_str(item, "document_id"),
                chunk_tokenizer=_optional_str(item, "chunk_tokenizer") or _optional_str(item, "tokenizer"),
                source_start=_optional_int(item, "source_start"),
                source_end=_optional_int(item, "source_end"),
                chunk_start=_optional_int(item, "chunk_start"),
                chunk_end=_optional_int(item, "chunk_end"),
                expected_overlap_tokens=_optional_int(item, "expected_overlap_tokens"),
                actual_overlap_tokens=_optional_int(item, "actual_overlap_tokens")
                if "actual_overlap_tokens" in item
                else _optional_int(item, "overlap_tokens"),
                citation=_optional_str(item, "citation"),
                citation_required=_bool(item, "citation_required", default=False),
                metadata_tokens=_int(item, "metadata_tokens", default=0),
                template_overhead_tokens=_int(item, "template_overhead_tokens", default=0),
                retrieval_payload_limit_tokens=_optional_int(item, "retrieval_payload_limit_tokens"),
            )
        )
    return tuple(segments)


def _training_datasets(spec: dict[str, Any]) -> tuple[TrainingDatasetSpec, ...]:
    raw_datasets = spec.get("datasets", [])
    if not isinstance(raw_datasets, list):
        raise ValueError("artifact field 'datasets' must be a list")
    datasets: list[TrainingDatasetSpec] = []
    for item in raw_datasets:
        if not isinstance(item, dict):
            raise ValueError("training dataset entries must be objects")
        datasets.append(
            TrainingDatasetSpec(
                name=_str(item, "name"),
                kind=TrainingDatasetKind(_str(item, "kind", default=TrainingDatasetKind.SUPERVISED.value)),
                path=_optional_str(item, "path"),
                split=_optional_str(item, "split"),
                format=_str(item, "format", default=_str(spec, "dataset_format", default="chat-jsonl")),
                example_count=_optional_int(item, "example_count"),
                content_fields=_tuple_of_str(item, "content_fields"),
                preference_fields=_tuple_of_str(item, "preference_fields"),
            )
        )
    return tuple(datasets)


def _system_message_policy(spec: dict[str, Any]) -> SystemMessagePolicy | None:
    raw_policy = spec.get("system_message_policy", spec.get("system_policy"))
    if raw_policy is None:
        return None
    if not isinstance(raw_policy, dict):
        raise ValueError("artifact field 'system_message_policy' must be an object")
    return SystemMessagePolicy(
        required=_bool(raw_policy, "required", default=False),
        allow_override=_bool(raw_policy, "allow_override", default=False),
        default=_optional_text(raw_policy, "default"),
        allowed_hashes=_tuple_of_str(raw_policy, "allowed_hashes"),
        max_tokens=_optional_int(raw_policy, "max_tokens"),
    )


def _role_labels(spec: dict[str, Any]) -> tuple[RoleLabel, ...]:
    raw_labels = spec.get("role_labels", [])
    if not isinstance(raw_labels, list):
        raise ValueError("artifact field 'role_labels' must be a list")
    labels: list[RoleLabel] = []
    for item in raw_labels:
        if not isinstance(item, dict):
            raise ValueError("role label entries must be objects")
        labels.append(
            RoleLabel(
                source_role=_str(item, "source_role"),
                canonical_role=_str(item, "canonical_role"),
                supervised_target=_bool(item, "supervised_target", default=False),
                trainable=_bool(item, "trainable", default=True),
                required=_bool(item, "required", default=False),
            )
        )
    return tuple(labels)


def _training_span_contracts(spec: dict[str, Any]) -> tuple[TrainingSpanContract, ...]:
    raw_spans = spec.get("supervised_spans", spec.get("target_spans", []))
    if not isinstance(raw_spans, list):
        raise ValueError("artifact field 'supervised_spans' must be a list")
    spans: list[TrainingSpanContract] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            raise ValueError("supervised span entries must be objects")
        spans.append(
            TrainingSpanContract(
                span_id=_str(item, "span_id", default=_str(item, "name") if "name" in item else None),
                target_role=_str(item, "target_role"),
                rendered_region_role=_str(item, "rendered_region_role"),
                start_token=_int(item, "start_token", default=0),
                end_token=_int(item, "end_token", default=0),
                region_start_token=_int(item, "region_start_token", default=0),
                region_end_token=_int(item, "region_end_token", default=0),
                supervised_target=_bool(item, "supervised_target", default=True),
                loss_masked=_bool(item, "loss_masked", default=True),
                packed_example_id=_optional_str(item, "packed_example_id"),
                crosses_packing_boundary=_bool(item, "crosses_packing_boundary", default=False),
                source_contributions=_training_source_contributions(item),
            )
        )
    return tuple(spans)


def _training_source_contributions(spec: dict[str, Any]) -> tuple[TrainingSourceContribution, ...]:
    raw_contributions = spec.get("source_contributions", spec.get("source_segments", []))
    if not isinstance(raw_contributions, list):
        raise ValueError("training span field 'source_contributions' must be a list")
    contributions: list[TrainingSourceContribution] = []
    for item in raw_contributions:
        if not isinstance(item, dict):
            raise ValueError("training source contribution entries must be objects")
        contributions.append(
            TrainingSourceContribution(
                source_id=_str(item, "source_id", default=_str(item, "id") if "id" in item else None),
                source_kind=TrainingTextSourceKind(_str(item, "source_kind", default=_str(item, "kind") if "kind" in item else None)),
                start_token=_int(item, "start_token", default=0),
                end_token=_int(item, "end_token", default=0),
                transform=_str(item, "transform", default=_str(item, "transform_name") if "transform_name" in item else "unknown-transform"),
                source_field=_optional_str(item, "source_field"),
                text_sha256=_optional_str(item, "text_sha256"),
            )
        )
    return tuple(contributions)


def _synthetic_generators(spec: dict[str, Any]) -> tuple[SyntheticGeneratorSpec, ...]:
    raw_generators = spec.get("synthetic_generators", spec.get("synthetic_data_generators", []))
    if not isinstance(raw_generators, list):
        raise ValueError("artifact field 'synthetic_generators' must be a list")
    generators: list[SyntheticGeneratorSpec] = []
    for item in raw_generators:
        if not isinstance(item, dict):
            raise ValueError("synthetic generator entries must be objects")
        generators.append(
            SyntheticGeneratorSpec(
                name=_str(item, "name"),
                generator_type=_str(item, "generator_type", default=_str(item, "type") if "type" in item else "synthetic-chat"),
                output_roles=_tuple_of_str(item, "output_roles"),
                required_roles=_tuple_of_str(item, "required_roles"),
                forbidden_roles=_tuple_of_str(item, "forbidden_roles"),
                max_prompt_tokens=_optional_int(item, "max_prompt_tokens"),
                max_completion_tokens=_optional_int(item, "max_completion_tokens"),
                schema_outputs=_synthetic_schema_outputs(item),
                tool_calls=_synthetic_tool_calls(item),
                truncation_cases=_synthetic_truncation_cases(item),
                metadata=_metadata(item),
            )
        )
    return tuple(generators)


def _evaluation_few_shot_examples(spec: dict[str, Any]) -> tuple[EvaluationFewShotExample, ...]:
    raw_examples = spec.get("few_shot_examples", spec.get("few_shots", []))
    if not isinstance(raw_examples, list):
        raise ValueError("artifact field 'few_shot_examples' must be a list")
    examples: list[EvaluationFewShotExample] = []
    for index, item in enumerate(raw_examples):
        if not isinstance(item, dict):
            raise ValueError("evaluation few-shot entries must be objects")
        examples.append(
            EvaluationFewShotExample(
                example_id=_case_id(item, index),
                role=_str(item, "role"),
                content=_optional_text(item, "content") or "",
                token_count=_optional_int(item, "token_count"),
            )
        )
    return tuple(examples)


def _synthetic_schema_outputs(spec: dict[str, Any]) -> tuple[SyntheticSchemaOutputContract, ...]:
    raw_cases = spec.get("schema_outputs", spec.get("json_outputs", []))
    if not isinstance(raw_cases, list):
        raise ValueError("synthetic generator field 'schema_outputs' must be a list")
    cases: list[SyntheticSchemaOutputContract] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError("synthetic schema output entries must be objects")
        cases.append(
            SyntheticSchemaOutputContract(
                case_id=_case_id(item, index),
                valid=_optional_bool(item, "valid"),
                parses=_optional_bool(item, "parses"),
                schema_valid=_optional_bool(item, "schema_valid"),
                reason=_optional_str(item, "reason") or _optional_str(item, "parser_error") or _optional_str(item, "schema_error"),
            )
        )
    return tuple(cases)


def _synthetic_tool_calls(spec: dict[str, Any]) -> tuple[SyntheticToolCallContract, ...]:
    raw_cases = spec.get("tool_calls", spec.get("tool_call_outputs", []))
    if not isinstance(raw_cases, list):
        raise ValueError("synthetic generator field 'tool_calls' must be a list")
    cases: list[SyntheticToolCallContract] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError("synthetic tool-call entries must be objects")
        cases.append(
            SyntheticToolCallContract(
                case_id=_case_id(item, index),
                valid=_optional_bool(item, "valid"),
                malformed=_optional_bool(item, "malformed"),
                reason=_optional_str(item, "reason") or _optional_str(item, "error"),
            )
        )
    return tuple(cases)


def _synthetic_truncation_cases(spec: dict[str, Any]) -> tuple[SyntheticTruncationContract, ...]:
    raw_cases = spec.get("truncation_cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("synthetic generator field 'truncation_cases' must be a list")
    cases: list[SyntheticTruncationContract] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError("synthetic truncation entries must be objects")
        cases.append(
            SyntheticTruncationContract(
                case_id=_case_id(item, index),
                input_tokens=_optional_int(item, "input_tokens"),
                output_tokens=_optional_int(item, "output_tokens"),
                max_context_tokens=_optional_int(item, "max_context_tokens"),
                preserved_required_roles=_tuple_of_str(item, "preserved_required_roles"),
                truncated_required_roles=_tuple_of_str(item, "truncated_required_roles"),
                reason=_optional_str(item, "reason"),
            )
        )
    return tuple(cases)


def _case_id(spec: dict[str, Any], index: int) -> str:
    return (
        _optional_str(spec, "id")
        or _optional_str(spec, "case_id")
        or _optional_str(spec, "example_id")
        or f"{index}"
    )


def _preference_pair_contracts(spec: dict[str, Any]) -> tuple[PreferencePairContract, ...]:
    raw_pairs = spec.get("preference_pairs", spec.get("preference_pair_contracts", []))
    if not isinstance(raw_pairs, list):
        raise ValueError("artifact field 'preference_pairs' must be a list")
    pairs: list[PreferencePairContract] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise ValueError("preference pair entries must be objects")
        pairs.append(
            PreferencePairContract(
                pair_id=_str(item, "pair_id", default=_str(item, "id") if "id" in item else None),
                prompt_sha256=_str(item, "prompt_sha256"),
                chosen_sha256=_str(item, "chosen_sha256"),
                rejected_sha256=_str(item, "rejected_sha256"),
                chosen_role_layout=_tuple_of_str(item, "chosen_role_layout"),
                rejected_role_layout=_tuple_of_str(item, "rejected_role_layout"),
                chosen_tokenizer=_str(item, "chosen_tokenizer"),
                rejected_tokenizer=_str(item, "rejected_tokenizer"),
                chosen_mask_policy=_str(item, "chosen_mask_policy"),
                rejected_mask_policy=_str(item, "rejected_mask_policy"),
                chosen_prompt_tokens=_int(item, "chosen_prompt_tokens", default=0),
                rejected_prompt_tokens=_int(item, "rejected_prompt_tokens", default=0),
                chosen_response_start_token=_int(item, "chosen_response_start_token", default=0),
                rejected_response_start_token=_int(item, "rejected_response_start_token", default=0),
                chosen_response_end_token=_int(item, "chosen_response_end_token", default=0),
                rejected_response_end_token=_int(item, "rejected_response_end_token", default=0),
                chosen_prompt_sha256=_optional_str(item, "chosen_prompt_sha256"),
                rejected_prompt_sha256=_optional_str(item, "rejected_prompt_sha256"),
                chosen_truncated=_bool(item, "chosen_truncated", default=False),
                rejected_truncated=_bool(item, "rejected_truncated", default=False),
                chosen_packed_example_id=_optional_str(item, "chosen_packed_example_id"),
                rejected_packed_example_id=_optional_str(item, "rejected_packed_example_id"),
            )
        )
    return tuple(pairs)


def _loss_mask_policy(spec: dict[str, Any]) -> LossMaskPolicy | None:
    raw_policy = spec.get("loss_mask_policy", spec.get("loss_mask"))
    if raw_policy is None:
        if "target_roles" not in spec:
            return None
        return LossMaskPolicy(target_roles=_tuple_of_str(spec, "target_roles"))
    if not isinstance(raw_policy, dict):
        raise ValueError("artifact field 'loss_mask_policy' must be an object")
    return LossMaskPolicy(
        strategy=LossMaskStrategy(_str(raw_policy, "strategy", default=LossMaskStrategy.ASSISTANT_ONLY.value)),
        target_roles=_tuple_of_str(raw_policy, "target_roles"),
        ignored_roles=_tuple_of_str(raw_policy, "ignored_roles"),
        explicit_mask_field=_optional_str(raw_policy, "explicit_mask_field"),
        label_pad_token_id=_int(raw_policy, "label_pad_token_id", default=-100),
    )


def _packing_window(spec: dict[str, Any]) -> PackingWindow | None:
    raw_window = spec.get("packing_window", spec.get("packing"))
    if raw_window is None:
        return None
    if not isinstance(raw_window, dict):
        raise ValueError("artifact field 'packing_window' must be an object")
    return PackingWindow(
        strategy=PackingStrategy(_str(raw_window, "strategy", default=PackingStrategy.NONE.value)),
        max_tokens=_optional_int(raw_window, "max_tokens"),
        stride_tokens=_int(raw_window, "stride_tokens", default=0),
        boundary_token=_optional_str(raw_window, "boundary_token"),
        preserve_example_boundaries=_bool(raw_window, "preserve_example_boundaries", default=True),
        reset_position_ids=_bool(raw_window, "reset_position_ids", default=True),
    )


def _chat_template_version(spec: dict[str, Any]) -> ChatTemplateVersion | None:
    raw_version = spec.get("chat_template_version", spec.get("chat_template"))
    if raw_version is None:
        return None
    if not isinstance(raw_version, dict):
        raise ValueError("artifact field 'chat_template_version' must be an object")
    return ChatTemplateVersion(
        name=_str(raw_version, "name"),
        version=_optional_str(raw_version, "version"),
        revision=_optional_str(raw_version, "revision"),
        sha256=_optional_str(raw_version, "sha256"),
        tokenizer_name=_optional_str(raw_version, "tokenizer_name"),
        add_generation_prompt=_optional_bool(raw_version, "add_generation_prompt"),
    )


def _training_pipeline_stages(spec: dict[str, Any]) -> tuple[TrainingPipelineStageVersion, ...]:
    raw_stages = spec.get("pipeline_stages", spec.get("stage_versions", []))
    if not isinstance(raw_stages, list):
        raise ValueError("artifact field 'pipeline_stages' must be a list")
    stages: list[TrainingPipelineStageVersion] = []
    for item in raw_stages:
        if not isinstance(item, dict):
            raise ValueError("training pipeline stage entries must be objects")
        stages.append(
            TrainingPipelineStageVersion(
                stage=_str(item, "stage"),
                tokenizer_name=_optional_str(item, "tokenizer_name"),
                tokenizer_version=_optional_str(item, "tokenizer_version"),
                tokenizer_revision=_optional_str(item, "tokenizer_revision"),
                tokenizer_sha256=_optional_str(item, "tokenizer_sha256"),
                chat_template_name=_optional_str(item, "chat_template_name"),
                chat_template_version=_optional_str(item, "chat_template_version"),
                chat_template_revision=_optional_str(item, "chat_template_revision"),
                chat_template_sha256=_optional_str(item, "chat_template_sha256"),
                add_generation_prompt=_optional_bool(item, "add_generation_prompt"),
            )
        )
    return tuple(stages)


def _training_redaction_policy(spec: dict[str, Any]) -> TrainingRedactionPolicy | None:
    raw_policy = spec.get("redaction_policy", spec.get("witness_redaction"))
    if raw_policy is None:
        return None
    if not isinstance(raw_policy, dict):
        raise ValueError("artifact field 'redaction_policy' must be an object")
    return TrainingRedactionPolicy(
        mode=TrainingRedactionMode(_str(raw_policy, "mode", default=TrainingRedactionMode.HASH_ONLY.value)),
        require_text_hashes=_bool(raw_policy, "require_text_hashes", default=True),
        allow_raw_text_in_witnesses=_bool(raw_policy, "allow_raw_text_in_witnesses", default=False),
        allowed_report_fields=_tuple_of_str(raw_policy, "allowed_report_fields"),
        forbidden_report_fields=_tuple_of_str(raw_policy, "forbidden_report_fields"),
        restricted_metadata_keys=_tuple_of_str(raw_policy, "restricted_metadata_keys"),
        secret_patterns=_tuple_of_str(raw_policy, "secret_patterns"),
    )


def _optional_text(spec: dict[str, Any], key: str) -> str | None:
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"artifact field '{key}' must be a string")
    return value
