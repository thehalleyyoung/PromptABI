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
    loss_mask_policy: LossMaskPolicy | None = None
    packing_window: PackingWindow | None = None
    chat_template_version: ChatTemplateVersion | None = None

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
        if self.loss_mask_policy is not None:
            data["loss_mask_policy"] = self.loss_mask_policy.to_dict()
        if self.packing_window is not None:
            data["packing_window"] = self.packing_window.to_dict()
        if self.chat_template_version is not None:
            data["chat_template_version"] = self.chat_template_version.to_dict()
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
            loss_mask_policy=_loss_mask_policy(spec),
            packing_window=_packing_window(spec),
            chat_template_version=_chat_template_version(spec),
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


def _optional_text(spec: dict[str, Any], key: str) -> str | None:
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"artifact field '{key}' must be a string")
    return value
