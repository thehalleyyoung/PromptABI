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


class TruncationStrategy(StrEnum):
    """Framework prompt-budget strategies represented before semantic checks exist."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    OLDEST_MESSAGE = "oldest-message"
    MIDDLE = "middle"
    CUSTOM = "custom"


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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("prompt segment name must be non-empty")
        if self.role is not None and not self.role:
            raise ValueError("prompt segment role must be non-empty")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("prompt segment max_tokens must be positive")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "required": self.required}
        if self.role is not None:
            data["role"] = self.role
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
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
        data = super().to_dict()
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
        data = super().to_dict()
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
        data = super().to_dict()
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
        data = super().to_dict()
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
        data = super().to_dict()
        data["dialect"] = self.dialect
        return data


@dataclass(frozen=True, slots=True)
class GrammarArtifact(BaseArtifact):
    grammar_type: str = "promptabi"

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.GRAMMAR)
        if not self.grammar_type:
            raise ValueError("grammar type must be non-empty")

    def to_dict(self) -> dict[str, object]:
        data = super().to_dict()
        data["grammar_type"] = self.grammar_type
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
        data = super().to_dict()
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
        object.__setattr__(self, "segments", tuple(sorted(self.segments, key=lambda segment: segment.name)))

    @property
    def required_segments(self) -> tuple[PromptSegment, ...]:
        return tuple(segment for segment in self.segments if segment.required)

    def to_dict(self) -> dict[str, object]:
        data = super().to_dict()
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
        data = super().to_dict()
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

    def __post_init__(self) -> None:
        BaseArtifact.__post_init__(self)
        _require_kind(self.kind, ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG)
        if not self.framework:
            raise ValueError("framework name must be non-empty")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.reserve_output_tokens < 0:
            raise ValueError("reserve_output_tokens must be non-negative")

    def to_dict(self) -> dict[str, object]:
        data = super().to_dict()
        data["framework"] = self.framework
        data["strategy"] = self.strategy.value
        if self.max_context_tokens is not None:
            data["max_context_tokens"] = self.max_context_tokens
        if self.reserve_output_tokens:
            data["reserve_output_tokens"] = self.reserve_output_tokens
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
        return GrammarArtifact(**common, grammar_type=_str(spec, "grammar_type", default="promptabi"))
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
        segments.append(
            PromptSegment(
                name=_str(item, "name"),
                role=_optional_str(item, "role"),
                required=_bool(item, "required", default=False),
                max_tokens=_optional_int(item, "max_tokens"),
            )
        )
    return tuple(segments)
