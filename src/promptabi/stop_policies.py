"""Stop-policy parsing across provider and framework config shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .diagnostics import SourceSpan
from .source import JsonSourceMap


STOP_STRING_KEYS = frozenset(
    {
        "stop",
        "stops",
        "stop_sequences",
        "stop_strings",
        "stopping_strings",
        "reverse_prompt",
        "reverse_prompts",
        "antiprompt",
        "anti_prompt",
    }
)
STOP_TOKEN_ID_KEYS = frozenset({"stop_token_id", "stop_token_ids"})
EOS_TOKEN_KEYS = frozenset({"eos_token_id", "eos_token_ids", "forced_eos_token_id"})
WRAPPER_KEYS = frozenset(
    {
        "completion_kwargs",
        "config",
        "default_options",
        "extra_body",
        "generation_config",
        "generation_kwargs",
        "invocation_params",
        "litellm_params",
        "llm_kwargs",
        "model_config",
        "model_kwargs",
        "options",
        "parameters",
        "params",
        "request",
        "sampling_params",
    }
)


class StopPolicyParseError(ValueError):
    """Raised when a stop-policy config has a supported key with invalid shape."""

    def __init__(
        self,
        message: str,
        *,
        path: tuple[str, ...] = (),
        span: SourceSpan | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.span = span


@dataclass(frozen=True, slots=True)
class StopPolicySource:
    """One concrete source location that contributed stop-policy semantics."""

    family: str
    path: tuple[str, ...]
    key: str
    stop_sequences: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    include_eos: bool | None = None
    span: SourceSpan | None = None

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path) or "<root>"

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        values: list[tuple[str, object]] = [
            ("family", self.family),
            ("key", self.key),
            ("path", self.dotted_path),
        ]
        if self.stop_sequences:
            values.append(("stop_sequences", self.stop_sequences))
        if self.stop_token_ids:
            values.append(("stop_token_ids", self.stop_token_ids))
        if self.include_eos is not None:
            values.append(("include_eos", self.include_eos))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class StopPolicyParseResult:
    """Normalized stop policy extracted from provider/framework configuration."""

    source_family: str
    stop_sequences: tuple[str, ...]
    stop_token_ids: tuple[int, ...]
    include_eos: bool
    sources: tuple[StopPolicySource, ...]
    ignored_empty_sequences: tuple[str, ...] = ()

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("include_eos", self.include_eos),
            ("ignored_empty_sequences", self.ignored_empty_sequences),
            ("source_family", self.source_family),
            ("source_paths", tuple(source.dotted_path for source in self.sources)),
            ("stop_sequence_count", len(self.stop_sequences)),
            ("stop_sequences", self.stop_sequences),
            ("stop_token_ids", self.stop_token_ids),
            ("stop_token_id_count", len(self.stop_token_ids)),
        )


def parse_stop_policy_config(
    data: Mapping[str, Any],
    *,
    source_map: JsonSourceMap | None = None,
    declared_family: str | None = None,
) -> StopPolicyParseResult:
    """Parse common stop-policy config surfaces into a deterministic model.

    The parser is intentionally static and offline. It accepts OpenAI-compatible
    request snapshots, Hugging Face generation configs, llama.cpp/Ollama-style
    request options, vLLM sampling params, LiteLLM passthrough params, and common
    framework wrapper dictionaries that nest those shapes under kwargs-like keys.
    """

    if not isinstance(data, Mapping):
        raise StopPolicyParseError("stop-policy config must be a JSON object")

    sources: list[StopPolicySource] = []
    ignored_empty: list[str] = []
    visit_roots = [(data, (), _family_from_mapping(data, declared_family=declared_family))]
    seen: set[int] = set()
    while visit_roots:
        current, path, family_hint = visit_roots.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        for key, value in sorted(current.items()):
            key_path = (*path, str(key))
            family = _family_for_key(key, key_path, current, family_hint)
            span = _span_for(source_map, key_path)
            if key in STOP_STRING_KEYS:
                strings, empty = _coerce_stop_strings(value, key_path, span)
                ignored_empty.extend(f"{'.'.join(key_path)}[{index}]" for index in empty)
                if strings:
                    sources.append(
                        StopPolicySource(
                            family=family,
                            path=key_path,
                            key=str(key),
                            stop_sequences=tuple(strings),
                            span=span,
                        )
                    )
            elif key in STOP_TOKEN_ID_KEYS:
                token_ids = _coerce_token_ids(value, key_path, span)
                if token_ids:
                    sources.append(
                        StopPolicySource(
                            family=family,
                            path=key_path,
                            key=str(key),
                            stop_token_ids=tuple(token_ids),
                            span=span,
                        )
                    )
            elif key in EOS_TOKEN_KEYS and value is not None:
                token_ids = _coerce_token_ids(value, key_path, span)
                sources.append(
                    StopPolicySource(
                        family=family,
                        path=key_path,
                        key=str(key),
                        stop_token_ids=tuple(token_ids),
                        include_eos=True,
                        span=span,
                    )
                )
            elif key == "ignore_eos":
                if not isinstance(value, bool):
                    raise StopPolicyParseError(
                        f"stop-policy field '{'.'.join(key_path)}' must be a boolean",
                        path=key_path,
                        span=span,
                    )
                sources.append(
                    StopPolicySource(
                        family=family,
                        path=key_path,
                        key=str(key),
                        include_eos=not value,
                        span=span,
                    )
                )
            elif key == "include_eos":
                if not isinstance(value, bool):
                    raise StopPolicyParseError(
                        f"stop-policy field '{'.'.join(key_path)}' must be a boolean",
                        path=key_path,
                        span=span,
                    )
                sources.append(
                    StopPolicySource(
                        family=family,
                        path=key_path,
                        key=str(key),
                        include_eos=value,
                        span=span,
                    )
                )

            if key in WRAPPER_KEYS and isinstance(value, Mapping):
                visit_roots.append((value, key_path, _family_for_wrapper(key, family_hint)))

    stop_sequences = tuple(sorted(dict.fromkeys(sequence for source in sources for sequence in source.stop_sequences)))
    stop_token_ids = tuple(sorted(dict.fromkeys(token_id for source in sources for token_id in source.stop_token_ids)))
    eos_values = tuple(source.include_eos for source in sources if source.include_eos is not None)
    include_eos = eos_values[-1] if eos_values else True
    family = _dominant_family(sources, declared_family=declared_family)
    return StopPolicyParseResult(
        source_family=family,
        stop_sequences=stop_sequences,
        stop_token_ids=stop_token_ids,
        include_eos=include_eos,
        sources=tuple(sources),
        ignored_empty_sequences=tuple(sorted(dict.fromkeys(ignored_empty))),
    )


def _coerce_stop_strings(
    value: Any,
    path: tuple[str, ...],
    span: SourceSpan | None,
) -> tuple[list[str], list[int]]:
    if isinstance(value, str):
        return ([value] if value else []), ([0] if not value else [])
    if isinstance(value, list):
        strings: list[str] = []
        empty: list[int] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise StopPolicyParseError(
                    f"stop-policy field '{'.'.join(path)}' must contain only strings",
                    path=(*path, str(index)),
                    span=span,
                )
            if item:
                strings.append(item)
            else:
                empty.append(index)
        return strings, empty
    if value is None:
        return [], []
    raise StopPolicyParseError(
        f"stop-policy field '{'.'.join(path)}' must be a string or list of strings",
        path=path,
        span=span,
    )


def _coerce_token_ids(value: Any, path: tuple[str, ...], span: SourceSpan | None) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise StopPolicyParseError(
                f"stop-policy field '{'.'.join(path)}' must contain non-negative token ids",
                path=path,
                span=span,
            )
        return (value,)
    if isinstance(value, list):
        token_ids: list[int] = []
        for index, item in enumerate(value):
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise StopPolicyParseError(
                    f"stop-policy field '{'.'.join(path)}' must contain non-negative integer token ids",
                    path=(*path, str(index)),
                    span=span,
                )
            token_ids.append(item)
        return tuple(token_ids)
    raise StopPolicyParseError(
        f"stop-policy field '{'.'.join(path)}' must be an integer or list of integers",
        path=path,
        span=span,
    )


def _family_from_mapping(data: Mapping[str, Any], *, declared_family: str | None) -> str:
    if declared_family:
        return declared_family
    provider = data.get("provider") or data.get("provider_name")
    framework = data.get("framework") or data.get("backend") or data.get("api_family")
    raw = f"{provider or ''} {framework or ''}".lower()
    if "litellm" in raw:
        return "litellm"
    if "vllm" in raw:
        return "vllm"
    if "llama" in raw or "ollama" in raw:
        return "llama.cpp"
    if "huggingface" in raw or "transformers" in raw or "generation" in raw:
        return "huggingface"
    if "openai" in raw or "responses" in raw or "chat-completions" in raw:
        return "openai-compatible"
    return "framework-wrapper"


def _family_for_key(
    key: object,
    path: tuple[str, ...],
    current: Mapping[str, Any],
    family_hint: str,
) -> str:
    if "litellm_params" in path:
        return "litellm"
    if "sampling_params" in path or key in STOP_TOKEN_ID_KEYS:
        return "vllm"
    if "generation_config" in path or key in {"stop_strings", *EOS_TOKEN_KEYS}:
        return "huggingface"
    if key in {"reverse_prompt", "reverse_prompts", "antiprompt", "anti_prompt", "ignore_eos"}:
        return "llama.cpp"
    return _family_from_mapping(current, declared_family=family_hint)


def _family_for_wrapper(key: object, family_hint: str) -> str:
    if key == "litellm_params":
        return "litellm"
    if key == "sampling_params":
        return "vllm"
    if key == "generation_config":
        return "huggingface"
    return family_hint


def _dominant_family(sources: list[StopPolicySource], *, declared_family: str | None) -> str:
    if declared_family:
        return declared_family
    if not sources:
        return "framework-wrapper"
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.family] = counts.get(source.family, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _span_for(source_map: JsonSourceMap | None, path: tuple[str, ...]) -> SourceSpan | None:
    if source_map is None:
        return None
    return source_map.span_for(path) or source_map.key_span_for(path)
