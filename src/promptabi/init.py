"""Project scaffolding for PromptABI configs and local fixture stubs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class InitError(ValueError):
    """Raised when a scaffold cannot be written safely."""


@dataclass(frozen=True, slots=True)
class ScaffoldFile:
    """A deterministic file emitted by ``promptabi init``."""

    path: str
    data: object

    @property
    def text(self) -> str:
        return _json_text(self.data)


@dataclass(frozen=True, slots=True)
class ScaffoldTemplate:
    """A named PromptABI scaffold for one application stack."""

    stack: str
    description: str
    checks: tuple[str, ...]
    max_context_tokens: int | None
    files: tuple[ScaffoldFile, ...]
    artifacts: Mapping[str, object]

    def config(self, name: str) -> dict[str, object]:
        config: dict[str, object] = {
            "name": name,
            "checks": list(self.checks),
            "artifacts": dict(self.artifacts),
        }
        if self.max_context_tokens is not None:
            config["max_context_tokens"] = self.max_context_tokens
        return config


def available_stacks() -> tuple[str, ...]:
    """Return supported scaffold stack names in CLI display order."""

    return tuple(SCAFFOLDS)


def scaffold_promptabi_project(
    *,
    stack: str,
    output_dir: str | Path,
    name: str | None = None,
    config_filename: str = "promptabi.json",
    force: bool = False,
) -> tuple[Path, ...]:
    """Write a self-contained PromptABI scaffold and return created paths."""

    template = _template(stack)
    target_dir = Path(output_dir).expanduser().resolve()
    if not config_filename or Path(config_filename).name != config_filename:
        raise InitError("config filename must be a single file name")
    project_name = name.strip() if name is not None else f"{template.stack}-promptabi"
    if not project_name:
        raise InitError("project name must be non-empty")

    writes: list[tuple[Path, str]] = [
        (target_dir / config_filename, _json_text(template.config(project_name))),
    ]
    writes.extend((target_dir / item.path, item.text) for item in template.files)
    collisions = [path for path, _text in writes if path.exists() and not force]
    if collisions:
        formatted = ", ".join(str(path) for path in collisions)
        raise InitError(f"refusing to overwrite existing file(s): {formatted}")

    for path, text in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tuple(path for path, _text in writes)


def _template(stack: str) -> ScaffoldTemplate:
    try:
        return SCAFFOLDS[stack]
    except KeyError as exc:
        allowed = ", ".join(available_stacks())
        raise InitError(f"unknown stack '{stack}' (expected one of {allowed})") from exc


def _json_text(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


_ANSWER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "citations": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
}

_OPENAI_TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup_policy",
                "description": "Look up a local policy document by slug.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["slug"],
                    "properties": {"slug": {"type": "string", "enum": ["refunds", "shipping"]}},
                },
            },
        }
    ]
}

_MESSAGES = [
    {"role": "system", "content": "Answer with JSON that matches the declared schema."},
    {"role": "user", "content": "Summarize the refund policy and cite sources."},
]

_MESSAGE_SEGMENTS = [
    {"name": "system-policy", "role": "system", "required": True, "content": _MESSAGES[0]["content"]},
    {"name": "user-request", "role": "user", "required": True, "content": _MESSAGES[1]["content"]},
]

_RAG_SEGMENTS = {
    "segments": [
        {"name": "system-policy", "role": "system", "required": True, "token_count": 24},
        {
            "name": "retrieval-chunk",
            "role": "retrieval",
            "required": True,
            "token_count": 64,
            "max_tokens": 96,
            "chunk_id": "refunds-0",
            "document_id": "refunds",
            "chunk_tokenizer": "serving-tokenizer",
            "source_start": 0,
            "source_end": 64,
            "chunk_start": 0,
            "chunk_end": 64,
            "expected_overlap_tokens": 8,
            "actual_overlap_tokens": 8,
            "citation": "refunds#0",
            "citation_required": True,
            "metadata_tokens": 4,
            "template_overhead_tokens": 3,
            "retrieval_payload_limit_tokens": 128,
        },
        {"name": "user-question", "role": "user", "required": True, "token_count": 18},
    ]
}

_BUDGET = {
    "framework": "langchain",
    "strategy": "priority",
    "preserve_system": True,
    "preserve_tools": True,
    "drop_roles": ["retrieval"],
    "max_context_tokens": 512,
    "reserve_output_tokens": 96,
    "reserved_tool_tokens": 32,
    "generation_prompt_tokens": 4,
    "special_token_overhead": 8,
}

def _provider_fixture(
    *,
    provider: str,
    provider_family: str,
    endpoint: str,
    supports_parallel_tool_calls: bool,
    emits_argument_fragments: bool,
) -> dict[str, object]:
    return {
        "provider": provider,
        "provider_family": provider_family,
        "request": {
            "method": "POST",
            "endpoint": endpoint,
            "fields": ["messages", "model", "tools", "tool_choice", "response_format", "stream", "stop"],
        },
        "response": {
            "fields": ["choices", "finish_reason", "message", "tool_calls", "usage"],
            "finish_reasons": ["stop", "tool_calls", "length"],
            "tool_calls": {
                "name_path": "choices[].message.tool_calls[].function.name",
                "arguments_path": "choices[].message.tool_calls[].function.arguments",
                "argument_encoding": "json-string",
                "id_path": "choices[].message.tool_calls[].id",
                "supports_parallel_tool_calls": supports_parallel_tool_calls,
            },
        },
        "stops": {
            "sequences": ["</tool_call>"],
            "finish_reason_path": "choices[].finish_reason",
            "truncates_before_parser": True,
        },
        "streaming": {
            "delta_path": "choices[].delta",
            "emits_argument_fragments": emits_argument_fragments,
            **({"assembly_key": "tool_calls[].index"} if emits_argument_fragments else {}),
        },
        "errors": {
            "code_path": "error.code",
            "message_path": "error.message",
            "rate_limit_path": "error.type",
            "sample": {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "redacted"}},
        },
        "limits": {
            "max_input_tokens": 8192,
            "max_output_tokens": 1024,
            "parallel_tool_call_limit": 8,
        },
        "edge_cases": [
            {
                "id": "tool-call-stop-boundary",
                "surface": "stops.finish_reason",
                "expected_behavior": "stop finish can terminate before an application parser is complete",
            }
        ],
    }


_OPENAI_PROVIDER = _provider_fixture(
    provider="openai",
    provider_family="openai",
    endpoint="/v1/chat/completions",
    supports_parallel_tool_calls=True,
    emits_argument_fragments=True,
)

_VLLM_PROVIDER = _provider_fixture(
    provider="vllm",
    provider_family="vllm-openai-server",
    endpoint="/v1/chat/completions",
    supports_parallel_tool_calls=True,
    emits_argument_fragments=True,
)

_LLAMA_CPP_PROVIDER = _provider_fixture(
    provider="llama.cpp",
    provider_family="llama.cpp-server",
    endpoint="/v1/chat/completions",
    supports_parallel_tool_calls=False,
    emits_argument_fragments=False,
)

_STOP_POLICY = {"stop": ["</tool_call>"], "include_eos": True}

_TOKENIZER_CONFIG = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "chat_template": (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    ),
}

_LLAMAINDEX_SEGMENTS = {
    "segments": [
        {"name": "system-policy", "role": "system", "required": True, "token_count": 20},
        {"name": "tool-catalog", "role": "tool", "required": True, "token_count": 72},
        {"name": "index-context", "role": "retrieval", "token_count": 96, "max_tokens": 128},
        {"name": "user-question", "role": "user", "required": True, "token_count": 16},
    ]
}

SCAFFOLDS: dict[str, ScaffoldTemplate] = {
    "openai-tools": ScaffoldTemplate(
        stack="openai-tools",
        description="OpenAI-compatible tool calls with JSON structured output.",
        checks=("repository-skeleton", "tool-schema-ingestion", "provider-fixture-replay"),
        max_context_tokens=4096,
        files=(
            ScaffoldFile("messages.json", _MESSAGES),
            ScaffoldFile("answer.schema.json", _ANSWER_SCHEMA),
            ScaffoldFile("tools.json", _OPENAI_TOOLS),
            ScaffoldFile("provider-openai.json", _OPENAI_PROVIDER),
        ),
        artifacts={
            "messages": {"kind": "prompt-segment", "path": "messages.json", "segments": _MESSAGE_SEGMENTS},
            "answer-schema": {"kind": "schema", "path": "answer.schema.json", "dialect": "json-schema-2020-12"},
            "tools": {"kind": "tool-definition", "path": "tools.json", "provider": "openai"},
            "provider-openai": {"kind": "provider-config", "path": "provider-openai.json", "provider": "openai"},
        },
    ),
    "huggingface-local": ScaffoldTemplate(
        stack="huggingface-local",
        description="Local Hugging Face tokenizer directory and chat template.",
        checks=("repository-skeleton", "role-boundary-nonforgeability"),
        max_context_tokens=8192,
        files=(ScaffoldFile("tokenizer/tokenizer_config.json", _TOKENIZER_CONFIG),),
        artifacts={
            "local-tokenizer": {"kind": "tokenizer", "path": "tokenizer", "family": "huggingface"},
            "chat-template": {
                "kind": "chat-template",
                "path": "tokenizer/tokenizer_config.json",
                "roles": ["system", "user", "assistant", "tool"],
            },
        },
    ),
    "vllm-openai": ScaffoldTemplate(
        stack="vllm-openai",
        description="vLLM OpenAI-compatible server with stop and provider fixtures.",
        checks=("repository-skeleton", "stop-differential", "provider-fixture-replay"),
        max_context_tokens=8192,
        files=(ScaffoldFile("provider-vllm.json", _VLLM_PROVIDER), ScaffoldFile("stop-policy.json", _STOP_POLICY)),
        artifacts={
            "provider-vllm": {"kind": "provider-config", "path": "provider-vllm.json", "provider": "vllm"},
            "stop-policy": {
                "kind": "stop-policy",
                "path": "stop-policy.json",
                "source_family": "vllm",
                "stop_sequences": ["</tool_call>"],
            },
        },
    ),
    "llama-cpp": ScaffoldTemplate(
        stack="llama-cpp",
        description="llama.cpp/OpenAI-compatible local server with explicit stops.",
        checks=("repository-skeleton", "stop-differential", "provider-fixture-replay"),
        max_context_tokens=4096,
        files=(ScaffoldFile("provider-llama-cpp.json", _LLAMA_CPP_PROVIDER), ScaffoldFile("stop-policy.json", _STOP_POLICY)),
        artifacts={
            "provider-llama-cpp": {
                "kind": "provider-config",
                "path": "provider-llama-cpp.json",
                "provider": "llama.cpp",
            },
            "stop-policy": {
                "kind": "stop-policy",
                "path": "stop-policy.json",
                "source_family": "llama.cpp",
                "stop_sequences": ["</tool_call>"],
            },
        },
    ),
    "langchain-rag": ScaffoldTemplate(
        stack="langchain-rag",
        description="LangChain-style RAG prompt segments and truncation budget.",
        checks=("repository-skeleton", "token-budget-model", "rag-chunking-compatibility"),
        max_context_tokens=512,
        files=(ScaffoldFile("segments.json", _RAG_SEGMENTS), ScaffoldFile("runtime-budget.json", _BUDGET)),
        artifacts={
            "segments": {"kind": "prompt-segment", "path": "segments.json", "segments": _RAG_SEGMENTS["segments"]},
            "runtime-budget": {
                "kind": "framework-truncation-config",
                "path": "runtime-budget.json",
                "framework": "langchain",
                "strategy": "priority",
                "max_context_tokens": 512,
                "preserve_system": True,
                "preserve_tools": True,
                "drop_roles": ["retrieval"],
            },
            "serving-tokenizer": {"kind": "tokenizer", "uri": "memory://serving-tokenizer", "family": "byte-level"},
        },
    ),
    "llamaindex-agent": ScaffoldTemplate(
        stack="llamaindex-agent",
        description="LlamaIndex agent prompt budget plus OpenAI-compatible tools.",
        checks=("repository-skeleton", "tool-schema-ingestion", "token-budget-model"),
        max_context_tokens=4096,
        files=(
            ScaffoldFile("agent-segments.json", _LLAMAINDEX_SEGMENTS),
            ScaffoldFile("runtime-budget.json", {**_BUDGET, "framework": "llamaindex"}),
            ScaffoldFile("tools.json", _OPENAI_TOOLS),
        ),
        artifacts={
            "agent-segments": {
                "kind": "prompt-segment",
                "path": "agent-segments.json",
                "segments": _LLAMAINDEX_SEGMENTS["segments"],
            },
            "runtime-budget": {
                "kind": "framework-truncation-config",
                "path": "runtime-budget.json",
                "framework": "llamaindex",
                "strategy": "priority",
                "max_context_tokens": 512,
                "preserve_system": True,
                "preserve_tools": True,
                "drop_roles": ["retrieval"],
            },
            "tools": {"kind": "tool-definition", "path": "tools.json", "provider": "openai"},
        },
    ),
    "json-schema": ScaffoldTemplate(
        stack="json-schema",
        description="Custom JSON Schema structured-output contract.",
        checks=("repository-skeleton", "parser-compatibility"),
        max_context_tokens=4096,
        files=(ScaffoldFile("answer.schema.json", _ANSWER_SCHEMA),),
        artifacts={
            "answer-schema": {"kind": "schema", "path": "answer.schema.json", "dialect": "json-schema-2020-12"},
        },
    ),
}
