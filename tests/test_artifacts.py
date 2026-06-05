from pathlib import Path

import pytest

from promptabi.artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ChatTemplateArtifact,
    EvaluationFewShotExample,
    EvaluationHarnessArtifact,
    FrameworkTruncationConfigArtifact,
    GrammarArtifact,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
    PromptSegment,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    SchemaArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TruncationStrategy,
    artifact_from_config,
)


def test_core_artifact_model_serializes_every_kind_deterministically() -> None:
    location = ArtifactLocation(path="/tmp/promptabi-artifact.json")
    provenance = ArtifactProvenance(version="v1", sha256="abc123")
    artifacts = (
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="tok",
            location=location,
            provenance=provenance,
            family="byte-bpe",
            added_tokens=("<eos>", "<bos>", "<eos>"),
        ),
        ChatTemplateArtifact(
            kind=ArtifactKind.CHAT_TEMPLATE,
            name="template",
            location=location,
            roles=("user", "assistant", "user"),
            add_generation_prompt=True,
        ),
        SpecialTokenMapArtifact(
            kind=ArtifactKind.SPECIAL_TOKEN_MAP,
            name="specials",
            location=location,
            tokens=(SpecialToken("eos", "</s>", 2), SpecialToken("bos", "<s>", 1)),
        ),
        StopPolicyArtifact(
            kind=ArtifactKind.STOP_POLICY,
            name="stops",
            location=location,
            stop_sequences=("</tool_call>", "\n\n", "</tool_call>"),
            stop_token_ids=(2, 2, 128001),
            include_eos=False,
            source_family="openai-compatible",
        ),
        SchemaArtifact(
            kind=ArtifactKind.SCHEMA,
            name="schema",
            location=location,
            dialect="json-schema-2020-12",
        ),
        GrammarArtifact(
            kind=ArtifactKind.GRAMMAR,
            name="grammar",
            location=location,
            grammar_type="ebnf",
        ),
        ToolDefinitionArtifact(
            kind=ArtifactKind.TOOL_DEFINITION,
            name="tools",
            location=location,
            provider="openai",
            tool_names=("refund_user", "lookup_order"),
        ),
        PromptSegmentArtifact(
            kind=ArtifactKind.PROMPT_SEGMENT,
            name="segments",
            location=location,
            segments=(
                PromptSegment("system-policy", role="system", required=True, max_tokens=128),
                PromptSegment("retrieval", role="user", required=False),
            ),
        ),
        ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name="provider",
            location=location,
            provider="openai-compatible",
            api_family="chat-completions",
        ),
        FrameworkTruncationConfigArtifact(
            kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
            name="budget",
            location=location,
            framework="vllm",
            strategy=TruncationStrategy.LEFT,
            max_context_tokens=8192,
            reserve_output_tokens=512,
        ),
        TrainingManifestArtifact(
            kind=ArtifactKind.TRAINING_MANIFEST,
            name="train",
            location=location,
            dataset_format="chat-jsonl",
            message_roles=("system", "user", "assistant"),
            target_roles=("assistant",),
            example_count=12,
        ),
        EvaluationHarnessArtifact(
            kind=ArtifactKind.EVALUATION_HARNESS,
            name="eval",
            location=location,
            benchmark_name="contract-bench",
            provider="openai-compatible",
            tokenizer="byte-bpe",
            prompt_template="template",
            answer_parser="json-schema",
            answer_schema="schema",
            stop_sequences=("</answer>",),
            allowed_roles=("user", "assistant"),
            required_prompt_variables=("question",),
            prompt_variables=("question",),
            few_shot_examples=(EvaluationFewShotExample("ex1", "user", "What?", 2),),
            max_prompt_tokens=16,
        ),
        PromptPackArtifact(
            kind=ArtifactKind.PROMPT_PACK,
            name="pack",
            location=location,
            pack_name="support-pack",
            pack_version="1.0.0",
            exported_templates=(
                PromptPackTemplate(
                    name="support-chat",
                    template="{{ messages }}",
                    roles=("system", "user", "assistant"),
                    variables=("messages",),
                    required_regions=("system-policy",),
                    supported_model_families=("openai-compatible",),
                ),
            ),
            expected_roles=("system", "user", "assistant"),
            tool_schemas=(PromptPackToolSchema("refund_user", provider="openai"),),
            stop_policies=(PromptPackStopPolicy("tool-json", stop_sequences=("</tool_call>",)),),
            supported_model_families=("openai-compatible",),
        ),
    )

    bundle = ArtifactBundle(artifacts)

    payload = bundle.to_dict()
    assert [item["kind"] for item in payload["artifacts"]] == sorted(kind.value for kind in ArtifactKind)
    assert bundle.by_name("tok").to_ref().to_dict() == {
        "kind": "tokenizer",
        "name": "tok",
        "path": "/tmp/promptabi-artifact.json",
        "version": "v1",
        "sha256": "abc123",
    }
    assert bundle.by_name("tok").to_dict()["added_tokens"] == ["<bos>", "<eos>"]
    assert bundle.by_name("segments").required_segments == (
        PromptSegment("system-policy", role="system", required=True, max_tokens=128),
    )


def test_config_artifact_parser_accepts_the_same_kinds_as_the_model(tmp_path: Path) -> None:
    artifact_file = tmp_path / "artifact.json"
    artifact_file.write_text("{}", encoding="utf-8")
    specs = {
        ArtifactKind.TOKENIZER: {"family": "byte-bpe", "added_tokens": ["<s>"]},
        ArtifactKind.CHAT_TEMPLATE: {"roles": ["system", "user"], "add_generation_prompt": True},
        ArtifactKind.SPECIAL_TOKEN_MAP: {"tokens": {"bos": "<s>", "eos": "</s>"}},
        ArtifactKind.STOP_POLICY: {
            "stop_sequences": ["</tool_call>"],
            "stop_token_ids": [128001],
            "include_eos": False,
            "source_family": "vllm",
        },
        ArtifactKind.SCHEMA: {"dialect": "json-schema"},
        ArtifactKind.GRAMMAR: {"grammar_type": "regex"},
        ArtifactKind.TOOL_DEFINITION: {"provider": "anthropic", "tool_names": ["refund_user"]},
        ArtifactKind.PROMPT_SEGMENT: {
            "segments": [{"name": "system-policy", "role": "system", "required": True}]
        },
        ArtifactKind.PROVIDER_CONFIG: {"provider": "openai", "api_family": "responses"},
        ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG: {
            "framework": "langchain",
            "strategy": "oldest-message",
            "max_context_tokens": 4096,
        },
        ArtifactKind.TRAINING_MANIFEST: {
            "dataset_format": "chat-jsonl",
            "message_roles": ["system", "user", "assistant"],
            "target_roles": ["assistant"],
            "example_count": 12,
        },
        ArtifactKind.EVALUATION_HARNESS: {
            "benchmark_name": "contract-bench",
            "provider": "openai-compatible",
            "tokenizer": "byte-bpe",
            "prompt_template": "template",
            "answer_parser": "json-schema",
            "answer_schema": "schema",
            "stop_sequences": ["</answer>"],
            "allowed_roles": ["user", "assistant"],
            "required_prompt_variables": ["question"],
            "prompt_variables": ["question"],
            "few_shot_examples": [{"id": "ex1", "role": "user", "content": "What?", "token_count": 2}],
            "max_prompt_tokens": 16,
        },
        ArtifactKind.PROMPT_PACK: {
            "pack_name": "support-pack",
            "pack_version": "1.0.0",
            "exported_templates": [
                {
                    "name": "support-chat",
                    "template": "{{ messages }}",
                    "roles": ["system", "user", "assistant"],
                    "variables": ["messages"],
                    "required_regions": ["system-policy"],
                    "supported_model_families": ["openai-compatible"],
                }
            ],
            "expected_roles": ["system", "user", "assistant"],
            "tool_schemas": [{"name": "refund_user", "provider": "openai"}],
            "stop_policies": [{"name": "tool-json", "stop_sequences": ["</tool_call>"]}],
            "supported_model_families": ["openai-compatible"],
        },
    }

    parsed = {
        kind: artifact_from_config(
            kind.value,
            {"kind": kind.value, "path": artifact_file.name, **extra},
            base_dir=tmp_path,
        )
        for kind, extra in specs.items()
    }

    assert set(parsed) == set(ArtifactKind)
    assert {artifact.kind for artifact in parsed.values()} == set(ArtifactKind)
    assert all(artifact.location.path == str(artifact_file.resolve()) for artifact in parsed.values())


def test_artifact_model_rejects_invalid_locations_and_segments() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactLocation(path="/tmp/a", uri="hf://model")

    with pytest.raises(ValueError, match="special token map values"):
        artifact_from_config(
            "specials",
            {"kind": "special-token-map", "path": "specials.json", "tokens": {"eos": 2}},
            base_dir=Path("."),
        )

    with pytest.raises(ValueError, match="at least one segment"):
        PromptSegmentArtifact(
            kind=ArtifactKind.PROMPT_SEGMENT,
            name="empty",
            location=ArtifactLocation(path="/tmp/segments.json"),
        )
