from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactLoader,
    ChatTemplateArtifact,
    ChatTemplateParseError,
    load_seed_corpus,
    parse_hf_chat_template_config,
    parse_hf_tokenizer_config_chat_template,
)


def test_hf_tokenizer_config_parser_extracts_template_contract() -> None:
    parsed = parse_hf_tokenizer_config_chat_template(
        "fixtures/seed_corpus/openai-compatible/tokenizer_config.json"
    )

    assert parsed.supported
    assert parsed.uses_generation_prompt is True
    assert parsed.uses_tools is False
    assert parsed.role_assumptions == ("assistant", "tool")
    assert {field.field for field in parsed.message_fields} == {"content", "role"}
    assert {token.text for token in parsed.special_tokens} >= {"<|start|>", "<|end|>"}
    assert parsed.generation_prompt_excerpts == ("<|start|>assistant",)
    assert parsed.source_span is not None
    assert parsed.source_span.path.endswith("tokenizer_config.json")


def test_parser_identifies_tools_filters_whitespace_and_unsupported_constructs() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{%- for tool in tools %}{{ tool['function']['name']|tojson }}{% endfor %}"
                "{% macro unsafe() %}x{% endmacro %}"
                "{% if add_generation_prompt %}assistant{% endif %}"
                "{{ raise_exception('bad') }}"
            ),
            "bos_token": {"content": "<s>"},
            "additional_special_tokens": [{"content": "<tool>"}, "</tool>"],
        }
    )

    assert parsed.supported is False
    assert parsed.uses_tools is True
    assert parsed.uses_whitespace_control is True
    assert parsed.filters == ("tojson",)
    assert {field.field for field in parsed.tool_fields} == {"function"}
    assert {token.text for token in parsed.special_tokens} == {"<s>", "<tool>", "</tool>"}
    assert {item.kind for item in parsed.unsupported_constructs} == {"global", "tag"}


def test_parser_covers_every_seed_corpus_chat_template() -> None:
    corpus = load_seed_corpus()

    parsed = [
        parse_hf_tokenizer_config_chat_template(entry.path / "tokenizer_config.json")
        for entry in corpus.entries
    ]

    assert len(parsed) == len(corpus.entries)
    assert all(template.supported for template in parsed)
    assert all({field.field for field in template.message_fields} >= {"content", "role"} for template in parsed)
    assert all(template.special_tokens for template in parsed)
    assert {role for template in parsed for role in template.role_assumptions} >= {
        "assistant",
        "system",
        "tool",
        "user",
    }


def test_chat_template_loader_attaches_parsed_metadata() -> None:
    artifact = ChatTemplateArtifact(
        kind=ArtifactKind.CHAT_TEMPLATE,
        name="qwen-template",
        location=ArtifactLocation(path=str(Path("fixtures/seed_corpus/qwen/tokenizer_config.json").resolve())),
    )

    loaded = ArtifactLoader().load(artifact)
    metadata = dict(loaded.metadata)

    assert loaded.source_type == "huggingface-tokenizer-config-chat-template"
    assert metadata["template_format"] == "jinja"
    assert metadata["message_fields"] == ("content", "role")
    assert metadata["role_assumptions"] == ("assistant",)
    assert metadata["uses_generation_prompt"] is True
    assert metadata["supported_fragment"] is True
    assert any(name == "chat_template" for name, _span in loaded.source_spans)


def test_parser_rejects_tokenizer_config_without_chat_template() -> None:
    with pytest.raises(ChatTemplateParseError, match="chat_template"):
        parse_hf_chat_template_config({"eos_token": "</s>"})
