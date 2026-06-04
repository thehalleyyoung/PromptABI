from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactLoader,
    ChatTemplateArtifact,
    ChatTemplateParseError,
    ChatTemplateSymbolicBounds,
    load_seed_corpus,
    parse_hf_chat_template_config,
    parse_hf_tokenizer_config_chat_template,
    symbolically_execute_chat_template,
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
    assert metadata["symbolic_supported_fragment"] is True
    assert metadata["symbolic_path_count"] > 0
    assert metadata["symbolic_abstentions"] == ()
    assert metadata["supported_fragment"] is True
    assert any(name == "chat_template" for name, _span in loaded.source_spans)


def test_parser_rejects_tokenizer_config_without_chat_template() -> None:
    with pytest.raises(ChatTemplateParseError, match="chat_template"):
        parse_hf_chat_template_config({"eos_token": "</s>"})


def test_symbolic_executor_reconstructs_hf_whitespace_and_special_tokens() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{{ bos_token }}\n"
                "{% for message in messages %}\n"
                "  <|{{ message['role'] }}|>\n"
                "  {{ message['content'] }}\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
            ),
            "bos_token": "<s>",
            "additional_special_tokens": ["<|user|>", "<|assistant|>"],
        }
    )

    execution = symbolically_execute_chat_template(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1),
    )

    assert execution.supported
    assert len(execution.paths) == 4
    one_message_with_generation = [
        path
        for path in execution.paths
        if ("messages", 1) in path.loop_iterations and "add_generation_prompt" in path.conditions
    ][0]
    assert one_message_with_generation.rendered_pattern == (
        "<s>\n  <|{messages[0].role}|>\n"
        "  {messages[0].content}\n"
        "<|assistant|>\n"
    )
    assert one_message_with_generation.segments[0].kind == "constant"
    assert one_message_with_generation.segments[0].value == "<s>"


def test_symbolic_executor_bounds_seed_corpus_conditionals_and_loops() -> None:
    parsed = parse_hf_tokenizer_config_chat_template("fixtures/seed_corpus/mistral/tokenizer_config.json")

    execution = symbolically_execute_chat_template(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=2, max_tools=0, max_loop_iterations=2, max_paths=64),
    )

    assert execution.supported
    assert len(execution.paths) == 21
    assert any("messages[0]['role'] == 'user'" in path.conditions for path in execution.paths)
    assert any("messages[1]['role'] == 'assistant'" in path.conditions for path in execution.paths)
    assert any("{messages[0].content}" in path.rendered_pattern for path in execution.paths)
    assert any(path.loop_iterations == (("messages", 2),) for path in execution.paths)


def test_symbolic_executor_records_filters_sets_tools_and_path_budget() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% set prefix = '<tool>' %}"
                "{% for tool in tools %}"
                "{{ prefix }}{{ tool['function']['name']|tojson }}"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<tool>"],
        }
    )

    execution = symbolically_execute_chat_template(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=0, max_tools=2, max_loop_iterations=2, max_paths=2),
    )

    assert not execution.supported
    assert any(item.kind == "bounds" for item in execution.abstentions)
    assert len(execution.paths) == 2
    assert any(
        segment.value == "{tools[0].function.name}" and segment.filters == ("tojson",)
        for path in execution.paths
        for segment in path.segments
    )


def test_symbolic_executor_abstains_on_unsupported_constructs() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages recursive %}"
                "{{ raise_exception('bad') }}"
                "{% endfor %}"
            )
        }
    )

    execution = symbolically_execute_chat_template(parsed)

    assert not execution.supported
    assert execution.paths == ()
    assert {item.kind for item in execution.abstentions} >= {"loop", "global"}
