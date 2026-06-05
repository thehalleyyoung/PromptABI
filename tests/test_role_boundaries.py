import json

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactLoader,
    ChatTemplateArtifact,
    ChatTemplateSymbolicBounds,
    analyze_role_boundary_nonforgeability,
    build_role_boundary_model,
    load_seed_corpus,
    parse_hf_chat_template_config,
    parse_hf_tokenizer_config_chat_template,
)
from promptabi.render import render_json, render_text
from promptabi.session import VerificationSession


ROLE_BOUNDARY_REGRESSION_CASES = (
    (
        "chatml",
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        },
        {
            ("{messages[0].role}", "assistant", "role-header"),
            ("{messages[0].content}", "<|im_start|>", "assistant-prefix"),
            ("{messages[0].content}", "<|im_end|>", "special-token"),
        },
    ),
    (
        "llama-header-tokens",
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
            ),
            "additional_special_tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        },
        {
            ("{messages[0].role}", "assistant", "role-header"),
            ("{messages[0].content}", "<|start_header_id|>", "assistant-prefix"),
            ("{messages[0].content}", "<|end_header_id|>", "assistant-prefix"),
            ("{messages[0].content}", "<|eot_id|>", "special-token"),
        },
    ),
    (
        "mistral-instruction-tags",
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}"
                "{% endfor %}"
            ),
            "eos_token": "</s>",
            "additional_special_tokens": ["[INST]", "[/INST]"],
        },
        {
            ("{messages[0].content}", "[INST]", "special-token"),
            ("{messages[0].content}", "[/INST]", "special-token"),
            ("{messages[0].content}", "</s>", "assistant-prefix"),
        },
    ),
    (
        "xml-tool-tags",
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'tool' %}<tool_call>{{ message['content'] }}</tool_call>"
                "{% else %}<message role=\"{{ message['role'] }}\">{{ message['content'] }}</message>{% endif %}"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<tool_call>", "</tool_call>"],
        },
        {
            ("{messages[0].role}", "assistant", "role-header"),
            ("{messages[0].content}", "<tool_call>", "tool-call-sentinel"),
            ("{messages[0].content}", "</tool_call>", "tool-call-sentinel"),
        },
    ),
    (
        "markdown-fence-roles",
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}```user\n{{ message['content'] }}\n```"
                "{% elif message['role'] == 'assistant' %}```assistant\n{{ message['content'] }}\n```{% endif %}"
                "{% endfor %}"
            ),
        },
        {
            ("{messages[0].content}", "```user", "role-header"),
            ("{messages[0].content}", "```assistant", "assistant-prefix"),
        },
    ),
    (
        "custom-finetune-hash-headers",
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n{% endif %}"
                "{% endfor %}"
            ),
        },
        {
            ("{messages[0].content}", "### User:", "role-header"),
            ("{messages[0].content}", "### Assistant:", "assistant-prefix"),
        },
    ),
)


def test_role_boundary_regression_suite_covers_known_delimiter_collisions() -> None:
    bounds = ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=32)

    for case_name, config, expected_witnesses in ROLE_BOUNDARY_REGRESSION_CASES:
        parsed = parse_hf_chat_template_config(config)
        report = analyze_role_boundary_nonforgeability(parsed, bounds=bounds)
        actual_witnesses = {
            (finding.input_expression, finding.marker, finding.marker_kind)
            for finding in report.findings
        }

        assert report.model.supported, case_name
        assert expected_witnesses <= actual_witnesses, case_name
        for input_expression, marker, marker_kind in expected_witnesses:
            finding = next(
                finding
                for finding in report.findings
                if (
                    finding.input_expression == input_expression
                    and finding.marker == marker
                    and finding.marker_kind == marker_kind
                )
            )
            assert finding.malicious_input == marker, case_name
            assert marker in finding.rendered_excerpt, case_name
            assert marker in finding.tokenized_representation, case_name
            assert finding.marker_start_offset < finding.marker_end_offset, case_name


def test_role_boundary_model_tracks_condition_variable_and_generation_regions() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|end|>"
                "{% elif message['role'] == 'developer' %}<|developer|>{{ message['content'] }}<|end|>"
                "{% elif message['role'] == 'tool' %}<tool>{{ message['content'] }}</tool>"
                "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}"
            ),
            "additional_special_tokens": ["<|system|>", "<|developer|>", "<|assistant|>", "<|end|>"],
        }
    )

    model = build_role_boundary_model(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    assert model.supported
    assert set(model.roles) >= {"assistant", "developer", "system", "tool"}

    system_path = next(path for path in model.paths if "messages[0]['role'] == 'system'" in path.conditions)
    system_region = system_path.regions[0]
    assert system_region.role == "system"
    assert system_region.role_source == "condition"
    assert system_region.content_expressions == ("{messages[0].content}",)
    assert system_path.rendered_pattern[system_region.start_offset : system_region.end_offset] == (
        "<|system|>{messages[0].content}<|end|>"
    )

    residual_path = next(path for path in model.paths if any(condition.startswith("else after") for condition in path.conditions))
    residual_region = residual_path.regions[0]
    assert residual_region.role == "{messages[0].role}"
    assert residual_region.role_source == "variable"
    assert residual_region.excluded_roles == ("developer", "system", "tool")

    generation_path = next(
        path
        for path in model.paths
        if "add_generation_prompt" in path.conditions
        and any(region.role_source == "generation-prompt" for region in path.regions)
    )
    generation_region = next(region for region in generation_path.regions if region.role_source == "generation-prompt")
    assert generation_region.role == "assistant"
    assert generation_path.rendered_pattern[generation_region.start_offset : generation_region.end_offset] == "<|assistant|>"


def test_role_boundary_model_preserves_seed_corpus_region_invariants() -> None:
    corpus = load_seed_corpus()

    for entry in corpus.entries:
        parsed = parse_hf_tokenizer_config_chat_template(entry.path / "tokenizer_config.json")
        model = build_role_boundary_model(
            parsed,
            bounds=ChatTemplateSymbolicBounds(max_messages=2, max_tools=1, max_loop_iterations=2, max_paths=64),
        )

        assert model.supported
        assert model.paths
        assert set(parsed.role_assumptions).issubset(model.roles)
        for path in model.paths:
            seen_segments: set[int] = set()
            for region in path.regions:
                assert 0 <= region.start_offset <= region.end_offset <= len(path.rendered_pattern)
                assert region.role_source in {"condition", "literal", "variable", "generation-prompt", "residual"}
                assert region.role
                assert seen_segments.isdisjoint(region.segment_indexes)
                seen_segments.update(region.segment_indexes)


def test_chat_template_loader_publishes_role_boundary_metadata() -> None:
    artifact = ChatTemplateArtifact(
        kind=ArtifactKind.CHAT_TEMPLATE,
        name="llama-template",
        location=ArtifactLocation(path="fixtures/seed_corpus/llama/tokenizer_config.json"),
    )

    loaded = ArtifactLoader().load(artifact)
    metadata = dict(loaded.metadata)

    assert metadata["role_boundary_supported"] is True
    assert metadata["role_boundary_path_count"] > 0
    assert metadata["role_boundary_region_count"] > 0
    assert "assistant" in metadata["role_boundary_roles"]


def test_role_boundary_nonforgeability_flags_raw_role_and_content_controls() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )

    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    assert not report.ok
    assert any(
        finding.input_expression == "{messages[0].role}"
        and finding.marker_kind == "role-header"
        and finding.marker == "assistant"
        for finding in report.findings
    )
    assert any(
        finding.input_expression == "{messages[0].content}"
        and finding.marker == "<|im_start|>"
        and finding.marker_kind in {"assistant-prefix", "special-token"}
        for finding in report.findings
    )
    assert any(
        finding.input_expression == "{messages[0].content}"
        and "<|im_start|>" in finding.rendered_excerpt
        for finding in report.findings
    )
    content_forgery = next(
        finding
        for finding in report.findings
        if finding.input_expression == "{messages[0].content}" and finding.marker == "<|im_start|>"
    )
    assert content_forgery.malicious_input == "<|im_start|>"
    assert content_forgery.forged_boundary.startswith("assistant-prefix '<|im_start|>'")
    assert content_forgery.marker_start_offset < content_forgery.marker_end_offset
    assert content_forgery.marker_start_offset > 0
    assert "256:'<|im_start|>'/special,added" in content_forgery.tokenized_representation
    assert content_forgery.to_dict()["malicious_input"] == "<|im_start|>"
    assert content_forgery.token_ids
    assert 256 in content_forgery.token_ids
    assert content_forgery.role_region["role"] == "{messages[0].role}"


def test_role_boundary_diagnostic_emits_rich_witness_formats() -> None:
    result = VerificationSession.from_config_file("examples/role-boundary/unsafe.promptabi.json").run()
    diagnostic = next(
        item
        for item in result.diagnostics
        if item.rule_id == "role-boundary-nonforgeability"
        and item.message.startswith("{messages[0].content} can forge assistant-prefix '<|im_start|>'")
    )

    assert diagnostic.witness is not None
    assert diagnostic.witness.rendered_strings == (
        "<|im_start|>{messages[0].role}\n<|im_start|><|im_end|>\n<|im_start|>assistant\n",
    )
    assert diagnostic.witness.token_ids
    assert 256 in diagnostic.witness.token_ids
    assert diagnostic.witness.role_regions[0]["role"] == "{messages[0].role}"
    assert diagnostic.witness.role_regions[0]["content_expressions"] == ["{messages[0].content}"]
    assert diagnostic.witness.minimal_fixes[0].startswith(
        "Apply escaping or encoding to {messages[0].content}"
    )

    text = render_text(result)
    assert "token_ids:" in text
    assert "role_region: path=" in text
    assert "role={messages[0].role} source=variable chars=" in text
    assert "minimal_fix: Apply escaping or encoding to {messages[0].content}" in text

    payload = json.loads(render_json(result))
    rendered = [
        item["witness"]
        for item in payload["diagnostics"]
        if item["fingerprint"] == diagnostic.fingerprint
    ][0]
    assert rendered["rendered_strings"] == list(diagnostic.witness.rendered_strings)
    assert rendered["token_ids"] == list(diagnostic.witness.token_ids)
    assert rendered["role_regions"] == list(diagnostic.witness.role_regions)


def test_role_boundary_nonforgeability_does_not_concatenate_across_content_placeholders() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<safe>{{ message['content'] }}</safe>"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<safe>", "</safe>"],
        }
    )

    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    markers = {finding.marker for finding in report.findings}
    assert "<safe>" in markers
    assert "</safe>" in markers
    assert "<safe></safe>" not in markers


def test_role_boundary_nonforgeability_recognizes_content_sanitizer_filters() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content']|tojson }}<|im_end|>\n"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )

    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    assert any(
        finding.input_expression == "{messages[0].role}" and finding.marker_kind == "role-header"
        for finding in report.findings
    )
    assert not any(finding.input_expression == "{messages[0].content}" for finding in report.findings)

    region = next(region for path in report.model.paths for region in path.regions if region.message_index == 0)
    sanitizer = next(
        item for item in region.input_sanitizers if item.input_expression == "{messages[0].content}"
    )
    assert sanitizer.field == "content"
    assert sanitizer.filters == ("tojson",)
    assert "JSON-encodes" in sanitizer.reason


def test_role_boundary_nonforgeability_recognizes_sanitized_role_and_set_bindings() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "{% set safe_content = message['content']|escape %}"
                "<|im_start|>{{ message['role']|tojson }}\n"
                "{{ safe_content }}<|im_end|>\n"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )

    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    assert report.ok
    region = next(region for path in report.model.paths for region in path.regions if region.message_index == 0)
    sanitizers = {item.input_expression: item for item in region.input_sanitizers}
    assert sanitizers["{messages[0].role}"].filters == ("tojson",)
    assert sanitizers["{messages[0].content}"].filters == ("escape",)
    assert region.to_dict()["input_sanitizers"]


def test_role_boundary_nonforgeability_recognizes_delimiter_safe_wrapper_filters() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|im_start|>user\n"
                "{{ message['content']|base64 }}<|im_end|>\n"
                "{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )

    report = analyze_role_boundary_nonforgeability(
        parsed,
        bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1, max_paths=16),
    )

    assert report.ok
    region = next(region for path in report.model.paths for region in path.regions if region.message_index == 0)
    sanitizer = next(
        item for item in region.input_sanitizers if item.input_expression == "{messages[0].content}"
    )
    assert sanitizer.filters == ("base64",)
    assert "excludes role and tool delimiter punctuation" in sanitizer.reason
