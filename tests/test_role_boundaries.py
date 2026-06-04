from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactLoader,
    ChatTemplateArtifact,
    ChatTemplateSymbolicBounds,
    build_role_boundary_model,
    load_seed_corpus,
    parse_hf_chat_template_config,
    parse_hf_tokenizer_config_chat_template,
)


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
