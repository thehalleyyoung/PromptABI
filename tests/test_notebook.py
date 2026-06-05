from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    BoolDomain,
    CheckMode,
    DeterministicFiniteAutomaton,
    Eq,
    FiniteContractProblem,
    NamedConstraint,
    NotebookVisualization,
    StopPolicyArtifact,
    TokenBudgetVisualization,
    TokenBudgetVisualizationRow,
    Value,
    Var,
    parse_hf_chat_template_config,
    render_notebook_visualization_html,
    render_notebook_visualization_text,
    visualize_grammar_product,
    visualize_smt_constraints,
    visualize_stop_reachability,
    visualize_template_rendering,
    visualize_tokenization,
    visualize_truncation,
)
from promptabi.tokenizers import ByteLevelTokenizer


def test_notebook_tokenization_uses_adapter_data_and_escapes_html() -> None:
    tokenizer = ByteLevelTokenizer(added_tokens=("<|im_start|>",), special_tokens={"<|im_start|>": 50256})
    text = "<script>alert(1)</script><|im_start|>"

    visualization = visualize_tokenization(text, tokenizer)
    encoded = tokenizer.encode(text)
    rendered = visualization._repr_html_()

    assert isinstance(visualization, NotebookVisualization)
    assert visualization.mode is CheckMode.BOUNDED
    assert visualization.payload["tokens"] == [token.to_dict() for token in encoded.tokens]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script" not in rendered
    assert "<link" not in rendered
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert render_notebook_visualization_text(visualization) == visualization.render_text()
    assert render_notebook_visualization_html(visualization) == rendered


def test_notebook_template_rendering_wraps_concrete_and_symbolic_paths() -> None:
    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}assistant: {% endif %}"
            )
        }
    )

    visualization = visualize_template_rendering(
        parsed,
        ({"role": "user", "content": "hello"},),
        add_generation_prompt=True,
    )

    assert visualization.mode is CheckMode.BOUNDED
    assert visualization.payload["rendered"] == "user: hello\nassistant: "
    assert visualization.payload["symbolic_execution"]["paths"]
    assert "bounded" in str(visualization)


def test_notebook_stop_reachability_uses_stop_analyzer() -> None:
    tokenizer = ByteLevelTokenizer(added_tokens=("END",), special_tokens={"END": 300})
    stop_policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("END", "END!"),
        stop_token_ids=(300,),
    )

    visualization = visualize_stop_reachability(stop_policy, tokenizer)

    assert visualization.mode is CheckMode.SOUND
    assert visualization.payload["tokenizer_backend"] == "byte-level"
    assert visualization.payload["sequences"][0]["token_ids"] == [300]
    assert visualization.payload["collisions"]
    assert "unreachable token id(s)" in visualization.summary


def test_notebook_grammar_product_and_smt_constraints_are_deterministic() -> None:
    left = DeterministicFiniteAutomaton.literal("ab", alphabet=("a", "b"), name="schema")
    right = DeterministicFiniteAutomaton.prefix_closed_literal("ab", alphabet=("a", "b"), name="tokenizer")
    product_view = visualize_grammar_product(left, right)

    problem = FiniteContractProblem(
        variables=(BoolDomain("allowed"),),
        constraints=(NamedConstraint("must_be_allowed", Eq(Var("allowed"), Value(True))),),
        name="policy",
    )
    smt_view = visualize_smt_constraints(problem, prefer_z3=False)

    assert product_view.mode is CheckMode.COMPLETE
    assert product_view.payload["intersection"]["witness"]["text"] == "ab"
    assert product_view.render_html() == visualize_grammar_product(left, right).render_html()
    assert smt_view.mode is CheckMode.BOUNDED
    assert smt_view.payload["result"]["status"] == "sat"
    assert smt_view.payload["result"]["assignment"] == {"allowed": True}
    assert "finite-enumeration" in smt_view.render_text()


def test_notebook_truncation_accepts_existing_budget_visualization() -> None:
    budget = TokenBudgetVisualization(
        budget_source="unit-test",
        framework="langchain",
        strategy="oldest-message",
        max_context_tokens=12,
        reserved_total=2,
        input_budget_tokens=10,
        total_prompt_tokens=14,
        required_prompt_tokens=4,
        overflow_tokens=4,
        truncation_boundary_tokens=10,
        must_survive_status="proved",
        rows=(
            TokenBudgetVisualizationRow(
                index=0,
                name="system",
                role="system",
                required=True,
                token_count=4,
                overhead_tokens=0,
                metadata_tokens=0,
                template_overhead_tokens=0,
                total_tokens=4,
                source="declared",
                start_token=0,
                end_token=4,
                status="kept",
                survival="guaranteed",
            ),
            TokenBudgetVisualizationRow(
                index=1,
                name="history",
                role="user",
                required=False,
                token_count=10,
                overhead_tokens=0,
                metadata_tokens=0,
                template_overhead_tokens=0,
                total_tokens=10,
                source="declared",
                start_token=4,
                end_token=14,
                status="dropped",
                survival="optional-dropped",
            ),
        ),
    )

    visualization = visualize_truncation(budget)

    assert visualization.mode is CheckMode.BOUNDED
    assert visualization.payload == budget.to_dict()
    assert "drops 1" in visualization.summary
    assert "history" in visualization.render_html()
