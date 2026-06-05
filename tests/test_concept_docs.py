from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ByteLevelTokenizer,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    VerificationConfig,
    analyze_token_budget,
)
from promptabi.budgets import TokenBudgetReservation
from promptabi.formal import (
    BoundedStringDomain,
    Contains,
    DeterministicFiniteAutomaton,
    FiniteContractProblem,
    NamedConstraint,
    SolverStatus,
    Value,
    Var,
)
from promptabi.loaders import ArtifactLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DOCS = {
    "concepts/formalism.md": (
        "DeterministicFiniteAutomaton",
        "FiniteStateTransducer",
        "FiniteContractProblem",
        "sound",
        "abstaining",
    ),
    "concepts/tokenizer-template-composition.md": (
        "messages -> chat template -> rendered prompt -> tokenizer -> token stream",
        "ByteLevelTokenizer",
        "grammar emptiness",
    ),
    "concepts/static-contracts.md": (
        "analyze_static_contracts",
        "StaticContractFinding",
        "SolverStatus.SAT",
    ),
    "concepts/role-boundary-nonforgeability.md": (
        "RoleBoundaryModel",
        "RoleBoundaryRegion",
        "forged_boundary",
    ),
    "concepts/stop-reachability.md": (
        "analyze_stop_overreachability",
        "StructuredOutputRegion",
        "line/column firing point",
    ),
    "concepts/grammar-emptiness.md": (
        "analyze_tokenizer_grammar_emptiness",
        "satisfiable",
        "empty",
        "abstained",
    ),
    "concepts/must-survive-budgets.md": (
        "TokenBudgetReservation",
        "TruncationPolicy",
        "MustSurviveProof",
    ),
}


def test_concept_docs_are_linked_from_mkdocs_nav_and_check_families() -> None:
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    checks = (REPO_ROOT / "docs" / "checks.md").read_text(encoding="utf-8")

    for relative_path, required_phrases in CONCEPT_DOCS.items():
        doc_text = (REPO_ROOT / "docs" / relative_path).read_text(encoding="utf-8")
        assert relative_path in mkdocs
        for phrase in required_phrases:
            assert phrase.lower() in doc_text.lower()

    for link in (
        "concepts/formalism.md",
        "concepts/static-contracts.md",
        "concepts/stop-reachability.md",
        "concepts/grammar-emptiness.md",
        "concepts/must-survive-budgets.md",
    ):
        assert link in checks


def test_formalism_doc_minimal_dfa_example_matches_real_code() -> None:
    stop = DeterministicFiniteAutomaton.literal("</s>", alphabet=set("</s>abc"))
    prefixes = DeterministicFiniteAutomaton.prefix_closed_literal("</s>", alphabet=set("</s>abc"))
    witness = stop.intersect(prefixes).shortest_witness()

    assert witness is not None
    assert witness.text == "</s>"


def test_static_contract_doc_minimal_solver_example_matches_real_code() -> None:
    problem = FiniteContractProblem(
        name="delimiter-forgery",
        variables=(BoundedStringDomain("content", tuple("<a>bc"), min_length=0, max_length=3),),
        constraints=(NamedConstraint("contains-marker", Contains(Var("content"), Value("<a>"))),),
    )

    result = problem.solve(prefer_z3=False)

    assert result.status is SolverStatus.SAT
    assert result.assignment == {"content": "<a>"}


def test_budget_doc_arithmetic_and_survival_terms_match_real_analyzer() -> None:
    reservation = TokenBudgetReservation(
        max_context_tokens=100,
        reserve_output_tokens=20,
        reserved_tool_tokens=7,
        generation_prompt_tokens=3,
        special_token_overhead=5,
    )
    assert reservation.input_budget_tokens == 65

    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="doc-budget",
        location=ArtifactLocation(uri="memory://doc-budget"),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=18),
            PromptSegment("old-required", role="user", required=True, token_count=26),
            PromptSegment("latest", role="user", required=True, token_count=20),
        ),
    )
    truncation = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="doc-langchain",
        location=ArtifactLocation(uri="memory://doc-langchain"),
        framework="langchain",
        max_context_tokens=50,
    )

    report = analyze_token_budget(
        VerificationConfig(name="doc-budget", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(truncation)),
    )

    assert report.policy is not None
    assert report.policy.strategy == "oldest-message"
    assert report.must_survive_proof is not None
    assert report.must_survive_proof.status == "violated"
    assert report.must_survive_proof.dropped_segments == ("old-required",)


def test_tokenizer_template_composition_doc_names_real_tokenizer_behavior() -> None:
    tokenizer = ByteLevelTokenizer(normalization=("lowercase",))
    encoded = tokenizer.encode('"OK"', add_special_tokens=False)

    assert encoded.normalized_text == '"ok"'
    assert tokenizer.decode(encoded.token_ids).text == '"ok"'
