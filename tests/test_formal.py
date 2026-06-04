import pytest

from promptabi.formal import (
    And,
    BoolDomain,
    BoundedStringDomain,
    Contains,
    DeterministicFiniteAutomaton,
    EnumDomain,
    Eq,
    FiniteContractProblem,
    FiniteStateTransducer,
    Implies,
    IntRangeDomain,
    Length,
    NamedConstraint,
    Ne,
    SolverStatus,
    Value,
    Var,
)


def test_dfa_intersection_extracts_shortest_reachability_witness() -> None:
    stop = DeterministicFiniteAutomaton.literal("</s>", alphabet=set("</s>abc"), name="stop")
    grammar_prefixes = DeterministicFiniteAutomaton.prefix_closed_literal("</s>", alphabet=set("</s>abc"), name="json-prefix")

    product = stop.intersect(grammar_prefixes, name="reachable-stop")
    witness = product.shortest_witness()

    assert witness is not None
    assert witness.text == "</s>"
    assert product.accepts_text("</s>") is True
    assert product.accepts_text("</") is False
    assert product.to_dict()["name"] == "reachable-stop"


def test_dfa_difference_proves_literal_excluded_from_safe_language() -> None:
    role_delimiter = DeterministicFiniteAutomaton.literal("<|assistant|>", alphabet=set("<|assistant|>user"))
    user_language = DeterministicFiniteAutomaton.finite_language(
        ["hello", "safe user text"],
        alphabet=set("<|assistant|>safe user texthello"),
        name="user-fields",
    )

    forged = role_delimiter.intersect(user_language, name="forged-role")
    safe_without_delimiter = user_language.difference(role_delimiter, name="safe-minus-delimiter")

    assert forged.is_empty() is True
    assert safe_without_delimiter.accepts_text("hello") is True
    assert safe_without_delimiter.accepts_text("<|assistant|>") is False


def test_transducer_composition_reconstructs_render_then_tokenize_witness() -> None:
    render = FiniteStateTransducer.literal_mapping("U", "<user>", name="render-role")
    tokenize = FiniteStateTransducer.finite_relation((("<user>", "T"), ("<assistant>", "A")), name="tokenize-control")

    composed = render.compose(tokenize, name="render-tokenize")
    witness = composed.shortest_witness()

    assert composed.accepts_pair("U", "T") is True
    assert composed.accepts_pair("U", "A") is False
    assert witness is not None
    assert witness.input_text == "U"
    assert witness.output_text == "T"
    assert witness.labels[0].to_dict() == {"input": "U", "output": "T"}
    assert any(label.input_symbol is None and label.output_symbol is None for label in witness.labels)
    assert composed.to_dict()["approximation"] == "exact"


def test_transducer_projections_track_input_and_output_languages_with_epsilons() -> None:
    relation = FiniteStateTransducer.finite_relation((("ab", "X"), ("c", "YZ")), name="toy-codec")

    input_language = relation.project_input(name="codec-inputs")
    output_language = relation.project_output(name="codec-outputs")

    assert input_language.accepts_text("ab") is True
    assert input_language.accepts_text("c") is True
    assert input_language.accepts_text("a") is False
    assert output_language.accepts_text("X") is True
    assert output_language.accepts_text("YZ") is True
    assert output_language.accepts_text("Y") is False


def test_transducer_overapproximation_pairs_independent_projections() -> None:
    exact = FiniteStateTransducer.finite_relation((("safe", "OK"), ("unsafe", "BAD")), name="classified")
    overapprox = exact.overapproximate_by_projections(name="classified-approx")

    assert exact.accepts_pair("safe", "BAD") is False
    assert overapprox.accepts_pair("safe", "BAD") is True
    assert overapprox.to_dict()["approximation"] == "overapproximation"


def test_finite_contract_solver_finds_counterexample_for_role_forgery() -> None:
    problem = FiniteContractProblem(
        name="role-boundary-nonforgeability",
        variables=(
            EnumDomain("role", ("system", "user", "assistant")),
            BoundedStringDomain("content", tuple("<a>bc"), min_length=0, max_length=3),
            IntRangeDomain("max_length", 0, 8),
            BoolDomain("escaped"),
        ),
        constraints=(
            NamedConstraint("attacker-controlled-role", Eq(Var("role"), Value("user"))),
            NamedConstraint("content-fits-field", Eq(Length(Var("content")), Var("max_length"))),
            NamedConstraint("unescaped-content", Eq(Var("escaped"), Value(False))),
            NamedConstraint("forgeable-delimiter", Contains(Var("content"), Value("<a>"))),
        ),
    )

    result = problem.solve(prefer_z3=False)

    assert result.status is SolverStatus.SAT
    assert result.assignment == {
        "content": "<a>",
        "escaped": False,
        "max_length": 3,
        "role": "user",
    }
    assert result.to_dict()["backend"] == "finite-enumeration"


def test_finite_contract_solver_reports_minimal_unsat_core() -> None:
    problem = FiniteContractProblem(
        variables=(EnumDomain("role", ("user", "assistant")),),
        constraints=(
            NamedConstraint("must-be-user", Eq(Var("role"), Value("user"))),
            NamedConstraint("must-be-assistant", Eq(Var("role"), Value("assistant"))),
            NamedConstraint("vacuous-implication", Implies(Eq(Var("role"), Value("user")), Eq(Var("role"), Value("user")))),
        ),
    )

    result = problem.solve(prefer_z3=False)

    assert result.unsat is True
    assert result.unsat_core == ("must-be-user", "must-be-assistant")
    assert result.checked_assignments == 2


def test_finite_contract_solver_supports_boolean_composition() -> None:
    problem = FiniteContractProblem(
        variables=(BoolDomain("has_system"), BoolDomain("drops_oldest")),
        constraints=(
            NamedConstraint(
                "unsafe-truncation",
                And(
                    Eq(Var("has_system"), Value(True)),
                    Eq(Var("drops_oldest"), Value(True)),
                    Ne(Var("has_system"), Var("drops_oldest")),
                ),
            ),
        ),
    )

    result = problem.solve(prefer_z3=False)

    assert result.unsat is True


def test_finite_contract_solver_uses_z3_when_available() -> None:
    pytest.importorskip("z3")
    problem = FiniteContractProblem(
        variables=(BoolDomain("escaped"), EnumDomain("role", ("assistant", "user"))),
        constraints=(
            NamedConstraint("user-controlled", Eq(Var("role"), Value("user"))),
            NamedConstraint("not-escaped", Eq(Var("escaped"), Value(False))),
        ),
    )

    result = problem.solve(prefer_z3=True)

    assert result.sat is True
    assert result.to_dict()["backend"] == "z3"
    assert result.assignment == {"escaped": False, "role": "user"}
