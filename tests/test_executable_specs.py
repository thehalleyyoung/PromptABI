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
    NamedConstraint,
    SolverConclusion,
    SolverStatus,
    Value,
    Var,
)
from promptabi.specs import (
    assert_spec_report,
    check_contract_result,
    check_dfa_product_language,
    check_dfa_witness,
    check_transducer_witness,
)


def test_executable_spec_checks_dfa_reachability_witness() -> None:
    delimiter = DeterministicFiniteAutomaton.literal("<|im_start|>", alphabet=set("<|im_start|>abc"))
    witness = delimiter.shortest_witness()

    report = check_dfa_witness(delimiter, witness)

    assert report.passed
    assert report.to_dict()["passed"] is True


def test_executable_spec_rejects_invalid_dfa_witness_path() -> None:
    delimiter = DeterministicFiniteAutomaton.literal("<x>")
    witness = delimiter.shortest_witness()
    assert witness is not None
    bad_witness = type(witness)(
        symbols=witness.symbols,
        states=(delimiter.start,) + witness.states[2:] + (delimiter.start,),
    )

    report = check_dfa_witness(delimiter, bad_witness)

    assert not report.passed
    assert {failure.name for failure in report.failures} >= {"path-follows-transitions", "path-ends-accepting"}


def test_executable_spec_checks_bounded_product_language_laws() -> None:
    control = DeterministicFiniteAutomaton.finite_language(["AB", "BA"], alphabet={"A", "B", "x"})
    user_text = DeterministicFiniteAutomaton.finite_language(["x", "AB"], alphabet={"A", "B", "x"})

    forged = control.intersect(user_text, name="forged-control")
    safe_minus_control = user_text.difference(control, name="safe-minus-control")
    either = control.union(user_text, name="either-control-or-user")

    assert_spec_report(check_dfa_product_language(forged, control, user_text, operation="intersection", max_depth=2))
    assert_spec_report(check_dfa_product_language(safe_minus_control, user_text, control, operation="difference", max_depth=2))
    assert_spec_report(check_dfa_product_language(either, control, user_text, operation="union", max_depth=2))


def test_executable_spec_detects_product_language_mismatch() -> None:
    left = DeterministicFiniteAutomaton.literal("a", alphabet={"a", "b"})
    right = DeterministicFiniteAutomaton.literal("b", alphabet={"a", "b"})
    wrong_product = left.union(right, name="not-an-intersection")

    report = check_dfa_product_language(wrong_product, left, right, operation="intersection", max_depth=1)

    assert not report.passed
    assert report.failures[0].name == "bounded-product-language"


def test_executable_spec_checks_transducer_witness_with_epsilons() -> None:
    render = FiniteStateTransducer.literal_mapping("U", "<user>")
    tokenize = FiniteStateTransducer.finite_relation((("<user>", "T"), ("<assistant>", "A")))
    composed = render.compose(tokenize, name="render-then-tokenize")

    report = check_transducer_witness(composed, composed.shortest_witness())

    assert report.passed


def test_executable_spec_checks_sat_contract_assignment() -> None:
    problem = FiniteContractProblem(
        name="role-boundary-counterexample",
        variables=(
            EnumDomain("role", ("user", "assistant")),
            BoundedStringDomain("content", tuple("<a>"), min_length=0, max_length=3),
        ),
        constraints=(
            NamedConstraint("attacker-role", Eq(Var("role"), Value("user"))),
            NamedConstraint("contains-delimiter", Contains(Var("content"), Value("<a>"))),
        ),
    )

    result = problem.solve(prefer_z3=False)
    report = check_contract_result(problem, result)

    assert result.status is SolverStatus.SAT
    assert report.passed


def test_executable_spec_checks_minimal_unsat_core() -> None:
    problem = FiniteContractProblem(
        variables=(BoolDomain("flag"),),
        constraints=(
            NamedConstraint("flag-true", Eq(Var("flag"), Value(True))),
            NamedConstraint("flag-false", Eq(Var("flag"), Value(False))),
        ),
    )

    result = problem.solve(prefer_z3=False)
    report = check_contract_result(problem, result)

    assert result.status is SolverStatus.UNSAT
    assert result.unsat_core == ("flag-true", "flag-false")
    assert report.passed


def test_executable_spec_checks_unknown_is_only_abstention() -> None:
    problem = FiniteContractProblem(
        variables=(BoolDomain("left"), BoolDomain("right")),
        constraints=(
            NamedConstraint(
                "left-and-right",
                And(Eq(Var("left"), Value(True)), Eq(Var("right"), Value(True))),
            ),
        ),
    )

    result = problem.solve(prefer_z3=False, max_assignments=1)
    report = check_contract_result(problem, result)

    assert result.status is SolverStatus.UNKNOWN
    assert result.conclusion is SolverConclusion.ABSTENTION
    assert report.passed
