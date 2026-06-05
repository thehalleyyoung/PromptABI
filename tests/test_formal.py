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
    Ge,
    Gt,
    Implies,
    InSet,
    IntRangeDomain,
    Length,
    Le,
    Lt,
    NamedConstraint,
    Ne,
    Not,
    Or,
    SolverConclusion,
    SolverBudgetOutcome,
    SolverReplayFile,
    SolverQueryCache,
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


def test_dfa_lazy_intersection_uses_compressed_alphabet_without_building_product() -> None:
    alphabet = {chr(codepoint) for codepoint in range(32, 127)}
    transitions = {("start", symbol): "accept" for symbol in alphabet}
    left = DeterministicFiniteAutomaton(
        states=frozenset({"start", "accept"}),
        alphabet=tuple(alphabet),
        start="start",
        accepts=frozenset({"accept"}),
        transitions=transitions,
        name="left",
    )
    right = DeterministicFiniteAutomaton(
        states=frozenset({"start", "accept"}),
        alphabet=tuple(alphabet),
        start="start",
        accepts=frozenset({"accept"}),
        transitions=transitions,
        name="right",
    )

    compressed = left.intersection_witness(right, compress_alphabet=True)
    uncompressed = left.intersection_witness(right, compress_alphabet=False)
    eager = left.intersect(right)

    assert compressed.witness is not None
    assert len(compressed.witness.text) == 1
    assert eager.accepts_text(compressed.witness.text) is True
    assert compressed.explored_transitions < uncompressed.explored_transitions
    assert compressed.representative_symbols < uncompressed.representative_symbols


def test_dfa_minimization_preserves_language_and_removes_equivalent_sink_states() -> None:
    dfa = DeterministicFiniteAutomaton(
        states=frozenset({"start", "accept", "dead1", "dead2"}),
        alphabet=("a", "b"),
        start="start",
        accepts=frozenset({"accept"}),
        transitions={
            ("start", "a"): "accept",
            ("start", "b"): "dead1",
            ("accept", "a"): "dead1",
            ("accept", "b"): "dead2",
            ("dead1", "a"): "dead1",
            ("dead1", "b"): "dead2",
            ("dead2", "a"): "dead1",
            ("dead2", "b"): "dead2",
        },
        name="redundant",
    )

    minimized = dfa.minimize()

    assert len(minimized.states) < len(dfa.states)
    for text in ("", "a", "b", "aa", "ab", "ba"):
        assert minimized.accepts_text(text) is dfa.accepts_text(text)


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
    assert result.conclusion is SolverConclusion.COUNTEREXAMPLE
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


def test_finite_contract_solver_prunes_partial_assignments_by_constraint_slice() -> None:
    problem = FiniteContractProblem(
        variables=(
            BoolDomain("gate"),
            EnumDomain("payload", tuple(f"value-{index}" for index in range(100))),
        ),
        constraints=(
            NamedConstraint("gate-must-be-true", Eq(Var("gate"), Value(True))),
            NamedConstraint("gate-must-be-false", Eq(Var("gate"), Value(False))),
        ),
    )

    result = problem.solve(prefer_z3=False)

    assert result.unsat is True
    assert result.checked_assignments == 0
    assert result.unsat_core == ("gate-must-be-true", "gate-must-be-false")


def test_finite_contract_solver_abstains_when_assignment_limit_is_reached() -> None:
    problem = FiniteContractProblem(
        variables=(
            BoolDomain("left"),
            BoolDomain("right"),
        ),
        constraints=(
            NamedConstraint(
                "both-true",
                And(Eq(Var("left"), Value(True)), Eq(Var("right"), Value(True))),
            ),
        ),
    )

    result = problem.solve(prefer_z3=False, max_assignments=1)

    assert result.status is SolverStatus.UNKNOWN
    assert result.conclusion is SolverConclusion.ABSTENTION
    assert result.budget_outcome is SolverBudgetOutcome.BOUNDED
    assert "assignment budget" in (result.budget_reason or "")
    assert result.checked_assignments == 2


def test_finite_contract_solver_reports_timeout_budget_without_false_unsat() -> None:
    problem = FiniteContractProblem(
        variables=(BoundedStringDomain("payload", tuple("abcdef"), min_length=1, max_length=5),),
        constraints=(NamedConstraint("payload-contains-z", Contains(Var("payload"), Value("z"))),),
    )

    result = problem.solve(prefer_z3=False, timeout_seconds=1e-12)

    assert result.status is SolverStatus.UNKNOWN
    assert result.budget_outcome is SolverBudgetOutcome.TIMED_OUT
    assert result.unsat_core == ()
    assert "timed out" in (result.budget_reason or "")


def test_solver_result_serializes_budget_classification() -> None:
    problem = FiniteContractProblem(
        variables=(BoolDomain("flag"),),
        constraints=(NamedConstraint("flag-true", Eq(Var("flag"), Value(True))),),
    )

    result = problem.solve(prefer_z3=False)
    round_trip = type(result).from_dict(result.to_dict())

    assert round_trip.budget_outcome is result.budget_outcome
    assert round_trip.budget_reason == result.budget_reason
    assert round_trip.to_dict()["solver_budget_outcome"] == "proved"


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


def test_solver_replay_round_trips_full_supported_expression_fragment(tmp_path) -> None:
    problem = FiniteContractProblem(
        name="full-supported-fragment",
        variables=(
            BoolDomain("enabled"),
            EnumDomain("role", ("assistant", "user")),
            IntRangeDomain("tokens", 0, 4),
            BoundedStringDomain("content", tuple("ab"), min_length=1, max_length=2),
        ),
        constraints=(
            NamedConstraint("enabled", Eq(Var("enabled"), Value(True))),
            NamedConstraint("role-is-user", InSet(Var("role"), ("user",))),
            NamedConstraint("content-has-a", Contains(Var("content"), Value("a"))),
            NamedConstraint("length-upper-bound", Le(Length(Var("content")), Var("tokens"))),
            NamedConstraint("positive-budget", Gt(Var("tokens"), Value(0))),
            NamedConstraint("non-assistant", Ne(Var("role"), Value("assistant"))),
            NamedConstraint("bounded-range", And(Ge(Var("tokens"), Value(1)), Lt(Var("tokens"), Value(4)))),
            NamedConstraint("or-guard", Or(Eq(Var("content"), Value("a")), Eq(Var("content"), Value("ba")))),
            NamedConstraint("not-disabled", Not(Eq(Var("enabled"), Value(False)))),
            NamedConstraint("implication", Implies(Eq(Var("role"), Value("user")), Contains(Var("content"), Value("a")))),
        ),
    )
    replay = SolverReplayFile.from_problem(
        problem,
        replay_id="unit-full-fragment",
        prefer_z3=False,
        artifact_hashes={"promptabi.json": "sha256:abc123"},
        supported_fragment_metadata={"fragment": "finite-contract-v1"},
    )
    path = tmp_path / "replay.json"

    replay.write_json(path)
    loaded = SolverReplayFile.read_json(path)
    report = loaded.replay()

    assert loaded.problem.to_dict() == problem.to_dict()
    assert report.ok
    assert report.status_matches
    assert report.stored_sat_witness_valid is True
    assert loaded.to_dict()["privacy"]["stores_provider_credentials"] is False
    assert loaded.to_dict()["privacy"]["stores_reduced_formula_literals"] is True


def test_solver_replay_treats_backend_and_query_environment_as_provenance() -> None:
    problem = FiniteContractProblem(
        name="role-region-nonforgeability",
        variables=(
            EnumDomain("boundary_marker", ("<|im_start|>assistant",)),
            EnumDomain("controlled_region", ("user\x1fhello <|im_start|>assistant",)),
        ),
        constraints=(
            NamedConstraint("controlled-region-contains-boundary-marker", Contains(Var("controlled_region"), Var("boundary_marker"))),
        ),
    )
    replay = SolverReplayFile.from_problem(problem, replay_id="portable-sat-replay", prefer_z3=False)
    stale_environment = dict(replay.normalized_query)
    stale_environment["solver_version_fingerprints"] = (("z3", "different"),)
    replay = SolverReplayFile.from_dict({**replay.to_dict(), "normalized_query": stale_environment})

    report = replay.replay()

    assert report.ok
    assert report.status_matches
    assert report.stored_sat_witness_valid is True
    assert report.query_environment_matches is False


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
    assert result.to_dict()["conclusion"] == "concrete-counterexample"
    assert result.assignment == {"escaped": False, "role": "user"}


def test_finite_contract_solver_minimizes_z3_unsat_core_when_available() -> None:
    pytest.importorskip("z3")
    problem = FiniteContractProblem(
        variables=(EnumDomain("role", ("user", "assistant")),),
        constraints=(
            NamedConstraint("must-be-user", Eq(Var("role"), Value("user"))),
            NamedConstraint("must-be-assistant", Eq(Var("role"), Value("assistant"))),
            NamedConstraint("tautological-role-domain", InSet(Var("role"), ("user", "assistant"))),
        ),
    )

    result = problem.solve(prefer_z3=True)

    assert result.unsat is True
    assert result.backend.value == "z3"
    assert result.conclusion is SolverConclusion.UNSAT_CORE_PROOF
    assert result.unsat_core == ("must-be-user", "must-be-assistant")


def test_solver_query_cache_reuses_normalized_problem_results() -> None:
    cache = SolverQueryCache()
    problem = FiniteContractProblem(
        name="cached-role-forgery",
        variables=(
            EnumDomain("role", ("assistant", "user")),
            BoundedStringDomain("content", tuple("<a>"), min_length=0, max_length=3),
        ),
        constraints=(
            NamedConstraint("attacker-role", Eq(Var("role"), Value("user"))),
            NamedConstraint("contains-delimiter", Contains(Var("content"), Value("<a>"))),
        ),
    )

    first = problem.solve(
        prefer_z3=False,
        query_cache=cache,
        artifact_hashes={"template": "sha256:abc", "tokenizer": "sha256:def"},
        supported_fragment_metadata={"strings": {"max_length": 3}, "backend": "finite"},
    )
    second = problem.solve(
        prefer_z3=False,
        query_cache=cache,
        artifact_hashes={"tokenizer": "sha256:def", "template": "sha256:abc"},
        supported_fragment_metadata={"backend": "finite", "strings": {"max_length": 3}},
    )

    assert first.sat is True
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.assignment == first.assignment
    assert second.cache_key == first.cache_key
    assert cache.hits == 1
    assert cache.misses == 1


def test_solver_query_cache_key_includes_artifact_and_fragment_metadata() -> None:
    problem = FiniteContractProblem(
        variables=(BoolDomain("flag"),),
        constraints=(NamedConstraint("flag-true", Eq(Var("flag"), Value(True))),),
    )

    base = problem.solver_query_key(
        prefer_z3=False,
        artifact_hashes={"manifest": "sha256:one"},
        supported_fragment_metadata={"fragment": "bool-v1"},
    )
    changed_artifact = problem.solver_query_key(
        prefer_z3=False,
        artifact_hashes={"manifest": "sha256:two"},
        supported_fragment_metadata={"fragment": "bool-v1"},
    )
    changed_fragment = problem.solver_query_key(
        prefer_z3=False,
        artifact_hashes={"manifest": "sha256:one"},
        supported_fragment_metadata={"fragment": "bool-v2"},
    )

    assert base != changed_artifact
    assert base != changed_fragment
    payload = problem.normalized_solver_query(
        prefer_z3=False,
        artifact_hashes={"manifest": "sha256:one"},
        supported_fragment_metadata={"fragment": "bool-v1"},
    )
    assert ("promptabi-finite-contract-solver", "v2-query-cache") in payload["solver_version_fingerprints"]


def test_solver_query_cache_round_trips_to_json(tmp_path) -> None:
    cache_path = tmp_path / "solver-cache.json"
    cache = SolverQueryCache()
    problem = FiniteContractProblem(
        variables=(BoolDomain("flag"),),
        constraints=(NamedConstraint("flag-false", Eq(Var("flag"), Value(False))),),
    )

    first = cache.solve(problem, prefer_z3=False)
    cache.write_json(cache_path)
    restored = SolverQueryCache.read_json(cache_path)
    second = restored.solve(problem, prefer_z3=False)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.assignment == {"flag": False}
