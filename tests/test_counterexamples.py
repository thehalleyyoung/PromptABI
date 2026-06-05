from promptabi import (
    ArtifactRef,
    BoundedStringDomain,
    Contains,
    CounterexampleMetric,
    CounterexampleProduct,
    CounterexampleProductArtifact,
    CounterexampleShrinkError,
    DeterministicFiniteAutomaton,
    EnumDomain,
    Eq,
    FiniteContractProblem,
    FiniteStateTransducer,
    IntRangeDomain,
    NamedConstraint,
    SolverBackend,
    SolverResult,
    SolverStatus,
    Value,
    Var,
    slice_counterexample_product,
    shrink_automaton_counterexample,
    shrink_finite_contract_counterexample,
    shrink_transducer_counterexample,
)
from promptabi.formal import AutomatonWitness, TransducerWitness


def test_automaton_counterexample_shrinker_replaces_nonminimal_path() -> None:
    automaton = DeterministicFiniteAutomaton.finite_language(
        ["long-control-token", "x"],
        alphabet=set("long-control-tokenx"),
        name="role-control-language",
    )
    original = AutomatonWitness(
        symbols=tuple("long-control-token"),
        states=tuple(f"q{index}" for index in range(len("long-control-token") + 1)),
    )
    assert automaton.accepts_text(original.text)

    report = shrink_automaton_counterexample(automaton, original, metric=CounterexampleMetric.STRING_LENGTH)

    assert report.changed
    assert report.minimized["text"] == "x"
    assert report.original_cost == len("long-control-token")
    assert report.minimized_cost == 1
    assert report.certificate["accepted"] is True
    assert "globally shortest" in report.steps[0].action
    assert report.witness().steps[-1].action == "certify minimized counterexample"


def test_transducer_counterexample_shrinker_minimizes_token_pair_cost() -> None:
    transducer = FiniteStateTransducer.finite_relation(
        (
            ("verbose-user-field", "ASSISTANT_CONTROL"),
            ("u", "A"),
        ),
        name="render-tokenize-relation",
    )
    original = transducer.shortest_witness()
    assert original is not None
    nonminimal = TransducerWitness(
        input_symbols=tuple("verbose-user-field"),
        output_symbols=tuple("ASSISTANT_CONTROL"),
        states=tuple(f"q{index}" for index in range(max(len("verbose-user-field"), len("ASSISTANT_CONTROL")) + 1)),
        labels=original.labels,
    )

    report = shrink_transducer_counterexample(transducer, nonminimal, metric=CounterexampleMetric.TOKEN_COUNT)

    assert report.changed
    assert report.minimized["input_text"] == "u"
    assert report.minimized["output_text"] == "A"
    assert report.minimized_cost == 2
    assert report.certificate["accepted"] is True
    assert report.certificate["minimality"] == "uniform-cost reachability over FST states"


def test_finite_contract_counterexample_shrinker_enumerates_minimal_assignment() -> None:
    problem = FiniteContractProblem(
        name="role-forgery-smt-counterexample",
        variables=(
            EnumDomain("role", ("assistant", "user")),
            BoundedStringDomain("content", tuple("<>abc"), min_length=0, max_length=6),
            IntRangeDomain("length", 0, 6),
        ),
        constraints=(
            NamedConstraint("attacker-role", Eq(Var("role"), Value("user"))),
            NamedConstraint("forged-delimiter", Contains(Var("content"), Value("<>"))),
            NamedConstraint("length-tracks-content", Eq(Var("length"), Value(2))),
        ),
    )
    nonminimal = SolverResult(
        status=SolverStatus.SAT,
        backend=SolverBackend.FINITE_ENUMERATION,
        assignment={"role": "user", "content": "abc<>", "length": 2},
    )

    report = shrink_finite_contract_counterexample(
        problem,
        nonminimal,
        metric=CounterexampleMetric.STRING_LENGTH,
    )

    assert report.changed
    assert report.minimized["assignment"] == {"content": "<>", "length": 2, "role": "user"}
    assert report.minimized_cost < report.original_cost
    assert report.certificate["satisfies_constraints"] is True
    assert report.certificate["minimality"] == "exhaustive finite-domain enumeration"


def test_counterexample_product_slicer_keeps_minimal_artifact_cover() -> None:
    template = CounterexampleProductArtifact(
        ArtifactRef(kind="template", name="chatml", path="tokenizer_config.json"),
        facts=("renders-user-region", "emits-im-start"),
        cost=3,
    )
    tokenizer = CounterexampleProductArtifact(
        ArtifactRef(kind="tokenizer", name="byte-level", path="tokenizer.json"),
        facts=("emits-im-start", "special-token-boundary"),
        cost=2,
    )
    provider = CounterexampleProductArtifact(
        ArtifactRef(kind="provider", name="openai-compatible", path="provider.json"),
        facts=("accepts-assistant-prefix",),
        cost=1,
    )
    redundant_policy = CounterexampleProductArtifact(
        ArtifactRef(kind="policy", name="org-baseline", path="policy.pabi"),
        facts=("renders-user-region",),
        cost=1,
    )
    product = CounterexampleProduct(
        name="chat-template-tokenizer-provider-forgery",
        artifacts=(template, tokenizer, provider, redundant_policy),
        failing_facts=("emits-im-start", "accepts-assistant-prefix"),
        edges=(
            (template.key, tokenizer.key),
            (tokenizer.key, provider.key),
            (redundant_policy.key, template.key),
        ),
    )

    report = slice_counterexample_product(product)

    assert report.artifact_keys == (tokenizer.key, provider.key)
    assert report.omitted_artifacts == (template.ref, redundant_policy.ref)
    assert report.cut_edges == ((template.key, tokenizer.key),)
    assert report.certificate["covers_required_facts"] is True
    assert report.certificate["minimality"] == "exhaustive minimum-cost artifact cover"
    assert report.certificate["candidate_slices_examined"] == 15
    assert report.to_dict()["required_facts"] == ["emits-im-start", "accepts-assistant-prefix"]

    witness = report.witness()
    assert witness.artifacts == (provider.ref, tokenizer.ref)
    assert witness.steps[-1].action == "certify sliced counterexample"


def test_counterexample_product_slicer_rejects_uncovered_required_fact() -> None:
    product = CounterexampleProduct(
        name="schema-tool-product",
        artifacts=(
            CounterexampleProductArtifact(
                ArtifactRef(kind="schema", name="tool", path="schema.json"),
                facts=("json-string-field",),
            ),
        ),
        failing_facts=("json-string-field",),
    )

    try:
        slice_counterexample_product(product, required_facts=("missing-provider-envelope",))
    except CounterexampleShrinkError as error:
        assert "missing-provider-envelope" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected uncovered product fact to be rejected")
