import json

import promptabi
from promptabi.cli import main
from promptabi.prompt_calculus import (
    PROMPT_CALCULUS_METATHEORY_VERSION,
    CONTROL_DELIMITERS,
    CompiledSchemaChecker,
    Concat,
    Data,
    Esc,
    FieldSchema,
    Lit,
    MigrationPatch,
    ObjectSchema,
    ProviderContract,
    Region,
    Seg,
    ToolEvent,
    denotational_truncate,
    drift_distance,
    fallback,
    feature_set_is_decidable,
    forged_delimiters,
    is_well_typed,
    lean_role_nonforgeability_source,
    maximal_decidable_feature_sets,
    negotiate,
    operational_truncate,
    refines,
    render,
    render_formal_appendix_markdown,
    render_metatheory_text,
    request_is_well_formed,
    run_metatheory,
    schema_accepts_reference,
    session_type_check,
    small_step,
    type_of,
)


def test_metatheory_report_all_theorems_pass_and_cover_steps_301_314() -> None:
    report = run_metatheory()
    by_step = {theorem.step: theorem for theorem in report.theorems}

    assert report.passed
    assert report.theorem_count == 14
    assert set(by_step) == set(range(301, 315))
    # Every theorem carries at least one executable obligation over a real domain.
    for theorem in report.theorems:
        assert theorem.checks
        assert theorem.domain_size > 0
        assert all(check.passed for check in theorem.checks)
    assert report.check_count >= 35
    assert report.domain_total > 10_000


def test_small_step_is_deterministic_and_adequate() -> None:
    term = Seg("system", Concat(Lit("be helpful"), Esc(Data("<|user|>"))))
    # Re-running a single step gives the identical successor (function, not relation).
    from promptabi.prompt_calculus import Configuration

    config = Configuration((term,), ())
    assert small_step(config) == small_step(config)
    # Operational normal form equals the denotational renderer.
    from promptabi.prompt_calculus import reduce_term

    final, steps = reduce_term(term)
    assert final.is_final
    assert steps > 0
    assert final.output == render(term)


def test_well_typed_prompts_never_forge_but_raw_data_can() -> None:
    guarded = Seg("user", Esc(Data("<|assistant|>")))
    assert is_well_typed(guarded)
    assert type_of(guarded) is Region.CONTROL
    assert forged_delimiters(render(guarded)) == ()

    raw = Seg("user", Data("<|assistant|>"))
    assert not is_well_typed(raw)
    assert forged_delimiters(render(raw))  # the raw injection forges a delimiter


def test_lean_artifact_states_the_theorem() -> None:
    source = lean_role_nonforgeability_source()
    assert "theorem role_nonforgeable" in source
    assert "namespace PromptABI" in source
    assert "∀" in source  # delimiters rendered with proof-assistant notation


def test_stop_policy_denotation_matches_operational_scan() -> None:
    text = "ab<|end|>cd"
    stops = ("<|end|>",)
    assert denotational_truncate(text, stops) == "ab"
    assert operational_truncate(text, stops) == "ab"
    # Property: result is a prefix and contains no stop.
    out = denotational_truncate(text, stops)
    assert text.startswith(out)
    assert "<|end|>" not in out


def test_schema_checker_soundness_and_completeness_witnesses() -> None:
    schema = ObjectSchema(
        fields=(
            FieldSchema("name", "string", required=True),
            FieldSchema("age", "integer", required=True),
        ),
        additional_properties=False,
    )
    checker = CompiledSchemaChecker.compile(schema)
    valid = {"name": "x", "age": 3}
    invalid_type = {"name": "x", "age": "three"}
    invalid_extra = {"name": "x", "age": 3, "extra": 1}

    assert schema_accepts_reference(schema, valid) is checker.accepts(valid) is True
    assert schema_accepts_reference(schema, invalid_type) is checker.accepts(invalid_type) is False
    assert schema_accepts_reference(schema, invalid_extra) is checker.accepts(invalid_extra) is False


def test_grammar_feature_lattice_is_down_closed() -> None:
    assert feature_set_is_decidable(frozenset())
    assert not feature_set_is_decidable(frozenset({"lookahead"}))
    assert not feature_set_is_decidable(frozenset({"complement", "recursion"}))
    maximal = maximal_decidable_feature_sets()
    assert maximal
    # Removing a feature from a decidable set stays decidable.
    big = frozenset({"recursion", "unbounded_repeat", "intersection"})
    assert feature_set_is_decidable(big)
    for feature in big:
        assert feature_set_is_decidable(big - {feature})


def test_capability_fallback_is_monotone_and_bounded() -> None:
    assert fallback("grammar_constrained") == "json_mode"
    assert fallback("best_effort_text") == "best_effort_text"
    # Negotiation never returns a tier stronger than requested.
    assert negotiate("json_mode", {"grammar_constrained"}) == "best_effort_text"
    assert negotiate("json_mode", {"json_mode", "grammar_constrained"}) == "json_mode"


def test_session_typed_tool_traces_are_balanced() -> None:
    good = (ToolEvent("open", 0), ToolEvent("arg", 0), ToolEvent("close", 0))
    bad = (ToolEvent("open", 0), ToolEvent("close", 1))
    assert session_type_check(good)
    assert not session_type_check(bad)


def test_migration_patch_safety_preserves_well_formedness() -> None:
    safe = MigrationPatch(renames=(("old_model", "model"),), adds=("temperature",))
    unsafe = MigrationPatch(renames=(("model", "temperature"),))
    assert safe.is_safe()
    assert not unsafe.is_safe()
    request = frozenset({"old_model", "messages"})
    patched = safe.apply(request)
    assert request_is_well_formed(patched)
    # The unsafe patch can break a well-formed request.
    well_formed = frozenset({"model", "messages"})
    assert request_is_well_formed(well_formed)
    assert not request_is_well_formed(unsafe.apply(well_formed))


def test_provider_contract_refinement_is_a_preorder() -> None:
    weak = ProviderContract(requires=frozenset({"auth", "tools"}), guarantees=frozenset({"streaming"}))
    strong = ProviderContract(requires=frozenset({"auth"}), guarantees=frozenset({"streaming", "tools"}))
    assert refines(strong, weak)
    assert not refines(weak, strong)
    assert refines(weak, weak)  # reflexive


def test_drift_distance_is_an_ultrametric() -> None:
    assert drift_distance(("a", "b", "c"), ("a", "b", "c")) == 0.0
    assert drift_distance(("a", "b"), ("a", "c")) == drift_distance(("a", "c"), ("a", "b"))
    x, y, z = ("a", "b", "c"), ("a", "x", "y"), ("p", "q", "r")
    assert drift_distance(x, z) <= max(drift_distance(x, y), drift_distance(y, z))


def test_control_delimiters_are_stable() -> None:
    assert CONTROL_DELIMITERS == ("<|assistant|>", "<|system|>", "<|user|>", "<|end|>")


def test_public_api_and_renderers_are_stable() -> None:
    report = run_metatheory()
    text = render_metatheory_text(report)
    payload = json.loads(promptabi.prompt_calculus_metatheory(output_format="json"))

    assert f"PromptABI prompt-assembly metatheory ({PROMPT_CALCULUS_METATHEORY_VERSION})" in text
    assert "301. operational-semantics-adequacy: PASS" in text
    assert payload["passed"] is True
    assert payload["theorem_count"] == report.theorem_count
    assert {item["theorem_id"] for item in payload["theorems"]} == {
        theorem.theorem_id for theorem in report.theorems
    }


def test_formal_appendix_markdown_lists_every_theorem() -> None:
    appendix = render_formal_appendix_markdown()
    assert "PromptABI Formal Appendix" in appendix
    for step in range(301, 315):
        assert f"### {step}." in appendix
    assert "promptabi metatheory --format json" in appendix


def test_metatheory_cli_outputs_json(capsys) -> None:
    exit_code = main(["metatheory", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert payload["version"] == PROMPT_CALCULUS_METATHEORY_VERSION
    assert payload["theorem_count"] == 14


def test_metatheory_cli_appendix(capsys) -> None:
    exit_code = main(["metatheory", "--appendix"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI Formal Appendix" in captured.out
