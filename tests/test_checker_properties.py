import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import promptabi
from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    BoolDomain,
    ByteLevelTokenizer,
    ChatTemplateRenderCase,
    ChatTemplateSymbolicBounds,
    Eq,
    FiniteContractProblem,
    FrameworkTruncationConfigArtifact,
    GrammarArtifact,
    GrammarDifferentialStatus,
    GrammarTokenizerAmbiguityKind,
    GrammarTokenizerEmptinessStatus,
    IntRangeDomain,
    Le,
    NamedConstraint,
    ParserCompatibilityStatus,
    PromptSegment,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    SchemaArtifact,
    SolverStatus,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    TokenizerDifferentialCase,
    TokenizerExpectation,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TruncationStrategy,
    Value,
    Var,
    VerificationConfig,
    analyze_grammar_differential_mapping,
    analyze_parser_compatibility,
    analyze_provider_fixture_replay,
    analyze_provider_migration,
    analyze_role_boundary_nonforgeability,
    analyze_static_contracts,
    analyze_stop_differential,
    analyze_stop_overreachability,
    analyze_stop_policy_tokenizer,
    analyze_token_budget,
    analyze_tokenizer_config_drift,
    analyze_tokenizer_grammar_ambiguity,
    analyze_tokenizer_grammar_emptiness,
    analyze_tool_call_serialization,
    parse_hf_chat_template_config,
    run_chat_template_differential,
    run_evaluation,
    run_mutation_fuzzing,
    run_tokenizer_differential,
    run_verification,
)
from promptabi.artifacts import ArtifactProvenance
from promptabi.loaders import ArtifactLoader, LoadedArtifact


PUBLIC_CHECKER_EXPORTS = {
    "analyze_grammar_differential_corpus",
    "analyze_grammar_differential_mapping",
    "analyze_parser_compatibility",
    "analyze_provider_fixture_replay",
    "analyze_provider_migration",
    "analyze_role_boundary_nonforgeability",
    "analyze_stop_differential",
    "analyze_stop_overreachability",
    "analyze_stop_policy_tokenizer",
    "analyze_token_budget",
    "analyze_tokenizer_config_drift",
    "analyze_tokenizer_grammar_ambiguity",
    "analyze_tokenizer_grammar_emptiness",
    "analyze_tool_call_serialization",
    "analyze_training_inference_bridge",
    "analyze_training_invalid_interface",
    "analyze_synthetic_generators",
    "analyze_training_metadata_drift",
    "analyze_training_packing",
    "run_beta_program",
    "run_chat_template_differential",
    "run_compatibility_audit",
    "run_corpus_verification",
    "run_evaluation",
    "run_mutation_fuzzing",
    "run_tokenizer_differential",
    "run_verification",
}


@dataclass(frozen=True, slots=True)
class _GeneratedCheckerCase:
    checker: str
    status: str


def test_property_suite_tracks_every_public_checker_export() -> None:
    exported = {name for name in promptabi.__all__ if name.startswith(("analyze_", "run_"))}

    assert exported == PUBLIC_CHECKER_EXPORTS


def test_generated_tokenizer_and_template_differentials_cover_safe_and_unsafe_cases() -> None:
    tokenizer = ByteLevelTokenizer(added_tokens=("<stop>",), special_tokens={"<stop>": 2})
    cases = []
    for text in ("plain", "hi<stop>", "unicode-é"):
        encoded = tokenizer.encode(text)
        cases.append(
            TokenizerDifferentialCase(
                name=f"byte-{len(cases)}",
                text=text,
                expectation=TokenizerExpectation(
                    token_ids=encoded.token_ids,
                    decoded_text=tokenizer.decode(encoded.token_ids).text,
                    token_texts=encoded.token_texts,
                    normalized_text=encoded.normalized_text,
                    special_token_ids=frozenset(token.token_id for token in encoded.tokens if token.special),
                    added_token_ids=frozenset(token.token_id for token in encoded.tokens if token.added),
                    byte_spans_required=True,
                    round_trip_normalized=True,
                ),
            )
        )

    token_report = run_tokenizer_differential(tokenizer, cases)
    assert token_report.ok
    assert token_report == run_tokenizer_differential(tokenizer, cases)

    parsed = parse_hf_chat_template_config(
        {
            "chat_template": (
                "{% for message in messages %}"
                "<|{{ message['role'] }}|>{{ message['content'] }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}"
            )
        }
    )
    safe_case = ChatTemplateRenderCase(
        name="expected-render",
        messages=({"role": "user", "content": "hello"},),
        expected_rendered="<|user|>hello<|assistant|>",
        add_generation_prompt=True,
    )
    unsafe_case = ChatTemplateRenderCase(
        name="wrong-oracle",
        messages=({"role": "user", "content": "hello"},),
        expected_rendered="<|user|>HELLO<|assistant|>",
        add_generation_prompt=True,
    )

    ok_report = run_chat_template_differential(parsed, (safe_case,))
    mismatch_report = run_chat_template_differential(parsed, (unsafe_case,))

    assert ok_report.mismatches == ()
    assert [mismatch.field for mismatch in mismatch_report.mismatches] == ["rendered"]


def test_generated_role_stop_and_budget_properties_exercise_safe_unsafe_and_abstaining_cases(
    tmp_path: Path,
) -> None:
    generated = []
    for marker in ("<|im_start|>", "<tool_call>"):
        unsafe_template = parse_hf_chat_template_config(
            {
                "chat_template": (
                    "{% for message in messages %}"
                    f"{marker}{{{{ message['role'] }}}}\n"
                    f"{{{{ message['content'] }}}}{marker}"
                    "{% endfor %}"
                ),
                "additional_special_tokens": [marker],
            }
        )
        safe_template = parse_hf_chat_template_config(
            {
                "chat_template": (
                    "{% for message in messages %}"
                    f"user\n{{{{ message['content']|tojson }}}}{marker}"
                    "{% endfor %}"
                ),
                "additional_special_tokens": [marker],
            }
        )

        unsafe = analyze_role_boundary_nonforgeability(
            unsafe_template,
            bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1),
        )
        safe = analyze_role_boundary_nonforgeability(
            safe_template,
            bounds=ChatTemplateSymbolicBounds(max_messages=1, max_tools=0, max_loop_iterations=1),
        )

        assert not unsafe.ok
        assert safe.ok
        generated.extend(
            [
                _GeneratedCheckerCase("analyze_role_boundary_nonforgeability", "unsafe"),
                _GeneratedCheckerCase("analyze_role_boundary_nonforgeability", "safe"),
            ]
        )

    tokenizer = ByteLevelTokenizer(added_tokens=("<STOP>",), special_tokens={"<STOP>": 2})
    generated_stop_policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="generated-stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("END", "ENDIF", "<STOP>"),
        stop_token_ids=(2, 999999),
    )
    stop_report = analyze_stop_policy_tokenizer(generated_stop_policy, tokenizer)
    assert stop_report.collisions
    assert stop_report.special_interactions
    assert stop_report.unreachable_token_ids
    assert stop_report == analyze_stop_policy_tokenizer(generated_stop_policy, tokenizer)
    generated.append(_GeneratedCheckerCase("analyze_stop_policy_tokenizer", "ambiguous"))

    schema_path = tmp_path / "stop.schema.json"
    unsupported_schema_path = tmp_path / "unsupported-stop.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["comment"],
                "properties": {"comment": {"type": "string"}},
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    unsupported_schema_path.write_text(json.dumps({"type": "array", "items": {"type": "integer"}}), encoding="utf-8")
    overreach = analyze_stop_overreachability(
        StopPolicyArtifact(
            kind=ArtifactKind.STOP_POLICY,
            name="content-stop",
            location=ArtifactLocation(uri="memory://stops"),
            stop_sequences=("</tool_call>",),
        ),
        (
            SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name="generated-schema",
                location=ArtifactLocation(path=str(schema_path)),
            ),
            SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name="unsupported",
                location=ArtifactLocation(path=str(unsupported_schema_path)),
            ),
        ),
    )
    assert overreach.content_findings
    assert overreach.abstentions
    generated.extend(
        [
            _GeneratedCheckerCase("analyze_stop_overreachability", "unsafe"),
            _GeneratedCheckerCase("analyze_stop_overreachability", "abstaining"),
        ]
    )

    provider_path = tmp_path / "stop-provider.json"
    provider_path.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "request_shape": {"fields": ["messages", "stop"]},
                "response_shape": {"fields": ["choices"]},
                "stop_trace": {
                    "name": "generated-stop-mismatch",
                    "family": "openai-compatible",
                    "chunks": ["safe", "END", "tail"],
                    "expected": {
                        "stopped": True,
                        "output": "safeEND",
                        "matched_stop": "END",
                        "finish_reason": "stop",
                        "include_stop_in_output": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    differential = analyze_stop_differential(
        StopPolicyArtifact(
            kind=ArtifactKind.STOP_POLICY,
            name="stops",
            location=ArtifactLocation(uri="memory://stops"),
            stop_sequences=("END",),
        ),
        (
            ProviderConfigArtifact(
                kind=ArtifactKind.PROVIDER_CONFIG,
                name="provider",
                location=ArtifactLocation(path=str(provider_path)),
                provider="openai-compatible",
            ),
        ),
    )
    assert differential.mismatches
    generated.append(_GeneratedCheckerCase("analyze_stop_differential", "unsafe"))

    for total_tokens, budget, expected_status in ((20, 32, "safe"), (80, 32, "unsafe")):
        budget_report = analyze_token_budget(
            VerificationConfig(name=f"budget-{expected_status}"),
            (
                LoadedArtifact(
                    artifact=PromptSegmentArtifact(
                        kind=ArtifactKind.PROMPT_SEGMENT,
                        name="segments",
                        location=ArtifactLocation(uri="memory://segments"),
                        segments=(
                            PromptSegment(
                                "system",
                                role="system",
                                required=True,
                                token_count=total_tokens,
                            ),
                        ),
                    ),
                    source_type="memory",
                    pinned=True,
                    resolved=True,
                ),
                LoadedArtifact(
                    artifact=FrameworkTruncationConfigArtifact(
                        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                        name="budget",
                        location=ArtifactLocation(uri="memory://budget"),
                        framework="vllm",
                        strategy=TruncationStrategy.LEFT,
                        max_context_tokens=budget,
                    ),
                    source_type="memory",
                    pinned=True,
                    resolved=True,
                ),
            ),
        )
        has_error = any(finding.severity == "error" for finding in budget_report.findings)
        assert has_error is (expected_status == "unsafe")
        generated.append(_GeneratedCheckerCase("analyze_token_budget", expected_status))

    assert {case.status for case in generated} >= {"safe", "unsafe", "ambiguous", "abstaining"}


def test_generated_grammar_parser_and_solver_properties_cover_sat_empty_ambiguous_and_abstain(
    tmp_path: Path,
) -> None:
    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte",
        location=ArtifactLocation(uri="memory://byte"),
        family="byte-level",
    )
    byte = ByteLevelTokenizer()
    lowercase = ByteLevelTokenizer(normalization=("lowercase",))

    sat_path = _write_json(tmp_path / "sat.schema.json", {"const": "OK"})
    empty_path = _write_json(tmp_path / "empty.schema.json", {"const": "OK"})
    enum_path = _write_json(tmp_path / "enum.schema.json", {"enum": ["OK", "ok"]})

    sat_report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        SchemaArtifact(kind=ArtifactKind.SCHEMA, name="sat", location=ArtifactLocation(path=str(sat_path))),
        byte,
    )
    empty_report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        SchemaArtifact(kind=ArtifactKind.SCHEMA, name="empty", location=ArtifactLocation(path=str(empty_path))),
        lowercase,
    )
    abstain_report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        GrammarArtifact(
            kind=ArtifactKind.GRAMMAR,
            name="remote-regex",
            location=ArtifactLocation(uri="memory://regex"),
            grammar_type="regex",
        ),
        byte,
    )

    assert sat_report.status is GrammarTokenizerEmptinessStatus.SATISFIABLE
    assert empty_report.status is GrammarTokenizerEmptinessStatus.EMPTY
    assert abstain_report.status is GrammarTokenizerEmptinessStatus.ABSTAINED
    assert sat_report == analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        SchemaArtifact(kind=ArtifactKind.SCHEMA, name="sat", location=ArtifactLocation(path=str(sat_path))),
        byte,
    )

    ambiguity = analyze_tokenizer_grammar_ambiguity(
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="lower-byte",
            location=ArtifactLocation(uri="memory://byte"),
            family="byte-level",
            metadata=(("normalization", ("lowercase",)),),
        ),
        SchemaArtifact(kind=ArtifactKind.SCHEMA, name="enum", location=ArtifactLocation(path=str(enum_path))),
        lowercase,
    )
    if importlib.util.find_spec("jsonschema") is None:
        assert ambiguity.abstained
    else:
        assert any(
            finding.kind is GrammarTokenizerAmbiguityKind.TOKEN_PATH_CONFLICT
            for finding in ambiguity.findings
        )

    agreement = analyze_grammar_differential_mapping(
        {
            "version": 1,
            "cases": [
                {
                    "id": "generated-choice",
                    "backend_family": "outlines",
                    "declared_type": "outlines",
                    "artifact": {"choices": ["yes"]},
                    "accepts": ["yes"],
                    "rejects": ["no"],
                }
            ],
        }
    )
    mismatch = analyze_grammar_differential_mapping(
        {
            "version": 1,
            "cases": [
                {
                    "id": "generated-mismatch",
                    "backend_family": "outlines",
                    "declared_type": "outlines",
                    "artifact": {"choices": ["yes"]},
                    "accepts": ["yes"],
                    "rejects": ["yes"],
                }
            ],
        }
    )
    unsupported = analyze_grammar_differential_mapping(
        {
            "version": 1,
            "cases": [
                {
                    "id": "generated-unsupported",
                    "backend_family": "custom",
                    "declared_type": "unknown",
                    "artifact": {"pattern": "(?=unsafe)"},
                    "accepts": ["unsafe"],
                    "rejects": ["safe"],
                }
            ],
        }
    )
    assert agreement.cases[0].status is GrammarDifferentialStatus.AGREEMENT
    assert mismatch.mismatches
    assert unsupported.abstentions

    parser_mismatch = analyze_parser_compatibility(
        SchemaArtifact(
            kind=ArtifactKind.SCHEMA,
            name="parser-mismatch",
            location=ArtifactLocation(path=str(sat_path)),
            metadata=(("parser_format", "json"),),
        )
    )
    parser_abstain = analyze_parser_compatibility(
        GrammarArtifact(
            kind=ArtifactKind.GRAMMAR,
            name="parser-abstain",
            location=ArtifactLocation(uri="memory://missing"),
            grammar_type="regex",
            metadata=(("parser_format", "custom-delimited"),),
        )
    )
    assert parser_mismatch.status is ParserCompatibilityStatus.MISMATCH
    assert parser_abstain.status is ParserCompatibilityStatus.ABSTAINED

    sat_problem = FiniteContractProblem(
        name="sat",
        variables=(BoolDomain("unsafe"),),
        constraints=(NamedConstraint("unsafe", Eq(Var("unsafe"), Value(True))),),
    )
    unsat_problem = FiniteContractProblem(
        name="unsat",
        variables=(IntRangeDomain("tokens", 0, 4),),
        constraints=(NamedConstraint("too-large", Le(Value(10), Var("tokens"))),),
    )
    assert sat_problem.solve(prefer_z3=False).status is SolverStatus.SAT
    assert unsat_problem.solve(prefer_z3=False).status is SolverStatus.UNSAT

    class _UnsupportedExpression:
        def evaluate(self, assignment):
            del assignment
            return False

        def to_z3(self, context):
            del context
            raise TypeError("unsupported property expression")

        def to_dict(self):
            return {"unsupported": True}

    unknown = FiniteContractProblem(
        name="unknown",
        variables=(BoolDomain("flag"),),
        constraints=(NamedConstraint("unsupported", _UnsupportedExpression()),),
    ).solve(prefer_z3=True)
    if importlib.util.find_spec("z3") is None:
        assert unknown.status in {SolverStatus.SAT, SolverStatus.UNSAT}
    else:
        assert unknown.status is SolverStatus.UNKNOWN


def test_generated_static_provider_tool_and_drift_properties(tmp_path: Path) -> None:
    loaded_static_safe = (
        _loaded(
            PromptSegmentArtifact(
                kind=ArtifactKind.PROMPT_SEGMENT,
                name="segments",
                location=ArtifactLocation(uri="memory://segments"),
                segments=(PromptSegment("system", role="system", required=True, token_count=8),),
            )
        ),
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="budget",
                location=ArtifactLocation(uri="memory://budget"),
                framework="vllm",
                max_context_tokens=32,
            )
        ),
        _loaded(
            StopPolicyArtifact(
                kind=ArtifactKind.STOP_POLICY,
                name="stops",
                location=ArtifactLocation(uri="memory://stops"),
                stop_sequences=("END",),
            )
        ),
        _loaded(
            SpecialTokenMapArtifact(
                kind=ArtifactKind.SPECIAL_TOKEN_MAP,
                name="specials",
                location=ArtifactLocation(uri="memory://specials"),
                tokens=(SpecialToken("eos", "</s>", 2),),
            )
        ),
    )
    loaded_static_unsafe = (
        *loaded_static_safe[:1],
        _loaded(
            FrameworkTruncationConfigArtifact(
                kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
                name="budget",
                location=ArtifactLocation(uri="memory://budget"),
                framework="vllm",
                max_context_tokens=4,
            )
        ),
        _loaded(
            StopPolicyArtifact(
                kind=ArtifactKind.STOP_POLICY,
                name="stops",
                location=ArtifactLocation(uri="memory://stops"),
                stop_sequences=("</s>",),
            )
        ),
        loaded_static_safe[-1],
        _loaded(
            TrainingManifestArtifact(
                kind=ArtifactKind.TRAINING_MANIFEST,
                name="train",
                location=ArtifactLocation(uri="memory://train"),
                target_roles=("critic",),
            )
        ),
    )

    safe_static = analyze_static_contracts(VerificationConfig(name="static-safe"), loaded_static_safe, prefer_z3=False)
    unsafe_static = analyze_static_contracts(VerificationConfig(name="static-unsafe"), loaded_static_unsafe, prefer_z3=False)
    assert not safe_static.violations
    assert unsafe_static.violations

    tool_path = _write_json(
        tmp_path / "tools.json",
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "parameters": {
                        "type": "object",
                        "required": ["order_id"],
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            }
        ],
    )
    source_provider_path = _write_json(tmp_path / "source-provider.json", _provider_fixture("openai", target="target"))
    target_provider_path = _write_json(
        tmp_path / "target-provider.json",
        _provider_fixture(
            "anthropic",
            request_fields=("messages",),
            response_fields=("content",),
            argument_encoding="json-object",
        ),
    )
    bad_replay_path = _write_json(
        tmp_path / "bad-replay.json",
        _provider_fixture("openai", edge_surface="response.not_recorded"),
    )
    loader = ArtifactLoader()
    loaded_tool = loader.load(
        ToolDefinitionArtifact(
            kind=ArtifactKind.TOOL_DEFINITION,
            name="tools",
            location=ArtifactLocation(path=str(tool_path)),
            provider="openai",
        )
    )
    loaded_source = loader.load(
        ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name="source",
            location=ArtifactLocation(path=str(source_provider_path)),
            provider="openai",
        )
    )
    loaded_target = loader.load(
        ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name="target",
            location=ArtifactLocation(path=str(target_provider_path)),
            provider="anthropic",
        )
    )
    loaded_bad_replay = loader.load(
        ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name="bad-replay",
            location=ArtifactLocation(path=str(bad_replay_path)),
            provider="openai",
        )
    )

    tool_report = analyze_tool_call_serialization((loaded_tool, loaded_source))
    migration_report = analyze_provider_migration((loaded_source, loaded_target))
    replay_report = analyze_provider_fixture_replay((loaded_bad_replay,))
    assert tool_report.findings
    assert migration_report.findings
    assert replay_report.findings

    clean = tmp_path / "clean-tokenizer"
    drifted = tmp_path / "drifted-tokenizer"
    _write_tokenizer_revision(clean, eos_id=2, template="{{ messages[0].content }}")
    _write_tokenizer_revision(drifted, eos_id=3, template="<|assistant|>{{ messages[0].content }}")
    clean_report = analyze_tokenizer_config_drift(
        (
            loader.load(
                TokenizerArtifact(
                    kind=ArtifactKind.TOKENIZER,
                    name="clean",
                    location=ArtifactLocation(path=str(clean)),
                    provenance=ArtifactProvenance(version="same"),
                    metadata=(("drift_baseline_path", "."),),
                )
            ),
        )
    )
    drift_report = analyze_tokenizer_config_drift(
        (
            loader.load(
                TokenizerArtifact(
                    kind=ArtifactKind.TOKENIZER,
                    name="drifted",
                    location=ArtifactLocation(path=str(drifted)),
                    provenance=ArtifactProvenance(version="current"),
                    metadata=(("drift_baseline_path", "../clean-tokenizer"),),
                )
            ),
        )
    )
    abstain_report = analyze_tokenizer_config_drift(
        (
            LoadedArtifact(
                TokenizerArtifact(
                    kind=ArtifactKind.TOKENIZER,
                    name="missing-baseline",
                    location=ArtifactLocation(path=str(clean)),
                    metadata=(("drift_baseline_path", "../does-not-exist"),),
                ),
                source_type="tokenizer-directory",
                pinned=True,
                resolved=True,
            ),
        )
    )
    assert clean_report.findings == ()
    assert drift_report.findings
    assert abstain_report.abstentions


def test_generated_end_to_end_run_properties_cover_api_evaluation_and_fuzzing(tmp_path: Path) -> None:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(json.dumps({"name": "generated-run", "checks": ["repository-skeleton"]}), encoding="utf-8")
    result = run_verification(config_path)
    assert result.ok
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["repository-skeleton"]

    evaluation = run_evaluation()
    assert evaluation.results
    assert evaluation.score.precision >= 0
    assert evaluation.score.recall >= 0

    fuzz_report = run_mutation_fuzzing(("stop-policies", "smt-encodings"))
    assert fuzz_report.baseline_results
    assert fuzz_report.mutation_results
    assert {"stop-policies", "smt-encodings"} == {surface.value for surface in fuzz_report.surfaces}


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _loaded(artifact) -> LoadedArtifact:
    return LoadedArtifact(artifact=artifact, source_type="memory", pinned=True, resolved=True)


def _provider_fixture(
    provider: str,
    *,
    target: str | None = None,
    request_fields: tuple[str, ...] = ("messages", "tools", "response_format"),
    response_fields: tuple[str, ...] = ("choices", "tool_calls"),
    argument_encoding: str = "json-string",
    edge_surface: str = "response.fields",
) -> dict[str, object]:
    migration = {"targets": [target]} if target else {}
    return {
        "provider": provider,
        "provider_family": provider,
        "api_family": provider,
        "request": {"method": "POST", "endpoint": "/v1/chat/completions", "fields": list(request_fields)},
        "response": {
            "fields": list(response_fields),
            "finish_reasons": ["stop"],
            "tool_calls": {
                "name_path": "choices[].message.tool_calls[].function.name",
                "arguments_path": "choices[].message.tool_calls[].function.arguments",
                "argument_encoding": argument_encoding,
                "supports_parallel_tool_calls": provider == "openai",
            },
        },
        "stops": {
            "sequences": ["END"],
            "finish_reason_path": "choices[].finish_reason",
            "truncates_before_parser": True,
        },
        "streaming": {
            "delta_path": "choices[].delta",
            "emits_argument_fragments": True,
            "assembly_key": "tool_calls[].index",
        },
        "errors": {"code_path": "error.code", "message_path": "error.message", "rate_limit_path": "error.type"},
        "limits": {"max_input_tokens": 1000 if provider == "openai" else 100, "max_output_tokens": 256},
        "edge_cases": [{"id": "generated-edge", "surface": edge_surface, "expected_behavior": "recorded"}],
        "tool_serialization": {
            "request": {"tool_names": ["lookup_order"], "argument_encoding": "json-object"},
            "response": {
                "tool_names": ["lookup_order"],
                "argument_encoding": argument_encoding,
                "argument_escaping": "raw",
                "id_path": "choices[].message.tool_calls[].id",
                "supports_parallel_tool_calls": provider == "openai",
                "observed_parallel_tool_calls": provider == "openai",
                "tool_call_stop_sequence": "END",
            },
            "parser": {
                "accepted_tool_names": ["lookup_order"],
                "argument_encoding": "json-object",
                "require_tool_call_id": True,
                "allow_parallel_tool_calls": False,
                "streaming_mode": "complete-json",
            },
            "streaming": {"emits_argument_fragments": True},
        },
        "migration_compatibility": {
            "provider_family": provider,
            "request": {"fields": list(request_fields)},
            "response": {"fields": list(response_fields)},
            "tools": {
                "argument_encoding": argument_encoding,
                "id_path": "choices[].message.tool_calls[].id",
                "supports_parallel_tool_calls": provider == "openai",
            },
            "streaming": {"emits_argument_fragments": True},
            "stops": {"sequences": ["END"]},
            "limits": {"max_input_tokens": 1000 if provider == "openai" else 100, "max_output_tokens": 256},
            "structured_outputs": {"modes": ["json_schema"] if provider == "openai" else []},
            "errors": {"code_path": "error.code", "rate_limit_path": "error.type"},
            "routing": {"routes_to": [provider]},
        },
        "provider_migration": migration,
    }


def _write_tokenizer_revision(root: Path, *, eos_id: int, template: str) -> None:
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "</s>",
                "eos_token_id": eos_id,
                "chat_template": template,
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {"normalizer": {"type": "NFC"}, "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos_id, "stop_strings": ["</s>"]}, sort_keys=True),
        encoding="utf-8",
    )
