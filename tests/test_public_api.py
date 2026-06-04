from promptabi import (
    ArtifactBundle,
    ArtifactRef,
    ArtifactKind,
    ArtifactLocation,
    CheckContext,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    SchemaArtifact,
    SourceSpan,
    VerificationConfig,
    VerificationSession,
    WitnessStep,
    WitnessTrace,
    collect_diagnostics,
    create_session,
    load_artifacts,
    render_result,
    run_verification,
)


def test_public_api_result_is_typed_and_deterministic() -> None:
    config = VerificationConfig(
        name="api-smoke",
        artifact_bundle=ArtifactBundle(
            (
                SchemaArtifact(
                    kind=ArtifactKind.SCHEMA,
                    name="schema",
                    location=ArtifactLocation(uri="memory://schema"),
                ),
            )
        ),
    )

    result = VerificationSession(config).run()

    assert result.ok
    assert result.to_dict()["config"]["name"] == "api-smoke"
    assert result.to_dict()["config"]["artifact_bundle"]["artifacts"][0]["kind"] == "schema"
    assert result.diagnostics[0].rule_id == "repository-skeleton"
    assert result.diagnostics[0].check_modes == (CheckMode.HEURISTIC,)


def test_embedding_api_loads_real_artifacts_and_renders_result() -> None:
    config_path = "examples/minimal/promptabi.json"

    loaded = load_artifacts(config_path)
    result = run_verification(config_path)
    rendered = render_result(result, output_format="json")

    assert [artifact.artifact.name for artifact in loaded] == ["messages", "schema", "tools"]
    assert result.ok
    assert '"minimal-chat-template"' in rendered


def test_embedding_api_supports_custom_checks() -> None:
    def requires_schema(context: CheckContext):
        schema = context.artifact("schema")
        yield Diagnostic(
            rule_id="embedded-schema-present",
            severity=DiagnosticSeverity.INFO,
            message=f"loaded {schema.artifact.kind.value} artifact",
            artifact=schema.artifact.to_ref(),
            check_modes=(CheckMode.SOUND,),
        )

    session = create_session(
        "examples/minimal/promptabi.json",
        checks={"embedded-schema-present": requires_schema},
    )

    result = session.run(checks=["embedded-schema-present"])

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["embedded-schema-present"]
    assert result.diagnostics[0].artifact is not None
    assert result.diagnostics[0].artifact.name == "schema"


def test_collect_diagnostics_reports_unknown_embedded_check() -> None:
    diagnostics = collect_diagnostics(
        "examples/minimal/promptabi.json",
        selected_checks=["not-registered"],
    )

    assert diagnostics[0].rule_id == "check-unknown"
    assert diagnostics[0].severity is DiagnosticSeverity.ERROR


def test_diagnostic_to_dict_omits_absent_optional_fields() -> None:
    diagnostic = Diagnostic(
        rule_id="demo",
        severity=DiagnosticSeverity.WARNING,
        message="example",
        artifact=ArtifactRef(kind="config", name="promptabi"),
        span=SourceSpan(path="promptabi.json"),
    )

    assert diagnostic.to_dict() == {
        "rule_id": "demo",
        "severity": "warning",
        "message": "example",
        "fingerprint": diagnostic.fingerprint,
        "suggestions": [],
        "check_modes": [],
        "artifact": {"kind": "config", "name": "promptabi"},
        "span": {"path": "promptabi.json", "start_line": 1, "start_column": 1},
    }


def test_diagnostic_model_preserves_provenance_witness_steps_and_stable_fingerprint() -> None:
    artifact = ArtifactRef(
        kind="chat-template",
        name="llama-template",
        uri="hf://meta-llama/example",
        revision="abc123",
        sha256="deadbeef",
        license="llama",
        source="huggingface",
    )
    witness = WitnessTrace(
        summary="user content reaches a role delimiter",
        steps=(
            WitnessStep(action="render template", input="user.content", output="<|assistant|>"),
            "tokenize rendered prompt",
        ),
        artifacts=(artifact,),
    )
    first = Diagnostic(
        rule_id="role-boundary-nonforgeability",
        severity=DiagnosticSeverity.ERROR,
        message="user content can render an assistant delimiter",
        artifact=artifact,
        span=SourceSpan(path="tokenizer_config.json", start_line=12, start_column=3),
        witness=witness,
        suggestions=("Escape user content before inserting role delimiters.",),
        check_modes=(CheckMode.SOUND, "bounded", CheckMode.Z3_BACKED_SMT),
    )
    second = Diagnostic(
        rule_id=first.rule_id,
        severity=first.severity,
        message=first.message,
        artifact=artifact,
        span=SourceSpan(path="tokenizer_config.json", start_line=12, start_column=3),
        witness=witness,
        suggestions=first.suggestions,
        check_modes=("z3-backed-smt", CheckMode.BOUNDED, CheckMode.SOUND),
    )

    payload = first.to_dict()

    assert first.fingerprint == second.fingerprint
    assert payload["artifact"] == {
        "kind": "chat-template",
        "name": "llama-template",
        "uri": "hf://meta-llama/example",
        "revision": "abc123",
        "sha256": "deadbeef",
        "license": "llama",
        "source": "huggingface",
    }
    assert payload["witness"]["steps"] == [
        {"action": "render template", "input": "user.content", "output": "<|assistant|>"},
        {"action": "tokenize rendered prompt"},
    ]
    assert payload["check_modes"] == ["bounded", "sound", "z3-backed-smt"]
    assert CheckMode.ABSTAINING.description.startswith("The check explicitly declines")
