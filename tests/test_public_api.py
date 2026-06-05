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
    UpstreamIssueLink,
    VerificationConfig,
    VerificationResult,
    VerificationSession,
    WitnessStep,
    WitnessTrace,
    collect_diagnostics,
    create_bug_report,
    create_session,
    diagnostic_message_catalog,
    load_artifacts,
    render_result,
    render_bug_report,
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


def test_embedding_api_creates_upstream_bug_report() -> None:
    result = run_verification("examples/minimal/promptabi.json")

    report = create_bug_report(
        result,
        config_path="examples/minimal/promptabi.json",
        rule_id="repository-skeleton",
    )
    rendered = render_bug_report(report)

    assert report.diagnostic.rule_id == "repository-skeleton"
    assert "## Reproduction" in rendered
    assert "promptabi verify --config examples/minimal/promptabi.json" in rendered
    assert "Privacy note" in rendered


def test_collect_diagnostics_reports_unknown_embedded_check() -> None:
    diagnostics = collect_diagnostics(
        "examples/minimal/promptabi.json",
        selected_checks=["not-registered"],
    )

    assert diagnostics[0].rule_id == "check-unknown"
    assert diagnostics[0].severity is DiagnosticSeverity.ERROR


def test_public_api_builds_diagnostic_message_catalog() -> None:
    result = run_verification("examples/minimal/promptabi.json")

    rendered = diagnostic_message_catalog(result.diagnostics, output_format="text")

    assert "PromptABI diagnostic message catalog" in rendered
    assert "promptabi.diagnostic.repository.skeleton" in rendered


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
        "witness_digest": diagnostic.witness_digest,
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


def test_diagnostic_model_preserves_upstream_issue_links_without_changing_fingerprint() -> None:
    link = UpstreamIssueLink(
        url="https://github.com/example/upstream/pull/42",
        title="Fix tokenizer control-token regression",
        status="merged",
        affected_versions=("tokenizer-fixture-v1",),
        fixed_versions=("tokenizer-fixture-v2",),
        workarounds=("Pin tokenizer revisions until the provider patch is deployed.",),
    )
    base = Diagnostic(
        rule_id="tokenizer-differential-mismatch",
        severity=DiagnosticSeverity.ERROR,
        message="tokenizer fixture drifted",
    )
    linked = Diagnostic(
        rule_id=base.rule_id,
        severity=base.severity,
        message=base.message,
        upstream_issues=(link,),
    )

    payload = linked.to_dict()
    round_tripped = Diagnostic.from_dict(payload)

    assert linked.fingerprint == base.fingerprint
    assert payload["upstream_issues"] == [link.to_dict()]
    assert round_tripped.upstream_issues == (link,)


def test_bug_report_includes_upstream_issue_status_and_workarounds() -> None:
    link = UpstreamIssueLink(
        url="https://github.com/example/upstream/issues/77",
        title="Structured-output parser mismatch",
        status="fixed",
        fixed_versions=("schema-fixture-v2",),
        workarounds=("Normalize the app parser and constrained decoder against the same JSON Schema.",),
    )
    result = VerificationResult(
        config=VerificationConfig(name="linked-report", checks=("parser-compatibility-mismatch",)),
        diagnostics=(
            Diagnostic(
                rule_id="parser-compatibility-mismatch",
                severity=DiagnosticSeverity.ERROR,
                message="parser accepts output outside the grammar",
                upstream_issues=(link,),
            ),
        ),
    )

    rendered = render_bug_report(create_bug_report(result, rule_id="parser-compatibility-mismatch"))

    assert "## Upstream status" in rendered
    assert "[Structured-output parser mismatch](https://github.com/example/upstream/issues/77)" in rendered
    assert "Normalize the app parser" in rendered
