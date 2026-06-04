from promptabi import (
    ArtifactRef,
    Diagnostic,
    DiagnosticSeverity,
    SourceSpan,
    VerificationConfig,
    VerificationSession,
)


def test_public_api_result_is_typed_and_deterministic() -> None:
    config = VerificationConfig(name="api-smoke")

    result = VerificationSession(config).run()

    assert result.ok
    assert result.to_dict()["config"]["name"] == "api-smoke"
    assert result.diagnostics[0].rule_id == "repository-skeleton"


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
        "suggestions": [],
        "artifact": {"kind": "config", "name": "promptabi"},
        "span": {"path": "promptabi.json", "start_line": 1, "start_column": 1},
    }

