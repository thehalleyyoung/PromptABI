import json
from pathlib import Path

from promptabi import analyze_tool_call_serialization
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader
from promptabi.session import VerificationSession


FIXTURE_CONFIG = Path("fixtures/structured_schemas/tool-serialization-contract/promptabi.json")


def test_tool_serialization_analyzer_finds_recorded_contract_mismatches() -> None:
    config = load_config(FIXTURE_CONFIG)
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in config.artifact_bundle)

    report = analyze_tool_call_serialization(loaded)

    assert report.checked_pairs == 1
    assert report.providers_checked == ("openai-provider",)
    assert report.tool_artifacts_checked == ("openai-tools",)
    assert {finding.kind.value for finding in report.findings} >= {
        "tool-name-mismatch",
        "argument-encoding-mismatch",
        "argument-escaping-risk",
        "tool-id-mismatch",
        "parallel-call-mismatch",
        "streaming-chunk-mismatch",
        "template-tool-mismatch",
        "stop-serialization-mismatch",
    }
    assert any(finding.span and finding.span.path.endswith("provider.json") for finding in report.findings)
    assert any("cancel_order" in value for finding in report.findings for _key, value in finding.evidence)


def test_tool_serialization_session_diagnostics_are_stable() -> None:
    result = VerificationSession.from_config_file(FIXTURE_CONFIG).run()

    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "tool-serialization"]

    assert result.ok is False
    assert len(diagnostics) >= 8
    assert any("tool-name-mismatch" in diagnostic.message for diagnostic in diagnostics)
    assert any("streaming-chunk-mismatch" in diagnostic.message for diagnostic in diagnostics)
    assert diagnostics[0].check_modes[0].value == "bounded"
    assert all(diagnostic.witness is not None for diagnostic in diagnostics)


def test_tool_serialization_cli_reports_real_fixture(capsys) -> None:
    exit_code = main(["verify", "--config", str(FIXTURE_CONFIG), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [item for item in payload["diagnostics"] if item["rule_id"] == "tool-serialization"]

    assert exit_code == 1
    assert captured.err == ""
    assert diagnostics
    assert any("argument-encoding-mismatch" in item["message"] for item in diagnostics)
    assert any(item["span"]["path"].endswith("provider.json") for item in diagnostics if "span" in item)
    assert any(
        step["action"] == "compare serialization field"
        for item in diagnostics
        for step in item["witness"]["steps"]
    )
