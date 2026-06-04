import json
from pathlib import Path

from promptabi import (
    SUPPORTED_PROVIDER_FAMILIES,
    analyze_provider_migration,
    canonical_provider_family,
)
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader
from promptabi.session import VerificationSession


FIXTURE_CONFIG = Path("fixtures/provider_migration/promptabi.json")


def test_provider_family_aliases_cover_named_migration_targets() -> None:
    aliases = {
        "OpenAI": "openai",
        "Azure OpenAI": "azure-openai",
        "Anthropic": "anthropic",
        "Gemini": "gemini",
        "Bedrock": "bedrock",
        "Together": "together",
        "Groq": "groq",
        "Ollama": "ollama",
        "llama.cpp server": "llama.cpp-server",
        "vLLM OpenAI server": "vllm-openai-server",
        "LiteLLM": "litellm",
    }

    assert set(aliases.values()) == set(SUPPORTED_PROVIDER_FAMILIES)
    assert {canonical_provider_family(alias) for alias in aliases} == set(SUPPORTED_PROVIDER_FAMILIES)
    assert canonical_provider_family("experimental-internal-provider") is None


def test_provider_migration_analyzer_finds_recorded_contract_mismatches() -> None:
    config = load_config(FIXTURE_CONFIG)
    loaded = tuple(ArtifactLoader().load(artifact) for artifact in config.artifact_bundle)

    report = analyze_provider_migration(loaded)

    assert report.migrations_checked == 12
    assert set(report.supported_targets_seen) == set(SUPPORTED_PROVIDER_FAMILIES)
    assert "openai-source" in report.providers_checked
    assert {finding.kind.value for finding in report.findings} >= {
        "unsupported-provider",
        "request-field-loss",
        "response-field-loss",
        "tool-argument-encoding-mismatch",
        "tool-id-mismatch",
        "parallel-tool-call-mismatch",
        "streaming-chunk-mismatch",
        "stop-behavior-mismatch",
        "context-limit-regression",
        "structured-output-mismatch",
        "error-shape-mismatch",
        "routing-target-missing",
    }
    assert any(finding.target_artifact_name == "azure-openai-target" for finding in report.findings) is False
    assert any(
        finding.span and finding.span.path.endswith("anthropic-target.json")
        for finding in report.findings
    )
    assert any("response_format" in value for finding in report.findings for _key, value in finding.evidence)


def test_provider_migration_session_diagnostics_are_stable() -> None:
    result = VerificationSession.from_config_file(FIXTURE_CONFIG).run()

    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "provider-migration"]

    assert result.ok is False
    assert len(diagnostics) >= 20
    assert any("tool-argument-encoding-mismatch" in diagnostic.message for diagnostic in diagnostics)
    assert any("routing-target-missing" in diagnostic.message for diagnostic in diagnostics)
    assert diagnostics[0].check_modes[0].value == "bounded"
    assert all(diagnostic.witness is not None for diagnostic in diagnostics)


def test_provider_migration_cli_reports_real_fixture(capsys) -> None:
    exit_code = main(["verify", "--config", str(FIXTURE_CONFIG), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    diagnostics = [item for item in payload["diagnostics"] if item["rule_id"] == "provider-migration"]

    assert exit_code == 1
    assert captured.err == ""
    assert diagnostics
    assert any("context-limit-regression" in item["message"] for item in diagnostics)
    assert any(item["span"]["path"].endswith("ollama-target.json") for item in diagnostics if "span" in item)
    assert any(
        step["action"] == "compare provider migration field"
        for item in diagnostics
        for step in item["witness"]["steps"]
    )
