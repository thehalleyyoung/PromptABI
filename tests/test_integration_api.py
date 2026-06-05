import json

import promptabi
from promptabi import (
    IntegrationGate,
    IntegrationSurface,
    build_integration_report,
    build_public_api_manifest,
    render_integration_report_json,
    render_integration_report_text,
)


def test_integration_api_builds_ci_ide_registry_and_platform_payloads() -> None:
    report = build_integration_report(
        "examples/minimal/promptabi.json",
        surfaces=[
            IntegrationSurface.CI_PROVIDER,
            IntegrationSurface.IDE_EXTENSION,
            IntegrationSurface.MODEL_REGISTRY,
            IntegrationSurface.INTERNAL_PLATFORM,
        ],
        bundle_key="local-test-key",
        bundle_key_id="ci-test",
    )

    payload = report.to_dict()
    ci = payload["surfaces"]["ci-provider"]
    ide = payload["surfaces"]["ide-extension"]
    registry = payload["surfaces"]["model-registry"]
    platform = payload["surfaces"]["internal-platform"]

    assert report.gate is IntegrationGate.PASS
    assert ci["exit_code"] == 0
    assert ci["sarif"]["version"] == "2.1.0"
    assert ci["check_runtimes"][0]["check"] == "repository-skeleton"
    assert ide["protocol"] == "promptabi.inlineDiagnostics.v1"
    assert ide["diagnostic_count"] == 1
    assert registry["signed_bundle"]["available"] is True
    assert registry["signed_bundle"]["signing_key_id"] == "ci-test"
    assert registry["reproducibility_hash"]
    assert platform["guarantee_modes"]["heuristic"] == 1
    assert "No telemetry is sent" in " ".join(platform["privacy_guarantees"])


def test_integration_api_dataset_surface_uses_real_training_contracts() -> None:
    report = build_integration_report(
        "examples/end-to-end/training-quickstart/fixed.promptabi.json",
        surfaces=["dataset-platform"],
    )

    dataset = report.to_dict()["surfaces"]["dataset-platform"]
    rule_ids = {diagnostic["rule_id"] for diagnostic in dataset["diagnostics"]}

    assert any(artifact["kind"] == "training-manifest" for artifact in dataset["artifacts"])
    assert dataset["diagnostic_count"] > 0
    assert "training-redaction-verified" in rule_ids
    assert all("dataset rows" not in json.dumps(diagnostic) for diagnostic in dataset["diagnostics"])


def test_integration_report_renderers_are_deterministic() -> None:
    report = build_integration_report(
        "examples/minimal/promptabi.json",
        surfaces=["ci-provider", "internal-platform"],
        fail_on="never",
    )

    rendered = render_integration_report_json(report)
    text = render_integration_report_text(report)
    payload = json.loads(rendered)

    assert payload["protocol"] == "promptabi.integration.v1"
    assert payload["gate"] == "pass"
    assert "PromptABI integration report:" in text
    assert render_integration_report_json(report) == rendered


def test_integration_api_symbols_are_stable_public_api() -> None:
    manifest = build_public_api_manifest()
    symbols = manifest.symbol_map()

    required = {
        "IntegrationSurface",
        "IntegrationGate",
        "IntegrationCapability",
        "IntegrationRequest",
        "IntegrationArtifactSummary",
        "IntegrationReport",
        "build_integration_report",
        "render_integration_report_json",
        "render_integration_report_text",
    }

    assert required.issubset(symbols)
    assert all(symbols[name].stability.value == "stable" for name in required)
    assert set(promptabi.STABLE_PUBLIC_API).issubset(set(promptabi.__all__))
