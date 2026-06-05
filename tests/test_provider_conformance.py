import json
from pathlib import Path

import promptabi
from promptabi import (
    REQUIRED_PROVIDER_CONFORMANCE_SURFACES,
    REQUIRED_PROVIDER_FIXTURE_FAMILIES,
    ProviderConformanceReport,
    build_provider_conformance_report,
    render_provider_conformance_json,
    render_provider_conformance_text,
    write_provider_conformance_manifest,
)
from promptabi.cli import main


FIXTURE_ROOT = Path("fixtures/provider_fixture_packs")


def test_provider_conformance_suite_covers_required_surfaces_and_families() -> None:
    report = build_provider_conformance_report(FIXTURE_ROOT)
    coverage = {surface.surface: surface for surface in report.surface_coverage}

    assert report.all_cases_passed is True
    assert report.provider_count >= 6
    assert set(report.provider_families) >= set(REQUIRED_PROVIDER_FIXTURE_FAMILIES)
    assert set(coverage) == set(REQUIRED_PROVIDER_CONFORMANCE_SURFACES)
    assert report.missing_provider_families == ()
    assert report.missing_surfaces == ()
    assert report.replay_findings == ()
    assert len(report.manifest_sha256) == 64
    assert all(surface.passed for surface in report.surface_coverage)


def test_provider_conformance_uses_semantic_provider_behavior_not_presence_only() -> None:
    report = build_provider_conformance_report(FIXTURE_ROOT)
    coverage = {surface.surface: surface for surface in report.surface_coverage}

    parallel = coverage["parallel-tool-calls"]
    json_mode = coverage["json-mode"]
    stop_handling = coverage["stop-handling"]

    assert "gemini-generate-content" not in parallel.provider_ids
    assert "vllm-openai-server" not in parallel.provider_ids
    assert "openai-chat-completions" in parallel.provider_ids
    assert "vllm-openai-server" in json_mode.provider_ids
    assert "anthropic-messages" not in json_mode.provider_ids
    assert set(stop_handling.provider_ids) == {
        "anthropic-messages",
        "bedrock-converse",
        "gemini-generate-content",
        "litellm-router",
        "openai-chat-completions",
        "vllm-openai-server",
    }
    assert any("limit=1" not in detail for _, detail in parallel.evidence)


def test_provider_conformance_renderers_writer_cli_and_public_api(tmp_path: Path, capsys) -> None:
    report = build_provider_conformance_report(FIXTURE_ROOT)
    text = render_provider_conformance_text(report)
    payload = json.loads(render_provider_conformance_json(report))

    assert "PromptABI provider fixture conformance" in text
    assert "tool-call-streaming: PASS" in text
    assert payload["all_cases_passed"] is True
    assert payload["manifest_sha256"] == report.manifest_sha256

    output = tmp_path / "provider-conformance.json"
    written = write_provider_conformance_manifest(output, root=FIXTURE_ROOT)
    assert json.loads(output.read_text(encoding="utf-8")) == written

    api_report = promptabi.provider_conformance_suite(FIXTURE_ROOT)
    api_rendered = promptabi.provider_conformance_suite(FIXTURE_ROOT, output_format="json")
    assert isinstance(api_report, ProviderConformanceReport)
    assert json.loads(api_rendered)["all_cases_passed"] is True

    exit_code = main(["corpus", "provider-conformance", "--root", str(FIXTURE_ROOT), "--format", "text"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "parallel-tool-calls: PASS" in captured.out

    cli_output = tmp_path / "cli-provider-conformance.json"
    exit_code = main(["corpus", "provider-conformance", "--root", str(FIXTURE_ROOT), "--output", str(cli_output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote provider conformance manifest" in captured.out
    assert json.loads(cli_output.read_text(encoding="utf-8"))["all_cases_passed"] is True


def test_provider_conformance_release_gate_reports_malformed_root(tmp_path: Path) -> None:
    root = tmp_path / "provider-fixtures"
    root.mkdir()

    report = promptabi.verify_corpora(provider_fixture_root=root)
    check = next(check for check in report.checks if check.name == "provider-conformance")

    assert report.ok is False
    assert check.passed is False
    assert check.coverage_count == 0
    assert "provider fixture conformance suite could not be replayed" in check.summary
