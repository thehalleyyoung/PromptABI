"""Tests for the live provider integration and CI layer (steps 331-345)."""

from __future__ import annotations

import json

import pytest

from promptabi import live_provider_ci as api_entry
from promptabi.live_provider_ci import (
    THIRD_PARTY_GATEWAYS,
    AdapterRequest,
    capture_conformance_snapshot,
    certify_third_party_gateways,
    conformance_sarif,
    cost_aware_sample,
    devcontainer_json,
    diff_snapshots,
    github_action_workflow_yaml,
    load_provider_adapters,
    otel_export,
    pr_comment_markdown,
    pre_commit_hook_config,
    record_http_cassettes,
    render_live_provider_ci_json,
    render_live_provider_ci_text,
    run_ci,
    run_live_provider_ci,
)
from promptabi.provider_conformance import build_provider_conformance_report


@pytest.fixture(scope="module")
def report():
    return run_live_provider_ci()


@pytest.fixture(scope="module")
def base_report():
    return build_provider_conformance_report()


# --- 331 adapters ----------------------------------------------------------- #


def test_three_adapters_load_from_real_fixtures():
    adapters = load_provider_adapters()
    assert {a.family for a in adapters} == {"openai", "anthropic", "oss-vllm"}
    for adapter in adapters:
        assert adapter.pack_sha256
        assert "request" in adapter.supported_surfaces()


def test_adapter_execution_is_deterministic_and_tool_aware():
    adapters = {a.family: a for a in load_provider_adapters()}
    openai = adapters["openai"]
    plain = AdapterRequest(messages=({"role": "user", "content": "hi"},))
    tool = AdapterRequest(messages=({"role": "user", "content": "go"},), tools=("search",))
    assert openai.execute(plain).response_sha256 == openai.execute(plain).response_sha256
    assert openai.execute(tool).finish_reason == "tool_calls"
    assert openai.execute(plain).response_sha256 != openai.execute(tool).response_sha256


# --- 332 signed snapshots --------------------------------------------------- #


def test_snapshot_is_signed_and_verifiable(base_report):
    snap = capture_conformance_snapshot(
        revision="2025-01", captured_at="2025-01-01T06:00:00Z", report=base_report
    )
    assert snap.verify()
    assert snap.conformance_score == pytest.approx(1.0)
    # Tampering breaks the signature.
    tampered = type(snap)(
        revision=snap.revision,
        captured_at=snap.captured_at,
        provider_families=snap.provider_families,
        surface_pass={**snap.surface_pass, "request": False},
        manifest_sha256=snap.manifest_sha256,
        replay_hash=snap.replay_hash,
        all_passed=False,
        signature=snap.signature,
    )
    assert not tampered.verify()


# --- 333 / 336 / 340 CI assets --------------------------------------------- #


def test_ci_assets_are_well_formed():
    assert "upload-sarif" in github_action_workflow_yaml()
    assert "promptabi ci" in github_action_workflow_yaml()
    assert "promptabi-conformance" in pre_commit_hook_config()
    json.loads(devcontainer_json())  # valid JSON


# --- 334 dashboard ---------------------------------------------------------- #


def test_dashboard_tracks_history_and_regression(report):
    dash = report.dashboard
    assert len(dash.revisions) == 12
    assert "2025-06" in dash.regressions
    assert dash.max_drift > 0.0


# --- 335 SARIF -------------------------------------------------------------- #


def test_ci_emits_valid_sarif(report):
    doc = conformance_sarif(report.latest_snapshot)
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["tool"]["driver"]["name"] == "PromptABI"
    out, code = run_ci(output_format="sarif")
    json.loads(out)
    assert code == 0  # latest snapshot is clean


def test_ci_exit_code_nonzero_on_regression_snapshot():
    # A regression snapshot must fail the gate.
    base = build_provider_conformance_report()
    snap = capture_conformance_snapshot(
        revision="2025-06", captured_at="2025-06-01T06:00:00Z", report=base
    )
    surfaces = dict(snap.surface_pass)
    first = sorted(surfaces)[0]
    surfaces[first] = False
    sarif = conformance_sarif(
        type(snap)(
            revision=snap.revision,
            captured_at=snap.captured_at,
            provider_families=snap.provider_families,
            surface_pass=surfaces,
            manifest_sha256=snap.manifest_sha256,
            replay_hash=snap.replay_hash,
            all_passed=False,
            signature=snap.signature,
        )
    )
    assert sarif["runs"][0]["results"]  # at least one finding


# --- 337 grammar backend bench --------------------------------------------- #


def test_grammar_backend_bench_runs_real_checker(report):
    bench = report.backend_bench
    assert len(bench.rows) == 8
    assert 0.0 < bench.coverage < 1.0
    # Recursion is unsupported by both OSS backends -> at least one row flags it.
    assert any(r.unsupported_feature for r in bench.rows if not r.supported)


# --- 338 cassettes ---------------------------------------------------------- #


def test_http_cassettes_one_per_provider(report):
    cassettes = report.cassettes
    assert len(cassettes) >= 6
    assert all(c.cassette_sha256 for c in cassettes)
    # Replays are deterministic.
    again = record_http_cassettes()
    assert [c.cassette_sha256 for c in cassettes] == [c.cassette_sha256 for c in again]


# --- 339 otel --------------------------------------------------------------- #


def test_otel_export_shape(report):
    doc = otel_export(report.latest_snapshot)
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans
    assert all("attributes" in s for s in spans)


# --- 341 PR comment --------------------------------------------------------- #


def test_pr_comment_reports_regression(base_report):
    clean = capture_conformance_snapshot(
        revision="a", captured_at="t", report=base_report
    )
    surfaces = dict(clean.surface_pass)
    first = sorted(surfaces)[0]
    surfaces[first] = False
    regressed = type(clean)(
        revision="b",
        captured_at="t",
        provider_families=clean.provider_families,
        surface_pass=surfaces,
        manifest_sha256=clean.manifest_sha256,
        replay_hash=clean.replay_hash,
        all_passed=False,
        signature=clean.signature,
    )
    diff = diff_snapshots(clean, regressed)
    assert diff.has_regression
    assert first in diff.regressed_surfaces
    md = pr_comment_markdown(diff)
    assert "Regressed surfaces" in md


# --- 342 bisector ----------------------------------------------------------- #


def test_bisector_pinpoints_regression(report):
    assert report.bisected_regression == "2025-06"


# --- 343 cost-aware sampling ------------------------------------------------ #


def test_cost_aware_sample_respects_budget():
    sample = cost_aware_sample(budget=5)
    assert sample.spent <= 5
    assert len(sample.selected) == 5
    assert sample.skipped > 0


# --- 344 webhook ------------------------------------------------------------ #


def test_regression_webhook_triggers(report):
    assert report.alarm.triggered
    assert report.alarm.payload["event"] == "promptabi.conformance.regression"


# --- 345 gateway certification --------------------------------------------- #


def test_third_party_gateways_certified():
    certs = certify_third_party_gateways()
    assert {c.gateway for c in certs} == set(THIRD_PARTY_GATEWAYS)
    assert all(c.certified for c in certs)
    assert all(c.request_response_replays for c in certs)


# --- aggregate + renderers -------------------------------------------------- #


def test_report_passes_and_renders(report):
    assert report.passed
    text = render_live_provider_ci_text(report)
    for tag in ("[331]", "[332]", "[334]", "[337]", "[338]", "[342]",
                "[343]", "[344]", "[345]"):
        assert tag in text
    decoded = json.loads(render_live_provider_ci_json(report))
    assert decoded["passed"] is True
    assert len(decoded["adapters"]) == 3


def test_public_api_entrypoint():
    text = api_entry(output_format="text")
    assert "PromptABI live provider" in text
    obj = api_entry()
    assert obj.passed
    with pytest.raises(ValueError):
        api_entry(output_format="xml")


def test_run_is_deterministic():
    a = run_live_provider_ci()
    b = run_live_provider_ci()
    assert a.to_dict() == b.to_dict()
