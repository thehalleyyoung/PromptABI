"""Tests for the upstream interface-safety bug confirmation campaign.

These tests prove the campaign engine works against *real* pinned upstream source
captured in ``fixtures/upstream_bug_campaign/`` and that the full methodology
(scope definitions, target selection, provenance pinning, inventories, flag-rule
candidates, duplicate search, reproduction plans, disclosure routing, report
drafting, and honest triage) is enforced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import promptabi
from promptabi.upstream_bug_campaign import (
    EXCLUDED_BUG_CLASSES,
    FLAG_RULES,
    INTERFACE_SAFETY_BUG_CLASSES,
    CampaignDossier,
    UpstreamBugCampaignError,
    _detect_token_id_only_transition,
    load_campaign_dossier,
    render_campaign_json,
    render_campaign_text,
    render_candidate_report_markdown,
    run_campaign,
)


DOSSIER_PATH = Path("fixtures/upstream_bug_campaign/campaign.json")


def _raw_dossier() -> dict:
    return json.loads(DOSSIER_PATH.read_text(encoding="utf-8"))


def _write_tmp_dossier(tmp_path: Path, raw: dict) -> Path:
    # Mirror the captured sources next to the temp dossier so capture
    # verification (full-file hashes) still resolves.
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    real_sources = Path("fixtures/upstream_bug_campaign/sources")
    for child in real_sources.iterdir():
        (src_dir / child.name).write_text(child.read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_dossier_loads_and_validates() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    assert isinstance(dossier, CampaignDossier)
    assert dossier.version == promptabi.UPSTREAM_BUG_CAMPAIGN_VERSION
    assert len(dossier.targets) >= 3
    target_ids = {t.target_id for t in dossier.targets}
    assert {"vllm", "llama_cpp", "transformers"} <= target_ids


def test_scope_definitions_match_taxonomy() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    assert set(dossier.definitions.bug_class_taxonomy) <= INTERFACE_SAFETY_BUG_CLASSES
    assert set(dossier.definitions.excluded_classes) <= EXCLUDED_BUG_CLASSES
    # Working notes must live outside the committed tree (step 5).
    assert not dossier.definitions.working_notes_path.startswith(("src/", "fixtures/", "tests/"))


def test_targets_record_security_and_contribution_policy() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    for target in dossier.targets:
        assert target.repository_url.startswith("https://github.com/")
        assert target.security_policy_url.startswith("https://github.com/")
        assert target.contribution_guide_url.startswith("https://github.com/")
        assert target.activity_signals


def test_scanned_sources_are_pinned_and_hash_locked() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    assert dossier.scanned_sources
    for source in dossier.scanned_sources:
        assert len(source.commit_sha) >= 16
        assert source.public_url.startswith("https://github.com/")
        # Captured-file verification (full-file hash + excerpt membership).
        source.verify_capture(base_dir=dossier.base_dir)


def test_run_campaign_produces_honest_triage() -> None:
    result = run_campaign(dossier_path=DOSSIER_PATH)
    counts = result.outcome_counts
    assert counts["rejected"] == 1
    assert counts["abstained"] == 1
    assert counts["duplicate"] == 1
    assert counts["confirmed"] == 0
    # No new bug claimed from this scan, but detection of real bugs is proven.
    assert result.reportable == ()
    assert len(result.confirmed_detection_references) >= 1
    for ref in result.confirmed_detection_references:
        assert ref.public_reference.startswith("https://github.com/")
        assert ref.witness_count > 0


def test_deepseek_token_id_transition_detected_on_real_source() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    source = dossier.source("vllm_deepseek_r1_reasoning")
    # The real captured parser branches on token-id membership for the transition.
    assert _detect_token_id_only_transition(source.excerpt) is True
    triage = {t.candidate_id: t for t in run_campaign(dossier=dossier).triage}
    rejected = triage["vllm_deepseek_reasoning_tokenid"]
    assert rejected.outcome == "rejected"
    assert "atomic special token" in rejected.evidence


def test_hermes_template_triages_to_abstention() -> None:
    triage = {t.candidate_id: t for t in run_campaign(dossier_path=DOSSIER_PATH).triage}
    abstained = triage["vllm_hermes_tool_template_abstain"]
    assert abstained.outcome == "abstained"
    assert "supported Jinja fragment" in abstained.evidence


def test_duplicate_candidate_routes_to_existing_issue() -> None:
    triage = {t.candidate_id: t for t in run_campaign(dossier_path=DOSSIER_PATH).triage}
    dup = triage["vllm_qwen_multi_function_block_dup"]
    assert dup.outcome == "duplicate"
    assert dup.reportable is False


def test_report_draft_renders_for_duplicate_contribution() -> None:
    dossier = load_campaign_dossier(DOSSIER_PATH)
    candidate = next(c for c in dossier.candidates if c.report is not None)
    source = dossier.source(candidate.source_id)
    markdown = render_candidate_report_markdown(candidate, source)
    assert markdown.startswith("# ")
    assert source.commit_sha in markdown
    assert "## Reproduction" in markdown
    assert "Severity:" in markdown


def test_flag_rules_cover_taxonomy() -> None:
    # Every flag rule maps to a real interface-safety bug class.
    assert set(FLAG_RULES.values()) <= INTERFACE_SAFETY_BUG_CLASSES
    # The taxonomy is exercised across several distinct flag rules.
    assert len(set(FLAG_RULES.values())) >= 6


def test_render_helpers_emit_text_and_json() -> None:
    result = run_campaign(dossier_path=DOSSIER_PATH)
    text = render_campaign_text(result)
    assert "Upstream interface-safety bug campaign" in text
    payload = json.loads(render_campaign_json(result))
    assert payload["outcome_counts"]["confirmed"] == 0
    assert payload["targets"] == 3


def test_excerpt_hash_tampering_is_rejected(tmp_path: Path) -> None:
    raw = _raw_dossier()
    raw["scanned_sources"][0]["excerpt"] = raw["scanned_sources"][0]["excerpt"] + "\n# tampered"
    path = _write_tmp_dossier(tmp_path, raw)
    with pytest.raises(UpstreamBugCampaignError, match="sha256 mismatch"):
        load_campaign_dossier(path)


def test_capture_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    raw = _raw_dossier()
    raw["scanned_sources"][0]["full_file_sha256"] = "sha256:" + "0" * 64
    path = _write_tmp_dossier(tmp_path, raw)
    dossier = load_campaign_dossier(path)
    with pytest.raises(UpstreamBugCampaignError, match="captured file hash mismatch"):
        run_campaign(dossier=dossier)


def test_outcome_mismatch_is_rejected(tmp_path: Path) -> None:
    raw = _raw_dossier()
    # Claim a confirmed outcome the analyzer will not support.
    for cand in raw["candidates"]:
        if cand["candidate_id"] == "vllm_deepseek_reasoning_tokenid":
            cand["expected_outcome"] = "confirmed"
            cand["report"] = {
                "title": "x",
                "summary": "x",
                "why_it_matters": "x",
                "actual_output": "x",
                "expected_output": "x",
                "promptabi_attribution": "x",
                "severity_claim": "defer-to-maintainers",
            }
    path = _write_tmp_dossier(tmp_path, raw)
    dossier = load_campaign_dossier(path)
    with pytest.raises(UpstreamBugCampaignError, match="expected 'confirmed' but analysis produced 'rejected'"):
        run_campaign(dossier=dossier)


def test_requires_three_targets(tmp_path: Path) -> None:
    raw = _raw_dossier()
    raw["targets"] = raw["targets"][:2]
    path = _write_tmp_dossier(tmp_path, raw)
    with pytest.raises(UpstreamBugCampaignError, match="three upstream targets"):
        load_campaign_dossier(path)


def test_overclaimed_severity_is_rejected(tmp_path: Path) -> None:
    raw = _raw_dossier()
    for cand in raw["candidates"]:
        if cand.get("report"):
            cand["report"]["severity_claim"] = "critical"
    path = _write_tmp_dossier(tmp_path, raw)
    with pytest.raises(UpstreamBugCampaignError, match="overclaim"):
        load_campaign_dossier(path)


def test_public_api_exports_campaign() -> None:
    for name in (
        "run_campaign",
        "load_campaign_dossier",
        "CampaignResult",
        "CandidateFinding",
        "UpstreamBugCampaignError",
        "render_candidate_report_markdown",
    ):
        assert hasattr(promptabi, name)
