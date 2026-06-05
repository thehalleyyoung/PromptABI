"""Tests for the community, standardization, and outreach surfaces (steps 391-400)."""

from __future__ import annotations

import json

import pytest

from promptabi import api
from promptabi.community_outreach import (
    COMMUNITY_OUTREACH_VERSION,
    CommunityOutreachReport,
    camera_ready_kit,
    collect_case_studies,
    conformance_challenge,
    docs_site_config,
    governance_documents,
    launch_plan,
    outreach_media,
    render_community_outreach_json,
    render_community_outreach_text,
    rfc_draft,
    run_community_outreach,
    standards_submission,
    working_group_charter,
    write_outreach_artifacts,
)


def test_rfc_draft_has_normative_requirements():
    rfc = rfc_draft()
    assert rfc.startswith("# RFC-0001")
    assert "MUST" in rfc
    assert "Conformance" in rfc


def test_working_group_charter_stakeholders():
    charter = working_group_charter()
    assert len(charter["stakeholders"]) >= 4
    assert charter["ip_policy"] == "Apache-2.0"


def test_standards_submission_open_review():
    submission = standards_submission()
    assert submission["open_review"] is True
    assert submission["venues"]


def test_camera_ready_kit_has_both_parts():
    kit = camera_ready_kit()
    assert kit["camera_ready_checklist"]
    assert kit["rebuttal_template"]
    assert "soundness" in kit["claims_to_evidence"]


def test_outreach_media_complete():
    media = outreach_media()
    for key in ("talk", "poster", "demo_video"):
        assert media[key]


def test_governance_documents():
    gov = governance_documents()
    assert set(gov) == {"CODE_OF_CONDUCT.md", "GOVERNANCE.md", "CONTRIBUTION_LADDER.md"}
    assert all(v.strip() for v in gov.values())


def test_docs_site_config_is_versioned():
    docs = docs_site_config()
    assert "site_name: PromptABI" in docs
    assert "mike" in docs  # versioned docs plugin


def test_conformance_challenge_backed_by_real_ctf():
    challenge = conformance_challenge()
    assert challenge.ctf_levels > 0
    assert challenge.ctf_sound is True
    assert challenge.program["safe_harbor"] is True


def test_case_studies_run_real_verifier():
    studies = collect_case_studies()
    assert len(studies) == 3
    total = sum(s.total_diagnostics for s in studies)
    assert total > 0
    for study in studies:
        assert study.config.endswith("promptabi.json")
        assert study.error_diagnostics <= study.total_diagnostics


def test_launch_plan_targets_stars():
    plan = launch_plan()
    assert plan["goal"].endswith("stars")
    assert len(plan["phases"]) >= 3


def test_suite_passes_all_ten_steps():
    report = run_community_outreach()
    assert isinstance(report, CommunityOutreachReport)
    assert [s.step for s in report.steps] == list(range(391, 401))
    assert report.passed is True
    assert all(s.ok for s in report.steps)


def test_suite_is_deterministic():
    a = run_community_outreach()
    b = run_community_outreach()
    assert a.to_dict() == b.to_dict()


def test_renderers():
    report = run_community_outreach()
    text = render_community_outreach_text(report)
    assert "PASS" in text
    payload = json.loads(render_community_outreach_json(report))
    assert payload["version"] == COMMUNITY_OUTREACH_VERSION
    assert len(payload["steps"]) == 10


def test_public_api_entrypoint():
    report = api.community_outreach()
    assert isinstance(report, CommunityOutreachReport)
    text = api.community_outreach(output_format="text")
    assert isinstance(text, str) and "outreach" in text.lower()
    with pytest.raises(ValueError):
        api.community_outreach(output_format="xml")


def test_write_outreach_artifacts(tmp_path):
    written = write_outreach_artifacts(tmp_path)
    assert "RFC-0001.md" in written
    assert "GOVERNANCE.md" in written
    assert (tmp_path / "RFC-0001.md").read_text().startswith("# RFC-0001")
    json.loads((tmp_path / "working-group-charter.json").read_text())
