"""Tests for the security and red-team research surfaces (steps 361-375)."""

from __future__ import annotations

import json

import pytest

from promptabi import red_team_research as api_entry
from promptabi.red_team_research import (
    attest_prompt_pack,
    build_attack_taxonomy,
    build_disclosure_record,
    coordinated_disclosure_policy_markdown,
    detect_homoglyph_control_tokens,
    detect_multi_agent_confusion,
    detect_refusal_confusion,
    detect_streaming_desync,
    detect_template_injection,
    detect_tokenizer_smuggling,
    measure_defense_coverage,
    prove_hardened_assembly_safe,
    render_red_team_research_json,
    render_red_team_research_text,
    run_ctf_benchmark,
    run_differential_harness,
    run_red_team_research,
    security_whitepaper_markdown,
)


@pytest.fixture(scope="module")
def report():
    return run_red_team_research()


# --- 361 taxonomy ----------------------------------------------------------- #


def test_attack_taxonomy_all_detectable():
    taxonomy = build_attack_taxonomy()
    assert len(taxonomy) == 6
    assert all(a.detected for a in taxonomy)
    assert {a.attack_id for a in taxonomy} == {
        "role-forgery",
        "tokenizer-smuggling",
        "homoglyph-control",
        "refusal-confusion",
        "multi-agent-confusion",
        "streaming-desync",
    }


# --- 362 differential ------------------------------------------------------- #


def test_differential_harness_sound_across_families():
    diff = run_differential_harness()
    assert len(diff.results) == 9
    assert diff.all_raw_caught  # every raw template is flagged
    assert diff.all_hardened_safe  # every hardened template is cleared


# --- 363 disclosure --------------------------------------------------------- #


def test_disclosure_record_reproduced():
    rec = build_disclosure_record()
    assert rec.reproduced
    assert rec.advisory_id.startswith("PROMPTABI-ADV")
    assert len(rec.timeline) >= 3


# --- 364 template injection ------------------------------------------------- #


def test_template_injection_detection():
    inj = detect_template_injection()
    assert inj.any_forgeable
    assert len(inj.raw_forgeable) == 9
    assert len(inj.hardened_safe) == 9


# --- 365 tokenizer smuggling ------------------------------------------------ #


def test_tokenizer_smuggling_and_defense():
    smug = detect_tokenizer_smuggling()
    assert smug.smuggling_possible  # naive tokenizer reproduces the marker
    assert smug.neutralized  # escaping breaks the channel


# --- 366 homoglyph ---------------------------------------------------------- #


def test_homoglyph_control_detection():
    homo = detect_homoglyph_control_tokens()
    assert homo.any_detected
    assert homo.clean_inputs  # benign inputs are not flagged
    # A purely ASCII benign string is never a finding.
    only_clean = detect_homoglyph_control_tokens(inputs=("hello world",))
    assert not only_clean.any_detected


# --- 367 hardened library --------------------------------------------------- #


def test_hardened_assembly_proven_safe():
    proof = prove_hardened_assembly_safe()
    assert proof.all_safe
    assert proof.families_checked == 9
    assert proof.unsafe_families == ()


# --- 368 CTF ---------------------------------------------------------------- #


def test_ctf_benchmark_no_false_negatives():
    ctf = run_ctf_benchmark()
    assert ctf.total == 5
    assert ctf.no_false_negatives
    # The two genuinely-vulnerable levels are flagged by the analyzer.
    vulnerable = [c for c in ctf.challenges if c.expected_vulnerable]
    assert all(c.analyzer_flagged for c in vulnerable)


# --- 369 defense coverage --------------------------------------------------- #


def test_defense_coverage_full():
    cov = measure_defense_coverage()
    assert cov.coverage == pytest.approx(1.0)
    assert cov.uncovered == ()


# --- 370 attestation -------------------------------------------------------- #


def test_supply_chain_attestation():
    att = attest_prompt_pack()
    assert att.verified
    assert att.tamper_detected
    assert len(att.digest) == 64


# --- 371 refusal ------------------------------------------------------------ #


def test_refusal_channel_confusion():
    res = detect_refusal_confusion()
    assert res.confusion_detected
    verdicts = dict(res.classifications)
    assert "ambiguous" in verdicts["ambiguous-bypass"]


# --- 372 multi-agent -------------------------------------------------------- #


def test_multi_agent_confusion():
    res = detect_multi_agent_confusion()
    assert res.violations > 0
    assert not res.ok


# --- 373 streaming desync --------------------------------------------------- #


def test_streaming_desync():
    res = detect_streaming_desync()
    assert res.desync_detected
    assert res.chunks == 3


# --- 374 / 375 docs --------------------------------------------------------- #


def test_whitepaper_and_policy():
    wp = security_whitepaper_markdown()
    assert "Attack taxonomy" in wp
    assert "disclosure timeline" in wp.lower()
    policy = coordinated_disclosure_policy_markdown()
    assert "security@promptabi.dev" in policy
    assert "90 days" in policy


# --- aggregate + API -------------------------------------------------------- #


def test_report_passes_all_fifteen(report):
    assert report.passed
    assert len(report.steps) == 15
    assert {s.step for s in report.steps} == set(range(361, 376))
    text = render_red_team_research_text(report)
    for step in range(361, 376):
        assert f"[{step}]" in text
    assert json.loads(render_red_team_research_json(report))["passed"] is True


def test_public_api_entrypoint():
    text = api_entry(output_format="text")
    assert "red-team" in text.lower()
    obj = api_entry()
    assert obj.passed
    with pytest.raises(ValueError):
        api_entry(output_format="xml")


def test_run_is_deterministic():
    assert run_red_team_research().to_dict() == run_red_team_research().to_dict()
