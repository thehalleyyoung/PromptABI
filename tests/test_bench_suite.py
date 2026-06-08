"""Tests for PromptABI-Bench: benchmarks, leaderboards, competitions (steps 476-490)."""

from __future__ import annotations

from promptabi import bench_suite as bs


def test_benchmark_has_public_and_hidden_split() -> None:
    benchmark = bs.build_benchmark()
    assert len(benchmark.cases) >= 40
    assert benchmark.public
    assert benchmark.hidden
    assert len(benchmark.public) + len(benchmark.hidden) == len(benchmark.cases)
    # Deterministic digest.
    assert benchmark.digest == bs.build_benchmark().digest
    # Cases carry ground truth and difficulty.
    assert all(c.difficulty in {"easy", "medium", "hard"} for c in benchmark.cases)


def test_promptabi_is_the_only_sound_baseline() -> None:
    results = {r.name: r for r in bs.baseline_results()}
    assert results["PromptABI"].sound
    assert results["PromptABI"].recall == 1.0
    # The naive linter and LLM grader both miss genuine forgeries.
    assert not results["llm-grader"].sound


def test_soundness_weighted_score_ranks_sound_above_unsound() -> None:
    results = {r.name: r for r in bs.baseline_results()}
    pa = bs.score_submission(results["PromptABI"])
    llm = bs.score_submission(results["llm-grader"])
    assert pa > llm


def test_sarif_leaderboard_interface() -> None:
    import hashlib
    import json

    benchmark = bs.build_benchmark()
    forgeable_case = next(c for c in benchmark.cases if c.forgeable)
    key = hashlib.sha256(
        json.dumps(forgeable_case.config, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()
    sarif_by_case = {key: {"runs": [{"results": [{"ruleId": "x"}]}]}}
    submission = bs.submission_from_sarif(sarif_by_case)
    assert submission(forgeable_case.config) is True
    # Unknown config -> no result -> predicted safe.
    assert submission({"chat_template": "unseen"}) is False


def test_ctf_challenges_are_gradable() -> None:
    challenges = bs.ctf_challenges()
    assert len(challenges) == 3
    assert {c.tier for c in challenges} == {1, 2, 3}
    grade = bs.grade_ctf(bs.promptabi_submission)
    assert grade["solved"] == grade["total"]


def test_competition_rules_have_safe_harbor() -> None:
    rules = bs.competition_rules()
    assert rules["cadence"] == "quarterly"
    assert rules["safe_harbor"]


def test_artifact_evaluation_passes() -> None:
    checklist = bs.artifact_evaluation_checklist()
    assert checklist
    assert bs.artifact_evaluation_passed()


def test_per_rule_difficulty_calibration_vs_human() -> None:
    buckets = {b.difficulty: b for b in bs.per_rule_difficulty()}
    # PromptABI nails easy/medium and conservatively over-warns on the hard,
    # unmodeled strip-replace class -- where humans do better.
    assert buckets["easy"].promptabi_accuracy == 1.0
    assert buckets["hard"].human_accuracy >= buckets["hard"].promptabi_accuracy


def test_state_of_prompt_safety_report() -> None:
    report = bs.state_of_prompt_safety_report()
    assert "PromptABI" in report["sound_tools"]
    assert report["promptabi_score"] > 0.9


def test_adversarial_submission_cannot_bypass_promptabi() -> None:
    # PromptABI never misses a genuine forgery.
    assert not bs.evaluate_adversarial_submission(bs.promptabi_submission).bypassed_soundness
    # An unsound grader does bypass (misses forgeries).
    adv = bs.evaluate_adversarial_submission(bs.llm_grader_submission)
    assert adv.bypassed_soundness
    assert adv.missed_cases


def test_evaluation_container_is_pinned_and_offline() -> None:
    spec = bs.evaluation_container_spec()
    assert spec["network"] == "disabled"
    assert spec["gpu"] == "none"
    assert "promptabi" in spec["pinned"]
    assert "FROM python:3.12-slim" in bs.render_dockerfile()


def test_bootstrap_leaderboard_ranks_promptabi_first() -> None:
    leaderboard = bs.bootstrap_leaderboard(
        {
            "PromptABI": bs.promptabi_submission,
            "naive": bs.naive_linter_submission,
            "llm": bs.llm_grader_submission,
        },
        limit=300,
        resamples=100,
    )
    assert leaderboard[0].name == "PromptABI"
    for row in leaderboard:
        assert row.f1_ci_low <= row.f1 <= row.f1_ci_high + 1e-9


def test_benchmark_doi_metadata() -> None:
    meta = bs.benchmark_doi_metadata()
    assert meta["doi"].startswith("10.5281/zenodo")
    assert meta["license"] == "Apache-2.0"
    assert meta["content_digest"]


def test_certification_gate_requires_soundness_and_threshold() -> None:
    certified = bs.certification_gate(bs.promptabi_submission)
    assert certified.certified
    assert certified.sound
    # The naive linter is not sound on the full corpus -> rejected.
    rejected = bs.certification_gate(bs.llm_grader_submission)
    assert not rejected.certified
    assert "not sound" in rejected.reason
