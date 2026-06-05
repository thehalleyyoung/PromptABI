"""Tests for the scaled empirical evaluation (roadmap steps 316-330)."""

from __future__ import annotations

import json

import pytest

from promptabi import scaled_empirical_evaluation
from promptabi.scaled_evaluation import (
    TARGET_CORPUS_SIZE,
    ConfusionMatrix,
    GroundTruth,
    build_scaled_prompt_corpus,
    render_scaled_evaluation_json,
    render_scaled_evaluation_text,
    run_scaled_evaluation,
)


@pytest.fixture(scope="module")
def full_report():
    return run_scaled_evaluation()


# --- Step 316: corpus ------------------------------------------------------- #


def test_corpus_is_at_least_ten_thousand_cases():
    corpus = build_scaled_prompt_corpus()
    assert len(corpus) == TARGET_CORPUS_SIZE
    assert len(corpus) >= 10_000


def test_corpus_is_deterministic():
    first = build_scaled_prompt_corpus()
    second = build_scaled_prompt_corpus()
    assert [c.case_id for c in first] == [c.case_id for c in second]
    assert [c.sanitizer for c in first] == [c.sanitizer for c in second]


def test_corpus_case_ids_unique():
    corpus = build_scaled_prompt_corpus()
    assert len({c.case_id for c in corpus}) == len(corpus)


def test_corpus_has_both_labels_for_every_family():
    corpus = build_scaled_prompt_corpus()
    families = {c.family for c in corpus}
    assert len(families) == 9
    for family in families:
        labels = {c.label for c in corpus if c.family == family}
        assert GroundTruth.VULNERABLE in labels
        assert GroundTruth.SAFE in labels


# --- Step 317 / 321: prevalence + soundness --------------------------------- #


def test_role_analyzer_is_sound_no_false_negatives(full_report):
    role = next(s for s in full_report.analyzer_scores if s.analyzer == "role-boundary")
    assert role.matrix.false_negative == 0
    assert role.matrix.recall == 1.0


def test_role_analyzer_f1_clears_floor(full_report):
    role = next(s for s in full_report.analyzer_scores if s.analyzer == "role-boundary")
    assert role.matrix.f1 >= 0.80
    assert role.matrix.true_positive > 0
    # Conservative incompleteness: some over-warnings exist and are bounded.
    assert role.matrix.false_positive > 0
    assert role.matrix.precision < 1.0


def test_prevalence_is_reported_per_family_and_sanitizer(full_report):
    assert 0.0 < full_report.prevalence.prevalence < 1.0
    assert len(full_report.prevalence.by_family) == 9
    # Recognized sanitizers are never predicted vulnerable.
    for name in ("tojson", "escape", "urlencode", "base64"):
        assert full_report.prevalence.by_sanitizer[name] == 0.0
    # Raw interpolation is always predicted vulnerable.
    assert full_report.prevalence.by_sanitizer["raw"] == 1.0


def test_analyzer_invocations_are_memoized(full_report):
    # 9 families x <=7 sanitizer classes => far fewer than the corpus size.
    assert full_report.prevalence.analyzer_invocations <= 63
    assert full_report.corpus_size >= 10_000


# --- Step 322: ablation ----------------------------------------------------- #


def test_sanitizer_pass_has_positive_marginal_contribution(full_report):
    assert full_report.ablation.precision_gain > 0.3
    assert full_report.ablation.full.precision > (
        full_report.ablation.without_sanitizer_pass.precision
    )


# --- Step 320: inter-rater -------------------------------------------------- #


def test_inter_rater_kappa_is_high_but_below_one(full_report):
    assert 0.7 <= full_report.inter_rater.cohen_kappa < 1.0
    # Disagreements are confined to the unmodeled strip-replace class.
    assert set(full_report.inter_rater.disagreement_classes) <= {"strip-replace"}


# --- Step 318 / 319 --------------------------------------------------------- #


def test_schema_violation_rate_is_a_fraction(full_report):
    assert 0.0 <= full_report.schema.overall_violation_rate <= 1.0
    assert full_report.schema.revisions


def test_longitudinal_drift_is_bounded(full_report):
    assert len(full_report.drift.months) == 12
    assert len(full_report.drift.prevalence_series) == 12
    assert full_report.drift.within_stability_band


# --- Step 323: throughput --------------------------------------------------- #


def test_throughput_round_trips_a_million_tokens(full_report):
    assert full_report.throughput.input_tokens >= 1_000_000
    assert full_report.throughput.round_trip_exact
    assert full_report.throughput.tokens_per_second > 0


# --- Step 324: fuzzing ------------------------------------------------------ #


def test_fuzzing_campaign_introduces_violations(full_report):
    assert full_report.fuzzing.mutation_cases > 0
    assert full_report.fuzzing.introduced_violations > 0


# --- Step 325: leaderboard -------------------------------------------------- #


def test_leaderboard_ranks_all_families_and_all_are_sound(full_report):
    assert len(full_report.leaderboard) == 9
    assert all(e.sound_no_false_negatives for e in full_report.leaderboard)
    scores = [e.conformance_score for e in full_report.leaderboard]
    assert scores == sorted(scores, reverse=True)


# --- Step 326: CVE regressions ---------------------------------------------- #


def test_all_cve_vectors_are_detected(full_report):
    assert len(full_report.cve_regressions) == 3
    assert all(c.detected for c in full_report.cve_regressions)


# --- Step 327: false-positive cost ------------------------------------------ #


def test_false_positive_cost_is_quantified(full_report):
    cost = full_report.false_positive_cost
    assert 0.0 <= cost.false_discovery_rate < 0.5
    assert cost.total_minutes > 0


# --- Step 328: cross-tokenizer ---------------------------------------------- #


def test_cross_tokenizer_alignment_on_multilingual_corpus(full_report):
    study = full_report.cross_tokenizer
    assert study.samples >= 10
    assert study.alignment_error_rate == 0.0
    assert len(study.by_locale) == study.samples


# --- Step 329: training contracts ------------------------------------------- #


def test_training_contract_violation_rate_is_reported(full_report):
    study = full_report.training_contracts
    assert study.records > 0
    assert 0.0 <= study.violation_rate <= 1.0


# --- Aggregate + renderers -------------------------------------------------- #


def test_report_passes_and_round_trips_json(full_report):
    assert full_report.passed
    payload = render_scaled_evaluation_json(full_report)
    decoded = json.loads(payload)
    assert decoded["corpus_size"] >= 10_000
    assert decoded["passed"] is True
    assert decoded["analyzer_scores"][0]["analyzer"] == "role-boundary"


def test_text_render_contains_every_study(full_report):
    text = render_scaled_evaluation_text(full_report)
    for tag in ("[317]", "[318]", "[319]", "[320]", "[321]", "[322]",
                "[323]", "[324]", "[325]", "[326]", "[327]", "[328]", "[329]"):
        assert tag in text


def test_public_api_entrypoint(full_report):
    text = scaled_empirical_evaluation(output_format="text", corpus_limit=200)
    assert isinstance(text, str)
    report = scaled_empirical_evaluation()
    assert report.corpus_size >= 10_000
    with pytest.raises(ValueError):
        scaled_empirical_evaluation(output_format="yaml")


def test_confusion_matrix_metrics():
    m = ConfusionMatrix(true_positive=8, false_positive=2, true_negative=10, false_negative=0)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(0.8)
    assert m.f1 == pytest.approx(2 * 0.8 / 1.8)
    assert m.specificity == pytest.approx(10 / 12)


def test_golden_summary_is_stable():
    a = run_scaled_evaluation()
    b = run_scaled_evaluation()
    role_a = next(s for s in a.analyzer_scores if s.analyzer == "role-boundary")
    role_b = next(s for s in b.analyzer_scores if s.analyzer == "role-boundary")
    assert role_a.matrix.to_dict() == role_b.matrix.to_dict()
    assert a.prevalence.to_dict() == b.prevalence.to_dict()
