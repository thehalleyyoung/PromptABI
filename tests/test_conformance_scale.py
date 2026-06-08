"""Tests for the cross-provider conformance corpus and differential testing layer
(steps 416-430).  Every assertion exercises the real analyzers."""

from __future__ import annotations

import json

from promptabi import conformance_scale as cs
from promptabi.scaled_evaluation import SEED_FAMILIES


def test_conformance_corpus_spans_every_provider_family() -> None:
    corpus = cs.build_conformance_corpus()
    assert corpus.total_cases >= 10_000
    assert corpus.spans_every_seed_family()
    assert {s.seed_family for s in corpus.slices} == set(SEED_FAMILIES)
    # Every major provider runtime family is represented.
    assert "openai" in corpus.provider_families
    assert "meta-llama" in corpus.provider_families
    # Each slice has both safe and vulnerable templates.
    assert all(s.case_count == s.vulnerable_count + s.safe_count for s in corpus.slices)
    assert all(s.distinct_templates >= 1 for s in corpus.slices)
    # Serialisable and digest-stamped.
    json.dumps(corpus.to_dict())
    assert corpus.digest


def test_differential_oracle_high_agreement_only_on_unmodeled_class() -> None:
    report = cs.run_differential_oracle(limit=600)
    assert report.compared == 600
    assert report.agreement_rate > 0.9
    # The analyzer is sound: it never misses a forgery the oracle catches.
    assert report.oracle_only == 0
    # Disagreement is confined to the known unmodeled sanitizer class.
    assert set(report.disagreement_classes) <= {"strip-replace"}


def test_mine_and_normalize_dedups_semantically_identical_templates() -> None:
    raw = {
        "model-a": "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% endfor %}",
        "model-b": "{% for m in messages %}<|im_start|>{{ m['role'] }}   \n{% endfor %}",
        "model-c": "{# header #}{% for m in messages %}[INST]{{ m['content'] }}{% endfor %}",
        "model-d": "{% for m in messages %}[INST]{{ m['content'] }}{% endfor %}",
    }
    normalized = cs.mine_and_normalize(raw)
    # a/b collapse to one structural template; c/d collapse to another.
    assert len(normalized) == 2
    groups = {tuple(n.sources) for n in normalized}
    assert ("model-a", "model-b") in groups
    assert ("model-c", "model-d") in groups


def test_labeled_suites_have_provenance_and_balance() -> None:
    suites = cs.build_labeled_suites(limit=400)
    assert len(suites) == 1
    suite = suites[0]
    assert suite.balanced
    for example in suite.positives + suite.negatives:
        assert example.provenance.startswith("corpus:")
        assert example.rule_id == "role-boundary-nonforgeability"


def test_metamorphic_rewrites_preserve_verdicts() -> None:
    report = cs.run_metamorphic_suite(limit=300)
    assert report.checked > 0
    assert report.ok
    assert report.preserved == report.checked


def test_fuzzing_harness_never_crashes_and_finds_forgeries() -> None:
    report = cs.run_fuzzing_harness(count=200, seed=7)
    assert report.ok
    assert report.crashes == 0
    assert report.flagged_forgeable > 0


def test_per_rule_metrics_report_wilson_intervals() -> None:
    metrics = cs.per_rule_metrics(limit=600)
    # Sound detector: no false negatives -> perfect recall.
    assert metrics.false_negatives == 0
    assert metrics.recall == 1.0
    lo, hi = metrics.precision_ci()
    assert 0.0 <= lo <= metrics.precision <= hi <= 1.0
    lo_r, hi_r = metrics.recall_ci()
    assert lo_r <= 1.0 <= hi_r + 1e-9


def test_wilson_interval_known_values() -> None:
    lo, hi = cs.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert cs.wilson_interval(0, 0) == (0.0, 0.0)


def test_regression_museum_catches_every_historical_bug() -> None:
    results = cs.replay_regression_museum()
    assert len(results) == 4
    assert all(r.caught for r in results)


def test_detect_pairing_drift_flags_verdict_change() -> None:
    pairings = [
        {
            "provider_family": "openai",
            "revision": "2025-01",
            "chat_template": (
                "{% for m in messages %}<|im_start|>{{ m['role'] | tojson }}\n"
                "{{ m['content'] | tojson }}<|im_end|>{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        },
        {
            "provider_family": "openai",
            "revision": "2025-02",
            "chat_template": (
                "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                "{{ m['content'] }}<|im_end|>{% endfor %}"
            ),
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        },
    ]
    alerts = cs.detect_pairing_drift(pairings)
    assert len(alerts) == 1
    assert alerts[0].from_revision == "2025-01"
    assert alerts[0].to_revision == "2025-02"
    assert "forgeable" in alerts[0].reason


def test_golden_encoding_roundtrip_holds() -> None:
    report = cs.cross_validate_golden_encodings()
    assert report.ok
    assert report.checked >= 5


def test_test_vector_package_is_versioned_and_verifiable() -> None:
    package = cs.build_test_vector_package()
    assert package.version == cs.CONFORMANCE_SCALE_VERSION
    assert package.digest
    assert cs.verify_test_vector_package(package)
    # Round-trips through JSON deterministically.
    again = cs.build_test_vector_package()
    assert again.digest == package.digest
    json.loads(package.to_json())


def test_mcnemar_shows_significant_improvement_over_baseline() -> None:
    report = cs.mcnemar_vs_baseline(limit=600)
    assert report.significant_at_05
    # PromptABI is strictly better than the naive delimiter linter.
    assert report.analyzer_correct_baseline_wrong > report.baseline_correct_analyzer_wrong


def test_inter_rater_reliability_is_substantial() -> None:
    report = cs.inter_rater_reliability(limit=600)
    assert 0.6 < report.cohens_kappa <= 1.0
    assert report.observed_agreement > 0.9


def test_corpus_snapshot_is_reproducible() -> None:
    a = cs.corpus_snapshot(limit=500)
    b = cs.corpus_snapshot(limit=500)
    assert a.snapshot_id == b.snapshot_id
    assert a.template_digest == b.template_digest


def test_conformance_dashboard_aggregates_live_runs() -> None:
    dashboard = cs.conformance_dashboard(limit=300)
    assert len(dashboard.points) == 1
    point = dashboard.points[0]
    assert point.metamorphic_ok
    assert point.golden_ok
    assert 0.0 <= point.precision <= 1.0
    text = cs.render_dashboard_text(dashboard)
    assert "conformance dashboard" in text
    json.dumps(dashboard.to_dict())
