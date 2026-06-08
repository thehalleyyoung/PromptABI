"""Tests for the performance, scaling, and incremental verification layer
(steps 461-475)."""

from __future__ import annotations

from promptabi import performance_scaling as ps
from promptabi.scaled_evaluation import build_scaled_prompt_corpus


def _configs(n: int) -> list[dict]:
    return [c.config() for c in build_scaled_prompt_corpus(limit=n)]


def test_persistent_cache_is_sound_and_dedups() -> None:
    configs = _configs(60)
    assert ps.cache_is_sound(configs)
    cache = ps.PersistentAnalysisCache()
    for cfg in configs:
        cache.verify(cfg)
    # Re-verifying the same configs is all hits.
    for cfg in configs:
        cache.verify(cfg)
    assert cache.hits >= len(configs)
    assert cache.size <= len(configs)


def test_work_stealing_merge_is_deterministic() -> None:
    configs = _configs(30)
    a = ps.work_stealing_schedule(configs, workers=4)
    b = ps.work_stealing_schedule(configs, workers=1)
    c = ps.work_stealing_schedule(configs, workers=8)
    assert [r.index for r in a] == list(range(len(configs)))
    assert a == b == c  # deterministic regardless of worker count


def test_monorepo_memoization_high_ratio() -> None:
    report = ps.monorepo_verify(_configs(120))
    assert report.config_count == 120
    # Templates repeat across families, so distinct artifacts << configs.
    assert report.distinct_artifacts < report.config_count
    assert report.memoization_ratio > 0.5


def test_single_config_profile_under_100ms() -> None:
    profile = ps.profile_single_config(_configs(1)[0], iterations=30)
    assert profile.iterations == 30
    assert profile.under_100ms


def test_daemon_warm_cache() -> None:
    daemon = ps.VerificationDaemon()
    assert not daemon.warm
    daemon.on_change(_configs(1)[0])
    assert daemon.warm


def test_impacted_checks_demand_driven() -> None:
    assert ps.impacted_checks(["chat_template"]) == ("role-boundary",)
    assert set(ps.impacted_checks(["tokenizer"])) == {"grammar", "stop-policy", "token-budget"}
    assert ps.impacted_checks(["nonexistent"]) == ()


def test_memory_bounded_vocab_streaming() -> None:
    vocab = [(f"t{i}", i) for i in range(5000)]
    assert ps.max_token_id_bounded(vocab, window=256) == 4999
    batches = list(ps.stream_vocabulary(vocab, window=1000))
    assert len(batches) == 5
    assert all(b[1] >= 0 for b in batches)


def test_scaling_curve_is_near_linear() -> None:
    curve = ps.scaling_curve(sizes=(40, 80, 160))
    assert len(curve.points) == 3
    assert curve.near_linear()


def test_shard_configs_partitions_exactly_once() -> None:
    ids = [f"config-{i}" for i in range(100)]
    shards = ps.shard_configs(ids, shards=5)
    assert len(shards) == 5
    flattened = [cid for shard in shards for cid in shard]
    assert sorted(flattened) == sorted(ids)
    assert len(set(flattened)) == len(ids)  # disjoint
    # Deterministic.
    assert ps.shard_configs(ids, shards=5) == shards


def test_fastpath_has_python_fallback_and_stable_abi() -> None:
    fp = ps.FastPath()
    # Native extension is optional; verdict comes from the fallback.
    findings = fp.verify(_configs(1)[0])
    assert isinstance(findings, tuple)
    assert "abi 1.0" in fp.abi_signature()


def test_minimal_recheck_transitive_dependents() -> None:
    edges = {
        "tokenizer": ["stop", "budget"],
        "stop": ["stop-policy-check"],
        "template": ["role-check"],
    }
    rechecked = ps.minimal_recheck_set(edges, ["tokenizer"])
    assert set(rechecked) == {"tokenizer", "stop", "budget", "stop-policy-check"}
    # An unrelated change does not pull in the tokenizer subtree.
    assert set(ps.minimal_recheck_set(edges, ["template"])) == {"template", "role-check"}


def test_trace_and_flamegraph_export() -> None:
    spans = ps.export_trace(_configs(1)[0])
    assert spans
    folded = ps.export_flamegraph(spans)
    assert "promptabi;verify_chat_template" in folded
    # Folded-stack format: "stack count".
    assert all(line.split(" ")[-1].isdigit() for line in folded.strip().splitlines())


def test_performance_regression_gate() -> None:
    assert ps.performance_regression_gate(measured_ms=5.0, golden_envelope_ms=10.0).passed
    assert not ps.performance_regression_gate(measured_ms=50.0, golden_envelope_ms=10.0).passed


def test_performance_whitepaper_contains_empirical_numbers() -> None:
    paper = ps.performance_whitepaper()
    assert "Performance White Paper" in paper
    assert "Single-config median latency" in paper
    assert "near-linear" in paper
