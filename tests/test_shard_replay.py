"""Tests for streaming-shard witness replay (step 268)."""

from __future__ import annotations

import dataclasses

from promptabi.shard_replay import (
    ReplayFindingKind,
    build_witness,
    derive_shard,
    render_replay_text,
    replay_witness,
)

IDS = tuple(f"ex-{i}" for i in range(20))


def test_replay_reproduces_witness() -> None:
    witness = build_witness(IDS, seed=42, shard_index=0, num_shards=4)
    result = replay_witness(witness, IDS)
    assert result.reproduced, result.findings


def test_shard_partition_is_deterministic() -> None:
    a = derive_shard(IDS, 7, 1, 3)
    b = derive_shard(IDS, 7, 1, 3)
    assert a == b


def test_order_mismatch_detected() -> None:
    witness = build_witness(IDS, seed=42, shard_index=0, num_shards=4)
    tampered = dataclasses.replace(
        witness, example_order=tuple(reversed(witness.example_order))
    )
    result = replay_witness(tampered, IDS)
    assert any(f.kind is ReplayFindingKind.ORDER_MISMATCH for f in result.findings)


def test_different_seed_diverges() -> None:
    witness = build_witness(IDS, seed=1, shard_index=0, num_shards=4)
    other = dataclasses.replace(witness, seed=2)
    result = replay_witness(other, IDS)
    assert not result.reproduced


def test_render_text_smoke() -> None:
    witness = build_witness(IDS, seed=1, shard_index=0, num_shards=2)
    assert "replay" in render_replay_text(replay_witness(witness, IDS))
