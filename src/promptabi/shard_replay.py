"""Streaming-shard witness replay (step 268).

Streaming dataset pipelines shuffle and shard data on the fly, which makes runs
hard to reproduce and audit.  A *witness* records exactly what a shard produced:
the seed, the shard index, the global order of example ids, and a content digest.
Replay re-derives the shard deterministically from the witness's seed and proves
the reconstruction matches the recorded order and digest -- so a regulator or a
debugging engineer can reproduce a training shard bit-for-bit without the
original infrastructure.

The shuffle is a deterministic, dependency-free permutation seeded by the
witness, so replay is fully reproducible in pure Python.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import StrEnum

SHARD_REPLAY_VERSION = "promptabi.shard-replay.v1"


class ReplayFindingKind(StrEnum):
    ORDER_MISMATCH = "order-mismatch"
    DIGEST_MISMATCH = "digest-mismatch"
    SIZE_MISMATCH = "size-mismatch"


@dataclass(frozen=True, slots=True)
class ShardWitness:
    seed: int
    shard_index: int
    num_shards: int
    example_order: tuple[str, ...]
    digest: str


def _digest(order: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for item in order:
        h.update(item.encode("utf-8"))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


def derive_shard(
    example_ids: tuple[str, ...],
    seed: int,
    shard_index: int,
    num_shards: int,
) -> tuple[str, ...]:
    """Deterministically shuffle then take every ``num_shards``-th example."""

    rng = random.Random(seed)
    shuffled = list(example_ids)
    rng.shuffle(shuffled)
    return tuple(shuffled[shard_index::num_shards])


def build_witness(
    example_ids: tuple[str, ...],
    seed: int,
    shard_index: int,
    num_shards: int,
) -> ShardWitness:
    order = derive_shard(example_ids, seed, shard_index, num_shards)
    return ShardWitness(
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
        example_order=order,
        digest=_digest(order),
    )


@dataclass(frozen=True, slots=True)
class ReplayFinding:
    kind: ReplayFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ReplayResult:
    version: str
    reproduced: bool
    findings: tuple[ReplayFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "reproduced": self.reproduced,
            "findings": [f.to_dict() for f in self.findings],
        }


def replay_witness(
    witness: ShardWitness,
    example_ids: tuple[str, ...],
) -> ReplayResult:
    findings: list[ReplayFinding] = []
    derived = derive_shard(
        example_ids, witness.seed, witness.shard_index, witness.num_shards
    )

    if len(derived) != len(witness.example_order):
        findings.append(
            ReplayFinding(
                ReplayFindingKind.SIZE_MISMATCH,
                f"replayed {len(derived)} examples != recorded "
                f"{len(witness.example_order)}",
            )
        )
    if derived != witness.example_order:
        findings.append(
            ReplayFinding(
                ReplayFindingKind.ORDER_MISMATCH,
                "replayed example order does not match the witness",
            )
        )
    if _digest(derived) != witness.digest:
        findings.append(
            ReplayFinding(
                ReplayFindingKind.DIGEST_MISMATCH,
                f"replayed digest {_digest(derived)} != witness {witness.digest}",
            )
        )

    return ReplayResult(
        version=SHARD_REPLAY_VERSION,
        reproduced=not findings,
        findings=tuple(findings),
    )


def render_replay_text(result: ReplayResult) -> str:
    lines = [
        f"PromptABI streaming-shard replay ({result.version})",
        f"result: {'REPRODUCED' if result.reproduced else 'DIVERGED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
