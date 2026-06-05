"""Prompt-pack interoperability leaderboard (step 259).

Given a set of packs and the targets (provider x model) each is expected to work
on, the leaderboard scores each pack on objective, reproducible criteria and
ranks them.  Scoring is deterministic and explainable: every point is tied to a
named criterion, so a pack author can see exactly why they rank where they do.

Criteria (each worth a fixed weight):

* ``certified`` -- the reusable-pack battery passes (steps 251/252/257),
* ``target_breadth`` -- proportion of declared targets actually covered,
* ``schema_coverage`` -- ships at least one certified structured-output schema,
* ``sanitized`` -- every extension point routes through a sanitizer.

The leaderboard is the kind of artifact that drives a community standard: it is
comparable across packs and stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .demo_packs import DemoPack, certify_demo_pack

INTEROP_LEADERBOARD_VERSION = "promptabi.interop-leaderboard.v1"

_WEIGHTS = {
    "certified": 40,
    "target_breadth": 30,
    "schema_coverage": 20,
    "sanitized": 10,
}


@dataclass(frozen=True, slots=True)
class TargetMatrix:
    """Which provider x model targets are actually validated for a pack."""

    declared: frozenset[str]
    validated: frozenset[str]

    def breadth(self) -> float:
        if not self.declared:
            return 0.0
        return len(self.validated & self.declared) / len(self.declared)


@dataclass(frozen=True, slots=True)
class PackScore:
    pack: str
    version: str
    score: int
    breakdown: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "pack": self.pack,
            "version": self.version,
            "score": self.score,
            "breakdown": [{"criterion": k, "points": v} for k, v in self.breakdown],
        }


@dataclass(frozen=True, slots=True)
class Leaderboard:
    version: str
    entries: tuple[PackScore, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "entries": [e.to_dict() for e in self.entries],
        }


def score_pack(pack: DemoPack, matrix: TargetMatrix) -> PackScore:
    breakdown: list[tuple[str, int]] = []

    cert = certify_demo_pack(pack)
    breakdown.append(("certified", _WEIGHTS["certified"] if cert.certified else 0))

    breakdown.append(
        ("target_breadth", round(_WEIGHTS["target_breadth"] * matrix.breadth()))
    )

    has_schema = bool(pack.schemas)
    breakdown.append(
        ("schema_coverage", _WEIGHTS["schema_coverage"] if has_schema else 0)
    )

    fully_sanitized = bool(pack.extension_points) and all(
        ep.sanitizer is not None for ep in pack.extension_points
    )
    breakdown.append(("sanitized", _WEIGHTS["sanitized"] if fully_sanitized else 0))

    total = sum(points for _, points in breakdown)
    return PackScore(
        pack=pack.name,
        version=pack.version,
        score=total,
        breakdown=tuple(breakdown),
    )


def build_leaderboard(
    packs: tuple[tuple[DemoPack, TargetMatrix], ...],
) -> Leaderboard:
    scores = [score_pack(pack, matrix) for pack, matrix in packs]
    # Stable, deterministic ranking: score desc, then name asc.
    scores.sort(key=lambda s: (-s.score, s.pack))
    return Leaderboard(version=INTEROP_LEADERBOARD_VERSION, entries=tuple(scores))


def render_leaderboard_text(board: Leaderboard) -> str:
    lines = [f"PromptABI interop leaderboard ({board.version})"]
    for rank, entry in enumerate(board.entries, start=1):
        lines.append(f"  {rank}. {entry.pack}@{entry.version}  {entry.score}/100")
        for criterion, points in entry.breakdown:
            lines.append(f"       {criterion}: {points}")
    return "\n".join(lines) + "\n"
