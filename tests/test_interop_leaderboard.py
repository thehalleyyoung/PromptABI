"""Tests for the prompt-pack interoperability leaderboard (step 259)."""

from __future__ import annotations

from promptabi.demo_packs import load_demo_packs
from promptabi.interop_leaderboard import (
    TargetMatrix,
    build_leaderboard,
    render_leaderboard_text,
    score_pack,
)

PACKS = {p.name: p for p in load_demo_packs()}


def test_certified_pack_scores_high() -> None:
    pack = PACKS["support-triage"]
    matrix = TargetMatrix(
        declared=frozenset({"openai:gpt", "hf:llama"}),
        validated=frozenset({"openai:gpt", "hf:llama"}),
    )
    score = score_pack(pack, matrix)
    assert score.score == 100


def test_partial_target_breadth_reduces_score() -> None:
    pack = PACKS["support-triage"]
    matrix = TargetMatrix(
        declared=frozenset({"a", "b", "c", "d"}),
        validated=frozenset({"a"}),
    )
    score = score_pack(pack, matrix)
    assert score.score < 100
    breakdown = dict(score.breakdown)
    assert breakdown["target_breadth"] < 30


def test_leaderboard_is_sorted() -> None:
    full = TargetMatrix(frozenset({"a"}), frozenset({"a"}))
    partial = TargetMatrix(frozenset({"a", "b", "c", "d"}), frozenset())
    board = build_leaderboard(
        (
            (PACKS["support-triage"], partial),
            (PACKS["json-extractor"], full),
        )
    )
    scores = [e.score for e in board.entries]
    assert scores == sorted(scores, reverse=True)


def test_render_text_smoke() -> None:
    full = TargetMatrix(frozenset({"a"}), frozenset({"a"}))
    board = build_leaderboard(((PACKS["json-extractor"], full),))
    assert "leaderboard" in render_leaderboard_text(board)
