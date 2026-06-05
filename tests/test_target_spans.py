"""Tests for supervised target-span survival after truncation (step 270)."""

from __future__ import annotations

from promptabi.target_spans import (
    Span,
    TargetSpanFindingKind,
    TruncationSide,
    render_target_span_text,
    verify_target_spans,
)


def test_right_truncation_preserves_span() -> None:
    result = verify_target_spans(
        total_length=10,
        max_length=8,
        side=TruncationSide.RIGHT,
        target_spans=(Span("resp", 2, 6),),
    )
    assert result.preserved


def test_right_truncation_severs_span() -> None:
    result = verify_target_spans(
        total_length=10,
        max_length=5,
        side=TruncationSide.RIGHT,
        target_spans=(Span("resp", 3, 8),),
    )
    assert any(
        f.kind is TargetSpanFindingKind.SPAN_SEVERED for f in result.findings
    )


def test_span_dropped_entirely() -> None:
    result = verify_target_spans(
        total_length=10,
        max_length=4,
        side=TruncationSide.RIGHT,
        target_spans=(Span("resp", 6, 9),),
    )
    assert any(f.kind is TargetSpanFindingKind.SPAN_DROPPED for f in result.findings)


def test_left_truncation_drops_prompt() -> None:
    result = verify_target_spans(
        total_length=10,
        max_length=4,
        side=TruncationSide.LEFT,
        target_spans=(Span("resp", 7, 10),),
        prompt_span=Span("prompt", 0, 3),
    )
    assert any(f.kind is TargetSpanFindingKind.PROMPT_DROPPED for f in result.findings)


def test_kept_window_right() -> None:
    result = verify_target_spans(10, 6, TruncationSide.RIGHT, ())
    assert result.kept_window == (0, 6)


def test_kept_window_left() -> None:
    result = verify_target_spans(10, 6, TruncationSide.LEFT, ())
    assert result.kept_window == (4, 10)


def test_render_text_smoke() -> None:
    result = verify_target_spans(10, 8, TruncationSide.RIGHT, (Span("r", 0, 4),))
    assert "target-span" in render_target_span_text(result)
