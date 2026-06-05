"""Tests for data-loader conformance badges (step 271)."""

from __future__ import annotations

from promptabi.loader_conformance_badge import (
    BadgeColor,
    CheckOutcome,
    build_loader_badge,
    render_badge_text,
)


def test_all_pass_is_green() -> None:
    badge = build_loader_badge(
        "trl",
        (CheckOutcome("loss-mask", True), CheckOutcome("target-span", True)),
    )
    assert badge.color is BadgeColor.GREEN
    assert badge.passed == 2


def test_partial_is_yellow() -> None:
    badge = build_loader_badge(
        "custom",
        (CheckOutcome("loss-mask", True), CheckOutcome("target-span", False)),
    )
    assert badge.color is BadgeColor.YELLOW
    assert badge.failing == ("target-span",)


def test_all_fail_is_red() -> None:
    badge = build_loader_badge("bad", (CheckOutcome("x", False),))
    assert badge.color is BadgeColor.RED


def test_empty_is_red() -> None:
    badge = build_loader_badge("none", ())
    assert badge.color is BadgeColor.RED


def test_shields_endpoint() -> None:
    badge = build_loader_badge("trl", (CheckOutcome("x", True),))
    endpoint = badge.to_shields_endpoint()
    assert endpoint["schemaVersion"] == 1
    assert endpoint["color"] == "green"


def test_render_text_smoke() -> None:
    badge = build_loader_badge("trl", (CheckOutcome("x", True),))
    assert "badge" in render_badge_text(badge)
