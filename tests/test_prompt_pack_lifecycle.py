"""Tests for prompt-pack deprecation and LTS metadata (step 248)."""

from __future__ import annotations

from datetime import date

from promptabi.prompt_pack_lifecycle import (
    LifecycleStatus,
    LifecycleValidationKind,
    PackLifecycle,
    evaluate_lifecycle,
    render_lifecycle_text,
)


def test_supported_version() -> None:
    lc = PackLifecycle("p", "1.0.0", released_on=date(2024, 1, 1))
    ev = evaluate_lifecycle(lc, as_of=date(2024, 6, 1))
    assert ev.status is LifecycleStatus.SUPPORTED
    assert ev.supported
    assert ev.coherent


def test_lts_version() -> None:
    lc = PackLifecycle("p", "1.0.0", released_on=date(2024, 1, 1), lts=True)
    ev = evaluate_lifecycle(lc, as_of=date(2024, 6, 1))
    assert ev.status is LifecycleStatus.LTS


def test_prerelease() -> None:
    lc = PackLifecycle("p", "2.0.0", released_on=date(2025, 1, 1))
    ev = evaluate_lifecycle(lc, as_of=date(2024, 12, 1))
    assert ev.status is LifecycleStatus.PRERELEASE
    assert not ev.supported


def test_deprecated_recommends_successor() -> None:
    lc = PackLifecycle(
        "p",
        "1.0.0",
        released_on=date(2024, 1, 1),
        deprecated_on=date(2024, 6, 1),
        end_of_life_on=date(2025, 1, 1),
        superseded_by="2.0.0",
    )
    ev = evaluate_lifecycle(lc, as_of=date(2024, 7, 1))
    assert ev.status is LifecycleStatus.DEPRECATED
    assert ev.supported
    assert ev.recommended_successor == "2.0.0"
    assert ev.days_until_end_of_life == (date(2025, 1, 1) - date(2024, 7, 1)).days


def test_end_of_life() -> None:
    lc = PackLifecycle(
        "p",
        "1.0.0",
        released_on=date(2024, 1, 1),
        deprecated_on=date(2024, 6, 1),
        end_of_life_on=date(2025, 1, 1),
        superseded_by="2.0.0",
    )
    ev = evaluate_lifecycle(lc, as_of=date(2025, 2, 1))
    assert ev.status is LifecycleStatus.END_OF_LIFE
    assert not ev.supported
    assert ev.days_until_end_of_life < 0


def test_incoherent_eol_before_deprecation() -> None:
    lc = PackLifecycle(
        "p",
        "1.0.0",
        released_on=date(2024, 1, 1),
        deprecated_on=date(2024, 12, 1),
        end_of_life_on=date(2024, 6, 1),
        superseded_by="2.0.0",
    )
    ev = evaluate_lifecycle(lc, as_of=date(2024, 7, 1))
    kinds = {v.kind for v in ev.validation}
    assert LifecycleValidationKind.EOL_BEFORE_DEPRECATION in kinds
    assert not ev.coherent


def test_self_supersede_flagged() -> None:
    lc = PackLifecycle(
        "p",
        "1.0.0",
        released_on=date(2024, 1, 1),
        deprecated_on=date(2024, 6, 1),
        superseded_by="1.0.0",
    )
    ev = evaluate_lifecycle(lc, as_of=date(2024, 7, 1))
    assert any(
        v.kind is LifecycleValidationKind.SELF_SUPERSEDE for v in ev.validation
    )


def test_deprecated_without_successor_flagged() -> None:
    lc = PackLifecycle(
        "p", "1.0.0", released_on=date(2024, 1, 1), deprecated_on=date(2024, 6, 1)
    )
    ev = evaluate_lifecycle(lc, as_of=date(2024, 7, 1))
    assert any(
        v.kind is LifecycleValidationKind.MISSING_SUCCESSOR for v in ev.validation
    )


def test_render_text() -> None:
    lc = PackLifecycle("p", "1.0.0", released_on=date(2024, 1, 1), lts=True)
    ev = evaluate_lifecycle(lc, as_of=date(2024, 6, 1))
    assert "lts" in render_lifecycle_text(ev)
