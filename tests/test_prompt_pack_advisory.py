"""Tests for the prompt-pack vulnerability advisory format (step 253)."""

from __future__ import annotations

import pytest

from promptabi.prompt_pack_advisory import (
    AdvisoryClass,
    AdvisorySeverity,
    PromptPackAdvisory,
    VersionRange,
    match_advisory,
    render_advisory_text,
    scan_advisories,
    version_lt,
)

ADVISORY = PromptPackAdvisory(
    advisory_id="PROMPTABI-2024-0001",
    pack="support-triage",
    title="User content can forge the assistant role header",
    advisory_class=AdvisoryClass.ROLE_FORGERY,
    severity=AdvisorySeverity.HIGH,
    affected=(VersionRange(introduced="1.0.0", fixed="1.2.1"),),
    first_fixed="1.2.1",
)


def test_version_ordering() -> None:
    assert version_lt("1.2.0", "1.2.1")
    assert not version_lt("2.0.0", "1.9.9")


def test_affected_version_matches() -> None:
    m = match_advisory(ADVISORY, "1.2.0")
    assert m.affected
    assert m.recommended_upgrade == "1.2.1"


def test_unaffected_version() -> None:
    m = match_advisory(ADVISORY, "1.2.1")
    assert not m.affected
    assert m.recommended_upgrade is None


def test_first_fixed_must_be_outside_affected() -> None:
    with pytest.raises(ValueError):
        PromptPackAdvisory(
            advisory_id="x",
            pack="p",
            title="t",
            advisory_class=AdvisoryClass.STOP_LEAK,
            severity=AdvisorySeverity.LOW,
            affected=(VersionRange("1.0.0", "2.0.0"),),
            first_fixed="1.5.0",
        )


def test_scan_filters_by_pack_and_version() -> None:
    matches = scan_advisories((ADVISORY,), "support-triage", "1.1.0")
    assert len(matches) == 1
    assert scan_advisories((ADVISORY,), "other-pack", "1.1.0") == ()


def test_render_text_smoke() -> None:
    assert "advisory" in render_advisory_text(ADVISORY).lower()
