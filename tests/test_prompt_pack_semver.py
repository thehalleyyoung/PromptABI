"""Tests for prompt-pack semantic-version impact rules (step 255)."""

from __future__ import annotations

from promptabi.prompt_pack_semver import (
    BumpLevel,
    CapabilitySurface,
    ImpactKind,
    actual_bump,
    check_semver_compliance,
    diff_surfaces,
)

BASE = CapabilitySurface(
    roles=frozenset({"system", "user", "assistant"}),
    schemas=frozenset({"triage"}),
    sanitizers=frozenset({"json_escape"}),
    stop_tokens=("<|im_end|>",),
)


def test_no_change_is_patch() -> None:
    result = diff_surfaces(BASE, BASE)
    assert result.required_bump is BumpLevel.PATCH


def test_role_removed_is_major() -> None:
    new = CapabilitySurface(
        roles=frozenset({"system", "user"}),
        schemas=BASE.schemas,
        sanitizers=BASE.sanitizers,
        stop_tokens=BASE.stop_tokens,
    )
    result = diff_surfaces(BASE, new)
    assert result.required_bump is BumpLevel.MAJOR
    assert any(i.kind is ImpactKind.ROLE_REMOVED for i in result.impacts)


def test_schema_added_is_minor() -> None:
    new = CapabilitySurface(
        roles=BASE.roles,
        schemas=frozenset({"triage", "summary"}),
        sanitizers=BASE.sanitizers,
        stop_tokens=BASE.stop_tokens,
    )
    result = diff_surfaces(BASE, new)
    assert result.required_bump is BumpLevel.MINOR


def test_stop_token_change_is_major() -> None:
    new = CapabilitySurface(
        roles=BASE.roles,
        schemas=BASE.schemas,
        sanitizers=BASE.sanitizers,
        stop_tokens=("</s>",),
    )
    assert diff_surfaces(BASE, new).required_bump is BumpLevel.MAJOR


def test_actual_bump_detection() -> None:
    assert actual_bump("1.0.0", "2.0.0") is BumpLevel.MAJOR
    assert actual_bump("1.0.0", "1.1.0") is BumpLevel.MINOR
    assert actual_bump("1.0.0", "1.0.1") is BumpLevel.PATCH


def test_breaking_change_as_patch_is_noncompliant() -> None:
    new = CapabilitySurface(
        roles=frozenset({"system", "user"}),
        schemas=BASE.schemas,
        sanitizers=BASE.sanitizers,
        stop_tokens=BASE.stop_tokens,
    )
    result = check_semver_compliance(BASE, new, "1.0.0", "1.0.1")
    assert not result.compliant
    assert result.required_bump is BumpLevel.MAJOR


def test_sufficient_bump_is_compliant() -> None:
    new = CapabilitySurface(
        roles=frozenset({"system", "user"}),
        schemas=BASE.schemas,
        sanitizers=BASE.sanitizers,
        stop_tokens=BASE.stop_tokens,
    )
    result = check_semver_compliance(BASE, new, "1.0.0", "2.0.0")
    assert result.compliant
