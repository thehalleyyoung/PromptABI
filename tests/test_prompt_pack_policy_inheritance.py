"""Tests for prompt-pack policy inheritance (step 244)."""

from __future__ import annotations

from promptabi.diagnostics import DiagnosticSeverity
from promptabi.prompt_pack_policy_inheritance import (
    InheritanceFindingKind,
    PackPolicy,
    render_inheritance_text,
    resolve_policy_chain,
    verify_inheritance,
)


def _base() -> PackPolicy:
    return PackPolicy(
        name="base",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=("llama", "mistral"),
        severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),),
    )


def test_tightening_child_is_sound() -> None:
    child = PackPolicy(
        name="team",
        required_tools=("audit_log", "rate_limit"),
        banned_roles=("system_override", "developer"),
        allowed_model_families=("llama",),
        severity_floors=(
            ("role-forgery", DiagnosticSeverity.ERROR),
            ("stop-overreach", DiagnosticSeverity.WARNING),
        ),
    )
    findings = verify_inheritance(_base(), child)
    assert findings == ()


def test_dropping_required_tool_is_flagged() -> None:
    child = PackPolicy(name="team", banned_roles=("system_override",),
                       allowed_model_families=("llama",),
                       severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),))
    findings = verify_inheritance(_base(), child)
    assert any(
        f.kind is InheritanceFindingKind.RELAXED_REQUIRED_TOOL for f in findings
    )


def test_unbanning_role_is_flagged() -> None:
    child = PackPolicy(name="team", required_tools=("audit_log",),
                       allowed_model_families=("llama",),
                       severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),))
    findings = verify_inheritance(_base(), child)
    assert any(f.kind is InheritanceFindingKind.UNBANNED_ROLE for f in findings)


def test_widening_model_family_is_flagged() -> None:
    child = PackPolicy(
        name="team",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=("llama", "gpt"),
        severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),),
    )
    findings = verify_inheritance(_base(), child)
    assert any(
        f.kind is InheritanceFindingKind.WIDENED_MODEL_FAMILY for f in findings
    )


def test_removing_model_restriction_is_flagged() -> None:
    child = PackPolicy(
        name="team",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=None,
        severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),),
    )
    findings = verify_inheritance(_base(), child)
    assert any(
        f.kind is InheritanceFindingKind.WIDENED_MODEL_FAMILY for f in findings
    )


def test_lowering_severity_floor_is_flagged() -> None:
    child = PackPolicy(
        name="team",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=("llama",),
        severity_floors=(("role-forgery", DiagnosticSeverity.WARNING),),
    )
    findings = verify_inheritance(_base(), child)
    assert any(f.kind is InheritanceFindingKind.LOWERED_SEVERITY for f in findings)


def test_unset_inherited_floor_is_flagged() -> None:
    child = PackPolicy(
        name="team",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=("llama",),
    )
    findings = verify_inheritance(_base(), child)
    assert any(f.kind is InheritanceFindingKind.LOWERED_SEVERITY for f in findings)


def test_resolve_chain_effective_policy() -> None:
    base = _base()
    team = PackPolicy(
        name="team",
        required_tools=("audit_log", "rate_limit"),
        banned_roles=("system_override",),
        allowed_model_families=("llama",),
        severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),),
    )
    project = PackPolicy(
        name="project",
        required_tools=("audit_log", "rate_limit", "pii_scan"),
        banned_roles=("system_override",),
        allowed_model_families=("llama",),
        severity_floors=(
            ("role-forgery", DiagnosticSeverity.ERROR),
            ("stop-overreach", DiagnosticSeverity.ERROR),
        ),
    )
    result = resolve_policy_chain([base, team, project])
    assert result.sound
    eff = result.effective
    assert set(eff.required_tools) == {"audit_log", "rate_limit", "pii_scan"}
    assert eff.allowed_model_families == ("llama",)
    assert eff.floor_map["stop-overreach"] is DiagnosticSeverity.ERROR


def test_resolve_chain_detects_violation_deep() -> None:
    base = _base()
    team = PackPolicy(
        name="team",
        required_tools=("audit_log",),
        banned_roles=("system_override",),
        allowed_model_families=("llama",),
        severity_floors=(("role-forgery", DiagnosticSeverity.ERROR),),
    )
    rogue = PackPolicy(
        name="rogue",
        required_tools=(),  # drops audit_log inherited from base/team
        allowed_model_families=("llama",),
    )
    result = resolve_policy_chain([base, team, rogue])
    assert not result.sound
    assert "RELAXED" in render_inheritance_text(result)
