"""Tests for marketplace trust tiers (step 245)."""

from __future__ import annotations

from promptabi.prompt_pack_trust_tiers import (
    TrustEvidence,
    TrustTier,
    evaluate_trust_tier,
    render_trust_text,
    tier_requirements,
)


def test_no_evidence_is_untrusted() -> None:
    a = evaluate_trust_tier("p", TrustEvidence())
    assert a.awarded_tier is TrustTier.UNTRUSTED
    assert a.next_tier is TrustTier.COMMUNITY
    assert "has_capability_signature" in a.missing_for_next


def test_community_tier() -> None:
    ev = TrustEvidence(has_capability_signature=True, lockfile_verified=True)
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.COMMUNITY
    assert a.next_tier is TrustTier.VERIFIED


def test_partial_community_stays_untrusted() -> None:
    ev = TrustEvidence(has_capability_signature=True)  # missing lockfile
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.UNTRUSTED
    assert "lockfile_verified" in a.missing_for_next


def test_verified_requires_one_review() -> None:
    ev = TrustEvidence(
        has_capability_signature=True,
        lockfile_verified=True,
        in_transparency_log=True,
        policy_inheritance_sound=True,
        independent_reviews=1,
    )
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.VERIFIED


def test_verified_blocked_without_review() -> None:
    ev = TrustEvidence(
        has_capability_signature=True,
        lockfile_verified=True,
        in_transparency_log=True,
        policy_inheritance_sound=True,
        independent_reviews=0,
    )
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.COMMUNITY
    assert "independent_review" in a.missing_for_next


def test_certified_full_stack() -> None:
    ev = TrustEvidence(
        has_capability_signature=True,
        lockfile_verified=True,
        in_transparency_log=True,
        policy_inheritance_sound=True,
        examples_certified=True,
        signed_mirror=True,
        independent_reviews=2,
    )
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.CERTIFIED
    assert a.next_tier is None
    assert a.missing_for_next == ()


def test_certified_needs_two_reviews() -> None:
    ev = TrustEvidence(
        has_capability_signature=True,
        lockfile_verified=True,
        in_transparency_log=True,
        policy_inheritance_sound=True,
        examples_certified=True,
        signed_mirror=True,
        independent_reviews=1,
    )
    a = evaluate_trust_tier("p", ev)
    assert a.awarded_tier is TrustTier.VERIFIED
    assert "two_independent_reviews" in a.missing_for_next


def test_tiers_are_cumulative() -> None:
    verified = set(tier_requirements(TrustTier.VERIFIED))
    certified = set(tier_requirements(TrustTier.CERTIFIED))
    assert verified < certified


def test_render_text() -> None:
    a = evaluate_trust_tier("p", TrustEvidence())
    assert "UNTRUSTED" in render_trust_text(a)
