"""Add marketplace trust tiers (step 245).

A prompt-pack marketplace needs an honest, *evidence-backed* trust label so a
consumer can tell a hardened, audited pack from an anonymous upload.  This module
defines four ascending tiers and awards them purely from objective evidence the
earlier steps already produce -- a capability signature, a verified transitive
lockfile, an entry in a transparency log, sound policy inheritance, certified
examples, and a signed offline mirror.

Tiers are **cumulative**: every requirement of a lower tier is also required by a
higher one, so a ``CERTIFIED`` badge strictly implies everything ``VERIFIED``
promised.  :func:`evaluate_trust_tier` awards the highest tier whose full
requirement set is satisfied and reports exactly which criteria are still missing
for the next tier -- the badge can never outrun its evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_TRUST_TIER_VERSION = "promptabi.prompt-pack-trust-tier.v1"


class TrustTier(StrEnum):
    UNTRUSTED = "untrusted"
    COMMUNITY = "community"
    VERIFIED = "verified"
    CERTIFIED = "certified"


_TIER_ORDER = (
    TrustTier.UNTRUSTED,
    TrustTier.COMMUNITY,
    TrustTier.VERIFIED,
    TrustTier.CERTIFIED,
)


@dataclass(frozen=True, slots=True)
class TrustEvidence:
    """Objective, machine-checkable facts about a published pack."""

    has_capability_signature: bool = False
    lockfile_verified: bool = False
    in_transparency_log: bool = False
    policy_inheritance_sound: bool = False
    examples_certified: bool = False
    signed_mirror: bool = False
    independent_reviews: int = 0

    def criterion(self, name: str) -> bool:
        if name == "independent_review":
            return self.independent_reviews >= 1
        if name == "two_independent_reviews":
            return self.independent_reviews >= 2
        return bool(getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "has_capability_signature": self.has_capability_signature,
            "lockfile_verified": self.lockfile_verified,
            "in_transparency_log": self.in_transparency_log,
            "policy_inheritance_sound": self.policy_inheritance_sound,
            "examples_certified": self.examples_certified,
            "signed_mirror": self.signed_mirror,
            "independent_reviews": self.independent_reviews,
        }


# Requirements *introduced* at each tier (cumulative with all lower tiers).
_TIER_REQUIREMENTS: dict[TrustTier, tuple[str, ...]] = {
    TrustTier.UNTRUSTED: (),
    TrustTier.COMMUNITY: ("has_capability_signature", "lockfile_verified"),
    TrustTier.VERIFIED: (
        "in_transparency_log",
        "policy_inheritance_sound",
        "independent_review",
    ),
    TrustTier.CERTIFIED: (
        "examples_certified",
        "signed_mirror",
        "two_independent_reviews",
    ),
}


def _cumulative_requirements(tier: TrustTier) -> tuple[str, ...]:
    reqs: list[str] = []
    for level in _TIER_ORDER:
        reqs.extend(_TIER_REQUIREMENTS[level])
        if level is tier:
            break
    return tuple(reqs)


@dataclass(frozen=True, slots=True)
class TrustAssessment:
    version: str
    pack_name: str
    awarded_tier: TrustTier
    next_tier: TrustTier | None
    missing_for_next: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pack_name": self.pack_name,
            "awarded_tier": self.awarded_tier.value,
            "next_tier": self.next_tier.value if self.next_tier else None,
            "missing_for_next": list(self.missing_for_next),
        }


def _satisfies(evidence: TrustEvidence, tier: TrustTier) -> bool:
    return all(evidence.criterion(name) for name in _cumulative_requirements(tier))


def evaluate_trust_tier(pack_name: str, evidence: TrustEvidence) -> TrustAssessment:
    """Award the highest tier fully supported by ``evidence``."""

    awarded = TrustTier.UNTRUSTED
    for tier in _TIER_ORDER:
        if _satisfies(evidence, tier):
            awarded = tier
        else:
            break

    awarded_index = _TIER_ORDER.index(awarded)
    next_tier = _TIER_ORDER[awarded_index + 1] if awarded_index + 1 < len(_TIER_ORDER) else None
    missing: tuple[str, ...] = ()
    if next_tier is not None:
        missing = tuple(
            name
            for name in _cumulative_requirements(next_tier)
            if not evidence.criterion(name)
        )

    return TrustAssessment(
        version=PROMPT_PACK_TRUST_TIER_VERSION,
        pack_name=pack_name,
        awarded_tier=awarded,
        next_tier=next_tier,
        missing_for_next=missing,
    )


def tier_requirements(tier: TrustTier) -> tuple[str, ...]:
    """Public view of the cumulative requirements for ``tier``."""

    return _cumulative_requirements(tier)


def render_trust_json(assessment: TrustAssessment) -> str:
    return json.dumps(assessment.to_dict(), indent=2, sort_keys=True) + "\n"


def render_trust_text(assessment: TrustAssessment) -> str:
    lines = [
        f"PromptABI prompt-pack trust tier ({assessment.version})",
        f"pack: {assessment.pack_name}",
        f"tier: {assessment.awarded_tier.value.upper()}",
    ]
    if assessment.next_tier is not None:
        lines.append(f"next tier: {assessment.next_tier.value}")
        for name in assessment.missing_for_next:
            lines.append(f"  - missing: {name}")
    else:
        lines.append("highest tier achieved")
    return "\n".join(lines) + "\n"
