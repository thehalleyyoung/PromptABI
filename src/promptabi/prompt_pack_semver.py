"""Semantic-version impact rules for prompt packs (step 255).

A prompt pack's *capability surface* -- the roles it exports, the special/stop
tokens it relies on, the structured-output schemas it ships, and the sanitizers
it guarantees -- is a contract.  Changing it requires a semantic-version bump
proportional to the breakage:

* removing an exported role, removing/renaming a schema, dropping a sanitizer,
  or changing a stop token is **breaking** -> requires a MAJOR bump;
* adding a new role/schema/sanitizer is **additive** -> requires at least MINOR;
* anything else (docs, metadata) is a PATCH.

This module diffs two capability surfaces, computes the *required* bump, and
checks that a *declared* old->new version pair actually performs at least that
bump -- so a pack cannot ship a breaking change as a patch release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from .prompt_pack_advisory import _parse_version

PROMPT_PACK_SEMVER_VERSION = "promptabi.prompt-pack-semver.v1"


class BumpLevel(IntEnum):
    PATCH = 0
    MINOR = 1
    MAJOR = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class ImpactKind(StrEnum):
    ROLE_REMOVED = "role-removed"
    ROLE_ADDED = "role-added"
    SCHEMA_REMOVED = "schema-removed"
    SCHEMA_ADDED = "schema-added"
    STOP_TOKEN_CHANGED = "stop-token-changed"
    SANITIZER_DROPPED = "sanitizer-dropped"
    SANITIZER_ADDED = "sanitizer-added"


_IMPACT_LEVEL: dict[ImpactKind, BumpLevel] = {
    ImpactKind.ROLE_REMOVED: BumpLevel.MAJOR,
    ImpactKind.SCHEMA_REMOVED: BumpLevel.MAJOR,
    ImpactKind.STOP_TOKEN_CHANGED: BumpLevel.MAJOR,
    ImpactKind.SANITIZER_DROPPED: BumpLevel.MAJOR,
    ImpactKind.ROLE_ADDED: BumpLevel.MINOR,
    ImpactKind.SCHEMA_ADDED: BumpLevel.MINOR,
    ImpactKind.SANITIZER_ADDED: BumpLevel.MINOR,
}


@dataclass(frozen=True, slots=True)
class CapabilitySurface:
    roles: frozenset[str] = frozenset()
    schemas: frozenset[str] = frozenset()
    sanitizers: frozenset[str] = frozenset()
    stop_tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilityImpact:
    kind: ImpactKind
    subject: str
    level: BumpLevel

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "subject": self.subject,
            "level": self.level.label,
        }


@dataclass(frozen=True, slots=True)
class SemverImpactResult:
    version: str
    required_bump: BumpLevel
    impacts: tuple[CapabilityImpact, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "required_bump": self.required_bump.label,
            "impacts": [i.to_dict() for i in self.impacts],
        }


def diff_surfaces(old: CapabilitySurface, new: CapabilitySurface) -> SemverImpactResult:
    impacts: list[CapabilityImpact] = []

    for role in sorted(old.roles - new.roles):
        impacts.append(CapabilityImpact(ImpactKind.ROLE_REMOVED, role, BumpLevel.MAJOR))
    for role in sorted(new.roles - old.roles):
        impacts.append(CapabilityImpact(ImpactKind.ROLE_ADDED, role, BumpLevel.MINOR))

    for schema in sorted(old.schemas - new.schemas):
        impacts.append(
            CapabilityImpact(ImpactKind.SCHEMA_REMOVED, schema, BumpLevel.MAJOR)
        )
    for schema in sorted(new.schemas - old.schemas):
        impacts.append(
            CapabilityImpact(ImpactKind.SCHEMA_ADDED, schema, BumpLevel.MINOR)
        )

    for san in sorted(old.sanitizers - new.sanitizers):
        impacts.append(
            CapabilityImpact(ImpactKind.SANITIZER_DROPPED, san, BumpLevel.MAJOR)
        )
    for san in sorted(new.sanitizers - old.sanitizers):
        impacts.append(
            CapabilityImpact(ImpactKind.SANITIZER_ADDED, san, BumpLevel.MINOR)
        )

    if old.stop_tokens != new.stop_tokens:
        impacts.append(
            CapabilityImpact(
                ImpactKind.STOP_TOKEN_CHANGED,
                f"{list(old.stop_tokens)} -> {list(new.stop_tokens)}",
                BumpLevel.MAJOR,
            )
        )

    required = max((i.level for i in impacts), default=BumpLevel.PATCH)
    return SemverImpactResult(
        version=PROMPT_PACK_SEMVER_VERSION,
        required_bump=required,
        impacts=tuple(impacts),
    )


def actual_bump(old_version: str, new_version: str) -> BumpLevel:
    o = _parse_version(old_version)
    n = _parse_version(new_version)
    if n[0] != o[0]:
        return BumpLevel.MAJOR
    if n[1] != o[1]:
        return BumpLevel.MINOR
    return BumpLevel.PATCH


@dataclass(frozen=True, slots=True)
class SemverComplianceResult:
    version: str
    compliant: bool
    required_bump: BumpLevel
    declared_bump: BumpLevel
    detail: str
    impacts: tuple[CapabilityImpact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "compliant": self.compliant,
            "required_bump": self.required_bump.label,
            "declared_bump": self.declared_bump.label,
            "detail": self.detail,
            "impacts": [i.to_dict() for i in self.impacts],
        }


def check_semver_compliance(
    old: CapabilitySurface,
    new: CapabilitySurface,
    old_version: str,
    new_version: str,
) -> SemverComplianceResult:
    impact = diff_surfaces(old, new)
    declared = actual_bump(old_version, new_version)
    compliant = declared >= impact.required_bump
    detail = (
        "declared bump satisfies required capability impact"
        if compliant
        else f"{old_version}->{new_version} is a {declared.label} bump but the "
        f"capability change requires at least a {impact.required_bump.label} bump"
    )
    return SemverComplianceResult(
        version=PROMPT_PACK_SEMVER_VERSION,
        compliant=compliant,
        required_bump=impact.required_bump,
        declared_bump=declared,
        detail=detail,
        impacts=impact.impacts,
    )
