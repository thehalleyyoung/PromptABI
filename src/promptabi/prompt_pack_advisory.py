"""A vulnerability advisory format for prompt packs (step 253).

When a prompt pack is found to forge roles, leak a stop token, or ship an
unsatisfiable schema, downstream consumers need a machine-readable advisory --
the prompt-interface analogue of a GHSA/OSV record.  This module defines that
format and the matcher that answers the only question a consumer cares about:
*"is the version I depend on affected, and what do I upgrade to?"*

Version ranges use an explicit, inclusive/exclusive interval model and ordinary
semantic-version comparison; ``first_fixed`` is validated to actually lie outside
the affected range so an advisory can never tell a consumer to upgrade to a still
vulnerable version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AdvisorySeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdvisoryClass(StrEnum):
    ROLE_FORGERY = "role-forgery"
    STOP_LEAK = "stop-leak"
    SCHEMA_UNSAT = "schema-unsatisfiable"
    SANITIZER_BYPASS = "sanitizer-bypass"
    PROVENANCE = "provenance"


PROMPT_PACK_ADVISORY_VERSION = "promptabi.prompt-pack-advisory.v1"


def _parse_version(version: str) -> tuple[int, ...]:
    parts = version.split("-", 1)[0].split(".")
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def version_lt(a: str, b: str) -> bool:
    return _parse_version(a) < _parse_version(b)


@dataclass(frozen=True, slots=True)
class VersionRange:
    """Affected range: [introduced, fixed) with optional bounds."""

    introduced: str = "0.0.0"
    fixed: str | None = None

    def contains(self, version: str) -> bool:
        if version_lt(version, self.introduced):
            return False
        if self.fixed is not None and not version_lt(version, self.fixed):
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {"introduced": self.introduced, "fixed": self.fixed}


@dataclass(frozen=True, slots=True)
class PromptPackAdvisory:
    advisory_id: str
    pack: str
    title: str
    advisory_class: AdvisoryClass
    severity: AdvisorySeverity
    affected: tuple[VersionRange, ...]
    first_fixed: str | None = None
    witness_digest: str | None = None
    references: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.first_fixed is not None:
            for rng in self.affected:
                if rng.contains(self.first_fixed):
                    raise ValueError(
                        f"first_fixed {self.first_fixed} lies inside an affected "
                        "range; advisory would recommend a vulnerable upgrade"
                    )

    def affects(self, version: str) -> bool:
        return any(rng.contains(version) for rng in self.affected)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROMPT_PACK_ADVISORY_VERSION,
            "advisory_id": self.advisory_id,
            "pack": self.pack,
            "title": self.title,
            "class": self.advisory_class.value,
            "severity": self.severity.value,
            "affected": [r.to_dict() for r in self.affected],
            "first_fixed": self.first_fixed,
            "witness_digest": self.witness_digest,
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class AdvisoryMatch:
    advisory_id: str
    affected: bool
    severity: AdvisorySeverity
    recommended_upgrade: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "advisory_id": self.advisory_id,
            "affected": self.affected,
            "severity": self.severity.value,
            "recommended_upgrade": self.recommended_upgrade,
        }


def match_advisory(advisory: PromptPackAdvisory, version: str) -> AdvisoryMatch:
    affected = advisory.affects(version)
    return AdvisoryMatch(
        advisory_id=advisory.advisory_id,
        affected=affected,
        severity=advisory.severity,
        recommended_upgrade=advisory.first_fixed if affected else None,
    )


def scan_advisories(
    advisories: tuple[PromptPackAdvisory, ...],
    pack: str,
    version: str,
) -> tuple[AdvisoryMatch, ...]:
    return tuple(
        match_advisory(a, version)
        for a in advisories
        if a.pack == pack and a.affects(version)
    )


def render_advisory_text(advisory: PromptPackAdvisory) -> str:
    lines = [
        f"PromptABI prompt-pack advisory {advisory.advisory_id} "
        f"({advisory.severity.value.upper()})",
        f"pack: {advisory.pack}",
        f"class: {advisory.advisory_class.value}",
        f"title: {advisory.title}",
    ]
    for rng in advisory.affected:
        fixed = rng.fixed or "unfixed"
        lines.append(f"  affected: >= {rng.introduced}, < {fixed}")
    if advisory.first_fixed:
        lines.append(f"  upgrade to: {advisory.first_fixed}")
    return "\n".join(lines) + "\n"
