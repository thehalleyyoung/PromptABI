"""Certify prompt-pack upgrade compatibility (step 241).

When a prompt pack is upgraded, consumers need a machine-checkable answer to one
question: *will my existing dependence on this pack keep working?*  Building on
the canonical capability signatures from step 240, this module certifies an
upgrade by comparing the old and new signatures as sets of promised
capabilities.

* A **removed** capability (a template, expected role, *required* tool, or stop
  policy that the new version no longer provides) is a breaking change -- any
  consumer that relied on it is now broken.
* An **added** capability is backward compatible -- it can only help consumers.
* Dropping a previously *optional* tool, or narrowing the supported model
  families, is flagged as a compatibility risk but not an outright break.

The certificate assigns a SemVer-style impact (``major`` for any break,
``minor`` for pure additions, ``patch`` for no capability change) and -- when a
declared version is present -- checks that the version bump is *at least* as
large as the impact demands, so a silent breaking change in a "patch" release is
caught.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .prompt_pack_capability import CapabilitySignature

PROMPT_PACK_UPGRADE_VERSION = "promptabi.prompt-pack-upgrade.v1"


class UpgradeImpact(StrEnum):
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


_IMPACT_ORDER = {UpgradeImpact.PATCH: 0, UpgradeImpact.MINOR: 1, UpgradeImpact.MAJOR: 2}


class UpgradeFindingKind(StrEnum):
    REMOVED_TEMPLATE = "removed-template"
    REMOVED_ROLE = "removed-role"
    REMOVED_REQUIRED_TOOL = "removed-required-tool"
    REMOVED_STOP_POLICY = "removed-stop-policy"
    REMOVED_OPTIONAL_TOOL = "removed-optional-tool"
    NARROWED_MODEL_FAMILY = "narrowed-model-family"
    ADDED_CAPABILITY = "added-capability"
    VERSION_UNDERSTATES_IMPACT = "version-understates-impact"


_BREAKING = frozenset(
    {
        UpgradeFindingKind.REMOVED_TEMPLATE,
        UpgradeFindingKind.REMOVED_ROLE,
        UpgradeFindingKind.REMOVED_REQUIRED_TOOL,
        UpgradeFindingKind.REMOVED_STOP_POLICY,
    }
)


@dataclass(frozen=True, slots=True)
class UpgradeFinding:
    kind: UpgradeFindingKind
    name: str
    breaking: bool

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "name": self.name, "breaking": self.breaking}


@dataclass(frozen=True, slots=True)
class UpgradeCertificate:
    version: str
    old_pack_version: str | None
    new_pack_version: str | None
    impact: UpgradeImpact
    compatible: bool
    findings: tuple[UpgradeFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "old_pack_version": self.old_pack_version,
            "new_pack_version": self.new_pack_version,
            "impact": self.impact.value,
            "compatible": self.compatible,
            "findings": [f.to_dict() for f in self.findings],
        }


def _removed(old: tuple[str, ...], new: tuple[str, ...]) -> list[str]:
    new_set = set(new)
    return [name for name in old if name not in new_set]


def _added(old: tuple[str, ...], new: tuple[str, ...]) -> list[str]:
    old_set = set(old)
    return [name for name in new if name not in old_set]


def _parse_semver(version: str | None) -> tuple[int, int, int] | None:
    if not version:
        return None
    core = version.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _declared_impact(
    old: tuple[int, int, int], new: tuple[int, int, int]
) -> UpgradeImpact:
    if new[0] != old[0]:
        return UpgradeImpact.MAJOR
    if new[1] != old[1]:
        return UpgradeImpact.MINOR
    return UpgradeImpact.PATCH


def certify_upgrade(
    old: CapabilitySignature, new: CapabilitySignature
) -> UpgradeCertificate:
    """Certify whether upgrading from ``old`` to ``new`` is backward compatible."""

    findings: list[UpgradeFinding] = []

    for name in _removed(old.templates, new.templates):
        findings.append(
            UpgradeFinding(UpgradeFindingKind.REMOVED_TEMPLATE, name, True)
        )
    for name in _removed(old.roles, new.roles):
        findings.append(UpgradeFinding(UpgradeFindingKind.REMOVED_ROLE, name, True))
    for name in _removed(old.required_tools, new.required_tools):
        findings.append(
            UpgradeFinding(UpgradeFindingKind.REMOVED_REQUIRED_TOOL, name, True)
        )
    for name in _removed(old.stop_policies, new.stop_policies):
        findings.append(
            UpgradeFinding(UpgradeFindingKind.REMOVED_STOP_POLICY, name, True)
        )

    # Optional-tool removal and model-family narrowing are risks, not breaks.
    old_optional = tuple(t for t in old.tools if t not in old.required_tools)
    for name in _removed(old_optional, new.tools):
        findings.append(
            UpgradeFinding(UpgradeFindingKind.REMOVED_OPTIONAL_TOOL, name, False)
        )
    for name in _removed(old.model_families, new.model_families):
        findings.append(
            UpgradeFinding(UpgradeFindingKind.NARROWED_MODEL_FAMILY, name, False)
        )

    added = (
        _added(old.templates, new.templates)
        + _added(old.roles, new.roles)
        + _added(old.tools, new.tools)
        + _added(old.stop_policies, new.stop_policies)
        + _added(old.model_families, new.model_families)
    )
    for name in added:
        findings.append(
            UpgradeFinding(UpgradeFindingKind.ADDED_CAPABILITY, name, False)
        )

    breaking = any(f.breaking for f in findings)
    if breaking:
        impact = UpgradeImpact.MAJOR
    elif added or any(
        f.kind
        in (
            UpgradeFindingKind.REMOVED_OPTIONAL_TOOL,
            UpgradeFindingKind.NARROWED_MODEL_FAMILY,
        )
        for f in findings
    ):
        impact = UpgradeImpact.MINOR
    else:
        impact = UpgradeImpact.PATCH

    old_ver = _parse_semver(old.pack_version)
    new_ver = _parse_semver(new.pack_version)
    if old_ver is not None and new_ver is not None:
        declared = _declared_impact(old_ver, new_ver)
        if _IMPACT_ORDER[declared] < _IMPACT_ORDER[impact]:
            findings.append(
                UpgradeFinding(
                    UpgradeFindingKind.VERSION_UNDERSTATES_IMPACT,
                    f"{old.pack_version}->{new.pack_version} declares {declared.value}"
                    f" but capabilities changed at {impact.value} level",
                    impact is UpgradeImpact.MAJOR,
                )
            )

    return UpgradeCertificate(
        version=PROMPT_PACK_UPGRADE_VERSION,
        old_pack_version=old.pack_version,
        new_pack_version=new.pack_version,
        impact=impact,
        compatible=not any(f.breaking for f in findings),
        findings=tuple(findings),
    )


def render_upgrade_json(cert: UpgradeCertificate) -> str:
    return json.dumps(cert.to_dict(), indent=2, sort_keys=True) + "\n"


def render_upgrade_text(cert: UpgradeCertificate) -> str:
    lines = [
        f"PromptABI prompt-pack upgrade certificate ({cert.version})",
        f"{cert.old_pack_version} -> {cert.new_pack_version}",
        f"impact: {cert.impact.value}",
        f"compatible: {'YES' if cert.compatible else 'NO (breaking change)'}",
    ]
    for finding in cert.findings:
        mark = "BREAK" if finding.breaking else "note"
        lines.append(f"  [{mark}] {finding.kind.value}: {finding.name}")
    return "\n".join(lines) + "\n"
