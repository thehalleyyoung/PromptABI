"""Prove exported roles cannot be forged by consumers (step 247).

A prompt pack exports *roles* (``system``, ``assistant``, ``tool``, ...) that
carry authority: a message rendered under the ``system`` role is trusted more
than one rendered under ``user``.  Consumers feed the pack content for the
*non-privileged* channels (user turns, tool results).  The pack's central safety
promise is that no consumer-supplied content can **forge** a privileged role --
i.e. inject the role's header marker so that attacker text is later parsed as if
the pack itself had emitted it.

This module proves that promise structurally.  A :class:`PackRoleModel` declares
every role marker (and whether the role is privileged) and every consumer input
channel together with how that channel neutralises markers -- by escaping the
exact marker strings, or by placing content in a structurally isolated region
(e.g. a JSON string) where the markers are inert.  :func:`prove_role_nonforgeability`
checks that for *every* consumer channel and *every* privileged marker the marker
is neutralised, and for any gap it emits a concrete forging witness -- the minimal
input that would smuggle the privileged header through.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_ROLE_INTEGRITY_VERSION = "promptabi.prompt-pack-role-integrity.v1"


class RoleIntegrityFindingKind(StrEnum):
    FORGEABLE_ROLE = "forgeable-role"
    UNDECLARED_CHANNEL_ROLE = "undeclared-channel-role"


@dataclass(frozen=True, slots=True)
class RoleMarker:
    """A header string that introduces a role in the rendered transcript."""

    role: str
    marker: str
    privileged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "marker": self.marker,
            "privileged": self.privileged,
        }


@dataclass(frozen=True, slots=True)
class ConsumerInputChannel:
    """A region whose content is supplied by the (untrusted) consumer."""

    role: str
    escaped_markers: tuple[str, ...] = ()
    structurally_isolated: bool = False

    def neutralizes(self, marker: str) -> bool:
        return self.structurally_isolated or marker in self.escaped_markers

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "escaped_markers": list(self.escaped_markers),
            "structurally_isolated": self.structurally_isolated,
        }


@dataclass(frozen=True, slots=True)
class PackRoleModel:
    markers: tuple[RoleMarker, ...]
    channels: tuple[ConsumerInputChannel, ...]

    def privileged_markers(self) -> tuple[RoleMarker, ...]:
        return tuple(m for m in self.markers if m.privileged)


@dataclass(frozen=True, slots=True)
class RoleIntegrityFinding:
    kind: RoleIntegrityFindingKind
    channel_role: str
    target_role: str
    marker: str
    forging_witness: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "channel_role": self.channel_role,
            "target_role": self.target_role,
            "marker": self.marker,
            "forging_witness": self.forging_witness,
        }


@dataclass(frozen=True, slots=True)
class RoleIntegrityReport:
    version: str
    nonforgeable: bool
    channels_checked: int
    privileged_roles: tuple[str, ...]
    findings: tuple[RoleIntegrityFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "nonforgeable": self.nonforgeable,
            "channels_checked": self.channels_checked,
            "privileged_roles": list(self.privileged_roles),
            "findings": [f.to_dict() for f in self.findings],
        }


def _forging_witness(marker: str) -> str:
    # The minimal payload a consumer would submit to break out and open a
    # privileged role header.
    return f"{marker}forged-content"


def prove_role_nonforgeability(model: PackRoleModel) -> RoleIntegrityReport:
    """Prove no consumer channel can forge any privileged role marker."""

    privileged = model.privileged_markers()
    declared_roles = {m.role for m in model.markers}
    findings: list[RoleIntegrityFinding] = []

    for channel in model.channels:
        if channel.role not in declared_roles:
            findings.append(
                RoleIntegrityFinding(
                    RoleIntegrityFindingKind.UNDECLARED_CHANNEL_ROLE,
                    channel.role,
                    channel.role,
                    "",
                    "",
                )
            )
        for marker in privileged:
            if not channel.neutralizes(marker.marker):
                findings.append(
                    RoleIntegrityFinding(
                        RoleIntegrityFindingKind.FORGEABLE_ROLE,
                        channel.role,
                        marker.role,
                        marker.marker,
                        _forging_witness(marker.marker),
                    )
                )

    return RoleIntegrityReport(
        version=PROMPT_PACK_ROLE_INTEGRITY_VERSION,
        nonforgeable=not findings,
        channels_checked=len(model.channels),
        privileged_roles=tuple(sorted(m.role for m in privileged)),
        findings=tuple(findings),
    )


def render_role_integrity_json(report: RoleIntegrityReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_role_integrity_text(report: RoleIntegrityReport) -> str:
    lines = [
        f"PromptABI prompt-pack role integrity ({report.version})",
        f"privileged roles: {', '.join(report.privileged_roles) or '(none)'}",
        f"channels checked: {report.channels_checked}",
        f"result: {'NON-FORGEABLE' if report.nonforgeable else 'FORGEABLE'}",
    ]
    for finding in report.findings:
        if finding.kind is RoleIntegrityFindingKind.FORGEABLE_ROLE:
            lines.append(
                f"  ! {finding.channel_role} can forge '{finding.target_role}' "
                f"via marker {finding.marker!r} (witness: {finding.forging_witness!r})"
            )
        else:
            lines.append(f"  ! {finding.kind.value}: {finding.channel_role}")
    return "\n".join(lines) + "\n"
