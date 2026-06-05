"""Verify prompt-pack policy inheritance (step 244).

Prompt packs are layered: an organisation publishes a hardened *base* pack, a
team extends it, and a project extends that.  Each layer declares a
:class:`PackPolicy` -- the safety constraints it enforces (mandatory tools,
forbidden roles, the model families it is allowed to target, and per-rule
diagnostic-severity floors).

Inheritance must be **monotone**: a descendant may *tighten* an inherited
constraint but never *relax* it.  A project that quietly drops a tool its base
pack made mandatory, re-permits a role the base pack banned, widens the set of
allowed model families, or lowers a severity floor has broken the contract its
ancestors promised.

:func:`verify_inheritance` checks a single parent/child link and names every
relaxation.  :func:`resolve_policy_chain` walks a whole ancestry from root to
leaf, verifying each link and -- only if every link is sound -- returning the
*effective* policy (the tightest combination of every layer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .diagnostics import DiagnosticSeverity

PROMPT_PACK_POLICY_INHERITANCE_VERSION = "promptabi.prompt-pack-policy-inheritance.v1"

_SEVERITY_RANK = {
    DiagnosticSeverity.INFO: 0,
    DiagnosticSeverity.WARNING: 1,
    DiagnosticSeverity.ERROR: 2,
}


class InheritanceFindingKind(StrEnum):
    RELAXED_REQUIRED_TOOL = "relaxed-required-tool"
    UNBANNED_ROLE = "unbanned-role"
    WIDENED_MODEL_FAMILY = "widened-model-family"
    LOWERED_SEVERITY = "lowered-severity"


@dataclass(frozen=True, slots=True)
class InheritanceFinding:
    kind: InheritanceFindingKind
    child: str
    parent: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "child": self.child,
            "parent": self.parent,
            "detail": self.detail,
        }


def _norm_strs(values) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(values)))


@dataclass(frozen=True, slots=True)
class PackPolicy:
    """The safety constraints one pack layer enforces."""

    name: str
    required_tools: tuple[str, ...] = ()
    banned_roles: tuple[str, ...] = ()
    # None means "no restriction" (any model family is allowed).
    allowed_model_families: tuple[str, ...] | None = None
    severity_floors: tuple[tuple[str, DiagnosticSeverity], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_tools", _norm_strs(self.required_tools))
        object.__setattr__(self, "banned_roles", _norm_strs(self.banned_roles))
        if self.allowed_model_families is not None:
            object.__setattr__(
                self,
                "allowed_model_families",
                _norm_strs(self.allowed_model_families),
            )
        floors: dict[str, DiagnosticSeverity] = {}
        for rule, sev in self.severity_floors:
            if rule in floors and floors[rule] != sev:
                # keep the stricter of duplicate declarations
                if _SEVERITY_RANK[sev] > _SEVERITY_RANK[floors[rule]]:
                    floors[rule] = sev
            else:
                floors[rule] = sev
        object.__setattr__(
            self,
            "severity_floors",
            tuple(sorted(floors.items(), key=lambda kv: kv[0])),
        )

    @property
    def floor_map(self) -> dict[str, DiagnosticSeverity]:
        return dict(self.severity_floors)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required_tools": list(self.required_tools),
            "banned_roles": list(self.banned_roles),
            "allowed_model_families": (
                None
                if self.allowed_model_families is None
                else list(self.allowed_model_families)
            ),
            "severity_floors": {r: s.value for r, s in self.severity_floors},
        }


@dataclass(frozen=True, slots=True)
class InheritanceResult:
    version: str
    sound: bool
    effective: PackPolicy
    findings: tuple[InheritanceFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sound": self.sound,
            "effective": self.effective.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_inheritance(parent: PackPolicy, child: PackPolicy) -> tuple[InheritanceFinding, ...]:
    """Return every way ``child`` relaxes a constraint inherited from ``parent``."""

    findings: list[InheritanceFinding] = []

    child_tools = set(child.required_tools)
    for tool in parent.required_tools:
        if tool not in child_tools:
            findings.append(
                InheritanceFinding(
                    InheritanceFindingKind.RELAXED_REQUIRED_TOOL,
                    child.name,
                    parent.name,
                    f"required tool '{tool}' dropped",
                )
            )

    child_banned = set(child.banned_roles)
    for role in parent.banned_roles:
        if role not in child_banned:
            findings.append(
                InheritanceFinding(
                    InheritanceFindingKind.UNBANNED_ROLE,
                    child.name,
                    parent.name,
                    f"banned role '{role}' re-permitted",
                )
            )

    if parent.allowed_model_families is not None:
        if child.allowed_model_families is None:
            findings.append(
                InheritanceFinding(
                    InheritanceFindingKind.WIDENED_MODEL_FAMILY,
                    child.name,
                    parent.name,
                    "child removes model-family restriction inherited from parent",
                )
            )
        else:
            allowed = set(parent.allowed_model_families)
            for fam in child.allowed_model_families:
                if fam not in allowed:
                    findings.append(
                        InheritanceFinding(
                            InheritanceFindingKind.WIDENED_MODEL_FAMILY,
                            child.name,
                            parent.name,
                            f"model family '{fam}' not permitted by parent",
                        )
                    )

    child_floors = child.floor_map
    for rule, parent_sev in parent.floor_map.items():
        child_sev = child_floors.get(rule)
        if child_sev is None or _SEVERITY_RANK[child_sev] < _SEVERITY_RANK[parent_sev]:
            found = child_sev.value if child_sev is not None else "unset"
            findings.append(
                InheritanceFinding(
                    InheritanceFindingKind.LOWERED_SEVERITY,
                    child.name,
                    parent.name,
                    f"severity floor for '{rule}' lowered from {parent_sev.value} to {found}",
                )
            )

    return tuple(findings)


def _merge_effective(parent: PackPolicy, child: PackPolicy) -> PackPolicy:
    if parent.allowed_model_families is None:
        families = child.allowed_model_families
    elif child.allowed_model_families is None:
        families = parent.allowed_model_families
    else:
        families = tuple(
            sorted(set(parent.allowed_model_families) & set(child.allowed_model_families))
        )

    floors = dict(parent.floor_map)
    for rule, sev in child.floor_map.items():
        if rule not in floors or _SEVERITY_RANK[sev] > _SEVERITY_RANK[floors[rule]]:
            floors[rule] = sev

    return PackPolicy(
        name=child.name,
        required_tools=parent.required_tools + child.required_tools,
        banned_roles=parent.banned_roles + child.banned_roles,
        allowed_model_families=families,
        severity_floors=tuple(floors.items()),
    )


def resolve_policy_chain(chain: "list[PackPolicy] | tuple[PackPolicy, ...]") -> InheritanceResult:
    """Verify a root-to-leaf policy chain and return the effective policy.

    ``chain[0]`` is the root ancestor and ``chain[-1]`` is the leaf.  Each link
    is checked for monotone refinement; the effective policy is only meaningful
    when the whole chain is sound.
    """

    if not chain:
        raise ValueError("policy chain must contain at least one policy")

    findings: list[InheritanceFinding] = []
    effective = chain[0]
    for child in chain[1:]:
        findings.extend(verify_inheritance(effective, child))
        effective = _merge_effective(effective, child)

    return InheritanceResult(
        version=PROMPT_PACK_POLICY_INHERITANCE_VERSION,
        sound=not findings,
        effective=effective,
        findings=tuple(findings),
    )


def render_inheritance_json(result: InheritanceResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


def render_inheritance_text(result: InheritanceResult) -> str:
    lines = [
        f"PromptABI prompt-pack policy inheritance ({result.version})",
        f"result: {'SOUND' if result.sound else 'RELAXED (inheritance violated)'}",
    ]
    for finding in result.findings:
        lines.append(
            f"  ! {finding.kind.value}: {finding.child} <- {finding.parent}: {finding.detail}"
        )
    return "\n".join(lines) + "\n"
