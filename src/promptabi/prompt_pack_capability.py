"""Define prompt-pack capability signatures (step 240).

A reusable prompt pack makes ABI *promises*: the templates it exports, the
conversation roles it expects, the tools it can call, the stop policies it
enforces, and the model families it supports.  Consumers need to ask one
question against those promises -- "does this pack provide everything I depend
on?" -- without re-reading the whole artifact.

This module distills a :class:`~promptabi.artifacts.PromptPackArtifact` into a
canonical **capability signature**: normalized, order-independent sets of the
capabilities a pack provides, plus a stable digest.  A
:class:`CapabilityRequirement` describes what a consumer needs, and
:func:`match_capability` proves -- by set containment -- whether the pack
satisfies it, naming every missing capability.  Because signatures are canonical
and hashable, two packs that promise the same ABI share a digest, which is the
foundation later steps build on for upgrade compatibility and registries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

from .artifacts import PromptPackArtifact

PROMPT_PACK_CAPABILITY_VERSION = "promptabi.prompt-pack-capability.v1"


class CapabilityFindingKind(StrEnum):
    MISSING_TEMPLATE = "missing-template"
    MISSING_ROLE = "missing-role"
    MISSING_TOOL = "missing-tool"
    MISSING_STOP_POLICY = "missing-stop-policy"
    MISSING_MODEL_FAMILY = "missing-model-family"


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    version: str
    pack_name: str
    pack_version: str | None
    templates: tuple[str, ...]
    roles: tuple[str, ...]
    tools: tuple[str, ...]
    required_tools: tuple[str, ...]
    stop_policies: tuple[str, ...]
    model_families: tuple[str, ...]

    @property
    def digest(self) -> str:
        payload = json.dumps(self._canonical(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _canonical(self) -> dict[str, object]:
        return {
            "templates": list(self.templates),
            "roles": list(self.roles),
            "tools": list(self.tools),
            "required_tools": list(self.required_tools),
            "stop_policies": list(self.stop_policies),
            "model_families": list(self.model_families),
        }

    def to_dict(self) -> dict[str, object]:
        data = {
            "version": self.version,
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "digest": self.digest,
        }
        data.update(self._canonical())
        return data


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    templates: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    stop_policies: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "templates": list(self.templates),
            "roles": list(self.roles),
            "tools": list(self.tools),
            "stop_policies": list(self.stop_policies),
            "model_families": list(self.model_families),
        }


@dataclass(frozen=True, slots=True)
class CapabilityFinding:
    kind: CapabilityFindingKind
    name: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "name": self.name}


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    version: str
    satisfied: bool
    findings: tuple[CapabilityFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "satisfied": self.satisfied,
            "findings": [f.to_dict() for f in self.findings],
        }


def _sorted_unique(values) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(values)))


def derive_capability_signature(pack: PromptPackArtifact) -> CapabilitySignature:
    """Distill a prompt pack into a canonical capability signature."""

    templates = _sorted_unique(t.name for t in pack.exported_templates)
    roles = set(pack.expected_roles)
    for template in pack.exported_templates:
        roles.update(template.roles)
    tools = _sorted_unique(t.name for t in pack.tool_schemas)
    required_tools = _sorted_unique(
        t.name for t in pack.tool_schemas if t.required
    )
    stop_policies = _sorted_unique(p.name for p in pack.stop_policies)
    model_families = set(pack.supported_model_families)
    for template in pack.exported_templates:
        model_families.update(template.supported_model_families)
    return CapabilitySignature(
        version=PROMPT_PACK_CAPABILITY_VERSION,
        pack_name=pack.pack_name,
        pack_version=pack.pack_version,
        templates=templates,
        roles=_sorted_unique(roles),
        tools=tools,
        required_tools=required_tools,
        stop_policies=stop_policies,
        model_families=_sorted_unique(model_families),
    )


def match_capability(
    signature: CapabilitySignature, requirement: CapabilityRequirement
) -> CapabilityMatch:
    """Prove (by set containment) whether a pack satisfies a requirement."""

    findings: list[CapabilityFinding] = []
    for name in requirement.templates:
        if name not in signature.templates:
            findings.append(
                CapabilityFinding(CapabilityFindingKind.MISSING_TEMPLATE, name)
            )
    for name in requirement.roles:
        if name not in signature.roles:
            findings.append(CapabilityFinding(CapabilityFindingKind.MISSING_ROLE, name))
    for name in requirement.tools:
        if name not in signature.tools:
            findings.append(CapabilityFinding(CapabilityFindingKind.MISSING_TOOL, name))
    for name in requirement.stop_policies:
        if name not in signature.stop_policies:
            findings.append(
                CapabilityFinding(CapabilityFindingKind.MISSING_STOP_POLICY, name)
            )
    for name in requirement.model_families:
        if name not in signature.model_families:
            findings.append(
                CapabilityFinding(CapabilityFindingKind.MISSING_MODEL_FAMILY, name)
            )
    return CapabilityMatch(
        version=PROMPT_PACK_CAPABILITY_VERSION,
        satisfied=not findings,
        findings=tuple(findings),
    )


def render_capability_json(signature: CapabilitySignature) -> str:
    return json.dumps(signature.to_dict(), indent=2, sort_keys=True) + "\n"


def render_capability_match_text(match: CapabilityMatch) -> str:
    lines = [
        f"PromptABI capability match ({match.version})",
        f"result: {'SATISFIED' if match.satisfied else 'UNSATISFIED'}",
    ]
    for finding in match.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.name}")
    return "\n".join(lines) + "\n"
