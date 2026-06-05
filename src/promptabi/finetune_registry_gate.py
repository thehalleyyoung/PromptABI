"""Connect fine-tune manifests to model-registry gates (step 275).

Before a fine-tuned model is admitted to a model registry, the registry runs a
*gate* over the fine-tune manifest: the base model must be on an allow-list, the
manifest must declare required provenance fields (dataset digest, tokenizer pin,
template digest), the training contract must have passed, and any required
approvals must be present.  This module evaluates a manifest against a gate
policy and returns an admit/deny decision with the precise failed obligations --
the kind of artifact a release pipeline blocks on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

REGISTRY_GATE_VERSION = "promptabi.registry-gate.v1"


class GateFindingKind(StrEnum):
    BASE_NOT_ALLOWED = "base-model-not-allowed"
    MISSING_FIELD = "missing-provenance-field"
    CONTRACT_NOT_PASSED = "training-contract-not-passed"
    MISSING_APPROVAL = "missing-approval"


@dataclass(frozen=True, slots=True)
class GatePolicy:
    allowed_base_models: frozenset[str]
    required_fields: frozenset[str]
    required_approvals: frozenset[str]
    require_contract_passed: bool = True


@dataclass(frozen=True, slots=True)
class FineTuneManifest:
    name: str
    base_model: str
    fields: dict[str, str]
    contract_passed: bool
    approvals: frozenset[str]


@dataclass(frozen=True, slots=True)
class GateFinding:
    kind: GateFindingKind
    subject: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class GateDecision:
    version: str
    admitted: bool
    manifest: str
    findings: tuple[GateFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "admitted": self.admitted,
            "manifest": self.manifest,
            "findings": [f.to_dict() for f in self.findings],
        }


def evaluate_gate(policy: GatePolicy, manifest: FineTuneManifest) -> GateDecision:
    findings: list[GateFinding] = []

    if manifest.base_model not in policy.allowed_base_models:
        findings.append(
            GateFinding(
                GateFindingKind.BASE_NOT_ALLOWED,
                manifest.base_model,
                "base model is not on the registry allow-list",
            )
        )

    for fld in sorted(policy.required_fields):
        if not manifest.fields.get(fld):
            findings.append(
                GateFinding(
                    GateFindingKind.MISSING_FIELD,
                    fld,
                    f"required provenance field {fld!r} is missing or empty",
                )
            )

    if policy.require_contract_passed and not manifest.contract_passed:
        findings.append(
            GateFinding(
                GateFindingKind.CONTRACT_NOT_PASSED,
                manifest.name,
                "training-contract verification did not pass",
            )
        )

    for approval in sorted(policy.required_approvals - manifest.approvals):
        findings.append(
            GateFinding(
                GateFindingKind.MISSING_APPROVAL,
                approval,
                f"required approval {approval!r} is absent",
            )
        )

    return GateDecision(
        version=REGISTRY_GATE_VERSION,
        admitted=not findings,
        manifest=manifest.name,
        findings=tuple(findings),
    )


def render_gate_text(decision: GateDecision) -> str:
    lines = [
        f"PromptABI fine-tune registry gate ({decision.version})",
        f"manifest: {decision.manifest}",
        f"decision: {'ADMIT' if decision.admitted else 'DENY'}",
    ]
    for f in decision.findings:
        lines.append(f"  ! {f.kind.value} [{f.subject}]: {f.detail}")
    return "\n".join(lines) + "\n"
