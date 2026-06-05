"""Regional / enterprise provider variant compatibility (step 289).

The "same" model is offered through many deployment variants -- public cloud,
regional endpoints, enterprise/VPC, sovereign clouds.  Each variant can differ in
available parameters, data-residency guarantees, and feature flags while claiming
API compatibility.  This module compares a variant's declared capabilities
against the baseline contract and a residency requirement, reporting feature
regressions and residency violations so a deployment can pick a safe variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROVIDER_VARIANT_VERSION = "promptabi.provider-variant.v1"


class VariantFindingKind(StrEnum):
    MISSING_FEATURE = "missing-feature"
    RESIDENCY_VIOLATION = "residency-violation"
    UNSUPPORTED_PARAM = "unsupported-param"


@dataclass(frozen=True, slots=True)
class ProviderVariant:
    provider: str
    variant: str
    region: str
    features: frozenset[str]
    params: frozenset[str]
    data_residency: str  # e.g. "eu", "us", "global"


@dataclass(frozen=True, slots=True)
class VariantRequirement:
    required_features: frozenset[str]
    required_params: frozenset[str]
    allowed_residencies: frozenset[str]


@dataclass(frozen=True, slots=True)
class VariantFinding:
    kind: VariantFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class VariantResult:
    version: str
    compatible: bool
    findings: tuple[VariantFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "compatible": self.compatible,
            "findings": [f.to_dict() for f in self.findings],
        }


def check_variant(
    variant: ProviderVariant, requirement: VariantRequirement
) -> VariantResult:
    findings: list[VariantFinding] = []

    for feature in sorted(requirement.required_features - variant.features):
        findings.append(
            VariantFinding(
                VariantFindingKind.MISSING_FEATURE,
                f"variant {variant.variant!r} lacks feature {feature!r}",
            )
        )
    for param in sorted(requirement.required_params - variant.params):
        findings.append(
            VariantFinding(
                VariantFindingKind.UNSUPPORTED_PARAM,
                f"variant {variant.variant!r} does not accept param {param!r}",
            )
        )
    if variant.data_residency not in requirement.allowed_residencies:
        findings.append(
            VariantFinding(
                VariantFindingKind.RESIDENCY_VIOLATION,
                f"residency {variant.data_residency!r} not in allowed "
                f"{sorted(requirement.allowed_residencies)}",
            )
        )

    return VariantResult(
        version=PROVIDER_VARIANT_VERSION,
        compatible=not findings,
        findings=tuple(findings),
    )


def render_variant_text(result: VariantResult) -> str:
    lines = [
        f"PromptABI provider-variant compatibility ({result.version})",
        f"result: {'COMPATIBLE' if result.compatible else 'INCOMPATIBLE'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
