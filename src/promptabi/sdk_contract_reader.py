"""Language-SDK provider-contract readers (step 297).

A provider contract authored once must be *readable* by SDKs in many languages.
This module emits a language-neutral, flat intermediate representation (IR) of a
provider contract -- obligations, capability requirements, stop policy, context
window -- that SDK code generators in any language can consume, plus a validator
that checks a round-tripped IR for the fields every reader depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

SDK_CONTRACT_READER_VERSION = "promptabi.sdk-contract-reader.v1"


class IRFindingKind(StrEnum):
    MISSING_FIELD = "missing-field"
    EMPTY_OBLIGATIONS = "empty-obligations"
    BAD_VERSION = "bad-version"


_REQUIRED_IR_FIELDS = (
    "version",
    "contract_id",
    "obligations",
    "required_capabilities",
    "max_total_tokens",
    "stop_sequences",
)


@dataclass(frozen=True, slots=True)
class ProviderContract:
    contract_id: str
    obligations: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    max_total_tokens: int
    stop_sequences: tuple[str, ...]


def to_ir(contract: ProviderContract) -> dict[str, object]:
    """Flat, language-neutral IR consumable by any SDK reader."""

    return {
        "version": SDK_CONTRACT_READER_VERSION,
        "contract_id": contract.contract_id,
        "obligations": list(contract.obligations),
        "required_capabilities": list(contract.required_capabilities),
        "max_total_tokens": contract.max_total_tokens,
        "stop_sequences": list(contract.stop_sequences),
    }


@dataclass(frozen=True, slots=True)
class IRFinding:
    kind: IRFindingKind
    field: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "field": self.field, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class IRValidation:
    version: str
    readable: bool
    findings: tuple[IRFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "readable": self.readable,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_ir(ir: dict[str, object]) -> IRValidation:
    findings: list[IRFinding] = []

    for fld in _REQUIRED_IR_FIELDS:
        if fld not in ir:
            findings.append(
                IRFinding(
                    IRFindingKind.MISSING_FIELD,
                    fld,
                    f"IR is missing required field {fld!r}",
                )
            )

    if ir.get("version") != SDK_CONTRACT_READER_VERSION:
        findings.append(
            IRFinding(
                IRFindingKind.BAD_VERSION,
                "version",
                f"unexpected IR version {ir.get('version')!r}",
            )
        )

    obligations = ir.get("obligations")
    if isinstance(obligations, list) and not obligations:
        findings.append(
            IRFinding(
                IRFindingKind.EMPTY_OBLIGATIONS,
                "obligations",
                "contract declares no obligations",
            )
        )

    return IRValidation(
        version=SDK_CONTRACT_READER_VERSION,
        readable=not findings,
        findings=tuple(findings),
    )


def render_ir_validation_text(result: IRValidation) -> str:
    lines = [
        f"PromptABI SDK contract reader ({result.version})",
        f"readable: {'YES' if result.readable else 'NO'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.field}]: {f.detail}")
    return "\n".join(lines) + "\n"
