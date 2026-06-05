"""Model dataset transforms as contract-preserving passes (step 262).

A training pipeline applies a sequence of *transforms* to raw examples (role
normalization, template application, truncation, packing).  Each transform is a
pass that must **preserve** the dataset's declared interface contract: it may not
introduce a new role, drop a required special token, or strip the EOS the trainer
relies on.  This module models a pass pipeline and proves the contract is an
invariant of every pass, pinpointing the first pass that breaks it.

A :class:`DatasetContract` is the declared invariant; a :class:`TransformPass`
exposes the *interface footprint* it produces; :func:`verify_pipeline` checks the
footprint of every pass against the contract and reports the earliest violation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

DATASET_TRANSFORM_VERSION = "promptabi.dataset-transform.v1"


class TransformViolationKind(StrEnum):
    NEW_ROLE_INTRODUCED = "new-role-introduced"
    REQUIRED_TOKEN_DROPPED = "required-token-dropped"
    EOS_STRIPPED = "eos-stripped"


@dataclass(frozen=True, slots=True)
class DatasetContract:
    allowed_roles: frozenset[str]
    required_tokens: frozenset[str]
    require_eos: bool = True


@dataclass(frozen=True, slots=True)
class InterfaceFootprint:
    roles: frozenset[str]
    tokens: frozenset[str]
    has_eos: bool


@dataclass(frozen=True, slots=True)
class TransformPass:
    name: str
    apply: Callable[[InterfaceFootprint], InterfaceFootprint]


@dataclass(frozen=True, slots=True)
class TransformViolation:
    kind: TransformViolationKind
    pass_name: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "pass": self.pass_name,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    version: str
    preserved: bool
    final: InterfaceFootprint
    violations: tuple[TransformViolation, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "preserved": self.preserved,
            "final_roles": sorted(self.final.roles),
            "violations": [v.to_dict() for v in self.violations],
        }


def _check(contract: DatasetContract, name: str, fp: InterfaceFootprint) -> list[TransformViolation]:
    out: list[TransformViolation] = []
    extra = fp.roles - contract.allowed_roles
    if extra:
        out.append(
            TransformViolation(
                TransformViolationKind.NEW_ROLE_INTRODUCED,
                name,
                f"introduced disallowed role(s): {sorted(extra)}",
            )
        )
    missing = contract.required_tokens - fp.tokens
    if missing:
        out.append(
            TransformViolation(
                TransformViolationKind.REQUIRED_TOKEN_DROPPED,
                name,
                f"dropped required token(s): {sorted(missing)}",
            )
        )
    if contract.require_eos and not fp.has_eos:
        out.append(
            TransformViolation(
                TransformViolationKind.EOS_STRIPPED,
                name,
                "transform removed the trailing EOS token",
            )
        )
    return out


def verify_pipeline(
    contract: DatasetContract,
    initial: InterfaceFootprint,
    passes: tuple[TransformPass, ...],
) -> PipelineResult:
    fp = initial
    violations: list[TransformViolation] = []
    for tp in passes:
        fp = tp.apply(fp)
        step_violations = _check(contract, tp.name, fp)
        if step_violations:
            violations.extend(step_violations)
            break  # report the earliest breaking pass
    return PipelineResult(
        version=DATASET_TRANSFORM_VERSION,
        preserved=not violations,
        final=fp,
        violations=tuple(violations),
    )


def render_pipeline_text(result: PipelineResult) -> str:
    lines = [
        f"PromptABI dataset-transform pipeline ({result.version})",
        f"result: {'PRESERVED' if result.preserved else 'BROKEN'}",
    ]
    for v in result.violations:
        lines.append(f"  ! {v.kind.value} [{v.pass_name}]: {v.detail}")
    return "\n".join(lines) + "\n"
