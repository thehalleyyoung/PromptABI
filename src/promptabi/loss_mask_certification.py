"""Certify loss-mask semantics across data loaders (step 263).

Different loaders (a TRL collator, a custom packing collator, a HF default) all
claim to supervise *only the assistant/target spans* of a conversation.  If two
loaders disagree about which token positions carry loss, the same data trains two
different objectives.  This module certifies loss-mask agreement *differentially*:
given a reference span specification (the ground-truth target ranges) and each
loader's produced mask, it proves every loader supervises exactly the target
positions -- no more (leaking prompt tokens), no less (dropping supervision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

LOSS_MASK_VERSION = "promptabi.loss-mask-cert.v1"


class LossMaskFindingKind(StrEnum):
    SUPERVISES_PROMPT = "supervises-prompt"
    DROPS_TARGET = "drops-target"
    LENGTH_MISMATCH = "length-mismatch"
    LOADER_DISAGREEMENT = "loader-disagreement"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Ground-truth: positions that *should* be supervised."""

    length: int
    target_positions: frozenset[int]


@dataclass(frozen=True, slots=True)
class LoaderMask:
    loader: str
    mask: tuple[int, ...]

    def supervised(self) -> frozenset[int]:
        return frozenset(i for i, m in enumerate(self.mask) if m)


@dataclass(frozen=True, slots=True)
class LossMaskFinding:
    kind: LossMaskFindingKind
    loader: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "loader": self.loader, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class LossMaskCertification:
    version: str
    certified: bool
    findings: tuple[LossMaskFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "certified": self.certified,
            "findings": [f.to_dict() for f in self.findings],
        }


def certify_loss_masks(
    spec: TargetSpec,
    loaders: tuple[LoaderMask, ...],
) -> LossMaskCertification:
    findings: list[LossMaskFinding] = []
    supervised_sets: dict[str, frozenset[int]] = {}

    for loader in loaders:
        if len(loader.mask) != spec.length:
            findings.append(
                LossMaskFinding(
                    LossMaskFindingKind.LENGTH_MISMATCH,
                    loader.loader,
                    f"mask length {len(loader.mask)} != expected {spec.length}",
                )
            )
            continue
        supervised = loader.supervised()
        supervised_sets[loader.loader] = supervised

        leaked = supervised - spec.target_positions
        if leaked:
            findings.append(
                LossMaskFinding(
                    LossMaskFindingKind.SUPERVISES_PROMPT,
                    loader.loader,
                    f"supervises non-target positions {sorted(leaked)}",
                )
            )
        dropped = spec.target_positions - supervised
        if dropped:
            findings.append(
                LossMaskFinding(
                    LossMaskFindingKind.DROPS_TARGET,
                    loader.loader,
                    f"fails to supervise target positions {sorted(dropped)}",
                )
            )

    distinct = set(supervised_sets.values())
    if len(distinct) > 1:
        findings.append(
            LossMaskFinding(
                LossMaskFindingKind.LOADER_DISAGREEMENT,
                ",".join(sorted(supervised_sets)),
                "loaders supervise different position sets",
            )
        )

    return LossMaskCertification(
        version=LOSS_MASK_VERSION,
        certified=not findings,
        findings=tuple(findings),
    )


def render_loss_mask_text(result: LossMaskCertification) -> str:
    lines = [
        f"PromptABI loss-mask certification ({result.version})",
        f"result: {'CERTIFIED' if result.certified else 'REJECTED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.loader}]: {f.detail}")
    return "\n".join(lines) + "\n"
