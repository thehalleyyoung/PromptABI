"""Verify distillation prompt packs (step 273).

Knowledge distillation trains a *student* on outputs produced by a *teacher*.
The distillation is only sound if the teacher and student share a compatible
prompt interface for the data being transferred: the same rendered prompt
structure (so the student sees what the teacher saw) and a student stop policy
that does not truncate inside the teacher's response format.  A mismatch means
the student is trained on targets it can never legitimately reproduce.

This module verifies a :class:`DistillationPack` by comparing the teacher and
student interfaces and checking the student's stop sequences are a superset
of (or equal to) the teacher's required terminators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

DISTILLATION_VERSION = "promptabi.distillation-pack.v1"


class DistillationFindingKind(StrEnum):
    PROMPT_DIGEST_MISMATCH = "prompt-digest-mismatch"
    ROLE_SET_MISMATCH = "role-set-mismatch"
    STOP_POLICY_TOO_WEAK = "stop-policy-too-weak"
    TOKENIZER_MISMATCH = "tokenizer-mismatch"


@dataclass(frozen=True, slots=True)
class ModelInterface:
    name: str
    prompt_digest: str
    roles: frozenset[str]
    stop_sequences: frozenset[str]
    tokenizer: str


@dataclass(frozen=True, slots=True)
class DistillationPack:
    teacher: ModelInterface
    student: ModelInterface
    require_same_tokenizer: bool = False


@dataclass(frozen=True, slots=True)
class DistillationFinding:
    kind: DistillationFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DistillationResult:
    version: str
    compatible: bool
    findings: tuple[DistillationFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "compatible": self.compatible,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_distillation(pack: DistillationPack) -> DistillationResult:
    findings: list[DistillationFinding] = []
    t, s = pack.teacher, pack.student

    if t.prompt_digest != s.prompt_digest:
        findings.append(
            DistillationFinding(
                DistillationFindingKind.PROMPT_DIGEST_MISMATCH,
                f"teacher prompt digest {t.prompt_digest} != student {s.prompt_digest}",
            )
        )
    if t.roles != s.roles:
        findings.append(
            DistillationFinding(
                DistillationFindingKind.ROLE_SET_MISMATCH,
                f"teacher roles {sorted(t.roles)} != student {sorted(s.roles)}",
            )
        )
    missing_stops = t.stop_sequences - s.stop_sequences
    if missing_stops:
        findings.append(
            DistillationFinding(
                DistillationFindingKind.STOP_POLICY_TOO_WEAK,
                f"student lacks teacher terminators {sorted(missing_stops)}; it may "
                "not terminate the distilled response format",
            )
        )
    if pack.require_same_tokenizer and t.tokenizer != s.tokenizer:
        findings.append(
            DistillationFinding(
                DistillationFindingKind.TOKENIZER_MISMATCH,
                f"teacher tokenizer {t.tokenizer!r} != student {s.tokenizer!r}",
            )
        )

    return DistillationResult(
        version=DISTILLATION_VERSION,
        compatible=not findings,
        findings=tuple(findings),
    )


def render_distillation_text(result: DistillationResult) -> str:
    lines = [
        f"PromptABI distillation-pack check ({result.version})",
        f"result: {'COMPATIBLE' if result.compatible else 'INCOMPATIBLE'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
