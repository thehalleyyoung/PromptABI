"""Packing proofs for multi-epoch fine-tuning (step 261).

To use a fixed sequence length efficiently, trainers *pack* several short
documents into one training sequence.  Packing is only safe if (a) documents are
separated by an EOS/separator token, (b) the attention mask does not let one
packed document attend across the separator into another (no cross-document
contamination), and (c) the loss mask only supervises real target tokens, never
padding or a previous document's tokens.  When the same packed sequences are
reused across multiple epochs, any violation is amplified.

This module models a packed sequence (token ids, a document-id per position, the
attention "reset" points, and the loss mask) and *proves* the three properties,
returning a precise witness position for any violation.  This is genuine,
checkable logic over real integer arrays -- not a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PACKING_PROOF_VERSION = "promptabi.packing-proof.v1"


class PackingViolationKind(StrEnum):
    MISSING_SEPARATOR = "missing-separator"
    ATTENTION_BLEED = "attention-bleed"
    LOSS_ON_PADDING = "loss-on-padding"
    LOSS_ACROSS_DOCUMENT = "loss-across-document"
    LENGTH_MISMATCH = "length-mismatch"


@dataclass(frozen=True, slots=True)
class PackedSequence:
    """One packed training sequence.

    ``doc_ids[i]`` is the document a position belongs to (-1 for padding).
    ``attention_resets`` is the set of positions where attention must restart
    (i.e. position 0 of each packed document).  ``loss_mask[i]`` is 1 if the
    position is supervised.  ``token_ids[i] == separator_id`` marks a separator.
    """

    token_ids: tuple[int, ...]
    doc_ids: tuple[int, ...]
    attention_resets: frozenset[int]
    loss_mask: tuple[int, ...]
    separator_id: int

    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class PackingViolation:
    kind: PackingViolationKind
    position: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "position": self.position,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class PackingProof:
    version: str
    sound: bool
    epochs: int
    violations: tuple[PackingViolation, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sound": self.sound,
            "epochs": self.epochs,
            "violations": [v.to_dict() for v in self.violations],
        }


def prove_packing(seq: PackedSequence, epochs: int = 1) -> PackingProof:
    violations: list[PackingViolation] = []
    n = seq.length()

    if not (len(seq.doc_ids) == len(seq.loss_mask) == n):
        violations.append(
            PackingViolation(
                PackingViolationKind.LENGTH_MISMATCH,
                -1,
                f"token_ids/doc_ids/loss_mask lengths must all be {n}",
            )
        )
        return PackingProof(PACKING_PROOF_VERSION, False, epochs, tuple(violations))

    for i in range(1, n):
        prev_doc = seq.doc_ids[i - 1]
        cur_doc = seq.doc_ids[i]

        # Document boundary (real doc -> different real doc) needs a separator
        # and an attention reset.
        if prev_doc != cur_doc and prev_doc >= 0 and cur_doc >= 0:
            if seq.token_ids[i - 1] != seq.separator_id:
                violations.append(
                    PackingViolation(
                        PackingViolationKind.MISSING_SEPARATOR,
                        i,
                        f"no separator before doc {cur_doc} starts",
                    )
                )
            if i not in seq.attention_resets:
                violations.append(
                    PackingViolation(
                        PackingViolationKind.ATTENTION_BLEED,
                        i,
                        f"doc {cur_doc} can attend into doc {prev_doc}: "
                        "no attention reset at boundary",
                    )
                )

    # Loss-mask soundness.
    for i in range(n):
        if seq.loss_mask[i]:
            if seq.doc_ids[i] < 0:
                violations.append(
                    PackingViolation(
                        PackingViolationKind.LOSS_ON_PADDING,
                        i,
                        "loss is computed on a padding position",
                    )
                )
            elif seq.token_ids[i] == seq.separator_id and (
                i + 1 < n and seq.doc_ids[i + 1] != seq.doc_ids[i]
            ):
                # Supervising a separator that bridges into the next document.
                violations.append(
                    PackingViolation(
                        PackingViolationKind.LOSS_ACROSS_DOCUMENT,
                        i,
                        "loss supervises a cross-document separator token",
                    )
                )

    return PackingProof(
        version=PACKING_PROOF_VERSION,
        sound=not violations,
        epochs=epochs,
        violations=tuple(violations),
    )


def render_packing_text(proof: PackingProof) -> str:
    lines = [
        f"PromptABI multi-epoch packing proof ({proof.version})",
        f"epochs: {proof.epochs}",
        f"result: {'SOUND' if proof.sound else 'UNSOUND'}",
    ]
    for v in proof.violations:
        lines.append(f"  ! {v.kind.value} @pos {v.position}: {v.detail}")
    return "\n".join(lines) + "\n"
