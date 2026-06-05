"""Prove training/eval tokenizer alignment over releases (step 267).

If the tokenizer used to *train* a model differs -- even subtly -- from the one
used to *evaluate* or *serve* it, every benchmark number and every served
response is measured against a different interface than the one trained.  Drift
sneaks in across releases: an added token, a changed merge, a flipped
``add_bos_token`` flag.

This module proves alignment by comparing two tokenizer *fingerprints* (vocab
digest, added-token list, special-token map, and the encodings of a fixed probe
set) and reporting the exact axis that diverged between a training pin and an
eval pin.  The probe-set comparison is a concrete differential: if two tokenizers
encode any probe string differently, they are not aligned, full stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

TOKENIZER_ALIGNMENT_VERSION = "promptabi.tokenizer-alignment.v1"


class AlignmentFindingKind(StrEnum):
    VOCAB_DIGEST_MISMATCH = "vocab-digest-mismatch"
    ADDED_TOKEN_MISMATCH = "added-token-mismatch"
    SPECIAL_TOKEN_MISMATCH = "special-token-mismatch"
    BOS_EOS_FLAG_MISMATCH = "bos-eos-flag-mismatch"
    PROBE_ENCODING_MISMATCH = "probe-encoding-mismatch"


@dataclass(frozen=True, slots=True)
class TokenizerFingerprint:
    release: str
    vocab_digest: str
    added_tokens: tuple[str, ...]
    special_tokens: dict[str, str]
    add_bos: bool
    add_eos: bool
    probe_encodings: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class AlignmentFinding:
    kind: AlignmentFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    version: str
    aligned: bool
    train_release: str
    eval_release: str
    findings: tuple[AlignmentFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "aligned": self.aligned,
            "train_release": self.train_release,
            "eval_release": self.eval_release,
            "findings": [f.to_dict() for f in self.findings],
        }


def prove_alignment(
    train: TokenizerFingerprint,
    eval_: TokenizerFingerprint,
) -> AlignmentResult:
    findings: list[AlignmentFinding] = []

    if train.vocab_digest != eval_.vocab_digest:
        findings.append(
            AlignmentFinding(
                AlignmentFindingKind.VOCAB_DIGEST_MISMATCH,
                f"{train.vocab_digest} != {eval_.vocab_digest}",
            )
        )
    if train.added_tokens != eval_.added_tokens:
        added = sorted(set(eval_.added_tokens) - set(train.added_tokens))
        removed = sorted(set(train.added_tokens) - set(eval_.added_tokens))
        findings.append(
            AlignmentFinding(
                AlignmentFindingKind.ADDED_TOKEN_MISMATCH,
                f"added={added} removed={removed}",
            )
        )
    if train.special_tokens != eval_.special_tokens:
        findings.append(
            AlignmentFinding(
                AlignmentFindingKind.SPECIAL_TOKEN_MISMATCH,
                f"{train.special_tokens} != {eval_.special_tokens}",
            )
        )
    if (train.add_bos, train.add_eos) != (eval_.add_bos, eval_.add_eos):
        findings.append(
            AlignmentFinding(
                AlignmentFindingKind.BOS_EOS_FLAG_MISMATCH,
                f"train(bos={train.add_bos},eos={train.add_eos}) != "
                f"eval(bos={eval_.add_bos},eos={eval_.add_eos})",
            )
        )

    shared_probes = set(train.probe_encodings) & set(eval_.probe_encodings)
    for probe in sorted(shared_probes):
        if train.probe_encodings[probe] != eval_.probe_encodings[probe]:
            findings.append(
                AlignmentFinding(
                    AlignmentFindingKind.PROBE_ENCODING_MISMATCH,
                    f"probe {probe!r}: {list(train.probe_encodings[probe])} != "
                    f"{list(eval_.probe_encodings[probe])}",
                )
            )

    return AlignmentResult(
        version=TOKENIZER_ALIGNMENT_VERSION,
        aligned=not findings,
        train_release=train.release,
        eval_release=eval_.release,
        findings=tuple(findings),
    )


def render_alignment_text(result: AlignmentResult) -> str:
    lines = [
        f"PromptABI train/eval tokenizer alignment ({result.version})",
        f"{result.train_release} vs {result.eval_release}: "
        f"{'ALIGNED' if result.aligned else 'DIVERGED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
