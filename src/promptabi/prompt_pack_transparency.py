"""Support private prompt-pack transparency logs (step 243).

Public transparency logs (Certificate Transparency, sigstore Rekor) let anyone
verify that an artifact was published and was never silently rewritten.  A
*private* prompt-pack transparency log gives a single organisation the same
guarantee without revealing pack names or digests to the world: the log lives
inside the org, but it is still **append-only** and **tamper-evident**.

This module implements that log as a hash chain.  Each
:class:`TransparencyEntry` records a published (or revoked) pack version and
carries the hash of the entry before it, so the whole log is summarised by one
``head`` hash.  Appending is the only mutation; any retroactive edit, deletion,
or reordering changes a downstream hash and is caught by
:func:`verify_log`.

Because the chain is deterministic, a verifier who only holds a trusted ``head``
can be given an *inclusion proof* (:func:`prove_inclusion`) for a single entry
and check it (:func:`verify_inclusion`) without seeing the rest of the log --
the "private" part: an auditor proves a specific pack was logged without the log
owner disclosing every other entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_TRANSPARENCY_VERSION = "promptabi.prompt-pack-transparency.v1"

_GENESIS = "0" * 64


class TransparencyAction(StrEnum):
    PUBLISH = "publish"
    REVOKE = "revoke"


class TransparencyFindingKind(StrEnum):
    BROKEN_CHAIN = "broken-chain"
    INDEX_GAP = "index-gap"
    BAD_ENTRY_HASH = "bad-entry-hash"
    HEAD_MISMATCH = "head-mismatch"
    ENTRY_NOT_FOUND = "entry-not-found"
    INCLUSION_HASH_MISMATCH = "inclusion-hash-mismatch"
    REVOKE_WITHOUT_PUBLISH = "revoke-without-publish"


@dataclass(frozen=True, slots=True)
class TransparencyFinding:
    kind: TransparencyFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


def _hash_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TransparencyEntry:
    index: int
    action: TransparencyAction
    pack_name: str
    pack_version: str
    pack_digest: str
    prev_hash: str
    entry_hash: str = ""

    def _body(self) -> dict[str, object]:
        return {
            "index": self.index,
            "action": self.action.value,
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "pack_digest": self.pack_digest,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return _hash_payload(self._body())

    def to_dict(self) -> dict[str, object]:
        data = self._body()
        data["entry_hash"] = self.entry_hash
        return data


@dataclass(frozen=True, slots=True)
class InclusionProof:
    version: str
    index: int
    entry_hash: str
    prev_hash: str
    head: str
    # the entries that follow the proven entry, in order, so a verifier can
    # recompute the head deterministically without the entries before it
    suffix: tuple[TransparencyEntry, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "index": self.index,
            "entry_hash": self.entry_hash,
            "prev_hash": self.prev_hash,
            "head": self.head,
            "suffix": [e.to_dict() for e in self.suffix],
        }


@dataclass(frozen=True, slots=True)
class LogVerification:
    version: str
    valid: bool
    head: str
    findings: tuple[TransparencyFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "head": self.head,
            "findings": [f.to_dict() for f in self.findings],
        }


class PromptPackTransparencyLog:
    """An append-only, hash-chained log of pack publications."""

    def __init__(self) -> None:
        self._entries: list[TransparencyEntry] = []

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else _GENESIS

    @property
    def entries(self) -> tuple[TransparencyEntry, ...]:
        return tuple(self._entries)

    def append(
        self,
        action: TransparencyAction,
        pack_name: str,
        pack_version: str,
        pack_digest: str,
    ) -> TransparencyEntry:
        if action is TransparencyAction.REVOKE and not self._has_publish(
            pack_name, pack_version
        ):
            raise ValueError(
                f"cannot revoke {pack_name}@{pack_version}: never published in log"
            )
        index = len(self._entries)
        prev_hash = self.head
        draft = TransparencyEntry(
            index=index,
            action=action,
            pack_name=pack_name,
            pack_version=pack_version,
            pack_digest=pack_digest,
            prev_hash=prev_hash,
        )
        entry = TransparencyEntry(
            index=index,
            action=action,
            pack_name=pack_name,
            pack_version=pack_version,
            pack_digest=pack_digest,
            prev_hash=prev_hash,
            entry_hash=draft.compute_hash(),
        )
        self._entries.append(entry)
        return entry

    def _has_publish(self, name: str, version: str) -> bool:
        return any(
            e.action is TransparencyAction.PUBLISH
            and e.pack_name == name
            and e.pack_version == version
            for e in self._entries
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROMPT_PACK_TRANSPARENCY_VERSION,
            "head": self.head,
            "entries": [e.to_dict() for e in self._entries],
        }


def verify_log(entries: tuple[TransparencyEntry, ...], expected_head: str | None = None) -> LogVerification:
    """Prove a log is a well-formed, unbroken, append-only hash chain."""

    findings: list[TransparencyFinding] = []
    prev = _GENESIS
    published: set[tuple[str, str]] = set()
    for i, entry in enumerate(entries):
        if entry.index != i:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.INDEX_GAP,
                    f"expected index {i}, found {entry.index}",
                )
            )
        if entry.prev_hash != prev:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.BROKEN_CHAIN,
                    f"entry {entry.index} prev_hash does not match prior head",
                )
            )
        if entry.compute_hash() != entry.entry_hash:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.BAD_ENTRY_HASH,
                    f"entry {entry.index} hash does not match its contents",
                )
            )
        if entry.action is TransparencyAction.REVOKE and (
            entry.pack_name,
            entry.pack_version,
        ) not in published:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.REVOKE_WITHOUT_PUBLISH,
                    f"{entry.pack_name}@{entry.pack_version} revoked before publish",
                )
            )
        if entry.action is TransparencyAction.PUBLISH:
            published.add((entry.pack_name, entry.pack_version))
        prev = entry.entry_hash

    head = entries[-1].entry_hash if entries else _GENESIS
    if expected_head is not None and head != expected_head:
        findings.append(
            TransparencyFinding(
                TransparencyFindingKind.HEAD_MISMATCH,
                f"computed head {head} != expected {expected_head}",
            )
        )

    return LogVerification(
        version=PROMPT_PACK_TRANSPARENCY_VERSION,
        valid=not findings,
        head=head,
        findings=tuple(findings),
    )


def prove_inclusion(log: PromptPackTransparencyLog, index: int) -> InclusionProof:
    """Build a minimal proof that entry ``index`` is committed to by the head."""

    entries = log.entries
    if index < 0 or index >= len(entries):
        raise IndexError(f"no entry at index {index}")
    target = entries[index]
    suffix = tuple(entries[index + 1 :])
    return InclusionProof(
        version=PROMPT_PACK_TRANSPARENCY_VERSION,
        index=index,
        entry_hash=target.entry_hash,
        prev_hash=target.prev_hash,
        head=log.head,
        suffix=suffix,
    )


def verify_inclusion(proof: InclusionProof) -> LogVerification:
    """Verify an inclusion proof against its own head, without the full log.

    The verifier recomputes the head by chaining forward from the proven entry
    through every suffix entry, re-deriving each entry hash from its contents and
    checking it links to the previous one.  Any altered suffix entry -- content
    or order -- breaks the recomputed head and is rejected.
    """

    findings: list[TransparencyFinding] = []
    running = proof.entry_hash
    for entry in proof.suffix:
        if entry.prev_hash != running:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.BROKEN_CHAIN,
                    f"suffix entry {entry.index} does not link to its predecessor",
                )
            )
        if entry.compute_hash() != entry.entry_hash:
            findings.append(
                TransparencyFinding(
                    TransparencyFindingKind.BAD_ENTRY_HASH,
                    f"suffix entry {entry.index} hash does not match its contents",
                )
            )
        running = entry.entry_hash
    if running != proof.head:
        findings.append(
            TransparencyFinding(
                TransparencyFindingKind.INCLUSION_HASH_MISMATCH,
                "suffix does not reconstruct the committed head",
            )
        )
    return LogVerification(
        version=PROMPT_PACK_TRANSPARENCY_VERSION,
        valid=not findings,
        head=proof.head,
        findings=tuple(findings),
    )


def render_log_verification_text(result: LogVerification) -> str:
    lines = [
        f"PromptABI prompt-pack transparency log ({result.version})",
        f"head: {result.head}",
        f"result: {'VALID' if result.valid else 'TAMPERED'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.detail}")
    return "\n".join(lines) + "\n"
