"""Support signed offline prompt-pack mirrors (step 249).

An air-gapped or regulated consumer cannot reach the upstream marketplace, so it
installs from an **offline mirror**: a self-contained snapshot of selected pack
versions plus a manifest that pins each one's content digest.  For the mirror to
be trustworthy without a network it must be *signed* -- the consumer verifies the
manifest's authenticity with a shared key it already holds, then verifies each
downloaded artifact against the digest the signed manifest pins.

This module builds canonical mirror manifests, signs them with an HMAC over their
canonical bytes (no third-party crypto dependency), and verifies them offline:

* :func:`verify_mirror` proves the signature matches the manifest contents, so
  any tampering with the entry list or digests is caught.
* :func:`verify_mirrored_artifact` proves a specific downloaded artifact's actual
  digest matches the digest the signed mirror pins for that name and version.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_MIRROR_VERSION = "promptabi.prompt-pack-mirror.v1"


class MirrorFindingKind(StrEnum):
    BAD_SIGNATURE = "bad-signature"
    DUPLICATE_ENTRY = "duplicate-entry"
    ARTIFACT_NOT_IN_MIRROR = "artifact-not-in-mirror"
    ARTIFACT_DIGEST_MISMATCH = "artifact-digest-mismatch"


@dataclass(frozen=True, slots=True)
class MirrorFinding:
    kind: MirrorFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class MirrorEntry:
    name: str
    version: str
    digest: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class MirrorManifest:
    mirror_id: str
    created: str
    entries: tuple[MirrorEntry, ...]
    registry_head: str | None = None

    def canonical_entries(self) -> tuple[MirrorEntry, ...]:
        return tuple(sorted(self.entries, key=lambda e: (e.name, e.version)))

    def canonical_bytes(self) -> bytes:
        payload = {
            "version": PROMPT_PACK_MIRROR_VERSION,
            "mirror_id": self.mirror_id,
            "created": self.created,
            "registry_head": self.registry_head,
            "entries": [e.to_dict() for e in self.canonical_entries()],
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROMPT_PACK_MIRROR_VERSION,
            "mirror_id": self.mirror_id,
            "created": self.created,
            "registry_head": self.registry_head,
            "entries": [e.to_dict() for e in self.canonical_entries()],
        }


@dataclass(frozen=True, slots=True)
class SignedMirror:
    manifest: MirrorManifest
    key_id: str
    signature: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "key_id": self.key_id,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class MirrorVerification:
    version: str
    valid: bool
    findings: tuple[MirrorFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def _compute_signature(manifest: MirrorManifest, key: bytes) -> str:
    return hmac.new(key, manifest.canonical_bytes(), hashlib.sha256).hexdigest()


def sign_mirror(manifest: MirrorManifest, key: bytes, key_id: str) -> SignedMirror:
    """Produce a signed mirror an offline consumer can verify with ``key``."""

    return SignedMirror(
        manifest=manifest,
        key_id=key_id,
        signature=_compute_signature(manifest, key),
    )


def verify_mirror(signed: SignedMirror, key: bytes) -> MirrorVerification:
    """Prove the mirror's signature matches its manifest and is internally sound."""

    findings: list[MirrorFinding] = []

    expected = _compute_signature(signed.manifest, key)
    if not hmac.compare_digest(expected, signed.signature):
        findings.append(
            MirrorFinding(
                MirrorFindingKind.BAD_SIGNATURE,
                "signature does not match manifest contents or key",
            )
        )

    seen: set[tuple[str, str]] = set()
    for entry in signed.manifest.entries:
        key_pair = (entry.name, entry.version)
        if key_pair in seen:
            findings.append(
                MirrorFinding(
                    MirrorFindingKind.DUPLICATE_ENTRY,
                    f"{entry.name}@{entry.version} listed more than once",
                )
            )
        seen.add(key_pair)

    return MirrorVerification(
        version=PROMPT_PACK_MIRROR_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def verify_mirrored_artifact(
    signed: SignedMirror,
    key: bytes,
    name: str,
    version: str,
    actual_digest: str,
) -> MirrorVerification:
    """Prove a downloaded artifact matches the digest the signed mirror pins."""

    findings: list[MirrorFinding] = []

    base = verify_mirror(signed, key)
    findings.extend(base.findings)

    entry = next(
        (
            e
            for e in signed.manifest.entries
            if e.name == name and e.version == version
        ),
        None,
    )
    if entry is None:
        findings.append(
            MirrorFinding(
                MirrorFindingKind.ARTIFACT_NOT_IN_MIRROR,
                f"{name}@{version} is not pinned by this mirror",
            )
        )
    elif entry.digest != actual_digest:
        findings.append(
            MirrorFinding(
                MirrorFindingKind.ARTIFACT_DIGEST_MISMATCH,
                f"{name}@{version}: pinned {entry.digest} != actual {actual_digest}",
            )
        )

    return MirrorVerification(
        version=PROMPT_PACK_MIRROR_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def render_mirror_verification_text(result: MirrorVerification) -> str:
    lines = [
        f"PromptABI signed offline mirror ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.detail}")
    return "\n".join(lines) + "\n"
