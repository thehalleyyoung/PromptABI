"""Third-party prompt-pack certification (step 258).

A certification authority (the PromptABI project, or any org running PromptABI in
CI) runs the reusable-pack battery on a *third-party* pack and, if it passes,
issues a signed certificate binding the pack's content digest to the battery
result and the toolchain version that produced it.  A relying party can then
verify the certificate offline with the authority's key, and -- crucially --
re-bind it to the pack they actually downloaded by comparing digests.

The certificate is signed with an HMAC over its canonical bytes (consistent with
the offline-mirror signing in step 249), so tampering with the verdict, the
digest, or the battery list is detected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import StrEnum

from .demo_packs import DemoPack, certify_demo_pack

THIRD_PARTY_CERT_VERSION = "promptabi.third-party-cert.v1"


class CertRejectionKind(StrEnum):
    BATTERY_FAILED = "battery-failed"
    BAD_SIGNATURE = "bad-signature"
    DIGEST_MISMATCH = "digest-mismatch"


def pack_content_digest(pack: DemoPack) -> str:
    payload = {
        "name": pack.name,
        "version": pack.version,
        "template": pack.template,
        "schemas": sorted(s.name for s in pack.schemas),
        "stop_sequences": list(pack.stop_sequences),
        "sanitizers": sorted(pack.sanitizers),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class CertificateBody:
    version: str
    pack: str
    pack_version: str
    pack_digest: str
    battery: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]
    authority: str
    toolchain: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "version": self.version,
            "pack": self.pack,
            "pack_version": self.pack_version,
            "pack_digest": self.pack_digest,
            "battery": list(self.battery),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "authority": self.authority,
            "toolchain": self.toolchain,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class SignedCertificate:
    body: CertificateBody
    key_id: str
    signature: str

    def to_dict(self) -> dict[str, object]:
        return {
            "body": self.body.to_dict(),
            "key_id": self.key_id,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class CertVerification:
    version: str
    valid: bool
    findings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": list(self.findings),
        }


_BATTERY = ("rag-extension-points", "structured-output-schemas", "stop-and-roles")


def _sign(body: CertificateBody, key: bytes) -> str:
    return hmac.new(key, body.canonical_bytes(), hashlib.sha256).hexdigest()


def issue_certificate(
    pack: DemoPack,
    key: bytes,
    key_id: str,
    authority: str,
    toolchain: str,
) -> SignedCertificate:
    """Run the battery and issue a signed certificate (pass or fail)."""

    result = certify_demo_pack(pack)
    body = CertificateBody(
        version=THIRD_PARTY_CERT_VERSION,
        pack=pack.name,
        pack_version=pack.version,
        pack_digest=pack_content_digest(pack),
        battery=_BATTERY,
        passed=result.certified,
        reasons=result.reasons,
        authority=authority,
        toolchain=toolchain,
    )
    return SignedCertificate(body=body, key_id=key_id, signature=_sign(body, key))


def verify_certificate(
    signed: SignedCertificate,
    key: bytes,
    downloaded_pack: DemoPack | None = None,
) -> CertVerification:
    """Verify signature, battery verdict, and (optionally) digest binding."""

    findings: list[str] = []

    expected = _sign(signed.body, key)
    if not hmac.compare_digest(expected, signed.signature):
        findings.append(CertRejectionKind.BAD_SIGNATURE.value)

    if not signed.body.passed:
        findings.append(
            f"{CertRejectionKind.BATTERY_FAILED.value}: "
            + ",".join(signed.body.reasons)
        )

    if downloaded_pack is not None:
        actual = pack_content_digest(downloaded_pack)
        if actual != signed.body.pack_digest:
            findings.append(
                f"{CertRejectionKind.DIGEST_MISMATCH.value}: "
                f"{actual} != {signed.body.pack_digest}"
            )

    return CertVerification(
        version=THIRD_PARTY_CERT_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )
