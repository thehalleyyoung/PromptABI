"""Provider fixture provenance attestations (step 288).

Conformance fixtures are only trustworthy if their provenance is verifiable: who
captured them, against which provider revision, when, and whether the captured
bytes have been altered since.  This module wraps a raw fixture payload in a
signed attestation (HMAC over canonical bytes plus metadata) and verifies it,
detecting tampering, stale revisions, and unknown signers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import StrEnum

FIXTURE_PROVENANCE_VERSION = "promptabi.fixture-provenance.v1"


class ProvenanceFindingKind(StrEnum):
    BAD_SIGNATURE = "bad-signature"
    PAYLOAD_TAMPERED = "payload-tampered"
    UNKNOWN_SIGNER = "unknown-signer"
    STALE_REVISION = "stale-revision"


@dataclass(frozen=True, slots=True)
class FixtureAttestation:
    fixture_id: str
    provider: str
    provider_revision: str
    captured_by: str
    captured_at: str
    payload_digest: str
    signature: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "provider": self.provider,
            "provider_revision": self.provider_revision,
            "captured_by": self.captured_by,
            "captured_at": self.captured_at,
            "payload_digest": self.payload_digest,
            "signature": self.signature,
        }


def _canonical_payload_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _signing_message(
    fixture_id: str,
    provider: str,
    provider_revision: str,
    captured_by: str,
    captured_at: str,
    payload_digest: str,
) -> bytes:
    return "\n".join(
        [
            FIXTURE_PROVENANCE_VERSION,
            fixture_id,
            provider,
            provider_revision,
            captured_by,
            captured_at,
            payload_digest,
        ]
    ).encode("utf-8")


def sign_fixture(
    *,
    fixture_id: str,
    provider: str,
    provider_revision: str,
    captured_by: str,
    captured_at: str,
    payload: object,
    secret: bytes,
) -> FixtureAttestation:
    digest = _canonical_payload_digest(payload)
    msg = _signing_message(
        fixture_id, provider, provider_revision, captured_by, captured_at, digest
    )
    signature = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return FixtureAttestation(
        fixture_id=fixture_id,
        provider=provider,
        provider_revision=provider_revision,
        captured_by=captured_by,
        captured_at=captured_at,
        payload_digest=digest,
        signature=signature,
    )


@dataclass(frozen=True, slots=True)
class ProvenanceFinding:
    kind: ProvenanceFindingKind
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ProvenanceResult:
    version: str
    trusted: bool
    findings: tuple[ProvenanceFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "trusted": self.trusted,
            "findings": [f.to_dict() for f in self.findings],
        }


def verify_fixture(
    attestation: FixtureAttestation,
    payload: object,
    *,
    known_signers: dict[str, bytes],
    current_revisions: dict[str, str] | None = None,
) -> ProvenanceResult:
    findings: list[ProvenanceFinding] = []

    secret = known_signers.get(attestation.captured_by)
    if secret is None:
        findings.append(
            ProvenanceFinding(
                ProvenanceFindingKind.UNKNOWN_SIGNER,
                f"signer {attestation.captured_by!r} not in trust store",
            )
        )
    else:
        msg = _signing_message(
            attestation.fixture_id,
            attestation.provider,
            attestation.provider_revision,
            attestation.captured_by,
            attestation.captured_at,
            attestation.payload_digest,
        )
        expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, attestation.signature):
            findings.append(
                ProvenanceFinding(
                    ProvenanceFindingKind.BAD_SIGNATURE,
                    "signature does not match attestation metadata",
                )
            )

    actual_digest = _canonical_payload_digest(payload)
    if not hmac.compare_digest(actual_digest, attestation.payload_digest):
        findings.append(
            ProvenanceFinding(
                ProvenanceFindingKind.PAYLOAD_TAMPERED,
                "payload digest differs from attested digest",
            )
        )

    if current_revisions is not None:
        current = current_revisions.get(attestation.provider)
        if current is not None and current != attestation.provider_revision:
            findings.append(
                ProvenanceFinding(
                    ProvenanceFindingKind.STALE_REVISION,
                    f"attested revision {attestation.provider_revision!r} != "
                    f"current {current!r}",
                )
            )

    return ProvenanceResult(
        version=FIXTURE_PROVENANCE_VERSION,
        trusted=not findings,
        findings=tuple(findings),
    )


def render_provenance_text(result: ProvenanceResult) -> str:
    lines = [
        f"PromptABI fixture provenance ({result.version})",
        f"result: {'TRUSTED' if result.trusted else 'UNTRUSTED'}",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value}: {f.detail}")
    return "\n".join(lines) + "\n"
