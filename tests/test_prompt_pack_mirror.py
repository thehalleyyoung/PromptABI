"""Tests for signed offline prompt-pack mirrors (step 249)."""

from __future__ import annotations

import dataclasses

from promptabi.prompt_pack_mirror import (
    MirrorEntry,
    MirrorFindingKind,
    MirrorManifest,
    render_mirror_verification_text,
    sign_mirror,
    verify_mirror,
    verify_mirrored_artifact,
)

KEY = b"shared-offline-key"


def _manifest() -> MirrorManifest:
    return MirrorManifest(
        mirror_id="airgap-2024",
        created="2024-06-01",
        entries=(
            MirrorEntry("alpha", "1.0.0", "d-alpha", 1024),
            MirrorEntry("beta", "2.0.0", "d-beta", 2048),
        ),
        registry_head="head-hash",
    )


def test_sign_and_verify_roundtrip() -> None:
    signed = sign_mirror(_manifest(), KEY, key_id="org-key-1")
    result = verify_mirror(signed, KEY)
    assert result.valid


def test_signature_is_order_independent() -> None:
    a = sign_mirror(_manifest(), KEY, "k")
    reordered = MirrorManifest(
        mirror_id="airgap-2024",
        created="2024-06-01",
        entries=(
            MirrorEntry("beta", "2.0.0", "d-beta", 2048),
            MirrorEntry("alpha", "1.0.0", "d-alpha", 1024),
        ),
        registry_head="head-hash",
    )
    b = sign_mirror(reordered, KEY, "k")
    assert a.signature == b.signature


def test_wrong_key_rejected() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    result = verify_mirror(signed, b"attacker-key")
    assert not result.valid
    assert any(f.kind is MirrorFindingKind.BAD_SIGNATURE for f in result.findings)


def test_tampered_manifest_rejected() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    tampered_entries = (
        MirrorEntry("alpha", "1.0.0", "FORGED", 1024),
        MirrorEntry("beta", "2.0.0", "d-beta", 2048),
    )
    tampered = dataclasses.replace(
        signed, manifest=dataclasses.replace(signed.manifest, entries=tampered_entries)
    )
    result = verify_mirror(tampered, KEY)
    assert not result.valid
    assert any(f.kind is MirrorFindingKind.BAD_SIGNATURE for f in result.findings)


def test_artifact_matches_pinned_digest() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    result = verify_mirrored_artifact(signed, KEY, "alpha", "1.0.0", "d-alpha")
    assert result.valid


def test_artifact_digest_mismatch_rejected() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    result = verify_mirrored_artifact(signed, KEY, "alpha", "1.0.0", "wrong")
    assert not result.valid
    assert any(
        f.kind is MirrorFindingKind.ARTIFACT_DIGEST_MISMATCH for f in result.findings
    )


def test_artifact_not_in_mirror_rejected() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    result = verify_mirrored_artifact(signed, KEY, "ghost", "9.9.9", "x")
    assert any(
        f.kind is MirrorFindingKind.ARTIFACT_NOT_IN_MIRROR for f in result.findings
    )


def test_render_text() -> None:
    signed = sign_mirror(_manifest(), KEY, "k")
    result = verify_mirror(signed, KEY)
    assert "VALID" in render_mirror_verification_text(result)
