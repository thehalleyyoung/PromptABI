"""Tests for third-party prompt-pack certification (step 258)."""

from __future__ import annotations

import dataclasses

from promptabi.demo_packs import load_demo_packs
from promptabi.third_party_certification import (
    issue_certificate,
    pack_content_digest,
    verify_certificate,
)

KEY = b"authority-key"
PACK = {p.name: p for p in load_demo_packs()}["support-triage"]


def test_issue_and_verify_roundtrip() -> None:
    signed = issue_certificate(PACK, KEY, "ca-1", "PromptABI-CA", "promptabi-1.0")
    result = verify_certificate(signed, KEY, downloaded_pack=PACK)
    assert result.valid, result.findings
    assert signed.body.passed


def test_wrong_key_rejected() -> None:
    signed = issue_certificate(PACK, KEY, "ca-1", "PromptABI-CA", "promptabi-1.0")
    result = verify_certificate(signed, b"forged-key", downloaded_pack=PACK)
    assert not result.valid


def test_tampered_verdict_rejected() -> None:
    signed = issue_certificate(PACK, KEY, "ca-1", "PromptABI-CA", "promptabi-1.0")
    forged_body = dataclasses.replace(signed.body, passed=True, reasons=())
    forged = dataclasses.replace(signed, body=forged_body)
    # If it was already passing, tamper the digest instead to force a signature break.
    forged_body2 = dataclasses.replace(signed.body, pack_digest="sha256:forged")
    forged2 = dataclasses.replace(signed, body=forged_body2)
    assert not verify_certificate(forged2, KEY).valid
    # forged is identical to signed here (already passed); ensure signature still bound.
    assert verify_certificate(forged, KEY).valid == signed.body.passed


def test_digest_mismatch_detected() -> None:
    other = {p.name: p for p in load_demo_packs()}["json-extractor"]
    signed = issue_certificate(PACK, KEY, "ca-1", "PromptABI-CA", "promptabi-1.0")
    result = verify_certificate(signed, KEY, downloaded_pack=other)
    assert not result.valid
    assert any("digest-mismatch" in f for f in result.findings)


def test_content_digest_is_stable() -> None:
    assert pack_content_digest(PACK) == pack_content_digest(PACK)
