"""Tests for private prompt-pack transparency logs (step 243)."""

from __future__ import annotations

import dataclasses

import pytest

from promptabi.prompt_pack_transparency import (
    PromptPackTransparencyLog,
    TransparencyAction,
    TransparencyFindingKind,
    prove_inclusion,
    render_log_verification_text,
    verify_inclusion,
    verify_log,
)


def _log() -> PromptPackTransparencyLog:
    log = PromptPackTransparencyLog()
    log.append(TransparencyAction.PUBLISH, "alpha", "1.0.0", "d-alpha-1")
    log.append(TransparencyAction.PUBLISH, "beta", "2.0.0", "d-beta-2")
    log.append(TransparencyAction.PUBLISH, "alpha", "1.1.0", "d-alpha-11")
    log.append(TransparencyAction.REVOKE, "alpha", "1.0.0", "d-alpha-1")
    return log


def test_log_is_hash_chained_and_verifies() -> None:
    log = _log()
    result = verify_log(log.entries, expected_head=log.head)
    assert result.valid
    assert result.head == log.head


def test_head_changes_with_each_append() -> None:
    log = PromptPackTransparencyLog()
    h0 = log.head
    log.append(TransparencyAction.PUBLISH, "a", "1.0.0", "d")
    h1 = log.head
    log.append(TransparencyAction.PUBLISH, "b", "1.0.0", "d")
    h2 = log.head
    assert len({h0, h1, h2}) == 3


def test_retroactive_edit_breaks_chain() -> None:
    log = _log()
    entries = list(log.entries)
    entries[1] = dataclasses.replace(entries[1], pack_digest="forged")
    result = verify_log(tuple(entries), expected_head=log.head)
    assert not result.valid
    kinds = {f.kind for f in result.findings}
    assert TransparencyFindingKind.BAD_ENTRY_HASH in kinds


def test_reordering_entries_is_detected() -> None:
    log = _log()
    entries = list(log.entries)
    entries[0], entries[1] = entries[1], entries[0]
    result = verify_log(tuple(entries))
    assert not result.valid


def test_expected_head_mismatch_detected() -> None:
    log = _log()
    result = verify_log(log.entries, expected_head="deadbeef")
    assert not result.valid
    assert any(
        f.kind is TransparencyFindingKind.HEAD_MISMATCH for f in result.findings
    )


def test_revoke_without_publish_rejected() -> None:
    log = PromptPackTransparencyLog()
    with pytest.raises(ValueError):
        log.append(TransparencyAction.REVOKE, "ghost", "9.9.9", "d")


def test_inclusion_proof_verifies_without_full_log() -> None:
    log = _log()
    proof = prove_inclusion(log, 1)
    result = verify_inclusion(proof)
    assert result.valid
    assert result.head == log.head


def test_inclusion_proof_of_head_entry() -> None:
    log = _log()
    last = len(log.entries) - 1
    proof = prove_inclusion(log, last)
    assert proof.suffix == ()
    assert verify_inclusion(proof).valid


def test_tampered_inclusion_proof_rejected() -> None:
    log = _log()
    proof = prove_inclusion(log, 0)
    bad_suffix = (dataclasses.replace(proof.suffix[0], pack_digest="x"),) + proof.suffix[1:]
    tampered = dataclasses.replace(proof, suffix=bad_suffix)
    assert not verify_inclusion(tampered).valid


def test_forged_head_in_proof_rejected() -> None:
    log = _log()
    proof = prove_inclusion(log, 1)
    forged = dataclasses.replace(proof, head="0" * 64)
    assert not verify_inclusion(forged).valid


def test_renderer_text() -> None:
    log = _log()
    result = verify_log(log.entries, expected_head=log.head)
    assert "VALID" in render_log_verification_text(result)
