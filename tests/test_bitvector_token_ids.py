"""Tests for bit-vector token-id encodings (step 229)."""

from __future__ import annotations

import json

import pytest

from promptabi.bitvector_token_ids import (
    TokenIdContract,
    TokenIdFindingKind,
    TokenIdRange,
    bit_width,
    encode_token_id,
    render_token_id_report_json,
    render_token_id_report_text,
    verify_token_id_contract,
)


def test_bit_width() -> None:
    assert bit_width(1) == 1
    assert bit_width(2) == 1
    assert bit_width(256) == 8
    assert bit_width(257) == 9
    assert bit_width(32000) == 15


def test_encode_token_id_big_endian() -> None:
    assert encode_token_id(5, 4) == (0, 1, 0, 1)
    assert encode_token_id(0, 3) == (0, 0, 0)
    with pytest.raises(ValueError):
        encode_token_id(16, 4)


def test_safe_contract_passes() -> None:
    contract = TokenIdContract(
        vocab_size=32000,
        content_ranges=(TokenIdRange(name="content", low=3, high=31999),),
        special_ids=frozenset({0, 1, 2}),
        name="llama-like",
    )
    report = verify_token_id_contract(contract)
    assert report.ok
    assert report.exact_safe
    # When z3 is present, the bit-vector path must agree.
    if report.smt_safe is not None:
        assert report.smt_safe is True


def test_special_collision_detected() -> None:
    contract = TokenIdContract(
        vocab_size=1000,
        content_ranges=(TokenIdRange(name="content", low=0, high=999),),
        special_ids=frozenset({2}),
    )
    report = verify_token_id_contract(contract)
    assert not report.ok
    assert any(f.kind is TokenIdFindingKind.SPECIAL_COLLISION for f in report.findings)
    if report.smt_safe is not None:
        assert report.smt_safe is False


def test_out_of_vocab_detected() -> None:
    contract = TokenIdContract(
        vocab_size=100,
        content_ranges=(TokenIdRange(name="content", low=50, high=120),),
        special_ids=frozenset(),
    )
    report = verify_token_id_contract(contract)
    assert not report.ok
    assert any(f.kind is TokenIdFindingKind.OUT_OF_VOCAB for f in report.findings)


def test_smt_agrees_with_exact() -> None:
    contract = TokenIdContract(
        vocab_size=50257,
        content_ranges=(TokenIdRange(name="content", low=0, high=50255),),
        special_ids=frozenset({50256}),
        name="gpt2-like",
    )
    report = verify_token_id_contract(contract)
    # no SMT_DISAGREEMENT finding regardless of backend availability
    assert all(f.kind is not TokenIdFindingKind.SMT_DISAGREEMENT for f in report.findings)
    assert report.ok


def test_render_round_trips() -> None:
    contract = TokenIdContract(
        vocab_size=1000,
        content_ranges=(TokenIdRange(name="content", low=0, high=999),),
        special_ids=frozenset({2}),
    )
    report = verify_token_id_contract(contract)
    payload = json.loads(render_token_id_report_json(report))
    assert payload["ok"] is False
    assert "bit-vector token-id" in render_token_id_report_text(report)
