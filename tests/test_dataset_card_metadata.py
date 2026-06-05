"""Tests for dataset-card PromptABI metadata (step 272)."""

from __future__ import annotations

from promptabi.dataset_card_metadata import (
    DatasetCardFindingKind,
    parse_dataset_card,
    render_dataset_card_text,
)

GOOD_CARD = {
    "promptabi": {
        "schema_version": "1",
        "template_digest": "sha256:abc",
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "supervised_role": "assistant",
        "special_tokens": ["<|eot_id|>"],
    }
}


def test_valid_card() -> None:
    result = parse_dataset_card(GOOD_CARD)
    assert result.valid
    assert result.metadata is not None
    assert result.metadata.supervised_role == "assistant"


def test_missing_block() -> None:
    result = parse_dataset_card({"license": "apache-2.0"})
    assert any(
        f.kind is DatasetCardFindingKind.MISSING_BLOCK for f in result.findings
    )


def test_missing_field() -> None:
    card = {"promptabi": {"template_digest": "x", "tokenizer": "y"}}
    result = parse_dataset_card(card)
    assert any(f.kind is DatasetCardFindingKind.MISSING_FIELD for f in result.findings)


def test_wrong_type() -> None:
    card = {
        "promptabi": {
            "template_digest": 123,
            "tokenizer": "y",
            "supervised_role": "assistant",
        }
    }
    result = parse_dataset_card(card)
    assert any(f.kind is DatasetCardFindingKind.WRONG_TYPE for f in result.findings)


def test_unknown_schema_version() -> None:
    card = {
        "promptabi": {
            "schema_version": "99",
            "template_digest": "x",
            "tokenizer": "y",
            "supervised_role": "assistant",
        }
    }
    result = parse_dataset_card(card)
    assert any(
        f.kind is DatasetCardFindingKind.UNKNOWN_SCHEMA_VERSION for f in result.findings
    )


def test_render_text_smoke() -> None:
    assert "dataset-card" in render_dataset_card_text(parse_dataset_card(GOOD_CARD))
