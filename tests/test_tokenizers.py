from pathlib import Path

import pytest

from promptabi import ArtifactKind, ArtifactLocation, TokenizerArtifact
from promptabi.tokenizer_diff import (
    TokenizerDifferentialCase,
    TokenizerExpectation,
    run_tokenizer_differential,
)
from promptabi.tokenizers import (
    ByteLevelTokenizer,
    NormalizationRule,
    TokenizerBackend,
    apply_normalization,
    load_tokenizer,
)


def test_byte_level_tokenizer_tracks_added_tokens_utf8_spans_and_round_trip() -> None:
    tokenizer = ByteLevelTokenizer(
        added_tokens=("<eos>",),
        special_tokens={"<eos>": 2},
        normalization=(NormalizationRule.NFC,),
    )

    encoded = tokenizer.encode("Cafe\u0301<eos>")

    assert encoded.normalized_text == "Café<eos>"
    assert encoded.normalization_steps == ("nfc",)
    assert encoded.token_ids == (67, 97, 102, 195, 169, 2)
    assert encoded.tokens[-1].to_dict() == {
        "id": 2,
        "text": "<eos>",
        "byte_span": [5, 10],
        "special": True,
        "added": True,
    }
    assert tokenizer.decode(encoded.token_ids).text == "Café<eos>"
    assert tokenizer.decode(encoded.token_ids, skip_special_tokens=True).text == "Café"
    assert tokenizer.round_trip("Café<eos>").normalized_match is True


def test_apply_normalization_is_deterministic_and_ordered() -> None:
    normalized, steps = apply_normalization("  ℌELLO  ", ("strip", "nfkc", "lowercase"))

    assert normalized == "hello"
    assert steps == ("strip", "nfkc", "lowercase")


def test_huggingface_tokenizers_adapter_matches_real_byte_level_bpe(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

    raw = tokenizers.Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.normalizer = normalizers.Lowercase()
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=80, special_tokens=["[UNK]", "<s>"], show_progress=False)
    raw.train_from_iterator(
        [
            "Hello <tool> café",
            "Tool calls must preserve bytes.",
            "HELLO promptabi",
        ],
        trainer=trainer,
    )
    raw.add_tokens(["<tool>"])
    tokenizer_path = tmp_path / "tokenizer.json"
    raw.save(str(tokenizer_path))

    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="hf-byte-bpe",
        location=ArtifactLocation(path=str(tokenizer_path)),
        added_tokens=("<tool>",),
    )
    adapter = load_tokenizer(artifact)
    text = "Hello <tool> café"

    expected = raw.encode(text)
    actual = adapter.encode(text)

    assert actual.backend is TokenizerBackend.HUGGINGFACE_TOKENIZERS
    assert actual.token_ids == tuple(expected.ids)
    assert actual.token_texts == tuple(expected.tokens)
    assert actual.normalized_text == "hello <tool> café"
    assert any(token.added and token.text == "<tool>" for token in actual.tokens)
    assert adapter.decode(actual.token_ids).text == raw.decode(expected.ids)


def test_tiktoken_adapter_matches_real_cl100k_base() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="cl100k",
        location=ArtifactLocation(uri="memory://cl100k_base"),
        family="tiktoken",
        metadata=(("encoding", "cl100k_base"),),
    )
    adapter = load_tokenizer(artifact)
    text = "hello 🌍<|endoftext|>"

    expected_ids = tuple(encoding.encode(text, allowed_special="all"))
    actual = adapter.encode(text, add_special_tokens=True)

    assert actual.backend is TokenizerBackend.TIKTOKEN
    assert actual.token_ids == expected_ids
    assert actual.tokens[-1].special is True
    assert adapter.decode(actual.token_ids).text == encoding.decode(list(expected_ids))
    assert adapter.round_trip("plain unicode 🌍").exact_match is True


def test_sentencepiece_adapter_matches_real_model(tmp_path: Path) -> None:
    sentencepiece = pytest.importorskip("sentencepiece")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "promptabi verifies tokenizer contracts",
                "sentencepiece round trips unicode cafe",
                "tool schemas and stop strings compose",
            ]
        ),
        encoding="utf-8",
    )
    model_prefix = tmp_path / "spm"
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        hard_vocab_limit=False,
    )
    model_path = tmp_path / "spm.model"
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="sentencepiece",
        location=ArtifactLocation(path=str(model_path)),
        family="sentencepiece",
    )
    adapter = load_tokenizer(artifact)
    text = "promptabi verifies unicode"

    actual = adapter.encode(text)

    assert actual.backend is TokenizerBackend.SENTENCEPIECE
    assert actual.token_ids == tuple(processor.EncodeAsIds(text))
    assert actual.token_texts == tuple(processor.EncodeAsPieces(text))
    assert adapter.decode(actual.token_ids).text == processor.DecodeIds(list(actual.token_ids))


def test_tokenizer_differential_harness_covers_byte_level_normalization_and_flags() -> None:
    tokenizer = ByteLevelTokenizer(
        added_tokens=("<eos>", "<tool>"),
        special_tokens={"<eos>": 2},
        normalization=(NormalizationRule.STRIP, NormalizationRule.NFC),
    )
    text = " Cafe\u0301<tool><eos> "
    normalized = "Café<tool><eos>"
    expected_ids = tuple(normalized[:4].encode("utf-8")) + (256, 2)

    report = run_tokenizer_differential(
        tokenizer,
        [
            TokenizerDifferentialCase(
                name="byte-normalization-added-special",
                text=text,
                expectation=TokenizerExpectation(
                    token_ids=expected_ids,
                    token_texts=("0x43", "0x61", "0x66", "0xc3", "0xa9", "<tool>", "<eos>"),
                    decoded_text=normalized,
                    normalized_text=normalized,
                    added_token_ids=frozenset({256, 2}),
                    special_token_ids=frozenset({2}),
                    normalization_steps=("strip", "nfc"),
                    byte_spans_required=True,
                    round_trip_normalized=True,
                ),
            )
        ],
    )

    report.assert_ok()
    assert report.to_dict() == {
        "backend": "byte-level",
        "cases_run": 1,
        "ok": True,
        "mismatches": [],
    }


def test_tokenizer_differential_harness_reports_stable_mismatches() -> None:
    report = run_tokenizer_differential(
        ByteLevelTokenizer(),
        [
            TokenizerDifferentialCase(
                name="intentional-mismatch",
                text="A",
                expectation=TokenizerExpectation(token_ids=(66,), decoded_text="B"),
            )
        ],
    )

    assert report.ok is False
    assert report.to_dict()["mismatches"] == [
        {
            "case_name": "intentional-mismatch",
            "field": "token_ids",
            "expected": [66],
            "actual": [65],
        },
        {
            "case_name": "intentional-mismatch",
            "field": "decoded_text",
            "expected": "B",
            "actual": "A",
        },
    ]


def test_tokenizer_differential_harness_compares_huggingface_specials_and_added_tokens(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers

    raw = tokenizers.Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=96, special_tokens=["[UNK]", "<s>", "</s>"], show_progress=False)
    raw.train_from_iterator(
        [
            "Hello <tool> café",
            "Tool calls preserve bytes",
            "PromptABI checks special tokens",
        ],
        trainer=trainer,
    )
    raw.add_tokens(["<tool>"])
    raw.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", raw.token_to_id("<s>")),
            ("</s>", raw.token_to_id("</s>")),
        ],
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    raw.save(str(tokenizer_path))
    adapter = load_tokenizer(
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="hf-byte-bpe",
            location=ArtifactLocation(path=str(tokenizer_path)),
            added_tokens=("<tool>",),
        )
    )
    text = "Hello <tool> café"
    encoded = raw.encode(text, add_special_tokens=True)

    report = run_tokenizer_differential(
        adapter,
        [
            TokenizerDifferentialCase(
                name="hf-byte-bpe-special-added-normalized",
                text=text,
                add_special_tokens=True,
                expectation=TokenizerExpectation(
                    token_ids=tuple(encoded.ids),
                    token_texts=tuple(encoded.tokens),
                    decoded_text=raw.decode(encoded.ids, skip_special_tokens=False),
                    normalized_text=raw.normalizer.normalize_str(text),
                    special_token_ids=frozenset(
                        token_id for token_id, special in zip(encoded.ids, encoded.special_tokens_mask, strict=True) if special
                    ),
                    added_token_ids=frozenset({raw.token_to_id("<tool>")}),
                ),
            ),
            TokenizerDifferentialCase(
                name="hf-byte-bpe-skip-special-decode",
                text=text,
                add_special_tokens=True,
                skip_special_tokens=True,
                expectation=TokenizerExpectation(
                    token_ids=tuple(encoded.ids),
                    decoded_text=raw.decode(encoded.ids, skip_special_tokens=True),
                    special_token_ids=frozenset(
                        token_id for token_id, special in zip(encoded.ids, encoded.special_tokens_mask, strict=True) if special
                    ),
                    added_token_ids=frozenset({raw.token_to_id("<tool>")}),
                ),
            ),
        ],
    )

    report.assert_ok()


def test_tokenizer_differential_harness_compares_tiktoken_special_and_byte_fallback() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    adapter = load_tokenizer(
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="cl100k",
            location=ArtifactLocation(uri="memory://cl100k_base"),
            family="tiktoken",
            metadata=(("encoding", "cl100k_base"),),
        )
    )
    text = "bytes:\xff emoji:🌍 <|endoftext|>"
    ids = tuple(encoding.encode(text, allowed_special="all"))
    special_id = encoding._special_tokens["<|endoftext|>"]

    report = run_tokenizer_differential(
        adapter,
        [
            TokenizerDifferentialCase(
                name="tiktoken-special-byte-fallback",
                text=text,
                add_special_tokens=True,
                expectation=TokenizerExpectation(
                    token_ids=ids,
                    decoded_text=encoding.decode(list(ids)),
                    special_token_ids=frozenset({special_id}),
                    added_token_ids=frozenset({special_id}),
                    byte_spans_required=True,
                    round_trip_normalized=True,
                ),
            )
        ],
    )

    report.assert_ok()


def test_tokenizer_differential_harness_compares_sentencepiece_decode_and_pieces(tmp_path: Path) -> None:
    sentencepiece = pytest.importorskip("sentencepiece")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "promptabi differential harness",
                "sentencepiece tests decode pieces",
                "unicode café and emoji earth",
            ]
        ),
        encoding="utf-8",
    )
    model_prefix = tmp_path / "spm-diff"
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=72,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        hard_vocab_limit=False,
    )
    model_path = tmp_path / "spm-diff.model"
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    adapter = load_tokenizer(
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="sentencepiece",
            location=ArtifactLocation(path=str(model_path)),
            family="sentencepiece",
        )
    )
    text = "promptabi unicode café"
    ids = tuple(processor.EncodeAsIds(text))

    report = run_tokenizer_differential(
        adapter,
        [
            TokenizerDifferentialCase(
                name="sentencepiece-pieces-decode",
                text=text,
                expectation=TokenizerExpectation(
                    token_ids=ids,
                    token_texts=tuple(processor.EncodeAsPieces(text)),
                    decoded_text=processor.DecodeIds(list(ids)),
                    special_token_ids=frozenset(token_id for token_id in ids if processor.IsControl(token_id) or processor.IsUnknown(token_id)),
                ),
            )
        ],
    )

    report.assert_ok()
