import json
from pathlib import Path

import pytest

from promptabi import ArtifactKind, ArtifactLocation, StopPolicyArtifact
from promptabi.cli import main
from promptabi.stop_analysis import analyze_stop_policy_tokenizer
from promptabi.tokenizers import ByteLevelTokenizer, NormalizationRule


def test_stop_analysis_reports_alignment_collisions_specials_and_unreachable_ids() -> None:
    tokenizer = ByteLevelTokenizer(
        added_tokens=("<eos>",),
        special_tokens={"<eos>": 2},
        normalization=(NormalizationRule.NFC,),
    )
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("END", "ENDIF", "<eos>", "e\u0301", "é"),
        stop_token_ids=(2, 999999),
    )

    report = analyze_stop_policy_tokenizer(policy, tokenizer)

    assert report.tokenizer_backend == "byte-level"
    assert {item.stop_sequence: item.token_ids for item in report.sequences}["<eos>"] == (2,)
    assert [item.token_id for item in report.unreachable_token_ids] == [999999]
    assert any(
        collision.level == "string"
        and collision.relation == "prefix"
        and collision.shorter == "END"
        and collision.longer == "ENDIF"
        for collision in report.collisions
    )
    assert any(collision.level == "normalized-string" for collision in report.normalization_collisions)
    assert [item.stop_sequence for item in report.special_interactions] == ["<eos>"]
    normalizing = {item.stop_sequence for item in report.lossy_or_normalizing_sequences}
    assert "e\u0301" in normalizing
    assert {item.stop_sequence for item in report.multi_token_sequences} >= {"END", "ENDIF"}


def test_stop_tokenizer_cli_check_emits_concrete_diagnostics(tmp_path: Path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "stop-tokenizer-fixture",
                "checks": ["stop-tokenizer-analysis"],
                "artifacts": {
                    "byte-tokenizer": {
                        "kind": "tokenizer",
                        "uri": "memory://byte",
                        "family": "byte-level",
                        "added_tokens": ["<STOP>"],
                    },
                    "stop-policy": {
                        "kind": "stop-policy",
                        "uri": "memory://stops",
                        "stop_sequences": ["END", "ENDIF", "<STOP>"],
                        "stop_token_ids": [999999],
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = {diagnostic["rule_id"] for diagnostic in payload["diagnostics"]}
    assert exit_code == 0
    assert payload["ok"] is True
    assert {
        "stop-tokenizer-alignment",
        "stop-tokenizer-collision",
        "stop-tokenizer-special-interaction",
        "stop-tokenizer-unreachable",
    }.issubset(rule_ids)
    unreachable = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "stop-tokenizer-unreachable"
    ][0]
    assert unreachable["check_modes"] == ["sound"]
    assert "999999" in unreachable["message"]
    assert captured.err == ""


def test_stop_analysis_matches_real_huggingface_tokenizer(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    raw = tokenizers.Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=80,
        special_tokens=["[UNK]", "<|stop|>"],
        show_progress=False,
    )
    raw.train_from_iterator(
        [
            "Observation: done",
            "Observation",
            "Observe",
            "<|stop|>",
        ],
        trainer=trainer,
    )
    raw.add_special_tokens(["<|stop|>"])

    from promptabi import ArtifactLocation, TokenizerArtifact, load_tokenizer

    tokenizer_path = tmp_path / "tokenizer.json"
    raw.save(str(tokenizer_path))
    tokenizer = load_tokenizer(
        TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name="hf-stop-tokenizer",
            location=ArtifactLocation(path=str(tokenizer_path)),
            added_tokens=("<|stop|>",),
        )
    )
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("Observation", "Observation:", "<|stop|>"),
    )

    report = analyze_stop_policy_tokenizer(policy, tokenizer)

    assert any(item.stop_sequence == "<|stop|>" and item.has_special_interaction for item in report.sequences)
    assert any(
        collision.level in {"string", "byte", "token"}
        and collision.shorter == "Observation"
        and collision.longer == "Observation:"
        for collision in report.collisions
    )
