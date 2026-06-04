import json
from pathlib import Path

from promptabi import ArtifactKind, ArtifactLocation, ArtifactProvenance, TokenizerArtifact
from promptabi.cli import main
from promptabi.loaders import ArtifactLoader
from promptabi.tokenizer_drift import (
    TokenizerDriftKind,
    analyze_tokenizer_config_drift,
    load_tokenizer_config_snapshot,
)


def test_tokenizer_config_snapshot_extracts_real_config_surfaces(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tok"
    _write_tokenizer_revision(
        tokenizer_dir,
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        eos_id=2,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["</tool_call>"],
        add_bos=True,
    )

    snapshot = load_tokenizer_config_snapshot(tokenizer_dir, revision="rev-a")

    assert snapshot.revision == "rev-a"
    assert snapshot.eos_token == "</s>"
    assert snapshot.eos_token_id == 2
    assert snapshot.added_tokens == (("</s>", 2, True), ("<tool>", 32000, False))
    assert snapshot.normalizer_signature == '{"type":"Lowercase"}'
    assert snapshot.chat_template_length == 62
    assert snapshot.stop_sequences == ("</tool_call>",)
    assert snapshot.stop_token_ids == (2,)


def test_tokenizer_drift_detects_special_template_normalizer_added_bos_and_stop_changes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_tokenizer_revision(
        baseline,
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        eos_id=2,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    _write_tokenizer_revision(
        current,
        chat_template="<|start_header_id|>{{ messages[-1].role }}<|end_header_id|>",
        eos_id=128009,
        added_tool_id=32042,
        normalizer={"type": "NFC"},
        stop_strings=["<|eot_id|>"],
        add_bos=True,
    )
    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="llama-current",
        location=ArtifactLocation(path=str(current)),
        provenance=ArtifactProvenance(version="current-rev"),
        metadata=(("drift_baseline_path", "../baseline"), ("drift_baseline_revision", "baseline-rev")),
    )
    loaded = ArtifactLoader().load(artifact)

    report = analyze_tokenizer_config_drift((loaded,))

    fields = {finding.field: finding for finding in report.findings}
    assert report.abstentions == ()
    assert fields["special_tokens"].kind is TokenizerDriftKind.SPECIAL_TOKEN_ID
    assert fields["chat_template_sha256"].kind is TokenizerDriftKind.CHAT_TEMPLATE
    assert fields["normalizer_signature"].kind is TokenizerDriftKind.NORMALIZATION
    assert fields["added_tokens"].kind is TokenizerDriftKind.ADDED_TOKENS
    assert fields["add_bos_token"].kind is TokenizerDriftKind.BOS_EOS
    assert fields["stop_sequences"].kind is TokenizerDriftKind.STOP_POLICY
    assert fields["stop_token_ids"].baseline == (2,)
    assert fields["stop_token_ids"].current == (128009,)
    assert all(finding.baseline_revision == "baseline-rev" for finding in report.findings)
    assert all(finding.current_revision == "current-rev" for finding in report.findings)


def test_verify_tokenizer_drift_runs_through_real_cli(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_tokenizer_revision(
        baseline,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    _write_tokenizer_revision(
        current,
        chat_template="{{ '<s>' + messages[0].content }}",
        eos_id=128009,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["<|eot_id|>"],
        add_bos=False,
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "tokenizer-drift-demo",
                "checks": ["tokenizer-config-drift"],
                "artifacts": {
                    "tok": {
                        "kind": "tokenizer",
                        "path": "current",
                        "version": "current-rev",
                        "metadata": {
                            "drift_baseline_path": "../baseline",
                            "drift_baseline_revision": "baseline-rev",
                        },
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    drift_diagnostics = [
        diagnostic for diagnostic in payload["diagnostics"] if diagnostic["rule_id"] == "tokenizer-drift"
    ]
    assert exit_code == 1
    assert {diagnostic["properties"]["kind"] for diagnostic in drift_diagnostics} >= {
        "chat-template-change",
        "special-token-id-change",
        "stop-policy-change",
    }
    assert any("chat_template_sha256" in diagnostic["message"] for diagnostic in drift_diagnostics)
    assert any(step["action"] == "compare artifact revisions" for step in drift_diagnostics[0]["witness"]["steps"])
    assert captured.err == ""


def test_verify_tokenizer_drift_reports_clean_baseline(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_tokenizer_revision(
        baseline,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    _write_tokenizer_revision(
        current,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        added_tool_id=32000,
        normalizer={"type": "Lowercase"},
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "tokenizer-drift-clean",
                "checks": ["tokenizer-config-drift"],
                "artifacts": {
                    "tok": {
                        "kind": "tokenizer",
                        "path": "current",
                        "version": "current-rev",
                        "metadata": {"drift_baseline_path": "../baseline"},
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]] == ["tokenizer-drift-clean"]


def _write_tokenizer_revision(
    root: Path,
    *,
    chat_template: str,
    eos_id: int,
    added_tool_id: int,
    normalizer: dict[str, str],
    stop_strings: list[str],
    add_bos: bool,
) -> None:
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "eos_token_id": eos_id,
                "add_bos_token": add_bos,
                "chat_template": chat_template,
                "added_tokens": [
                    {"id": eos_id, "content": "</s>", "special": True},
                    {"id": added_tool_id, "content": "<tool>", "special": False},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "normalizer": normalizer,
                "added_tokens": [
                    {"id": eos_id, "content": "</s>", "special": True},
                    {"id": added_tool_id, "content": "<tool>", "special": False},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"stop_strings": stop_strings, "eos_token_id": eos_id}, sort_keys=True),
        encoding="utf-8",
    )
