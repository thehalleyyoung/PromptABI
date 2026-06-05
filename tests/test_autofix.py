import json
from hashlib import sha256
from pathlib import Path

from promptabi import AutoFixKind, guarded_autofix_preview, low_risk_autofix
from promptabi.cli import main


def test_autofix_preview_and_write_lockfile_against_real_config(tmp_path: Path, capsys) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "autofix-lock",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": schema.name,
                        "sha256": sha256(schema.read_bytes()).hexdigest(),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lockfile = tmp_path / "promptabi.lock.json"

    preview_exit = main(["fix", "--config", str(config), "--kind", "lockfile", "--format", "json"])
    preview = capsys.readouterr()
    preview_payload = json.loads(preview.out)
    assert preview_exit == 0
    assert preview_payload["applied"] is False
    assert preview_payload["changes"][0]["status"] == "planned"
    assert not lockfile.exists()

    write_exit = main(["fix", "--config", str(config), "--kind", "lockfile", "--write"])
    written = capsys.readouterr()
    assert write_exit == 0
    assert written.err == ""
    assert lockfile.is_file()
    payload = json.loads(lockfile.read_text(encoding="utf-8"))
    assert payload["config_name"] == "autofix-lock"
    assert payload["artifacts"][0]["sha256"] == sha256(schema.read_bytes()).hexdigest()


def test_autofix_writes_special_token_map_and_reverified_config(tmp_path: Path, capsys) -> None:
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps(
            {
                "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>{% endfor %}",
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "autofix-specials",
                "checks": ["repository-skeleton"],
                "artifacts": {"template": {"kind": "chat-template", "path": tokenizer_config.name}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["fix", "--config", str(config), "--kind", "special-tokens", "--write", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["changes"][0]["action"] == "write-special-token-map"
    special_map = tmp_path / "promptabi.special-tokens.json"
    assert special_map.is_file()
    special_payload = json.loads(special_map.read_text(encoding="utf-8"))
    assert {token["text"] for token in special_payload["tokens"]} == {"<|im_start|>", "<|im_end|>"}
    updated_config = json.loads(config.read_text(encoding="utf-8"))
    assert updated_config["artifacts"]["promptabi-special-tokens"]["kind"] == "special-token-map"
    assert main(["verify", "--config", str(config), "--fail-on", "never"]) == 0
    capsys.readouterr()


def test_autofix_writes_unsupported_annotations_and_docs_stub(tmp_path: Path, capsys) -> None:
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"chat_template": "{{ raise_exception('unsupported') }}", "additional_special_tokens": ["<x>"]}),
        encoding="utf-8",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "autofix-annotations",
                "checks": ["repository-skeleton"],
                "artifacts": {"template": {"kind": "chat-template", "path": tokenizer_config.name}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = low_risk_autofix(config, kinds=[AutoFixKind.UNSUPPORTED_FRAGMENTS, AutoFixKind.DOCS_STUB], write=True)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert report.applied is True
    assert (tmp_path / "promptabi.unsupported-fragments.json").is_file()
    annotations = json.loads((tmp_path / "promptabi.unsupported-fragments.json").read_text(encoding="utf-8"))
    assert annotations["annotations"][0]["fragments"][0]["source"] == "unsupported_constructs"
    notes = tmp_path / "promptabi-fix-notes.md"
    assert notes.is_file()
    assert "artifact-unpinned" in notes.read_text(encoding="utf-8")


def test_guarded_autofix_preview_reports_before_after_witnesses_for_high_risk_template_fix(capsys) -> None:
    config = Path("examples/role-boundary/unsafe.promptabi.json")

    exit_code = main(["fix", "--config", str(config), "--preview-risk", "high", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["applied"] is False
    assert payload["risk"] == "high"
    assert payload["preview_count"] >= 1
    preview = payload["previews"][0]
    assert preview["changes_user_visible_prompt_behavior"] is True
    assert preview["diagnostics"][0]["rule_id"] == "role-boundary-nonforgeability"
    assert preview["before_witnesses"][0]["rendered_strings"]
    assert preview["after_witnesses"][0]["minimal_fixes"]
    assert any("does not write files" in guardrail for guardrail in preview["guardrails"])


def test_guarded_autofix_preview_rejects_write_mode(capsys) -> None:
    config = Path("examples/role-boundary/unsafe.promptabi.json")

    exit_code = main(["fix", "--config", str(config), "--preview-risk", "high", "--write"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "cannot be combined with --write" in captured.err


def test_guarded_autofix_preview_has_successful_empty_preview_for_safe_config(capsys) -> None:
    config = Path("examples/role-boundary/safe.promptabi.json")

    exit_code = main(["fix", "--config", str(config), "--preview-risk", "high"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status: no high-risk fix previews found" in captured.out
    assert captured.err == ""


def test_guarded_autofix_preview_api_can_render_text() -> None:
    rendered = guarded_autofix_preview(
        Path("examples/role-boundary/unsafe.promptabi.json"),
        output_format="text",
    )

    assert "PromptABI guarded auto-fix preview (high risk)" in rendered
    assert "before witness:" in rendered
    assert "after witness:" in rendered
