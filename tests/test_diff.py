import json
from pathlib import Path

from promptabi.cli import main


def test_diff_cli_reports_contract_breaking_real_artifact_changes(tmp_path: Path, capsys) -> None:
    baseline_tok = tmp_path / "baseline_tok"
    current_tok = tmp_path / "current_tok"
    _write_tokenizer_revision(
        baseline_tok,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    _write_tokenizer_revision(
        current_tok,
        chat_template="<|start_header_id|>{{ messages[0].role }}<|end_header_id|>",
        eos_id=128009,
        stop_strings=["<|eot_id|>"],
        add_bos=True,
    )
    baseline_provider = tmp_path / "baseline-provider.json"
    current_provider = tmp_path / "current-provider.json"
    _write_provider_fixture(
        baseline_provider,
        provider="OpenAI",
        request_fields=["messages", "tools", "response_format"],
        response_fields=["choices", "tool_calls"],
        argument_encoding="json-object",
        max_input_tokens=128000,
    )
    _write_provider_fixture(
        current_provider,
        provider="vLLM OpenAI server",
        request_fields=["messages"],
        response_fields=["choices"],
        argument_encoding="json-string",
        max_input_tokens=8192,
    )
    baseline = tmp_path / "baseline.promptabi.json"
    current = tmp_path / "current.promptabi.json"
    _write_config(
        baseline,
        name="baseline-stack",
        checks=["tokenizer-config-drift", "provider-migration", "token-budget-model"],
        tokenizer_path="baseline_tok",
        provider_path="baseline-provider.json",
        max_context_tokens=128000,
        framework_max_context=128000,
        preserve_system=True,
    )
    _write_config(
        current,
        name="current-stack",
        checks=["tokenizer-config-drift", "token-budget-model"],
        tokenizer_path="current_tok",
        provider_path="current-provider.json",
        max_context_tokens=8192,
        framework_max_context=8192,
        preserve_system=False,
    )

    exit_code = main(["diff", str(baseline), str(current), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]]
    assert exit_code == 1
    assert payload["config"]["name"] == "baseline-stack -> current-stack"
    assert "diff-check-removed" in rule_ids
    assert "diff-context-regression" in rule_ids
    assert "diff-tokenizer-drift" in rule_ids
    assert "diff-provider-contract" in rule_ids
    assert "diff-framework-truncation" in rule_ids
    assert any(
        diagnostic["properties"].get("kind") == "chat-template-change"
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "diff-tokenizer-drift"
    )
    assert any(
        step["action"] == "compare provider contract field"
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "diff-provider-contract"
        for step in diagnostic["witness"]["steps"]
    )
    assert captured.err == ""


def test_diff_cli_reports_clean_configs_and_text_heading(tmp_path: Path, capsys) -> None:
    tok = tmp_path / "tok"
    provider = tmp_path / "provider.json"
    _write_tokenizer_revision(
        tok,
        chat_template="{{ messages[0].content }}",
        eos_id=2,
        stop_strings=["</tool_call>"],
        add_bos=False,
    )
    _write_provider_fixture(
        provider,
        provider="OpenAI",
        request_fields=["messages", "tools"],
        response_fields=["choices", "tool_calls"],
        argument_encoding="json-object",
        max_input_tokens=128000,
    )
    baseline = tmp_path / "baseline.promptabi.json"
    current = tmp_path / "current.promptabi.json"
    _write_config(
        baseline,
        name="same-baseline",
        checks=["tokenizer-config-drift", "provider-migration"],
        tokenizer_path="tok",
        provider_path="provider.json",
        max_context_tokens=128000,
        framework_max_context=128000,
        preserve_system=True,
    )
    _write_config(
        current,
        name="same-current",
        checks=["tokenizer-config-drift", "provider-migration"],
        tokenizer_path="tok",
        provider_path="provider.json",
        max_context_tokens=128000,
        framework_max_context=128000,
        preserve_system=True,
    )

    exit_code = main(["diff", str(baseline), str(current)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI diff: same-baseline -> same-current" in captured.out
    assert "status: PASS" in captured.out
    assert "INFO diff-clean" in captured.out
    assert captured.err == ""


def test_diff_cli_abstains_when_an_artifact_cannot_load(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.promptabi.json"
    current = tmp_path / "current.promptabi.json"
    baseline.write_text(
        json.dumps(
            {
                "name": "bad-baseline",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "missing.schema.json",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    current_schema = tmp_path / "schema.json"
    current_schema.write_text("{}", encoding="utf-8")
    current.write_text(
        json.dumps(
            {
                "name": "current",
                "checks": ["repository-skeleton"],
                "artifacts": {
                    "schema": {
                        "kind": "schema",
                        "path": "schema.json",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["diff", str(baseline), str(current), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(diagnostic["rule_id"] == "diff-abstained" for diagnostic in payload["diagnostics"])
    assert any(
        step["action"] == "preserve original loader finding"
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "diff-abstained"
        for step in diagnostic["witness"]["steps"]
    )
    assert captured.err == ""


def _write_config(
    path: Path,
    *,
    name: str,
    checks: list[str],
    tokenizer_path: str,
    provider_path: str,
    max_context_tokens: int,
    framework_max_context: int,
    preserve_system: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "name": name,
                "checks": checks,
                "max_context_tokens": max_context_tokens,
                "artifacts": {
                    "tok": {
                        "kind": "tokenizer",
                        "path": tokenizer_path,
                        "version": name,
                    },
                    "provider": {
                        "kind": "provider-config",
                        "path": provider_path,
                        "provider": name,
                    },
                    "framework": {
                        "kind": "framework-truncation-config",
                        "path": provider_path,
                        "framework": "vllm",
                        "strategy": "left",
                        "max_context_tokens": framework_max_context,
                        "preserve_system": preserve_system,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_tokenizer_revision(
    root: Path,
    *,
    chat_template: str,
    eos_id: int,
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
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "normalizer": {"type": "NFC"},
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"stop_strings": stop_strings, "eos_token_id": eos_id}, sort_keys=True),
        encoding="utf-8",
    )


def _write_provider_fixture(
    path: Path,
    *,
    provider: str,
    request_fields: list[str],
    response_fields: list[str],
    argument_encoding: str,
    max_input_tokens: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "provider": provider,
                "request_shape": {"required_fields": request_fields},
                "response_shape": {"required_fields": response_fields},
                "migration_compatibility": {
                    "provider_family": provider,
                    "request": {"required_fields": request_fields},
                    "response": {"required_fields": response_fields},
                    "tools": {
                        "argument_encoding": argument_encoding,
                        "id_path": "choices[].message.tool_calls[].id",
                        "supports_parallel_tool_calls": True,
                    },
                    "streaming": {"emits_argument_fragments": True},
                    "stops": {"sequences": ["</tool_call>"]},
                    "limits": {
                        "max_input_tokens": max_input_tokens,
                        "max_output_tokens": 4096,
                    },
                    "structured_outputs": {"modes": ["json_schema"]},
                    "errors": {"code_path": "error.code", "rate_limit_path": "error.type"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
