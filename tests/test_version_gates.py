import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.version_gates import (
    SemverImpact,
    load_version_gate_policy,
    render_version_gate_json,
    render_version_gate_text,
    run_version_gate,
    version_gate_policy_from_mapping,
)


def test_version_gate_classifies_tokenizer_kinds_by_semver_impact(tmp_path: Path) -> None:
    baseline, current = _write_tokenizer_pair(
        tmp_path,
        baseline_template="{{ messages[0].content }}",
        current_template="<|start_header_id|>{{ messages[0].role }}<|end_header_id|>",
        baseline_eos=2,
        current_eos=2,
    )

    patch_report = run_version_gate(baseline, current, allowed_impact="patch-safe")
    minor_report = run_version_gate(baseline, current, allowed_impact="minor-breaking")

    assert patch_report.ok is False
    assert minor_report.ok is True
    assert patch_report.max_required_impact is SemverImpact.MINOR_BREAKING
    assert {
        finding.diagnostic.to_dict()["properties"]["kind"]: finding.required_impact
        for finding in patch_report.findings
        if finding.diagnostic.rule_id == "diff-tokenizer-drift"
    } == {"chat-template-change": SemverImpact.MINOR_BREAKING}


def test_version_gate_treats_special_token_drift_and_abstention_as_major(tmp_path: Path) -> None:
    baseline, current = _write_tokenizer_pair(
        tmp_path,
        baseline_template="{{ messages[0].content }}",
        current_template="{{ messages[0].content }}",
        baseline_eos=2,
        current_eos=128009,
    )
    major_report = run_version_gate(baseline, current, allowed_impact="minor-breaking")

    assert major_report.ok is False
    assert any(
        finding.required_impact is SemverImpact.MAJOR_BREAKING
        and finding.diagnostic.to_dict().get("properties", {}).get("kind") == "special-token-id-change"
        for finding in major_report.findings
    )

    remote_baseline = tmp_path / "remote-baseline.promptabi.json"
    remote_current = tmp_path / "remote-current.promptabi.json"
    _write_remote_tokenizer_config(remote_baseline, "hf://org/model-a")
    _write_remote_tokenizer_config(remote_current, "hf://org/model-b")

    abstained = run_version_gate(remote_baseline, remote_current, allowed_impact="minor-breaking")

    assert abstained.ok is False
    assert any(
        finding.required_impact is SemverImpact.MAJOR_BREAKING
        and finding.diagnostic.rule_id == "diff-tokenizer-abstained"
        for finding in abstained.findings
    )


def test_version_gate_policy_overrides_are_specific_and_stable(tmp_path: Path) -> None:
    baseline, current = _write_tokenizer_pair(
        tmp_path,
        baseline_template="{{ messages[0].content }}",
        current_template="<|start_header_id|>{{ messages[0].role }}<|end_header_id|>",
        baseline_eos=2,
        current_eos=2,
    )
    policy_path = tmp_path / "version-gate.policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "version_gate_policy_version": 1,
                "default_unknown_impact": "minor-breaking",
                "rules": [
                    {
                        "match": {"rule_id": "diff-tokenizer-drift", "kind": "chat-template-change"},
                        "impact": "patch-safe",
                        "rationale": "this deployment keeps template delimiters backward-compatible",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    policy = load_version_gate_policy(policy_path)
    report = run_version_gate(baseline, current, allowed_impact="patch-safe", policy=policy)
    payload = json.loads(render_version_gate_json(report))
    text = render_version_gate_text(report)
    api_payload = json.loads(
        promptabi.semantic_version_gate(
            baseline,
            current,
            allowed_impact="patch-safe",
            policy_path=policy_path,
            output_format="json",
        )
    )

    assert report.ok is True
    assert payload == api_payload
    assert payload["findings"][0]["source"] == "policy"
    assert "this deployment keeps template delimiters" in text
    assert render_version_gate_json(report) == render_version_gate_json(report)


def test_version_gate_cli_writes_json_and_uses_exit_codes(tmp_path: Path, capsys) -> None:
    baseline, current = _write_tokenizer_pair(
        tmp_path,
        baseline_template="{{ messages[0].content }}",
        current_template="<|start_header_id|>{{ messages[0].role }}<|end_header_id|>",
        baseline_eos=2,
        current_eos=2,
    )
    output = tmp_path / "gate.json"

    exit_code = main(
        [
            "version-gate",
            str(baseline),
            str(current),
            "--allowed-impact",
            "patch-safe",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert "wrote version-gate report" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["max_required_impact"] == "minor-breaking"

    exit_code = main(["version-gate", str(baseline), str(current), "--allowed-impact", "minor-breaking"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PromptABI semantic version gate" in captured.out


def test_version_gate_policy_validation_rejects_unsound_match_keys() -> None:
    try:
        version_gate_policy_from_mapping(
            {
                "rules": [
                    {
                        "match": {"rule_id": "diff-tokenizer-drift", "unsupported_selector": "x"},
                        "impact": "patch-safe",
                    }
                ]
            }
        )
    except ValueError as exc:
        assert "unknown match keys" in str(exc)
    else:
        raise AssertionError("invalid version-gate policy was accepted")


def _write_tokenizer_pair(
    root: Path,
    *,
    baseline_template: str,
    current_template: str,
    baseline_eos: int,
    current_eos: int,
) -> tuple[Path, Path]:
    baseline_tok = root / "baseline_tok"
    current_tok = root / "current_tok"
    _write_tokenizer_revision(baseline_tok, chat_template=baseline_template, eos_id=baseline_eos)
    _write_tokenizer_revision(current_tok, chat_template=current_template, eos_id=current_eos)
    baseline = root / "baseline.promptabi.json"
    current = root / "current.promptabi.json"
    _write_config(baseline, name="baseline-stack", tokenizer_path="baseline_tok")
    _write_config(current, name="current-stack", tokenizer_path="current_tok")
    return baseline, current


def _write_config(path: Path, *, name: str, tokenizer_path: str) -> None:
    provider_path = path.parent / "provider.json"
    if not provider_path.exists():
        provider_path.write_text(
            json.dumps(
                {
                    "provider": "OpenAI",
                    "request_shape": {"required_fields": ["messages", "tools"]},
                    "response_shape": {"required_fields": ["choices", "tool_calls"]},
                    "migration_compatibility": {
                        "provider_family": "openai",
                        "request": {"required_fields": ["messages", "tools"]},
                        "response": {"required_fields": ["choices", "tool_calls"]},
                        "tools": {
                            "argument_encoding": "json-object",
                            "id_path": "choices[].message.tool_calls[].id",
                            "supports_parallel_tool_calls": True,
                        },
                        "streaming": {"emits_argument_fragments": True},
                        "stops": {"sequences": ["</tool_call>"]},
                        "limits": {"max_input_tokens": 128000, "max_output_tokens": 4096},
                        "structured_outputs": {"modes": ["json_schema"]},
                        "errors": {"code_path": "error.code", "rate_limit_path": "error.type"},
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    path.write_text(
        json.dumps(
            {
                "name": name,
                "checks": ["tokenizer-config-drift", "provider-migration", "token-budget-model"],
                "max_context_tokens": 128000,
                "artifacts": {
                    "tok": {"kind": "tokenizer", "path": tokenizer_path, "version": name},
                    "provider": {"kind": "provider-config", "path": "provider.json", "provider": "OpenAI"},
                    "framework": {
                        "kind": "framework-truncation-config",
                        "path": "provider.json",
                        "framework": "vllm",
                        "strategy": "left",
                        "max_context_tokens": 128000,
                        "preserve_system": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_tokenizer_revision(root: Path, *, chat_template: str, eos_id: int) -> None:
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "eos_token_id": eos_id,
                "add_bos_token": False,
                "chat_template": chat_template,
                "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {"normalizer": {"type": "NFC"}, "added_tokens": [{"id": eos_id, "content": "</s>", "special": True}]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"stop_strings": ["</tool_call>"], "eos_token_id": eos_id}, sort_keys=True),
        encoding="utf-8",
    )


def _write_remote_tokenizer_config(path: Path, uri: str) -> None:
    path.write_text(
        json.dumps(
            {
                "name": path.stem,
                "checks": ["tokenizer-config-drift"],
                "artifacts": {"tok": {"kind": "tokenizer", "uri": uri}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
