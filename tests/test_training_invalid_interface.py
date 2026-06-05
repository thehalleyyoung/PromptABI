import json
from pathlib import Path

from promptabi.config import load_config
from promptabi.session import VerificationSession


def _write_config(tmp_path: Path, manifest_path: Path) -> Path:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "training-invalid-interface",
                "checks": ["training-invalid-interface"],
                "artifacts": {"train": {"kind": "training-manifest", "path": manifest_path.name}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def test_training_invalid_interface_verifies_clean_finite_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "clean-training-interface.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_format": "chat-jsonl",
                "datasets": [{"name": "sft", "kind": "supervised", "format": "chat-jsonl"}],
                "role_labels": [
                    {"source_role": "human", "canonical_role": "user", "trainable": False},
                    {"source_role": "assistant", "canonical_role": "assistant", "supervised_target": True},
                    {"source_role": "tool_result", "canonical_role": "tool", "trainable": False},
                ],
                "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["assistant"]},
                "supervised_spans": [
                    {
                        "span_id": "clean-1.assistant",
                        "target_role": "assistant",
                        "rendered_region_role": "assistant",
                        "start_token": 8,
                        "end_token": 16,
                        "region_start_token": 7,
                        "region_end_token": 18,
                    }
                ],
                "preference_pairs": [
                    {
                        "pair_id": "prefs-clean",
                        "prompt_sha256": "sha256:prompt",
                        "chosen_sha256": "sha256:chosen",
                        "rejected_sha256": "sha256:rejected",
                        "chosen_role_layout": ["system", "user", "assistant"],
                        "rejected_role_layout": ["system", "user", "assistant"],
                        "chosen_tokenizer": "tok@sha256:clean",
                        "rejected_tokenizer": "tok@sha256:clean",
                        "chosen_mask_policy": "dpo-response-only",
                        "rejected_mask_policy": "dpo-response-only",
                        "chosen_prompt_tokens": 12,
                        "rejected_prompt_tokens": 12,
                        "chosen_response_start_token": 12,
                        "rejected_response_start_token": 12,
                        "chosen_response_end_token": 22,
                        "rejected_response_end_token": 20,
                    }
                ],
                "metadata": {
                    "training_interface_contract": {
                        "tool_calls": [{"id": "tool-clean", "valid": True}],
                        "json_outputs": [{"id": "json-clean", "valid": True, "parses": True, "schema_valid": True}],
                        "stop_sequences": [{"sequence": "</tool_call>", "reachable": True, "matching_examples": 3}],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id.startswith("training-invalid-interface")
    ]

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in diagnostics] == ["training-invalid-interface-verified"]
    assert dict(diagnostics[0].properties)["kind"] == "verified"


def test_training_invalid_interface_catches_roles_tools_json_and_stops(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad-training-interface.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_format": "chat-jsonl",
                "datasets": [{"name": "sft", "kind": "supervised", "format": "chat-jsonl"}],
                "message_roles": ["user", "asistant"],
                "role_labels": [
                    {"source_role": "human", "canonical_role": "user", "trainable": False},
                    {"source_role": "bot", "canonical_role": "asistant", "supervised_target": True},
                ],
                "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["asistant"]},
                "supervised_spans": [
                    {
                        "span_id": "bad-1.asistant",
                        "target_role": "asistant",
                        "rendered_region_role": "asistant",
                        "start_token": 8,
                        "end_token": 16,
                        "region_start_token": 7,
                        "region_end_token": 18,
                    }
                ],
                "preference_pairs": [
                    {
                        "pair_id": "prefs-bad",
                        "prompt_sha256": "sha256:prompt",
                        "chosen_sha256": "sha256:chosen",
                        "rejected_sha256": "sha256:rejected",
                        "chosen_role_layout": ["user", "asistant"],
                        "rejected_role_layout": ["user", "assistant"],
                        "chosen_tokenizer": "tok@sha256:bad",
                        "rejected_tokenizer": "tok@sha256:bad",
                        "chosen_mask_policy": "dpo-response-only",
                        "rejected_mask_policy": "dpo-response-only",
                        "chosen_prompt_tokens": 12,
                        "rejected_prompt_tokens": 12,
                        "chosen_response_start_token": 12,
                        "rejected_response_start_token": 12,
                        "chosen_response_end_token": 22,
                        "rejected_response_end_token": 20,
                    }
                ],
                "metadata": {
                    "training_interface_contract": {
                        "allowed_roles": ["system", "user", "assistant", "tool"],
                        "tool_calls": [{"id": "tool-bad", "valid": False, "reason": "arguments are not JSON"}],
                        "json_outputs": [{"id": "json-bad", "parses": False, "parser_error": "unterminated object"}],
                        "stop_sequences": [{"sequence": "</tool_call>", "reachable": False, "matching_examples": 0}],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id.startswith("training-invalid-interface")
    ]
    kinds = {dict(diagnostic.properties)["kind"] for diagnostic in diagnostics}

    assert not result.ok
    assert {
        "impossible-role",
        "malformed-tool-call",
        "invalid-json-output",
        "unreachable-stop-sequence",
    }.issubset(kinds)
    assert {
        "training-invalid-interface-impossible-role",
        "training-invalid-interface-malformed-tool-call",
        "training-invalid-interface-invalid-json-output",
        "training-invalid-interface-unreachable-stop-sequence",
    } == {diagnostic.rule_id for diagnostic in diagnostics}
