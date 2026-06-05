import json
from pathlib import Path

from promptabi.artifacts import ArtifactKind, artifact_from_config
from promptabi.loaders import ArtifactLoader
from promptabi.session import VerificationSession


def _write_prompt_pack(path: Path, *, expected_roles=None, tool_name="refund_user", stop="</tool_call>") -> None:
    path.write_text(
        json.dumps(
            {
                "name": "support-pack",
                "version": "1.0.0",
                "exported_templates": [
                    {
                        "name": "support-chat",
                        "template": "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
                        "roles": ["system", "user", "assistant"],
                        "variables": ["messages"],
                        "required_regions": ["system-policy"],
                        "supported_model_families": ["openai-compatible"],
                    }
                ],
                "expected_roles": expected_roles or ["system", "user", "assistant"],
                "tool_schemas": [{"name": tool_name, "provider": "openai"}],
                "stop_policies": [{"name": "tool-json", "stop_sequences": [stop]}],
                "supported_model_families": ["openai-compatible"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_prompt_pack_loader_materializes_reusable_contracts(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    artifact = artifact_from_config(
        "support",
        {"kind": "prompt-pack", "path": pack.name, "version": "1.0.0"},
        base_dir=tmp_path,
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "prompt-pack"
    assert loaded.pinned is True
    assert dict(loaded.metadata) == {
        "stop_policy_count": 1,
        "template_count": 1,
        "tool_schema_count": 1,
    }
    assert loaded.artifact.kind is ArtifactKind.PROMPT_PACK
    assert loaded.artifact.exported_templates[0].roles == ("assistant", "system", "user")
    assert {"expected_roles", "tool_schemas.refund_user", "stop_policies.tool-json"}.issubset(
        {name for name, _span in loaded.source_spans}
    )


def test_prompt_pack_contracts_verify_against_downstream_artifacts(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-safe",
                "checks": ["prompt-pack-contracts"],
                "artifacts": {
                    "support": {"kind": "prompt-pack", "path": pack.name, "version": "1.0.0"},
                    "messages": {
                        "kind": "prompt-segment",
                        "uri": "memory://messages",
                        "segments": [
                            {"name": "system-policy", "role": "system", "required": True},
                            {"name": "user-request", "role": "user", "required": True},
                            {"name": "assistant-answer", "role": "assistant"},
                        ],
                    },
                    "tools": {
                        "kind": "tool-definition",
                        "uri": "memory://tools",
                        "provider": "openai",
                        "tool_names": ["refund_user"],
                    },
                    "stops": {
                        "kind": "stop-policy",
                        "uri": "memory://stops",
                        "stop_sequences": ["</tool_call>"],
                    },
                    "provider": {
                        "kind": "provider-config",
                        "uri": "memory://provider",
                        "provider": "openai-compatible",
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config).run()

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["prompt-pack-verified"]
    assert "compatible with configured app artifacts" in result.diagnostics[0].message


def test_prompt_pack_contracts_report_real_app_mismatches(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-unsafe",
                "checks": ["prompt-pack-contracts"],
                "artifacts": {
                    "support": {"kind": "prompt-pack", "path": pack.name, "version": "1.0.0"},
                    "messages": {
                        "kind": "prompt-segment",
                        "uri": "memory://messages",
                        "segments": [{"name": "user-request", "role": "user", "required": True}],
                    },
                    "tools": {
                        "kind": "tool-definition",
                        "uri": "memory://tools",
                        "provider": "openai",
                        "tool_names": ["lookup_order"],
                    },
                    "stops": {
                        "kind": "stop-policy",
                        "uri": "memory://stops",
                        "stop_sequences": ["<END>"],
                    },
                    "provider": {
                        "kind": "provider-config",
                        "uri": "memory://provider",
                        "provider": "anthropic",
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config).run()

    assert not result.ok
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == [
        "prompt-pack-app-role-missing",
        "prompt-pack-model-family-unsupported",
        "prompt-pack-stop-missing",
        "prompt-pack-tool-missing",
    ]
    assert "assistant, system" in result.diagnostics[0].message
    assert all(diagnostic.witness is not None for diagnostic in result.diagnostics)
