import json
from pathlib import Path

from promptabi.artifacts import ArtifactKind, artifact_from_config
from promptabi.cli import main
from promptabi.loaders import ArtifactLoader
from promptabi.prompt_packs import (
    build_prompt_pack_mirror,
    create_signed_prompt_pack_provenance,
    build_prompt_pack_lockfile,
    build_prompt_pack_registry,
    certify_prompt_pack_monotonicity,
    compare_prompt_pack_lockfile,
    compare_prompt_pack_upgrade,
    load_signed_prompt_pack_provenance,
    load_prompt_pack_mirror_manifest,
    verify_signed_prompt_pack_provenance,
    verify_prompt_pack_mirror,
    prompt_pack_registry_to_json,
    render_prompt_pack_monotonicity_json,
)
from promptabi.session import VerificationSession


def _write_prompt_pack(
    path: Path,
    *,
    expected_roles=None,
    tool_name="refund_user",
    stop="</tool_call>",
    template="support-chat",
    template_source="{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
    template_roles=None,
    supported_model_families=None,
    schema_digest=None,
    include_extra_template=False,
    extra_template_roles=None,
    extra_template_required_regions=None,
    extra_tool_required=False,
    extra_stop_sequence=None,
) -> None:
    template_entries = [
        {
            "name": template,
            "template": template_source,
            "roles": template_roles or ["system", "user", "assistant"],
            "variables": ["messages"],
            "required_regions": ["system-policy"],
            "supported_model_families": supported_model_families or ["openai-compatible"],
        }
    ]
    if include_extra_template:
        template_entries.append(
            {
                "name": "handoff-chat",
                "template": "{{ system }}\n{{ transcript }}",
                "roles": extra_template_roles or ["system", "user", "assistant", "tool"],
                "variables": ["system", "transcript"],
                "required_regions": extra_template_required_regions or ["system-policy", "tool-result"],
                "supported_model_families": ["openai-compatible"],
            }
        )
    tool_schema = {"name": tool_name, "provider": "openai"}
    if schema_digest is not None:
        tool_schema["schema_digest"] = schema_digest
    tool_schemas = [tool_schema]
    if extra_tool_required:
        tool_schemas.append({"name": "lookup_order", "provider": "openai", "required": True})
    stop_policies = [{"name": "tool-json", "stop_sequences": [stop]}]
    if extra_stop_sequence is not None:
        stop_policies.append({"name": "handoff-json", "stop_sequences": [extra_stop_sequence]})
    path.write_text(
        json.dumps(
            {
                "name": "support-pack",
                "version": "1.0.0",
                "exported_templates": template_entries,
                "expected_roles": expected_roles or ["system", "user", "assistant"],
                "tool_schemas": tool_schemas,
                "stop_policies": stop_policies,
                "supported_model_families": supported_model_families or ["openai-compatible"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_safe_config(path: Path, pack: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "prompt-pack-upgrade",
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
                            {"name": "tool-result", "role": "tool"},
                        ],
                    },
                    "tools": {
                        "kind": "tool-definition",
                        "uri": "memory://tools",
                        "provider": "openai",
                        "tool_names": ["refund_user", "lookup_order"],
                    },
                    "stops": {
                        "kind": "stop-policy",
                        "uri": "memory://stops",
                        "stop_sequences": ["</tool_call>", "</handoff>"],
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


def _prompt_pack_lock_for_config(config: Path, base_dir: Path):
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _ = session.load_artifacts_with_diagnostics()
    return build_prompt_pack_lockfile(loaded, result.diagnostics, base_dir=base_dir), result, loaded


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


def test_prompt_pack_composition_reuses_guarantees_with_bounded_rag_and_truncation(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-composed-safe",
                "checks": ["prompt-pack-contracts"],
                "artifacts": {
                    "support": {"kind": "prompt-pack", "path": pack.name, "version": "1.0.0"},
                    "messages": {
                        "kind": "prompt-segment",
                        "uri": "memory://messages",
                        "segments": [
                            {"name": "system-policy", "role": "system", "required": True, "token_count": 10},
                            {"name": "retrieval-a", "role": "retrieval", "token_count": 8, "max_tokens": 16, "chunk_id": "doc:a"},
                            {"name": "user-request", "role": "user", "required": True, "token_count": 7},
                            {"name": "assistant-answer", "role": "assistant", "token_count": 4},
                        ],
                    },
                    "budget": {
                        "kind": "framework-truncation-config",
                        "uri": "memory://budget",
                        "framework": "custom-rag-pipeline",
                        "strategy": "priority",
                        "max_context_tokens": 64,
                        "drop_roles": ["retrieval"],
                        "preserve_system": True,
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
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["prompt-pack-composition-verified"]
    assert "context, RAG, and truncation" in result.diagnostics[0].message
    assert dict(result.diagnostics[0].properties)["finding_kind"] == "composition-verified"


def test_prompt_pack_composition_reports_rag_and_truncation_regressions(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-composed-unsafe",
                "checks": ["prompt-pack-contracts"],
                "artifacts": {
                    "support": {"kind": "prompt-pack", "path": pack.name, "version": "1.0.0"},
                    "messages": {
                        "kind": "prompt-segment",
                        "uri": "memory://messages",
                        "segments": [
                            {"name": "system-policy", "role": "system", "required": True, "token_count": 30},
                            {"name": "user-request", "role": "user", "required": True, "token_count": 20},
                            {"name": "retrieval-a", "role": "retrieval", "token_count": 25, "chunk_id": "doc:a"},
                            {"name": "assistant-answer", "role": "assistant", "token_count": 4},
                        ],
                    },
                    "budget": {
                        "kind": "framework-truncation-config",
                        "uri": "memory://budget",
                        "framework": "vllm",
                        "max_context_tokens": 45,
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

    assert not result.ok
    rule_ids = {diagnostic.rule_id for diagnostic in result.diagnostics}
    assert "prompt-pack-composition-rag-unbounded" in rule_ids
    assert "prompt-pack-composition-required-region-truncated" in rule_ids
    truncated = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id == "prompt-pack-composition-required-region-truncated"
    )
    assert "system-policy" in truncated.message
    assert truncated.witness is not None


def test_prompt_pack_composes_template_and_tokenizer_proofs(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(
        pack,
        template_source="{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-template-tokenizer-safe",
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
                    "tokenizer": {
                        "kind": "tokenizer",
                        "uri": "memory://tokenizer",
                        "family": "byte-level",
                        "added_tokens": ["<|im_start|>", "<|im_end|>"],
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
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["prompt-pack-template-tokenizer-verified"]
    diagnostic = result.diagnostics[0]
    assert "chat-template/tokenizer proof" in diagnostic.message
    assert dict(diagnostic.properties)["finding_kind"] == "template-tokenizer-verified"
    assert diagnostic.witness is not None
    assert "support-chat" in diagnostic.witness.steps[0].output
    assert diagnostic.witness.token_ids


def test_prompt_pack_template_tokenizer_composition_reports_control_token_collision(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(
        pack,
        template_source="{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-template-tokenizer-unsafe",
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
                    "tokenizer": {
                        "kind": "tokenizer",
                        "uri": "memory://tokenizer",
                        "family": "byte-level",
                        "added_tokens": ["<|im_start|>", "<|im_end|>", "<|system|>"],
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
        "prompt-pack-template-tokenizer-control-token"
    ]
    diagnostic = result.diagnostics[0]
    assert "can inject tokenizer control token '<|system|>'" in diagnostic.message
    assert dict(diagnostic.properties)["subject"] == "support-chat:user content"
    assert diagnostic.witness is not None
    assert diagnostic.witness.rendered_strings == ("<|system|>",)
    assert diagnostic.witness.token_ids == (258,)


def test_prompt_pack_lockfile_pins_package_contracts_and_diagnostics(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-lock",
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
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _ = session.load_artifacts_with_diagnostics()

    lockfile = build_prompt_pack_lockfile(loaded, result.diagnostics, base_dir=tmp_path)

    assert len(lockfile.entries) == 1
    entry = lockfile.entries[0]
    assert entry.package_name == "support-pack"
    assert entry.version == "1.0.0"
    assert entry.location == "support.prompt-pack.json"
    assert entry.contract_hash
    assert entry.diagnostic_baseline[0][0] == "prompt-pack-verified"

    _write_prompt_pack(pack, expected_roles=["system", "user", "assistant", "tool"])
    changed_session = VerificationSession.from_config_file(config)
    changed_result = changed_session.run()
    changed_loaded, _ = changed_session.load_artifacts_with_diagnostics()
    drift = compare_prompt_pack_lockfile(lockfile, changed_loaded, changed_result.diagnostics, lockfile_path=tmp_path / "prompt-pack.lock.json")

    assert any(diagnostic.rule_id == "prompt-pack-lock-drift" for diagnostic in drift)
    assert any(dict(diagnostic.properties)["field"] == "contract_hash" for diagnostic in drift)


def test_cli_prompt_pack_lock_writes_and_checks_real_drift(tmp_path: Path, capsys) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    _write_prompt_pack(pack)
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "prompt-pack-cli-lock",
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
    lockfile = tmp_path / "prompt-pack.lock.json"

    assert main(["prompt-pack", "lock", "--config", str(config), "--lockfile", str(lockfile), "--write", "--format", "json"]) == 0
    capsys.readouterr()
    assert lockfile.is_file()

    assert main(["prompt-pack", "lock", "--config", str(config), "--lockfile", str(lockfile), "--check", "--format", "json"]) == 0
    checked = capsys.readouterr()
    assert "prompt-pack-lock-verified" in checked.out

    _write_prompt_pack(pack, tool_name="lookup_order")
    exit_code = main(["prompt-pack", "lock", "--config", str(config), "--lockfile", str(lockfile), "--check", "--format", "json"])
    drift = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(drift.out)
    assert any(diagnostic["rule_id"] == "prompt-pack-lock-drift" for diagnostic in payload["diagnostics"])


def test_prompt_pack_upgrade_allows_additive_compatible_changes(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    baseline, _baseline_result, _baseline_loaded = _prompt_pack_lock_for_config(config, tmp_path)

    _write_prompt_pack(
        pack,
        expected_roles=["system", "user", "assistant", "tool"],
        schema_digest="sha256:refund-v1",
        supported_model_families=["openai-compatible", "vllm"],
        include_extra_template=True,
    )
    current_lock, current_result, current_loaded = _prompt_pack_lock_for_config(config, tmp_path)

    assert current_lock.entries[0].contract_hash != baseline.entries[0].contract_hash
    upgrade = compare_prompt_pack_upgrade(baseline, current_loaded, current_result.diagnostics, baseline_path=tmp_path / "baseline.lock.json")

    assert [diagnostic.rule_id for diagnostic in upgrade] == ["prompt-pack-upgrade-compatible"]
    assert upgrade[0].severity.value == "info"


def test_prompt_pack_upgrade_rejects_role_stop_schema_and_budget_regressions(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    baseline, _baseline_result, _baseline_loaded = _prompt_pack_lock_for_config(config, tmp_path)

    _write_prompt_pack(
        pack,
        expected_roles=["system", "user"],
        tool_name="refund_user",
        stop="</different>",
        template_source="{{ message.content }}",
        template_roles=["system", "user"],
        supported_model_families=["anthropic"],
        schema_digest="sha256:refund-v2",
    )
    changed_session = VerificationSession.from_config_file(config)
    changed_result = changed_session.run()
    changed_loaded, _ = changed_session.load_artifacts_with_diagnostics()

    upgrade = compare_prompt_pack_upgrade(baseline, changed_loaded, changed_result.diagnostics, baseline_path=tmp_path / "baseline.lock.json")
    rule_ids = {diagnostic.rule_id for diagnostic in upgrade}

    assert "prompt-pack-upgrade-role-regression" in rule_ids
    assert "prompt-pack-upgrade-model-family-regression" in rule_ids
    assert "prompt-pack-upgrade-template-regression" in rule_ids
    assert "prompt-pack-upgrade-tool-schema-regression" in rule_ids
    assert "prompt-pack-upgrade-stop-regression" in rule_ids
    assert "prompt-pack-upgrade-diagnostic-regression" in rule_ids
    assert all(diagnostic.witness is not None for diagnostic in upgrade)
    assert any(dict(diagnostic.properties)["field"] == "stop_policies.tool-json" for diagnostic in upgrade)


def test_prompt_pack_monotonicity_certifies_append_only_extensions(tmp_path: Path, capsys) -> None:
    baseline_pack = tmp_path / "baseline.prompt-pack.json"
    candidate_pack = tmp_path / "candidate.prompt-pack.json"
    _write_prompt_pack(baseline_pack)
    _write_prompt_pack(
        candidate_pack,
        include_extra_template=True,
        extra_template_roles=["system", "user", "assistant"],
        extra_template_required_regions=["system-policy"],
        supported_model_families=["openai-compatible", "vllm-openai-compatible"],
    )
    baseline_config = tmp_path / "baseline.promptabi.json"
    candidate_config = tmp_path / "candidate.promptabi.json"
    _write_safe_config(baseline_config, baseline_pack)
    _write_safe_config(candidate_config, candidate_pack)
    baseline_lock, _baseline_result, _baseline_loaded = _prompt_pack_lock_for_config(baseline_config, tmp_path)
    candidate_session = VerificationSession.from_config_file(candidate_config)
    candidate_result = candidate_session.run()
    candidate_loaded, _load_diagnostics = candidate_session.load_artifacts_with_diagnostics()

    certificate = certify_prompt_pack_monotonicity(baseline_lock, candidate_loaded, candidate_result.diagnostics)
    payload = json.loads(render_prompt_pack_monotonicity_json(certificate))

    assert certificate.ok
    assert payload["ok"] is True
    assert [diagnostic.rule_id for diagnostic in certificate.diagnostics] == [
        "prompt-pack-monotonicity-certified"
    ]

    lock_path = tmp_path / "prompt-pack.lock.json"
    lock_path.write_text(json.dumps(baseline_lock.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    exit_code = main(
        [
            "prompt-pack",
            "monotonicity",
            "--config",
            str(candidate_config),
            "--baseline-lockfile",
            str(lock_path),
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["diagnostics"][0]["rule_id"] == "prompt-pack-monotonicity-certified"


def test_prompt_pack_monotonicity_rejects_new_required_obligations(tmp_path: Path) -> None:
    baseline_pack = tmp_path / "baseline.prompt-pack.json"
    candidate_pack = tmp_path / "candidate.prompt-pack.json"
    _write_prompt_pack(baseline_pack)
    _write_prompt_pack(
        candidate_pack,
        include_extra_template=True,
        extra_tool_required=True,
        extra_stop_sequence="</handoff>",
    )
    baseline_config = tmp_path / "baseline.promptabi.json"
    candidate_config = tmp_path / "candidate.promptabi.json"
    _write_safe_config(baseline_config, baseline_pack)
    _write_safe_config(candidate_config, candidate_pack)
    baseline_lock, _baseline_result, _baseline_loaded = _prompt_pack_lock_for_config(baseline_config, tmp_path)
    candidate_session = VerificationSession.from_config_file(candidate_config)
    candidate_result = candidate_session.run()
    candidate_loaded, _load_diagnostics = candidate_session.load_artifacts_with_diagnostics()

    certificate = certify_prompt_pack_monotonicity(baseline_lock, candidate_loaded, candidate_result.diagnostics)

    assert not certificate.ok
    assert {
        "prompt-pack-monotonicity-required-tool-added",
        "prompt-pack-monotonicity-stop-obligation-added",
        "prompt-pack-monotonicity-template-obligation-added",
    }.issubset({diagnostic.rule_id for diagnostic in certificate.diagnostics})


def test_cli_prompt_pack_upgrade_gates_real_candidate(tmp_path: Path, capsys) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    baseline_lock = tmp_path / "baseline.prompt-pack.lock.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)

    assert main(["prompt-pack", "lock", "--config", str(config), "--lockfile", str(baseline_lock), "--write", "--format", "json"]) == 0
    capsys.readouterr()

    _write_prompt_pack(
        pack,
        expected_roles=["system", "user", "assistant", "tool"],
        schema_digest="sha256:refund-v1",
        include_extra_template=True,
    )
    assert main(["prompt-pack", "upgrade", "--config", str(config), "--baseline-lockfile", str(baseline_lock), "--format", "json"]) == 0
    compatible = json.loads(capsys.readouterr().out)
    assert any(diagnostic["rule_id"] == "prompt-pack-upgrade-compatible" for diagnostic in compatible["diagnostics"])

    _write_prompt_pack(pack, stop="</different>", schema_digest="sha256:refund-v2")
    exit_code = main(["prompt-pack", "upgrade", "--config", str(config), "--baseline-lockfile", str(baseline_lock), "--format", "json"])
    regressed = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert any(diagnostic["rule_id"] == "prompt-pack-upgrade-stop-regression" for diagnostic in regressed["diagnostics"])
    assert any(diagnostic["rule_id"] == "prompt-pack-upgrade-tool-schema-regression" for diagnostic in regressed["diagnostics"])


def test_prompt_pack_registry_publishes_hashes_without_private_contract_contents(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()

    registry = build_prompt_pack_registry(loaded, result.diagnostics, base_dir=tmp_path)
    payload = registry.to_dict()
    encoded = prompt_pack_registry_to_json(registry)

    assert payload["registry_version"] == 1
    entry = payload["prompt_packs"][0]
    assert entry["package_name"] == "support-pack"
    assert entry["supported_fragments"] == {
        "diagnostic_count": 1,
        "model_family_count": 1,
        "role_count": 3,
        "stop_policy_count": 1,
        "template_count": 1,
        "tool_schema_count": 1,
    }
    assert len(entry["contract_hash"]) == 64
    assert len(entry["proof_hash"]) == 64
    assert entry["diagnostics"][0]["rule_id"] == "prompt-pack-verified"
    assert "template_hash" in entry["proofs"]["template_proofs"][0]
    assert "tool_name_hash" in entry["proofs"]["tool_schema_proofs"][0]
    assert "stop_sequence_set_hash" in entry["proofs"]["stop_policy_proofs"][0]
    assert "refund_user" not in encoded
    assert "</tool_call>" not in encoded
    assert "{% for message in messages %}" not in encoded


def test_cli_prompt_pack_registry_writes_public_manifest(tmp_path: Path, capsys) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    output = tmp_path / "registry.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)

    assert main(["prompt-pack", "registry", "--config", str(config), "--output", str(output), "--format", "text"]) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert "PromptABI prompt-pack registry" in stdout
    assert payload["registry_version"] == 1
    assert payload["prompt_packs"][0]["package_name"] == "support-pack"
    assert "refund_user" not in output.read_text(encoding="utf-8")
    assert "</tool_call>" not in output.read_text(encoding="utf-8")


def test_signed_prompt_pack_provenance_trusts_registry_metadata_without_raw_prompts(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()

    provenance = create_signed_prompt_pack_provenance(
        loaded,
        result.diagnostics,
        key="prompt-pack-secret",
        key_id="registry-review",
        base_dir=tmp_path,
    )
    verification = verify_signed_prompt_pack_provenance(provenance, key="prompt-pack-secret")
    encoded = json.dumps(provenance.to_dict(), sort_keys=True)

    assert verification.ok
    assert verification.signing_key_id == "registry-review"
    assert verification.package_count == 1
    assert provenance.payload["package_count"] == 1
    assert provenance.payload["prompt_packs"][0]["package_name"] == "support-pack"
    assert provenance.to_dict()["provenance_hash"] == provenance.provenance_hash
    assert "refund_user" not in encoded
    assert "</tool_call>" not in encoded
    assert "{% for message in messages %}" not in encoded


def test_signed_prompt_pack_provenance_rejects_tampering(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()
    provenance = create_signed_prompt_pack_provenance(loaded, result.diagnostics, key="prompt-pack-secret")
    tampered = provenance.to_dict()
    tampered["payload"]["prompt_packs"][0]["contract_hash"] = "0" * 64

    verification = verify_signed_prompt_pack_provenance(tampered, key="prompt-pack-secret")

    assert verification.ok is False
    assert verification.reason == "signature mismatch"


def test_cli_prompt_pack_provenance_create_and_verify(tmp_path: Path, capsys) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    output = tmp_path / "prompt-pack.provenance.json"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)

    assert main(
        [
            "prompt-pack",
            "provenance",
            "create",
            "--config",
            str(config),
            "--output",
            str(output),
            "--key",
            "prompt-pack-secret",
            "--key-id",
            "cli-review",
        ]
    ) == 0
    created = capsys.readouterr()
    assert "wrote signed prompt-pack provenance" in created.out
    assert load_signed_prompt_pack_provenance(output).signing_key_id == "cli-review"

    assert main(["prompt-pack", "provenance", "verify", str(output), "--key", "prompt-pack-secret", "--format", "json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["package_count"] == 1
    assert verified["signing_key_id"] == "cli-review"

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["payload"]["prompt_packs"][0]["proof_hash"] = "1" * 64
    output.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")
    assert main(["prompt-pack", "provenance", "verify", str(output), "--key", "prompt-pack-secret", "--format", "json"]) == 1
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["ok"] is False
    assert rejected["reason"] == "signature mismatch"


def test_prompt_pack_mirror_copies_local_packs_and_verifies_offline(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    mirror_dir = tmp_path / "mirror"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()

    mirror = build_prompt_pack_mirror(loaded, result.diagnostics, mirror_dir=mirror_dir, base_dir=tmp_path)
    manifest = load_prompt_pack_mirror_manifest(mirror_dir / "prompt-pack-mirror.json")
    verification = verify_prompt_pack_mirror(mirror_dir / "prompt-pack-mirror.json")

    assert mirror.entries[0].package_name == "support-pack"
    assert manifest.entries[0].mirror_path.startswith("packs/")
    assert (mirror_dir / manifest.entries[0].mirror_path).read_text(encoding="utf-8") == pack.read_text(encoding="utf-8")
    assert verification.ok
    assert verification.diagnostics[0].rule_id == "prompt-pack-mirror-verified"
    assert "refund_user" not in json.dumps(manifest.to_dict(), sort_keys=True)
    assert "</tool_call>" not in json.dumps(manifest.to_dict(), sort_keys=True)


def test_prompt_pack_mirror_detects_tampered_local_pack(tmp_path: Path) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    mirror_dir = tmp_path / "mirror"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)
    session = VerificationSession.from_config_file(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()
    mirror = build_prompt_pack_mirror(loaded, result.diagnostics, mirror_dir=mirror_dir, base_dir=tmp_path)
    mirrored_file = mirror_dir / mirror.entries[0].mirror_path

    mirrored_file.write_text(mirrored_file.read_text(encoding="utf-8").replace("refund-v1", "refund-v2"), encoding="utf-8")
    verification = verify_prompt_pack_mirror(mirror_dir / "prompt-pack-mirror.json")

    assert not verification.ok
    assert [diagnostic.rule_id for diagnostic in verification.diagnostics] == ["prompt-pack-mirror-sha256-drift"]


def test_cli_prompt_pack_mirror_build_and_verify(tmp_path: Path, capsys) -> None:
    pack = tmp_path / "support.prompt-pack.json"
    config = tmp_path / "promptabi.json"
    mirror_dir = tmp_path / "mirror"
    _write_prompt_pack(pack, schema_digest="sha256:refund-v1")
    _write_safe_config(config, pack)

    assert main(["prompt-pack", "mirror", "build", "--config", str(config), "--mirror-dir", str(mirror_dir), "--format", "json"]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["mirror_version"] == 1
    assert built["prompt_packs"][0]["package_name"] == "support-pack"
    assert (mirror_dir / "prompt-pack-mirror.json").is_file()

    assert main(["prompt-pack", "mirror", "verify", "--manifest", str(mirror_dir / "prompt-pack-mirror.json"), "--format", "json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["diagnostics"][0]["rule_id"] == "prompt-pack-mirror-verified"

    mirrored_file = mirror_dir / built["prompt_packs"][0]["mirror_path"]
    mirrored_file.write_text(mirrored_file.read_text(encoding="utf-8").replace("refund_user", "refund_usfr"), encoding="utf-8")
    assert main(["prompt-pack", "mirror", "verify", "--manifest", str(mirror_dir / "prompt-pack-mirror.json"), "--format", "json"]) == 1
    tampered = json.loads(capsys.readouterr().out)
    assert tampered["ok"] is False
    assert tampered["diagnostics"][0]["rule_id"] == "prompt-pack-mirror-sha256-drift"
