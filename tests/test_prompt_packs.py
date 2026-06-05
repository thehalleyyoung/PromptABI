import json
from pathlib import Path

from promptabi.artifacts import ArtifactKind, artifact_from_config
from promptabi.cli import main
from promptabi.loaders import ArtifactLoader
from promptabi.prompt_packs import (
    build_prompt_pack_lockfile,
    build_prompt_pack_registry,
    compare_prompt_pack_lockfile,
    compare_prompt_pack_upgrade,
    prompt_pack_registry_to_json,
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
                "roles": ["system", "user", "assistant", "tool"],
                "variables": ["system", "transcript"],
                "required_regions": ["system-policy", "tool-result"],
                "supported_model_families": ["openai-compatible"],
            }
        )
    tool_schema = {"name": tool_name, "provider": "openai"}
    if schema_digest is not None:
        tool_schema["schema_digest"] = schema_digest
    path.write_text(
        json.dumps(
            {
                "name": "support-pack",
                "version": "1.0.0",
                "exported_templates": template_entries,
                "expected_roles": expected_roles or ["system", "user", "assistant"],
                "tool_schemas": [tool_schema],
                "stop_policies": [{"name": "tool-json", "stop_sequences": [stop]}],
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
