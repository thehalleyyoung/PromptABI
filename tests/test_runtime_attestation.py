import json
import re
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.runtime_attestation import (
    RUNTIME_CONTRACT_FAMILIES,
    RuntimeAttestationError,
    build_runtime_attestation_report,
    render_runtime_attestation_json,
    render_runtime_attestation_text,
    runtime_contract_refs_from_cli,
    write_runtime_attestation_hooks,
)


ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _attestation_config(tmp_path: Path) -> Path:
    config = {
        "name": "runtime-attested-service",
        "checks": ["repository-skeleton"],
        "artifacts": {
            "system-prompt": {
                "kind": "prompt-segment",
                "uri": "memory://runtime/system-prompt",
                "segments": [{"name": "system", "role": "system", "required": True}],
            },
            "runtime-tokenizer": {
                "kind": "tokenizer",
                "uri": "memory://runtime/tokenizer",
                "family": "byte-bpe",
            },
            "chat-template": {
                "kind": "chat-template",
                "uri": "memory://runtime/template",
                "roles": ["system", "user", "assistant"],
            },
            "answer-schema": {
                "kind": "schema",
                "uri": "memory://runtime/schema",
                "dialect": "json-schema",
            },
            "provider-config": {
                "kind": "provider-config",
                "uri": "memory://runtime/provider",
                "provider": "openai-compatible",
            },
        },
    }
    path = tmp_path / "promptabi.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_runtime_attestation_reports_verified_contract_families_and_hooks(tmp_path: Path) -> None:
    config = _attestation_config(tmp_path)
    report = build_runtime_attestation_report(
        config,
        bundle_key="runtime-secret",
        service="checkout-agent",
        environment="prod",
        revision="image@sha256:abc",
        instance_id="pod-123",
        runtime_contract_refs={
            "runtime-tokenizer": "hf://org/model@abc/tokenizer.json",
            "provider-config": "openai-compatible:llama-3.1",
        },
    )
    payload = json.loads(render_runtime_attestation_json(report))
    families = payload["contract_families"]

    assert report.ok is True
    assert set(families) == set(RUNTIME_CONTRACT_FAMILIES)
    assert all(families[family] == 1 for family in RUNTIME_CONTRACT_FAMILIES)
    assert len(payload["bundle"]["bundle_hash"]) == 64
    assert len(payload["manifest_sha256"]) == 64
    assert {contract["family"] for contract in payload["contracts"]} == set(RUNTIME_CONTRACT_FAMILIES)
    assert all(len(contract["contract_hash"]) == 64 for contract in payload["contracts"])
    assert any(contract.get("runtime_ref") == "openai-compatible:llama-3.1" for contract in payload["contracts"])
    assert {hook["kind"] for hook in payload["hooks"]} == {
        "env-file",
        "http-json",
        "kubernetes-annotations",
        "opentelemetry-attributes",
    }
    assert "PromptABI runtime attestation" in render_runtime_attestation_text(report)


def test_runtime_attestation_is_deterministic_and_env_keys_are_valid(tmp_path: Path) -> None:
    config = _attestation_config(tmp_path)
    first = build_runtime_attestation_report(config, bundle_key="runtime-secret", service="svc", environment="prod")
    second = build_runtime_attestation_report(config, bundle_key="runtime-secret", service="svc", environment="prod")

    assert render_runtime_attestation_json(first) == render_runtime_attestation_json(second)
    assert first.manifest_sha256 == second.manifest_sha256

    env_hook = next(hook for hook in first.hooks if hook.kind.value == "env-file")
    for line in env_hook.content.splitlines():
        key, _value = line.split("=", 1)
        assert ENV_KEY_RE.fullmatch(key), key
    assert "PROMPTABI_CONTRACT_CHAT_TEMPLATE_HASH" in env_hook.content
    assert "PROMPTABI_CONTRACT_SYSTEM_PROMPT_HASH" in env_hook.content


def test_runtime_attestation_requires_signed_bundle_evidence(tmp_path: Path) -> None:
    config = _attestation_config(tmp_path)

    try:
        build_runtime_attestation_report(config)
    except RuntimeAttestationError as exc:
        assert "bundle signing key" in str(exc)
    else:
        raise AssertionError("expected runtime attestation to require signed bundle evidence")


def test_runtime_attestation_writer_and_cli_create_real_hooks(tmp_path: Path, capsys) -> None:
    config = _attestation_config(tmp_path)
    written = write_runtime_attestation_hooks(
        tmp_path / "attestation",
        config,
        bundle_key="runtime-secret",
        service="support-agent",
        runtime_contract_refs={"system-prompt": "prompt-pack:support@1.0.0"},
    )

    assert sorted(path.name for path in written.written_files) == [
        "kubernetes-annotations.yaml",
        "opentelemetry-attributes.json",
        "runtime-attestation.env",
        "runtime-attestation.json",
        "well-known-promptabi-attestation.json",
    ]
    manifest = json.loads((written.output_dir / "runtime-attestation.json").read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == written.report.manifest_sha256
    assert "prompt-pack:support@1.0.0" in (written.output_dir / "runtime-attestation.env").read_text(encoding="utf-8")

    output_dir = tmp_path / "cli-attestation"
    exit_code = main(
        [
            "runtime-attestation",
            "--config",
            str(config),
            "--bundle-key",
            "runtime-secret",
            "--service",
            "support-agent",
            "--runtime-contract",
            "chat-template=tokenizer_config.json",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "PromptABI runtime attestation" in captured.out
    assert (output_dir / "runtime-attestation.json").is_file()

    exit_code = main(["runtime-attestation", "--config", str(config), "--bundle-key", "runtime-secret", "--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["ok"] is True


def test_runtime_contract_cli_parser_rejects_malformed_values(tmp_path: Path, capsys) -> None:
    assert runtime_contract_refs_from_cli(["name=ref=with=equals"]) == {"name": "ref=with=equals"}

    exit_code = main(
        [
            "runtime-attestation",
            "--config",
            str(_attestation_config(tmp_path)),
            "--bundle-key",
            "runtime-secret",
            "--runtime-contract",
            "malformed",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--runtime-contract values must be NAME=REF" in captured.err


def test_runtime_attestation_public_api_renders_json(tmp_path: Path) -> None:
    payload = json.loads(
        promptabi.runtime_attestation(
            _attestation_config(tmp_path),
            bundle_key="runtime-secret",
            service="api-agent",
            output_format="json",
        )
    )

    assert payload["service"] == "api-agent"
    assert payload["contract_families"]["provider"] == 1
