import json
from pathlib import Path

from promptabi import TrainingManifestArtifact, analyze_synthetic_generators
from promptabi.artifacts import artifact_from_config
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.session import VerificationSession


def _write_config(tmp_path: Path, manifest_path: Path, checks: list[str] | None = None) -> Path:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "synthetic-generator-contracts",
                "checks": checks or ["synthetic-generator-contracts"],
                "artifacts": {"train": {"kind": "training-manifest", "path": manifest_path.name}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def _clean_manifest() -> dict[str, object]:
    return {
        "dataset_format": "chat-jsonl",
        "datasets": [{"name": "synthetic-sft", "kind": "supervised", "format": "chat-jsonl"}],
        "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["assistant"]},
        "packing_window": {"strategy": "none", "preserve_example_boundaries": True, "reset_position_ids": True},
        "chat_template_version": {
            "name": "synthetic-chat-template",
            "version": "2026-06-05",
            "tokenizer_name": "synthetic-tokenizer",
            "add_generation_prompt": False,
        },
        "pipeline_stages": [
            {
                "stage": "training",
                "tokenizer_name": "synthetic-tokenizer",
                "tokenizer_sha256": "sha256:tok",
                "chat_template_name": "synthetic-chat-template",
                "chat_template_sha256": "sha256:tmpl",
                "add_generation_prompt": False,
            },
            {
                "stage": "serving",
                "tokenizer_name": "synthetic-tokenizer",
                "tokenizer_sha256": "sha256:tok",
                "chat_template_name": "synthetic-chat-template",
                "chat_template_sha256": "sha256:tmpl",
                "add_generation_prompt": False,
            },
        ],
        "metadata": {
            "training_interface_contract": {
                "allowed_roles": ["system", "user", "assistant", "tool"],
            }
        },
        "synthetic_generators": [
            {
                "name": "tool-and-json-sft-generator",
                "generator_type": "self-instruct-tool-sft",
                "output_roles": ["system", "user", "assistant", "tool"],
                "required_roles": ["user", "assistant"],
                "forbidden_roles": ["developer"],
                "max_prompt_tokens": 512,
                "max_completion_tokens": 256,
                "schema_outputs": [
                    {"id": "answer-json", "valid": True, "parses": True, "schema_valid": True}
                ],
                "tool_calls": [{"id": "search-call", "valid": True, "malformed": False}],
                "truncation_cases": [
                    {
                        "id": "long-user-short-answer",
                        "input_tokens": 480,
                        "output_tokens": 128,
                        "max_context_tokens": 1024,
                        "preserved_required_roles": ["user", "assistant"],
                    }
                ],
                "metadata": {"generator_revision": "sha256:generator"},
            }
        ],
    }


def test_synthetic_generator_contracts_verify_clean_generator(tmp_path: Path) -> None:
    manifest_path = tmp_path / "synthetic-generator.json"
    manifest_path.write_text(json.dumps(_clean_manifest(), sort_keys=True), encoding="utf-8")

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id.startswith("synthetic-generator-contracts")
    ]

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in diagnostics] == ["synthetic-generator-contracts-verified"]
    assert dict(diagnostics[0].properties)["kind"] == "verified"
    assert dict(diagnostics[0].properties)["generator_name"] == "tool-and-json-sft-generator"


def test_synthetic_generator_contracts_catch_roles_schema_tools_and_truncation(tmp_path: Path) -> None:
    manifest = _clean_manifest()
    manifest["synthetic_generators"] = [
        {
            "name": "bad-generator",
            "generator_type": "weak-self-instruct",
            "output_roles": ["user", "assistant", "asistant", "developer"],
            "required_roles": ["system", "user", "assistant"],
            "forbidden_roles": ["developer"],
            "schema_outputs": [{"id": "bad-json", "parses": False, "schema_error": "unterminated object"}],
            "tool_calls": [{"id": "bad-tool", "valid": False, "reason": "arguments are not JSON"}],
            "truncation_cases": [
                {
                    "id": "overflow",
                    "input_tokens": 900,
                    "output_tokens": 300,
                    "max_context_tokens": 1024,
                    "preserved_required_roles": ["user"],
                    "truncated_required_roles": ["assistant"],
                }
            ],
        }
    ]
    manifest_path = tmp_path / "bad-synthetic-generator.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.rule_id.startswith("synthetic-generator-contracts")
    ]
    rule_ids = {diagnostic.rule_id for diagnostic in diagnostics}
    kinds = {dict(diagnostic.properties)["kind"] for diagnostic in diagnostics}

    assert not result.ok
    assert {
        "role-contract-violation",
        "schema-contract-violation",
        "tool-call-contract-violation",
        "truncation-contract-violation",
    }.issubset(kinds)
    assert {
        "synthetic-generator-contracts-role-contract-violation",
        "synthetic-generator-contracts-schema-contract-violation",
        "synthetic-generator-contracts-tool-call-contract-violation",
        "synthetic-generator-contracts-truncation-contract-violation",
    } == rule_ids


def test_synthetic_generator_manifest_round_trips_through_artifact_model(tmp_path: Path) -> None:
    manifest_path = tmp_path / "synthetic-generator.json"
    manifest = _clean_manifest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    artifact = artifact_from_config(
        "train",
        {"kind": "training-manifest", "path": str(manifest_path), **manifest},
        base_dir=tmp_path,
    )

    assert isinstance(artifact, TrainingManifestArtifact)
    assert artifact.synthetic_generators[0].name == "tool-and-json-sft-generator"
    assert artifact.to_dict()["synthetic_generators"] == [
        artifact.synthetic_generators[0].to_dict()
    ]
    report = analyze_synthetic_generators(artifact)
    assert report.verified


def test_verify_training_runs_synthetic_generator_contracts(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "synthetic-generator.json"
    manifest_path.write_text(json.dumps(_clean_manifest(), sort_keys=True), encoding="utf-8")

    exit_code = main(["verify-training", "--manifest", str(manifest_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = {diagnostic["rule_id"] for diagnostic in payload["diagnostics"]}
    assert exit_code == 0
    assert "synthetic-generator-contracts" in payload["config"]["checks"]
    assert "synthetic-generator-contracts-verified" in rule_ids
    assert captured.err == ""
