import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader
from promptabi.session import CHECK_MODE_CATALOG
from promptabi.session import VerificationSession
from promptabi.training_streaming import TrainingStreamingFindingKind, analyze_training_streaming


SHA = "sha256:" + "a" * 64


def test_training_streaming_samples_jsonl_chunks_and_verifies_hashes(tmp_path: Path) -> None:
    manifest_path = _write_streaming_manifest(tmp_path, clean=True)
    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith("training-streaming")]

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in diagnostics] == ["training-streaming-verified"]
    properties = dict(diagnostics[0].properties)
    assert properties["rows_sampled"] == 5
    assert properties["chunks_sampled"] == 3
    assert properties["sample_name"] == "streamed-chat"
    assert "validate chunks" in [step.action for step in diagnostics[0].witness.steps]


def test_training_streaming_reports_real_sample_invariant_failures(tmp_path: Path) -> None:
    manifest_path = _write_streaming_manifest(tmp_path, clean=False)
    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith("training-streaming")]
    rule_ids = {diagnostic.rule_id for diagnostic in diagnostics}
    kinds = {dict(diagnostic.properties)["kind"] for diagnostic in diagnostics}

    assert not result.ok
    assert {
        "training-streaming-field-missing",
        "training-streaming-role-violation",
        "training-streaming-hash-missing",
    }.issubset(rule_ids)
    assert {"field-missing", "role-violation", "hash-missing"}.issubset(kinds)
    assert all(dict(diagnostic.properties)["rows_sampled"] <= 5 for diagnostic in diagnostics)


def test_training_streaming_api_and_cli_use_bounded_manifest_samples(tmp_path: Path, capsys) -> None:
    manifest_path = _write_streaming_manifest(tmp_path, clean=True)
    config_path = _write_config(tmp_path, manifest_path)
    artifact = ArtifactLoader().load(load_config(config_path).artifact_bundle.artifacts[0]).artifact
    report = analyze_training_streaming(artifact, base_dir=tmp_path)

    assert report.verified
    assert promptabi.analyze_training_streaming(artifact, base_dir=tmp_path).verified

    exit_code = main(["verify-training", "--manifest", str(manifest_path), "--format", "json", "--fail-on", "error"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    streaming_rules = [diagnostic["rule_id"] for diagnostic in payload["diagnostics"] if diagnostic["rule_id"].startswith("training-streaming")]

    assert exit_code == 0
    assert captured.err == ""
    assert "training-streaming-verified" in streaming_rules


def test_training_streaming_abstains_without_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "no-streaming.training-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_format": "chat-jsonl",
                "datasets": [{"name": "sft", "kind": "supervised", "format": "chat-jsonl"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith("training-streaming")]

    assert result.ok
    assert [diagnostic.rule_id for diagnostic in diagnostics] == ["training-streaming-contract-missing"]
    assert diagnostics[0].check_modes[0].value == "abstaining"


def test_training_streaming_verifies_proof_carrying_dataset_shard(tmp_path: Path) -> None:
    manifest_path = _write_streaming_manifest(tmp_path, clean=True)
    _attach_proof_sidecar(tmp_path, manifest_path)

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith("training-streaming")]
    rule_ids = [diagnostic.rule_id for diagnostic in diagnostics]

    assert result.ok
    assert rule_ids == ["training-streaming-shard-proof-verified", "training-streaming-verified"]
    proof_properties = dict(diagnostics[0].properties)
    assert proof_properties["rows_sampled"] == 5
    assert proof_properties["chunks_sampled"] == 3
    assert proof_properties["sample_name"] == "streamed-chat"
    assert "stream shard digest" in [step.action for step in diagnostics[0].witness.steps]

    artifact = ArtifactLoader().load(load_config(_write_config(tmp_path, manifest_path)).artifact_bundle.artifacts[0]).artifact
    assert analyze_training_streaming(artifact, base_dir=tmp_path).verified


def test_training_streaming_rejects_stale_or_sensitive_shard_proofs(tmp_path: Path) -> None:
    manifest_path = _write_streaming_manifest(tmp_path, clean=True)
    sidecar_path = _attach_proof_sidecar(tmp_path, manifest_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["artifact_hashes"]["dataset"] = "sha256:" + "b" * 64
    payload["counterexample_fingerprints"] = ["raw prompt excerpt"]
    sidecar_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = VerificationSession(load_config(_write_config(tmp_path, manifest_path))).run()
    diagnostics = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id.startswith("training-streaming")]
    rule_ids = {diagnostic.rule_id for diagnostic in diagnostics}

    assert not result.ok
    assert "training-streaming-shard-proof-hash-mismatch" in rule_ids
    assert "training-streaming-shard-proof-unsafe-fingerprint" in rule_ids
    assert "training-streaming-verified" not in rule_ids


def test_training_streaming_finding_kinds_are_registered_for_diagnostics() -> None:
    missing = [
        f"training-streaming-{kind.value}"
        for kind in TrainingStreamingFindingKind
        if f"training-streaming-{kind.value}" not in CHECK_MODE_CATALOG
    ]

    assert missing == []


def _write_config(tmp_path: Path, manifest_path: Path) -> Path:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "training-streaming",
                "checks": ["training-streaming"],
                "artifacts": {"train": {"kind": "training-manifest", "path": manifest_path.name}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_streaming_manifest(tmp_path: Path, *, clean: bool) -> Path:
    sample_path = tmp_path / ("clean.jsonl" if clean else "bad.jsonl")
    rows = [
        {
            "messages": [{"role": "system", "content": "policy"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "assistant_target_sha256": SHA,
            "loss_mask_sha256": SHA,
        },
        {
            "messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
            "assistant_target_sha256": SHA,
            "loss_mask_sha256": SHA,
        },
        {
            "messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}],
            "assistant_target_sha256": SHA,
            "loss_mask_sha256": SHA,
        },
        {
            "messages": [{"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"}],
            "assistant_target_sha256": SHA,
            "loss_mask_sha256": SHA,
        },
        {
            "messages": [{"role": "user", "content": "q5"}, {"role": "assistant", "content": "a5"}],
            "assistant_target_sha256": SHA,
            "loss_mask_sha256": SHA,
        },
    ]
    if not clean:
        rows[1].pop("assistant_target_sha256")
        rows[2]["messages"] = [{"role": "attacker", "content": "bad"}]
        rows[3]["loss_mask_sha256"] = "not-a-hash"
    sample_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    manifest_path = tmp_path / ("clean.training-manifest.json" if clean else "bad.training-manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_format": "streaming-chat-jsonl",
                "datasets": [
                    {
                        "name": "sft-stream",
                        "kind": "supervised",
                        "format": "chat-jsonl",
                        "path": sample_path.name,
                        "content_fields": ["messages", "assistant_target_sha256", "loss_mask_sha256"],
                    }
                ],
                "role_labels": [
                    {"source_role": "system", "canonical_role": "system"},
                    {"source_role": "user", "canonical_role": "user"},
                    {"source_role": "assistant", "canonical_role": "assistant", "supervised_target": True},
                ],
                "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["assistant"]},
                "packing_window": {
                    "strategy": "sample-packing",
                    "max_tokens": 2048,
                    "boundary_token": "<|eot_id|>",
                    "preserve_example_boundaries": True,
                    "reset_position_ids": True,
                },
                "chat_template_version": {
                    "name": "chatml-fixture",
                    "version": "template-v1",
                    "tokenizer_name": "tok",
                    "add_generation_prompt": False,
                },
                "pipeline_stages": [
                    {
                        "stage": "training",
                        "tokenizer_name": "tok",
                        "tokenizer_version": "tok-v1",
                        "chat_template_name": "chatml-fixture",
                        "chat_template_version": "template-v1",
                        "add_generation_prompt": False,
                    },
                    {
                        "stage": "serving",
                        "tokenizer_name": "tok",
                        "tokenizer_version": "tok-v1",
                        "chat_template_name": "chatml-fixture",
                        "chat_template_version": "template-v1",
                        "add_generation_prompt": False,
                    },
                ],
                "metadata": {
                    "training_interface_contract": {
                        "allowed_roles": ["system", "user", "assistant"],
                        "tool_calls": [{"id": "none", "valid": True}],
                        "json_outputs": [{"id": "none", "valid": True, "parses": True, "schema_valid": True}],
                        "stop_sequences": [{"sequence": "<|eot_id|>", "reachable": True, "matching_examples": 5}],
                    },
                    "streaming_dataset_verification": [
                        {
                            "name": "streamed-chat",
                            "dataset": "sft-stream",
                            "sample_rows": 5,
                            "expected_sample_rows": 5,
                            "chunk_size": 2,
                            "max_row_bytes": 512,
                            "hash_fields": ["assistant_target_sha256", "loss_mask_sha256"],
                            "allowed_roles": ["system", "user", "assistant"],
                        }
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _attach_proof_sidecar(tmp_path: Path, manifest_path: Path) -> Path:
    sidecar_path = tmp_path / "clean.jsonl.promptabi-proof.json"
    proof = promptabi.build_dataset_shard_proof(
        shard_path="clean.jsonl",
        sample_name="streamed-chat",
        rows_sampled=5,
        chunks_sampled=3,
        hash_fields=("assistant_target_sha256", "loss_mask_sha256"),
        allowed_roles=("system", "user", "assistant"),
        chunk_size=2,
        sample_rows=5,
        counterexample_fingerprints=(),
        base_dir=tmp_path,
    )
    sidecar_path.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["streaming_dataset_verification"][0]["proof_sidecars"] = [sidecar_path.name]
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return sidecar_path
