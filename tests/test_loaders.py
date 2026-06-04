import json
import struct
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ProviderConfigArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
)
from promptabi.artifacts import SchemaArtifact, artifact_from_config
from promptabi.loaders import ArtifactLoadError, ArtifactLoader
from promptabi.session import VerificationSession
from promptabi.stop_policies import parse_stop_policy_config


def test_loader_hashes_and_validates_pinned_local_files(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    digest = sha256(schema.read_bytes()).hexdigest()
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="schema",
        location=ArtifactLocation(path=str(schema)),
        provenance=ArtifactProvenance(sha256=digest),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "json-schema"
    assert loaded.pinned is True
    assert loaded.resolved is True
    assert loaded.actual_sha256 == digest
    assert loaded.size_bytes == len(schema.read_bytes())
    assert dict(loaded.metadata)["root_kind"] == "object"
    assert loaded.warnings == ()


def test_loader_rejects_hash_mismatches_and_malformed_pins(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="malformed sha256"):
        ArtifactLoader().load(
            SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name="schema",
                location=ArtifactLocation(path=str(schema)),
                provenance=ArtifactProvenance(sha256="abc"),
            )
        )

    with pytest.raises(ArtifactLoadError, match="does not match"):
        ArtifactLoader().load(
            SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name="schema",
                location=ArtifactLocation(path=str(schema)),
                provenance=ArtifactProvenance(sha256="0" * 64),
            )
        )


def test_loader_summarizes_tokenizer_directories_deterministically(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tok"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer_config.json").write_text('{"model_max_length":4096}', encoding="utf-8")
    (tokenizer_dir / "special_tokens_map.json").write_text('{"eos_token":"</s>"}', encoding="utf-8")
    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="tok",
        location=ArtifactLocation(path=str(tokenizer_dir)),
        provenance=ArtifactProvenance(version="local-test"),
    )

    first = ArtifactLoader().load(artifact)
    second = ArtifactLoader().load(artifact)

    assert first.source_type == "tokenizer-directory"
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.members == ("special_tokens_map.json", "tokenizer_config.json")
    assert first.warnings == ()


def test_loader_parses_huggingface_model_repo_refs_without_network() -> None:
    commit = "a" * 40
    artifact = artifact_from_config(
        "llama-tokenizer",
        {
            "kind": "tokenizer",
            "uri": f"hf://meta-llama/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json?revision={commit}",
        },
        base_dir=Path("."),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "huggingface-model-repo"
    assert loaded.resolved is False
    assert loaded.pinned is True
    assert dict(loaded.metadata) == {
        "artifact_path": "tokenizer_config.json",
        "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "revision": commit,
    }
    assert loaded.warnings == ()


def test_loader_warns_on_movable_huggingface_refs() -> None:
    artifact = artifact_from_config(
        "llama-tokenizer",
        {
            "kind": "tokenizer",
            "uri": "hf://meta-llama/Meta-Llama-3.1-8B-Instruct?revision=main",
        },
        base_dir=Path("."),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.pinned is True
    assert [warning.rule_id for warning in loaded.warnings] == ["artifact-weak-pin"]


def test_loader_treats_memory_refs_as_embedded_artifacts() -> None:
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="schema",
        location=ArtifactLocation(uri="memory://schema"),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "memory"
    assert loaded.pinned is True
    assert loaded.resolved is True
    assert loaded.warnings == ()


def test_loader_reads_gguf_metadata_stub_header(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 12, 5))
    artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="gguf-tokenizer",
        location=ArtifactLocation(path=str(gguf)),
        provenance=ArtifactProvenance(version="stub-v1"),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "gguf-metadata-stub"
    assert dict(loaded.metadata) == {
        "gguf_version": 3,
        "metadata_kv_count": 5,
        "tensor_count": 12,
    }


def test_loader_validates_provider_config_snapshots(tmp_path: Path) -> None:
    snapshot = tmp_path / "openai.snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "captured_at": "2026-06-04T00:00:00Z",
                "request_shape": {"messages": "array"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = ProviderConfigArtifact(
        kind=ArtifactKind.PROVIDER_CONFIG,
        name="provider",
        location=ArtifactLocation(path=str(snapshot)),
        provenance=ArtifactProvenance(sha256=sha256(snapshot.read_bytes()).hexdigest()),
        provider="openai-compatible",
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "provider-config-snapshot"
    assert dict(loaded.metadata) == {"provider": "openai-compatible"}
    assert loaded.warnings == ()


def test_stop_policy_parser_covers_common_provider_and_framework_shapes() -> None:
    openai = parse_stop_policy_config(
        {
            "provider": "openai",
            "request_shape": {"messages": "array"},
            "stop": ["</tool_call>", "\n\n"],
        }
    )
    assert openai.source_family == "openai-compatible"
    assert openai.stop_sequences == ("\n\n", "</tool_call>")
    assert openai.include_eos is True

    huggingface = parse_stop_policy_config(
        {
            "generation_config": {
                "stop_strings": ["<|eot_id|>"],
                "eos_token_id": [128001, 128009],
            }
        }
    )
    assert huggingface.source_family == "huggingface"
    assert huggingface.stop_sequences == ("<|eot_id|>",)
    assert huggingface.stop_token_ids == (128001, 128009)

    llama_cpp = parse_stop_policy_config(
        {"backend": "llama.cpp", "reverse_prompt": ["User:"], "ignore_eos": True}
    )
    assert llama_cpp.source_family == "llama.cpp"
    assert llama_cpp.stop_sequences == ("User:",)
    assert llama_cpp.include_eos is False

    vllm = parse_stop_policy_config(
        {"framework": "vllm", "sampling_params": {"stop": "</s>", "stop_token_ids": [2], "ignore_eos": False}}
    )
    assert vllm.source_family == "vllm"
    assert vllm.stop_sequences == ("</s>",)
    assert vllm.stop_token_ids == (2,)
    assert vllm.include_eos is True

    litellm = parse_stop_policy_config({"litellm_params": {"stop": ["Observation:"], "include_eos": False}})
    assert litellm.source_family == "litellm"
    assert litellm.stop_sequences == ("Observation:",)
    assert litellm.include_eos is False

    wrapper = parse_stop_policy_config({"model_kwargs": {"generation_kwargs": {"stop_sequences": ["###"]}}})
    assert wrapper.source_family == "framework-wrapper"
    assert wrapper.stop_sequences == ("###",)


def test_loader_parses_stop_policy_config_and_preserves_source_metadata(tmp_path: Path) -> None:
    config = tmp_path / "vllm-stop-policy.json"
    config.write_text(
        json.dumps(
            {
                "framework": "vllm",
                "sampling_params": {
                    "stop": ["</tool_call>", ""],
                    "stop_token_ids": [128001, 128009],
                    "ignore_eos": True,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(path=str(config)),
        provenance=ArtifactProvenance(sha256=sha256(config.read_bytes()).hexdigest()),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "stop-policy-config"
    assert isinstance(loaded.artifact, StopPolicyArtifact)
    assert loaded.artifact.stop_sequences == ("</tool_call>",)
    assert loaded.artifact.stop_token_ids == (128001, 128009)
    assert loaded.artifact.include_eos is False
    metadata = dict(loaded.metadata)
    assert metadata["source_family"] == "vllm"
    assert metadata["stop_sequence_count"] == 1
    assert metadata["stop_token_id_count"] == 2
    assert metadata["ignored_empty_sequences"] == ("sampling_params.stop[1]",)
    assert any(name == "sampling_params.stop.0" for name, _span in loaded.source_spans)


def test_loader_rejects_malformed_stop_policy_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad-stop-policy.json"
    config.write_text('{"stop": [123]}', encoding="utf-8")
    artifact = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="bad-stops",
        location=ArtifactLocation(path=str(config)),
    )

    with pytest.raises(ArtifactLoadError, match="could not be parsed"):
        ArtifactLoader().load(artifact)


def test_loader_summarizes_archived_fixture_bundles(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixtures.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("bundle/metadata.json", "{}")
        archive.writestr("bundle/tokenizer_config.json", "{}")
    artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="fixtures",
        location=ArtifactLocation(path=str(archive_path)),
        provenance=ArtifactProvenance(sha256=sha256(archive_path.read_bytes()).hexdigest()),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "fixture-bundle-archive"
    assert loaded.members == ("bundle/metadata.json", "bundle/tokenizer_config.json")
    assert loaded.warnings == ()


def test_verification_session_reports_loader_warnings_and_keeps_missing_canonical(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    unpinned_config = tmp_path / "promptabi.json"
    unpinned_config.write_text(
        '{"name":"unpinned","artifacts":{"schema":{"kind":"schema","path":"schema.json"}}}',
        encoding="utf-8",
    )

    warning_result = VerificationSession.from_config_file(unpinned_config).run()

    assert warning_result.ok is True
    assert warning_result.diagnostics[0].rule_id == "artifact-unpinned"

    missing_config = tmp_path / "missing.promptabi.json"
    missing_config.write_text(
        '{"name":"missing","artifacts":{"schema":{"kind":"schema","path":"missing.json"}}}',
        encoding="utf-8",
    )

    missing_result = VerificationSession.from_config_file(missing_config).run()

    assert missing_result.ok is False
    assert missing_result.diagnostics[0].rule_id == "artifact-missing"
