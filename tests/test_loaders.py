import json
import struct
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

from promptabi import ArtifactKind, ArtifactLocation, ArtifactProvenance, ProviderConfigArtifact, TokenizerArtifact
from promptabi.artifacts import SchemaArtifact, artifact_from_config
from promptabi.loaders import ArtifactLoadError, ArtifactLoader
from promptabi.session import VerificationSession


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

    assert loaded.source_type == "local-file"
    assert loaded.pinned is True
    assert loaded.resolved is True
    assert loaded.actual_sha256 == digest
    assert loaded.size_bytes == len(schema.read_bytes())
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
