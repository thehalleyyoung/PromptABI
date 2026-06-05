import json
from hashlib import sha256
from pathlib import Path

from promptabi.cli import main
from promptabi.compatibility_matrix import CHECK_RULE_IDS
from promptabi.config import load_config
from promptabi.session import VerificationSession


def test_artifact_provenance_verifies_pinned_licensed_trusted_local_artifact(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object","properties":{"answer":{"type":"string"}}}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "trusted-local",
                "checks": ["artifact-provenance"],
                "artifacts": {
                    "answer-schema": {
                        "kind": "schema",
                        "path": schema.name,
                        "sha256": sha256(schema.read_bytes()).hexdigest(),
                        "license": "Apache-2.0",
                        "source": "https://github.com/example/app/schemas/answer.schema.json",
                        "metadata": {"trusted_source": True},
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config_path).run()

    assert result.ok is True
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["artifact-provenance-verified"]
    diagnostic = result.diagnostics[0]
    assert diagnostic.artifact is None
    assert diagnostic.witness is not None
    assert diagnostic.witness.steps[1].output == "all resolved artifacts pinned by sha256"
    assert diagnostic.witness.artifacts[0].sha256 == sha256(schema.read_bytes()).hexdigest()


def test_artifact_provenance_reports_missing_hash_license_and_source(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "incomplete-local",
                "checks": ["artifact-provenance"],
                "artifacts": {
                    "answer-schema": {
                        "kind": "schema",
                        "path": schema.name,
                        "version": "reviewed-but-not-hashed",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config_path).run()

    assert result.ok is True
    rule_ids = {diagnostic.rule_id for diagnostic in result.diagnostics}
    assert rule_ids == {
        "artifact-provenance-missing-hash",
        "artifact-provenance-missing-license",
        "artifact-provenance-missing-source",
    }
    hash_diagnostic = next(
        diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "artifact-provenance-missing-hash"
    )
    assert dict(hash_diagnostic.properties)["source_type"] == "json-schema"
    assert hash_diagnostic.witness is not None
    assert any(step.action == "hash resolved artifact" for step in hash_diagnostic.witness.steps)


def test_artifact_provenance_requires_explicit_trusted_source_annotation(tmp_path: Path) -> None:
    schema = tmp_path / "answer.schema.json"
    schema.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "untrusted-source",
                "checks": ["artifact-provenance"],
                "artifacts": {
                    "answer-schema": {
                        "kind": "schema",
                        "path": schema.name,
                        "sha256": sha256(schema.read_bytes()).hexdigest(),
                        "license": "MIT",
                        "source": "https://github.com/example/app/schema.json",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    diagnostics = VerificationSession.from_config_file(config_path).run().diagnostics

    assert [diagnostic.rule_id for diagnostic in diagnostics] == ["artifact-provenance-untrusted-source"]
    assert diagnostics[0].suggestions == (
        "Set metadata.trusted_source to true only for reviewed internal mirrors or approved upstreams.",
    )


def test_artifact_provenance_rejects_movable_remote_downloads(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "movable-remote",
                "checks": ["artifact-provenance"],
                "artifacts": {
                    "remote-tokenizer": {
                        "kind": "tokenizer",
                        "uri": "hf://org/model/tokenizer_config.json?revision=main",
                        "license": "apache-2.0",
                        "source": "https://huggingface.co/org/model",
                        "metadata": {"trusted_source": True},
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert captured.err == ""
    assert any(
        diagnostic["rule_id"] == "artifact-provenance-nonreproducible-remote"
        and diagnostic["severity"] == "error"
        and "movable Hugging Face revision" in diagnostic["witness"]["steps"][1]["output"]
        for diagnostic in payload["diagnostics"]
    )


def test_artifact_provenance_is_listed_in_compatibility_matrix() -> None:
    config = load_config("examples/minimal/promptabi.json")
    session = VerificationSession(config)

    assert "artifact-provenance" in session.checks
    assert CHECK_RULE_IDS["artifact-provenance"] == (
        "artifact-provenance-missing-hash",
        "artifact-provenance-missing-license",
        "artifact-provenance-missing-source",
        "artifact-provenance-nonreproducible-remote",
        "artifact-provenance-untrusted-source",
        "artifact-provenance-verified",
    )
