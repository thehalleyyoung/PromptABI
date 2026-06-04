from pathlib import Path

import pytest

from promptabi.config import ConfigError, VerificationConfig, load_config


def test_config_normalizes_artifact_paths(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text('{"name": "demo", "artifacts": {"schema": "schema.json"}}', encoding="utf-8")

    loaded = load_config(config)

    assert loaded.artifacts == {"schema": str(schema.resolve())}
    assert loaded.artifact_bundle.by_name("schema").kind == "schema"


def test_config_loads_typed_artifacts_and_keeps_legacy_path_map(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    provider_uri = "hf://meta-llama/Meta-Llama-3.1-8B-Instruct"
    config = tmp_path / "promptabi.json"
    config.write_text(
        """
        {
          "name": "typed",
          "artifacts": {
            "tok": {
              "kind": "tokenizer",
              "path": "tokenizer.json",
              "family": "byte-bpe",
              "version": "v1"
            },
            "provider": {
              "kind": "provider-config",
              "uri": "hf://meta-llama/Meta-Llama-3.1-8B-Instruct",
              "provider": "openai-compatible"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.artifacts == {"tok": str(tokenizer.resolve())}
    assert loaded.artifact_bundle.by_name("tok").to_ref().to_dict() == {
        "kind": "tokenizer",
        "name": "tok",
        "path": str(tokenizer.resolve()),
        "version": "v1",
    }
    assert loaded.artifact_bundle.by_name("provider").location.uri == provider_uri


def test_config_rejects_invalid_checks() -> None:
    with pytest.raises(ConfigError, match="checks"):
        VerificationConfig.from_mapping({"name": "demo", "checks": "all"}, base_dir=Path("."))


def test_config_rejects_unknown_artifact_kinds() -> None:
    with pytest.raises(ConfigError, match="unsupported kind"):
        VerificationConfig.from_mapping(
            {"name": "demo", "artifacts": {"bad": {"kind": "weights", "path": "model.bin"}}},
            base_dir=Path("."),
        )
