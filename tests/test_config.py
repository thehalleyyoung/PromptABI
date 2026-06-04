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


def test_config_rejects_invalid_checks() -> None:
    with pytest.raises(ConfigError, match="checks"):
        VerificationConfig.from_mapping({"name": "demo", "checks": "all"}, base_dir=Path("."))

