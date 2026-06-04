"""Configuration loading for the first PromptABI workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a PromptABI configuration cannot be loaded soundly."""


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """A minimal, versioned config object for step-one verification sessions."""

    name: str
    artifacts: dict[str, str] = field(default_factory=dict)
    checks: tuple[str, ...] = ("repository-skeleton",)
    max_context_tokens: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, base_dir: Path) -> "VerificationConfig":
        name = data.get("name", "unnamed")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("config field 'name' must be a non-empty string")

        raw_artifacts = data.get("artifacts", {})
        if not isinstance(raw_artifacts, dict):
            raise ConfigError("config field 'artifacts' must be an object")
        artifacts: dict[str, str] = {}
        for key, value in sorted(raw_artifacts.items()):
            if not isinstance(key, str) or not key:
                raise ConfigError("artifact names must be non-empty strings")
            if not isinstance(value, str) or not value:
                raise ConfigError(f"artifact '{key}' path must be a non-empty string")
            artifact_path = Path(value)
            if not artifact_path.is_absolute():
                artifact_path = base_dir / artifact_path
            artifacts[key] = str(artifact_path.resolve())

        raw_checks = data.get("checks", ["repository-skeleton"])
        if not isinstance(raw_checks, list) or not all(isinstance(item, str) for item in raw_checks):
            raise ConfigError("config field 'checks' must be a list of strings")
        checks = tuple(sorted(dict.fromkeys(raw_checks)))

        raw_max_context = data.get("max_context_tokens")
        if raw_max_context is not None and (
            not isinstance(raw_max_context, int) or raw_max_context <= 0
        ):
            raise ConfigError("config field 'max_context_tokens' must be a positive integer")

        return cls(
            name=name.strip(),
            artifacts=artifacts,
            checks=checks,
            max_context_tokens=raw_max_context,
        )


def load_config(path: str | Path) -> VerificationConfig:
    """Load a JSON PromptABI config file from disk."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    return VerificationConfig.from_mapping(raw, base_dir=config_path.parent)

