"""Configuration loading for the first PromptABI workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ArtifactBundle,
    artifact_from_cli_override,
    artifact_from_config,
    local_artifact_paths,
)


class ConfigError(ValueError):
    """Raised when a PromptABI configuration cannot be loaded soundly."""


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """A versioned config object for PromptABI verification sessions."""

    name: str
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_bundle: ArtifactBundle = field(default_factory=ArtifactBundle)
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
        typed_artifacts = []
        for key, value in sorted(raw_artifacts.items()):
            if not isinstance(key, str) or not key:
                raise ConfigError("artifact names must be non-empty strings")
            try:
                typed_artifacts.append(artifact_from_config(key, value, base_dir=base_dir))
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        artifact_bundle = ArtifactBundle(tuple(typed_artifacts))
        artifacts = local_artifact_paths(artifact_bundle)

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
            artifact_bundle=artifact_bundle,
            checks=checks,
            max_context_tokens=raw_max_context,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "checks": list(self.checks),
            "artifacts": dict(sorted(self.artifacts.items())),
            "artifact_bundle": self.artifact_bundle.to_dict(),
            "max_context_tokens": self.max_context_tokens,
        }

    def with_artifact_overrides(self, overrides: dict[str, str], *, base_dir: Path) -> "VerificationConfig":
        """Return a config with CLI-provided artifact paths or URIs applied."""

        if not overrides:
            return self
        artifacts_by_name: dict[str, Artifact] = {artifact.name: artifact for artifact in self.artifact_bundle}
        for name, value in sorted(overrides.items()):
            artifacts_by_name[name] = artifact_from_cli_override(
                name,
                value,
                base_dir=base_dir,
                existing=artifacts_by_name.get(name),
            )
        artifact_bundle = ArtifactBundle(tuple(artifacts_by_name.values()))
        return VerificationConfig(
            name=self.name,
            artifacts=local_artifact_paths(artifact_bundle),
            artifact_bundle=artifact_bundle,
            checks=self.checks,
            max_context_tokens=self.max_context_tokens,
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


CONFIG_FILENAMES = ("promptabi.json", ".promptabi.json")


def discover_config(start: str | Path = ".") -> Path:
    """Find a PromptABI config by walking from ``start`` toward the filesystem root."""

    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        for filename in CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    names = ", ".join(CONFIG_FILENAMES)
    raise ConfigError(f"no PromptABI config found from {current} (looked for {names})")
