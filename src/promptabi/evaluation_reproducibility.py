"""Evaluation-harness reproducibility pins for published benchmark runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .artifacts import (
    Artifact,
    ArtifactKind,
    ChatTemplateArtifact,
    EvaluationHarnessArtifact,
    ProviderConfigArtifact,
    SchemaArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
)
from .loaders import LoadedArtifact
from .session import VerificationSession


EVALUATION_REPRODUCIBILITY_VERSION = 1
DEFAULT_EVALUATION_REPRODUCIBILITY_CONFIGS = (
    Path(__file__).resolve().parents[2] / "examples" / "evaluation-harness" / "safe.promptabi.json",
)


class EvaluationReproducibilityError(ValueError):
    """Raised when an evaluation reproducibility manifest cannot be built."""


@dataclass(frozen=True, slots=True)
class EvaluationReproducibilityReport:
    """Pinned benchmark-interface surfaces for one or more evaluation harness configs."""

    configs: tuple[dict[str, object], ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": EVALUATION_REPRODUCIBILITY_VERSION,
            "promptabi_version": __version__,
            "purpose": (
                "Evaluation reproducibility pins for prompt rendering, tokenizer versions, "
                "provider fixtures, stop policies, and parser contracts."
            ),
            "config_count": len(self.configs),
            "configs": list(self.configs),
            "manifest_sha256": self.manifest_sha256,
        }


def build_evaluation_reproducibility_report(
    configs: Sequence[str | Path] | None = None,
) -> EvaluationReproducibilityReport:
    """Build deterministic reproducibility pins from real evaluation-harness configs."""

    config_paths = tuple(Path(path) for path in (configs or DEFAULT_EVALUATION_REPRODUCIBILITY_CONFIGS))
    if not config_paths:
        raise EvaluationReproducibilityError("at least one evaluation harness config is required")
    entries = tuple(_config_entry(path) for path in sorted(config_paths, key=lambda item: str(item)))
    payload_without_hash = {
        "manifest_version": EVALUATION_REPRODUCIBILITY_VERSION,
        "promptabi_version": __version__,
        "configs": entries,
    }
    return EvaluationReproducibilityReport(
        configs=entries,
        manifest_sha256=_stable_json_hash(payload_without_hash),
    )


def render_evaluation_reproducibility_json(report: EvaluationReproducibilityReport) -> str:
    """Render an evaluation reproducibility report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_evaluation_reproducibility_text(report: EvaluationReproducibilityReport) -> str:
    """Render a concise terminal summary of pinned evaluation surfaces."""

    lines = [
        "PromptABI evaluation reproducibility",
        f"configs: {len(report.configs)}",
        f"manifest sha256: {report.manifest_sha256}",
    ]
    for config in report.configs:
        surfaces = config["surfaces"]  # type: ignore[index]
        surface_hashes = {
            name: surface["surface_sha256"]  # type: ignore[index]
            for name, surface in sorted(surfaces.items())  # type: ignore[union-attr]
        }
        lines.append(f"- {config['config_path']}: {config['reproducibility_status']}")
        for name, digest in surface_hashes.items():
            lines.append(f"  {name}: {digest}")
    return "\n".join(lines) + "\n"


def _config_entry(config_path: Path) -> dict[str, object]:
    try:
        session = VerificationSession.from_config_file(config_path)
        loaded = session.load_artifacts()
    except Exception as exc:
        raise EvaluationReproducibilityError(f"cannot load evaluation config {config_path}: {exc}") from exc

    loaded_by_kind = _loaded_by_kind(loaded)
    harnesses = tuple(item for item in loaded if isinstance(item.artifact, EvaluationHarnessArtifact))
    if not harnesses:
        raise EvaluationReproducibilityError(
            f"config does not contain an evaluation-harness artifact: {config_path}"
        )

    surfaces = {
        "prompt_rendering": _prompt_rendering_surface(harnesses, loaded_by_kind.get(ArtifactKind.CHAT_TEMPLATE, ())),
        "tokenizer_versions": _tokenizer_surface(harnesses, loaded_by_kind.get(ArtifactKind.TOKENIZER, ())),
        "provider_fixtures": _provider_surface(harnesses, loaded_by_kind.get(ArtifactKind.PROVIDER_CONFIG, ())),
        "stop_policies": _stop_surface(harnesses, loaded_by_kind.get(ArtifactKind.STOP_POLICY, ())),
        "parser_contracts": _parser_surface(harnesses, loaded_by_kind.get(ArtifactKind.SCHEMA, ())),
    }
    missing = tuple(sorted(name for name, surface in surfaces.items() if not surface["complete"]))
    entry: dict[str, object] = {
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "session_name": session.config.name,
        "harness_names": [item.artifact.name for item in harnesses],
        "reproducibility_status": "complete" if not missing else "incomplete",
        "missing_surfaces": list(missing),
        "surfaces": surfaces,
    }
    entry["config_reproducibility_sha256"] = _stable_json_hash(entry)
    return entry


def _loaded_by_kind(loaded: Sequence[LoadedArtifact]) -> dict[ArtifactKind, tuple[LoadedArtifact, ...]]:
    grouped: dict[ArtifactKind, list[LoadedArtifact]] = {}
    for item in loaded:
        grouped.setdefault(item.artifact.kind, []).append(item)
    return {
        kind: tuple(sorted(items, key=lambda item: item.artifact.name))
        for kind, items in grouped.items()
    }


def _prompt_rendering_surface(
    harnesses: Sequence[LoadedArtifact],
    templates: Sequence[LoadedArtifact],
) -> dict[str, object]:
    harness_payload = []
    for loaded in harnesses:
        harness = _require_artifact(loaded.artifact, EvaluationHarnessArtifact)
        conversation_turns = [turn.to_dict() for turn in harness.conversation_turns]
        few_shots = [example.to_dict() for example in harness.few_shot_examples]
        harness_payload.append(
            {
                "artifact": loaded.artifact.name,
                "benchmark_name": harness.benchmark_name,
                "prompt_template": harness.prompt_template,
                "allowed_roles": list(harness.allowed_roles),
                "required_prompt_variables": list(harness.required_prompt_variables),
                "prompt_variables": list(harness.prompt_variables),
                "max_prompt_tokens": harness.max_prompt_tokens,
                "max_history_messages": harness.max_history_messages,
                "max_history_tokens": harness.max_history_tokens,
                "preserve_system_prompt": harness.preserve_system_prompt,
                "preserve_tool_messages": harness.preserve_tool_messages,
                "retained_turn_ids": list(harness.retained_turn_ids),
                "dropped_turn_ids": list(harness.dropped_turn_ids),
                "conversation_turns_sha256": _stable_json_hash(conversation_turns),
                "few_shot_examples_sha256": _stable_json_hash(few_shots),
            }
        )
    payload = {
        "harness_prompt_contracts": sorted(harness_payload, key=lambda item: str(item["artifact"])),
        "chat_templates": [_chat_template_pin(template) for template in templates],
    }
    return _surface(payload, complete=bool(harness_payload and templates))


def _tokenizer_surface(
    harnesses: Sequence[LoadedArtifact],
    tokenizers: Sequence[LoadedArtifact],
) -> dict[str, object]:
    snapshots = []
    for loaded in harnesses:
        harness = _require_artifact(loaded.artifact, EvaluationHarnessArtifact)
        if harness.benchmark_tokenizer is not None:
            snapshots.append({"artifact": harness.name, "benchmark_tokenizer": harness.benchmark_tokenizer.to_dict()})
    payload = {
        "harness_benchmark_tokenizers": sorted(snapshots, key=lambda item: str(item["artifact"])),
        "tokenizers": [_tokenizer_pin(tokenizer) for tokenizer in tokenizers],
    }
    return _surface(payload, complete=bool(snapshots and tokenizers))


def _provider_surface(
    harnesses: Sequence[LoadedArtifact],
    providers: Sequence[LoadedArtifact],
) -> dict[str, object]:
    harness_payload = []
    for loaded in harnesses:
        harness = _require_artifact(loaded.artifact, EvaluationHarnessArtifact)
        harness_payload.append(
            {
                "artifact": harness.name,
                "provider": harness.provider,
                "model": harness.model,
            }
        )
    payload = {
        "harness_provider_contracts": sorted(harness_payload, key=lambda item: str(item["artifact"])),
        "provider_configs": [_provider_pin(provider) for provider in providers],
    }
    return _surface(payload, complete=bool(harness_payload and providers))


def _stop_surface(
    harnesses: Sequence[LoadedArtifact],
    stop_policies: Sequence[LoadedArtifact],
) -> dict[str, object]:
    harness_payload = []
    for loaded in harnesses:
        harness = _require_artifact(loaded.artifact, EvaluationHarnessArtifact)
        harness_payload.append(
            {
                "artifact": harness.name,
                "stop_sequences": list(harness.stop_sequences),
            }
        )
    payload = {
        "harness_stop_contracts": sorted(harness_payload, key=lambda item: str(item["artifact"])),
        "stop_policies": [_stop_policy_pin(stop_policy) for stop_policy in stop_policies],
    }
    return _surface(payload, complete=bool(harness_payload and stop_policies))


def _parser_surface(
    harnesses: Sequence[LoadedArtifact],
    schemas: Sequence[LoadedArtifact],
) -> dict[str, object]:
    harness_payload = []
    for loaded in harnesses:
        harness = _require_artifact(loaded.artifact, EvaluationHarnessArtifact)
        harness_payload.append(
            {
                "artifact": harness.name,
                "answer_parser": harness.answer_parser,
                "answer_schema": harness.answer_schema,
            }
        )
    payload = {
        "harness_parser_contracts": sorted(harness_payload, key=lambda item: str(item["artifact"])),
        "schemas": [_schema_pin(schema) for schema in schemas],
    }
    return _surface(payload, complete=bool(harness_payload and schemas))


def _surface(payload: dict[str, object], *, complete: bool) -> dict[str, object]:
    return {
        **payload,
        "complete": complete,
        "surface_sha256": _stable_json_hash(payload),
    }


def _base_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = loaded.artifact
    pin: dict[str, object] = {
        "name": artifact.name,
        "kind": artifact.kind.value,
        "location": artifact.location.to_dict(),
        "provenance": artifact.provenance.to_dict(),
        "metadata": dict(artifact.metadata),
        "source_type": loaded.source_type,
        "pinned": loaded.pinned,
        "resolved": loaded.resolved,
        "actual_sha256": loaded.actual_sha256,
        "manifest_sha256": loaded.manifest_sha256,
        "members": list(loaded.members),
        "loaded_metadata": dict(loaded.metadata),
    }
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _chat_template_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = _require_artifact(loaded.artifact, ChatTemplateArtifact)
    pin = _base_pin(loaded)
    pin.update(
        {
            "template_format": artifact.template_format,
            "roles": list(artifact.roles),
            "add_generation_prompt": artifact.add_generation_prompt,
        }
    )
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _tokenizer_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = _require_artifact(loaded.artifact, TokenizerArtifact)
    pin = _base_pin(loaded)
    pin.update({"family": artifact.family, "added_tokens": list(artifact.added_tokens)})
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _provider_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = _require_artifact(loaded.artifact, ProviderConfigArtifact)
    pin = _base_pin(loaded)
    pin.update({"provider": artifact.provider, "api_family": artifact.api_family})
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _stop_policy_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = _require_artifact(loaded.artifact, StopPolicyArtifact)
    pin = _base_pin(loaded)
    pin.update(
        {
            "stop_sequences": list(artifact.stop_sequences),
            "stop_token_ids": list(artifact.stop_token_ids),
            "include_eos": artifact.include_eos,
            "source_family": artifact.source_family,
        }
    )
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _schema_pin(loaded: LoadedArtifact) -> dict[str, object]:
    artifact = _require_artifact(loaded.artifact, SchemaArtifact)
    pin = _base_pin(loaded)
    pin.update({"dialect": artifact.dialect})
    pin["artifact_pin_sha256"] = _stable_json_hash(pin)
    return pin


def _require_artifact(artifact: Artifact, expected: type[Any]) -> Any:
    if not isinstance(artifact, expected):
        raise TypeError(f"expected {expected.__name__}, got {type(artifact).__name__}")
    return artifact


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationReproducibilityError(f"cannot hash file {path}: {exc}") from exc


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
