"""Model-registry publication manifests for PromptABI verification evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .integration_api import IntegrationReport, build_integration_report


MODEL_REGISTRY_MANIFEST_VERSION = "promptabi.model-registry.v1"


class ModelRegistryKind(StrEnum):
    """Supported registry families for PromptABI publication examples."""

    HUGGING_FACE_HUB = "hugging-face-hub"
    INTERNAL = "internal-registry"
    MLFLOW = "mlflow-style"
    ARTIFACT_REPOSITORY = "artifact-repository"


class ModelRegistryError(ValueError):
    """Raised when model-registry publication evidence is invalid."""


@dataclass(frozen=True, slots=True)
class ModelRegistryTarget:
    """One registry target that will receive PromptABI evidence."""

    kind: ModelRegistryKind
    name: str
    model_ref: str
    evidence_path: str
    instructions: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, index: int) -> "ModelRegistryTarget":
        kind = _required_enum(data, "kind", ModelRegistryKind, f"targets[{index}]")
        name = _required_str(data, "name", f"targets[{index}]")
        model_ref = _required_str(data, "model_ref", f"targets[{index}]")
        evidence_path = _required_str(data, "evidence_path", f"targets[{index}]")
        instructions = _string_sequence(data.get("instructions", ()), f"targets[{index}].instructions")
        if not instructions:
            raise ModelRegistryError(f"targets[{index}].instructions must contain at least one command")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        ):
            raise ModelRegistryError(f"targets[{index}].metadata must be an object of strings")
        return cls(
            kind=kind,
            name=name,
            model_ref=model_ref,
            evidence_path=evidence_path,
            instructions=tuple(instructions),
            metadata=dict(sorted(metadata.items())),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "evidence_path": self.evidence_path,
            "instructions": list(self.instructions),
            "kind": self.kind.value,
            "model_ref": self.model_ref,
            "name": self.name,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ModelRegistryPublication:
    """A deterministic model-registry evidence manifest."""

    source_config: str
    report: IntegrationReport
    targets: tuple[ModelRegistryTarget, ...]
    checks_required: tuple[str, ...]
    publish_blockers: tuple[str, ...] = ()
    manifest_version: str = MODEL_REGISTRY_MANIFEST_VERSION

    @property
    def ok(self) -> bool:
        return not self.publish_blockers

    def to_dict(self) -> dict[str, object]:
        registry_surface = self.report.surfaces.get("model-registry", {})
        return {
            "checks_required": list(self.checks_required),
            "gate": self.report.gate.value,
            "manifest_version": self.manifest_version,
            "ok": self.ok,
            "protocol": self.report.to_dict()["protocol"],
            "publish_blockers": list(self.publish_blockers),
            "registry_evidence": registry_surface,
            "source_config": self.source_config,
            "targets": [target.to_dict() for target in self.targets],
        }


def load_model_registry_targets(path: str | Path) -> tuple[ModelRegistryTarget, ...]:
    """Load a registry target manifest."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelRegistryError(f"model-registry manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelRegistryError(
            f"model-registry manifest is not valid JSON at {manifest_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ModelRegistryError("model-registry manifest root must be an object")
    if raw.get("manifest_version") != MODEL_REGISTRY_MANIFEST_VERSION:
        raise ModelRegistryError(
            f"unsupported model-registry manifest version: {raw.get('manifest_version')!r}"
        )
    targets = raw.get("targets")
    if not isinstance(targets, list):
        raise ModelRegistryError("model-registry manifest field 'targets' must be a list")
    loaded = tuple(
        ModelRegistryTarget.from_mapping(target, index=index)
        for index, target in enumerate(targets)
        if isinstance(target, Mapping)
    )
    if len(loaded) != len(targets):
        raise ModelRegistryError("each model-registry target must be an object")
    if not loaded:
        raise ModelRegistryError("model-registry manifest must contain at least one target")
    return loaded


def build_model_registry_publication(
    config: str | Path,
    *,
    targets: Sequence[ModelRegistryTarget] | str | Path,
    fail_on: str = "error",
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "model-registry",
    workspace_root: str | Path | None = None,
) -> ModelRegistryPublication:
    """Run PromptABI and build registry-specific publication evidence."""

    config_path = Path(config).expanduser().resolve()
    registry_targets = load_model_registry_targets(targets) if isinstance(targets, (str, Path)) else tuple(targets)
    if not registry_targets:
        raise ModelRegistryError("at least one model-registry target is required")
    report = build_integration_report(
        config_path,
        surfaces=["model-registry"],
        fail_on=fail_on,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        workspace_root=workspace_root,
    )
    blockers = _publish_blockers(report)
    return ModelRegistryPublication(
        source_config=_relative_to_cwd(config_path),
        report=report,
        targets=tuple(sorted(registry_targets, key=lambda target: (target.kind.value, target.name))),
        checks_required=tuple(surface.value for surface in report.request.surfaces),
        publish_blockers=blockers,
    )


def render_model_registry_publication_json(publication: ModelRegistryPublication) -> str:
    """Render registry publication evidence as deterministic JSON."""

    return json.dumps(publication.to_dict(), indent=2, sort_keys=True) + "\n"


def render_model_registry_publication_text(publication: ModelRegistryPublication) -> str:
    """Render a compact registry publication checklist."""

    lines = [
        "PromptABI model-registry publication",
        f"status: {'PASS' if publication.ok else 'FAIL'}",
        f"config: {publication.source_config}",
        f"gate: {publication.report.gate.value}",
        f"targets: {len(publication.targets)}",
    ]
    for target in publication.targets:
        lines.append(f"- {target.kind.value} {target.name}: {target.model_ref}")
        lines.append(f"  evidence: {target.evidence_path}")
        lines.extend(f"  publish: {instruction}" for instruction in target.instructions)
    if publication.publish_blockers:
        lines.append("blockers:")
        lines.extend(f"- {blocker}" for blocker in publication.publish_blockers)
    return "\n".join(lines) + "\n"


def _publish_blockers(report: IntegrationReport) -> tuple[str, ...]:
    blockers: list[str] = []
    if not report.ok:
        blockers.append("PromptABI verification has error diagnostics")
    registry_surface = report.surfaces.get("model-registry")
    if not isinstance(registry_surface, Mapping):
        blockers.append("model-registry integration surface is missing")
        return tuple(blockers)
    signed_bundle = registry_surface.get("signed_bundle")
    if not isinstance(signed_bundle, Mapping) or signed_bundle.get("available") is not True:
        blockers.append("signed verification bundle evidence is required for registry publication")
    if not registry_surface.get("reproducibility_hash"):
        blockers.append("reproducibility hash is required for registry publication")
    return tuple(blockers)


def _required_str(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _required_enum(
    data: Mapping[str, Any],
    key: str,
    enum_type: type[ModelRegistryKind],
    context: str,
) -> ModelRegistryKind:
    value = _required_str(data, key, context)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ModelRegistryError(f"{context}.{key} must be one of: {allowed}") from exc


def _string_sequence(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ModelRegistryError(f"{context} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _relative_to_cwd(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
