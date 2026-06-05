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
from .diagnostics import SourceSpan
from .enterprise import EnterpriseConfigError, EnterpriseSettings, empty_enterprise_settings, enterprise_from_config_mapping
from .policies import PolicyError, VerificationPolicy, empty_policy, load_policy_file, merge_policies, policy_from_config_mapping
from .source import JsonSourceMap, build_json_source_map


class ConfigError(ValueError):
    """Raised when a PromptABI configuration cannot be loaded soundly."""


@dataclass(frozen=True, slots=True)
class ConfigInheritanceSource:
    """One config file participating in an inheritance chain."""

    path: str
    extends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"path": self.path}
        if self.extends:
            data["extends"] = list(self.extends)
        return data


@dataclass(frozen=True, slots=True)
class ProofObligationLineage:
    """Resolved provenance for fields that create proof obligations."""

    obligation: str
    field: str
    source_path: str
    detail: str
    status: str = "inherited"

    def to_dict(self) -> dict[str, str]:
        return {
            "obligation": self.obligation,
            "field": self.field,
            "source_path": self.source_path,
            "detail": self.detail,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """A versioned config object for PromptABI verification sessions."""

    name: str
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_bundle: ArtifactBundle = field(default_factory=ArtifactBundle)
    checks: tuple[str, ...] = ("repository-skeleton",)
    max_context_tokens: int | None = None
    policy: VerificationPolicy = field(default_factory=empty_policy)
    enterprise: EnterpriseSettings = field(default_factory=empty_enterprise_settings)
    inheritance_sources: tuple[ConfigInheritanceSource, ...] = ()
    proof_obligation_lineage: tuple[ProofObligationLineage, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path,
        source_map: JsonSourceMap | None = None,
    ) -> "VerificationConfig":
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
                typed_artifacts.append(
                    artifact_from_config(
                        key,
                        value,
                        base_dir=base_dir,
                        source_span=_artifact_config_span(source_map, key),
                    )
                )
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

        try:
            enterprise = enterprise_from_config_mapping(data, base_dir=base_dir)
        except EnterpriseConfigError as exc:
            raise ConfigError(str(exc)) from exc

        try:
            policy = policy_from_config_mapping(data, base_dir=base_dir, source_map=source_map)
            if enterprise.policy_packs:
                policy = merge_policies(
                    policy,
                    *(load_policy_file(pack.path) for pack in enterprise.policy_packs),
                )
        except PolicyError as exc:
            raise ConfigError(str(exc)) from exc

        return cls(
            name=name.strip(),
            artifacts=artifacts,
            artifact_bundle=artifact_bundle,
            checks=checks,
            max_context_tokens=raw_max_context,
            policy=policy,
            enterprise=enterprise,
            proof_obligation_lineage=_local_proof_obligation_lineage(
                data,
                artifact_bundle=artifact_bundle,
                checks=checks,
                source_path=str(base_dir / "<mapping>"),
                status="declared",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "checks": list(self.checks),
            "artifacts": dict(sorted(self.artifacts.items())),
            "artifact_bundle": self.artifact_bundle.to_dict(),
            "max_context_tokens": self.max_context_tokens,
        }
        if self.policy.active:
            data["policy"] = self.policy.to_dict()
        if self.enterprise.active:
            data["enterprise"] = self.enterprise.to_dict()
        if self.inheritance_sources:
            data["inheritance"] = {
                "sources": [source.to_dict() for source in self.inheritance_sources],
                "proof_obligations": [item.to_dict() for item in self.proof_obligation_lineage],
            }
        return data

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
            policy=self.policy,
            enterprise=self.enterprise,
            inheritance_sources=self.inheritance_sources,
            proof_obligation_lineage=(
                *self.proof_obligation_lineage,
                *(
                    ProofObligationLineage(
                        obligation=_obligation_for_artifact(artifact),
                        field=f"artifacts.{name}",
                        source_path=str(base_dir / "<cli-override>"),
                        detail=f"{name} ({artifact.kind.value}) supplied by CLI override",
                        status="override",
                    )
                    for name, artifact in sorted(artifacts_by_name.items())
                    if name in overrides
                ),
            ),
        )


def load_config(path: str | Path) -> VerificationConfig:
    """Load a JSON PromptABI config file from disk."""

    config_path = Path(path).expanduser()
    return _load_config_resolved(config_path, stack=())


def _load_config_resolved(path: Path, *, stack: tuple[Path, ...]) -> VerificationConfig:
    config_path = path.resolve()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ConfigError(f"config inheritance cycle detected: {cycle}")
    try:
        text = config_path.read_text(encoding="utf-8")
        raw = json.loads(text)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"config file is not valid JSON at {config_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    try:
        source_map = build_json_source_map(text, config_path)
    except ValueError as exc:
        raise ConfigError(f"config source map could not be built: {exc}") from exc
    parents = tuple(
        _load_config_resolved(parent, stack=(*stack, config_path))
        for parent in _extends_paths(raw.get("extends", ()), base_dir=config_path.parent)
    )
    current = VerificationConfig.from_mapping(raw, base_dir=config_path.parent, source_map=source_map)
    current_lineage = _local_proof_obligation_lineage(
        raw,
        artifact_bundle=current.artifact_bundle,
        checks=current.checks,
        source_path=str(config_path),
        status="declared",
    )
    current = VerificationConfig(
        name=current.name,
        artifacts=current.artifacts,
        artifact_bundle=current.artifact_bundle,
        checks=current.checks,
        max_context_tokens=current.max_context_tokens,
        policy=current.policy,
        enterprise=current.enterprise,
        proof_obligation_lineage=current_lineage,
    )
    if not parents:
        return current
    return _merge_inherited_configs(current, parents=parents, source_path=config_path, extends=raw.get("extends", ()))


def _artifact_config_span(source_map: JsonSourceMap | None, name: str) -> SourceSpan | None:
    if source_map is None:
        return None
    return source_map.span_for(("artifacts", name)) or source_map.key_span_for(("artifacts", name))


def _extends_paths(raw: Any, *, base_dir: Path) -> tuple[Path, ...]:
    if raw in (None, (), []):
        return ()
    if isinstance(raw, str):
        raw_paths = (raw,)
    elif isinstance(raw, list) and all(isinstance(item, str) and item for item in raw):
        raw_paths = tuple(raw)
    else:
        raise ConfigError("config field 'extends' must be a string or list of non-empty strings")
    return tuple((base_dir / item).expanduser().resolve() for item in raw_paths)


def _merge_inherited_configs(
    current: VerificationConfig,
    *,
    parents: tuple[VerificationConfig, ...],
    source_path: Path,
    extends: Any,
) -> VerificationConfig:
    artifacts_by_name: dict[str, Artifact] = {}
    for parent in parents:
        artifacts_by_name.update({artifact.name: artifact for artifact in parent.artifact_bundle})
    parent_artifact_names = set(artifacts_by_name)
    child_artifacts = {artifact.name: artifact for artifact in current.artifact_bundle}
    artifacts_by_name.update(child_artifacts)
    artifact_bundle = ArtifactBundle(tuple(artifacts_by_name.values()))

    checks = tuple(sorted(dict.fromkeys(check for config in (*parents, current) for check in config.checks)))
    max_context_tokens = (
        current.max_context_tokens
        if current.max_context_tokens is not None
        else next((parent.max_context_tokens for parent in reversed(parents) if parent.max_context_tokens is not None), None)
    )
    enterprise = _merge_enterprise(*(parent.enterprise for parent in parents), current.enterprise)
    policy = merge_policies(*(parent.policy for parent in parents), current.policy)

    extends_tuple = tuple(str(path) for path in _extends_paths(extends, base_dir=source_path.parent))
    inheritance_sources = (
        *(source for parent in parents for source in parent.inheritance_sources),
        *(ConfigInheritanceSource(path=str(Path(parent_path).resolve())) for parent_path in extends_tuple),
        ConfigInheritanceSource(path=str(source_path), extends=extends_tuple),
    )
    lineage = [
        *(lineage for parent in parents for lineage in parent.proof_obligation_lineage),
        *_override_lineage(current.proof_obligation_lineage, parent_artifact_names=parent_artifact_names, child_artifacts=child_artifacts),
    ]

    return VerificationConfig(
        name=current.name,
        artifacts=local_artifact_paths(artifact_bundle),
        artifact_bundle=artifact_bundle,
        checks=checks,
        max_context_tokens=max_context_tokens,
        policy=policy,
        enterprise=enterprise,
        inheritance_sources=tuple(_dedupe_sources(inheritance_sources)),
        proof_obligation_lineage=tuple(lineage),
    )


def _override_lineage(
    current_lineage: tuple[ProofObligationLineage, ...],
    *,
    parent_artifact_names: set[str],
    child_artifacts: dict[str, Artifact],
) -> tuple[ProofObligationLineage, ...]:
    overridden = set(parent_artifact_names & set(child_artifacts))
    rows: list[ProofObligationLineage] = []
    for item in current_lineage:
        artifact_name = item.field.removeprefix("artifacts.") if item.field.startswith("artifacts.") else None
        status = "override" if artifact_name in overridden else item.status
        rows.append(
            ProofObligationLineage(
                obligation=item.obligation,
                field=item.field,
                source_path=item.source_path,
                detail=item.detail,
                status=status,
            )
        )
    return tuple(rows)


def _dedupe_sources(sources: tuple[ConfigInheritanceSource, ...]) -> tuple[ConfigInheritanceSource, ...]:
    by_path: dict[str, ConfigInheritanceSource] = {}
    for source in sources:
        if source.path not in by_path:
            by_path[source.path] = source
    return tuple(by_path.values())


def _local_proof_obligation_lineage(
    data: dict[str, Any],
    *,
    artifact_bundle: ArtifactBundle,
    checks: tuple[str, ...],
    source_path: str,
    status: str,
) -> tuple[ProofObligationLineage, ...]:
    rows: list[ProofObligationLineage] = []
    raw_artifacts = data.get("artifacts", {})
    if isinstance(raw_artifacts, dict):
        for artifact in artifact_bundle:
            if artifact.name in raw_artifacts:
                rows.append(
                    ProofObligationLineage(
                        obligation=_obligation_for_artifact(artifact),
                        field=f"artifacts.{artifact.name}",
                        source_path=source_path,
                        detail=f"{artifact.name} ({artifact.kind.value})",
                        status=status,
                    )
                )
    if "max_context_tokens" in data:
        rows.append(
            ProofObligationLineage(
                obligation="prompt-segment-budget",
                field="max_context_tokens",
                source_path=source_path,
                detail=f"max_context_tokens={data['max_context_tokens']}",
                status=status,
            )
        )
    raw_checks = data.get("checks", [])
    if isinstance(raw_checks, list):
        for check in sorted({item for item in raw_checks if isinstance(item, str)}):
            rows.append(
                ProofObligationLineage(
                    obligation=_obligation_for_check(check),
                    field=f"checks.{check}",
                    source_path=source_path,
                    detail=check,
                    status=status,
                )
            )
    for field_name, obligation in (("policy", "policy-pack-semantic-preservation"), ("policy_files", "policy-pack-semantic-preservation"), ("enterprise", "enterprise-proof-environment")):
        if field_name in data:
            rows.append(
                ProofObligationLineage(
                    obligation=obligation,
                    field=field_name,
                    source_path=source_path,
                    detail=f"{field_name} contributes verification constraints",
                    status=status,
                )
            )
    return tuple(rows)


def _obligation_for_artifact(artifact: Artifact) -> str:
    mapping = {
        "chat-template": "role-region-nonforgeability",
        "special-token-map": "role-region-nonforgeability",
        "stop-policy": "stop-control-token-collision",
        "prompt-segment": "prompt-segment-budget",
        "framework-truncation-config": "prompt-segment-budget",
        "tool-definition": "tool-schema-precondition-satisfiability",
        "provider-config": "tool-provider-mismatch",
        "training-manifest": "training-target-role-alignment",
        "static-contract": "z3-backed-finite-contract",
        "schema": "grammar-tokenizer-emptiness",
        "grammar": "grammar-tokenizer-emptiness",
        "tokenizer": "tokenizer-contract-product",
        "prompt-pack": "prompt-pack-compositional-contract",
    }
    return mapping.get(artifact.kind.value, f"{artifact.kind.value}-obligation")


def _obligation_for_check(check: str) -> str:
    if "budget" in check or "truncation" in check:
        return "prompt-segment-budget"
    if "role" in check or "boundary" in check:
        return "role-region-nonforgeability"
    if "stop" in check:
        return "stop-control-token-collision"
    if "tool" in check:
        return "tool-schema-precondition-satisfiability"
    if "static" in check or "smt" in check or "z3" in check:
        return "z3-backed-finite-contract"
    return check


def _merge_enterprise(*settings: EnterpriseSettings) -> EnterpriseSettings:
    active = tuple(setting for setting in settings if setting.active)
    if not active:
        return empty_enterprise_settings()
    last = active[-1]
    return EnterpriseSettings(
        strict_no_network=any(setting.strict_no_network for setting in active),
        offline_mirrors=tuple(item for setting in active for item in setting.offline_mirrors),
        private_artifact_indexes=tuple(item for setting in active for item in setting.private_artifact_indexes),
        internal_prompt_packs=tuple(item for setting in active for item in setting.internal_prompt_packs),
        internal_provider_fixtures=tuple(item for setting in active for item in setting.internal_provider_fixtures),
        policy_packs=tuple(item for setting in active for item in setting.policy_packs),
        access_control=last.access_control,
        solver_sandbox=last.solver_sandbox,
    )


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
