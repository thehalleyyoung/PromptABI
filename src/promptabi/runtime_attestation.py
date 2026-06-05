"""Runtime attestation hooks for services running verified PromptABI contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._version import __version__
from .bundles import _stable_json_hash
from .integration_api import IntegrationArtifactSummary, IntegrationGate, build_integration_report


RUNTIME_ATTESTATION_MANIFEST_VERSION = "promptabi.runtime-attestation.v1"
RUNTIME_CONTRACT_FAMILIES: tuple[str, ...] = ("prompt", "tokenizer", "template", "schema", "provider")


class RuntimeAttestationError(ValueError):
    """Raised when runtime attestation hooks cannot be built from verified evidence."""


class RuntimeAttestationHookKind(StrEnum):
    """Service surfaces that can report PromptABI runtime attestation evidence."""

    ENV_FILE = "env-file"
    HTTP_JSON = "http-json"
    KUBERNETES_ANNOTATIONS = "kubernetes-annotations"
    OPENTELEMETRY_ATTRIBUTES = "opentelemetry-attributes"


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    """One verified prompt-interface contract a service declares at runtime."""

    name: str
    kind: str
    family: str
    contract_hash: str
    artifact: Mapping[str, object]
    runtime_ref: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact": dict(self.artifact),
            "contract_hash": self.contract_hash,
            "family": self.family,
            "kind": self.kind,
            "name": self.name,
        }
        if self.runtime_ref is not None:
            payload["runtime_ref"] = self.runtime_ref
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeAttestationHook:
    """One rendered hook payload a service can expose at runtime."""

    kind: RuntimeAttestationHookKind
    path: str
    description: str
    content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "description": self.description,
            "kind": self.kind.value,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class RuntimeAttestationReport:
    """Deterministic runtime attestation manifest for a verified PromptABI config."""

    service: str
    environment: str
    source_config: str
    gate: IntegrationGate
    ok: bool
    bundle_hash: str
    reproducibility_hash: str
    signing_key_id: str
    revision: str | None
    instance_id: str | None
    contracts: tuple[RuntimeContract, ...]
    hooks: tuple[RuntimeAttestationHook, ...]
    blockers: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_version": RUNTIME_ATTESTATION_MANIFEST_VERSION,
            "promptabi_version": __version__,
            "service": self.service,
            "environment": self.environment,
            "source_config": self.source_config,
            "gate": self.gate.value,
            "ok": self.ok,
            "bundle": {
                "bundle_hash": self.bundle_hash,
                "reproducibility_hash": self.reproducibility_hash,
                "signing_key_id": self.signing_key_id,
            },
            "contract_families": _family_counts(self.contracts),
            "contracts": [contract.to_dict() for contract in self.contracts],
            "hooks": [hook.to_dict() for hook in self.hooks],
            "blockers": list(self.blockers),
            "manifest_sha256": self.manifest_sha256,
        }
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.instance_id is not None:
            payload["instance_id"] = self.instance_id
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeAttestationWriteResult:
    """Files written by the runtime-attestation hook writer."""

    output_dir: Path
    report: RuntimeAttestationReport
    written_files: tuple[Path, ...]


def build_runtime_attestation_report(
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "runtime-attestation",
    service: str = "promptabi-service",
    environment: str = "production",
    revision: str | None = None,
    instance_id: str | None = None,
    runtime_contract_refs: Mapping[str, str] | None = None,
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
) -> RuntimeAttestationReport:
    """Build service runtime hooks from a live verification run and signed bundle evidence."""

    if bundle_key is None:
        raise RuntimeAttestationError("runtime attestation requires a bundle signing key")
    service = _required_label(service, "service")
    environment = _required_label(environment, "environment")
    refs = dict(sorted((runtime_contract_refs or {}).items()))
    config_path = Path(config).expanduser().resolve()
    report = build_integration_report(
        config_path,
        surfaces=("model-registry", "internal-platform"),
        fail_on=fail_on,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        workspace_root=workspace_root,
    )
    registry_surface = _mapping(report.surfaces.get("model-registry"), "model-registry surface")
    signed_bundle = _mapping(registry_surface.get("signed_bundle"), "signed bundle evidence")
    if signed_bundle.get("available") is not True:
        reason = signed_bundle.get("reason", "signed bundle evidence is unavailable")
        raise RuntimeAttestationError(f"runtime attestation requires current signed bundle evidence ({reason})")
    bundle_hash = _required_str(signed_bundle, "bundle_hash")
    signing_key_id = _required_str(signed_bundle, "signing_key_id")
    reproducibility_hash = _required_str(registry_surface, "reproducibility_hash")
    contracts = tuple(
        sorted(
            (_contract_from_summary(summary, runtime_ref=refs.get(summary.name)) for summary in report.artifacts),
            key=lambda contract: (contract.family, contract.kind, contract.name),
        )
    )
    blockers = _attestation_blockers(report.ok, report.gate, contracts)
    source_config = report.request.config_path or _relative_to_cwd(config_path)
    base_payload = _base_attestation_payload(
        service=service,
        environment=environment,
        source_config=source_config,
        gate=report.gate,
        ok=not blockers,
        bundle_hash=bundle_hash,
        reproducibility_hash=reproducibility_hash,
        signing_key_id=signing_key_id,
        revision=revision,
        instance_id=instance_id,
        contracts=contracts,
        blockers=blockers,
    )
    hooks = _build_hooks(base_payload, contracts)
    manifest_sha256 = _stable_json_hash({**base_payload, "hooks": [hook.to_dict() for hook in hooks]})
    return RuntimeAttestationReport(
        service=service,
        environment=environment,
        source_config=source_config,
        gate=report.gate,
        ok=not blockers,
        bundle_hash=bundle_hash,
        reproducibility_hash=reproducibility_hash,
        signing_key_id=signing_key_id,
        revision=revision,
        instance_id=instance_id,
        contracts=contracts,
        hooks=hooks,
        blockers=blockers,
        manifest_sha256=manifest_sha256,
    )


def render_runtime_attestation_json(report: RuntimeAttestationReport) -> str:
    """Render runtime attestation as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_runtime_attestation_text(report: RuntimeAttestationReport) -> str:
    """Render a compact runtime-attestation summary."""

    lines = [
        "PromptABI runtime attestation",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"service: {report.service}",
        f"environment: {report.environment}",
        f"config: {report.source_config}",
        f"gate: {report.gate.value}",
        f"bundle_hash: {report.bundle_hash}",
        f"reproducibility_hash: {report.reproducibility_hash}",
        "contract_families: "
        + ", ".join(f"{family}={count}" for family, count in _family_counts(report.contracts).items()),
        f"hooks: {len(report.hooks)}",
    ]
    for hook in report.hooks:
        lines.append(f"- {hook.kind.value}: {hook.path}")
    if report.blockers:
        lines.append("blockers:")
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    return "\n".join(lines) + "\n"


def write_runtime_attestation_hooks(
    output_dir: str | Path,
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "runtime-attestation",
    service: str = "promptabi-service",
    environment: str = "production",
    revision: str | None = None,
    instance_id: str | None = None,
    runtime_contract_refs: Mapping[str, str] | None = None,
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
    force: bool = False,
) -> RuntimeAttestationWriteResult:
    """Write runtime attestation hooks and a deterministic manifest."""

    destination = Path(output_dir)
    report = build_runtime_attestation_report(
        config,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        service=service,
        environment=environment,
        revision=revision,
        instance_id=instance_id,
        runtime_contract_refs=runtime_contract_refs,
        fail_on=fail_on,
        workspace_root=workspace_root,
    )
    filenames = tuple(["runtime-attestation.json", *(hook.path for hook in report.hooks)])
    _prepare_output_dir(destination, expected_filenames=filenames, force=force)
    written: list[Path] = []
    manifest_path = destination / "runtime-attestation.json"
    manifest_path.write_text(render_runtime_attestation_json(report), encoding="utf-8")
    written.append(manifest_path)
    for hook in report.hooks:
        path = destination / hook.path
        path.write_text(hook.content, encoding="utf-8")
        written.append(path)
    return RuntimeAttestationWriteResult(output_dir=destination, report=report, written_files=tuple(written))


def render_runtime_attestation_write_summary(result: RuntimeAttestationWriteResult) -> str:
    return (
        "PromptABI runtime attestation\n"
        f"output: {result.output_dir}\n"
        f"files: {len(result.written_files)}\n"
        f"bundle_hash: {result.report.bundle_hash}\n"
        f"reproducibility_hash: {result.report.reproducibility_hash}\n"
        f"manifest: {result.output_dir / 'runtime-attestation.json'}\n"
    )


def runtime_contract_refs_from_cli(values: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Parse repeated NAME=REF runtime-contract bindings."""

    refs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeAttestationError("--runtime-contract values must be NAME=REF")
        name, ref = value.split("=", 1)
        name = name.strip()
        ref = ref.strip()
        if not name or not ref:
            raise RuntimeAttestationError("--runtime-contract values must include non-empty NAME and REF")
        refs[name] = ref
    return refs


def _contract_from_summary(summary: IntegrationArtifactSummary, *, runtime_ref: str | None) -> RuntimeContract:
    artifact = summary.to_dict()
    return RuntimeContract(
        name=summary.name,
        kind=summary.kind,
        family=_contract_family(summary.kind),
        contract_hash=_stable_json_hash(artifact),
        artifact=artifact,
        runtime_ref=runtime_ref,
    )


def _contract_family(kind: str) -> str:
    if kind in {"prompt-segment", "prompt-pack", "static-contract"}:
        return "prompt"
    if kind in {"tokenizer", "special-token-map"}:
        return "tokenizer"
    if kind in {"chat-template"}:
        return "template"
    if kind in {"schema", "grammar", "tool-definition", "stop-policy"}:
        return "schema"
    if kind in {"provider-config", "evaluation-harness", "training-manifest"}:
        return "provider"
    return "prompt"


def _family_counts(contracts: tuple[RuntimeContract, ...]) -> dict[str, int]:
    return {family: sum(1 for contract in contracts if contract.family == family) for family in RUNTIME_CONTRACT_FAMILIES}


def _build_hooks(base_payload: Mapping[str, object], contracts: tuple[RuntimeContract, ...]) -> tuple[RuntimeAttestationHook, ...]:
    env = _env_content(base_payload, contracts)
    http = json.dumps(base_payload, indent=2, sort_keys=True) + "\n"
    k8s = _kubernetes_annotations(base_payload, contracts)
    otel = _otel_attributes(base_payload, contracts)
    return (
        RuntimeAttestationHook(
            RuntimeAttestationHookKind.ENV_FILE,
            "runtime-attestation.env",
            "Source this file or project it as environment variables in the service runtime.",
            env,
        ),
        RuntimeAttestationHook(
            RuntimeAttestationHookKind.HTTP_JSON,
            "well-known-promptabi-attestation.json",
            "Serve this JSON at /.well-known/promptabi-attestation for runtime inventory.",
            http,
        ),
        RuntimeAttestationHook(
            RuntimeAttestationHookKind.KUBERNETES_ANNOTATIONS,
            "kubernetes-annotations.yaml",
            "Attach these annotations to Pods or Deployments that run the verified contract.",
            k8s,
        ),
        RuntimeAttestationHook(
            RuntimeAttestationHookKind.OPENTELEMETRY_ATTRIBUTES,
            "opentelemetry-attributes.json",
            "Attach these resource attributes to traces, metrics, or logs emitted by the service.",
            otel,
        ),
    )


def _base_attestation_payload(
    *,
    service: str,
    environment: str,
    source_config: str,
    gate: IntegrationGate,
    ok: bool,
    bundle_hash: str,
    reproducibility_hash: str,
    signing_key_id: str,
    revision: str | None,
    instance_id: str | None,
    contracts: tuple[RuntimeContract, ...],
    blockers: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_version": RUNTIME_ATTESTATION_MANIFEST_VERSION,
        "promptabi_version": __version__,
        "service": service,
        "environment": environment,
        "source_config": source_config,
        "gate": gate.value,
        "ok": ok,
        "bundle": {
            "bundle_hash": bundle_hash,
            "reproducibility_hash": reproducibility_hash,
            "signing_key_id": signing_key_id,
        },
        "contract_families": _family_counts(contracts),
        "contracts": [contract.to_dict() for contract in contracts],
        "blockers": list(blockers),
    }
    if revision is not None:
        payload["revision"] = revision
    if instance_id is not None:
        payload["instance_id"] = instance_id
    return payload


def _env_content(base_payload: Mapping[str, object], contracts: tuple[RuntimeContract, ...]) -> str:
    bundle = _mapping(base_payload["bundle"], "bundle")
    values = {
        "PROMPTABI_ATTESTATION_VERSION": RUNTIME_ATTESTATION_MANIFEST_VERSION,
        "PROMPTABI_SERVICE": str(base_payload["service"]),
        "PROMPTABI_ENVIRONMENT": str(base_payload["environment"]),
        "PROMPTABI_GATE": str(base_payload["gate"]),
        "PROMPTABI_OK": "true" if base_payload["ok"] else "false",
        "PROMPTABI_BUNDLE_HASH": _required_str(bundle, "bundle_hash"),
        "PROMPTABI_REPRODUCIBILITY_HASH": _required_str(bundle, "reproducibility_hash"),
        "PROMPTABI_SIGNING_KEY_ID": _required_str(bundle, "signing_key_id"),
        "PROMPTABI_CONTRACT_COUNT": str(len(contracts)),
    }
    for contract in contracts:
        prefix = f"PROMPTABI_CONTRACT_{_env_key(contract.name)}"
        values[f"{prefix}_FAMILY"] = contract.family
        values[f"{prefix}_KIND"] = contract.kind
        values[f"{prefix}_HASH"] = contract.contract_hash
        if contract.runtime_ref is not None:
            values[f"{prefix}_RUNTIME_REF"] = contract.runtime_ref
    return "".join(f"{key}={json.dumps(value)}\n" for key, value in sorted(values.items()))


def _kubernetes_annotations(base_payload: Mapping[str, object], contracts: tuple[RuntimeContract, ...]) -> str:
    bundle = _mapping(base_payload["bundle"], "bundle")
    annotations = {
        "promptabi.dev/attestation-version": RUNTIME_ATTESTATION_MANIFEST_VERSION,
        "promptabi.dev/service": str(base_payload["service"]),
        "promptabi.dev/environment": str(base_payload["environment"]),
        "promptabi.dev/gate": str(base_payload["gate"]),
        "promptabi.dev/bundle-hash": _required_str(bundle, "bundle_hash"),
        "promptabi.dev/reproducibility-hash": _required_str(bundle, "reproducibility_hash"),
        "promptabi.dev/signing-key-id": _required_str(bundle, "signing_key_id"),
    }
    for contract in contracts:
        annotations[f"promptabi.dev/contract-{_annotation_name(contract.name)}"] = contract.contract_hash
    lines = ["metadata:", "  annotations:"]
    lines.extend(f"    {key}: {json.dumps(value)}" for key, value in sorted(annotations.items()))
    return "\n".join(lines) + "\n"


def _otel_attributes(base_payload: Mapping[str, object], contracts: tuple[RuntimeContract, ...]) -> str:
    bundle = _mapping(base_payload["bundle"], "bundle")
    attributes: dict[str, str | int | bool] = {
        "promptabi.attestation.version": RUNTIME_ATTESTATION_MANIFEST_VERSION,
        "promptabi.service": str(base_payload["service"]),
        "promptabi.environment": str(base_payload["environment"]),
        "promptabi.gate": str(base_payload["gate"]),
        "promptabi.ok": bool(base_payload["ok"]),
        "promptabi.bundle_hash": _required_str(bundle, "bundle_hash"),
        "promptabi.reproducibility_hash": _required_str(bundle, "reproducibility_hash"),
        "promptabi.signing_key_id": _required_str(bundle, "signing_key_id"),
        "promptabi.contract_count": len(contracts),
    }
    for family, count in _family_counts(contracts).items():
        attributes[f"promptabi.contract_family.{family}.count"] = count
    return json.dumps(attributes, indent=2, sort_keys=True) + "\n"


def _attestation_blockers(
    ok: bool,
    gate: IntegrationGate,
    contracts: tuple[RuntimeContract, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not ok:
        blockers.append("PromptABI verification has error diagnostics")
    if gate is IntegrationGate.FAIL:
        blockers.append("integration gate is fail")
    if not contracts:
        blockers.append("no prompt-interface contracts were available to attest")
    return tuple(dict.fromkeys(blockers))


def _prepare_output_dir(destination: Path, *, expected_filenames: tuple[str, ...], force: bool) -> None:
    if destination.exists():
        existing = [destination / filename for filename in expected_filenames if (destination / filename).exists()]
        if existing and not force:
            names = ", ".join(path.name for path in existing)
            raise RuntimeAttestationError(f"runtime-attestation files already exist ({names}); pass --force to overwrite")
    destination.mkdir(parents=True, exist_ok=True)


def _required_label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeAttestationError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeAttestationError(f"{context} is missing from integration evidence")
    return value


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeAttestationError(f"runtime attestation evidence field {key!r} is required")
    return value


def _env_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", value).upper()
    key = re.sub(r"_+", "_", key).strip("_")
    if not key or not re.match(r"^[A-Z_]", key):
        key = f"CONTRACT_{key}"
    return key[:80]


def _annotation_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9-]", "-", value.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    if not name or not re.match(r"^[a-z0-9]", name):
        name = f"contract-{name}"
    return name[:63].rstrip("-") or "contract"


def _relative_to_cwd(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
