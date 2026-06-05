"""Prompt-pack parsing, package locks, registries, mirrors, and compatibility checks."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
    PromptSegment,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
)
from .budgets import TokenBudgetReport, TokenBudgetSegment
from .diagnostics import (
    ArtifactRef,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    SourceSpan,
    WitnessStep,
    WitnessTrace,
)
from .loaders import LoadedArtifact


PROMPT_PACK_LOCKFILE_VERSION = 1
PROMPT_PACK_LOCK_CHECK_MODES = (CheckMode.SOUND, CheckMode.COMPLETE)
PROMPT_PACK_REGISTRY_VERSION = 1
PROMPT_PACK_MIRROR_VERSION = 1
PROMPT_PACK_PROVENANCE_VERSION = 1
PROMPT_PACK_SIGNATURE_ALGORITHM = "hmac-sha256"


class PromptPackLockError(ValueError):
    """Raised when a prompt-pack lockfile cannot be read or compared."""


class PromptPackMirrorError(ValueError):
    """Raised when a local prompt-pack registry mirror cannot be built or verified."""


class PromptPackProvenanceError(ValueError):
    """Raised when a signed prompt-pack provenance manifest cannot be built or verified."""


class PromptPackFindingKind(StrEnum):
    """Prompt-pack compatibility outcomes."""

    EMPTY_PACK = "empty-pack"
    TEMPLATE_ROLE_MISMATCH = "template-role-mismatch"
    APP_ROLE_MISSING = "app-role-missing"
    TOOL_MISSING = "tool-missing"
    STOP_MISSING = "stop-missing"
    MODEL_FAMILY_UNSUPPORTED = "model-family-unsupported"
    COMPOSITION_CONTEXT_ROLE_UNSUPPORTED = "composition-context-role-unsupported"
    COMPOSITION_REQUIRED_REGION_UNDECLARED = "composition-required-region-undeclared"
    COMPOSITION_REQUIRED_REGION_NOT_MUST_SURVIVE = "composition-required-region-not-must-survive"
    COMPOSITION_REQUIRED_REGION_TRUNCATED = "composition-required-region-truncated"
    COMPOSITION_RAG_UNBOUNDED = "composition-rag-unbounded"
    COMPOSITION_VERIFIED = "composition-verified"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class PromptPackFinding:
    """One prompt-pack finding against a downstream app contract."""

    kind: PromptPackFindingKind
    pack: PromptPackArtifact
    message: str
    template: PromptPackTemplate | None = None
    tool: PromptPackToolSchema | None = None
    stop_policy: PromptPackStopPolicy | None = None
    expected: tuple[str, ...] = ()
    observed: tuple[str, ...] = ()
    span: SourceSpan | None = None
    subject: str | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR


@dataclass(frozen=True, slots=True)
class PromptPackReport:
    """Compatibility report for reusable prompt-pack artifacts."""

    findings: tuple[PromptPackFinding, ...]


@dataclass(frozen=True, slots=True)
class PromptPackLockEntry:
    """Deterministic package-manager-style lock entry for one prompt pack."""

    name: str
    package_name: str
    version: str | None
    location: str
    sha256: str | None
    contract_hash: str
    exported_templates: tuple[tuple[str, str, str], ...]
    tool_schemas: tuple[tuple[str, str | None, str | None, bool], ...]
    stop_policies: tuple[tuple[str, tuple[str, ...], tuple[int, ...], bool], ...]
    supported_model_families: tuple[str, ...]
    expected_roles: tuple[str, ...]
    diagnostic_baseline: tuple[tuple[str, str, str], ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromptPackLockEntry":
        return cls(
            name=_required_str(data, "name"),
            package_name=_required_str(data, "package_name"),
            version=_optional_str(data, "version"),
            location=_required_str(data, "location"),
            sha256=_optional_str(data, "sha256"),
            contract_hash=_required_str(data, "contract_hash"),
            exported_templates=_template_lock_entries(data.get("exported_templates", [])),
            tool_schemas=_tool_lock_entries(data.get("tool_schemas", [])),
            stop_policies=_stop_lock_entries(data.get("stop_policies", [])),
            supported_model_families=_string_tuple(data.get("supported_model_families", []), "supported_model_families"),
            expected_roles=_string_tuple(data.get("expected_roles", []), "expected_roles"),
            diagnostic_baseline=_diagnostic_lock_entries(data.get("diagnostic_baseline", [])),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "contract_hash": self.contract_hash,
            "diagnostic_baseline": [
                {"fingerprint": fingerprint, "rule_id": rule_id, "severity": severity}
                for rule_id, severity, fingerprint in self.diagnostic_baseline
            ],
            "expected_roles": list(self.expected_roles),
            "exported_templates": [
                {
                    "name": name,
                    "roles_hash": roles_hash,
                    "template_hash": template_hash,
                }
                for name, template_hash, roles_hash in self.exported_templates
            ],
            "location": self.location,
            "name": self.name,
            "package_name": self.package_name,
            "stop_policies": [
                {
                    "include_eos": include_eos,
                    "name": name,
                    "stop_sequences": list(stop_sequences),
                    "stop_token_ids": list(stop_token_ids),
                }
                for name, stop_sequences, stop_token_ids, include_eos in self.stop_policies
            ],
            "supported_model_families": list(self.supported_model_families),
            "tool_schemas": [
                {
                    "name": name,
                    "provider": provider,
                    "required": required,
                    "schema_digest": schema_digest,
                }
                for name, provider, schema_digest, required in self.tool_schemas
            ],
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.version is not None:
            data["version"] = self.version
        return data


@dataclass(frozen=True, slots=True)
class PromptPackLockfile:
    """Lockfile that pins reusable prompt-pack packages and verified contracts."""

    entries: tuple[PromptPackLockEntry, ...]
    lockfile_version: int = PROMPT_PACK_LOCKFILE_VERSION

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromptPackLockfile":
        if data.get("lockfile_version") != PROMPT_PACK_LOCKFILE_VERSION:
            raise PromptPackLockError(f"unsupported prompt-pack lockfile version: {data.get('lockfile_version')!r}")
        raw_entries = data.get("prompt_packs")
        if not isinstance(raw_entries, list):
            raise PromptPackLockError("prompt-pack lockfile field 'prompt_packs' must be a list")
        entries: list[PromptPackLockEntry] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                raise PromptPackLockError("prompt-pack lock entries must be objects")
            entries.append(PromptPackLockEntry.from_mapping(item))
        return cls(entries=tuple(sorted(entries, key=lambda item: item.name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "lockfile_version": self.lockfile_version,
            "prompt_packs": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class PromptPackRegistryEntry:
    """Public, privacy-preserving registry entry for a verified prompt pack."""

    name: str
    package_name: str
    version: str | None
    location: str
    sha256: str | None
    contract_hash: str
    proof_hash: str
    supported_fragments: tuple[tuple[str, object], ...]
    reproducible_metadata: tuple[tuple[str, object], ...]
    proofs: tuple[tuple[str, object], ...]
    diagnostics: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "contract_hash": self.contract_hash,
            "diagnostics": [
                {"fingerprint": fingerprint, "rule_id": rule_id, "severity": severity}
                for rule_id, severity, fingerprint in self.diagnostics
            ],
            "location": self.location,
            "name": self.name,
            "package_name": self.package_name,
            "proof_hash": self.proof_hash,
            "proofs": _pairs_to_dict(self.proofs),
            "reproducible_metadata": _pairs_to_dict(self.reproducible_metadata),
            "supported_fragments": _pairs_to_dict(self.supported_fragments),
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.version is not None:
            data["version"] = self.version
        return data


@dataclass(frozen=True, slots=True)
class PromptPackRegistry:
    """Public registry manifest for verified prompt-pack packages."""

    entries: tuple[PromptPackRegistryEntry, ...]
    registry_version: int = PROMPT_PACK_REGISTRY_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_packs": [entry.to_dict() for entry in self.entries],
            "registry_version": self.registry_version,
        }


@dataclass(frozen=True, slots=True)
class PromptPackMirrorEntry:
    """One content-addressed local prompt-pack artifact in an offline mirror."""

    name: str
    package_name: str
    version: str | None
    source_location: str
    mirror_path: str
    sha256: str
    size_bytes: int
    contract_hash: str
    proof_hash: str
    registry_entry: PromptPackRegistryEntry

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "contract_hash": self.contract_hash,
            "mirror_path": self.mirror_path,
            "name": self.name,
            "package_name": self.package_name,
            "proof_hash": self.proof_hash,
            "registry_entry": self.registry_entry.to_dict(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_location": self.source_location,
        }
        if self.version is not None:
            data["version"] = self.version
        return data


@dataclass(frozen=True, slots=True)
class PromptPackMirrorManifest:
    """Manifest for an enterprise-local prompt-pack registry mirror."""

    entries: tuple[PromptPackMirrorEntry, ...]
    mirror_version: int = PROMPT_PACK_MIRROR_VERSION

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromptPackMirrorManifest":
        if data.get("mirror_version") != PROMPT_PACK_MIRROR_VERSION:
            raise PromptPackMirrorError(f"unsupported prompt-pack mirror version: {data.get('mirror_version')!r}")
        raw_entries = data.get("prompt_packs")
        if not isinstance(raw_entries, list):
            raise PromptPackMirrorError("prompt-pack mirror field 'prompt_packs' must be a list")
        entries: list[PromptPackMirrorEntry] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                raise PromptPackMirrorError("prompt-pack mirror entries must be objects")
            entries.append(_mirror_entry_from_mapping(item))
        return cls(entries=tuple(sorted(entries, key=lambda entry: (entry.package_name, entry.name))))

    def to_dict(self) -> dict[str, object]:
        return {
            "mirror_version": self.mirror_version,
            "prompt_packs": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class PromptPackMirrorVerification:
    """Result of validating a local prompt-pack registry mirror."""

    manifest: PromptPackMirrorManifest
    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return all(diagnostic.severity is not DiagnosticSeverity.ERROR for diagnostic in self.diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "mirror": self.manifest.to_dict(),
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class SignedPromptPackProvenance:
    """Signed provenance manifest for reviewed prompt-pack registry metadata."""

    payload: dict[str, object]
    signature: str
    signing_key_id: str
    algorithm: str = PROMPT_PACK_SIGNATURE_ALGORITHM

    @property
    def provenance_hash(self) -> str:
        return _hash_jsonable(self.payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "payload": self.payload,
            "provenance_hash": self.provenance_hash,
            "signature": self.signature,
            "signing_key_id": self.signing_key_id,
        }


@dataclass(frozen=True, slots=True)
class PromptPackProvenanceVerification:
    """Result of checking a signed prompt-pack provenance manifest."""

    ok: bool
    provenance_hash: str
    signing_key_id: str
    expected_signature: str
    actual_signature: str
    package_count: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "actual_signature": self.actual_signature,
            "expected_signature": self.expected_signature,
            "ok": self.ok,
            "package_count": self.package_count,
            "provenance_hash": self.provenance_hash,
            "signing_key_id": self.signing_key_id,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


def analyze_prompt_pack_contracts(
    prompt_pack: PromptPackArtifact,
    artifacts: tuple[Artifact, ...],
    *,
    source_spans: dict[str, SourceSpan] | None = None,
    token_budget_report: TokenBudgetReport | None = None,
) -> PromptPackReport:
    """Check a reusable prompt pack against its own and app-level ABI promises."""

    source_spans = source_spans or {}
    findings: list[PromptPackFinding] = []
    if not prompt_pack.exported_templates:
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.EMPTY_PACK,
                pack=prompt_pack,
                message=f"prompt pack '{prompt_pack.pack_name}' exports no templates",
                span=source_spans.get("exported_templates"),
            )
        )
        return PromptPackReport(tuple(findings))


    expected_roles = set(prompt_pack.expected_roles)
    for template in prompt_pack.exported_templates:
        if expected_roles and not set(template.roles).issubset(expected_roles):
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.TEMPLATE_ROLE_MISMATCH,
                    pack=prompt_pack,
                    template=template,
                    message=f"template '{template.name}' exports roles outside the pack expected role set",
                    expected=tuple(sorted(expected_roles)),
                    observed=template.roles,
                    span=source_spans.get(f"exported_templates.{template.name}.roles"),
                )
            )

    app_roles = _application_roles(artifacts)
    if app_roles and expected_roles:
        missing_roles = tuple(sorted(expected_roles - app_roles))
        if missing_roles:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.APP_ROLE_MISSING,
                    pack=prompt_pack,
                    message=f"downstream app is missing prompt-pack role(s): {', '.join(missing_roles)}",
                    expected=tuple(sorted(expected_roles)),
                    observed=tuple(sorted(app_roles)),
                    span=source_spans.get("expected_roles"),
                )
            )

    app_tools = _application_tools(artifacts)
    if prompt_pack.tool_schemas:
        for tool in prompt_pack.tool_schemas:
            if tool.required and app_tools and tool.name not in app_tools:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.TOOL_MISSING,
                        pack=prompt_pack,
                        tool=tool,
                        message=f"downstream app does not provide required prompt-pack tool '{tool.name}'",
                        expected=(tool.name,),
                        observed=tuple(sorted(app_tools)),
                        span=source_spans.get(f"tool_schemas.{tool.name}"),
                    )
                )
            elif tool.required and not app_tools:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.TOOL_MISSING,
                        pack=prompt_pack,
                        tool=tool,
                        message=f"prompt pack requires tool '{tool.name}' but no downstream tool-definition artifact is configured",
                        expected=(tool.name,),
                        span=source_spans.get(f"tool_schemas.{tool.name}"),
                    )
                )

    app_stops = _application_stop_sequences(artifacts)
    if prompt_pack.stop_policies:
        for stop_policy in prompt_pack.stop_policies:
            missing_stops = tuple(sequence for sequence in stop_policy.stop_sequences if sequence not in app_stops)
            if missing_stops and app_stops:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.STOP_MISSING,
                        pack=prompt_pack,
                        stop_policy=stop_policy,
                        message=f"downstream stop policy omits prompt-pack stop sequence(s): {', '.join(missing_stops)}",
                        expected=stop_policy.stop_sequences,
                        observed=tuple(sorted(app_stops)),
                        span=source_spans.get(f"stop_policies.{stop_policy.name}"),
                    )
                )
            elif stop_policy.stop_sequences and not app_stops:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.STOP_MISSING,
                        pack=prompt_pack,
                        stop_policy=stop_policy,
                        message=f"prompt pack declares stop policy '{stop_policy.name}' but no downstream stop-policy artifact is configured",
                        expected=stop_policy.stop_sequences,
                        span=source_spans.get(f"stop_policies.{stop_policy.name}"),
                    )
                )

    app_model_families = _application_model_families(artifacts)
    supported_families = set(prompt_pack.supported_model_families)
    if app_model_families and supported_families and app_model_families.isdisjoint(supported_families):
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.MODEL_FAMILY_UNSUPPORTED,
                pack=prompt_pack,
                message=f"downstream model/provider family is outside prompt-pack support: {', '.join(sorted(app_model_families))}",
                expected=tuple(sorted(supported_families)),
                observed=tuple(sorted(app_model_families)),
                span=source_spans.get("supported_model_families"),
            )
        )

    findings.extend(
        _prompt_pack_composition_findings(
            prompt_pack,
            artifacts,
            source_spans=source_spans,
            token_budget_report=token_budget_report,
        )
    )

    if not findings:
        if _has_composition_evidence(artifacts, token_budget_report):
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.COMPOSITION_VERIFIED,
                    pack=prompt_pack,
                    message=(
                        f"prompt pack '{prompt_pack.pack_name}' guarantees compose with downstream "
                        "context, RAG, and truncation artifacts"
                    ),
                    expected=tuple(sorted(expected_roles)),
                    observed=tuple(sorted(app_roles)),
                    severity=DiagnosticSeverity.INFO,
                    evidence=_composition_evidence(artifacts, token_budget_report),
                )
            )
        else:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.VERIFIED,
                    pack=prompt_pack,
                    message=(
                        f"prompt pack '{prompt_pack.pack_name}' exports {len(prompt_pack.exported_templates)} "
                        "template contract(s) compatible with configured app artifacts"
                    ),
                    expected=tuple(sorted(expected_roles)),
                    observed=tuple(sorted(app_roles)),
                    severity=DiagnosticSeverity.INFO,
                )
            )
    return PromptPackReport(tuple(findings))


def build_prompt_pack_lockfile(
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    base_dir: str | Path | None = None,
) -> PromptPackLockfile:
    """Build a deterministic lockfile for the configured prompt-pack package set."""

    resolved_base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    entries: list[PromptPackLockEntry] = []
    for loaded in loaded_artifacts:
        if isinstance(loaded.artifact, PromptPackArtifact):
            entries.append(_prompt_pack_lock_entry(loaded, diagnostics, base_dir=resolved_base))
    if not entries:
        raise PromptPackLockError("no prompt-pack artifacts are loaded; cannot write a prompt-pack lockfile")
    return PromptPackLockfile(tuple(sorted(entries, key=lambda entry: entry.name)))


def prompt_pack_lockfile_to_json(lockfile: PromptPackLockfile) -> str:
    return json.dumps(lockfile.to_dict(), indent=2, sort_keys=True) + "\n"


def load_prompt_pack_lockfile(path: str | Path) -> PromptPackLockfile:
    lockfile_path = Path(path)
    try:
        raw = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromptPackLockError(f"prompt-pack lockfile not found: {lockfile_path}") from exc
    except json.JSONDecodeError as exc:
        raise PromptPackLockError(
            f"prompt-pack lockfile is not valid JSON at {lockfile_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise PromptPackLockError("prompt-pack lockfile root must be a JSON object")
    return PromptPackLockfile.from_mapping(raw)


def write_prompt_pack_lockfile(path: str | Path, lockfile: PromptPackLockfile) -> None:
    lockfile_path = Path(path)
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile_path.write_text(prompt_pack_lockfile_to_json(lockfile), encoding="utf-8")


def build_prompt_pack_registry(
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    base_dir: str | Path | None = None,
) -> PromptPackRegistry:
    """Build a public registry manifest with hashes, proofs, and no prompt contents."""

    lockfile = build_prompt_pack_lockfile(loaded_artifacts, diagnostics, base_dir=base_dir)
    entries = tuple(_registry_entry_from_lock_entry(entry) for entry in lockfile.entries)
    return PromptPackRegistry(entries=entries)


def prompt_pack_registry_to_json(registry: PromptPackRegistry) -> str:
    return json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n"


def render_prompt_pack_registry_text(registry: PromptPackRegistry) -> str:
    lines = ["PromptABI prompt-pack registry", f"registry version: {registry.registry_version}"]
    for entry in registry.entries:
        fragments = dict(entry.supported_fragments)
        diagnostics = ", ".join(rule_id for rule_id, _severity, _fingerprint in entry.diagnostics) or "none"
        version = f"@{entry.version}" if entry.version is not None else ""
        lines.extend(
            [
                "",
                f"- {entry.package_name}{version}",
                f"  contract: {entry.contract_hash[:16]} proof: {entry.proof_hash[:16]}",
                f"  source: {entry.location}",
                (
                    "  fragments: "
                    f"templates={fragments.get('template_count', 0)} "
                    f"tools={fragments.get('tool_schema_count', 0)} "
                    f"stops={fragments.get('stop_policy_count', 0)} "
                    f"roles={fragments.get('role_count', 0)}"
                ),
                f"  diagnostics: {diagnostics}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_prompt_pack_registry(path: str | Path, registry: PromptPackRegistry) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(prompt_pack_registry_to_json(registry), encoding="utf-8")


def create_signed_prompt_pack_provenance(
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    key: str | bytes | None = None,
    key_id: str = "local",
    base_dir: str | Path | None = None,
) -> SignedPromptPackProvenance:
    """Create a signed, privacy-preserving provenance manifest for prompt packs."""

    registry = build_prompt_pack_registry(loaded_artifacts, diagnostics, base_dir=base_dir)
    payload = build_prompt_pack_provenance_payload(registry)
    return sign_prompt_pack_provenance_payload(payload, key=key, key_id=key_id)


def build_prompt_pack_provenance_payload(registry: PromptPackRegistry) -> dict[str, object]:
    """Build unsigned prompt-pack provenance from a registry manifest."""

    registry_dict = registry.to_dict()
    entries = registry_dict["prompt_packs"]
    payload: dict[str, object] = {
        "provenance_version": PROMPT_PACK_PROVENANCE_VERSION,
        "registry_hash": _hash_jsonable(registry_dict),
        "package_count": len(registry.entries),
        "prompt_packs": entries,
    }
    payload["proof_set_hash"] = _hash_jsonable(
        [
            {
                "contract_hash": entry["contract_hash"],
                "name": entry["name"],
                "package_name": entry["package_name"],
                "proof_hash": entry["proof_hash"],
                "sha256": entry.get("sha256"),
                "version": entry.get("version"),
            }
            for entry in entries  # type: ignore[union-attr]
            if isinstance(entry, dict)
        ]
    )
    return payload


def sign_prompt_pack_provenance_payload(
    payload: dict[str, object],
    *,
    key: str | bytes | None = None,
    key_id: str = "local",
) -> SignedPromptPackProvenance:
    """Sign a prompt-pack provenance payload with a local HMAC key."""

    _validate_prompt_pack_provenance_payload(payload)
    signature = _prompt_pack_signature(payload, _resolve_prompt_pack_key(key))
    return SignedPromptPackProvenance(payload=payload, signature=signature, signing_key_id=key_id)


def verify_signed_prompt_pack_provenance(
    provenance: SignedPromptPackProvenance | dict[str, object] | str | Path,
    *,
    key: str | bytes | None = None,
) -> PromptPackProvenanceVerification:
    """Verify a signed prompt-pack provenance manifest without reading prompt contents."""

    data = _prompt_pack_provenance_mapping(provenance)
    algorithm = data.get("algorithm")
    if algorithm != PROMPT_PACK_SIGNATURE_ALGORITHM:
        raise PromptPackProvenanceError(f"unsupported prompt-pack provenance signature algorithm: {algorithm!r}")
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise PromptPackProvenanceError("prompt-pack provenance payload must be an object")
    _validate_prompt_pack_provenance_payload(payload)
    actual = _provenance_required_str(data, "signature")
    key_id = _provenance_required_str(data, "signing_key_id")
    expected = _prompt_pack_signature(payload, _resolve_prompt_pack_key(key))
    ok = hmac.compare_digest(actual, expected)
    return PromptPackProvenanceVerification(
        ok=ok,
        provenance_hash=_hash_jsonable(payload),
        signing_key_id=key_id,
        expected_signature=expected,
        actual_signature=actual,
        package_count=_provenance_package_count(payload),
        reason=None if ok else "signature mismatch",
    )


def load_signed_prompt_pack_provenance(path: str | Path) -> SignedPromptPackProvenance:
    """Load a signed prompt-pack provenance manifest from JSON."""

    data = _prompt_pack_provenance_mapping(path)
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise PromptPackProvenanceError("prompt-pack provenance payload must be an object")
    _validate_prompt_pack_provenance_payload(payload)
    return SignedPromptPackProvenance(
        payload=payload,
        signature=_provenance_required_str(data, "signature"),
        signing_key_id=_provenance_required_str(data, "signing_key_id"),
        algorithm=_provenance_required_str(data, "algorithm"),
    )


def write_signed_prompt_pack_provenance(
    path: str | Path,
    provenance: SignedPromptPackProvenance,
    *,
    force: bool = False,
) -> None:
    """Write signed prompt-pack provenance JSON, refusing accidental overwrites by default."""

    destination = Path(path)
    if destination.exists() and not force:
        raise PromptPackProvenanceError(f"prompt-pack provenance already exists: {destination}; pass --force to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_prompt_pack_provenance_json(provenance), encoding="utf-8")


def render_prompt_pack_provenance_json(provenance: SignedPromptPackProvenance) -> str:
    return json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n"


def render_prompt_pack_provenance_verification_text(verification: PromptPackProvenanceVerification) -> str:
    status = "PASS" if verification.ok else "FAIL"
    lines = [
        "PromptABI signed prompt-pack provenance verification",
        f"status: {status}",
        f"packages: {verification.package_count}",
        f"provenance_hash: {verification.provenance_hash}",
        f"signing_key_id: {verification.signing_key_id}",
    ]
    if verification.reason is not None:
        lines.append(f"reason: {verification.reason}")
    return "\n".join(lines) + "\n"


def build_prompt_pack_mirror(
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    mirror_dir: str | Path,
    base_dir: str | Path | None = None,
) -> PromptPackMirrorManifest:
    """Copy configured local prompt-pack artifacts into a checksum-verified offline mirror."""

    resolved_mirror = Path(mirror_dir).expanduser().resolve()
    packages_dir = resolved_mirror / "packs"
    packages_dir.mkdir(parents=True, exist_ok=True)
    lockfile = build_prompt_pack_lockfile(loaded_artifacts, diagnostics, base_dir=base_dir)
    registry_by_name = {
        entry.name: _registry_entry_from_lock_entry(entry)
        for entry in lockfile.entries
    }
    entries: list[PromptPackMirrorEntry] = []
    for loaded in loaded_artifacts:
        artifact = loaded.artifact
        if not isinstance(artifact, PromptPackArtifact):
            continue
        if artifact.location.path is None:
            raise PromptPackMirrorError(
                f"prompt pack '{artifact.name}' is not a resolved local file and cannot be mirrored without network access"
            )
        source_path = Path(artifact.location.path)
        if not source_path.is_file():
            raise PromptPackMirrorError(f"prompt pack '{artifact.name}' source file is missing: {source_path}")
        sha256 = loaded.actual_sha256 or _sha256_file(source_path)
        lock_entry = next(entry for entry in lockfile.entries if entry.name == artifact.name)
        suffix = source_path.suffix or ".json"
        mirror_name = f"{_safe_filename(lock_entry.package_name)}-{sha256[:16]}{suffix}"
        mirror_path = packages_dir / mirror_name
        if mirror_path.exists():
            existing_sha = _sha256_file(mirror_path)
            if existing_sha != sha256:
                raise PromptPackMirrorError(f"mirror target collision for prompt pack '{artifact.name}': {mirror_path}")
        else:
            shutil.copyfile(source_path, mirror_path)
        size_bytes = mirror_path.stat().st_size
        source_location = _portable_path_string(str(source_path.resolve()), base_dir=Path(base_dir).resolve() if base_dir is not None else None)
        entries.append(
            PromptPackMirrorEntry(
                name=lock_entry.name,
                package_name=lock_entry.package_name,
                version=lock_entry.version,
                source_location=source_location,
                mirror_path=mirror_path.relative_to(resolved_mirror).as_posix(),
                sha256=sha256,
                size_bytes=size_bytes,
                contract_hash=lock_entry.contract_hash,
                proof_hash=registry_by_name[lock_entry.name].proof_hash,
                registry_entry=registry_by_name[lock_entry.name],
            )
        )
    if not entries:
        raise PromptPackMirrorError("no local prompt-pack artifacts are loaded; cannot build a local mirror")
    manifest = PromptPackMirrorManifest(tuple(sorted(entries, key=lambda entry: (entry.package_name, entry.name))))
    write_prompt_pack_mirror_manifest(resolved_mirror / "prompt-pack-mirror.json", manifest)
    return manifest


def prompt_pack_mirror_to_json(mirror: PromptPackMirrorManifest) -> str:
    return json.dumps(mirror.to_dict(), indent=2, sort_keys=True) + "\n"


def load_prompt_pack_mirror_manifest(path: str | Path) -> PromptPackMirrorManifest:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromptPackMirrorError(f"prompt-pack mirror manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise PromptPackMirrorError(
            f"prompt-pack mirror manifest is not valid JSON at {manifest_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise PromptPackMirrorError("prompt-pack mirror manifest root must be a JSON object")
    return PromptPackMirrorManifest.from_mapping(raw)


def write_prompt_pack_mirror_manifest(path: str | Path, mirror: PromptPackMirrorManifest) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(prompt_pack_mirror_to_json(mirror), encoding="utf-8")


def verify_prompt_pack_mirror(
    path: str | Path,
    *,
    manifest: PromptPackMirrorManifest | None = None,
) -> PromptPackMirrorVerification:
    """Verify mirrored prompt-pack files and privacy-preserving registry metadata offline."""

    manifest_path = Path(path).expanduser().resolve()
    mirror = manifest or load_prompt_pack_mirror_manifest(manifest_path)
    mirror_root = manifest_path.parent
    diagnostics: list[Diagnostic] = []
    for entry in mirror.entries:
        file_path = (mirror_root / entry.mirror_path).resolve()
        artifact_ref = ArtifactRef(kind="prompt-pack-mirror", name=entry.name, path=str(file_path))
        if not _is_relative_to(file_path, mirror_root):
            diagnostics.append(
                _prompt_pack_mirror_diagnostic(
                    "prompt-pack-mirror-path-escaped",
                    f"mirrored prompt pack '{entry.name}' points outside the mirror directory",
                    artifact_ref=artifact_ref,
                    expected="inside mirror",
                    actual=entry.mirror_path,
                )
            )
            continue
        if not file_path.is_file():
            diagnostics.append(
                _prompt_pack_mirror_diagnostic(
                    "prompt-pack-mirror-missing-file",
                    f"mirrored prompt pack '{entry.name}' is missing from the local mirror",
                    artifact_ref=artifact_ref,
                    expected=entry.sha256,
                    actual="missing",
                )
            )
            continue
        actual_sha = _sha256_file(file_path)
        actual_size = file_path.stat().st_size
        if actual_sha != entry.sha256:
            diagnostics.append(
                _prompt_pack_mirror_diagnostic(
                    "prompt-pack-mirror-sha256-drift",
                    f"mirrored prompt pack '{entry.name}' checksum differs from the mirror manifest",
                    artifact_ref=artifact_ref,
                    expected=entry.sha256,
                    actual=actual_sha,
                )
            )
        if actual_size != entry.size_bytes:
            diagnostics.append(
                _prompt_pack_mirror_diagnostic(
                    "prompt-pack-mirror-size-drift",
                    f"mirrored prompt pack '{entry.name}' size differs from the mirror manifest",
                    artifact_ref=artifact_ref,
                    expected=str(entry.size_bytes),
                    actual=str(actual_size),
                )
            )
    if not diagnostics:
        diagnostics.append(
            _prompt_pack_mirror_diagnostic(
                "prompt-pack-mirror-verified",
                f"prompt-pack mirror verifies {len(mirror.entries)} local package(s) without network access",
                artifact_ref=ArtifactRef(kind="prompt-pack-mirror", name="prompt-pack-mirror", path=str(manifest_path)),
                expected="matched",
                actual="matched",
                severity=DiagnosticSeverity.INFO,
            )
        )
    return PromptPackMirrorVerification(mirror, tuple(diagnostics))


def render_prompt_pack_mirror_text(mirror: PromptPackMirrorManifest) -> str:
    lines = ["PromptABI prompt-pack local mirror", f"mirror version: {mirror.mirror_version}"]
    for entry in mirror.entries:
        version = f"@{entry.version}" if entry.version is not None else ""
        lines.extend(
            [
                "",
                f"- {entry.package_name}{version}",
                f"  file: {entry.mirror_path}",
                f"  sha256: {entry.sha256}",
                f"  contract: {entry.contract_hash[:16]} proof: {entry.proof_hash[:16]}",
                f"  source: {entry.source_location}",
            ]
        )
    return "\n".join(lines) + "\n"


def render_prompt_pack_mirror_verification_text(verification: PromptPackMirrorVerification) -> str:
    status = "PASS" if verification.ok else "FAIL"
    lines = [
        "PromptABI prompt-pack local mirror verification",
        f"status: {status}",
        f"packages: {len(verification.manifest.entries)}",
    ]
    for diagnostic in verification.diagnostics:
        lines.append(f"{diagnostic.severity.value.upper()} {diagnostic.rule_id}: {diagnostic.message}")
    return "\n".join(lines) + "\n"


def compare_prompt_pack_lockfile(
    lockfile: PromptPackLockfile,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    lockfile_path: str | Path | None = None,
) -> tuple[Diagnostic, ...]:
    """Return diagnostics for prompt-pack package or verified-contract drift."""

    base_dir = Path(lockfile_path).expanduser().resolve().parent if lockfile_path is not None else None
    current = build_prompt_pack_lockfile(loaded_artifacts, diagnostics, base_dir=base_dir)
    drift: list[Diagnostic] = []
    expected_by_name = {entry.name: entry for entry in lockfile.entries}
    current_by_name = {entry.name: entry for entry in current.entries}
    for name in sorted(expected_by_name.keys() - current_by_name.keys()):
        drift.append(
            _prompt_pack_lock_diagnostic(
                "prompt-pack-lock-missing",
                f"prompt pack '{name}' is present in the lockfile but missing from current verification",
                "prompt_pack",
                "present",
                "missing",
                entry=expected_by_name[name],
                lockfile_path=lockfile_path,
            )
        )
    for name in sorted(current_by_name.keys() - expected_by_name.keys()):
        drift.append(
            _prompt_pack_lock_diagnostic(
                "prompt-pack-lock-added",
                f"prompt pack '{name}' is new relative to the lockfile",
                "prompt_pack",
                "missing",
                "present",
                entry=current_by_name[name],
                lockfile_path=lockfile_path,
            )
        )
    for name in sorted(expected_by_name.keys() & current_by_name.keys()):
        expected = expected_by_name[name]
        actual = current_by_name[name]
        for field in (
            "package_name",
            "version",
            "location",
            "sha256",
            "contract_hash",
            "exported_templates",
            "tool_schemas",
            "stop_policies",
            "supported_model_families",
            "expected_roles",
            "diagnostic_baseline",
        ):
            if getattr(expected, field) != getattr(actual, field):
                drift.append(
                    _prompt_pack_lock_diagnostic(
                        "prompt-pack-lock-drift",
                        f"prompt pack '{name}' {field.replace('_', ' ')} differs from the lockfile",
                        field,
                        str(getattr(expected, field)),
                        str(getattr(actual, field)),
                        entry=actual,
                        lockfile_path=lockfile_path,
                    )
                )
    if not drift:
        return (
            _prompt_pack_lock_diagnostic(
                "prompt-pack-lock-verified",
                "prompt-pack lockfile matches the current prompt-pack packages and verified contracts",
                "prompt_pack_lockfile",
                "matched",
                "matched",
                severity=DiagnosticSeverity.INFO,
                lockfile_path=lockfile_path,
            ),
        )
    return tuple(drift)


def compare_prompt_pack_upgrade(
    baseline: PromptPackLockfile,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    baseline_path: str | Path | None = None,
) -> tuple[Diagnostic, ...]:
    """Prove whether current prompt packs preserve baseline verified guarantees."""

    base_dir = Path(baseline_path).expanduser().resolve().parent if baseline_path is not None else None
    findings: list[Diagnostic] = []
    baseline_by_name = {entry.name: entry for entry in baseline.entries}
    if baseline_by_name and not any(isinstance(loaded.artifact, PromptPackArtifact) for loaded in loaded_artifacts):
        return tuple(
            _prompt_pack_upgrade_diagnostic(
                "prompt-pack-upgrade-missing-pack",
                f"prompt pack '{name}' is missing from the upgrade candidate",
                "prompt_pack",
                "present",
                "missing",
                entry=entry,
                baseline_path=baseline_path,
            )
            for name, entry in sorted(baseline_by_name.items())
        )
    current = build_prompt_pack_lockfile(loaded_artifacts, diagnostics, base_dir=base_dir)
    current_by_name = {entry.name: entry for entry in current.entries}
    for name in sorted(baseline_by_name.keys() - current_by_name.keys()):
        findings.append(
            _prompt_pack_upgrade_diagnostic(
                "prompt-pack-upgrade-missing-pack",
                f"prompt pack '{name}' is missing from the upgrade candidate",
                "prompt_pack",
                "present",
                "missing",
                entry=baseline_by_name[name],
                baseline_path=baseline_path,
            )
        )
    for name in sorted(baseline_by_name.keys() & current_by_name.keys()):
        expected = baseline_by_name[name]
        actual = current_by_name[name]
        if expected.package_name != actual.package_name:
            findings.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-package-regression",
                    f"prompt pack '{name}' resolves to a different package name",
                    "package_name",
                    expected.package_name,
                    actual.package_name,
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
        _append_subset_regression(
            findings,
            entry=actual,
            baseline_path=baseline_path,
            rule_id="prompt-pack-upgrade-role-regression",
            field="expected_roles",
            expected=expected.expected_roles,
            actual=actual.expected_roles,
            message=f"prompt pack '{name}' no longer preserves every baseline role-boundary guarantee",
        )
        _append_subset_regression(
            findings,
            entry=actual,
            baseline_path=baseline_path,
            rule_id="prompt-pack-upgrade-model-family-regression",
            field="supported_model_families",
            expected=expected.supported_model_families,
            actual=actual.supported_model_families,
            message=f"prompt pack '{name}' dropped baseline model-family support",
        )
        findings.extend(_template_upgrade_regressions(expected, actual, baseline_path=baseline_path))
        findings.extend(_tool_upgrade_regressions(expected, actual, baseline_path=baseline_path))
        findings.extend(_stop_upgrade_regressions(expected, actual, baseline_path=baseline_path))
        findings.extend(_diagnostic_upgrade_regressions(expected, actual, baseline_path=baseline_path))
    if findings:
        return tuple(findings)
    return (
        _prompt_pack_upgrade_diagnostic(
            "prompt-pack-upgrade-compatible",
            "prompt-pack upgrade preserves baseline role-boundary, stop, schema, budget, and diagnostic guarantees",
            "prompt_pack_upgrade",
            "baseline",
            "compatible",
            severity=DiagnosticSeverity.INFO,
            baseline_path=baseline_path,
        ),
    )


def prompt_pack_lock_error_diagnostic(
    exc: PromptPackLockError,
    *,
    lockfile_path: str | Path | None = None,
) -> Diagnostic:
    path = str(lockfile_path) if lockfile_path is not None else None
    return Diagnostic(
        rule_id="prompt-pack-lock-load-failed",
        severity=DiagnosticSeverity.ERROR,
        message=str(exc),
        artifact=ArtifactRef(kind="prompt-pack-lockfile", name="prompt-pack-lockfile", path=path) if path else None,
        check_modes=PROMPT_PACK_LOCK_CHECK_MODES,
        suggestions=("Run promptabi prompt-pack lock after reviewing current prompt-pack packages.",),
        witness=WitnessTrace(
            summary="PromptABI could not load the prompt-pack lockfile for enforcement.",
            steps=(WitnessStep(action="load prompt-pack lockfile", input=path, output=str(exc)),),
        ),
    )


def prompt_pack_mirror_error_diagnostic(
    exc: PromptPackMirrorError,
    *,
    manifest_path: str | Path | None = None,
) -> Diagnostic:
    path = str(manifest_path) if manifest_path is not None else None
    return Diagnostic(
        rule_id="prompt-pack-mirror-load-failed",
        severity=DiagnosticSeverity.ERROR,
        message=str(exc),
        artifact=ArtifactRef(kind="prompt-pack-mirror", name="prompt-pack-mirror", path=path) if path else None,
        check_modes=PROMPT_PACK_LOCK_CHECK_MODES,
        suggestions=("Rebuild the local prompt-pack mirror from reviewed prompt-pack artifacts.",),
        witness=WitnessTrace(
            summary="PromptABI could not load or verify the local prompt-pack mirror.",
            steps=(WitnessStep(action="load prompt-pack mirror", input=path, output=str(exc)),),
        ),
    )


def _append_subset_regression(
    findings: list[Diagnostic],
    *,
    entry: PromptPackLockEntry,
    baseline_path: str | Path | None,
    rule_id: str,
    field: str,
    expected: tuple[str, ...],
    actual: tuple[str, ...],
    message: str,
) -> None:
    missing = tuple(sorted(set(expected) - set(actual)))
    if missing:
        findings.append(
            _prompt_pack_upgrade_diagnostic(
                rule_id,
                message,
                field,
                ", ".join(expected),
                ", ".join(actual) if actual else "(none)",
                entry=entry,
                baseline_path=baseline_path,
                extra_properties=(("missing", ", ".join(missing)),),
            )
        )


def _template_upgrade_regressions(
    expected: PromptPackLockEntry,
    actual: PromptPackLockEntry,
    *,
    baseline_path: str | Path | None,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    actual_by_name = {name: (template_hash, roles_hash) for name, template_hash, roles_hash in actual.exported_templates}
    for name, template_hash, roles_hash in expected.exported_templates:
        observed = actual_by_name.get(name)
        if observed is None:
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-template-regression",
                    f"prompt pack '{actual.name}' removed baseline template '{name}'",
                    "exported_templates",
                    name,
                    "missing",
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
        elif observed != (template_hash, roles_hash):
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-template-regression",
                    f"prompt pack '{actual.name}' changed template '{name}' role or budget contract",
                    f"exported_templates.{name}",
                    f"template={template_hash}, roles={roles_hash}",
                    f"template={observed[0]}, roles={observed[1]}",
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
    return tuple(diagnostics)


def _tool_upgrade_regressions(
    expected: PromptPackLockEntry,
    actual: PromptPackLockEntry,
    *,
    baseline_path: str | Path | None,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    actual_by_name = {name: (provider, schema_digest, required) for name, provider, schema_digest, required in actual.tool_schemas}
    for name, provider, schema_digest, required in expected.tool_schemas:
        observed = actual_by_name.get(name)
        if observed is None:
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-tool-schema-regression",
                    f"prompt pack '{actual.name}' removed baseline tool schema '{name}'",
                    "tool_schemas",
                    name,
                    "missing",
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
        elif observed[0] != provider or observed[1] != schema_digest or (required and not observed[2]):
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-tool-schema-regression",
                    f"prompt pack '{actual.name}' changed baseline tool schema guarantee for '{name}'",
                    f"tool_schemas.{name}",
                    str((provider, schema_digest, required)),
                    str(observed),
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
    return tuple(diagnostics)


def _stop_upgrade_regressions(
    expected: PromptPackLockEntry,
    actual: PromptPackLockEntry,
    *,
    baseline_path: str | Path | None,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    actual_by_name = {
        name: (set(stop_sequences), set(stop_token_ids), include_eos)
        for name, stop_sequences, stop_token_ids, include_eos in actual.stop_policies
    }
    for name, stop_sequences, stop_token_ids, include_eos in expected.stop_policies:
        observed = actual_by_name.get(name)
        if observed is None:
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-stop-regression",
                    f"prompt pack '{actual.name}' removed baseline stop policy '{name}'",
                    "stop_policies",
                    name,
                    "missing",
                    entry=actual,
                    baseline_path=baseline_path,
                )
            )
            continue
        missing_sequences = set(stop_sequences) - observed[0]
        missing_token_ids = set(stop_token_ids) - observed[1]
        eos_regressed = include_eos and not observed[2]
        if missing_sequences or missing_token_ids or eos_regressed:
            diagnostics.append(
                _prompt_pack_upgrade_diagnostic(
                    "prompt-pack-upgrade-stop-regression",
                    f"prompt pack '{actual.name}' no longer preserves stop-policy guarantee '{name}'",
                    f"stop_policies.{name}",
                    str((stop_sequences, stop_token_ids, include_eos)),
                    str((tuple(sorted(observed[0])), tuple(sorted(observed[1])), observed[2])),
                    entry=actual,
                    baseline_path=baseline_path,
                    extra_properties=(
                        ("missing_stop_sequences", ", ".join(sorted(missing_sequences))),
                        ("missing_stop_token_ids", ", ".join(str(item) for item in sorted(missing_token_ids))),
                        ("include_eos_regressed", eos_regressed),
                    ),
                )
            )
    return tuple(diagnostics)


def _diagnostic_upgrade_regressions(
    expected: PromptPackLockEntry,
    actual: PromptPackLockEntry,
    *,
    baseline_path: str | Path | None,
) -> tuple[Diagnostic, ...]:
    expected_non_info = {(rule_id, severity) for rule_id, severity, _fingerprint in expected.diagnostic_baseline if severity != DiagnosticSeverity.INFO.value}
    actual_non_info = {(rule_id, severity) for rule_id, severity, _fingerprint in actual.diagnostic_baseline if severity != DiagnosticSeverity.INFO.value}
    new_non_info = tuple(sorted(actual_non_info - expected_non_info))
    expected_verified = any(rule_id == "prompt-pack-verified" for rule_id, _severity, _fingerprint in expected.diagnostic_baseline)
    actual_verified = any(rule_id == "prompt-pack-verified" for rule_id, _severity, _fingerprint in actual.diagnostic_baseline)
    if not new_non_info and (not expected_verified or actual_verified):
        return ()
    if expected_verified and not actual_verified:
        message = f"prompt pack '{actual.name}' lost its verified prompt-pack contract diagnostic"
        field = "diagnostic_baseline.prompt-pack-verified"
        expected_text = "present"
        actual_text = "missing"
    else:
        message = f"prompt pack '{actual.name}' introduced prompt-pack diagnostic regressions"
        field = "diagnostic_baseline"
        expected_text = str(tuple(sorted(expected_non_info)))
        actual_text = str(tuple(sorted(actual_non_info)))
    return (
        _prompt_pack_upgrade_diagnostic(
            "prompt-pack-upgrade-diagnostic-regression",
            message,
            field,
            expected_text,
            actual_text,
            entry=actual,
            baseline_path=baseline_path,
        ),
    )


def _application_roles(artifacts: tuple[Artifact, ...]) -> set[str]:
    roles: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ChatTemplateArtifact):
            roles.update(artifact.roles)
        elif isinstance(artifact, PromptSegmentArtifact):
            roles.update(segment.role for segment in artifact.segments if segment.role is not None)
    return roles


def _prompt_pack_composition_findings(
    prompt_pack: PromptPackArtifact,
    artifacts: tuple[Artifact, ...],
    *,
    source_spans: dict[str, SourceSpan],
    token_budget_report: TokenBudgetReport | None,
) -> tuple[PromptPackFinding, ...]:
    if not _has_composition_evidence(artifacts, token_budget_report):
        return ()

    findings: list[PromptPackFinding] = []
    segments = _application_prompt_segments(artifacts)
    segment_by_name = {segment.name: segment for segment in segments}
    pack_roles = _prompt_pack_declared_roles(prompt_pack)
    added_roles = tuple(
        sorted(
            {
                segment.role
                for segment in segments
                if segment.role is not None
                and segment.role not in pack_roles
                and not _raw_segment_is_retrieval(segment)
            }
        )
    )
    if added_roles:
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.COMPOSITION_CONTEXT_ROLE_UNSUPPORTED,
                pack=prompt_pack,
                message=(
                    f"downstream context adds role(s) outside prompt-pack guarantees: "
                    f"{', '.join(added_roles)}"
                ),
                expected=tuple(sorted(pack_roles)),
                observed=added_roles,
                span=source_spans.get("expected_roles"),
                subject="context roles",
                severity=DiagnosticSeverity.WARNING,
                evidence=(("added_roles", ", ".join(added_roles)),),
            )
        )

    required_regions = _prompt_pack_required_regions(prompt_pack)
    if required_regions:
        missing_regions = tuple(region for region in required_regions if region not in segment_by_name)
        if missing_regions:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.COMPOSITION_REQUIRED_REGION_UNDECLARED,
                    pack=prompt_pack,
                    message=(
                        "downstream composition does not declare prompt-pack required region(s): "
                        f"{', '.join(missing_regions)}"
                    ),
                    expected=required_regions,
                    observed=tuple(sorted(segment_by_name)),
                    span=source_spans.get("exported_templates"),
                    subject="required regions",
                    evidence=(("missing_regions", ", ".join(missing_regions)),),
                )
            )
        optional_regions = tuple(
            region
            for region in required_regions
            if (segment := segment_by_name.get(region)) is not None and not segment.required
        )
        if optional_regions:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.COMPOSITION_REQUIRED_REGION_NOT_MUST_SURVIVE,
                    pack=prompt_pack,
                    message=(
                        "downstream composition declares prompt-pack required region(s) without "
                        f"must-survive protection: {', '.join(optional_regions)}"
                    ),
                    expected=required_regions,
                    observed=tuple(
                        sorted(segment.name for segment in segments if segment.required)
                    ),
                    span=source_spans.get("exported_templates"),
                    subject="must-survive regions",
                    evidence=(("optional_required_regions", ", ".join(optional_regions)),),
                )
            )

    proof = token_budget_report.must_survive_proof if token_budget_report is not None else None
    if proof is not None and proof.status == "violated":
        dropped_required = tuple(region for region in required_regions if region in proof.dropped_segments)
        if dropped_required:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.COMPOSITION_REQUIRED_REGION_TRUNCATED,
                    pack=prompt_pack,
                    message=(
                        "framework truncation can drop prompt-pack required region(s): "
                        f"{', '.join(dropped_required)}"
                    ),
                    expected=required_regions,
                    observed=tuple(proof.survived_segments),
                    span=source_spans.get("exported_templates"),
                    subject="truncation proof",
                    evidence=(
                        ("dropped_required_regions", ", ".join(dropped_required)),
                        ("policy", f"{proof.policy.framework}:{proof.policy.strategy}"),
                        ("input_budget_tokens", str(proof.input_budget_tokens)),
                    ),
                )
            )

    unbounded_chunks = tuple(
        chunk.name
        for chunk in _composition_rag_chunks(segments, token_budget_report)
        if not _rag_chunk_has_runtime_bound(chunk)
    )
    if unbounded_chunks:
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.COMPOSITION_RAG_UNBOUNDED,
                pack=prompt_pack,
                message=(
                    "downstream RAG context has no runtime token bound for prompt-pack composition: "
                    f"{', '.join(unbounded_chunks)}"
                ),
                expected=("max_tokens", "retrieval_payload_limit_tokens"),
                observed=unbounded_chunks,
                span=source_spans.get("exported_templates"),
                subject="RAG bounds",
                evidence=(("unbounded_chunks", ", ".join(unbounded_chunks)),),
            )
        )

    return tuple(findings)


def _has_composition_evidence(
    artifacts: tuple[Artifact, ...],
    token_budget_report: TokenBudgetReport | None,
) -> bool:
    return any(isinstance(artifact, FrameworkTruncationConfigArtifact) for artifact in artifacts) or bool(
        _composition_rag_chunks(_application_prompt_segments(artifacts), token_budget_report)
    )


def _composition_evidence(
    artifacts: tuple[Artifact, ...],
    token_budget_report: TokenBudgetReport | None,
) -> tuple[tuple[str, str], ...]:
    rag_chunks = _composition_rag_chunks(_application_prompt_segments(artifacts), token_budget_report)
    budgets = tuple(artifact.name for artifact in artifacts if isinstance(artifact, FrameworkTruncationConfigArtifact))
    evidence = [
        ("framework_truncation_configs", ", ".join(budgets) or "<none>"),
        ("rag_chunks", ", ".join(chunk.name for chunk in rag_chunks) or "<none>"),
    ]
    if token_budget_report is not None and token_budget_report.must_survive_proof is not None:
        evidence.append(("must_survive_status", token_budget_report.must_survive_proof.status))
    return tuple(evidence)


def _application_prompt_segments(artifacts: tuple[Artifact, ...]) -> tuple[PromptSegment, ...]:
    return tuple(
        segment
        for artifact in artifacts
        if isinstance(artifact, PromptSegmentArtifact)
        for segment in artifact.segments
    )


def _prompt_pack_declared_roles(prompt_pack: PromptPackArtifact) -> set[str]:
    roles = set(prompt_pack.expected_roles)
    for template in prompt_pack.exported_templates:
        roles.update(template.roles)
    return roles


def _prompt_pack_required_regions(prompt_pack: PromptPackArtifact) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                region
                for template in prompt_pack.exported_templates
                for region in template.required_regions
            }
        )
    )


def _composition_rag_chunks(
    segments: tuple[PromptSegment, ...],
    token_budget_report: TokenBudgetReport | None,
) -> tuple[PromptSegment | TokenBudgetSegment, ...]:
    if token_budget_report is not None:
        return tuple(segment for segment in token_budget_report.segments if segment.is_retrieval_chunk)
    return tuple(segment for segment in segments if _raw_segment_is_retrieval(segment))


def _raw_segment_is_retrieval(segment: PromptSegment) -> bool:
    return (
        segment.role == "retrieval"
        or segment.chunk_id is not None
        or segment.document_id is not None
        or segment.chunk_tokenizer is not None
        or segment.source_start is not None
        or segment.source_end is not None
        or segment.chunk_start is not None
        or segment.chunk_end is not None
        or segment.expected_overlap_tokens is not None
        or segment.actual_overlap_tokens is not None
        or segment.citation is not None
        or segment.citation_required
        or segment.metadata_tokens > 0
        or segment.template_overhead_tokens > 0
        or segment.retrieval_payload_limit_tokens is not None
    )


def _rag_chunk_has_runtime_bound(segment: PromptSegment | TokenBudgetSegment) -> bool:
    return segment.max_tokens is not None or segment.retrieval_payload_limit_tokens is not None


def _application_tools(artifacts: tuple[Artifact, ...]) -> set[str]:
    tools: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ToolDefinitionArtifact):
            tools.update(artifact.tool_names)
    return tools


def _application_stop_sequences(artifacts: tuple[Artifact, ...]) -> set[str]:
    stops: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, StopPolicyArtifact):
            stops.update(artifact.stop_sequences)
    return stops


def _application_model_families(artifacts: tuple[Artifact, ...]) -> set[str]:
    families: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ProviderConfigArtifact):
            families.add(artifact.provider)
            if artifact.api_family is not None:
                families.add(artifact.api_family)
        elif isinstance(artifact, TokenizerArtifact) and artifact.family is not None:
            families.add(artifact.family)
    return families


def _prompt_pack_lock_entry(
    loaded: LoadedArtifact,
    diagnostics: tuple[Diagnostic, ...],
    *,
    base_dir: Path | None,
) -> PromptPackLockEntry:
    artifact = loaded.artifact
    if not isinstance(artifact, PromptPackArtifact):
        raise PromptPackLockError("prompt-pack lock entries can only be built from prompt-pack artifacts")
    payload = {
        "expected_roles": list(artifact.expected_roles),
        "exported_templates": [template.to_dict() for template in artifact.exported_templates],
        "pack_name": artifact.pack_name,
        "stop_policies": [stop_policy.to_dict() for stop_policy in artifact.stop_policies],
        "supported_model_families": list(artifact.supported_model_families),
        "tool_schemas": [tool.to_dict() for tool in artifact.tool_schemas],
        "version": artifact.pack_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return PromptPackLockEntry(
        name=artifact.name,
        package_name=artifact.pack_name,
        version=artifact.pack_version,
        location=_portable_path_string(artifact.location.ref_path or "", base_dir=base_dir),
        sha256=loaded.actual_sha256 or artifact.provenance.sha256,
        contract_hash=hashlib.sha256(encoded).hexdigest(),
        exported_templates=tuple(
            sorted(
                (
                    template.name,
                    _sha256_text(template.template),
                    _sha256_text("\n".join(template.roles)),
                )
                for template in artifact.exported_templates
            )
        ),
        tool_schemas=tuple(
            sorted((tool.name, tool.provider, tool.schema_digest, tool.required) for tool in artifact.tool_schemas)
        ),
        stop_policies=tuple(
            sorted(
                (
                    stop_policy.name,
                    stop_policy.stop_sequences,
                    stop_policy.stop_token_ids,
                    stop_policy.include_eos,
                )
                for stop_policy in artifact.stop_policies
            )
        ),
        supported_model_families=artifact.supported_model_families,
        expected_roles=artifact.expected_roles,
        diagnostic_baseline=_prompt_pack_diagnostic_baseline(artifact.name, diagnostics),
    )


def _registry_entry_from_lock_entry(entry: PromptPackLockEntry) -> PromptPackRegistryEntry:
    template_proofs = tuple(
        {
            "roles_hash": roles_hash,
            "template_hash": template_hash,
            "template_name_hash": _sha256_text(name),
        }
        for name, template_hash, roles_hash in entry.exported_templates
    )
    tool_proofs = tuple(
        {
            "provider": provider,
            "required": required,
            "schema_digest_hash": _sha256_text(schema_digest) if schema_digest is not None else None,
            "tool_name_hash": _sha256_text(name),
        }
        for name, provider, schema_digest, required in entry.tool_schemas
    )
    stop_proofs = tuple(
        {
            "include_eos": include_eos,
            "stop_policy_name_hash": _sha256_text(name),
            "stop_sequence_set_hash": _hash_jsonable(stop_sequences),
            "stop_token_id_set_hash": _hash_jsonable(stop_token_ids),
        }
        for name, stop_sequences, stop_token_ids, include_eos in entry.stop_policies
    )
    proof_payload = {
        "diagnostics": [
            {"fingerprint": fingerprint, "rule_id": rule_id, "severity": severity}
            for rule_id, severity, fingerprint in entry.diagnostic_baseline
        ],
        "model_family_hashes": [_sha256_text(family) for family in entry.supported_model_families],
        "role_hashes": [_sha256_text(role) for role in entry.expected_roles],
        "stop_policies": stop_proofs,
        "templates": template_proofs,
        "tools": tool_proofs,
    }
    proof_hash = _hash_jsonable(proof_payload)
    supported_fragments: tuple[tuple[str, object], ...] = (
        ("diagnostic_count", len(entry.diagnostic_baseline)),
        ("model_family_count", len(entry.supported_model_families)),
        ("role_count", len(entry.expected_roles)),
        ("stop_policy_count", len(entry.stop_policies)),
        ("template_count", len(entry.exported_templates)),
        ("tool_schema_count", len(entry.tool_schemas)),
    )
    reproducible_metadata: tuple[tuple[str, object], ...] = (
        ("contract_hash", entry.contract_hash),
        ("location", entry.location),
        ("package_name", entry.package_name),
        ("sha256", entry.sha256),
        ("version", entry.version),
    )
    proofs: tuple[tuple[str, object], ...] = (
        ("diagnostic_fingerprints", proof_payload["diagnostics"]),
        ("model_family_hashes", proof_payload["model_family_hashes"]),
        ("role_hashes", proof_payload["role_hashes"]),
        ("stop_policy_proofs", stop_proofs),
        ("template_proofs", template_proofs),
        ("tool_schema_proofs", tool_proofs),
    )
    return PromptPackRegistryEntry(
        name=entry.name,
        package_name=entry.package_name,
        version=entry.version,
        location=entry.location,
        sha256=entry.sha256,
        contract_hash=entry.contract_hash,
        proof_hash=proof_hash,
        supported_fragments=supported_fragments,
        reproducible_metadata=reproducible_metadata,
        proofs=proofs,
        diagnostics=entry.diagnostic_baseline,
    )


def _prompt_pack_diagnostic_baseline(
    artifact_name: str,
    diagnostics: tuple[Diagnostic, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (diagnostic.rule_id, diagnostic.severity.value, diagnostic.fingerprint)
            for diagnostic in diagnostics
            if diagnostic.rule_id.startswith("prompt-pack-")
            and not diagnostic.rule_id.startswith("prompt-pack-lock-")
            and diagnostic.artifact is not None
            and diagnostic.artifact.name == artifact_name
        )
    )


def _prompt_pack_lock_diagnostic(
    rule_id: str,
    message: str,
    field: str,
    expected: str,
    actual: str,
    *,
    entry: PromptPackLockEntry | None = None,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    lockfile_path: str | Path | None,
) -> Diagnostic:
    artifact = (
        ArtifactRef(kind="prompt-pack", name=entry.name, path=entry.location if "://" not in entry.location else None, uri=entry.location if "://" in entry.location else None)
        if entry is not None
        else ArtifactRef(kind="prompt-pack-lockfile", name="prompt-pack-lockfile", path=str(lockfile_path) if lockfile_path is not None else None)
    )
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=artifact,
        check_modes=PROMPT_PACK_LOCK_CHECK_MODES,
        suggestions=("Regenerate the prompt-pack lockfile only after reviewing package and contract changes.",),
        witness=WitnessTrace(
            summary="The enforced prompt-pack lockfile does not match the current package contract state."
            if severity is DiagnosticSeverity.ERROR
            else "The enforced prompt-pack lockfile matches the current package contract state.",
            steps=(
                WitnessStep(action="read prompt-pack lockfile", input=str(lockfile_path) if lockfile_path is not None else None),
                WitnessStep(action=f"compare {field}", input=expected, output=actual),
            ),
            artifacts=(artifact,),
        ),
        properties=(("actual", actual), ("expected", expected), ("field", field)),
    )


def _prompt_pack_upgrade_diagnostic(
    rule_id: str,
    message: str,
    field: str,
    expected: str,
    actual: str,
    *,
    entry: PromptPackLockEntry | None = None,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    baseline_path: str | Path | None,
    extra_properties: tuple[tuple[str, object], ...] = (),
) -> Diagnostic:
    artifact = (
        ArtifactRef(kind="prompt-pack", name=entry.name, path=entry.location if "://" not in entry.location else None, uri=entry.location if "://" in entry.location else None)
        if entry is not None
        else ArtifactRef(kind="prompt-pack-lockfile", name="prompt-pack-upgrade-baseline", path=str(baseline_path) if baseline_path is not None else None)
    )
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=artifact,
        check_modes=PROMPT_PACK_LOCK_CHECK_MODES,
        suggestions=("Keep old pack guarantees, or intentionally regenerate a new baseline after downstream apps are updated.",),
        witness=WitnessTrace(
            summary="PromptABI compared the baseline prompt-pack lock against the upgrade candidate.",
            steps=(
                WitnessStep(action="read baseline prompt-pack lockfile", input=str(baseline_path) if baseline_path is not None else None),
                WitnessStep(action=f"compare {field}", input=expected, output=actual),
            ),
            artifacts=(artifact,),
        ),
        properties=(("actual", actual), ("expected", expected), ("field", field), *extra_properties),
    )


def _prompt_pack_mirror_diagnostic(
    rule_id: str,
    message: str,
    *,
    artifact_ref: ArtifactRef,
    expected: str,
    actual: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=artifact_ref,
        check_modes=PROMPT_PACK_LOCK_CHECK_MODES,
        suggestions=("Use `promptabi prompt-pack mirror build` to refresh the local mirror after reviewing package changes.",),
        witness=WitnessTrace(
            summary="PromptABI verified local mirror files against the mirror manifest.",
            steps=(
                WitnessStep(action="read mirror manifest", input=artifact_ref.path),
                WitnessStep(action="compare mirrored artifact", input=expected, output=actual),
            ),
            artifacts=(artifact_ref,),
        ),
        properties=(("actual", actual), ("expected", expected)),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_jsonable(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prompt_pack_signature(payload: dict[str, object], key: bytes) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _resolve_prompt_pack_key(key: str | bytes | None) -> bytes:
    raw = key if key is not None else os.environ.get("PROMPTABI_PROMPT_PACK_KEY")
    if raw is None:
        raise PromptPackProvenanceError("prompt-pack provenance signing key is required; pass --key or set PROMPTABI_PROMPT_PACK_KEY")
    if isinstance(raw, bytes):
        if not raw:
            raise PromptPackProvenanceError("prompt-pack provenance signing key must be non-empty")
        return raw
    if not raw:
        raise PromptPackProvenanceError("prompt-pack provenance signing key must be non-empty")
    return raw.encode("utf-8")


def _prompt_pack_provenance_mapping(value: SignedPromptPackProvenance | dict[str, object] | str | Path) -> dict[str, object]:
    if isinstance(value, SignedPromptPackProvenance):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    path = Path(value)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromptPackProvenanceError(f"prompt-pack provenance not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PromptPackProvenanceError(
            f"prompt-pack provenance is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise PromptPackProvenanceError("prompt-pack provenance root must be a JSON object")
    return raw


def _validate_prompt_pack_provenance_payload(payload: dict[str, object]) -> None:
    if payload.get("provenance_version") != PROMPT_PACK_PROVENANCE_VERSION:
        raise PromptPackProvenanceError(f"unsupported prompt-pack provenance version: {payload.get('provenance_version')!r}")
    prompt_packs = payload.get("prompt_packs")
    if not isinstance(prompt_packs, list):
        raise PromptPackProvenanceError("prompt-pack provenance field 'prompt_packs' must be a list")
    for item in prompt_packs:
        if not isinstance(item, dict):
            raise PromptPackProvenanceError("prompt-pack provenance entries must be objects")
        _registry_entry_from_mapping(item)
    package_count = payload.get("package_count")
    if not isinstance(package_count, int) or isinstance(package_count, bool) or package_count != len(prompt_packs):
        raise PromptPackProvenanceError("prompt-pack provenance package_count must match prompt_packs")
    registry_hash = payload.get("registry_hash")
    if not isinstance(registry_hash, str) or not registry_hash:
        raise PromptPackProvenanceError("prompt-pack provenance field 'registry_hash' must be a non-empty string")
    proof_set_hash = payload.get("proof_set_hash")
    if not isinstance(proof_set_hash, str) or not proof_set_hash:
        raise PromptPackProvenanceError("prompt-pack provenance field 'proof_set_hash' must be a non-empty string")


def _provenance_package_count(payload: dict[str, object]) -> int:
    value = payload.get("package_count")
    if not isinstance(value, int) or isinstance(value, bool):
        raise PromptPackProvenanceError("prompt-pack provenance package_count must be an integer")
    return value


def _provenance_required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise PromptPackProvenanceError(f"prompt-pack provenance field '{key}' must be a non-empty string")
    return value


def _pairs_to_dict(pairs: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return {key: value for key, value in pairs if value is not None}


def _portable_path_string(value: str, *, base_dir: Path | None) -> str:
    if base_dir is None or "://" in value:
        return value
    try:
        path = Path(value)
    except ValueError:
        return value
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(base_dir).as_posix()
    except ValueError:
        return value


def _template_lock_entries(value: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        raise PromptPackLockError("prompt-pack lock field 'exported_templates' must be a list")
    entries: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise PromptPackLockError("prompt-pack lock exported template entries must be objects")
        entries.append((_required_str(item, "name"), _required_str(item, "template_hash"), _required_str(item, "roles_hash")))
    return tuple(sorted(entries))


def _tool_lock_entries(value: object) -> tuple[tuple[str, str | None, str | None, bool], ...]:
    if not isinstance(value, list):
        raise PromptPackLockError("prompt-pack lock field 'tool_schemas' must be a list")
    entries: list[tuple[str, str | None, str | None, bool]] = []
    for item in value:
        if not isinstance(item, dict):
            raise PromptPackLockError("prompt-pack lock tool schema entries must be objects")
        entries.append(
            (
                _required_str(item, "name"),
                _optional_str(item, "provider"),
                _optional_str(item, "schema_digest"),
                _required_bool(item, "required"),
            )
        )
    return tuple(sorted(entries))


def _stop_lock_entries(value: object) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...], bool], ...]:
    if not isinstance(value, list):
        raise PromptPackLockError("prompt-pack lock field 'stop_policies' must be a list")
    entries: list[tuple[str, tuple[str, ...], tuple[int, ...], bool]] = []
    for item in value:
        if not isinstance(item, dict):
            raise PromptPackLockError("prompt-pack lock stop policy entries must be objects")
        entries.append(
            (
                _required_str(item, "name"),
                _string_tuple(item.get("stop_sequences", []), "stop_sequences"),
                _int_tuple(item.get("stop_token_ids", []), "stop_token_ids"),
                _required_bool(item, "include_eos"),
            )
        )
    return tuple(sorted(entries))


def _diagnostic_lock_entries(value: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        raise PromptPackLockError("prompt-pack lock field 'diagnostic_baseline' must be a list")
    entries: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise PromptPackLockError("prompt-pack lock diagnostic baseline entries must be objects")
        entries.append((_required_str(item, "rule_id"), _required_str(item, "severity"), _required_str(item, "fingerprint")))
    return tuple(sorted(entries))


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PromptPackLockError(f"prompt-pack lock field '{field_name}' must be a list of strings")
    return tuple(value)


def _int_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise PromptPackLockError(f"prompt-pack lock field '{field_name}' must be a list of integers")
    return tuple(value)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise PromptPackLockError(f"prompt-pack lock field '{key}' must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PromptPackLockError(f"prompt-pack lock field '{key}' must be a non-empty string when present")
    return value


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise PromptPackLockError(f"prompt-pack lock field '{key}' must be a boolean")
    return value


def _mirror_entry_from_mapping(data: dict[str, Any]) -> PromptPackMirrorEntry:
    registry_raw = data.get("registry_entry")
    if not isinstance(registry_raw, dict):
        raise PromptPackMirrorError("prompt-pack mirror entry field 'registry_entry' must be an object")
    registry_entry = _registry_entry_from_mapping(registry_raw)
    size_bytes = data.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise PromptPackMirrorError("prompt-pack mirror entry field 'size_bytes' must be a non-negative integer")
    return PromptPackMirrorEntry(
        name=_mirror_required_str(data, "name"),
        package_name=_mirror_required_str(data, "package_name"),
        version=_mirror_optional_str(data, "version"),
        source_location=_mirror_required_str(data, "source_location"),
        mirror_path=_mirror_required_str(data, "mirror_path"),
        sha256=_mirror_required_str(data, "sha256"),
        size_bytes=size_bytes,
        contract_hash=_mirror_required_str(data, "contract_hash"),
        proof_hash=_mirror_required_str(data, "proof_hash"),
        registry_entry=registry_entry,
    )


def _registry_entry_from_mapping(data: dict[str, Any]) -> PromptPackRegistryEntry:
    return PromptPackRegistryEntry(
        name=_mirror_required_str(data, "name"),
        package_name=_mirror_required_str(data, "package_name"),
        version=_mirror_optional_str(data, "version"),
        location=_mirror_required_str(data, "location"),
        sha256=_mirror_optional_str(data, "sha256"),
        contract_hash=_mirror_required_str(data, "contract_hash"),
        proof_hash=_mirror_required_str(data, "proof_hash"),
        supported_fragments=_dict_to_pairs(data.get("supported_fragments", {}), "supported_fragments"),
        reproducible_metadata=_dict_to_pairs(data.get("reproducible_metadata", {}), "reproducible_metadata"),
        proofs=_dict_to_pairs(data.get("proofs", {}), "proofs"),
        diagnostics=_diagnostic_lock_entries(data.get("diagnostics", [])),
    )


def _dict_to_pairs(value: object, field_name: str) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, dict):
        raise PromptPackMirrorError(f"prompt-pack mirror registry field '{field_name}' must be an object")
    return tuple(sorted(value.items(), key=lambda item: item[0]))


def _mirror_required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise PromptPackMirrorError(f"prompt-pack mirror field '{key}' must be a non-empty string")
    return value


def _mirror_optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PromptPackMirrorError(f"prompt-pack mirror field '{key}' must be a non-empty string when present")
    return value


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in value.strip())
    return safe.strip(".-") or "prompt-pack"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
