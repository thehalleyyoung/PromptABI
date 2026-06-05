"""Prompt-pack parsing, package locks, and compatibility checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ChatTemplateArtifact,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
)
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


class PromptPackLockError(ValueError):
    """Raised when a prompt-pack lockfile cannot be read or compared."""


class PromptPackFindingKind(StrEnum):
    """Prompt-pack compatibility outcomes."""

    EMPTY_PACK = "empty-pack"
    TEMPLATE_ROLE_MISMATCH = "template-role-mismatch"
    APP_ROLE_MISSING = "app-role-missing"
    TOOL_MISSING = "tool-missing"
    STOP_MISSING = "stop-missing"
    MODEL_FAMILY_UNSUPPORTED = "model-family-unsupported"
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


def analyze_prompt_pack_contracts(
    prompt_pack: PromptPackArtifact,
    artifacts: tuple[Artifact, ...],
    *,
    source_spans: dict[str, SourceSpan] | None = None,
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

    if not findings:
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
