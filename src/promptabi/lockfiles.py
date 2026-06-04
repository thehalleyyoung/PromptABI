"""Deterministic lockfiles for verified PromptABI artifact contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import VerificationConfig
from .diagnostics import ArtifactRef, CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace
from .loaders import LoadedArtifact


LOCKFILE_VERSION = 1
LOCKFILE_CHECK_MODES = (CheckMode.SOUND, CheckMode.COMPLETE)


class LockfileError(ValueError):
    """Raised when a PromptABI lockfile cannot be read or written soundly."""


@dataclass(frozen=True, slots=True)
class LockfileArtifact:
    """The reproducibility-relevant state for one loaded artifact."""

    name: str
    kind: str
    location: str
    source_type: str
    resolved: bool
    pinned: bool
    sha256: str | None = None
    manifest_sha256: str | None = None
    size_bytes: int | None = None
    version: str | None = None
    revision: str | None = None
    license: str | None = None
    source: str | None = None
    members: tuple[str, ...] = ()
    supported_fragments: tuple[tuple[str, object], ...] = ()
    metadata_fingerprint: str | None = None

    @classmethod
    def from_loaded(cls, loaded: LoadedArtifact, *, base_dir: Path | None = None) -> "LockfileArtifact":
        artifact = loaded.artifact
        metadata = dict(loaded.metadata)
        sha256 = loaded.actual_sha256 or artifact.provenance.sha256
        manifest_sha256 = loaded.manifest_sha256
        revision = artifact.provenance.revision or _string_metadata(metadata, "revision")
        version = artifact.provenance.version or _string_metadata(metadata, "version")
        location = _portable_path_string(artifact.location.ref_path or "", base_dir=base_dir)
        return cls(
            name=artifact.name,
            kind=artifact.kind.value,
            location=location,
            source_type=loaded.source_type,
            resolved=loaded.resolved,
            pinned=loaded.pinned,
            sha256=sha256,
            manifest_sha256=manifest_sha256,
            size_bytes=loaded.size_bytes,
            version=version,
            revision=revision,
            license=artifact.provenance.license,
            source=artifact.provenance.source,
            members=loaded.members,
            supported_fragments=_supported_fragment_metadata(metadata),
            metadata_fingerprint=_metadata_fingerprint(metadata),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LockfileArtifact":
        required = ("name", "kind", "location", "source_type", "resolved", "pinned")
        missing = [key for key in required if key not in data]
        if missing:
            raise LockfileError(f"lockfile artifact entry is missing fields: {', '.join(missing)}")
        supported = data.get("supported_fragments", {})
        if not isinstance(supported, dict):
            raise LockfileError("lockfile artifact field 'supported_fragments' must be an object")
        members = data.get("members", [])
        if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
            raise LockfileError("lockfile artifact field 'members' must be a list of strings")
        return cls(
            name=_required_str(data, "name"),
            kind=_required_str(data, "kind"),
            location=_required_str(data, "location"),
            source_type=_required_str(data, "source_type"),
            resolved=_required_bool(data, "resolved"),
            pinned=_required_bool(data, "pinned"),
            sha256=_optional_str(data, "sha256"),
            manifest_sha256=_optional_str(data, "manifest_sha256"),
            size_bytes=_optional_int(data, "size_bytes"),
            version=_optional_str(data, "version"),
            revision=_optional_str(data, "revision"),
            license=_optional_str(data, "license"),
            source=_optional_str(data, "source"),
            members=tuple(members),
            supported_fragments=tuple(sorted(supported.items())),
            metadata_fingerprint=_optional_str(data, "metadata_fingerprint"),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "location": self.location,
            "name": self.name,
            "pinned": self.pinned,
            "resolved": self.resolved,
            "source_type": self.source_type,
        }
        for key in ("sha256", "manifest_sha256", "size_bytes", "version", "revision", "license", "source"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.members:
            data["members"] = list(self.members)
        if self.supported_fragments:
            data["supported_fragments"] = dict(self.supported_fragments)
        if self.metadata_fingerprint is not None:
            data["metadata_fingerprint"] = self.metadata_fingerprint
        return data


@dataclass(frozen=True, slots=True)
class Lockfile:
    """A reproducible PromptABI verification snapshot."""

    config_name: str
    config_hash: str
    artifacts: tuple[LockfileArtifact, ...]
    checks: tuple[str, ...]
    diagnostic_baseline: tuple[tuple[str, str, str, str | None], ...]
    library_versions: tuple[tuple[str, str], ...]
    provider_fixture_versions: tuple[tuple[str, str], ...] = ()
    promptabi_version: str = __version__
    lockfile_version: int = LOCKFILE_VERSION

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Lockfile":
        if data.get("lockfile_version") != LOCKFILE_VERSION:
            raise LockfileError(f"unsupported PromptABI lockfile version: {data.get('lockfile_version')!r}")
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise LockfileError("lockfile field 'artifacts' must be a list")
        raw_checks = data.get("checks", [])
        raw_baseline = data.get("diagnostic_baseline", [])
        raw_libraries = data.get("library_versions", {})
        raw_fixtures = data.get("provider_fixture_versions", {})
        if not isinstance(raw_checks, list) or not all(isinstance(item, str) for item in raw_checks):
            raise LockfileError("lockfile field 'checks' must be a list of strings")
        if not isinstance(raw_baseline, list):
            raise LockfileError("lockfile field 'diagnostic_baseline' must be a list")
        if not isinstance(raw_libraries, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_libraries.items()
        ):
            raise LockfileError("lockfile field 'library_versions' must be an object of strings")
        if not isinstance(raw_fixtures, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_fixtures.items()
        ):
            raise LockfileError("lockfile field 'provider_fixture_versions' must be an object of strings")
        return cls(
            config_name=_required_str(data, "config_name"),
            config_hash=_required_str(data, "config_hash"),
            artifacts=tuple(sorted((LockfileArtifact.from_mapping(item) for item in raw_artifacts), key=_artifact_key)),
            checks=tuple(raw_checks),
            diagnostic_baseline=tuple(sorted(_baseline_entry(item) for item in raw_baseline)),
            library_versions=tuple(sorted(raw_libraries.items())),
            provider_fixture_versions=tuple(sorted(raw_fixtures.items())),
            promptabi_version=_required_str(data, "promptabi_version"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checks": list(self.checks),
            "config_hash": self.config_hash,
            "config_name": self.config_name,
            "diagnostic_baseline": [
                {
                    "artifact": artifact_name,
                    "fingerprint": fingerprint,
                    "rule_id": rule_id,
                    "severity": severity,
                }
                for rule_id, severity, fingerprint, artifact_name in self.diagnostic_baseline
            ],
            "library_versions": dict(self.library_versions),
            "lockfile_version": self.lockfile_version,
            "promptabi_version": self.promptabi_version,
            "provider_fixture_versions": dict(self.provider_fixture_versions),
        }


def build_lockfile(
    config: VerificationConfig,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    base_dir: str | Path | None = None,
) -> Lockfile:
    """Build a deterministic lockfile from the exact artifacts and diagnostics just verified."""

    resolved_base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    return Lockfile(
        config_name=config.name,
        config_hash=_config_hash(config, base_dir=resolved_base),
        artifacts=tuple(
            sorted(
                (LockfileArtifact.from_loaded(loaded, base_dir=resolved_base) for loaded in loaded_artifacts),
                key=_artifact_key,
            )
        ),
        checks=tuple(config.checks),
        diagnostic_baseline=_diagnostic_baseline(diagnostics),
        library_versions=_library_versions(),
        provider_fixture_versions=_provider_fixture_versions(loaded_artifacts),
    )


def lockfile_to_json(lockfile: Lockfile) -> str:
    return json.dumps(lockfile.to_dict(), indent=2, sort_keys=True) + "\n"


def load_lockfile(path: str | Path) -> Lockfile:
    lockfile_path = Path(path)
    try:
        raw = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LockfileError(f"lockfile not found: {lockfile_path}") from exc
    except json.JSONDecodeError as exc:
        raise LockfileError(
            f"lockfile is not valid JSON at {lockfile_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise LockfileError("lockfile root must be a JSON object")
    return Lockfile.from_mapping(raw)


def write_lockfile(path: str | Path, lockfile: Lockfile) -> None:
    lockfile_path = Path(path)
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile_path.write_text(lockfile_to_json(lockfile), encoding="utf-8")


def compare_lockfile(
    lockfile: Lockfile,
    config: VerificationConfig,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    diagnostics: tuple[Diagnostic, ...] = (),
    *,
    lockfile_path: str | Path | None = None,
) -> tuple[Diagnostic, ...]:
    """Return diagnostics for lockfile drift against the current verification state."""

    base_dir = Path(lockfile_path).expanduser().resolve().parent if lockfile_path is not None else None
    current = build_lockfile(config, loaded_artifacts, diagnostics, base_dir=base_dir)
    drift: list[Diagnostic] = []
    if lockfile.config_name != current.config_name:
        drift.append(
            _lock_diagnostic(
                "lockfile-config-drift",
                "lockfile was created for a different PromptABI config name",
                "config_name",
                lockfile.config_name,
                current.config_name,
                lockfile_path=lockfile_path,
            )
        )
    if lockfile.config_hash != current.config_hash:
        drift.append(
            _lock_diagnostic(
                "lockfile-config-drift",
                "PromptABI config contract changed since the lockfile was written",
                "config_hash",
                lockfile.config_hash,
                current.config_hash,
                lockfile_path=lockfile_path,
            )
        )
    drift.extend(_artifact_drift_diagnostics(lockfile, current, lockfile_path=lockfile_path))
    if lockfile.diagnostic_baseline != current.diagnostic_baseline:
        drift.append(
            _lock_diagnostic(
                "lockfile-diagnostic-baseline-drift",
                "verification diagnostics differ from the lockfile baseline",
                "diagnostic_baseline",
                _baseline_summary(lockfile.diagnostic_baseline),
                _baseline_summary(current.diagnostic_baseline),
                lockfile_path=lockfile_path,
            )
        )
    if dict(lockfile.provider_fixture_versions) != dict(current.provider_fixture_versions):
        drift.append(
            _lock_diagnostic(
                "lockfile-provider-fixture-drift",
                "provider fixture versions differ from the lockfile",
                "provider_fixture_versions",
                str(dict(lockfile.provider_fixture_versions)),
                str(dict(current.provider_fixture_versions)),
                lockfile_path=lockfile_path,
            )
        )
    if dict(lockfile.library_versions) != dict(current.library_versions):
        drift.append(
            _lock_diagnostic(
                "lockfile-library-version-drift",
                "verification library versions differ from the lockfile",
                "library_versions",
                str(dict(lockfile.library_versions)),
                str(dict(current.library_versions)),
                lockfile_path=lockfile_path,
            )
        )
    if not drift:
        return (
            _lock_diagnostic(
                "lockfile-verified",
                "PromptABI lockfile matches the current verified artifacts and diagnostic baseline",
                "lockfile",
                "matched",
                "matched",
                severity=DiagnosticSeverity.INFO,
                lockfile_path=lockfile_path,
            ),
        )
    return tuple(drift)


def lockfile_error_diagnostic(exc: LockfileError, *, lockfile_path: str | Path | None = None) -> Diagnostic:
    path = str(lockfile_path) if lockfile_path is not None else None
    return Diagnostic(
        rule_id="lockfile-load-failed",
        severity=DiagnosticSeverity.ERROR,
        message=str(exc),
        artifact=ArtifactRef(kind="lockfile", name="promptabi-lockfile", path=path) if path else None,
        check_modes=LOCKFILE_CHECK_MODES,
        suggestions=("Run promptabi verify --write-lockfile after reviewing the current artifacts.",),
        witness=WitnessTrace(
            summary="PromptABI could not load the lockfile for enforcement.",
            steps=(WitnessStep(action="load lockfile", input=path, output=str(exc)),),
        ),
    )


def _artifact_drift_diagnostics(
    expected: Lockfile,
    current: Lockfile,
    *,
    lockfile_path: str | Path | None,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    expected_by_name = {artifact.name: artifact for artifact in expected.artifacts}
    current_by_name = {artifact.name: artifact for artifact in current.artifacts}
    for name in sorted(expected_by_name.keys() - current_by_name.keys()):
        diagnostics.append(
            _artifact_lock_diagnostic(
                expected_by_name[name],
                "lockfile-artifact-missing",
                f"artifact '{name}' is present in the lockfile but missing from current verification",
                "artifact",
                "present",
                "missing",
                lockfile_path=lockfile_path,
            )
        )
    for name in sorted(current_by_name.keys() - expected_by_name.keys()):
        diagnostics.append(
            _artifact_lock_diagnostic(
                current_by_name[name],
                "lockfile-artifact-added",
                f"artifact '{name}' is new relative to the lockfile",
                "artifact",
                "missing",
                "present",
                lockfile_path=lockfile_path,
            )
        )
    for name in sorted(expected_by_name.keys() & current_by_name.keys()):
        old = expected_by_name[name]
        new = current_by_name[name]
        for field in (
            "kind",
            "location",
            "source_type",
            "sha256",
            "manifest_sha256",
            "revision",
            "version",
            "metadata_fingerprint",
            "supported_fragments",
        ):
            if getattr(old, field) != getattr(new, field):
                diagnostics.append(
                    _artifact_lock_diagnostic(
                        new,
                        "lockfile-artifact-drift",
                        f"artifact '{name}' {field.replace('_', ' ')} differs from the lockfile",
                        field,
                        str(getattr(old, field)),
                        str(getattr(new, field)),
                        lockfile_path=lockfile_path,
                    )
                )
    return tuple(diagnostics)


def _artifact_lock_diagnostic(
    artifact: LockfileArtifact,
    rule_id: str,
    message: str,
    field: str,
    expected: str,
    actual: str,
    *,
    lockfile_path: str | Path | None,
) -> Diagnostic:
    return _lock_diagnostic(
        rule_id,
        message,
        field,
        expected,
        actual,
        artifact=ArtifactRef(kind=artifact.kind, name=artifact.name, path=artifact.location if "://" not in artifact.location else None, uri=artifact.location if "://" in artifact.location else None),
        lockfile_path=lockfile_path,
    )


def _lock_diagnostic(
    rule_id: str,
    message: str,
    field: str,
    expected: str,
    actual: str,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    artifact: ArtifactRef | None = None,
    lockfile_path: str | Path | None,
) -> Diagnostic:
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=artifact,
        check_modes=LOCKFILE_CHECK_MODES,
        suggestions=("Regenerate the lockfile only after reviewing the artifact and diagnostic changes.",),
        witness=WitnessTrace(
            summary="The enforced PromptABI lockfile does not match the current verification state."
            if severity is DiagnosticSeverity.ERROR
            else "The enforced PromptABI lockfile matches the current verification state.",
            steps=(
                WitnessStep(action="read lockfile", input=str(lockfile_path) if lockfile_path is not None else None),
                WitnessStep(action=f"compare {field}", input=expected, output=actual),
            ),
            artifacts=(artifact,) if artifact is not None else (),
        ),
        properties=(("actual", actual), ("expected", expected), ("field", field)),
    )


def _config_hash(config: VerificationConfig, *, base_dir: Path | None = None) -> str:
    payload = {
        "artifacts": config.artifact_bundle.to_dict(),
        "checks": list(config.checks),
        "max_context_tokens": config.max_context_tokens,
        "name": config.name,
    }
    encoded = json.dumps(_jsonable(payload, base_dir=base_dir), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _diagnostic_baseline(diagnostics: tuple[Diagnostic, ...]) -> tuple[tuple[str, str, str, str | None], ...]:
    return tuple(
        sorted(
            (
                diagnostic.rule_id,
                diagnostic.severity.value,
                diagnostic.fingerprint,
                diagnostic.artifact.name if diagnostic.artifact is not None else None,
            )
            for diagnostic in diagnostics
            if not diagnostic.rule_id.startswith("lockfile-")
        )
    )


def _library_versions() -> tuple[tuple[str, str], ...]:
    names = ("jsonschema", "sentencepiece", "tiktoken", "tokenizers", "z3-solver")
    versions = {
        "promptabi": __version__,
        "python": platform.python_version(),
    }
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return tuple(sorted(versions.items()))


def _provider_fixture_versions(loaded_artifacts: tuple[LoadedArtifact, ...]) -> tuple[tuple[str, str], ...]:
    versions: dict[str, str] = {}
    for loaded in loaded_artifacts:
        metadata = dict(loaded.metadata)
        provider = _string_metadata(metadata, "provider") or loaded.artifact.name
        revision = (
            _string_metadata(metadata, "fixture_revision")
            or _string_metadata(metadata, "upstream_revision")
            or _string_metadata(metadata, "version")
        )
        if loaded.source_type in {"provider-config-snapshot", "provider-fixture-pack"} and revision is not None:
            versions[provider] = revision
    return tuple(sorted(versions.items()))


def _supported_fragment_metadata(metadata: dict[str, object]) -> tuple[tuple[str, object], ...]:
    prefixes = ("supported", "symbolic_supported", "role_boundary_supported")
    suffixes = ("supported_fragment", "grammar_type", "source_family", "provider")
    selected = {
        key: value
        for key, value in metadata.items()
        if any(key.startswith(prefix) for prefix in prefixes) or any(key.endswith(suffix) for suffix in suffixes)
    }
    return tuple(sorted((key, _jsonable(value)) for key, value in selected.items()))


def _metadata_fingerprint(metadata: dict[str, object]) -> str | None:
    if not metadata:
        return None
    encoded = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object, *, base_dir: Path | None = None) -> object:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, base_dir=base_dir)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, base_dir=base_dir) for item in value]
    if isinstance(value, str):
        return _portable_path_string(value, base_dir=base_dir)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


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


def _string_metadata(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _artifact_key(artifact: LockfileArtifact) -> tuple[str, str]:
    return (artifact.kind, artifact.name)


def _baseline_entry(data: object) -> tuple[str, str, str, str | None]:
    if not isinstance(data, dict):
        raise LockfileError("diagnostic baseline entries must be objects")
    artifact = data.get("artifact")
    if artifact is not None and not isinstance(artifact, str):
        raise LockfileError("diagnostic baseline artifact must be a string when present")
    return (
        _required_str(data, "rule_id"),
        _required_str(data, "severity"),
        _required_str(data, "fingerprint"),
        artifact,
    )


def _baseline_summary(baseline: tuple[tuple[str, str, str, str | None], ...]) -> str:
    return ", ".join(f"{rule_id}:{severity}:{fingerprint}" for rule_id, severity, fingerprint, _ in baseline) or "(none)"


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise LockfileError(f"lockfile field '{key}' must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LockfileError(f"lockfile field '{key}' must be a non-empty string when present")
    return value


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise LockfileError(f"lockfile field '{key}' must be a boolean")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise LockfileError(f"lockfile field '{key}' must be an integer when present")
    return value
