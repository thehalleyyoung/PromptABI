"""Signed verification bundles for PromptABI audit trails."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import VerificationConfig, load_config
from .diagnostics import Diagnostic
from .loaders import LoadedArtifact
from .lockfiles import build_lockfile
from .session import VerificationResult, VerificationSession


VERIFICATION_BUNDLE_VERSION = 1
DEFAULT_BUNDLE_EXCERPT_BYTES = 4096
SIGNATURE_ALGORITHM = "hmac-sha256"


class VerificationBundleError(ValueError):
    """Raised when a signed verification bundle cannot be built or verified."""


@dataclass(frozen=True, slots=True)
class VerificationBundle:
    """A deterministic signed audit bundle for one PromptABI verification run."""

    payload: dict[str, object]
    signature: str
    signing_key_id: str
    algorithm: str = SIGNATURE_ALGORITHM

    @property
    def bundle_hash(self) -> str:
        return _stable_json_hash(self.payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "bundle_hash": self.bundle_hash,
            "payload": self.payload,
            "signature": self.signature,
            "signing_key_id": self.signing_key_id,
        }


@dataclass(frozen=True, slots=True)
class VerificationBundleVerification:
    """Result of checking a signed verification bundle."""

    ok: bool
    bundle_hash: str
    signing_key_id: str
    expected_signature: str
    actual_signature: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "actual_signature": self.actual_signature,
            "bundle_hash": self.bundle_hash,
            "expected_signature": self.expected_signature,
            "ok": self.ok,
            "signing_key_id": self.signing_key_id,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


def create_signed_verification_bundle(
    config_path: str | Path,
    *,
    key: str | bytes | None = None,
    key_id: str = "local",
    artifact_overrides: dict[str, str] | None = None,
    excerpt_bytes: int = DEFAULT_BUNDLE_EXCERPT_BYTES,
) -> VerificationBundle:
    """Run verification and package diagnostics, witnesses, lockfile state, and hashes."""

    if excerpt_bytes < 0:
        raise VerificationBundleError("excerpt_bytes must be non-negative")
    resolved_key = _resolve_key(key)
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    if artifact_overrides:
        config = config.with_artifact_overrides(artifact_overrides, base_dir=Path.cwd())
    session = VerificationSession(config)
    result = session.run()
    loaded_artifacts, _load_diagnostics = session.load_artifacts_with_diagnostics()
    payload = build_verification_bundle_payload(
        config,
        config_path=path,
        result=result,
        loaded_artifacts=loaded_artifacts,
        excerpt_bytes=excerpt_bytes,
    )
    return sign_verification_bundle_payload(payload, key=resolved_key, key_id=key_id)


def build_verification_bundle_payload(
    config: VerificationConfig,
    *,
    config_path: str | Path,
    result: VerificationResult,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    excerpt_bytes: int = DEFAULT_BUNDLE_EXCERPT_BYTES,
) -> dict[str, object]:
    """Build the unsigned, deterministic audit payload from a completed run."""

    path = Path(config_path).expanduser().resolve()
    lockfile = build_lockfile(config, loaded_artifacts, result.diagnostics, base_dir=path.parent)
    diagnostics = [diagnostic.to_dict() for diagnostic in result.diagnostics]
    witness_hashes = _witness_hashes(result.diagnostics)
    artifacts = [_bundle_artifact(loaded, base_dir=path.parent, excerpt_bytes=excerpt_bytes) for loaded in loaded_artifacts]
    payload: dict[str, object] = {
        "bundle_version": VERIFICATION_BUNDLE_VERSION,
        "config": {
            "hash": _hash_file(path),
            "name": config.name,
            "path": _portable_path(path, path.parent),
        },
        "diagnostics": diagnostics,
        "environment": _environment(),
        "lockfile": lockfile.to_dict(),
        "promptabi_version": __version__,
        "reproducibility": {
            "artifact_count": len(loaded_artifacts),
            "diagnostic_count": len(result.diagnostics),
            "error_count": sum(1 for diagnostic in result.diagnostics if diagnostic.severity.value == "error"),
            "ok": result.ok,
            "witness_hashes": witness_hashes,
        },
        "artifacts": artifacts,
        "solver_metadata": _solver_metadata(result.diagnostics),
    }
    payload["reproducibility_hash"] = _stable_json_hash(
        {
            "artifact_hashes": [
                {
                    "name": artifact["name"],
                    "sha256": artifact.get("sha256"),
                    "manifest_sha256": artifact.get("manifest_sha256"),
                }
                for artifact in artifacts
            ],
            "config_hash": payload["config"]["hash"],  # type: ignore[index]
            "diagnostic_fingerprints": [diagnostic["fingerprint"] for diagnostic in diagnostics],
            "lockfile_hash": _stable_json_hash(payload["lockfile"]),
            "witness_hashes": witness_hashes,
        }
    )
    return payload


def sign_verification_bundle_payload(
    payload: dict[str, object],
    *,
    key: str | bytes | None = None,
    key_id: str = "local",
) -> VerificationBundle:
    """Sign a bundle payload using a deterministic local HMAC key."""

    resolved_key = _resolve_key(key)
    signature = _signature(payload, resolved_key)
    return VerificationBundle(payload=payload, signature=signature, signing_key_id=key_id)


def verify_signed_verification_bundle(
    bundle: VerificationBundle | dict[str, object] | str | Path,
    *,
    key: str | bytes | None = None,
) -> VerificationBundleVerification:
    """Verify a signed bundle without rerunning PromptABI checks."""

    data = _bundle_mapping(bundle)
    algorithm = data.get("algorithm")
    if algorithm != SIGNATURE_ALGORITHM:
        raise VerificationBundleError(f"unsupported bundle signature algorithm: {algorithm!r}")
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise VerificationBundleError("bundle payload must be an object")
    actual = _required_str(data, "signature")
    key_id = _required_str(data, "signing_key_id")
    expected = _signature(payload, _resolve_key(key))
    ok = hmac.compare_digest(actual, expected)
    return VerificationBundleVerification(
        ok=ok,
        bundle_hash=_stable_json_hash(payload),
        signing_key_id=key_id,
        expected_signature=expected,
        actual_signature=actual,
        reason=None if ok else "signature mismatch",
    )


def write_signed_verification_bundle(
    path: str | Path,
    bundle: VerificationBundle,
    *,
    force: bool = False,
) -> None:
    """Write a signed bundle JSON file, refusing accidental overwrites by default."""

    destination = Path(path)
    if destination.exists() and not force:
        raise VerificationBundleError(f"bundle already exists: {destination}; pass --force to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_verification_bundle_json(bundle), encoding="utf-8")


def load_signed_verification_bundle(path: str | Path) -> VerificationBundle:
    """Load a signed bundle from JSON."""

    data = _bundle_mapping(path)
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise VerificationBundleError("bundle payload must be an object")
    return VerificationBundle(
        payload=payload,
        signature=_required_str(data, "signature"),
        signing_key_id=_required_str(data, "signing_key_id"),
        algorithm=_required_str(data, "algorithm"),
    )


def render_verification_bundle_json(bundle: VerificationBundle) -> str:
    return json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n"


def render_bundle_verification_text(result: VerificationBundleVerification) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [
        "PromptABI signed bundle verification",
        f"status: {status}",
        f"bundle_hash: {result.bundle_hash}",
        f"signing_key_id: {result.signing_key_id}",
    ]
    if result.reason is not None:
        lines.append(f"reason: {result.reason}")
    return "\n".join(lines) + "\n"


def _bundle_artifact(loaded: LoadedArtifact, *, base_dir: Path, excerpt_bytes: int) -> dict[str, object]:
    artifact = loaded.artifact
    data: dict[str, object] = {
        "kind": artifact.kind.value,
        "location": _portable_path_string(artifact.location.ref_path, base_dir=base_dir),
        "name": artifact.name,
        "pinned": loaded.pinned,
        "resolved": loaded.resolved,
        "source_type": loaded.source_type,
    }
    for key, value in (
        ("sha256", loaded.actual_sha256 or artifact.provenance.sha256),
        ("manifest_sha256", loaded.manifest_sha256),
        ("size_bytes", loaded.size_bytes),
        ("revision", artifact.provenance.revision),
        ("version", artifact.provenance.version),
        ("license", artifact.provenance.license),
        ("source", artifact.provenance.source),
    ):
        if value is not None:
            data[key] = value
    if loaded.members:
        data["members"] = list(loaded.members)
    excerpt = _artifact_excerpt(artifact.location.ref_path, base_dir=base_dir, max_bytes=excerpt_bytes)
    if excerpt is not None:
        data["excerpt"] = excerpt
    data["metadata_hash"] = _stable_json_hash(loaded.metadata)
    return data


def _artifact_excerpt(path_value: str | None, *, base_dir: Path, max_bytes: int) -> dict[str, object] | None:
    if path_value is None or "://" in path_value or max_bytes == 0:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        return None
    raw = path.read_bytes()
    excerpt = raw[:max_bytes]
    return {
        "base64": base64.b64encode(excerpt).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > len(excerpt),
    }


def _witness_hashes(diagnostics: tuple[Diagnostic, ...]) -> list[dict[str, object]]:
    entries = []
    for diagnostic in diagnostics:
        if diagnostic.witness is None:
            continue
        witness = diagnostic.witness.to_dict()
        entries.append(
            {
                "diagnostic_fingerprint": diagnostic.fingerprint,
                "rule_id": diagnostic.rule_id,
                "witness_sha256": _stable_json_hash(witness),
            }
        )
    return entries


def _solver_metadata(diagnostics: tuple[Diagnostic, ...]) -> dict[str, object]:
    solver_diagnostics = []
    for diagnostic in diagnostics:
        modes = [mode.value for mode in diagnostic.check_modes]
        if "z3-backed-smt" not in modes and not diagnostic.rule_id.startswith("static-contract"):
            continue
        solver_diagnostics.append(
            {
                "check_modes": modes,
                "fingerprint": diagnostic.fingerprint,
                "properties_hash": _stable_json_hash(dict(diagnostic.properties)),
                "rule_id": diagnostic.rule_id,
            }
        )
    return {
        "diagnostic_count": len(solver_diagnostics),
        "diagnostics": solver_diagnostics,
        "z3_version": _z3_version(),
    }


def _environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _z3_version() -> str | None:
    try:
        import z3  # type: ignore
    except Exception:
        return None
    return str(z3.get_version_string())


def _resolve_key(key: str | bytes | None) -> bytes:
    value = key if key is not None else os.environ.get("PROMPTABI_BUNDLE_KEY")
    if value is None:
        raise VerificationBundleError("a signing key is required via --key or PROMPTABI_BUNDLE_KEY")
    if isinstance(value, bytes):
        resolved = value
    else:
        resolved = value.encode("utf-8")
    if not resolved:
        raise VerificationBundleError("signing key must be non-empty")
    return resolved


def _signature(payload: dict[str, object], key: bytes) -> str:
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _bundle_mapping(bundle: VerificationBundle | dict[str, object] | str | Path) -> dict[str, object]:
    if isinstance(bundle, VerificationBundle):
        return bundle.to_dict()
    if isinstance(bundle, dict):
        return bundle
    path = Path(bundle)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationBundleError(f"bundle not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationBundleError(f"bundle is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise VerificationBundleError("bundle root must be an object")
    return data


def _required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise VerificationBundleError(f"bundle field '{key}' must be a non-empty string")
    return value


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_path_string(path_value: str | None, *, base_dir: Path) -> str | None:
    if path_value is None or "://" in path_value:
        return path_value
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    return _portable_path(path, base_dir)


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
