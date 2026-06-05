"""Add transitive lockfile proofs for prompt-pack registries (step 242).

A prompt-pack *registry* publishes versioned packs, and packs depend on other
packs.  A consumer that installs one root pack inherits its entire transitive
dependency closure -- and therefore the whole closure's ABI surface.  Before any
of that surface is trusted, the closure must be *resolved* and *pinned*: every
transitive dependency must exist at the requested version and be pinned to a
content digest.

This module resolves the transitive closure of a root pack against a
:class:`PackVersionRegistry`, refusing to produce a lockfile if anything is
unsound:

* a declared dependency is missing from the registry;
* two paths in the closure require *different* versions of the same pack (a
  diamond conflict);
* the dependency graph contains a cycle.

When resolution succeeds, :func:`resolve_transitive_closure` returns a
:class:`TransitiveLockfile` whose pins are canonically ordered and summarised by
a single Merkle-style ``root_digest``.  :func:`verify_transitive_lockfile`
re-checks an existing lockfile against the registry -- proving that every pin
still matches the published digest and that the recorded root digest is the one
implied by the pins -- so a tampered lockfile or a drifted registry is caught.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_REGISTRY_VERSION = "promptabi.prompt-pack-registry.v1"


class RegistryFindingKind(StrEnum):
    MISSING_DEPENDENCY = "missing-dependency"
    VERSION_CONFLICT = "version-conflict"
    DEPENDENCY_CYCLE = "dependency-cycle"
    DIGEST_MISMATCH = "digest-mismatch"
    ROOT_DIGEST_MISMATCH = "root-digest-mismatch"
    UNKNOWN_PIN = "unknown-pin"
    INCOMPLETE_CLOSURE = "incomplete-closure"


class RegistryResolutionError(ValueError):
    """Raised when a transitive closure cannot be resolved soundly."""

    def __init__(self, findings: "tuple[RegistryFinding, ...]") -> None:
        self.findings = findings
        super().__init__("; ".join(f.describe() for f in findings))


@dataclass(frozen=True, slots=True)
class RegistryFinding:
    kind: RegistryFindingKind
    pack: str
    detail: str = ""

    def describe(self) -> str:
        base = f"{self.kind.value}: {self.pack}"
        return f"{base} ({self.detail})" if self.detail else base

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "pack": self.pack, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PackDependency:
    name: str
    version: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One published pack version and the packs it depends on."""

    name: str
    version: str
    digest: str
    dependencies: tuple[PackDependency, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.version)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "dependencies": [d.to_dict() for d in self.dependencies],
        }


@dataclass(frozen=True, slots=True)
class LockfilePin:
    name: str
    version: str
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class TransitiveLockfile:
    version: str
    root_name: str
    root_version: str
    pins: tuple[LockfilePin, ...]
    root_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "root": {"name": self.root_name, "version": self.root_version},
            "pins": [p.to_dict() for p in self.pins],
            "root_digest": self.root_digest,
        }


@dataclass(frozen=True, slots=True)
class LockfileVerification:
    version: str
    valid: bool
    findings: tuple[RegistryFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "findings": [f.to_dict() for f in self.findings],
        }


class PackVersionRegistry:
    """An immutable index of published pack versions keyed by (name, version)."""

    def __init__(self, entries: "tuple[RegistryEntry, ...] | list[RegistryEntry]") -> None:
        index: dict[tuple[str, str], RegistryEntry] = {}
        for entry in entries:
            if entry.key in index:
                raise ValueError(f"duplicate registry entry: {entry.name}@{entry.version}")
            index[entry.key] = entry
        self._index = index

    def get(self, name: str, version: str) -> RegistryEntry | None:
        return self._index.get((name, version))

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._index


def _root_digest(pins: tuple[LockfilePin, ...]) -> str:
    payload = json.dumps([p.to_dict() for p in pins], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_pins(resolved: dict[tuple[str, str], RegistryEntry]) -> tuple[LockfilePin, ...]:
    return tuple(
        LockfilePin(name=entry.name, version=entry.version, digest=entry.digest)
        for entry in sorted(resolved.values(), key=lambda e: e.key)
    )


def resolve_transitive_closure(
    registry: PackVersionRegistry, root_name: str, root_version: str
) -> TransitiveLockfile:
    """Resolve and pin the full transitive closure of one root pack.

    Raises :class:`RegistryResolutionError` listing every missing dependency,
    version conflict, and cycle found, so the failure is fully diagnosable.
    """

    findings: list[RegistryFinding] = []
    resolved: dict[tuple[str, str], RegistryEntry] = {}
    # name -> the single version we have committed to (diamond-conflict guard)
    chosen_version: dict[str, str] = {}
    on_stack: set[tuple[str, str]] = set()

    def visit(name: str, version: str) -> None:
        prior = chosen_version.get(name)
        if prior is not None and prior != version:
            findings.append(
                RegistryFinding(
                    RegistryFindingKind.VERSION_CONFLICT,
                    name,
                    f"requires both {prior} and {version}",
                )
            )
            return
        key = (name, version)
        if key in resolved:
            return
        if key in on_stack:
            findings.append(
                RegistryFinding(
                    RegistryFindingKind.DEPENDENCY_CYCLE, name, f"at {name}@{version}"
                )
            )
            return
        entry = registry.get(name, version)
        if entry is None:
            findings.append(
                RegistryFinding(
                    RegistryFindingKind.MISSING_DEPENDENCY, name, f"version {version}"
                )
            )
            return
        chosen_version[name] = version
        on_stack.add(key)
        for dep in entry.dependencies:
            visit(dep.name, dep.version)
        on_stack.discard(key)
        resolved[key] = entry

    visit(root_name, root_version)

    if findings:
        raise RegistryResolutionError(tuple(findings))

    pins = _ordered_pins(resolved)
    return TransitiveLockfile(
        version=PROMPT_PACK_REGISTRY_VERSION,
        root_name=root_name,
        root_version=root_version,
        pins=pins,
        root_digest=_root_digest(pins),
    )


def verify_transitive_lockfile(
    lockfile: TransitiveLockfile, registry: PackVersionRegistry
) -> LockfileVerification:
    """Re-prove a lockfile against the registry it was resolved from."""

    findings: list[RegistryFinding] = []

    if _root_digest(lockfile.pins) != lockfile.root_digest:
        findings.append(
            RegistryFinding(
                RegistryFindingKind.ROOT_DIGEST_MISMATCH,
                lockfile.root_name,
                "recorded root digest does not match pins",
            )
        )

    for pin in lockfile.pins:
        entry = registry.get(pin.name, pin.version)
        if entry is None:
            findings.append(
                RegistryFinding(
                    RegistryFindingKind.UNKNOWN_PIN, pin.name, f"version {pin.version}"
                )
            )
            continue
        if entry.digest != pin.digest:
            findings.append(
                RegistryFinding(
                    RegistryFindingKind.DIGEST_MISMATCH,
                    pin.name,
                    f"registry {entry.digest} != pin {pin.digest}",
                )
            )

    # Completeness: every dependency reachable from the pinned set must itself
    # be pinned, otherwise the lockfile understates the trusted closure.
    pinned = {(p.name, p.version) for p in lockfile.pins}
    for pin in lockfile.pins:
        entry = registry.get(pin.name, pin.version)
        if entry is None:
            continue
        for dep in entry.dependencies:
            if (dep.name, dep.version) not in pinned:
                findings.append(
                    RegistryFinding(
                        RegistryFindingKind.INCOMPLETE_CLOSURE,
                        dep.name,
                        f"required by {pin.name}@{pin.version} but not pinned",
                    )
                )

    return LockfileVerification(
        version=PROMPT_PACK_REGISTRY_VERSION,
        valid=not findings,
        findings=tuple(findings),
    )


def render_lockfile_json(lockfile: TransitiveLockfile) -> str:
    return json.dumps(lockfile.to_dict(), indent=2, sort_keys=True) + "\n"


def render_verification_text(result: LockfileVerification) -> str:
    lines = [
        f"PromptABI prompt-pack lockfile verification ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'}",
    ]
    for finding in result.findings:
        lines.append(f"  ! {finding.describe()}")
    return "\n".join(lines) + "\n"
