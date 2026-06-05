"""Tests for transitive lockfile proofs for prompt-pack registries (step 242)."""

from __future__ import annotations

import dataclasses

import pytest

from promptabi.prompt_pack_registry import (
    LockfilePin,
    PackDependency,
    PackVersionRegistry,
    RegistryEntry,
    RegistryFindingKind,
    RegistryResolutionError,
    render_lockfile_json,
    render_verification_text,
    resolve_transitive_closure,
    verify_transitive_lockfile,
)


def _registry() -> PackVersionRegistry:
    return PackVersionRegistry(
        [
            RegistryEntry(
                "app",
                "1.0.0",
                "d-app",
                dependencies=(
                    PackDependency("base", "2.0.0"),
                    PackDependency("tools", "1.5.0"),
                ),
            ),
            RegistryEntry(
                "tools",
                "1.5.0",
                "d-tools",
                dependencies=(PackDependency("base", "2.0.0"),),
            ),
            RegistryEntry("base", "2.0.0", "d-base"),
        ]
    )


def test_resolves_full_closure_with_diamond() -> None:
    lock = resolve_transitive_closure(_registry(), "app", "1.0.0")
    names = {p.name for p in lock.pins}
    assert names == {"app", "base", "tools"}
    # base appears once despite two paths to it
    assert sum(1 for p in lock.pins if p.name == "base") == 1
    assert lock.root_digest


def test_pins_are_canonically_ordered_and_digest_stable() -> None:
    lock_a = resolve_transitive_closure(_registry(), "app", "1.0.0")
    lock_b = resolve_transitive_closure(_registry(), "app", "1.0.0")
    assert [p.name for p in lock_a.pins] == sorted(p.name for p in lock_a.pins)
    assert lock_a.root_digest == lock_b.root_digest


def test_missing_dependency_is_rejected() -> None:
    registry = PackVersionRegistry(
        [RegistryEntry("app", "1.0.0", "d", (PackDependency("gone", "9.9.9"),))]
    )
    with pytest.raises(RegistryResolutionError) as exc:
        resolve_transitive_closure(registry, "app", "1.0.0")
    kinds = {f.kind for f in exc.value.findings}
    assert RegistryFindingKind.MISSING_DEPENDENCY in kinds


def test_version_conflict_is_rejected() -> None:
    registry = PackVersionRegistry(
        [
            RegistryEntry(
                "app",
                "1.0.0",
                "d",
                (PackDependency("a", "1.0.0"), PackDependency("b", "1.0.0")),
            ),
            RegistryEntry("a", "1.0.0", "da", (PackDependency("base", "1.0.0"),)),
            RegistryEntry("b", "1.0.0", "db", (PackDependency("base", "2.0.0"),)),
            RegistryEntry("base", "1.0.0", "db1"),
            RegistryEntry("base", "2.0.0", "db2"),
        ]
    )
    with pytest.raises(RegistryResolutionError) as exc:
        resolve_transitive_closure(registry, "app", "1.0.0")
    assert any(
        f.kind is RegistryFindingKind.VERSION_CONFLICT for f in exc.value.findings
    )


def test_cycle_is_detected() -> None:
    registry = PackVersionRegistry(
        [
            RegistryEntry("a", "1.0.0", "da", (PackDependency("b", "1.0.0"),)),
            RegistryEntry("b", "1.0.0", "db", (PackDependency("a", "1.0.0"),)),
        ]
    )
    with pytest.raises(RegistryResolutionError) as exc:
        resolve_transitive_closure(registry, "a", "1.0.0")
    assert any(
        f.kind is RegistryFindingKind.DEPENDENCY_CYCLE for f in exc.value.findings
    )


def test_verify_accepts_clean_lockfile() -> None:
    registry = _registry()
    lock = resolve_transitive_closure(registry, "app", "1.0.0")
    result = verify_transitive_lockfile(lock, registry)
    assert result.valid
    assert result.findings == ()


def test_verify_detects_tampered_pin_digest() -> None:
    registry = _registry()
    lock = resolve_transitive_closure(registry, "app", "1.0.0")
    tampered_pins = tuple(
        dataclasses.replace(p, digest="forged") if p.name == "base" else p
        for p in lock.pins
    )
    tampered = dataclasses.replace(lock, pins=tampered_pins)
    result = verify_transitive_lockfile(tampered, registry)
    assert not result.valid
    kinds = {f.kind for f in result.findings}
    # root digest no longer matches the mutated pins, and the pin disagrees
    assert RegistryFindingKind.ROOT_DIGEST_MISMATCH in kinds
    assert RegistryFindingKind.DIGEST_MISMATCH in kinds


def test_verify_detects_registry_drift() -> None:
    registry = _registry()
    lock = resolve_transitive_closure(registry, "app", "1.0.0")
    drifted = PackVersionRegistry(
        [
            RegistryEntry(
                "app",
                "1.0.0",
                "d-app",
                (PackDependency("base", "2.0.0"), PackDependency("tools", "1.5.0")),
            ),
            RegistryEntry(
                "tools", "1.5.0", "d-tools", (PackDependency("base", "2.0.0"),)
            ),
            RegistryEntry("base", "2.0.0", "d-base-CHANGED"),
        ]
    )
    result = verify_transitive_lockfile(lock, drifted)
    assert not result.valid
    assert any(
        f.kind is RegistryFindingKind.DIGEST_MISMATCH for f in result.findings
    )


def test_verify_detects_incomplete_closure() -> None:
    registry = _registry()
    full = resolve_transitive_closure(registry, "app", "1.0.0")
    truncated = dataclasses.replace(
        full, pins=tuple(p for p in full.pins if p.name != "base")
    )
    # recompute root digest so only the completeness check fires
    from promptabi.prompt_pack_registry import _root_digest  # type: ignore

    truncated = dataclasses.replace(
        truncated, root_digest=_root_digest(truncated.pins)
    )
    result = verify_transitive_lockfile(truncated, registry)
    assert not result.valid
    assert any(
        f.kind is RegistryFindingKind.INCOMPLETE_CLOSURE for f in result.findings
    )


def test_renderers_emit_text() -> None:
    registry = _registry()
    lock = resolve_transitive_closure(registry, "app", "1.0.0")
    assert "root_digest" in render_lockfile_json(lock)
    result = verify_transitive_lockfile(lock, registry)
    assert "VALID" in render_verification_text(result)


def test_unknown_pin_in_lockfile_is_flagged() -> None:
    registry = _registry()
    lock = resolve_transitive_closure(registry, "app", "1.0.0")
    extra = lock.pins + (LockfilePin("ghost", "1.0.0", "x"),)
    from promptabi.prompt_pack_registry import _root_digest  # type: ignore

    mutated = dataclasses.replace(lock, pins=extra, root_digest=_root_digest(extra))
    result = verify_transitive_lockfile(mutated, registry)
    assert any(f.kind is RegistryFindingKind.UNKNOWN_PIN for f in result.findings)
