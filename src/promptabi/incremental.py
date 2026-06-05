"""Incremental verification planning and cache reuse for monorepos."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, WitnessStep, WitnessTrace, diagnostic_sort_key
from .loaders import LoadedArtifact
from .session import (
    CheckContext,
    ScheduledDiagnostic,
    VerificationResult,
    VerificationSession,
    _check_runtimes_from_scheduled,
)
from .policies import apply_org_policy_diagnostics, apply_policy_diagnostics


@dataclass(frozen=True, slots=True)
class IncrementalPlan:
    """A deterministic selection of checks to recompute for changed local inputs."""

    changed_paths: tuple[Path, ...]
    changed_artifacts: tuple[str, ...]
    changed_kinds: tuple[ArtifactKind, ...]
    selected_checks: tuple[str, ...]
    skipped_checks: tuple[str, ...]
    full_run_reason: str | None = None

    @property
    def full_run(self) -> bool:
        return self.full_run_reason is not None

    def to_properties(self) -> tuple[tuple[str, object], ...]:
        return (
            ("changed_artifacts", list(self.changed_artifacts)),
            ("changed_kinds", [kind.value for kind in self.changed_kinds]),
            ("changed_paths", [str(path) for path in self.changed_paths]),
            ("full_run", self.full_run),
            ("full_run_reason", self.full_run_reason),
            ("selected_checks", list(self.selected_checks)),
            ("skipped_checks", list(self.skipped_checks)),
        )


class IncrementalVerificationError(ValueError):
    """Raised when an incremental change set cannot be resolved deterministically."""


def git_changed_paths(ref: str, *, cwd: Path) -> tuple[Path, ...]:
    """Return changed tracked and untracked paths relative to ``cwd``."""

    diff = _run_git(("diff", "--name-only", "--relative", ref), cwd=cwd)
    status = _run_git(("status", "--porcelain=v1", "--untracked-files=all"), cwd=cwd)
    paths = [line for line in diff.splitlines() if line.strip()]
    paths.extend(_status_paths(status))
    return _normalize_changed_paths(paths, base_dir=cwd)


def explicit_changed_paths(paths: Iterable[str | Path], *, base_dir: Path) -> tuple[Path, ...]:
    """Normalize CLI-provided changed paths against the config directory."""

    return _normalize_changed_paths(paths, base_dir=base_dir)


def run_incremental_verification(
    session: VerificationSession,
    *,
    changed_paths: Sequence[Path],
    cache_dir: Path,
    config_path: Path,
) -> VerificationResult:
    """Run only affected checks, reusing cached diagnostics for skipped checks."""

    loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
    plan = plan_incremental_checks(
        session,
        changed_paths=changed_paths,
        config_path=config_path,
        loaded_artifacts=loaded_artifacts,
    )
    check_cache = _CheckDiagnosticCache(cache_dir / "incremental" / _config_cache_name(session, config_path))
    checks_to_run = list(plan.selected_checks)
    cached_scheduled: list[ScheduledDiagnostic] = []
    cache_notes: list[ScheduledDiagnostic] = []
    config_checks = tuple(session.config.checks)
    check_ordinals = {name: index for index, name in enumerate(config_checks)}

    for check_name in plan.skipped_checks:
        key = _check_cache_key(session, check_name, loaded_artifacts)
        cached = check_cache.get(key)
        if cached is None:
            checks_to_run.append(check_name)
            cache_notes.append(
                ScheduledDiagnostic(
                    check_ordinals.get(check_name, len(config_checks)),
                    0,
                    _incremental_cache_miss_diagnostic(check_name, plan),
                    check_name=check_name,
                )
            )
            continue
        ordinal = check_ordinals.get(check_name, len(config_checks))
        cached_scheduled.extend(
            ScheduledDiagnostic(ordinal, index, diagnostic, check_name=check_name)
            for index, diagnostic in enumerate(cached)
        )
        cache_notes.append(
            ScheduledDiagnostic(
                ordinal,
                len(cached),
                _incremental_reused_diagnostic(check_name, len(cached), plan),
                check_name=check_name,
            )
        )

    scheduled = [
        ScheduledDiagnostic(-1, index, diagnostic)
        for index, diagnostic in enumerate(load_diagnostics)
    ]
    if checks_to_run:
        context = CheckContext(config=session.config, loaded_artifacts=loaded_artifacts)
        run_scheduled = session._check_diagnostics(context, tuple(checks_to_run))
        _store_run_diagnostics(check_cache, session, tuple(checks_to_run), run_scheduled, loaded_artifacts)
        scheduled.extend(_remap_scheduled_ordinals(run_scheduled, checks_to_run, check_ordinals))
    scheduled.extend(cached_scheduled)
    scheduled.extend(cache_notes)
    return _result_from_scheduled(session, scheduled, selected_checks=config_checks)


def plan_incremental_checks(
    session: VerificationSession,
    *,
    changed_paths: Sequence[Path],
    config_path: Path,
    loaded_artifacts: Sequence[LoadedArtifact],
) -> IncrementalPlan:
    """Select checks affected by the changed local paths and dependency graph."""

    normalized_paths = tuple(sorted({path.resolve() for path in changed_paths}, key=str))
    config_path = config_path.resolve()
    requested_checks = tuple(session.config.checks)
    if any(path == config_path for path in normalized_paths):
        return IncrementalPlan(
            changed_paths=normalized_paths,
            changed_artifacts=(),
            changed_kinds=(),
            selected_checks=requested_checks,
            skipped_checks=(),
            full_run_reason="config changed",
        )

    changed_artifacts = tuple(
        loaded
        for loaded in loaded_artifacts
        if loaded.artifact.location.path is not None
        and any(_path_matches_artifact(path, Path(loaded.artifact.location.path)) for path in normalized_paths)
    )
    changed_names = tuple(sorted({loaded.artifact.name for loaded in changed_artifacts}))
    changed_kinds = tuple(sorted({loaded.artifact.kind for loaded in changed_artifacts}, key=lambda kind: kind.value))
    if not normalized_paths:
        selected = requested_checks
        reason = "no changed paths supplied"
    else:
        selected = _affected_checks(session, requested_checks, changed_kinds, loaded_artifacts)
        reason = None
    skipped = tuple(check for check in requested_checks if check not in selected)
    return IncrementalPlan(
        changed_paths=normalized_paths,
        changed_artifacts=changed_names,
        changed_kinds=changed_kinds,
        selected_checks=selected,
        skipped_checks=skipped,
        full_run_reason=reason,
    )


def _affected_checks(
    session: VerificationSession,
    requested_checks: Sequence[str],
    changed_kinds: Sequence[ArtifactKind],
    loaded_artifacts: Sequence[LoadedArtifact],
) -> tuple[str, ...]:
    changed_kind_set = set(changed_kinds)
    selected: set[str] = set()
    for check in requested_checks:
        dependency = session.check_dependencies.get(check)
        if dependency is None:
            continue
        dependency_kinds = set(dependency.artifact_kinds)
        if not dependency_kinds:
            continue
        has_remote_dependency = any(
            loaded.artifact.kind in dependency_kinds and loaded.artifact.location.uri is not None
            for loaded in loaded_artifacts
        )
        if has_remote_dependency or dependency_kinds.intersection(changed_kind_set):
            selected.add(check)

    changed = True
    while changed:
        changed = False
        for check in requested_checks:
            dependency = session.check_dependencies.get(check)
            if dependency is None:
                continue
            if any(prerequisite in selected for prerequisite in dependency.after) and check not in selected:
                selected.add(check)
                changed = True
            if check in selected:
                for prerequisite in dependency.after:
                    if prerequisite in requested_checks and prerequisite not in selected:
                        selected.add(prerequisite)
                        changed = True
    return tuple(check for check in requested_checks if check in selected)


def _store_run_diagnostics(
    cache: "_CheckDiagnosticCache",
    session: VerificationSession,
    checks: Sequence[str],
    scheduled: Sequence[ScheduledDiagnostic],
    loaded_artifacts: Sequence[LoadedArtifact],
) -> None:
    for ordinal, check_name in enumerate(checks):
        diagnostics = tuple(
            item.diagnostic
            for item in sorted(
                (item for item in scheduled if item.check_ordinal == ordinal),
                key=lambda item: item.emission_index,
            )
        )
        cache.set(_check_cache_key(session, check_name, loaded_artifacts), diagnostics)


def _remap_scheduled_ordinals(
    scheduled: Sequence[ScheduledDiagnostic],
    checks: Sequence[str],
    config_ordinals: dict[str, int],
) -> tuple[ScheduledDiagnostic, ...]:
    return tuple(
        ScheduledDiagnostic(
            config_ordinals.get(checks[item.check_ordinal], item.check_ordinal)
            if 0 <= item.check_ordinal < len(checks)
            else item.check_ordinal,
            item.emission_index,
            item.diagnostic,
            check_name=item.check_name,
            duration_ms=item.duration_ms,
        )
        for item in scheduled
    )


def _result_from_scheduled(
    session: VerificationSession,
    scheduled: Sequence[ScheduledDiagnostic],
    *,
    selected_checks: Sequence[str],
) -> VerificationResult:
    ordered = tuple(item.diagnostic for item in sorted(scheduled, key=_scheduled_sort_key))
    diagnostics = tuple(
        sorted(
            (
                *ordered,
                *apply_org_policy_diagnostics(
                    session.config,
                    session.config.policy,
                    selected_checks=tuple(selected_checks),
                    check_modes=session.check_modes,
                ),
            ),
            key=lambda item: item.sort_key,
        )
    )
    diagnostics = apply_policy_diagnostics(diagnostics, session.config.policy)
    return VerificationResult(
        config=session.config,
        diagnostics=tuple(diagnostics),
        check_runtimes=_check_runtimes_from_scheduled(scheduled),
    )


def _check_cache_key(
    session: VerificationSession,
    check_name: str,
    loaded_artifacts: Sequence[LoadedArtifact],
) -> str:
    dependency = session.check_dependencies.get(check_name)
    artifact_kinds = set(dependency.artifact_kinds if dependency is not None else ())
    relevant = [
        loaded
        for loaded in loaded_artifacts
        if not artifact_kinds or loaded.artifact.kind in artifact_kinds
    ]
    payload = {
        "check": check_name,
        "config": session.config.to_dict(),
        "artifacts": [
            {
                "kind": loaded.artifact.kind.value,
                "name": loaded.artifact.name,
                "path": loaded.artifact.location.path,
                "uri": loaded.artifact.location.uri,
                "actual_sha256": loaded.actual_sha256,
                "manifest_sha256": loaded.manifest_sha256,
                "version": loaded.artifact.provenance.ref_version,
                "source_type": loaded.source_type,
            }
            for loaded in sorted(relevant, key=lambda item: (item.artifact.kind.value, item.artifact.name))
        ],
        "dependency": {
            "artifact_kinds": sorted(kind.value for kind in artifact_kinds),
            "after": list(dependency.after) if dependency is not None else [],
            "resources": list(dependency.resources) if dependency is not None else [],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_cache_name(session: VerificationSession, config_path: Path) -> str:
    payload = {
        "config_path": str(config_path.resolve()),
        "checks": list(session.config.checks),
        "plugins": sorted(session.plugin_registry.checks),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest() + ".json"


def _incremental_cache_miss_diagnostic(check_name: str, plan: IncrementalPlan) -> Diagnostic:
    return Diagnostic(
        rule_id="incremental-cache-miss",
        severity=DiagnosticSeverity.INFO,
        message=f"incremental cache had no reusable result for '{check_name}', so the check was recomputed",
        check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        witness=_incremental_witness("cache miss", check_name, plan),
        properties=(("check", check_name), *plan.to_properties()),
    )


def _incremental_reused_diagnostic(check_name: str, diagnostic_count: int, plan: IncrementalPlan) -> Diagnostic:
    return Diagnostic(
        rule_id="incremental-check-reused",
        severity=DiagnosticSeverity.INFO,
        message=f"reused {diagnostic_count} cached diagnostic(s) for unchanged check '{check_name}'",
        check_modes=(CheckMode.HEURISTIC,),
        witness=_incremental_witness("reuse cached check", check_name, plan),
        properties=(("check", check_name), ("diagnostic_count", diagnostic_count), *plan.to_properties()),
    )


def _incremental_witness(action: str, check_name: str, plan: IncrementalPlan) -> WitnessTrace:
    return WitnessTrace(
        summary="PromptABI planned a monorepo incremental verification run from changed local inputs.",
        steps=(
            WitnessStep(action="normalize changed paths", output=str(len(plan.changed_paths))),
            WitnessStep(action="map paths to artifacts", output=", ".join(plan.changed_artifacts) or "none"),
            WitnessStep(action=action, input=check_name, output=", ".join(plan.changed_kinds) or "no artifact kind"),
        ),
    )


def _path_matches_artifact(changed_path: Path, artifact_path: Path) -> bool:
    artifact_path = artifact_path.resolve()
    if changed_path == artifact_path:
        return True
    try:
        changed_path.relative_to(artifact_path)
    except ValueError:
        return False
    return True


def _normalize_changed_paths(paths: Iterable[str | Path], *, base_dir: Path) -> tuple[Path, ...]:
    normalized = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        normalized.append(candidate.resolve())
    return tuple(sorted(set(normalized), key=str))


def _status_paths(output: str) -> tuple[str, ...]:
    paths = []
    for line in output.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:
            _old, path = path.split(" -> ", 1)
        paths.append(path)
    return tuple(paths)


def _run_git(args: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(cwd), "--no-pager", *args),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"git exited {completed.returncode}"
        raise IncrementalVerificationError(message)
    return completed.stdout


def _scheduled_sort_key(item: ScheduledDiagnostic) -> tuple[object, ...]:
    return (
        diagnostic_sort_key(item.diagnostic),
        item.check_ordinal,
        item.emission_index,
    )


class _CheckDiagnosticCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()

    def get(self, key: str) -> tuple[Diagnostic, ...] | None:
        raw = self._data.get(key)
        if not isinstance(raw, list):
            return None
        return tuple(Diagnostic.from_dict(item) for item in raw if isinstance(item, dict))

    def set(self, key: str, diagnostics: Sequence[Diagnostic]) -> None:
        self._data[key] = [diagnostic.to_dict() for diagnostic in diagnostics]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}
