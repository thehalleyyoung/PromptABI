"""Local developer workflows for PromptABI verification."""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, discover_config, load_config
from .diagnostics import Diagnostic
from .first_party_plugins import create_first_party_plugin_registry
from .github_action import relevant_promptabi_paths
from .lockfiles import (
    LockfileError,
    compare_lockfile,
    load_lockfile,
    lockfile_error_diagnostic,
)
from .session import VerificationResult, VerificationSession


HOOK_MARKER = "# PromptABI managed pre-commit hook"


class LocalWorkflowError(ValueError):
    """Raised when a local workflow cannot run soundly."""


@dataclass(frozen=True, slots=True)
class LocalWorkflowRun:
    """A completed local PromptABI workflow run."""

    result: VerificationResult | None
    skipped: bool
    changed_paths: tuple[str, ...]
    relevant_paths: tuple[str, ...]
    candidate_paths: tuple[str, ...]
    selected_paths: tuple[str, ...]
    exit_code: int

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return () if self.result is None else self.result.diagnostics


def run_local_workflow(
    *,
    config_path: str | Path | None = None,
    lockfile_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    fail_on: str = "error",
    require_lockfile: bool = False,
    changed_only: bool = False,
    mode: str = "staged",
    changed_paths: Sequence[str] | None = None,
    allow_unstaged: bool = False,
) -> LocalWorkflowRun:
    """Run local verification, optionally gated to changed PromptABI inputs."""

    root = resolve_repo_root(repo_root)
    resolved_config = Path(config_path).expanduser().resolve() if config_path else discover_config(root)
    resolved_lockfile = _resolve_lockfile(lockfile_path, resolved_config)
    if cache_dir is not None:
        Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)

    config = load_config(resolved_config)
    relevant = relevant_promptabi_paths(
        config_path=resolved_config,
        lockfile_path=resolved_lockfile if require_lockfile else None,
        artifact_paths=(artifact.location.path for artifact in config.artifact_bundle if artifact.location.path),
        repo_root=root,
    )
    changed = tuple(sorted(dict.fromkeys(changed_paths))) if changed_paths is not None else changed_local_paths(
        repo_root=root,
        mode=mode,
    )
    candidate = tuple(path for path in changed if is_promptabi_candidate_path(path))
    selected = tuple(path for path in changed if _touches_path(path, relevant) or path in candidate)

    if changed_only and not selected:
        return LocalWorkflowRun(
            result=None,
            skipped=True,
            changed_paths=changed,
            relevant_paths=relevant,
            candidate_paths=candidate,
            selected_paths=(),
            exit_code=0,
        )

    if mode == "staged" and selected and not allow_unstaged:
        dirty_selected = _unstaged_overlap(repo_root=root, selected_paths=selected)
        if dirty_selected:
            formatted = ", ".join(dirty_selected)
            raise LocalWorkflowError(
                "cannot verify staged PromptABI inputs while their working-tree copies differ; "
                f"stage or revert unstaged edits for: {formatted}"
            )

    registry = create_first_party_plugin_registry()
    session = VerificationSession(config, plugin_registry=registry)
    result = session.run()
    if require_lockfile:
        loaded_artifacts, _load_diagnostics = session.load_artifacts_with_diagnostics()
        try:
            lockfile = load_lockfile(resolved_lockfile)
            lock_diagnostics = compare_lockfile(
                lockfile,
                config,
                loaded_artifacts,
                result.diagnostics,
                lockfile_path=resolved_lockfile,
            )
        except LockfileError as exc:
            lock_diagnostics = (lockfile_error_diagnostic(exc, lockfile_path=resolved_lockfile),)
        if lock_diagnostics:
            result = type(result)(
                config=result.config,
                diagnostics=tuple(sorted((*result.diagnostics, *lock_diagnostics), key=lambda item: item.sort_key)),
            )

    return LocalWorkflowRun(
        result=result,
        skipped=False,
        changed_paths=changed,
        relevant_paths=relevant,
        candidate_paths=candidate,
        selected_paths=selected,
        exit_code=_exit_code(result, fail_on=fail_on),
    )


def install_pre_commit_hook(
    *,
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    fail_on: str = "error",
    require_lockfile: bool = False,
    changed_only: bool = True,
    force: bool = False,
) -> Path:
    """Install a PromptABI-managed git pre-commit hook."""

    root = resolve_repo_root(repo_root)
    hooks_path_setting = _git_optional(root, "config", "--get", "core.hooksPath")
    if hooks_path_setting:
        hooks_dir = Path(hooks_path_setting)
        if not hooks_dir.is_absolute():
            hooks_dir = root / hooks_dir
    else:
        hooks_dir_raw = _git_required(root, "rev-parse", "--git-path", "hooks")
        hooks_dir = Path(hooks_dir_raw)
        if not hooks_dir.is_absolute():
            hooks_dir = root / hooks_dir
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    if hook_path.exists():
        current = hook_path.read_text(encoding="utf-8", errors="replace")
        if HOOK_MARKER not in current and not force:
            raise LocalWorkflowError(
                f"refusing to overwrite existing non-PromptABI hook at {hook_path}; rerun with --force"
            )

    resolved_config = Path(config_path).expanduser().resolve() if config_path else discover_config(root)
    words = [
        sys.executable,
        "-m",
        "promptabi",
        "pre-commit",
        "run",
        "--config",
        str(resolved_config),
        "--repo-root",
        str(root),
        "--fail-on",
        fail_on,
    ]
    if cache_dir is not None:
        words.extend(["--cache-dir", str(Path(cache_dir).expanduser())])
    if require_lockfile:
        words.append("--require-lockfile")
    if changed_only:
        words.append("--changed-only")
    script = "#!/bin/sh\n" f"{HOOK_MARKER}\n" "set -eu\n" f"exec {_shell_join(words)}\n"
    hook_path.write_text(script, encoding="utf-8")
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def changed_local_paths(*, repo_root: str | Path, mode: str = "staged") -> tuple[str, ...]:
    """Return repo-relative paths changed in the selected local git view."""

    args = ["diff", "--name-only", "--diff-filter=ACMRT"]
    if mode == "staged":
        args.insert(1, "--cached")
    elif mode == "working-tree":
        args.extend(["HEAD"])
    elif mode != "unstaged":
        raise LocalWorkflowError(f"unsupported changed-path mode: {mode}")
    output = _git_required(Path(repo_root).expanduser().resolve(), *args)
    return tuple(sorted(path for path in output.splitlines() if path))


def is_promptabi_candidate_path(path: str) -> bool:
    """Return whether a changed path is likely to affect PromptABI verification."""

    normalized = path.strip("/")
    name = Path(normalized).name.lower()
    suffixes = Path(normalized).suffixes
    if name in {"promptabi.json", ".promptabi.json", "promptabi.lock.json"}:
        return True
    if name in {
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tools.json",
        "messages.json",
        "segments.json",
        "runtime-budget.json",
        "provider-fixtures.json",
        "training-manifest.json",
    }:
        return True
    if name.endswith((".schema.json", ".grammar.json", ".promptabi.json")):
        return True
    if any(part in name for part in ("template", "schema", "grammar", "tool", "tokenizer", "training")):
        return True
    return bool(suffixes and suffixes[-1] in {".jinja", ".j2", ".ebnf", ".bnf", ".lark", ".jsonl"})


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the repository root using git when possible."""

    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    try:
        return Path(_git_required(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    except LocalWorkflowError:
        return Path.cwd().resolve()


def render_local_workflow_text(run: LocalWorkflowRun) -> str:
    """Render a concise local workflow summary."""

    if run.skipped:
        lines = ["PromptABI pre-commit: skipped (no changed PromptABI inputs)"]
    else:
        assert run.result is not None
        errors = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "error")
        warnings = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "warning")
        lines = [
            "PromptABI pre-commit: "
            f"{'PASS' if run.result.ok else 'FAIL'} ({errors} errors, {warnings} warnings)"
        ]
        for diagnostic in run.result.diagnostics[:10]:
            lines.append(f"  {diagnostic.severity.value.upper()} {diagnostic.rule_id}: {diagnostic.message}")
    if run.selected_paths:
        lines.extend(["changed PromptABI inputs:", *[f"  {path}" for path in run.selected_paths[:20]]])
    elif run.changed_paths and run.skipped:
        lines.append(f"changed paths checked for relevance: {len(run.changed_paths)}")
    return "\n".join(lines) + "\n"


def _resolve_lockfile(lockfile_path: str | Path | None, config_path: Path) -> Path:
    if lockfile_path is not None:
        return Path(lockfile_path).expanduser().resolve()
    return config_path.with_name("promptabi.lock.json")


def _unstaged_overlap(*, repo_root: Path, selected_paths: Sequence[str]) -> tuple[str, ...]:
    unstaged = set(changed_local_paths(repo_root=repo_root, mode="unstaged"))
    return tuple(path for path in selected_paths if path in unstaged)


def _touches_path(changed_path: str, relevant_paths: Sequence[str]) -> bool:
    changed = changed_path.strip("/")
    for relevant_path in relevant_paths:
        relevant = relevant_path.rstrip("/")
        if changed == relevant or changed.startswith(f"{relevant}/") or relevant.startswith(f"{changed}/"):
            return True
    return False


def _git_required(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            message = exc.stderr.strip()
        raise LocalWorkflowError(f"git {' '.join(args)} failed: {message}") from exc
    return completed.stdout.strip()


def _git_optional(repo_root: Path, *args: str) -> str | None:
    try:
        return _git_required(repo_root, *args)
    except LocalWorkflowError:
        return None


def _shell_join(words: Sequence[str]) -> str:
    return " ".join(shlex.quote(word) for word in words)


def _exit_code(result: VerificationResult, *, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    severities = {diagnostic.severity.value for diagnostic in result.diagnostics}
    if fail_on == "any":
        return 1 if severities else 0
    if fail_on == "warning":
        return 1 if severities.intersection({"error", "warning"}) else 0
    return 0 if result.ok else 1
