"""GitHub Actions orchestration for PromptABI verification."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, discover_config, load_config
from .diagnostics import Diagnostic
from .first_party_plugins import create_first_party_plugin_registry
from .lockfiles import (
    LockfileError,
    compare_lockfile,
    load_lockfile,
    lockfile_error_diagnostic,
)
from .render import SarifRenderOptions, render_github_annotations, render_sarif
from .session import VerificationResult, VerificationSession


class GitHubActionError(ValueError):
    """Raised when the GitHub Action runner cannot complete soundly."""


@dataclass(frozen=True, slots=True)
class GitHubActionRun:
    """A completed PromptABI GitHub Action run."""

    result: VerificationResult | None
    skipped: bool
    changed_paths: tuple[str, ...]
    relevant_paths: tuple[str, ...]
    sarif_path: Path
    summary_path: Path | None
    exit_code: int

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return () if self.result is None else self.result.diagnostics


def run_github_action(
    *,
    config_path: str | Path | None = None,
    lockfile_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    sarif_output: str | Path = "promptabi.sarif",
    summary_output: str | Path | None = None,
    repo_root: str | Path | None = None,
    fail_on: str = "error",
    require_lockfile: bool = True,
    changed_only: bool = False,
    base_ref: str | None = None,
    head_ref: str | None = None,
    annotations: bool = False,
    argv: Sequence[str] | None = None,
) -> GitHubActionRun:
    """Run verification with GitHub-specific outputs and changed-artifact gating."""

    workspace = _resolve_repo_root(repo_root)
    resolved_config = Path(config_path).expanduser().resolve() if config_path else discover_config(workspace)
    resolved_lockfile = _resolve_lockfile(lockfile_path, resolved_config)
    resolved_cache = _resolve_cache_dir(cache_dir, workspace)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    resolved_sarif = Path(sarif_output).expanduser()
    if not resolved_sarif.is_absolute():
        resolved_sarif = workspace / resolved_sarif
    resolved_sarif = resolved_sarif.resolve()
    resolved_summary = _resolve_summary_path(summary_output)

    try:
        config = load_config(resolved_config)
    except ConfigError:
        raise

    relevant = relevant_promptabi_paths(
        config_path=resolved_config,
        lockfile_path=resolved_lockfile if require_lockfile else None,
        artifact_paths=(artifact.location.path for artifact in config.artifact_bundle if artifact.location.path),
        repo_root=workspace,
    )
    changed = ()
    if changed_only:
        changed = changed_promptabi_paths(repo_root=workspace, base_ref=base_ref, head_ref=head_ref)
        if changed and not _touches_relevant_path(changed, relevant):
            _write_skip_sarif(resolved_sarif, category="promptabi")
            _write_markdown_summary(
                resolved_summary,
                config_name=config.name,
                skipped=True,
                changed_paths=changed,
                relevant_paths=relevant,
                diagnostics=(),
                sarif_path=resolved_sarif,
            )
            _write_github_outputs(
                skipped=True,
                result=None,
                sarif_path=resolved_sarif,
                summary_path=resolved_summary,
            )
            return GitHubActionRun(
                result=None,
                skipped=True,
                changed_paths=changed,
                relevant_paths=relevant,
                sarif_path=resolved_sarif,
                summary_path=resolved_summary,
                exit_code=0,
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

    sarif = render_sarif(
        result,
        options=SarifRenderOptions(
            category="promptabi",
            checkout_uri_base=workspace,
            include_invocation=True,
            command_line=_command_line(argv),
            working_directory=workspace,
        ),
    )
    resolved_sarif.parent.mkdir(parents=True, exist_ok=True)
    resolved_sarif.write_text(sarif, encoding="utf-8")
    _write_markdown_summary(
        resolved_summary,
        config_name=config.name,
        skipped=False,
        changed_paths=changed,
        relevant_paths=relevant,
        diagnostics=result.diagnostics,
        sarif_path=resolved_sarif,
    )
    _write_github_outputs(
        skipped=False,
        result=result,
        sarif_path=resolved_sarif,
        summary_path=resolved_summary,
    )
    if annotations:
        print(render_github_annotations(result, checkout_uri_base=workspace), end="")

    return GitHubActionRun(
        result=result,
        skipped=False,
        changed_paths=changed,
        relevant_paths=relevant,
        sarif_path=resolved_sarif,
        summary_path=resolved_summary,
        exit_code=_exit_code(result, fail_on=fail_on),
    )


def relevant_promptabi_paths(
    *,
    config_path: Path,
    lockfile_path: Path | None,
    artifact_paths: Sequence[str | None],
    repo_root: Path,
) -> tuple[str, ...]:
    """Return repo-relative paths whose changes require PromptABI verification."""

    paths = {_repo_relative(config_path, repo_root)}
    if lockfile_path is not None:
        paths.add(_repo_relative(lockfile_path, repo_root))
    for artifact_path in artifact_paths:
        if artifact_path:
            paths.add(_repo_relative(Path(artifact_path), repo_root))
    return tuple(sorted(path for path in paths if path))


def changed_promptabi_paths(
    *,
    repo_root: str | Path,
    base_ref: str | None = None,
    head_ref: str | None = None,
) -> tuple[str, ...]:
    """Return changed repository paths from git using GitHub-friendly refs."""

    root = Path(repo_root).expanduser().resolve()
    base = base_ref or os.environ.get("PROMPTABI_BASE_REF") or os.environ.get("GITHUB_BASE_REF")
    head = head_ref or os.environ.get("PROMPTABI_HEAD_REF") or os.environ.get("GITHUB_SHA") or "HEAD"
    if not base:
        before = os.environ.get("GITHUB_EVENT_BEFORE")
        if before and set(before) != {"0"}:
            base = before
    if not base:
        return ()

    refspec = f"{base}...{head}" if head else base
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACMRT", refspec],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            message = exc.stderr.strip()
        raise GitHubActionError(f"could not detect changed PromptABI artifacts with git diff: {message}") from exc
    return tuple(sorted(path for path in completed.stdout.splitlines() if path))


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_lockfile(lockfile_path: str | Path | None, config_path: Path) -> Path:
    if lockfile_path is not None:
        return Path(lockfile_path).expanduser().resolve()
    return config_path.with_name("promptabi.lock.json")


def _resolve_cache_dir(cache_dir: str | Path | None, repo_root: Path) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir).expanduser()
        return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
    env_value = os.environ.get("PROMPTABI_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root / ".promptabi" / "cache").resolve()


def _resolve_summary_path(summary_output: str | Path | None) -> Path | None:
    raw = summary_output or os.environ.get("GITHUB_STEP_SUMMARY")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _touches_relevant_path(changed_paths: Sequence[str], relevant_paths: Sequence[str]) -> bool:
    relevant = tuple(path.rstrip("/") for path in relevant_paths)
    for changed in changed_paths:
        normalized = changed.strip("/")
        for path in relevant:
            if normalized == path or normalized.startswith(f"{path}/") or path.startswith(f"{normalized}/"):
                return True
    return False


def _write_skip_sarif(path: Path, *, category: str) -> None:
    payload = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "automationDetails": {"id": f"{category}/"},
                "tool": {"driver": {"name": "PromptABI", "rules": []}},
                "results": [],
                "invocations": [{"executionSuccessful": True}],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_summary(
    path: Path | None,
    *,
    config_name: str,
    skipped: bool,
    changed_paths: Sequence[str],
    relevant_paths: Sequence[str],
    diagnostics: Sequence[Diagnostic],
    sarif_path: Path,
) -> None:
    if path is None:
        return
    counts = _diagnostic_counts(diagnostics)
    status = "SKIPPED" if skipped else ("PASS" if counts["error"] == 0 else "FAIL")
    lines = [
        "## PromptABI verification",
        "",
        f"**Config:** `{config_name}`",
        f"**Status:** {status}",
        f"**Diagnostics:** {counts['error']} errors, {counts['warning']} warnings, {counts['info']} info",
        f"**SARIF:** `{sarif_path}`",
    ]
    if changed_paths:
        lines.extend(["", "### Changed PromptABI inputs", "", *_markdown_list(changed_paths)])
    if skipped:
        lines.extend(["", "No configured PromptABI artifact changed, so verification was skipped."])
    elif diagnostics:
        lines.extend(["", "### Findings", "", "| Severity | Rule | Message |", "| --- | --- | --- |"])
        for diagnostic in diagnostics[:20]:
            lines.append(
                f"| {diagnostic.severity.value} | `{diagnostic.rule_id}` | {_escape_markdown_table(diagnostic.message)} |"
            )
    if relevant_paths:
        lines.extend(["", "### Watched inputs", "", *_markdown_list(relevant_paths)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_github_outputs(
    *,
    skipped: bool,
    result: VerificationResult | None,
    sarif_path: Path,
    summary_path: Path | None,
) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    diagnostics = () if result is None else result.diagnostics
    counts = _diagnostic_counts(diagnostics)
    values = {
        "skipped": str(skipped).lower(),
        "ok": str(result is None or result.ok).lower(),
        "diagnostic-count": str(len(diagnostics)),
        "error-count": str(counts["error"]),
        "warning-count": str(counts["warning"]),
        "info-count": str(counts["info"]),
        "sarif": str(sarif_path),
    }
    if summary_path is not None:
        values["summary"] = str(summary_path)
    with Path(output_path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def _diagnostic_counts(diagnostics: Sequence[Diagnostic]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for diagnostic in diagnostics:
        counts[diagnostic.severity.value] += 1
    return counts


def _markdown_list(values: Sequence[str]) -> list[str]:
    return [f"- `{value}`" for value in values[:50]]


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _command_line(argv: Sequence[str] | None) -> str:
    words = ["promptabi", *(argv or ("github-action",))]
    return " ".join(shlex.quote(str(word)) for word in words)


def _exit_code(result: VerificationResult, *, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    severities = {diagnostic.severity.value for diagnostic in result.diagnostics}
    if fail_on == "any":
        return 1 if severities else 0
    if fail_on == "warning":
        return 1 if severities.intersection({"error", "warning"}) else 0
    return 0 if result.ok else 1
