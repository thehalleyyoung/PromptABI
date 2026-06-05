"""Upstream-ready markdown issue reports for PromptABI diagnostics."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import ArtifactRef, Diagnostic, WitnessStep
from .explain import DiagnosticExplanation, explain_diagnostic
from .minimization import MinimizationResult
from .session import VerificationResult


class BugReportError(ValueError):
    """Raised when an upstream issue report cannot be generated."""


@dataclass(frozen=True, slots=True)
class BugReport:
    """A sanitized markdown issue report for one PromptABI diagnostic."""

    title: str
    diagnostic: Diagnostic
    markdown: str


def generate_bug_report(
    result: VerificationResult,
    *,
    config_path: str | Path | None = None,
    fingerprint: str | None = None,
    rule_id: str | None = None,
    index: int | None = None,
    expected_behavior: str | None = None,
    actual_behavior: str | None = None,
    minimization: MinimizationResult | None = None,
    command: str | None = None,
    base_dir: str | Path | None = None,
    max_witness_chars: int = 240,
) -> BugReport:
    """Generate a deterministic, non-secret markdown issue for one diagnostic."""

    if max_witness_chars < 40:
        raise BugReportError("max_witness_chars must be at least 40")
    explanation = explain_diagnostic(
        result,
        fingerprint=fingerprint,
        rule_id=rule_id,
        index=index,
        base_dir=base_dir,
    )
    diagnostic = explanation.diagnostic
    title = _title_for(diagnostic)
    resolved_command = command or _default_command(config_path, diagnostic)
    markdown = _render_markdown(
        result,
        explanation,
        title=title,
        command=resolved_command,
        expected_behavior=expected_behavior,
        actual_behavior=actual_behavior,
        minimization=minimization,
        max_witness_chars=max_witness_chars,
    )
    return BugReport(title=title, diagnostic=diagnostic, markdown=markdown)


def render_bug_report(report: BugReport) -> str:
    """Render a bug report as markdown."""

    return report.markdown


def _render_markdown(
    result: VerificationResult,
    explanation: DiagnosticExplanation,
    *,
    title: str,
    command: str,
    expected_behavior: str | None,
    actual_behavior: str | None,
    minimization: MinimizationResult | None,
    max_witness_chars: int,
) -> str:
    diagnostic = explanation.diagnostic
    command_lines = tuple(line for line in command.splitlines() if line)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        _sanitize_text(diagnostic.message, limit=max_witness_chars),
        "",
        "## Affected artifact",
        "",
        _artifact_table(_affected_artifacts(result, diagnostic)),
        "",
        "## Reproduction",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Expected behavior",
        "",
        expected_behavior
        or "The artifact contract should not admit this structural prompt-interface state under the supported PromptABI fragment.",
        "",
        "## Actual behavior",
        "",
        actual_behavior
        or _sanitize_text(explanation.likely_symptom, limit=max_witness_chars),
        "",
        "## PromptABI diagnostic",
        "",
        f"- Rule: `{diagnostic.rule_id}`",
        f"- Severity: `{diagnostic.severity.value}`",
        f"- Fingerprint: `{diagnostic.fingerprint}`",
        f"- Modes: {_mode_list(diagnostic)}",
        "",
    ]
    if diagnostic.upstream_issues:
        lines.extend(["## Upstream status", ""])
        lines.extend(_upstream_issue_lines(diagnostic, max_chars=max_witness_chars))
        lines.append("")
    lines.extend(["## Non-sensitive witness trace", ""])
    lines.extend(_witness_lines(diagnostic, max_chars=max_witness_chars))
    lines.extend(
        [
            "",
            "## Minimized input",
            "",
        ]
    )
    if minimization is None:
        lines.append(
            "No separate minimization case was supplied; the diagnostic witness above is the minimized structural trace emitted by PromptABI."
        )
    else:
        lines.extend(_minimization_lines(minimization, max_chars=max_witness_chars))
    lines.extend(["", "## Suggested fix", ""])
    if diagnostic.suggestions:
        lines.extend(f"- {_sanitize_text(suggestion, limit=max_witness_chars)}" for suggestion in diagnostic.suggestions)
    else:
        lines.append("- No automatic fix suggestion is attached to this diagnostic.")
    lines.extend(
        [
            "",
            "## Privacy note",
            "",
            "This report was generated locally by PromptABI. Witness inputs and outputs are redacted and length-limited; no prompts, schemas, configs, solver constraints, or provider traces were transmitted.",
            "",
        ]
    )
    return "\n".join(lines)


def _title_for(diagnostic: Diagnostic) -> str:
    artifact = f" in {diagnostic.artifact.name}" if diagnostic.artifact is not None else ""
    return f"PromptABI {diagnostic.rule_id}{artifact}: {diagnostic.message[:80]}".rstrip()


def _default_command(config_path: str | Path | None, diagnostic: Diagnostic) -> str:
    verify_words = ["promptabi", "verify"]
    if config_path is not None:
        verify_words.extend(["--config", str(config_path)])
    verify_words.extend(["--fail-on", "never", "--format", "json"])
    explain_words = ["promptabi", "explain"]
    if config_path is not None:
        explain_words.extend(["--config", str(config_path)])
    explain_words.extend(["--fingerprint", diagnostic.fingerprint])
    return "\n".join(
        (
            " ".join(shlex.quote(word) for word in verify_words),
            " ".join(shlex.quote(word) for word in explain_words),
        )
    )


def _affected_artifacts(result: VerificationResult, diagnostic: Diagnostic) -> tuple[ArtifactRef, ...]:
    refs: dict[tuple[str, str, str | None, str | None], ArtifactRef] = {}
    if diagnostic.artifact is not None:
        refs[_artifact_key(diagnostic.artifact)] = diagnostic.artifact
    if diagnostic.witness is not None:
        for artifact in diagnostic.witness.artifacts:
            refs[_artifact_key(artifact)] = artifact
    for artifact in result.config.artifact_bundle:
        ref = artifact.to_ref()
        if diagnostic.artifact is None or artifact.name == diagnostic.artifact.name:
            refs[_artifact_key(ref)] = ref
    return tuple(sorted(refs.values(), key=lambda item: (item.kind, item.name, item.location_uri or "")))


def _artifact_key(ref: ArtifactRef) -> tuple[str, str, str | None, str | None]:
    return (ref.kind, ref.name, ref.location_uri, ref.sha256 or ref.revision or ref.version)


def _artifact_table(artifacts: tuple[ArtifactRef, ...]) -> str:
    if not artifacts:
        return "No artifact reference was attached to the diagnostic."
    lines = [
        "| Kind | Name | Location | Version/revision | SHA-256 | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in (
                    artifact.kind,
                    artifact.name,
                    artifact.location_uri or "",
                    artifact.version or artifact.revision or "",
                    artifact.sha256 or "",
                    artifact.source or "",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _witness_lines(diagnostic: Diagnostic, *, max_chars: int) -> list[str]:
    if diagnostic.witness is None:
        return ["PromptABI did not attach a witness trace to this diagnostic."]
    lines = [_sanitize_text(diagnostic.witness.summary, limit=max_chars), ""]
    for index, step in enumerate(diagnostic.witness.steps, start=1):
        assert isinstance(step, WitnessStep)
        rendered = f"{index}. {step.action}"
        details = []
        if step.input is not None:
            details.append(f"input: `{_sanitize_text(step.input, limit=max_chars)}`")
        if step.output is not None:
            details.append(f"output: `{_sanitize_text(step.output, limit=max_chars)}`")
        if details:
            rendered += " (" + "; ".join(details) + ")"
        lines.append(rendered)
    return lines


def _minimization_lines(result: MinimizationResult, *, max_chars: int) -> list[str]:
    payload = json.dumps(result.minimized, indent=2, sort_keys=True, ensure_ascii=False)
    return [
        f"PromptABI minimized a `{result.kind.value}` repro from {result.stats.original_size} to {result.stats.minimized_size} JSON bytes.",
        "",
        "```json",
        _sanitize_text(payload, limit=max(max_chars, 1200)),
        "```",
    ]


def _mode_list(diagnostic: Diagnostic) -> str:
    if not diagnostic.check_modes:
        return "_not declared_"
    return ", ".join(f"`{mode.value}`" for mode in diagnostic.check_modes)


def _upstream_issue_lines(diagnostic: Diagnostic, *, max_chars: int) -> list[str]:
    lines: list[str] = []
    for link in diagnostic.upstream_issues:
        lines.append(f"- [{_sanitize_text(link.title, limit=max_chars)}]({link.url})")
        lines.append(f"  - Status: `{_sanitize_text(link.status, limit=max_chars)}`")
        if link.affected_versions:
            lines.append(
                "  - Affected versions/artifacts: "
                + ", ".join(f"`{_sanitize_text(item, limit=max_chars)}`" for item in link.affected_versions)
            )
        if link.fixed_versions:
            lines.append(
                "  - Fixed versions/patches: "
                + ", ".join(f"`{_sanitize_text(item, limit=max_chars)}`" for item in link.fixed_versions)
            )
        if link.workarounds:
            lines.append("  - Local compatibility workarounds:")
            lines.extend(f"    - {_sanitize_text(item, limit=max_chars)}" for item in link.workarounds)
    return lines


def _md_cell(value: str) -> str:
    return _sanitize_text(value, limit=96).replace("|", "\\|") or "_not declared_"


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
)


def _sanitize_text(value: Any, *, limit: int) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: match.group(0).split(match.group(1))[0] + match.group(1) + "=<redacted>", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [truncated {omitted} chars]"
