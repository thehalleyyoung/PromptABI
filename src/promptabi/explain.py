"""Tutorial-style explanations for PromptABI diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import Diagnostic
from .session import VerificationResult


class ExplainError(ValueError):
    """Raised when a diagnostic explanation cannot be selected or rendered."""


@dataclass(frozen=True, slots=True)
class SourceSnippet:
    """A small source excerpt attached to a diagnostic explanation."""

    path: str
    start_line: int
    lines: tuple[str, ...]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "start_line": self.start_line,
            "lines": list(self.lines),
        }
        if self.note is not None:
            data["note"] = self.note
        return data


@dataclass(frozen=True, slots=True)
class DiagnosticExplanation:
    """A structured, tutorial-style expansion of one diagnostic."""

    diagnostic: Diagnostic
    property_checked: str
    proof_modes: tuple[str, ...]
    likely_symptom: str
    source_snippet: SourceSnippet | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "diagnostic": self.diagnostic.to_dict(),
            "property_checked": self.property_checked,
            "proof_modes": list(self.proof_modes),
            "likely_production_symptom": self.likely_symptom,
            "fix_suggestions": list(self.diagnostic.suggestions),
        }
        if self.source_snippet is not None:
            data["source_snippet"] = self.source_snippet.to_dict()
        return data


PROPERTY_BY_RULE_ID = {
    "artifact-missing": "Every declared local artifact must resolve to a readable file before PromptABI can reason about downstream contracts.",
    "artifact-load-failed": "Each declared artifact must parse inside its supported fragment with source locations preserved for diagnostics.",
    "artifact-unpinned": "Reproducible verification requires artifacts to carry stable provenance such as sha256 pins or immutable revisions.",
    "repository-skeleton": "The configured PromptABI project should load, normalize artifacts, and produce deterministic diagnostics through the public workflow.",
    "role-boundary-nonforgeability": "Attacker-controlled content must not render as structural role delimiters, assistant prefixes, special tokens, or tool-call sentinels under the selected chat template.",
    "stop-overreachability": "A stop policy must not be able to terminate generation inside a still-valid structured-output region before the application parser can see a complete value.",
    "grammar-tokenizer-emptiness": "The product of tokenizer assumptions and structured-output grammar must admit at least one valid serialized output.",
    "grammar-tokenizer-ambiguity": "Distinct token or byte paths must not collapse into conflicting structured outputs under the declared tokenizer and grammar assumptions.",
    "parser-compatibility": "The constrained-decoding grammar and application parser should agree on the bounded examples they are expected to accept or reject.",
    "must-survive-budget": "Prompt regions marked as required must survive the configured framework truncation and context-budget policy.",
}


SYMPTOM_BY_RULE_ID = {
    "artifact-missing": "CI fails before deployment, or a local run silently checks fewer artifacts than the production stack actually uses.",
    "artifact-load-failed": "A schema, template, tokenizer config, or provider fixture is skipped or mis-modeled, hiding the real interface risk.",
    "artifact-unpinned": "A future library, model, or artifact update can change verification results without an obvious code diff.",
    "repository-skeleton": "This informational diagnostic confirms the verification path is wired; it is not itself a production failure.",
    "role-boundary-nonforgeability": "A user, tool, or retrieved field can be serialized so downstream code or the model-facing transcript appears to contain a role boundary the app did not intend.",
    "stop-overreachability": "Generation can stop in the middle of JSON, XML-ish tool calls, markdown fences, or provider envelopes, causing flaky parsing or truncated tool arguments.",
    "grammar-tokenizer-emptiness": "A constrained decoder may be unable to emit any value satisfying the declared schema, producing retries, timeouts, or empty responses.",
    "grammar-tokenizer-ambiguity": "Two serialized forms can decode or parse as different application values, making validation disagree with runtime parsing.",
    "parser-compatibility": "The model can produce text accepted by one layer but rejected or interpreted differently by the application parser.",
    "must-survive-budget": "System instructions, tool definitions, citations, or format constraints can be dropped while the request still appears valid.",
}


def explain_diagnostic(
    result: VerificationResult,
    *,
    fingerprint: str | None = None,
    rule_id: str | None = None,
    index: int | None = None,
    base_dir: str | Path | None = None,
) -> DiagnosticExplanation:
    """Select and explain one logical diagnostic from a verification result."""

    selected = select_diagnostic(result, fingerprint=fingerprint, rule_id=rule_id, index=index)
    return DiagnosticExplanation(
        diagnostic=selected,
        property_checked=PROPERTY_BY_RULE_ID.get(
            selected.rule_id,
            "PromptABI checked a structural interface contract over the configured artifacts and reported how the observed state fits that contract.",
        ),
        proof_modes=tuple(mode.description for mode in selected.check_modes),
        likely_symptom=SYMPTOM_BY_RULE_ID.get(
            selected.rule_id,
            "The affected prompt-interface contract can behave differently from the assumptions encoded in tests, deployment gates, or downstream parsers.",
        ),
        source_snippet=_source_snippet(selected, base_dir=Path(base_dir) if base_dir is not None else None),
    )


def select_diagnostic(
    result: VerificationResult,
    *,
    fingerprint: str | None = None,
    rule_id: str | None = None,
    index: int | None = None,
) -> Diagnostic:
    """Select one de-duplicated diagnostic using a fingerprint, rule id, or one-based index."""

    selectors = [fingerprint is not None, rule_id is not None, index is not None]
    if sum(selectors) > 1:
        raise ExplainError("choose only one selector: --fingerprint, --rule-id, or --index")
    diagnostics = _deduplicate_diagnostics(result.diagnostics)
    if not diagnostics:
        raise ExplainError("verification produced no diagnostics to explain")
    if fingerprint is not None:
        matches = tuple(diagnostic for diagnostic in diagnostics if diagnostic.fingerprint == fingerprint)
        if not matches:
            raise ExplainError(f"no diagnostic has fingerprint {fingerprint!r}")
        return matches[0]
    if rule_id is not None:
        matches = tuple(diagnostic for diagnostic in diagnostics if diagnostic.rule_id == rule_id)
        if not matches:
            raise ExplainError(f"no diagnostic has rule id {rule_id!r}")
        if len(matches) > 1:
            options = ", ".join(diagnostic.fingerprint for diagnostic in matches[:8])
            suffix = " ..." if len(matches) > 8 else ""
            raise ExplainError(
                f"rule id {rule_id!r} matched {len(matches)} diagnostics; rerun with --fingerprint "
                f"or --index. fingerprints: {options}{suffix}"
            )
        return matches[0]
    if index is not None:
        if index < 1 or index > len(diagnostics):
            raise ExplainError(f"--index must be between 1 and {len(diagnostics)}")
        return diagnostics[index - 1]
    for diagnostic in diagnostics:
        if diagnostic.severity.value != "info":
            return diagnostic
    return diagnostics[0]


def render_explanation_text(explanation: DiagnosticExplanation) -> str:
    """Render an explanation as concise tutorial text."""

    diagnostic = explanation.diagnostic
    lines = [
        f"PromptABI explanation: {diagnostic.rule_id}",
        f"severity: {diagnostic.severity.value}",
        f"fingerprint: {diagnostic.fingerprint}",
        "",
        "Diagnostic",
        f"  {diagnostic.message}",
        "",
        "Formal property",
        f"  {explanation.property_checked}",
    ]
    if explanation.proof_modes:
        lines.extend(["", "Proof mode"])
        for mode in explanation.proof_modes:
            lines.append(f"  - {mode}")
    if diagnostic.artifact is not None:
        location = diagnostic.artifact.location_uri
        suffix = f" ({location})" if location is not None else ""
        lines.extend(["", "Artifact", f"  {diagnostic.artifact.kind}:{diagnostic.artifact.name}{suffix}"])
    if diagnostic.span is not None:
        span = diagnostic.span
        rendered = f"{span.path}:{span.start_line}:{span.start_column}"
        if span.end_line is not None:
            rendered += f"-{span.end_line}"
            if span.end_column is not None:
                rendered += f":{span.end_column}"
        lines.extend(["", "Location", f"  {rendered}"])
    if explanation.source_snippet is not None:
        lines.extend(["", "Artifact snippet"])
        snippet = explanation.source_snippet
        if snippet.note is not None:
            lines.append(f"  {snippet.note}")
        for offset, line in enumerate(snippet.lines):
            lines.append(f"  {snippet.start_line + offset}: {line}")
    if diagnostic.witness is not None:
        lines.extend(["", "Witness", f"  {diagnostic.witness.summary}"])
        for step_index, step in enumerate(diagnostic.witness.steps, start=1):
            rendered = step.action
            if step.input is not None:
                rendered += f" | input: {step.input}"
            if step.output is not None:
                rendered += f" | output: {step.output}"
            lines.append(f"  {step_index}. {rendered}")
    lines.extend(["", "Likely production symptom", f"  {explanation.likely_symptom}"])
    if diagnostic.suggestions:
        lines.extend(["", "Fix suggestions"])
        for suggestion in diagnostic.suggestions:
            lines.append(f"  - {suggestion}")
    else:
        lines.extend(["", "Fix suggestions", "  - No automatic fix suggestion is attached to this diagnostic."])
    return "\n".join(lines) + "\n"


def render_explanation_json(explanation: DiagnosticExplanation) -> str:
    """Render an explanation as stable JSON."""

    return json.dumps(explanation.to_dict(), indent=2, sort_keys=True) + "\n"


def _deduplicate_diagnostics(diagnostics: tuple[Diagnostic, ...]) -> tuple[Diagnostic, ...]:
    seen: set[str] = set()
    deduplicated = []
    for diagnostic in diagnostics:
        key = json.dumps(diagnostic.to_dict(), sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(diagnostic)
    return tuple(deduplicated)


def _source_snippet(diagnostic: Diagnostic, *, base_dir: Path | None) -> SourceSnippet | None:
    path_text: str | None = None
    start_line = 1
    end_line = 1
    if diagnostic.span is not None:
        path_text = diagnostic.span.path
        start_line = diagnostic.span.start_line
        end_line = diagnostic.span.end_line or start_line
    elif diagnostic.artifact is not None:
        path_text = diagnostic.artifact.path
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        return SourceSnippet(path=str(path), start_line=start_line, lines=(), note="source file is not available locally")
    try:
        source_lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return SourceSnippet(path=str(path), start_line=start_line, lines=(), note="source file is not UTF-8 text")
    except OSError as exc:
        return SourceSnippet(path=str(path), start_line=start_line, lines=(), note=f"source file could not be read: {exc}")
    context_start = max(1, start_line - 2)
    context_end = min(len(source_lines), max(end_line, start_line) + 2)
    if context_start > context_end:
        return SourceSnippet(path=str(path), start_line=start_line, lines=(), note="source span is outside the file")
    return SourceSnippet(
        path=str(path),
        start_line=context_start,
        lines=tuple(source_lines[context_start - 1 : context_end]),
    )
