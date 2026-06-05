"""CLI renderers for PromptABI diagnostics."""

from __future__ import annotations

import hashlib
import html
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from .session import VerificationResult

SARIF_CONTROL_PROPERTY_KEYS = frozenset(
    {
        "sarif_suppression",
        "sarif_suppressions",
        "sarif_suppression_justification",
    }
)
SARIF_PROJECT_ROOT_ID = "PROJECTROOT"


@dataclass(frozen=True, slots=True)
class SarifRenderOptions:
    """GitHub code-scanning options for deterministic SARIF rendering."""

    category: str | None = None
    checkout_uri_base: Path | None = None
    include_invocation: bool = False
    command_line: str | None = None
    working_directory: Path | None = None


def render_text(
    result: VerificationResult,
    *,
    verbosity: int = 0,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    heading: str = "PromptABI verification",
) -> str:
    lines = [
        f"{heading}: {result.config.name}",
        f"checks: {', '.join(result.config.checks) if result.config.checks else '(none)'}",
        f"status: {'PASS' if result.ok else 'FAIL'}",
    ]
    if verbosity > 0:
        if config_path is not None:
            lines.append(f"config: {config_path}")
        if cache_dir is not None:
            lines.append(f"cache: {cache_dir}")
        lines.append(f"artifacts: {len(result.config.artifact_bundle.artifacts)}")
    for diagnostic in result.diagnostics:
        if verbosity < 0 and diagnostic.severity.value == "info":
            continue
        mode_suffix = ""
        if diagnostic.check_modes:
            mode_suffix = " [" + ", ".join(mode.label for mode in diagnostic.check_modes) + "]"
        lines.append(
            f"{diagnostic.severity.value.upper()} {diagnostic.rule_id}{mode_suffix}: {diagnostic.message}"
        )
        lines.append(f"  fingerprint: {diagnostic.fingerprint}")
        if diagnostic.artifact is not None:
            location = diagnostic.artifact.location_uri
            suffix = f" ({location})" if location is not None else ""
            lines.append(f"  artifact: {diagnostic.artifact.kind}:{diagnostic.artifact.name}{suffix}")
        if diagnostic.span is not None:
            span = diagnostic.span
            location = f"{span.path}:{span.start_line}:{span.start_column}"
            if span.end_line is not None:
                location += f"-{span.end_line}"
                if span.end_column is not None:
                    location += f":{span.end_column}"
            lines.append(f"  span: {location}")
        if diagnostic.witness is not None:
            lines.append(f"  witness: {diagnostic.witness.summary}")
            for index, step in enumerate(diagnostic.witness.steps, start=1):
                rendered = step.action
                if step.input is not None:
                    rendered += f" | input: {step.input}"
                if step.output is not None:
                    rendered += f" | output: {step.output}"
                lines.append(f"    {index}. {rendered}")
            lines.extend(_witness_detail_lines(diagnostic.witness))
        for suggestion in diagnostic.suggestions:
            lines.append(f"  suggestion: {suggestion}")
    return "\n".join(lines) + "\n"


def render_json(result: VerificationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


def render_html(result: VerificationResult) -> str:
    """Render a self-contained, web-free HTML verification report."""

    severity_counts = _severity_counts(result)
    diagnostic_rows = "\n".join(_diagnostic_summary_row(diagnostic, index) for index, diagnostic in enumerate(result.diagnostics, start=1))
    diagnostic_details = "\n".join(_diagnostic_detail(diagnostic, index) for index, diagnostic in enumerate(result.diagnostics, start=1))
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>PromptABI report: {_escape(result.config.name)}</title>",
            "<style>",
            _HTML_STYLE,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>PromptABI static report: {_escape(result.config.name)}</h1>",
            '<section class="hero">',
            f'<div class="status {("pass" if result.ok else "fail")}">{"PASS" if result.ok else "FAIL"}</div>',
            _metric_card("Diagnostics", str(len(result.diagnostics))),
            _metric_card("Errors", str(severity_counts["error"])),
            _metric_card("Warnings", str(severity_counts["warning"])),
            _metric_card("Info", str(severity_counts["info"])),
            "</section>",
            "<section>",
            "<h2>Configured interface contract</h2>",
            _artifact_table(result),
            "</section>",
            "<section>",
            "<h2>Diagnostics</h2>",
            '<table class="diagnostics"><thead><tr><th>#</th><th>Severity</th><th>Rule</th><th>Message</th><th>Location</th><th>Fingerprint</th></tr></thead>',
            f"<tbody>{diagnostic_rows}</tbody></table>",
            diagnostic_details,
            "</section>",
            _specialized_section(
                "Interactive witness explorer",
                "Open each self-contained witness to inspect role-region overlays, token-boundary views, and solver-assignment tables without network assets or JavaScript.",
                _interactive_witness_explorer(result),
            ),
            _specialized_section(
                "Prompt and parser visualizations",
                "Rendered prompt, tokenizer, role-boundary, parser, and template observations extracted from witnesses.",
                _prompt_visualization_cards(result),
            ),
            _specialized_section(
                "Automata and grammar witnesses",
                "Finite-language, tokenizer x grammar, parser-state, and constrained-decoding evidence.",
                _witness_cards(result, _is_automata_or_grammar_diagnostic),
            ),
            _specialized_section(
                "SMT and finite-contract witnesses",
                "Z3-backed and finite-enumeration obligations with solver statuses, models, and unsat cores.",
                _witness_cards(result, _is_smt_diagnostic),
            ),
            _specialized_section(
                "Token-budget charts",
                "Context-window reservations, segment survival, truncation boundaries, and dropped prompt fields.",
                _token_budget_charts(result),
            ),
            _specialized_section(
                "Artifact diffs",
                "Contract-breaking changes between baseline and current PromptABI configurations.",
                _artifact_diff_rows(result),
                table_headers=("Rule", "Severity", "Change", "Properties", "Fingerprint"),
            ),
            _specialized_section(
                "Corpus and fixture summaries",
                "Seed corpus, structured-schema corpus, provider-fixture, and replay summary diagnostics.",
                _corpus_summary_rows(result),
                table_headers=("Rule", "Severity", "Summary", "Properties", "Fingerprint"),
            ),
            "</main>",
            "</body>",
            "</html>",
            "",
        )
    )


def render_sarif(result: VerificationResult, *, options: SarifRenderOptions | None = None) -> str:
    """Render a SARIF 2.1.0 log suitable for GitHub code scanning."""

    options = options or SarifRenderOptions()
    rule_ids = OrderedDict((diagnostic.rule_id, diagnostic) for diagnostic in result.diagnostics)
    rules = [
        {
            "id": rule_id,
            "name": rule_id,
            "shortDescription": {"text": diagnostic.message},
            "defaultConfiguration": {"level": diagnostic.severity.sarif_level},
            "properties": {
                "checkModes": [mode.value for mode in diagnostic.check_modes],
                "precision": "high",
            },
        }
        for rule_id, diagnostic in sorted(rule_ids.items())
    ]
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "PromptABI",
                "informationUri": "https://github.com/thehalleyyoung/PromptABI",
                "rules": rules,
            }
        },
        "results": [_diagnostic_to_sarif_result(diagnostic, options=options) for diagnostic in result.diagnostics],
    }
    if options.category is not None:
        category = _normalize_sarif_category(options.category)
        run["automationDetails"] = {
            "id": category,
            "guid": str(uuid5(NAMESPACE_URL, f"https://github.com/thehalleyyoung/PromptABI/sarif/{category}")),
        }
    if options.checkout_uri_base is not None:
        run["originalUriBaseIds"] = {
            SARIF_PROJECT_ROOT_ID: {
                "uri": _directory_file_uri(options.checkout_uri_base),
                "description": {"text": "Repository checkout root used by GitHub code scanning."},
            }
        }
    if options.include_invocation:
        invocation: dict[str, Any] = {"executionSuccessful": result.ok}
        if options.command_line is not None:
            invocation["commandLine"] = options.command_line
        if options.working_directory is not None:
            invocation["workingDirectory"] = {"uri": _directory_file_uri(options.working_directory)}
        run["invocations"] = [invocation]

    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }
    return json.dumps(sarif, indent=2, sort_keys=True) + "\n"


def render_github_annotations(result: VerificationResult, *, checkout_uri_base: Path | None = None) -> str:
    """Render GitHub Actions workflow commands for pull-request annotations."""

    lines = []
    for diagnostic in result.diagnostics:
        level = {
            "error": "error",
            "warning": "warning",
            "info": "notice",
        }[diagnostic.severity.value]
        properties = {"title": diagnostic.rule_id}
        location = _annotation_location(diagnostic, checkout_uri_base)
        properties.update(location)
        rendered_properties = ",".join(
            f"{key}={_escape_github_command_property(value)}" for key, value in properties.items()
        )
        lines.append(
            f"::{level} {rendered_properties}::{_escape_github_command_data(diagnostic.message)}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _diagnostic_to_sarif_result(diagnostic, *, options: SarifRenderOptions) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "severity": diagnostic.severity.value,
        "suggestions": list(diagnostic.suggestions),
        "checkModes": [mode.value for mode in diagnostic.check_modes],
    }
    if _sarif_github_enabled(options):
        properties["githubCodeScanning"] = {
            "annotationLevel": diagnostic.severity.sarif_level,
            "fingerprint": diagnostic.fingerprint,
        }
    result: dict[str, object] = {
        "ruleId": diagnostic.rule_id,
        "level": diagnostic.severity.sarif_level,
        "message": {"text": diagnostic.message},
        "partialFingerprints": {"promptabiFingerprint": diagnostic.fingerprint},
        "properties": properties,
    }
    if diagnostic.witness is not None:
        properties["witness"] = diagnostic.witness.to_dict()
    if diagnostic.properties:
        properties.update(
            {
                key: value
                for key, value in dict(diagnostic.properties).items()
                if key not in SARIF_CONTROL_PROPERTY_KEYS
            }
        )
    suppressions = _sarif_suppressions(dict(diagnostic.properties))
    if suppressions:
        result["suppressions"] = suppressions
    location = _sarif_location(diagnostic, options=options)
    if location is not None:
        result["locations"] = [location]
        if _sarif_github_enabled(options):
            result["partialFingerprints"]["promptabiLocationFingerprint"] = _location_fingerprint(
                diagnostic.rule_id,
                location,
                diagnostic.fingerprint,
            )
    return result


def _sarif_location(diagnostic, *, options: SarifRenderOptions) -> dict[str, object] | None:
    artifact_uri = None
    if diagnostic.span is not None:
        artifact_uri = diagnostic.span.path
    elif diagnostic.artifact is not None:
        artifact_uri = diagnostic.artifact.location_uri
    if artifact_uri is None:
        return None

    physical_location: dict[str, object] = {
        "artifactLocation": _sarif_artifact_location(artifact_uri, options=options)
    }
    if diagnostic.span is not None:
        region: dict[str, int] = {
            "startLine": diagnostic.span.start_line,
            "startColumn": diagnostic.span.start_column,
        }
        if diagnostic.span.end_line is not None:
            region["endLine"] = diagnostic.span.end_line
        if diagnostic.span.end_column is not None:
            region["endColumn"] = diagnostic.span.end_column
        physical_location["region"] = region
    return {"physicalLocation": physical_location}


def _sarif_artifact_location(uri: str, *, options: SarifRenderOptions) -> dict[str, str]:
    if options.checkout_uri_base is None:
        return {"uri": uri}
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return {"uri": uri}
    raw_path = Path(parsed.path) if parsed.scheme == "file" else Path(uri)
    if raw_path.is_absolute():
        base = options.checkout_uri_base.resolve()
        candidate = raw_path.resolve()
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            return {"uri": candidate.as_uri()}
        return {"uri": relative.as_posix(), "uriBaseId": SARIF_PROJECT_ROOT_ID}
    return {"uri": raw_path.as_posix(), "uriBaseId": SARIF_PROJECT_ROOT_ID}


def _annotation_location(diagnostic, checkout_uri_base: Path | None) -> dict[str, str]:
    location: dict[str, str] = {}
    artifact_uri = None
    if diagnostic.span is not None:
        artifact_uri = diagnostic.span.path
        location["file"] = _annotation_file(artifact_uri, checkout_uri_base)
        location["line"] = str(diagnostic.span.start_line)
        location["col"] = str(diagnostic.span.start_column)
        if diagnostic.span.end_line is not None:
            location["endLine"] = str(diagnostic.span.end_line)
        if diagnostic.span.end_column is not None:
            location["endColumn"] = str(diagnostic.span.end_column)
    elif diagnostic.artifact is not None:
        artifact_uri = diagnostic.artifact.location_uri
        if artifact_uri is not None:
            location["file"] = _annotation_file(artifact_uri, checkout_uri_base)
    return location


def _annotation_file(uri: str, checkout_uri_base: Path | None) -> str:
    if checkout_uri_base is None:
        return uri
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return uri
    raw_path = Path(parsed.path) if parsed.scheme == "file" else Path(uri)
    if raw_path.is_absolute():
        try:
            return raw_path.resolve().relative_to(checkout_uri_base.resolve()).as_posix()
        except ValueError:
            return raw_path.resolve().as_posix()
    return raw_path.as_posix()


def _sarif_suppressions(properties: dict[str, object]) -> list[dict[str, str]]:
    raw_suppressions = properties.get("sarif_suppressions")
    if isinstance(raw_suppressions, list):
        suppressions = []
        for item in raw_suppressions:
            if isinstance(item, dict):
                kind = item.get("kind")
                if kind in {"external", "inSource"}:
                    suppression = {"kind": kind}
                    justification = item.get("justification")
                    if isinstance(justification, str) and justification:
                        suppression["justification"] = justification
                    suppressions.append(suppression)
        if suppressions:
            return suppressions
    raw_suppression = properties.get("sarif_suppression")
    if isinstance(raw_suppression, dict):
        kind = raw_suppression.get("kind")
        if kind in {"external", "inSource"}:
            suppression = {"kind": kind}
            justification = raw_suppression.get("justification")
            if isinstance(justification, str) and justification:
                suppression["justification"] = justification
            return [suppression]
    justification = properties.get("sarif_suppression_justification")
    if isinstance(justification, str) and justification:
        return [{"kind": "external", "justification": justification}]
    return []


def _location_fingerprint(rule_id: str, location: dict[str, object], diagnostic_fingerprint: str) -> str:
    encoded = json.dumps(
        {"ruleId": rule_id, "location": location, "diagnostic": diagnostic_fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalize_sarif_category(category: str) -> str:
    stripped = category.strip().strip("/")
    if not stripped:
        stripped = "promptabi"
    return f"{stripped}/"


def _sarif_github_enabled(options: SarifRenderOptions) -> bool:
    return (
        options.category is not None
        or options.checkout_uri_base is not None
        or options.include_invocation
    )


def _directory_file_uri(path: Path) -> str:
    uri = path.expanduser().resolve().as_uri()
    return uri if uri.endswith("/") else f"{uri}/"


def _escape_github_command_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_github_command_property(value: str) -> str:
    return (
        _escape_github_command_data(value)
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


_HTML_STYLE = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f8fa; color: #1f2328; }
main { max-width: 1180px; margin: 0 auto; padding: 32px; }
h1 { margin: 0 0 16px; font-size: 2rem; letter-spacing: -0.03em; }
h2 { margin-top: 32px; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
h3 { margin: 0 0 8px; }
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0 28px; }
.status, .metric, .card, details { background: #fff; border: 1px solid #d0d7de; border-radius: 12px; box-shadow: 0 1px 2px #1f232812; }
.status { display: grid; place-items: center; min-height: 86px; font-size: 1.5rem; font-weight: 800; }
.pass { color: #116329; border-color: #2da44e; }
.fail { color: #cf222e; border-color: #cf222e; }
.metric { padding: 16px; }
.metric strong { display: block; font-size: 1.8rem; }
.metric span { color: #57606a; font-size: 0.9rem; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d7de; border-radius: 12px; overflow: hidden; }
th, td { border-bottom: 1px solid #d8dee4; padding: 10px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-size: 0.85rem; color: #57606a; }
tr:last-child td { border-bottom: 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
code { background: #f6f8fa; border-radius: 6px; padding: 2px 5px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8fa; padding: 10px; border-radius: 8px; border: 1px solid #d0d7de; }
details { margin: 12px 0; padding: 12px; }
summary { cursor: pointer; font-weight: 700; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; background: #eaeef2; margin: 0 4px 4px 0; }
.badge.error { background: #ffebe9; color: #cf222e; }
.badge.warning { background: #fff8c5; color: #7d4e00; }
.badge.info { background: #ddf4ff; color: #0969da; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.card { padding: 14px; overflow: hidden; }
.muted { color: #57606a; }
.bar-row { display: grid; grid-template-columns: minmax(120px, 1.2fr) minmax(160px, 3fr) 72px; gap: 10px; align-items: center; margin: 8px 0; }
.bar-track { background: #eaeef2; border-radius: 999px; overflow: hidden; min-height: 14px; }
.bar { min-height: 14px; border-radius: 999px; background: #0969da; }
.bar.dropped { background: #cf222e; }
.bar.kept { background: #2da44e; }
.bar.unknown { background: repeating-linear-gradient(45deg, #8c959f, #8c959f 4px, #afb8c1 4px, #afb8c1 8px); }
.empty { background: #fff; border: 1px dashed #afb8c1; border-radius: 12px; padding: 12px; color: #57606a; }
.explorer-grid { display: grid; gap: 12px; }
.explorer-card[open] { border-color: #0969da; }
.explorer-panel { margin-top: 12px; border-top: 1px solid #d8dee4; padding-top: 12px; }
.role-overlay { white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 10px; }
.role-region { border: 2px solid #0969da; background: #ddf4ff; border-radius: 5px; padding: 0 2px; }
.role-region.assistant { border-color: #8250df; background: #fbefff; }
.role-region.system { border-color: #bf8700; background: #fff8c5; }
.role-region.user { border-color: #1a7f37; background: #dafbe1; }
.token-boundaries { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.token-boundary { border: 1px solid #8c959f; border-radius: 8px; background: #fff; padding: 4px 7px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
.token-boundary::before { content: "|"; color: #cf222e; margin-right: 4px; font-weight: 800; }
.assignment-table th:first-child { width: 35%; }
""".strip()


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _severity_counts(result: VerificationResult) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for diagnostic in result.diagnostics:
        counts[diagnostic.severity.value] = counts.get(diagnostic.severity.value, 0) + 1
    return counts


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>'


def _artifact_table(result: VerificationResult) -> str:
    rows = []
    for artifact in sorted(result.config.artifact_bundle.artifacts, key=lambda item: (item.kind.value, item.name)):
        location = artifact.location.path or artifact.location.uri or ""
        provenance = artifact.provenance
        pin = provenance.ref_version or ""
        rows.append(
            "<tr>"
            f"<td>{_escape(artifact.kind.value)}</td>"
            f"<td><code>{_escape(artifact.name)}</code></td>"
            f"<td>{_escape(location)}</td>"
            f"<td>{_escape(pin or 'unversioned')}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="4" class="muted">No artifacts configured.</td></tr>')
    return (
        "<table><thead><tr><th>Kind</th><th>Name</th><th>Location</th><th>Pin</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _diagnostic_summary_row(diagnostic, index: int) -> str:
    return (
        f'<tr id="diag-{index}">'
        f"<td>{index}</td>"
        f'<td>{_severity_badge(diagnostic.severity.value)}</td>'
        f"<td><code>{_escape(diagnostic.rule_id)}</code></td>"
        f"<td>{_escape(diagnostic.message)}</td>"
        f"<td>{_escape(_diagnostic_location(diagnostic))}</td>"
        f"<td><code>{_escape(diagnostic.fingerprint)}</code></td>"
        "</tr>"
    )


def _diagnostic_detail(diagnostic, index: int) -> str:
    modes = "".join(f'<span class="badge">{_escape(mode.value)}</span>' for mode in diagnostic.check_modes)
    suggestions = "".join(f"<li>{_escape(suggestion)}</li>" for suggestion in diagnostic.suggestions)
    witness = _witness_block(diagnostic)
    properties = _properties_block(dict(diagnostic.properties))
    artifact = ""
    if diagnostic.artifact is not None:
        artifact = f"<p><strong>Artifact:</strong> {_escape(diagnostic.artifact.kind)}:<code>{_escape(diagnostic.artifact.name)}</code></p>"
    return (
        "<details>"
        f"<summary>#{index} {_severity_badge(diagnostic.severity.value)} <code>{_escape(diagnostic.rule_id)}</code> {_escape(diagnostic.message)}</summary>"
        f"<p><strong>Fingerprint:</strong> <code>{_escape(diagnostic.fingerprint)}</code></p>"
        f"<p><strong>Location:</strong> {_escape(_diagnostic_location(diagnostic))}</p>"
        f"{artifact}"
        f"{'<p><strong>Modes:</strong> ' + modes + '</p>' if modes else ''}"
        f"{'<h3>Suggestions</h3><ul>' + suggestions + '</ul>' if suggestions else ''}"
        f"{witness}{properties}"
        "</details>"
    )


def _severity_badge(severity: str) -> str:
    return f'<span class="badge {_escape(severity)}">{_escape(severity.upper())}</span>'


def _diagnostic_location(diagnostic) -> str:
    if diagnostic.span is not None:
        span = diagnostic.span
        location = f"{span.path}:{span.start_line}:{span.start_column}"
        if span.end_line is not None:
            location += f"-{span.end_line}"
            if span.end_column is not None:
                location += f":{span.end_column}"
        return location
    if diagnostic.artifact is not None and diagnostic.artifact.location_uri is not None:
        return diagnostic.artifact.location_uri
    return "run"


def _witness_block(diagnostic) -> str:
    if diagnostic.witness is None:
        return ""
    steps = "".join(
        "<li>"
        f"<strong>{_escape(step.action)}</strong>"
        f"{' input=' + _inline_code(step.input) if step.input is not None else ''}"
        f"{' output=' + _inline_code(step.output) if step.output is not None else ''}"
        "</li>"
        for step in diagnostic.witness.steps
    )
    artifacts = "".join(
        f"<li>{_escape(artifact.kind)}:<code>{_escape(artifact.name)}</code>"
        f"{' ' + _escape(artifact.location_uri) if artifact.location_uri is not None else ''}</li>"
        for artifact in diagnostic.witness.artifacts
    )
    return (
        "<h3>Witness</h3>"
        f"<p>{_escape(diagnostic.witness.summary)}</p>"
        f"<ol>{steps}</ol>"
        f"{_witness_details_html(diagnostic.witness)}"
        f"{'<h3>Witness artifacts</h3><ul>' + artifacts + '</ul>' if artifacts else ''}"
    )


def _witness_detail_lines(witness) -> list[str]:
    lines: list[str] = []
    for rendered in witness.rendered_strings:
        lines.append(f"    rendered: {_compact_text(rendered)}")
    if witness.token_ids:
        lines.append(f"    token_ids: {list(witness.token_ids)}")
    for region in witness.role_regions:
        role = region.get("role", "unknown")
        role_source = region.get("role_source", "unknown")
        path_index = region.get("path_index", "?")
        region_index = region.get("region_index", "?")
        start = region.get("start_offset", "?")
        end = region.get("end_offset", "?")
        lines.append(
            "    role_region: "
            f"path={path_index} region={region_index} role={role} source={role_source} chars={start}:{end}"
        )
    for state in witness.parser_states:
        lines.append(f"    parser_state: {state}")
    for assignment in witness.solver_assignments:
        lines.append(f"    solver_assignment: {_compact_json(assignment)}")
    for decision in witness.truncation_decisions:
        lines.append(f"    truncation_decision: {_compact_json(decision)}")
    for fix in witness.minimal_fixes:
        lines.append(f"    minimal_fix: {fix}")
    return lines


def _witness_details_html(witness) -> str:
    sections: list[str] = []
    if witness.rendered_strings:
        rendered = "".join(f"<li><pre>{_escape(item)}</pre></li>" for item in witness.rendered_strings)
        sections.append(f"<h3>Rendered strings</h3><ol>{rendered}</ol>")
    if witness.token_ids:
        sections.append(f"<h3>Token IDs</h3><pre>{_escape(list(witness.token_ids))}</pre>")
    if witness.role_regions:
        sections.append(f"<h3>Role regions</h3>{_render_value(list(witness.role_regions))}")
    if witness.parser_states:
        states = "".join(f"<li>{_escape(state)}</li>" for state in witness.parser_states)
        sections.append(f"<h3>Parser states</h3><ul>{states}</ul>")
    if witness.solver_assignments:
        sections.append(f"<h3>Solver assignments</h3>{_render_value(list(witness.solver_assignments))}")
    if witness.truncation_decisions:
        sections.append(f"<h3>Truncation decisions</h3>{_render_value(list(witness.truncation_decisions))}")
    if witness.minimal_fixes:
        fixes = "".join(f"<li>{_escape(fix)}</li>" for fix in witness.minimal_fixes)
        sections.append(f"<h3>Minimal fixes</h3><ul>{fixes}</ul>")
    return "".join(sections)


def _interactive_witness_explorer(result: VerificationResult) -> str:
    cards = []
    for index, diagnostic in enumerate(result.diagnostics, start=1):
        if diagnostic.witness is None:
            continue
        card = _witness_explorer_card(diagnostic, index)
        if card:
            cards.append(card)
    return f'<div class="explorer-grid">{"".join(cards)}</div>' if cards else ""


def _witness_explorer_card(diagnostic, index: int) -> str:
    witness = diagnostic.witness
    panels = "".join(
        panel
        for panel in (
            _role_region_overlay_panel(witness),
            _token_boundary_panel(witness),
            _solver_assignment_panel(witness),
        )
        if panel
    )
    if not panels:
        return ""
    return (
        '<details class="explorer-card">'
        f"<summary>#{index} {_severity_badge(diagnostic.severity.value)} <code>{_escape(diagnostic.rule_id)}</code> "
        f"{_escape(diagnostic.message)}</summary>"
        f"<p class=\"muted\">{_escape(witness.summary)}</p>"
        f"{panels}"
        "</details>"
    )


def _role_region_overlay_panel(witness) -> str:
    if not witness.rendered_strings and not witness.role_regions:
        return ""
    overlays = []
    regions = list(witness.role_regions)
    for path_index, rendered in enumerate(witness.rendered_strings):
        matching_regions = [
            region for region in regions if _region_int(region, "path_index", default=path_index) == path_index
        ]
        overlays.append(
            "<details open>"
            f"<summary>Rendered path {path_index} role-region overlay</summary>"
            f"{_render_role_overlay(rendered, matching_regions)}"
            "</details>"
        )
    if regions:
        overlays.append("<h3>Role-region table</h3>" + _role_region_table(regions))
    return '<div class="explorer-panel"><h3>Role-region overlays</h3>' + "".join(overlays) + "</div>"


def _render_role_overlay(rendered: str, regions: list[dict[str, Any]]) -> str:
    spans: list[tuple[int, int, dict[str, Any]]] = []
    last_end = 0
    for region in sorted(regions, key=lambda item: (_region_int(item, "start_offset"), _region_int(item, "end_offset"))):
        start = _region_int(region, "start_offset")
        end = _region_int(region, "end_offset")
        if start < last_end or start < 0 or end <= start or end > len(rendered):
            continue
        spans.append((start, end, region))
        last_end = end
    if not spans:
        return f'<pre class="role-overlay">{_escape(rendered)}</pre>'
    parts: list[str] = []
    cursor = 0
    for start, end, region in spans:
        if start > cursor:
            parts.append(_escape(rendered[cursor:start]))
        role = str(region.get("role", "unknown"))
        label = (
            f"role={role}; source={region.get('role_source', 'unknown')}; "
            f"region={region.get('region_index', '?')}; chars={start}:{end}"
        )
        parts.append(
            f'<span class="role-region {_css_identifier(role)}" title="{_escape(label)}">'
            f"{_escape(rendered[start:end])}</span>"
        )
        cursor = end
    if cursor < len(rendered):
        parts.append(_escape(rendered[cursor:]))
    return f'<pre class="role-overlay">{"".join(parts)}</pre>'


def _role_region_table(regions: list[dict[str, Any]]) -> str:
    rows = []
    for region in sorted(regions, key=lambda item: (_region_int(item, "path_index"), _region_int(item, "region_index"))):
        rows.append(
            "<tr>"
            f"<td>{_escape(region.get('path_index', '?'))}</td>"
            f"<td>{_escape(region.get('region_index', '?'))}</td>"
            f"<td>{_escape(region.get('role', 'unknown'))}</td>"
            f"<td>{_escape(region.get('role_source', 'unknown'))}</td>"
            f"<td>{_escape(region.get('start_offset', '?'))}:{_escape(region.get('end_offset', '?'))}</td>"
            f"<td>{_render_value(region.get('content_expressions', []))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Path</th><th>Region</th><th>Role</th><th>Source</th><th>Chars</th><th>Inputs</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _token_boundary_panel(witness) -> str:
    if not witness.token_ids:
        return ""
    tokens = "".join(
        f'<span class="token-boundary" title="token index {index}">{_escape(token_id)}</span>'
        for index, token_id in enumerate(witness.token_ids)
    )
    return (
        '<div class="explorer-panel">'
        "<h3>Token-boundary view</h3>"
        f'<div class="token-boundaries">{tokens}</div>'
        "</div>"
    )


def _solver_assignment_panel(witness) -> str:
    if not witness.solver_assignments:
        return ""
    tables = []
    for index, assignment in enumerate(witness.solver_assignments, start=1):
        rows = "".join(
            "<tr>"
            f"<th>{_escape(key)}</th>"
            f"<td>{_render_value(assignment[key])}</td>"
            "</tr>"
            for key in sorted(assignment, key=str)
        )
        tables.append(
            "<details open>"
            f"<summary>Solver assignment {index}</summary>"
            f'<table class="assignment-table">{rows}</table>'
            "</details>"
        )
    return '<div class="explorer-panel"><h3>Solver-assignment tables</h3>' + "".join(tables) + "</div>"


def _region_int(region: dict[str, Any], key: str, *, default: int = 0) -> int:
    value = region.get(key, default)
    return value if isinstance(value, int) else default


def _css_identifier(value: str) -> str:
    identifier = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return identifier.strip("-") or "unknown"


def _compact_text(value: str, *, limit: int = 240) -> str:
    normalized = value.replace("\n", "\\n").replace("\r", "\\r")
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _inline_code(value: object) -> str:
    return f"<code>{_escape(value)}</code>"


def _properties_block(properties: dict[str, Any]) -> str:
    if not properties:
        return ""
    return f"<h3>Properties</h3>{_render_value(properties)}"


def _render_value(value: Any) -> str:
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><th>{_escape(key)}</th><td>{_render_value(value[key])}</td></tr>"
            for key in sorted(value, key=str)
        )
        return f"<table>{rows}</table>"
    if isinstance(value, (list, tuple)):
        if not value:
            return '<span class="muted">[]</span>'
        return "<ol>" + "".join(f"<li>{_render_value(item)}</li>" for item in value) + "</ol>"
    return _inline_code(value)


def _specialized_section(
    title: str,
    description: str,
    body: str,
    *,
    table_headers: tuple[str, ...] | None = None,
) -> str:
    if not body:
        body = '<div class="empty">No matching diagnostics in this run.</div>'
    elif table_headers is not None:
        header = "".join(f"<th>{_escape(item)}</th>" for item in table_headers)
        body = f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
    return (
        "<section>"
        f"<h2>{_escape(title)}</h2>"
        f'<p class="muted">{_escape(description)}</p>'
        f"{body}"
        "</section>"
    )


def _prompt_visualization_cards(result: VerificationResult) -> str:
    cards = []
    for diagnostic in result.diagnostics:
        if diagnostic.witness is None:
            continue
        prompt_steps = [
            step
            for step in diagnostic.witness.steps
            if any(
                marker in step.action.lower()
                for marker in ("render", "prompt", "template", "tokenize", "role", "parser", "boundary")
            )
        ]
        if not prompt_steps:
            continue
        cards.append(_step_card(diagnostic, prompt_steps))
    return f'<div class="cards">{"".join(cards)}</div>' if cards else ""


def _witness_cards(result: VerificationResult, predicate) -> str:
    cards = []
    for diagnostic in result.diagnostics:
        if predicate(diagnostic):
            steps = list(diagnostic.witness.steps) if diagnostic.witness is not None else []
            cards.append(_step_card(diagnostic, steps))
    return f'<div class="cards">{"".join(cards)}</div>' if cards else ""


def _step_card(diagnostic, steps) -> str:
    rendered_steps = "".join(
        "<li>"
        f"<strong>{_escape(step.action)}</strong>"
        f"{' input=' + _inline_code(step.input) if step.input is not None else ''}"
        f"{' output=' + _inline_code(step.output) if step.output is not None else ''}"
        "</li>"
        for step in steps
    )
    if not rendered_steps:
        rendered_steps = '<li class="muted">No explicit witness steps were emitted.</li>'
    return (
        '<article class="card">'
        f"<h3><code>{_escape(diagnostic.rule_id)}</code></h3>"
        f"<p>{_severity_badge(diagnostic.severity.value)} {_escape(diagnostic.message)}</p>"
        f"<ol>{rendered_steps}</ol>"
        f"{_witness_details_html(diagnostic.witness) if diagnostic.witness is not None else ''}"
        "</article>"
    )


def _is_smt_diagnostic(diagnostic) -> bool:
    if any(mode.value == "z3-backed-smt" for mode in diagnostic.check_modes):
        return True
    if diagnostic.witness is None:
        return False
    return any(
        any(marker in step.action.lower() for marker in ("smt", "z3", "finite contract", "solver", "model", "unsat core"))
        for step in diagnostic.witness.steps
    )


def _is_automata_or_grammar_diagnostic(diagnostic) -> bool:
    if diagnostic.rule_id.startswith(("grammar-", "parser-compatibility")):
        return True
    if diagnostic.witness is None:
        return False
    return any(
        any(marker in step.action.lower() for marker in ("dfa", "automata", "automaton", "grammar", "parser state", "constrained"))
        for step in diagnostic.witness.steps
    )


def _token_budget_charts(result: VerificationResult) -> str:
    charts = []
    for diagnostic in result.diagnostics:
        visualization = dict(diagnostic.properties).get("token_budget_visualization")
        if isinstance(visualization, dict):
            charts.append(_token_budget_chart(diagnostic, visualization))
    return "".join(charts)


def _token_budget_chart(diagnostic, visualization: dict[str, Any]) -> str:
    rows = visualization.get("rows")
    if not isinstance(rows, list):
        rows = []
    totals = [
        int(row.get("total_tokens", 0))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("total_tokens"), int) and row.get("total_tokens", 0) > 0
    ]
    denominator = max([int(visualization.get("input_budget_tokens", 0) or 0), *totals, 1])
    bar_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        total = row.get("total_tokens")
        tokens = total if isinstance(total, int) and total >= 0 else 0
        width = max(2, min(100, round((tokens / denominator) * 100))) if tokens else 2
        status = str(row.get("status", "unknown"))
        css_status = status if status in {"kept", "dropped", "unknown"} else "unknown"
        label = f"{row.get('index', '?')}. {row.get('name', '<unnamed>')}"
        detail = "unknown" if total is None else f"{tokens} token(s)"
        bar_rows.append(
            '<div class="bar-row">'
            f"<div><strong>{_escape(label)}</strong><br><span class=\"muted\">{_escape(row.get('role', 'unknown-role'))} / {_escape(status)}</span></div>"
            '<div class="bar-track">'
            f'<div class="bar {css_status}" style="width: {width}%"></div>'
            "</div>"
            f"<div>{_escape(detail)}</div>"
            "</div>"
        )
    if not bar_rows:
        bar_rows.append('<div class="empty">No prompt segments were available for charting.</div>')
    metadata = (
        f"context={visualization.get('max_context_tokens', 'unknown')}, "
        f"reserved={visualization.get('reserved_total', 'unknown')}, "
        f"input={visualization.get('input_budget_tokens', 'unknown')}, "
        f"required={visualization.get('required_prompt_tokens', 'unknown')}, "
        f"total={visualization.get('total_prompt_tokens', 'unknown')}, "
        f"must-survive={visualization.get('must_survive_status', 'unknown')}"
    )
    return (
        '<article class="card">'
        f"<h3>{_escape(visualization.get('budget_source', 'context budget'))}</h3>"
        f"<p>{_severity_badge(diagnostic.severity.value)} <code>{_escape(diagnostic.rule_id)}</code> {_escape(metadata)}</p>"
        f"{''.join(bar_rows)}"
        "</article>"
    )


def _artifact_diff_rows(result: VerificationResult) -> str:
    rows = []
    for diagnostic in result.diagnostics:
        if not diagnostic.rule_id.startswith("diff-"):
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_escape(diagnostic.rule_id)}</code></td>"
            f"<td>{_severity_badge(diagnostic.severity.value)}</td>"
            f"<td>{_escape(diagnostic.message)}</td>"
            f"<td>{_render_value(dict(diagnostic.properties))}</td>"
            f"<td><code>{_escape(diagnostic.fingerprint)}</code></td>"
            "</tr>"
        )
    return "".join(rows)


def _corpus_summary_rows(result: VerificationResult) -> str:
    rows = []
    for diagnostic in result.diagnostics:
        searchable = " ".join(
            [
                diagnostic.rule_id,
                diagnostic.message,
                " ".join(str(key) for key, _value in diagnostic.properties),
            ]
        ).lower()
        if not any(marker in searchable for marker in ("corpus", "fixture", "provider-fixture", "structured-schema")):
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_escape(diagnostic.rule_id)}</code></td>"
            f"<td>{_severity_badge(diagnostic.severity.value)}</td>"
            f"<td>{_escape(diagnostic.message)}</td>"
            f"<td>{_render_value(dict(diagnostic.properties))}</td>"
            f"<td><code>{_escape(diagnostic.fingerprint)}</code></td>"
            "</tr>"
        )
    return "".join(rows)
