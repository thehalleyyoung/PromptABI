"""CLI renderers for PromptABI diagnostics."""

from __future__ import annotations

import hashlib
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
        for suggestion in diagnostic.suggestions:
            lines.append(f"  suggestion: {suggestion}")
    return "\n".join(lines) + "\n"


def render_json(result: VerificationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


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
