"""CLI renderers for PromptABI diagnostics."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from .session import VerificationResult


def render_text(
    result: VerificationResult,
    *,
    verbosity: int = 0,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
) -> str:
    lines = [
        f"PromptABI verification: {result.config.name}",
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


def render_sarif(result: VerificationResult) -> str:
    """Render a SARIF 2.1.0 log suitable for GitHub code scanning."""

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
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "PromptABI",
                        "informationUri": "https://github.com/thehalleyyoung/PromptABI",
                        "rules": rules,
                    }
                },
                "results": [_diagnostic_to_sarif_result(diagnostic) for diagnostic in result.diagnostics],
            }
        ],
    }
    return json.dumps(sarif, indent=2, sort_keys=True) + "\n"


def _diagnostic_to_sarif_result(diagnostic) -> dict[str, object]:
    result: dict[str, object] = {
        "ruleId": diagnostic.rule_id,
        "level": diagnostic.severity.sarif_level,
        "message": {"text": diagnostic.message},
        "partialFingerprints": {"promptabiFingerprint": diagnostic.fingerprint},
        "properties": {
            "severity": diagnostic.severity.value,
            "suggestions": list(diagnostic.suggestions),
            "checkModes": [mode.value for mode in diagnostic.check_modes],
        },
    }
    if diagnostic.witness is not None:
        result["properties"]["witness"] = diagnostic.witness.to_dict()
    if diagnostic.properties:
        result["properties"].update(dict(diagnostic.properties))
    location = _sarif_location(diagnostic)
    if location is not None:
        result["locations"] = [location]
    return result


def _sarif_location(diagnostic) -> dict[str, object] | None:
    artifact_uri = None
    if diagnostic.span is not None:
        artifact_uri = diagnostic.span.path
    elif diagnostic.artifact is not None:
        artifact_uri = diagnostic.artifact.location_uri
    if artifact_uri is None:
        return None

    physical_location: dict[str, object] = {"artifactLocation": {"uri": artifact_uri}}
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
