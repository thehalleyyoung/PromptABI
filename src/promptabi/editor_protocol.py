"""Language-server-style diagnostics for editor integrations."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import ConfigError, VerificationConfig, discover_config, load_config
from .diagnostics import ArtifactRef, CheckMode, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessTrace
from .plugins import PluginRegistry
from .session import VerificationResult, VerificationSession

EDITOR_PROTOCOL_VERSION = "promptabi.editorDiagnostics.v1"
PROMPTABI_LSP_SOURCE = "PromptABI"


class EditorProtocolError(ValueError):
    """Raised when an editor diagnostic report cannot be produced."""


@dataclass(frozen=True, slots=True)
class EditorDiagnosticReport:
    """A deterministic editor-facing diagnostic batch."""

    config_path: Path
    workspace_root: Path
    result: VerificationResult
    documents: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return self.result.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EDITOR_PROTOCOL_VERSION,
            "workspaceRoot": _document_uri(self.workspace_root),
            "config": _document_uri(self.config_path),
            "ok": self.ok,
            "documents": list(self.documents),
        }


def build_editor_diagnostic_report(
    *,
    config_path: str | Path | None = None,
    artifact_overrides: dict[str, str] | None = None,
    workspace_root: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> EditorDiagnosticReport:
    """Run PromptABI and render grouped LSP ``publishDiagnostics`` notifications.

    The report is intentionally not a long-running LSP server. It is a
    deterministic protocol payload that editor plugins, notebooks, and language
    server shims can request on save, on open, or from a background watcher.
    """

    resolved_config = Path(config_path).expanduser().resolve() if config_path else discover_config()
    resolved_workspace = Path(workspace_root).expanduser().resolve() if workspace_root else resolved_config.parent
    try:
        config = load_config(resolved_config)
        if artifact_overrides:
            config = config.with_artifact_overrides(artifact_overrides, base_dir=Path.cwd())
        session = VerificationSession(config, plugin_registry=plugin_registry)
        result = session.run()
    except ConfigError as exc:
        result = _config_error_result(resolved_config, exc)
    except OSError as exc:
        raise EditorProtocolError(f"cannot build editor diagnostics: {exc}") from exc

    documents = _publish_diagnostics_documents(
        result,
        config_path=resolved_config,
        workspace_root=resolved_workspace,
    )
    return EditorDiagnosticReport(
        config_path=resolved_config,
        workspace_root=resolved_workspace,
        result=result,
        documents=documents,
    )


def render_editor_diagnostic_json(report: EditorDiagnosticReport) -> str:
    """Render an editor diagnostic report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_editor_diagnostic_text(report: EditorDiagnosticReport) -> str:
    """Render a compact human summary of an editor diagnostic report."""

    total = sum(len(document["params"]["diagnostics"]) for document in report.documents)
    lines = [
        "PromptABI editor diagnostics:",
        f"protocol: {EDITOR_PROTOCOL_VERSION}",
        f"config: {report.config_path}",
        f"workspace: {report.workspace_root}",
        f"documents: {len(report.documents)}",
        f"diagnostics: {total}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
    ]
    for document in report.documents:
        diagnostics = document["params"]["diagnostics"]
        if not diagnostics:
            continue
        lines.append(f"{document['params']['uri']}: {len(diagnostics)}")
        for diagnostic in diagnostics:
            code = diagnostic.get("code", "promptabi")
            lines.append(f"  {code}: {diagnostic['message']}")
    return "\n".join(lines) + "\n"


def _publish_diagnostics_documents(
    result: VerificationResult,
    *,
    config_path: Path,
    workspace_root: Path,
) -> tuple[dict[str, Any], ...]:
    by_uri: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for uri in _watched_document_uris(result.config, config_path):
        by_uri.setdefault(uri, [])
    for diagnostic in result.diagnostics:
        uri = _diagnostic_uri(diagnostic, fallback=config_path)
        by_uri.setdefault(uri, []).append(_lsp_diagnostic(diagnostic, workspace_root=workspace_root))

    return tuple(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": diagnostics,
            },
        }
        for uri, diagnostics in sorted(by_uri.items(), key=lambda item: item[0])
    )


def _watched_document_uris(config: VerificationConfig, config_path: Path) -> tuple[str, ...]:
    uris = {_document_uri(config_path)}
    for artifact in config.artifact_bundle:
        if artifact.location.path is not None:
            uris.add(_document_uri(Path(artifact.location.path)))
    return tuple(sorted(uris))


def _diagnostic_uri(diagnostic: Diagnostic, *, fallback: Path) -> str:
    if diagnostic.span is not None:
        return _document_uri_or_raw(diagnostic.span.path)
    if diagnostic.artifact is not None and diagnostic.artifact.location_uri is not None:
        return _document_uri_or_raw(diagnostic.artifact.location_uri)
    return _document_uri(fallback)


def _lsp_diagnostic(diagnostic: Diagnostic, *, workspace_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "range": _lsp_range(diagnostic.span),
        "severity": _lsp_severity(diagnostic.severity),
        "source": PROMPTABI_LSP_SOURCE,
        "code": diagnostic.rule_id,
        "message": diagnostic.message,
        "data": {
            "fingerprint": diagnostic.fingerprint,
            "severity": diagnostic.severity.value,
            "checkModes": [mode.value for mode in diagnostic.check_modes],
            "suggestions": list(diagnostic.suggestions),
            "workspaceRoot": _document_uri(workspace_root),
            "protocol": EDITOR_PROTOCOL_VERSION,
        },
    }
    if diagnostic.artifact is not None:
        payload["data"]["artifact"] = diagnostic.artifact.to_dict()
    if diagnostic.witness is not None:
        payload["data"]["witness"] = _editor_witness(diagnostic.witness)
    related = _related_information(diagnostic)
    if related:
        payload["relatedInformation"] = related
    return payload


def _lsp_range(span: SourceSpan | None) -> dict[str, dict[str, int]]:
    if span is None:
        return {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        }
    end_line = span.end_line if span.end_line is not None else span.start_line
    end_column = span.end_column if span.end_column is not None else span.start_column
    return {
        "start": {"line": span.start_line - 1, "character": span.start_column - 1},
        "end": {"line": max(0, end_line - 1), "character": max(1, end_column)},
    }


def _lsp_severity(severity: DiagnosticSeverity) -> int:
    return {
        DiagnosticSeverity.ERROR: 1,
        DiagnosticSeverity.WARNING: 2,
        DiagnosticSeverity.INFO: 3,
    }[severity]


def _related_information(diagnostic: Diagnostic) -> list[dict[str, Any]]:
    if diagnostic.artifact is None or diagnostic.artifact.location_uri is None:
        return []
    return [
        {
            "location": {
                "uri": _document_uri_or_raw(diagnostic.artifact.location_uri),
                "range": _lsp_range(None),
            },
            "message": f"{diagnostic.artifact.kind}:{diagnostic.artifact.name}",
        }
    ]


def _editor_witness(witness: WitnessTrace) -> dict[str, Any]:
    return {
        "summary": witness.summary,
        "steps": [step.to_dict() for step in witness.steps],
        "artifacts": [artifact.to_dict() for artifact in witness.artifacts],
    }


def _config_error_result(config_path: Path, exc: ConfigError) -> VerificationResult:
    span = _config_error_span(config_path, str(exc))
    diagnostic = Diagnostic(
        rule_id="config-load-failed",
        severity=DiagnosticSeverity.ERROR,
        message=str(exc),
        artifact=ArtifactRef(kind="config", name=config_path.name, path=str(config_path)),
        span=span,
        check_modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        suggestions=("Fix the PromptABI JSON config before running editor diagnostics.",),
    )
    return VerificationResult(config=VerificationConfig(name=config_path.stem, checks=()), diagnostics=(diagnostic,))


def _config_error_span(config_path: Path, message: str) -> SourceSpan:
    match = re.search(r":(\d+):(\d+):", message)
    if match:
        return SourceSpan(path=str(config_path), start_line=int(match.group(1)), start_column=int(match.group(2)))
    return SourceSpan(path=str(config_path), start_line=1, start_column=1)


def _document_uri_or_raw(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return value
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve().as_uri()
    return _document_uri(Path(value))


def _document_uri(path: Path) -> str:
    return path.expanduser().resolve().as_uri()
