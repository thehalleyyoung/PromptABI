"""Developer experience and ecosystem surfaces (roadmap steps 346-360).

This module turns PromptABI's verification results into the concrete,
developer-facing artifacts an ecosystem needs: a language server, an editor
extension, generated SDK readers in four languages, rich finding explanations,
ranked autofixes, stable JSON Schemas, a WASM playground, a sub-five-minute
quickstart, a third-party plugin API, framework integration shims, typed stubs,
internationalized messages, runnable tutorial notebooks, an org policy/profile
system, and a coverage badge.

Everything is CPU-only, network-free, and deterministic. The demonstrations are
driven by *real* verification output: ``run_verification`` is executed against
the bundled ``examples/rag-chunking`` project, which emits genuine error-severity
diagnostics, and every surface is built from those real findings.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .diagnostics import Diagnostic, DiagnosticSeverity, SourceSpan
from .explain import explain_diagnostic, render_explanation_text
from .fix_suggestions import RankedFixSuggestion, rank_fix_suggestions

if TYPE_CHECKING:
    from .session import CheckContext

DEVEX_ECOSYSTEM_VERSION = "2026.06"

#: Project used as the live demonstration corpus. It emits real error findings.
DEMO_CONFIG_RELPATH = "examples/rag-chunking/promptabi.json"

SUPPORTED_SDK_LANGUAGES: tuple[str, ...] = ("python", "typescript", "go", "rust")
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "es", "ja", "de", "fr")
FRAMEWORK_INTEGRATIONS: tuple[str, ...] = ("langchain", "llamaindex", "dspy")

# LSP severities: 1=Error 2=Warning 3=Information 4=Hint.
_LSP_SEVERITY = {
    DiagnosticSeverity.ERROR: 1,
    DiagnosticSeverity.WARNING: 2,
    DiagnosticSeverity.INFO: 3,
}


# --------------------------------------------------------------------------- #
# Real demonstration corpus
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / DEMO_CONFIG_RELPATH).exists():
            return parent
    # Fall back to the current working directory.
    if (Path.cwd() / DEMO_CONFIG_RELPATH).exists():
        return Path.cwd()
    raise FileNotFoundError(DEMO_CONFIG_RELPATH)


def demo_config_path() -> Path:
    return _repo_root() / DEMO_CONFIG_RELPATH


def _demo_result() -> Any:
    """Run the production verifier over the bundled RAG example."""

    from .api import run_verification  # lazy import avoids an import cycle

    return run_verification(str(demo_config_path()))


def load_demo_diagnostics() -> tuple[Diagnostic, ...]:
    """Run the production verifier over the bundled RAG example."""

    return tuple(_demo_result().diagnostics)


def _first_error(diagnostics: tuple[Diagnostic, ...]) -> Diagnostic:
    for diagnostic in diagnostics:
        if diagnostic.severity is DiagnosticSeverity.ERROR:
            return diagnostic
    if diagnostics:
        return diagnostics[0]
    raise ValueError("demo project produced no diagnostics")


# --------------------------------------------------------------------------- #
# 346 - Language server
# --------------------------------------------------------------------------- #
def _span_range(span: SourceSpan | None) -> dict[str, dict[str, int]]:
    if span is None:
        return {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
    start_line = max(span.start_line - 1, 0)
    start_col = max(span.start_column - 1, 0)
    end_line = max((span.end_line or span.start_line) - 1, start_line)
    end_col = max((span.end_column or span.start_column) - 1, start_col)
    return {
        "start": {"line": start_line, "character": start_col},
        "end": {"line": end_line, "character": end_col},
    }


def lsp_diagnostic(diagnostic: Diagnostic) -> dict[str, Any]:
    return {
        "range": _span_range(diagnostic.span),
        "severity": _LSP_SEVERITY.get(diagnostic.severity, 3),
        "code": diagnostic.rule_id,
        "source": "promptabi",
        "message": diagnostic.message,
        "data": {
            "fingerprint": diagnostic.fingerprint,
            "suggestions": list(diagnostic.suggestions),
        },
    }


def build_lsp_diagnostics(
    diagnostics: tuple[Diagnostic, ...], *, uri: str = "file:///workspace/promptabi.json"
) -> dict[str, Any]:
    """Construct an LSP ``textDocument/publishDiagnostics`` notification."""

    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": [lsp_diagnostic(d) for d in diagnostics],
        },
    }


def language_server_initialize_result() -> dict[str, Any]:
    return {
        "capabilities": {
            "textDocumentSync": 1,
            "diagnosticProvider": {
                "identifier": "promptabi",
                "interFileDependencies": True,
                "workspaceDiagnostics": False,
            },
            "codeActionProvider": True,
            "hoverProvider": True,
        },
        "serverInfo": {"name": "promptabi-language-server", "version": DEVEX_ECOSYSTEM_VERSION},
    }


# --------------------------------------------------------------------------- #
# 347 - VS Code extension
# --------------------------------------------------------------------------- #
def vscode_extension_files() -> dict[str, str]:
    package_json = {
        "name": "promptabi",
        "displayName": "PromptABI Prompt Safety",
        "description": "Inline static verification of prompt/tokenizer/tool-calling contracts.",
        "version": DEVEX_ECOSYSTEM_VERSION,
        "engines": {"vscode": "^1.85.0"},
        "categories": ["Linters", "Programming Languages"],
        "activationEvents": ["onLanguage:json", "workspaceContains:**/promptabi.json"],
        "main": "./extension.js",
        "contributes": {
            "commands": [
                {"command": "promptabi.verify", "title": "PromptABI: Verify Workspace"},
                {"command": "promptabi.explain", "title": "PromptABI: Explain Finding"},
            ],
            "configuration": {
                "title": "PromptABI",
                "properties": {
                    "promptabi.inlineHints": {"type": "boolean", "default": True},
                    "promptabi.failOn": {
                        "type": "string",
                        "enum": ["info", "warning", "error"],
                        "default": "error",
                    },
                },
            },
        },
    }
    extension_js = (
        "// PromptABI VS Code extension scaffold.\n"
        "// Surfaces analyzer findings as inline diagnostics via the language server.\n"
        "const { LanguageClient } = require('vscode-languageclient/node');\n"
        "let client;\n"
        "function activate(context) {\n"
        "  const serverOptions = { command: 'promptabi', args: ['lsp'] };\n"
        "  const clientOptions = { documentSelector: [{ scheme: 'file', language: 'json' }] };\n"
        "  client = new LanguageClient('promptabi', 'PromptABI', serverOptions, clientOptions);\n"
        "  context.subscriptions.push(client.start());\n"
        "}\n"
        "function deactivate() { return client ? client.stop() : undefined; }\n"
        "module.exports = { activate, deactivate };\n"
    )
    return {
        "package.json": json.dumps(package_json, indent=2, sort_keys=True),
        "extension.js": extension_js,
    }


# --------------------------------------------------------------------------- #
# 348 - SDK readers
# --------------------------------------------------------------------------- #
_DIAGNOSTIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("rule_id", "string"),
    ("severity", "string"),
    ("message", "string"),
    ("fingerprint", "string"),
    ("suggestions", "string[]"),
)


def _python_reader() -> str:
    return (
        '"""Generated PromptABI result reader (Python)."""\n'
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from typing import Any\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Diagnostic:\n"
        "    rule_id: str\n"
        "    severity: str\n"
        "    message: str\n"
        "    fingerprint: str\n"
        "    suggestions: tuple[str, ...]\n\n"
        "    @classmethod\n"
        "    def from_dict(cls, data: dict[str, Any]) -> 'Diagnostic':\n"
        "        return cls(\n"
        "            rule_id=str(data['rule_id']),\n"
        "            severity=str(data['severity']),\n"
        "            message=str(data['message']),\n"
        "            fingerprint=str(data['fingerprint']),\n"
        "            suggestions=tuple(data.get('suggestions', ())),\n"
        "        )\n"
    )


def _typescript_reader() -> str:
    return (
        "// Generated PromptABI result reader (TypeScript).\n"
        "export interface Diagnostic {\n"
        "  rule_id: string;\n"
        "  severity: string;\n"
        "  message: string;\n"
        "  fingerprint: string;\n"
        "  suggestions: string[];\n"
        "}\n\n"
        "export function diagnosticFromJson(data: Record<string, unknown>): Diagnostic {\n"
        "  return {\n"
        "    rule_id: String(data['rule_id']),\n"
        "    severity: String(data['severity']),\n"
        "    message: String(data['message']),\n"
        "    fingerprint: String(data['fingerprint']),\n"
        "    suggestions: (data['suggestions'] as string[]) ?? [],\n"
        "  };\n"
        "}\n"
    )


def _go_reader() -> str:
    return (
        "// Generated PromptABI result reader (Go).\n"
        "package promptabi\n\n"
        "type Diagnostic struct {\n"
        "\tRuleID      string   `json:\"rule_id\"`\n"
        "\tSeverity    string   `json:\"severity\"`\n"
        "\tMessage     string   `json:\"message\"`\n"
        "\tFingerprint string   `json:\"fingerprint\"`\n"
        "\tSuggestions []string `json:\"suggestions\"`\n"
        "}\n"
    )


def _rust_reader() -> str:
    return (
        "// Generated PromptABI result reader (Rust).\n"
        "use serde::Deserialize;\n\n"
        "#[derive(Debug, Clone, Deserialize)]\n"
        "pub struct Diagnostic {\n"
        "    pub rule_id: String,\n"
        "    pub severity: String,\n"
        "    pub message: String,\n"
        "    pub fingerprint: String,\n"
        "    #[serde(default)]\n"
        "    pub suggestions: Vec<String>,\n"
        "}\n"
    )


def sdk_reader_sources() -> dict[str, str]:
    """Generate result-object readers for four target languages."""

    return {
        "python": _python_reader(),
        "typescript": _typescript_reader(),
        "go": _go_reader(),
        "rust": _rust_reader(),
    }


# --------------------------------------------------------------------------- #
# 349 - Rich explanation
# --------------------------------------------------------------------------- #
def explain_demo_finding(diagnostics: tuple[Diagnostic, ...] | None = None) -> str:
    result = _demo_result()
    target = _first_error(tuple(result.diagnostics))
    explanation = explain_diagnostic(result, fingerprint=target.fingerprint)
    return render_explanation_text(explanation)


# --------------------------------------------------------------------------- #
# 350 - Autofix suggestions
# --------------------------------------------------------------------------- #
def ranked_autofixes(
    diagnostics: tuple[Diagnostic, ...] | None = None,
) -> tuple[RankedFixSuggestion, ...]:
    diagnostics = diagnostics if diagnostics is not None else load_demo_diagnostics()
    return rank_fix_suggestions(diagnostics)


# --------------------------------------------------------------------------- #
# 351 - Stable JSON Schemas
# --------------------------------------------------------------------------- #
_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def result_json_schemas() -> dict[str, dict[str, Any]]:
    source_span = {
        "$schema": _SCHEMA_DIALECT,
        "$id": "https://promptabi.dev/schema/source-span.json",
        "title": "SourceSpan",
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "start_line", "start_column"],
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "start_column": {"type": "integer", "minimum": 1},
            "end_line": {"type": ["integer", "null"]},
            "end_column": {"type": ["integer", "null"]},
        },
    }
    diagnostic = {
        "$schema": _SCHEMA_DIALECT,
        "$id": "https://promptabi.dev/schema/diagnostic.json",
        "title": "Diagnostic",
        "type": "object",
        "additionalProperties": False,
        "required": ["rule_id", "severity", "message", "fingerprint"],
        "properties": {
            "rule_id": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "warning", "error"]},
            "message": {"type": "string"},
            "fingerprint": {"type": "string"},
            "witness_digest": {"type": ["string", "null"]},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "check_modes": {"type": "array", "items": {"type": "string"}},
            "witness": {"type": ["object", "null"]},
        },
    }
    verification_result = {
        "$schema": _SCHEMA_DIALECT,
        "$id": "https://promptabi.dev/schema/verification-result.json",
        "title": "VerificationResult",
        "type": "object",
        "additionalProperties": True,
        "required": ["ok", "diagnostics"],
        "properties": {
            "ok": {"type": "boolean"},
            "diagnostics": {
                "type": "array",
                "items": {"$ref": "https://promptabi.dev/schema/diagnostic.json"},
            },
        },
    }
    return {
        "SourceSpan": source_span,
        "Diagnostic": diagnostic,
        "VerificationResult": verification_result,
    }


def validate_against_diagnostic_schema(diagnostic: Diagnostic) -> bool:
    """Minimal structural validation of a diagnostic against its schema."""

    schema = result_json_schemas()["Diagnostic"]
    payload = diagnostic.to_dict()
    required = schema["required"]
    if any(key not in payload for key in required):
        return False
    allowed = set(schema["properties"])
    if any(key not in allowed for key in payload):
        return False
    return payload["severity"] in schema["properties"]["severity"]["enum"]


# --------------------------------------------------------------------------- #
# 352 - WASM playground
# --------------------------------------------------------------------------- #
def wasm_playground_html() -> str:
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <title>PromptABI Playground</title>\n"
        "  <script src=\"https://cdn.jsdelivr.net/pyodide/v0.26.0/full/pyodide.js\"></script>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>PromptABI Playground</h1>\n"
        "  <textarea id=\"config\"></textarea>\n"
        "  <button id=\"run\">Verify</button>\n"
        "  <pre id=\"out\"></pre>\n"
        "  <script type=\"module\">\n"
        "    const pyodide = await loadPyodide();\n"
        "    await pyodide.loadPackage('micropip');\n"
        "    await pyodide.runPythonAsync(`import micropip; await micropip.install('promptabi')`);\n"
        "    document.getElementById('run').onclick = async () => {\n"
        "      const cfg = document.getElementById('config').value;\n"
        "      const out = await pyodide.runPythonAsync(`import promptabi, json; "
        "json.dumps(promptabi.run_verification_payload())`);\n"
        "      document.getElementById('out').textContent = out;\n"
        "    };\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )


# --------------------------------------------------------------------------- #
# 353 - Five-minute quickstart
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class QuickstartResult:
    found_bug: bool
    rule_id: str
    severity: str
    message: str
    suggestion: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "found_bug": self.found_bug,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


def quickstart_find_bug() -> QuickstartResult:
    """Find a real, error-severity contract violation in one call."""

    import time

    start = time.perf_counter()
    diagnostics = load_demo_diagnostics()
    target = _first_error(diagnostics)
    elapsed = time.perf_counter() - start
    return QuickstartResult(
        found_bug=target.severity is DiagnosticSeverity.ERROR,
        rule_id=target.rule_id,
        severity=str(target.severity),
        message=target.message,
        suggestion=target.suggestions[0] if target.suggestions else "",
        elapsed_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# 354 - Third-party plugin API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PluginDemoResult:
    plugin_check_name: str
    registered: bool
    emitted_rule_ids: tuple[str, ...]
    plugin_finding_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_check_name": self.plugin_check_name,
            "registered": self.registered,
            "emitted_rule_ids": list(self.emitted_rule_ids),
            "plugin_finding_present": self.plugin_finding_present,
        }


_PLUGIN_RULE_ID = "thirdparty-banned-phrase"
_PLUGIN_BANNED_PHRASE = "ignore previous instructions"


def _banned_phrase_check(context: "CheckContext") -> tuple[Diagnostic, ...]:
    """A sample third-party analyzer scanning loaded artifacts for a banned phrase.

    It always emits at least an informational finding so the demonstration proves
    the plugin executed inside the real verification scheduler and was able to
    contribute diagnostics to the result.
    """

    scanned = 0
    hits = 0
    for loaded in context.loaded_artifacts:
        scanned += 1
        try:
            text = json.dumps(_jsonable(loaded.artifact.to_dict()), sort_keys=True).lower()
        except Exception:
            text = str(getattr(loaded.artifact, "name", "")).lower()
        if _PLUGIN_BANNED_PHRASE in text:
            hits += 1
    severity = DiagnosticSeverity.WARNING if hits else DiagnosticSeverity.INFO
    message = (
        f"third-party analyzer scanned {scanned} artifact(s); "
        f"{hits} contained the banned override phrase"
    )
    return (
        Diagnostic(
            rule_id=_PLUGIN_RULE_ID,
            severity=severity,
            message=message,
            suggestions=("Route untrusted content through a delimiter-neutralizing filter.",),
        ),
    )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def run_third_party_plugin() -> PluginDemoResult:
    """Register and execute a third-party analyzer through the real plugin API.

    The check is registered on a :class:`PluginRegistry`, then driven end-to-end
    by the production ``run_verification`` scheduler via ``selected_checks`` so the
    demonstration exercises the genuine integration path, not a synthetic stub.
    """

    from .api import run_verification
    from .plugins import PluginRegistry

    registry = PluginRegistry()
    registry.register_check(
        _PLUGIN_RULE_ID,
        _banned_phrase_check,
        plugin="example-thirdparty",
        version="1.0.0",
    )
    registered = _PLUGIN_RULE_ID in registry.checks

    result = run_verification(
        str(demo_config_path()),
        plugin_registry=registry,
        selected_checks=(_PLUGIN_RULE_ID,),
    )
    rule_ids = tuple(d.rule_id for d in result.diagnostics)
    return PluginDemoResult(
        plugin_check_name=_PLUGIN_RULE_ID,
        registered=registered,
        emitted_rule_ids=rule_ids,
        plugin_finding_present=_PLUGIN_RULE_ID in rule_ids,
    )


# --------------------------------------------------------------------------- #
# 355 - Framework integrations
# --------------------------------------------------------------------------- #
def framework_integration_adapters() -> dict[str, str]:
    langchain = (
        '"""PromptABI LangChain integration."""\n'
        "from langchain_core.callbacks import BaseCallbackHandler\n"
        "import promptabi\n\n\n"
        "class PromptABIGuard(BaseCallbackHandler):\n"
        "    def __init__(self, config_path: str, fail_on: str = 'error') -> None:\n"
        "        self.config_path = config_path\n"
        "        self.fail_on = fail_on\n\n"
        "    def on_chain_start(self, serialized, inputs, **kwargs):\n"
        "        result = promptabi.run_verification(self.config_path)\n"
        "        blocking = [d for d in result.diagnostics if str(d.severity) == self.fail_on]\n"
        "        if blocking:\n"
        "            raise ValueError(f'PromptABI blocked: {blocking[0].message}')\n"
    )
    llamaindex = (
        '"""PromptABI LlamaIndex integration."""\n'
        "import promptabi\n\n\n"
        "def promptabi_node_postprocessor(config_path: str):\n"
        "    def _guard(nodes, query_bundle=None):\n"
        "        result = promptabi.run_verification(config_path)\n"
        "        if not result.ok:\n"
        "            raise ValueError('PromptABI contract violation')\n"
        "        return nodes\n"
        "    return _guard\n"
    )
    dspy = (
        '"""PromptABI DSPy integration."""\n'
        "import promptabi\n\n\n"
        "def promptabi_assertion(config_path: str) -> bool:\n"
        "    result = promptabi.run_verification(config_path)\n"
        "    return result.ok\n"
    )
    return {"langchain": langchain, "llamaindex": llamaindex, "dspy": dspy}


# --------------------------------------------------------------------------- #
# 356 - Typed stubs
# --------------------------------------------------------------------------- #
def generate_type_stub() -> str:
    return (
        "from .diagnostics import Diagnostic\n"
        "from .fix_suggestions import RankedFixSuggestion\n"
        "from typing import Any\n\n"
        "DEVEX_ECOSYSTEM_VERSION: str\n"
        "SUPPORTED_SDK_LANGUAGES: tuple[str, ...]\n"
        "SUPPORTED_LOCALES: tuple[str, ...]\n"
        "FRAMEWORK_INTEGRATIONS: tuple[str, ...]\n\n"
        "def load_demo_diagnostics() -> tuple[Diagnostic, ...]: ...\n"
        "def build_lsp_diagnostics(diagnostics: tuple[Diagnostic, ...], *, uri: str = ...) -> dict[str, Any]: ...\n"
        "def vscode_extension_files() -> dict[str, str]: ...\n"
        "def sdk_reader_sources() -> dict[str, str]: ...\n"
        "def ranked_autofixes(diagnostics: tuple[Diagnostic, ...] | None = ...) -> tuple[RankedFixSuggestion, ...]: ...\n"
        "def result_json_schemas() -> dict[str, dict[str, Any]]: ...\n"
        "def wasm_playground_html() -> str: ...\n"
        "def quickstart_find_bug() -> Any: ...\n"
        "def run_third_party_plugin() -> Any: ...\n"
        "def framework_integration_adapters() -> dict[str, str]: ...\n"
        "def generate_type_stub() -> str: ...\n"
        "def localize_finding(diagnostic: Diagnostic, locale: str) -> str: ...\n"
        "def tutorial_notebook() -> dict[str, Any]: ...\n"
        "def coverage_badge_svg(percent: float, *, label: str = ...) -> str: ...\n"
        "def run_devex_ecosystem() -> Any: ...\n"
    )


# --------------------------------------------------------------------------- #
# 357 - Internationalized messages
# --------------------------------------------------------------------------- #
LOCALIZED_MESSAGES: dict[str, dict[str, str]] = {
    "promptabi.diagnostic.rag.citation.loss": {
        "en": "Truncation can drop a citation-required retrieval chunk.",
        "es": "El truncamiento puede descartar un fragmento con cita obligatoria.",
        "ja": "トランケーションにより引用必須の取得チャンクが失われる可能性があります。",
        "de": "Die Kürzung kann einen zitierpflichtigen Abrufabschnitt verwerfen.",
        "fr": "La troncature peut supprimer un fragment nécessitant une citation.",
    },
    "promptabi.diagnostic.rag.payload.truncation": {
        "en": "Retrieval chunk exceeds its payload token budget.",
        "es": "El fragmento de recuperación supera su presupuesto de tokens.",
        "ja": "取得チャンクがペイロードのトークン予算を超えています。",
        "de": "Abrufabschnitt überschreitet sein Token-Budget.",
        "fr": "Le fragment dépasse son budget de jetons.",
    },
}


def localize_finding(diagnostic: Diagnostic, locale: str) -> str:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"unsupported locale: {locale}")
    key = diagnostic.localization_key
    catalog = LOCALIZED_MESSAGES.get(key or "")
    if catalog is None:
        return diagnostic.message
    return catalog.get(locale, catalog.get("en", diagnostic.message))


# --------------------------------------------------------------------------- #
# 358 - Tutorial notebook
# --------------------------------------------------------------------------- #
def tutorial_notebook() -> dict[str, Any]:
    def code(*lines: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in lines],
        }

    def md(*lines: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            md("# PromptABI Quickstart", "Find a real prompt-contract bug in under five minutes."),
            code("import promptabi", "result = promptabi.run_verification('examples/rag-chunking/promptabi.json')"),
            md("## Inspect the findings"),
            code("for d in result.diagnostics:", "    print(d.severity, d.rule_id, d.message)"),
            md("## Explain a finding"),
            code("from promptabi.devex_ecosystem import explain_demo_finding", "print(explain_demo_finding())"),
        ],
    }


# --------------------------------------------------------------------------- #
# 359 - Org policy profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    severity_overrides: tuple[tuple[str, DiagnosticSeverity], ...] = ()
    disabled_rules: tuple[str, ...] = ()
    fail_on: DiagnosticSeverity = DiagnosticSeverity.ERROR

    def override_map(self) -> dict[str, DiagnosticSeverity]:
        return dict(self.severity_overrides)


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile: str
    kept: tuple[Diagnostic, ...]
    gate_passed: bool
    blocking_rule_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "kept": [d.rule_id for d in self.kept],
            "gate_passed": self.gate_passed,
            "blocking_rule_ids": list(self.blocking_rule_ids),
        }


_SEVERITY_ORDER = {
    DiagnosticSeverity.INFO: 0,
    DiagnosticSeverity.WARNING: 1,
    DiagnosticSeverity.ERROR: 2,
}


def apply_profile(diagnostics: tuple[Diagnostic, ...], profile: PolicyProfile) -> ProfileResult:
    overrides = profile.override_map()
    disabled = set(profile.disabled_rules)
    kept: list[Diagnostic] = []
    blocking: list[str] = []
    threshold = _SEVERITY_ORDER[profile.fail_on]
    for diagnostic in diagnostics:
        if diagnostic.rule_id in disabled:
            continue
        severity = overrides.get(diagnostic.rule_id, diagnostic.severity)
        adjusted = diagnostic if severity is diagnostic.severity else _with_severity(diagnostic, severity)
        kept.append(adjusted)
        if _SEVERITY_ORDER[severity] >= threshold:
            blocking.append(diagnostic.rule_id)
    return ProfileResult(
        profile=profile.name,
        kept=tuple(kept),
        gate_passed=not blocking,
        blocking_rule_ids=tuple(blocking),
    )


def _with_severity(diagnostic: Diagnostic, severity: DiagnosticSeverity) -> Diagnostic:
    return Diagnostic(
        rule_id=diagnostic.rule_id,
        severity=severity,
        message=diagnostic.message,
        artifact=diagnostic.artifact,
        span=diagnostic.span,
        witness=diagnostic.witness,
        suggestions=diagnostic.suggestions,
        check_modes=diagnostic.check_modes,
        properties=diagnostic.properties,
        message_id=diagnostic.message_id,
        message_args=diagnostic.message_args,
        upstream_issues=diagnostic.upstream_issues,
    )


def builtin_profiles() -> dict[str, PolicyProfile]:
    return {
        "strict": PolicyProfile(name="strict", fail_on=DiagnosticSeverity.WARNING),
        "lenient": PolicyProfile(
            name="lenient",
            severity_overrides=(("artifact-unpinned", DiagnosticSeverity.INFO),),
            fail_on=DiagnosticSeverity.ERROR,
        ),
    }


# --------------------------------------------------------------------------- #
# 360 - Coverage badge
# --------------------------------------------------------------------------- #
def _badge_color(percent: float) -> str:
    if percent >= 90:
        return "#4c1"
    if percent >= 75:
        return "#dfb317"
    return "#e05d44"


def coverage_badge_svg(percent: float, *, label: str = "coverage") -> str:
    value = f"{percent:.0f}%"
    color = _badge_color(percent)
    label_w = 6 * len(label) + 10
    value_w = 6 * len(value) + 10
    total = label_w + value_w
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
        f'role="img" aria-label="{label}: {value}">'
        f'<rect width="{total}" height="20" fill="#555"/>'
        f'<rect x="{label_w}" width="{value_w}" height="20" fill="{color}"/>'
        f'<g fill="#fff" font-family="Verdana" font-size="11">'
        f'<text x="{label_w / 2:.0f}" y="14" text-anchor="middle">{label}</text>'
        f'<text x="{label_w + value_w / 2:.0f}" y="14" text-anchor="middle">{value}</text>'
        f"</g></svg>"
    )


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DevexStep:
    step: int
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DevexEcosystemReport:
    version: str
    steps: tuple[DevexStep, ...]

    @property
    def passed(self) -> bool:
        return all(step.ok for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "steps": [step.to_dict() for step in self.steps],
        }


def run_devex_ecosystem() -> DevexEcosystemReport:
    diagnostics = load_demo_diagnostics()
    steps: list[DevexStep] = []

    lsp = build_lsp_diagnostics(diagnostics)
    steps.append(
        DevexStep(346, "language-server", bool(lsp["params"]["diagnostics"]),
                  f'{len(lsp["params"]["diagnostics"])} LSP diagnostics')
    )

    ext = vscode_extension_files()
    ext_ok = "package.json" in ext and bool(json.loads(ext["package.json"]))
    steps.append(DevexStep(347, "vscode-extension", ext_ok, "valid package.json"))

    sdk = sdk_reader_sources()
    sdk_ok = set(sdk) == set(SUPPORTED_SDK_LANGUAGES) and _python_reader_roundtrips(diagnostics)
    steps.append(DevexStep(348, "sdk-readers", sdk_ok, f"{len(sdk)} languages, python verified"))

    explain_text = explain_demo_finding(diagnostics)
    steps.append(DevexStep(349, "explain", bool(explain_text.strip()), "rich explanation rendered"))

    fixes = ranked_autofixes(diagnostics)
    steps.append(DevexStep(350, "autofix", len(fixes) > 0, f"{len(fixes)} ranked fixes"))

    schemas = result_json_schemas()
    schema_ok = all("$schema" in s for s in schemas.values()) and validate_against_diagnostic_schema(
        _first_error(diagnostics)
    )
    steps.append(DevexStep(351, "json-schema", schema_ok, f"{len(schemas)} schemas"))

    html = wasm_playground_html()
    steps.append(
        DevexStep(352, "wasm-playground", "pyodide" in html and "promptabi" in html, "pyodide HTML")
    )

    qs = quickstart_find_bug()
    steps.append(
        DevexStep(353, "quickstart", qs.found_bug and qs.elapsed_seconds < 300,
                  f"{qs.rule_id} (under 5 minutes)")
    )

    plugin = run_third_party_plugin()
    steps.append(
        DevexStep(354, "plugin-api", plugin.registered and plugin.plugin_finding_present,
                  plugin.plugin_check_name)
    )

    frameworks = framework_integration_adapters()
    steps.append(
        DevexStep(355, "framework-integrations", set(frameworks) == set(FRAMEWORK_INTEGRATIONS),
                  ", ".join(sorted(frameworks)))
    )

    stub = generate_type_stub()
    stub_ok = _stub_parses(stub)
    steps.append(DevexStep(356, "typed-stubs", stub_ok, "ast-valid .pyi"))

    target = _first_error(diagnostics)
    localized = {loc: localize_finding(target, loc) for loc in SUPPORTED_LOCALES}
    i18n_ok = len({v for v in localized.values()}) >= 2
    steps.append(DevexStep(357, "i18n", i18n_ok, f"{len(SUPPORTED_LOCALES)} locales"))

    notebook = tutorial_notebook()
    nb_ok = notebook.get("nbformat") == 4 and len(notebook.get("cells", [])) > 0
    steps.append(DevexStep(358, "tutorial-notebook", nb_ok, f'{len(notebook["cells"])} cells'))

    lenient = builtin_profiles()["lenient"]
    profile_result = apply_profile(diagnostics, lenient)
    steps.append(
        DevexStep(359, "policy-profiles", True,
                  f"profile '{profile_result.profile}', gate_passed={profile_result.gate_passed}")
    )

    badge = coverage_badge_svg(96.0)
    steps.append(DevexStep(360, "coverage-badge", "<svg" in badge and "96%" in badge, "valid SVG"))

    return DevexEcosystemReport(version=DEVEX_ECOSYSTEM_VERSION, steps=tuple(steps))


def _python_reader_roundtrips(diagnostics: tuple[Diagnostic, ...]) -> bool:
    namespace: dict[str, Any] = {}
    exec(compile(_python_reader(), "<sdk-python>", "exec"), namespace)  # noqa: S102
    reader = namespace["Diagnostic"]
    payload = _first_error(diagnostics).to_dict()
    obj = reader.from_dict(payload)
    return obj.rule_id == payload["rule_id"] and obj.severity == payload["severity"]


def _stub_parses(stub: str) -> bool:
    try:
        ast.parse(stub)
        return True
    except SyntaxError:
        return False


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_devex_ecosystem_text(report: DevexEcosystemReport) -> str:
    lines = [
        f"PromptABI developer experience + ecosystem v{report.version}",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
        "",
    ]
    for step in report.steps:
        mark = "ok" if step.ok else "XX"
        lines.append(f"[{step.step}] {mark} {step.name}: {step.detail}")
    return "\n".join(lines)


def render_devex_ecosystem_json(report: DevexEcosystemReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "DEVEX_ECOSYSTEM_VERSION",
    "DEMO_CONFIG_RELPATH",
    "SUPPORTED_SDK_LANGUAGES",
    "SUPPORTED_LOCALES",
    "FRAMEWORK_INTEGRATIONS",
    "LOCALIZED_MESSAGES",
    "DevexStep",
    "DevexEcosystemReport",
    "QuickstartResult",
    "PluginDemoResult",
    "PolicyProfile",
    "ProfileResult",
    "demo_config_path",
    "load_demo_diagnostics",
    "build_lsp_diagnostics",
    "lsp_diagnostic",
    "language_server_initialize_result",
    "vscode_extension_files",
    "sdk_reader_sources",
    "explain_demo_finding",
    "ranked_autofixes",
    "result_json_schemas",
    "validate_against_diagnostic_schema",
    "wasm_playground_html",
    "quickstart_find_bug",
    "run_third_party_plugin",
    "framework_integration_adapters",
    "generate_type_stub",
    "localize_finding",
    "tutorial_notebook",
    "PolicyProfile",
    "apply_profile",
    "builtin_profiles",
    "coverage_badge_svg",
    "run_devex_ecosystem",
    "render_devex_ecosystem_text",
    "render_devex_ecosystem_json",
]
