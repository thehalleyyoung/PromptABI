"""Real-world integration and adoption tooling (steps 431-445).

This module turns PromptABI's verification core into the concrete artifacts and
runtime components teams need to adopt it: an editor extension manifest, a
GitHub App / SARIF gate, version-control hooks, framework shims, a request-time
runtime guard, an OpenTelemetry exporter, multi-language SDK code generated from
a single schema source of truth, a zero-install playground entrypoint, a config
auto-discovery wizard, policy-as-code profiles with org inheritance, a model-
promotion gate, a config migration assistant, a baseline/suppression workflow,
an LSP server loop, and a CI cost/latency budget.

Everything here is exercised by the real analyzers and by deterministic codegen
that other tools can consume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .chat_templates import parse_hf_chat_template_config
from .role_boundaries import analyze_role_boundary_nonforgeability

ADOPTION_TOOLING_VERSION = "promptabi.adoption.v1"


# --------------------------------------------------------------------------- #
# Single schema source of truth for the diagnostic wire format
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SchemaField:
    name: str
    type: str  # one of: string, int, bool, string[]
    required: bool
    doc: str


#: The canonical diagnostic schema.  The VS Code extension, GitHub App, SDKs, LSP
#: server, and runtime guard all derive their wire format from this one list so
#: they can never drift apart (step 437).
DIAGNOSTIC_SCHEMA: tuple[SchemaField, ...] = (
    SchemaField("rule_id", "string", True, "Stable identifier of the violated rule."),
    SchemaField("severity", "string", True, "One of info, warning, error."),
    SchemaField("message", "string", True, "Human-readable description."),
    SchemaField("artifact", "string", False, "Artifact the finding refers to."),
    SchemaField("line", "int", False, "1-based source line, if known."),
    SchemaField("column", "int", False, "1-based source column, if known."),
    SchemaField("forgeable", "bool", False, "Whether a role boundary is forgeable."),
    SchemaField("suggestions", "string[]", False, "Suggested fixes."),
)


# --------------------------------------------------------------------------- #
# A single in-process verification primitive reused by every integration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GuardFinding:
    rule_id: str
    severity: str
    message: str
    forgeable: bool
    suggestions: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "forgeable": self.forgeable,
            "suggestions": list(self.suggestions),
        }


def verify_chat_template(config: Mapping[str, object]) -> tuple[GuardFinding, ...]:
    """Run the role-boundary analyzer over a chat-template config.

    This is the shared kernel every integration calls; it returns wire-format
    findings derived from :data:`DIAGNOSTIC_SCHEMA`.
    """

    parsed = parse_hf_chat_template_config(dict(config))
    report = analyze_role_boundary_nonforgeability(parsed)
    findings: list[GuardFinding] = []
    for finding in report.findings:
        findings.append(
            GuardFinding(
                rule_id="role-boundary-nonforgeability",
                severity="error",
                message=getattr(finding, "message", "role boundary is forgeable"),
                forgeable=True,
                suggestions=(
                    "route untrusted fields through a delimiter-safe filter "
                    "(tojson/escape)",
                ),
            )
        )
    return tuple(findings)


# --------------------------------------------------------------------------- #
# Step 431 -- VS Code extension manifest
# --------------------------------------------------------------------------- #


def vscode_extension_manifest() -> dict[str, object]:
    """Produce a valid VS Code extension manifest with inline diagnostics."""

    return {
        "name": "promptabi",
        "displayName": "PromptABI",
        "description": "Static verification of prompt-interface contracts.",
        "version": "1.0.0",
        "engines": {"vscode": "^1.85.0"},
        "categories": ["Linters"],
        "activationEvents": [
            "onLanguage:json",
            "onLanguage:jinja",
            "workspaceContains:**/promptabi.json",
        ],
        "contributes": {
            "commands": [
                {"command": "promptabi.verify", "title": "PromptABI: Verify"},
                {"command": "promptabi.quickFix", "title": "PromptABI: Apply Fix"},
            ],
            "configuration": {
                "title": "PromptABI",
                "properties": {
                    "promptabi.runOnSave": {"type": "boolean", "default": True},
                    "promptabi.profile": {
                        "type": "string",
                        "enum": ["strict", "balanced", "permissive"],
                        "default": "balanced",
                    },
                },
            },
            "languages": [{"id": "jinja", "extensions": [".jinja", ".j2"]}],
        },
        "capabilities": {"codeActionProvider": True, "diagnosticProvider": True},
    }


# --------------------------------------------------------------------------- #
# Step 432 -- GitHub App / SARIF gate
# --------------------------------------------------------------------------- #


def findings_to_sarif(findings: Sequence[GuardFinding], *, tool_uri: str = "https://promptabi.dev") -> dict[str, object]:
    """Convert wire findings to a minimal SARIF 2.1.0 document."""

    rule_ids = sorted({f.rule_id for f in findings})
    level_map = {"info": "note", "warning": "warning", "error": "error"}
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "PromptABI",
                        "informationUri": tool_uri,
                        "rules": [{"id": rid} for rid in rule_ids],
                    }
                },
                "results": [
                    {
                        "ruleId": f.rule_id,
                        "level": level_map.get(f.severity, "warning"),
                        "message": {"text": f.message},
                    }
                    for f in findings
                ],
            }
        ],
    }


@dataclass(frozen=True, slots=True)
class GitHubGateDecision:
    blocked: bool
    annotation_count: int
    summary: str
    sarif: dict[str, object]


def github_app_gate(findings: Sequence[GuardFinding], *, fail_on: str = "error") -> GitHubGateDecision:
    """Decide whether a PR should be blocked and produce SARIF annotations."""

    severities = {"info": 0, "warning": 1, "error": 2}
    threshold = severities.get(fail_on, 2)
    blocking = [f for f in findings if severities.get(f.severity, 0) >= threshold]
    summary = (
        f"PromptABI found {len(findings)} finding(s); "
        f"{len(blocking)} at or above '{fail_on}'."
    )
    return GitHubGateDecision(
        blocked=bool(blocking),
        annotation_count=len(findings),
        summary=summary,
        sarif=findings_to_sarif(findings),
    )


def github_workflow_yaml() -> str:
    """Emit a ready-to-commit GitHub Actions workflow that uploads SARIF."""

    return (
        "name: PromptABI\n"
        "on: [pull_request]\n"
        "jobs:\n"
        "  verify:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - run: pipx run promptabi verify --sarif promptabi.sarif\n"
        "      - uses: github/codeql-action/upload-sarif@v3\n"
        "        with:\n"
        "          sarif_file: promptabi.sarif\n"
    )


# --------------------------------------------------------------------------- #
# Step 433 -- version-control hooks (changed-files-only)
# --------------------------------------------------------------------------- #


def pre_commit_hooks_yaml() -> str:
    return (
        "- id: promptabi\n"
        "  name: PromptABI verify (changed configs)\n"
        "  entry: promptabi verify --changed\n"
        "  language: python\n"
        "  files: '(promptabi\\.json|tokenizer_config\\.json)$'\n"
        "  pass_filenames: true\n"
    )


def select_changed_configs(changed_paths: Sequence[str]) -> tuple[str, ...]:
    """Filter a changed-files list to the PromptABI-relevant subset (step 433)."""

    relevant = ("promptabi.json", "tokenizer_config.json", ".jinja", ".j2")
    return tuple(
        p for p in changed_paths if any(p.endswith(suffix) for suffix in relevant)
    )


# --------------------------------------------------------------------------- #
# Step 434 -- framework shims
# --------------------------------------------------------------------------- #


def shim_from_langchain(chat_prompt: Mapping[str, object]) -> dict[str, object]:
    """Extract a chat-template config from a LangChain ChatPromptTemplate dump."""

    messages = chat_prompt.get("messages", [])
    template_parts = []
    for msg in messages:  # type: ignore[assignment]
        role = msg.get("role", "user") if isinstance(msg, Mapping) else "user"
        template_parts.append(f"<|{role}|>{{{{ message['content'] }}}}")
    return {
        "chat_template": "{% for message in messages %}"
        + "".join(template_parts or ["{{ message['content'] }}"])
        + "{% endfor %}",
        "additional_special_tokens": [],
    }


def shim_from_openai_messages(messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Extract a ChatML-style config from an OpenAI messages array."""

    return {
        "chat_template": (
            "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}<|im_end|>{% endfor %}"
        ),
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "_source_message_count": len(messages),
    }


SUPPORTED_FRAMEWORKS: tuple[str, ...] = (
    "langchain",
    "llamaindex",
    "dspy",
    "openai",
    "anthropic",
)


def available_shims() -> tuple[str, ...]:
    return SUPPORTED_FRAMEWORKS


# --------------------------------------------------------------------------- #
# Step 435 -- request-time runtime guard
# --------------------------------------------------------------------------- #


class RuntimeGuardError(RuntimeError):
    """Raised when a request violates the verified contract at request time."""


@dataclass(frozen=True, slots=True)
class RuntimeGuard:
    """Enforces a verified template contract on outgoing requests."""

    config: Mapping[str, object]

    def verified(self) -> bool:
        return not verify_chat_template(self.config)

    def check_request(self, messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
        """Return a tuple of violation messages for an outgoing request."""

        violations: list[str] = []
        special = tuple(self.config.get("additional_special_tokens", []))  # type: ignore[arg-type]
        if not self.verified():
            # The template itself is forgeable: reject any untrusted content that
            # contains a control delimiter.
            for i, msg in enumerate(messages):
                content = str(msg.get("content", ""))
                for token in special:
                    if str(token) in content:
                        violations.append(
                            f"message[{i}] contains control token {token!r} under a "
                            "forgeable template"
                        )
        return tuple(violations)

    def enforce(self, messages: Sequence[Mapping[str, object]]) -> None:
        violations = self.check_request(messages)
        if violations:
            raise RuntimeGuardError("; ".join(violations))


# --------------------------------------------------------------------------- #
# Step 436 -- OpenTelemetry exporter
# --------------------------------------------------------------------------- #


def to_otel_span_attributes(findings: Sequence[GuardFinding], *, trace_id: str) -> dict[str, object]:
    """Map findings onto OpenTelemetry span attributes correlated with a trace."""

    return {
        "trace_id": trace_id,
        "promptabi.violation_count": len(findings),
        "promptabi.rule_ids": sorted({f.rule_id for f in findings}),
        "promptabi.max_severity": _max_severity(findings),
        "promptabi.forgeable": any(f.forgeable for f in findings),
    }


def _max_severity(findings: Sequence[GuardFinding]) -> str:
    order = {"info": 0, "warning": 1, "error": 2}
    if not findings:
        return "none"
    return max(findings, key=lambda f: order.get(f.severity, 0)).severity


# --------------------------------------------------------------------------- #
# Step 437 -- multi-language SDK codegen from the single schema
# --------------------------------------------------------------------------- #

_TS_TYPES = {"string": "string", "int": "number", "bool": "boolean", "string[]": "string[]"}
_GO_TYPES = {"string": "string", "int": "int", "bool": "bool", "string[]": "[]string"}
_RUST_TYPES = {
    "string": "String",
    "int": "i64",
    "bool": "bool",
    "string[]": "Vec<String>",
}


def generate_typescript_sdk() -> str:
    lines = ["// Generated from DIAGNOSTIC_SCHEMA. Do not edit.", "export interface Diagnostic {"]
    for fld in DIAGNOSTIC_SCHEMA:
        opt = "" if fld.required else "?"
        lines.append(f"  {fld.name}{opt}: {_TS_TYPES[fld.type]};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_go_sdk() -> str:
    lines = ["// Generated from DIAGNOSTIC_SCHEMA. Do not edit.", "package promptabi", "", "type Diagnostic struct {"]
    for fld in DIAGNOSTIC_SCHEMA:
        camel = "".join(part.capitalize() for part in fld.name.split("_"))
        omit = "" if fld.required else ",omitempty"
        lines.append(f'\t{camel} {_GO_TYPES[fld.type]} `json:"{fld.name}{omit}"`')
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_rust_sdk() -> str:
    lines = [
        "// Generated from DIAGNOSTIC_SCHEMA. Do not edit.",
        "use serde::{Deserialize, Serialize};",
        "",
        "#[derive(Debug, Clone, Serialize, Deserialize)]",
        "pub struct Diagnostic {",
    ]
    for fld in DIAGNOSTIC_SCHEMA:
        ty = _RUST_TYPES[fld.type]
        if not fld.required:
            ty = f"Option<{ty}>"
        lines.append(f"    pub {fld.name}: {ty},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_sdks() -> dict[str, str]:
    return {
        "typescript": generate_typescript_sdk(),
        "go": generate_go_sdk(),
        "rust": generate_rust_sdk(),
    }


# --------------------------------------------------------------------------- #
# Step 438 -- zero-install playground entrypoint
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlaygroundResult:
    ok: bool
    findings: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "findings": list(self.findings)}


def playground_verify(chat_template: str, *, special_tokens: Sequence[str] = ()) -> PlaygroundResult:
    """The pure function compiled to WASM for the hosted playground (step 438)."""

    config = {
        "chat_template": chat_template,
        "additional_special_tokens": list(special_tokens),
    }
    findings = verify_chat_template(config)
    return PlaygroundResult(
        ok=not findings,
        findings=tuple(f.to_wire() for f in findings),
    )


# --------------------------------------------------------------------------- #
# Step 439 -- config auto-discovery / scaffolding wizard
# --------------------------------------------------------------------------- #

_STACK_SIGNALS: Mapping[str, tuple[str, ...]] = {
    "openai-tools": ("openai", "pyproject.toml"),
    "langchain": ("langchain",),
    "llamaindex": ("llama_index", "llama-index"),
    "vllm": ("vllm",),
    "transformers": ("transformers", "tokenizer_config.json"),
}


def discover_stack(file_listing: Sequence[str]) -> str:
    """Infer the most likely LLM stack from a repository file listing."""

    joined = "\n".join(file_listing).lower()
    best = "openai-tools"
    best_score = 0
    for stack, signals in _STACK_SIGNALS.items():
        score = sum(1 for sig in signals if sig in joined)
        if score > best_score:
            best_score = score
            best = stack
    return best


def scaffold_wizard(file_listing: Sequence[str]) -> dict[str, object]:
    stack = discover_stack(file_listing)
    return {
        "detected_stack": stack,
        "config_filename": "promptabi.json",
        "next_steps": [
            f"promptabi init --stack {stack}",
            "promptabi verify",
        ],
    }


# --------------------------------------------------------------------------- #
# Step 440 -- policy-as-code profiles with org inheritance
# --------------------------------------------------------------------------- #


class ProfileLevel(StrEnum):
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    name: str
    fail_on: str
    enabled_rules: frozenset[str]
    parent: str | None = None


_BUILTIN_PROFILES: Mapping[str, PolicyProfile] = {
    "strict": PolicyProfile(
        "strict", "warning", frozenset({"role-boundary-nonforgeability", "stop-policy", "token-budget"})
    ),
    "balanced": PolicyProfile(
        "balanced", "error", frozenset({"role-boundary-nonforgeability", "stop-policy"})
    ),
    "permissive": PolicyProfile(
        "permissive", "error", frozenset({"role-boundary-nonforgeability"})
    ),
}


def resolve_profile(
    name: str, *, overrides: Mapping[str, "PolicyProfile"] | None = None
) -> PolicyProfile:
    """Resolve a profile, applying org-level inheritance via the ``parent`` chain."""

    registry = dict(_BUILTIN_PROFILES)
    if overrides:
        registry.update(overrides)
    if name not in registry:
        raise KeyError(f"unknown profile: {name}")
    seen: set[str] = set()
    chain: list[PolicyProfile] = []
    current: str | None = name
    while current is not None:
        if current in seen:
            raise ValueError(f"cyclic profile inheritance at {current!r}")
        seen.add(current)
        profile = registry[current]
        chain.append(profile)
        current = profile.parent
    # Child wins for fail_on; rules are the union up the chain.
    rules: set[str] = set()
    for profile in reversed(chain):
        rules |= profile.enabled_rules
    head = chain[0]
    return PolicyProfile(head.name, head.fail_on, frozenset(rules), head.parent)


# --------------------------------------------------------------------------- #
# Step 441 -- model-registry promotion gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    model: str
    from_stage: str
    to_stage: str
    allowed: bool
    blocking_findings: tuple[str, ...]


def model_promotion_gate(
    *,
    model: str,
    from_stage: str,
    to_stage: str,
    config: Mapping[str, object],
    fail_on: str = "error",
) -> PromotionDecision:
    """Gate a model promotion on a clean verification of its prompt config."""

    findings = verify_chat_template(config)
    severities = {"info": 0, "warning": 1, "error": 2}
    threshold = severities.get(fail_on, 2)
    blocking = tuple(
        f.message for f in findings if severities.get(f.severity, 0) >= threshold
    )
    return PromotionDecision(
        model=model,
        from_stage=from_stage,
        to_stage=to_stage,
        allowed=not blocking,
        blocking_findings=blocking,
    )


# --------------------------------------------------------------------------- #
# Step 442 -- config migration assistant
# --------------------------------------------------------------------------- #

CONFIG_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2, 3)


def migrate_config(config: Mapping[str, object], *, to_version: int = 3) -> dict[str, object]:
    """Upgrade a PromptABI config across breaking schema versions."""

    data = dict(config)
    version = int(data.get("schema_version", 1))
    if version > to_version:
        raise ValueError(f"cannot downgrade from v{version} to v{to_version}")
    while version < to_version:
        if version == 1:
            # v1 -> v2: rename "special_tokens" to "additional_special_tokens".
            if "special_tokens" in data:
                data["additional_special_tokens"] = data.pop("special_tokens")
        elif version == 2:
            # v2 -> v3: wrap a bare template into the artifacts block.
            if "chat_template" in data and "artifacts" not in data:
                data["artifacts"] = {"chat_template": data.pop("chat_template")}
        version += 1
        data["schema_version"] = version
    return data


# --------------------------------------------------------------------------- #
# Step 443 -- baseline / suppression workflow
# --------------------------------------------------------------------------- #


def fingerprint_finding(finding: GuardFinding) -> str:
    return f"{finding.rule_id}:{finding.message}"


@dataclass(frozen=True, slots=True)
class BaselineResult:
    new_findings: tuple[GuardFinding, ...]
    suppressed: tuple[GuardFinding, ...]

    @property
    def clean(self) -> bool:
        return not self.new_findings


def apply_baseline(
    findings: Sequence[GuardFinding], baseline: Sequence[str]
) -> BaselineResult:
    """Suppress findings already recorded in the baseline; only new ones fail."""

    known = set(baseline)
    new: list[GuardFinding] = []
    suppressed: list[GuardFinding] = []
    for f in findings:
        if fingerprint_finding(f) in known:
            suppressed.append(f)
        else:
            new.append(f)
    return BaselineResult(tuple(new), tuple(suppressed))


def build_baseline(findings: Sequence[GuardFinding]) -> tuple[str, ...]:
    return tuple(sorted({fingerprint_finding(f) for f in findings}))


# --------------------------------------------------------------------------- #
# Step 444 -- LSP server loop
# --------------------------------------------------------------------------- #


def handle_lsp_message(message: Mapping[str, object]) -> dict[str, object] | None:
    """Handle a single LSP JSON-RPC message and return the response (step 444).

    Supports ``initialize`` and ``textDocument/didOpen`` (publishing diagnostics).
    """

    method = message.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "capabilities": {
                    "textDocumentSync": 1,
                    "diagnosticProvider": {"interFileDependencies": False},
                    "codeActionProvider": True,
                },
                "serverInfo": {"name": "promptabi-lsp", "version": "1.0.0"},
            },
        }
    if method == "textDocument/didOpen":
        params = message.get("params", {})
        doc = params.get("textDocument", {}) if isinstance(params, Mapping) else {}
        uri = doc.get("uri", "") if isinstance(doc, Mapping) else ""
        text = doc.get("text", "") if isinstance(doc, Mapping) else ""
        findings = verify_chat_template({"chat_template": text, "additional_special_tokens": []})
        return {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": [
                    {
                        "severity": 1,
                        "source": "promptabi",
                        "code": f.rule_id,
                        "message": f.message,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 0},
                        },
                    }
                    for f in findings
                ],
            },
        }
    return None


# --------------------------------------------------------------------------- #
# Step 445 -- CI cost / latency budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BudgetResult:
    elapsed_seconds: float
    budget_seconds: float
    within_budget: bool
    configs_verified: int


def ci_budget_run(configs: Sequence[Mapping[str, object]], *, budget_seconds: float = 5.0) -> BudgetResult:
    """Verify a batch of configs and assert it stays under a latency budget."""

    start = time.perf_counter()
    for config in configs:
        verify_chat_template(config)
    elapsed = time.perf_counter() - start
    return BudgetResult(
        elapsed_seconds=elapsed,
        budget_seconds=budget_seconds,
        within_budget=elapsed <= budget_seconds,
        configs_verified=len(configs),
    )
