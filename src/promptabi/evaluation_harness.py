"""Static evaluation-harness contract checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ArtifactKind,
    EvaluationHarnessArtifact,
    ProviderConfigArtifact,
    SchemaArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    artifact_from_config,
)
from .diagnostics import SourceSpan
from .source import JsonSourceMap


class EvaluationHarnessError(ValueError):
    """Raised when an evaluation harness manifest has an unsupported shape."""


@dataclass(frozen=True, slots=True)
class EvaluationHarnessFinding:
    """One finite evaluation-harness contract finding."""

    rule_id: str
    severity: str
    message: str
    suggestion: str
    subject: str | None = None
    expected: str | None = None
    actual: str | None = None
    span: SourceSpan | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationHarnessReport:
    """Result of checking one harness against surrounding PromptABI artifacts."""

    harness_name: str
    benchmark_name: str
    findings: tuple[EvaluationHarnessFinding, ...]

    @property
    def verified(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


def parse_evaluation_harness_mapping(
    name: str,
    raw: Mapping[str, Any],
    *,
    path: Path,
    source_map: JsonSourceMap | None = None,
) -> EvaluationHarnessArtifact:
    """Parse a local evaluation harness manifest into the typed artifact model."""

    if not isinstance(raw, Mapping):
        raise EvaluationHarnessError("evaluation harness root must be a JSON object")
    try:
        return artifact_from_config(
            name,
            {**dict(raw), "kind": ArtifactKind.EVALUATION_HARNESS.value, "path": str(path)},
            base_dir=Path("."),
            source_span=source_map.span_for(()) if source_map is not None else None,
        )
    except ValueError as exc:
        raise EvaluationHarnessError(str(exc)) from exc


def analyze_evaluation_harness_contracts(
    harness: EvaluationHarnessArtifact,
    artifacts: Sequence[Artifact],
    *,
    source_spans: Mapping[str, SourceSpan] | None = None,
) -> EvaluationHarnessReport:
    """Check benchmark prompts, few-shots, parsers, and stops against contracts."""

    spans = dict(source_spans or {})
    findings: list[EvaluationHarnessFinding] = []
    providers = tuple(artifact for artifact in artifacts if isinstance(artifact, ProviderConfigArtifact))
    tokenizers = tuple(artifact for artifact in artifacts if isinstance(artifact, TokenizerArtifact))
    stop_policies = tuple(artifact for artifact in artifacts if isinstance(artifact, StopPolicyArtifact))
    schemas = tuple(artifact for artifact in artifacts if isinstance(artifact, SchemaArtifact))
    templates = tuple(artifact for artifact in artifacts if artifact.kind is ArtifactKind.CHAT_TEMPLATE)

    findings.extend(_provider_findings(harness, providers, spans))
    findings.extend(_tokenizer_findings(harness, tokenizers, spans))
    findings.extend(_template_findings(harness, templates, spans))
    findings.extend(_stop_findings(harness, stop_policies, spans))
    findings.extend(_parser_findings(harness, schemas, spans))
    findings.extend(_prompt_variable_findings(harness, spans))
    findings.extend(_few_shot_findings(harness, spans))

    if not any(finding.severity == "error" for finding in findings):
        findings.append(
            EvaluationHarnessFinding(
                rule_id="evaluation-harness-verified",
                severity="info",
                message=(
                    f"evaluation harness '{harness.name}' matches declared provider, tokenizer, "
                    "prompt, parser, few-shot, and stop contracts within the finite manifest"
                ),
                suggestion="Keep the harness manifest pinned beside benchmark releases so published scores remain reproducible.",
                witness=(
                    ("select benchmark harness", harness.benchmark_name, harness.name),
                    ("compare finite contract surfaces", None, "no contract-breaking mismatch found"),
                ),
            )
        )
    return EvaluationHarnessReport(
        harness_name=harness.name,
        benchmark_name=harness.benchmark_name,
        findings=tuple(sorted(findings, key=lambda finding: (finding.severity, finding.rule_id, finding.subject or ""))),
    )


def _provider_findings(
    harness: EvaluationHarnessArtifact,
    providers: Sequence[ProviderConfigArtifact],
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    findings: list[EvaluationHarnessFinding] = []
    if harness.provider is None:
        return (
            _missing("provider", "evaluation harness does not declare the provider used to run benchmark prompts", spans.get("provider")),
        )
    if not providers:
        findings.append(_missing("provider", "no provider-config artifact is available to compare with the harness provider", spans.get("provider")))
    provider_names = {provider.provider for provider in providers}
    if provider_names and harness.provider not in provider_names:
        findings.append(
            _mismatch(
                "evaluation-harness-provider-mismatch",
                "provider",
                f"evaluation harness provider '{harness.provider}' is not one of the configured provider contracts",
                expected=", ".join(sorted(provider_names)),
                actual=harness.provider,
                span=spans.get("provider"),
                suggestion="Run the benchmark through the same provider adapter that serving verification uses, or add a matching provider-config artifact.",
            )
        )
    model_contracts = tuple(
        str(value)
        for provider in providers
        for key, value in provider.metadata
        if key in {"model", "model_name", "model_id"} and isinstance(value, str) and value
    )
    if harness.model is not None and model_contracts and harness.model not in model_contracts:
        findings.append(
            _mismatch(
                "evaluation-harness-model-mismatch",
                "model",
                "evaluation harness model does not match model metadata declared by provider-config artifacts",
                expected=", ".join(sorted(model_contracts)),
                actual=harness.model,
                span=spans.get("model"),
                suggestion="Refresh the benchmark harness model field or the provider fixture metadata before comparing scores.",
            )
        )
    elif harness.model is not None and providers and not model_contracts:
        findings.append(
            _missing("model", "provider-config artifacts do not expose model metadata for this harness model contract", spans.get("model"))
        )
    return tuple(findings)


def _tokenizer_findings(
    harness: EvaluationHarnessArtifact,
    tokenizers: Sequence[TokenizerArtifact],
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    if harness.tokenizer is None:
        return (
            _missing("tokenizer", "evaluation harness does not declare the tokenizer used for prompt rendering and scoring", spans.get("tokenizer")),
        )
    if not tokenizers:
        return (
            _missing("tokenizer", "no tokenizer artifact is available to compare with the harness tokenizer", spans.get("tokenizer")),
        )
    tokenizer_facts = {tokenizer.name for tokenizer in tokenizers}
    tokenizer_facts.update(tokenizer.family for tokenizer in tokenizers if tokenizer.family)
    if harness.tokenizer not in tokenizer_facts:
        return (
            _mismatch(
                "evaluation-harness-tokenizer-mismatch",
                "tokenizer",
                "evaluation harness tokenizer is not represented by the configured tokenizer artifacts",
                expected=", ".join(sorted(tokenizer_facts)),
                actual=harness.tokenizer,
                span=spans.get("tokenizer"),
                suggestion="Use the same tokenizer artifact in the evaluation harness and production/provider contract.",
            ),
        )
    return ()


def _template_findings(
    harness: EvaluationHarnessArtifact,
    templates: Sequence[Artifact],
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    if harness.prompt_template is None:
        return (
            _missing("prompt_template", "evaluation harness does not declare the prompt template used for benchmark prompts", spans.get("prompt_template")),
        )
    if not templates:
        return (
            _missing("prompt_template", "no chat-template artifact is available to compare with the harness prompt template", spans.get("prompt_template")),
        )
    names = {template.name for template in templates}
    if harness.prompt_template not in names:
        return (
            _mismatch(
                "evaluation-harness-prompt-template-mismatch",
                "prompt_template",
                "evaluation harness prompt template does not match configured chat-template artifacts",
                expected=", ".join(sorted(names)),
                actual=harness.prompt_template,
                span=spans.get("prompt_template"),
                suggestion="Pin the benchmark prompt renderer to the same chat-template artifact used for serving verification.",
            ),
        )
    return ()


def _stop_findings(
    harness: EvaluationHarnessArtifact,
    stop_policies: Sequence[StopPolicyArtifact],
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    if not harness.stop_sequences:
        return (
            _missing("stop_sequences", "evaluation harness does not declare benchmark stop sequences", spans.get("stop_sequences")),
        )
    if not stop_policies:
        return (
            _missing("stop_sequences", "no stop-policy artifact is available to compare with benchmark stops", spans.get("stop_sequences")),
        )
    configured = {sequence for policy in stop_policies for sequence in policy.stop_sequences}
    observed = set(harness.stop_sequences)
    if observed != configured:
        return (
            _mismatch(
                "evaluation-harness-stop-policy-mismatch",
                "stop_sequences",
                "evaluation harness stop sequences differ from the configured stop-policy artifacts",
                expected=", ".join(sorted(configured)) or "<none>",
                actual=", ".join(sorted(observed)) or "<none>",
                span=spans.get("stop_sequences"),
                suggestion="Use the same stop policy for evaluation and provider requests so parsers see the same completions.",
            ),
        )
    return ()


def _parser_findings(
    harness: EvaluationHarnessArtifact,
    schemas: Sequence[SchemaArtifact],
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    if harness.answer_parser is None:
        return (
            _missing("answer_parser", "evaluation harness does not declare the parser used to grade answers", spans.get("answer_parser")),
        )
    parser = harness.answer_parser.lower().replace("_", "-")
    if parser in {"json", "json-schema", "pydantic", "pydantic-json-schema"}:
        if not schemas:
            return (
                _missing("answer_parser", "no schema artifact is available to compare with the harness answer parser", spans.get("answer_parser")),
            )
        compatible = tuple(schema for schema in schemas if "json" in schema.dialect.lower())
        if not compatible:
            return (
                _mismatch(
                    "evaluation-harness-answer-parser-mismatch",
                    "answer_parser",
                    "JSON-style evaluation answer parser has no JSON Schema-compatible artifact",
                    expected=", ".join(schema.dialect for schema in schemas),
                    actual=harness.answer_parser,
                    span=spans.get("answer_parser"),
                    suggestion="Declare a JSON Schema artifact that models the benchmark grader's accepted answer format.",
                ),
            )
        if harness.answer_schema is not None and harness.answer_schema not in {schema.name for schema in compatible}:
            return (
                _mismatch(
                    "evaluation-harness-answer-parser-mismatch",
                    "answer_schema",
                    "evaluation harness answer_schema does not name any compatible schema artifact",
                    expected=", ".join(sorted(schema.name for schema in compatible)),
                    actual=harness.answer_schema,
                    span=spans.get("answer_schema"),
                    suggestion="Point answer_schema at the exact schema artifact used by the benchmark grader.",
                ),
            )
    return ()


def _prompt_variable_findings(
    harness: EvaluationHarnessArtifact,
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    missing = tuple(sorted(set(harness.required_prompt_variables).difference(harness.prompt_variables)))
    if not missing:
        return ()
    return (
        _mismatch(
            "evaluation-harness-prompt-variable-missing",
            "prompt_variables",
            "evaluation benchmark prompt is missing required variables declared by the prompt renderer",
            expected=", ".join(harness.required_prompt_variables),
            actual=", ".join(harness.prompt_variables) or "<none>",
            span=spans.get("prompt_variables") or spans.get("required_prompt_variables"),
            suggestion=f"Populate benchmark prompt variables before rendering: {', '.join(missing)}.",
        ),
    )


def _few_shot_findings(
    harness: EvaluationHarnessArtifact,
    spans: Mapping[str, SourceSpan],
) -> tuple[EvaluationHarnessFinding, ...]:
    findings: list[EvaluationHarnessFinding] = []
    allowed = set(harness.allowed_roles or ("system", "developer", "user", "assistant", "tool", "function"))
    for index, example in enumerate(harness.few_shot_examples):
        if example.role not in allowed:
            findings.append(
                _mismatch(
                    "evaluation-harness-few-shot-role-mismatch",
                    f"few_shot_examples[{index}].role",
                    f"few-shot example '{example.example_id}' uses a role outside the declared model/provider contract",
                    expected=", ".join(sorted(allowed)),
                    actual=example.role,
                    span=spans.get(f"few_shot_examples.{index}.role"),
                    suggestion="Map few-shot roles to the same role vocabulary accepted by the chat template/provider contract.",
                )
            )
    if harness.max_prompt_tokens is not None:
        token_counts = tuple(example.token_count for example in harness.few_shot_examples)
        if any(count is None for count in token_counts):
            findings.append(
                _missing(
                    "few_shot_examples",
                    "few-shot token counts are incomplete, so the benchmark prompt budget cannot be proven statically",
                    spans.get("few_shot_examples"),
                )
            )
        else:
            total = sum(count for count in token_counts if count is not None)
            if total > harness.max_prompt_tokens:
                findings.append(
                    _mismatch(
                        "evaluation-harness-few-shot-budget-overflow",
                        "max_prompt_tokens",
                        "few-shot examples exceed the evaluation harness prompt budget",
                        expected=f"<= {harness.max_prompt_tokens}",
                        actual=str(total),
                        span=spans.get("max_prompt_tokens"),
                        suggestion="Reduce few-shot examples or raise the benchmark prompt budget to match the provider context contract.",
                    )
                )
    return tuple(findings)


def _missing(subject: str, message: str, span: SourceSpan | None) -> EvaluationHarnessFinding:
    return EvaluationHarnessFinding(
        rule_id="evaluation-harness-contract-missing",
        severity="warning",
        message=message,
        suggestion="Declare the missing evaluation harness contract field or add the matching PromptABI artifact so the check can decide.",
        subject=subject,
        span=span,
        witness=(("inspect evaluation harness contract", subject, "missing or no comparable artifact"),),
    )


def _mismatch(
    rule_id: str,
    subject: str,
    message: str,
    *,
    expected: str,
    actual: str,
    span: SourceSpan | None,
    suggestion: str,
) -> EvaluationHarnessFinding:
    return EvaluationHarnessFinding(
        rule_id=rule_id,
        severity="error",
        message=message,
        suggestion=suggestion,
        subject=subject,
        expected=expected,
        actual=actual,
        span=span,
        witness=(
            ("select evaluation harness field", subject, actual),
            ("compare against configured PromptABI contract", expected, "mismatch"),
        ),
    )
