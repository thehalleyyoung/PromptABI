"""Deterministic mutation-based fuzzing for PromptABI artifact contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactKind,
    ArtifactLocation,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TruncationStrategy,
)
from .chat_templates import (
    ChatTemplateParseError,
    ChatTemplateSymbolicBounds,
    parse_hf_chat_template_config,
    symbolically_execute_chat_template,
)
from .config import VerificationConfig
from .formal import FiniteContractProblem, Gt, IntRangeDomain, NamedConstraint, SolverStatus, Value, Var
from .grammars import GrammarIngestionError, ingest_grammar_mapping, ingest_grammar_text
from .json_schema import compile_json_schema_mapping, normalize_json_schema_mapping
from .loaders import LoadedArtifact
from .static_contracts import analyze_static_contracts
from .stop_analysis import analyze_stop_policy_tokenizer
from .stop_overreachability import analyze_stop_overreachability
from .tokenizers import ByteLevelTokenizer
from .tool_schemas import ToolSchemaIngestionError, ingest_tool_schema_mapping


class FuzzSurface(StrEnum):
    """Artifact families covered by the mutation fuzzer."""

    CHAT_TEMPLATES = "chat-templates"
    TOKENIZERS = "tokenizers"
    STOP_POLICIES = "stop-policies"
    SCHEMAS = "schemas"
    GRAMMARS = "grammars"
    TOOL_DEFINITIONS = "tool-definitions"
    TRUNCATION_CONFIGS = "truncation-configs"
    SMT_ENCODINGS = "smt-encodings"


ALL_FUZZ_SURFACES = tuple(surface for surface in FuzzSurface)
FUZZING_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class MutationCase:
    """One baseline or mutated artifact case."""

    case_id: str
    surface: FuzzSurface
    description: str
    payload: object
    mutation: str | None = None


@dataclass(frozen=True, slots=True)
class MutationObservation:
    """A normalized finding emitted by a parser, analyzer, or solver."""

    rule_id: str
    severity: str
    message: str
    evidence: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
        }
        if self.evidence:
            data["evidence"] = dict(self.evidence)
        return data


@dataclass(frozen=True, slots=True)
class MutationCaseResult:
    """Replay result for one mutation case."""

    case: MutationCase
    observations: tuple[MutationObservation, ...]
    introduced_rule_ids: tuple[str, ...] = ()

    @property
    def introduced_violation_count(self) -> int:
        introduced = set(self.introduced_rule_ids)
        return sum(1 for observation in self.observations if observation.rule_id in introduced and observation.severity != "info")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case.case_id,
            "surface": self.case.surface.value,
            "description": self.case.description,
            "mutation": self.case.mutation,
            "observations": [observation.to_dict() for observation in self.observations],
            "introduced_rule_ids": list(self.introduced_rule_ids),
            "introduced_violation_count": self.introduced_violation_count,
        }


@dataclass(frozen=True, slots=True)
class MutationFuzzReport:
    """Complete deterministic mutation-fuzzing report."""

    surfaces: tuple[FuzzSurface, ...]
    baseline_results: tuple[MutationCaseResult, ...]
    mutation_results: tuple[MutationCaseResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.baseline_results) + len(self.mutation_results)

    @property
    def mutation_count(self) -> int:
        return len(self.mutation_results)

    @property
    def introduced_violation_count(self) -> int:
        return sum(result.introduced_violation_count for result in self.mutation_results)

    @property
    def discovered_rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted({rule for result in self.mutation_results for rule in result.introduced_rule_ids}))

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": FUZZING_MANIFEST_VERSION,
            "surfaces": [surface.value for surface in self.surfaces],
            "case_count": self.case_count,
            "mutation_count": self.mutation_count,
            "introduced_violation_count": self.introduced_violation_count,
            "discovered_rule_ids": list(self.discovered_rule_ids),
            "baselines": [result.to_dict() for result in self.baseline_results],
            "mutations": [result.to_dict() for result in self.mutation_results],
        }


class MutationFuzzingError(ValueError):
    """Raised when a mutation-fuzzing request is malformed."""


def run_mutation_fuzzing(surfaces: Sequence[str | FuzzSurface] = ("all",)) -> MutationFuzzReport:
    """Run deterministic mutation fuzzing over selected artifact surfaces."""

    selected = _resolve_surfaces(surfaces)
    baseline_results: list[MutationCaseResult] = []
    mutation_results: list[MutationCaseResult] = []
    for surface in selected:
        baseline = _baseline_case(surface)
        baseline_observations = _run_case(baseline)
        baseline_rule_ids = {observation.rule_id for observation in baseline_observations}
        baseline_results.append(MutationCaseResult(case=baseline, observations=baseline_observations))
        for mutation in _mutation_cases(surface):
            observations = _run_case(mutation)
            introduced = tuple(
                sorted(
                    {
                        observation.rule_id
                        for observation in observations
                        if observation.rule_id not in baseline_rule_ids and observation.severity != "info"
                    }
                )
            )
            mutation_results.append(
                MutationCaseResult(case=mutation, observations=observations, introduced_rule_ids=introduced)
            )
    return MutationFuzzReport(
        surfaces=selected,
        baseline_results=tuple(baseline_results),
        mutation_results=tuple(mutation_results),
    )


def render_mutation_fuzz_json(report: MutationFuzzReport) -> str:
    """Render a mutation-fuzzing report as strict JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_mutation_fuzz_text(report: MutationFuzzReport) -> str:
    """Render a concise terminal mutation-fuzzing summary."""

    lines = [
        "PromptABI mutation fuzzing",
        f"surfaces: {', '.join(surface.value for surface in report.surfaces)}",
        f"cases: {report.case_count} ({report.mutation_count} mutations)",
        f"introduced violations: {report.introduced_violation_count}",
        f"discovered rules: {', '.join(report.discovered_rule_ids) or '<none>'}",
    ]
    for result in report.mutation_results:
        if result.introduced_rule_ids:
            lines.append(
                f"- {result.case.case_id}: {', '.join(result.introduced_rule_ids)}"
            )
    return "\n".join(lines) + "\n"


def _resolve_surfaces(values: Sequence[str | FuzzSurface]) -> tuple[FuzzSurface, ...]:
    requested = tuple(values or ("all",))
    if any(str(value) == "all" for value in requested):
        return ALL_FUZZ_SURFACES
    resolved: list[FuzzSurface] = []
    for value in requested:
        try:
            surface = value if isinstance(value, FuzzSurface) else FuzzSurface(str(value))
        except ValueError as exc:
            known = ", ".join(surface.value for surface in ALL_FUZZ_SURFACES)
            raise MutationFuzzingError(f"unknown fuzz surface {value!r}; expected one of: {known}, all") from exc
        if surface not in resolved:
            resolved.append(surface)
    return tuple(resolved)


def _baseline_case(surface: FuzzSurface) -> MutationCase:
    return _case_factories()[surface][0]()


def _mutation_cases(surface: FuzzSurface) -> tuple[MutationCase, ...]:
    return tuple(factory() for factory in _case_factories()[surface][1:])


def _case_factories() -> dict[FuzzSurface, tuple[Callable[[], MutationCase], ...]]:
    return {
        FuzzSurface.CHAT_TEMPLATES: (
            _chat_template_baseline,
            _chat_template_unsupported_macro,
            _chat_template_path_explosion,
        ),
        FuzzSurface.TOKENIZERS: (
            _tokenizer_baseline,
            _tokenizer_control_token_collision,
            _tokenizer_normalization_drift,
        ),
        FuzzSurface.STOP_POLICIES: (
            _stop_policy_baseline,
            _stop_policy_prefix_collision,
            _stop_policy_json_overreach,
        ),
        FuzzSurface.SCHEMAS: (
            _schema_baseline,
            _schema_recursive_ref,
            _schema_unsatisfiable_enum,
        ),
        FuzzSurface.GRAMMARS: (
            _grammar_baseline,
            _grammar_unsupported_lookbehind,
            _grammar_missing_start_rule,
        ),
        FuzzSurface.TOOL_DEFINITIONS: (
            _tool_definition_baseline,
            _tool_definition_missing_required,
            _tool_definition_open_argument_string,
        ),
        FuzzSurface.TRUNCATION_CONFIGS: (
            _truncation_baseline,
            _truncation_required_segment_overflow,
            _truncation_unknown_required_segment,
        ),
        FuzzSurface.SMT_ENCODINGS: (
            _smt_baseline,
            _smt_satisfiable_violation,
            _smt_unsat_core_conflict,
        ),
    }


def _run_case(case: MutationCase) -> tuple[MutationObservation, ...]:
    runners: dict[FuzzSurface, Callable[[object], tuple[MutationObservation, ...]]] = {
        FuzzSurface.CHAT_TEMPLATES: _run_chat_template,
        FuzzSurface.TOKENIZERS: _run_tokenizer,
        FuzzSurface.STOP_POLICIES: _run_stop_policy,
        FuzzSurface.SCHEMAS: _run_schema,
        FuzzSurface.GRAMMARS: _run_grammar,
        FuzzSurface.TOOL_DEFINITIONS: _run_tool_definition,
        FuzzSurface.TRUNCATION_CONFIGS: _run_truncation_config,
        FuzzSurface.SMT_ENCODINGS: _run_smt_encoding,
    }
    return runners[case.surface](case.payload)


def _chat_template_baseline() -> MutationCase:
    return MutationCase(
        case_id="chat-template-baseline",
        surface=FuzzSurface.CHAT_TEMPLATES,
        description="supported ChatML-style template with bounded messages",
        payload={
            "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
    )


def _chat_template_unsupported_macro() -> MutationCase:
    payload = dict(_chat_template_baseline().payload)  # type: ignore[arg-type]
    payload["chat_template"] = "{% macro role(x) %}{{ x }}{% endmacro %}{{ role(messages[0]['role']) }}"
    return MutationCase(
        case_id="chat-template-unsupported-macro",
        surface=FuzzSurface.CHAT_TEMPLATES,
        description="mutation injects a Jinja macro outside the supported symbolic fragment",
        payload=payload,
        mutation="insert-unsupported-macro",
    )


def _chat_template_path_explosion() -> MutationCase:
    payload = dict(_chat_template_baseline().payload)  # type: ignore[arg-type]
    payload["chat_template"] = "".join(
        f"{{% if messages[0]['role'] == 'role{i}' %}}role{i}{{% endif %}}" for i in range(8)
    )
    return MutationCase(
        case_id="chat-template-path-explosion",
        surface=FuzzSurface.CHAT_TEMPLATES,
        description="mutation creates more symbolic paths than the fuzzing bound allows",
        payload=payload,
        mutation="branch-amplification",
    )


def _tokenizer_baseline() -> MutationCase:
    return MutationCase(
        case_id="tokenizer-baseline",
        surface=FuzzSurface.TOKENIZERS,
        description="byte-level tokenizer with stable ASCII round trips",
        payload={"samples": ("hello", "assistant"), "added_tokens": (), "special_tokens": {}, "normalization": ()},
    )


def _tokenizer_control_token_collision() -> MutationCase:
    return MutationCase(
        case_id="tokenizer-control-token-collision",
        surface=FuzzSurface.TOKENIZERS,
        description="mutation turns a role delimiter into an added special token and stop string",
        payload={
            "samples": ("<|assistant|>", "user"),
            "added_tokens": ("<|assistant|>",),
            "special_tokens": {"<|assistant|>": 32001},
            "normalization": (),
        },
        mutation="add-control-token",
    )


def _tokenizer_normalization_drift() -> MutationCase:
    return MutationCase(
        case_id="tokenizer-normalization-drift",
        surface=FuzzSurface.TOKENIZERS,
        description="mutation introduces NFKC-sensitive text that changes before tokenization",
        payload={"samples": ("ℌello",), "added_tokens": (), "special_tokens": {}, "normalization": ("nfkc",)},
        mutation="nfkc-drift",
    )


def _stop_policy_baseline() -> MutationCase:
    return MutationCase(
        case_id="stop-policy-baseline",
        surface=FuzzSurface.STOP_POLICIES,
        description="single stop sequence with a decodable token id",
        payload={"stop_sequences": ("\nEND",), "stop_token_ids": (65,)},
    )


def _stop_policy_prefix_collision() -> MutationCase:
    return MutationCase(
        case_id="stop-policy-prefix-collision",
        surface=FuzzSurface.STOP_POLICIES,
        description="mutation adds proper-prefix stop strings and an unreachable token id",
        payload={"stop_sequences": ("}", "}}", "</tool_call>"), "stop_token_ids": (999999,)},
        mutation="prefix-stop-and-unreachable-id",
    )


def _stop_policy_json_overreach() -> MutationCase:
    return MutationCase(
        case_id="stop-policy-json-overreach",
        surface=FuzzSurface.STOP_POLICIES,
        description="mutation makes raw stops reachable inside structured JSON/tool output regions",
        payload={"stop_sequences": ("}", "```", "</tool_call>"), "stop_token_ids": ()},
        mutation="structured-output-overreach",
    )


def _schema_baseline() -> MutationCase:
    return MutationCase(
        case_id="schema-baseline",
        surface=FuzzSurface.SCHEMAS,
        description="supported closed JSON object schema",
        payload={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def _schema_recursive_ref() -> MutationCase:
    return MutationCase(
        case_id="schema-recursive-ref",
        surface=FuzzSurface.SCHEMAS,
        description="mutation adds recursive references beyond the bounded schema fragment",
        payload={
            "$defs": {"node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/node"}}}},
            "$ref": "#/$defs/node",
        },
        mutation="recursive-ref",
    )


def _schema_unsatisfiable_enum() -> MutationCase:
    return MutationCase(
        case_id="schema-unsatisfiable-enum",
        surface=FuzzSurface.SCHEMAS,
        description="mutation creates an empty enum accepted by JSON syntax but impossible as a value contract",
        payload={"enum": []},
        mutation="empty-enum",
    )


def _grammar_baseline() -> MutationCase:
    return MutationCase(
        case_id="grammar-baseline",
        surface=FuzzSurface.GRAMMARS,
        description="simple PromptABI grammar with a declared start rule",
        payload={"start": "answer", "rules": {"answer": '"yes" | "no"'}},
    )


def _grammar_unsupported_lookbehind() -> MutationCase:
    return MutationCase(
        case_id="grammar-unsupported-lookbehind",
        surface=FuzzSurface.GRAMMARS,
        description="mutation injects a regex lookbehind outside the local grammar fragment",
        payload={"type": "regex", "regex": "(?<=secret)answer"},
        mutation="regex-lookbehind",
    )


def _grammar_missing_start_rule() -> MutationCase:
    return MutationCase(
        case_id="grammar-missing-start-rule",
        surface=FuzzSurface.GRAMMARS,
        description="mutation points the grammar start symbol at an absent rule",
        payload={"start": "missing", "rules": {"answer": '"ok"'}},
        mutation="missing-start-rule",
    )


def _tool_definition_baseline() -> MutationCase:
    return MutationCase(
        case_id="tool-definition-baseline",
        surface=FuzzSurface.TOOL_DEFINITIONS,
        description="OpenAI-style tool schema with required parameters declared under properties",
        payload=[
            {
                "type": "function",
                "function": {
                    "name": "route_ticket",
                    "description": "Route a support ticket.",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )


def _tool_definition_missing_required() -> MutationCase:
    payload = json.loads(json.dumps(_tool_definition_baseline().payload))
    payload[0]["function"]["parameters"]["required"].append("priority")
    return MutationCase(
        case_id="tool-definition-missing-required",
        surface=FuzzSurface.TOOL_DEFINITIONS,
        description="mutation requires a tool parameter that has no declared property",
        payload=payload,
        mutation="required-property-drift",
    )


def _tool_definition_open_argument_string() -> MutationCase:
    return MutationCase(
        case_id="tool-definition-open-argument-string",
        surface=FuzzSurface.TOOL_DEFINITIONS,
        description="mutation converts the tool-call envelope into a string-encoded, open parameter surface",
        payload={
            "provider": "anthropic",
            "tools": [
                {
                    "name": "emit_json",
                    "input_schema": {"type": "string"},
                }
            ],
        },
        mutation="argument-string-envelope",
    )


def _truncation_baseline() -> MutationCase:
    return MutationCase(
        case_id="truncation-baseline",
        surface=FuzzSurface.TRUNCATION_CONFIGS,
        description="required prompt segments fit in the modeled framework budget",
        payload={"segment_tokens": (12, 8), "max_context_tokens": 64, "reserve_output_tokens": 8},
    )


def _truncation_required_segment_overflow() -> MutationCase:
    return MutationCase(
        case_id="truncation-required-segment-overflow",
        surface=FuzzSurface.TRUNCATION_CONFIGS,
        description="mutation makes must-survive prompt regions exceed the input budget",
        payload={"segment_tokens": (40, 35), "max_context_tokens": 64, "reserve_output_tokens": 16},
        mutation="required-budget-overflow",
    )


def _truncation_unknown_required_segment() -> MutationCase:
    return MutationCase(
        case_id="truncation-unknown-required-segment",
        surface=FuzzSurface.TRUNCATION_CONFIGS,
        description="mutation removes a required segment token count so survival cannot be proved",
        payload={"segment_tokens": (12, None), "max_context_tokens": 64, "reserve_output_tokens": 8},
        mutation="unknown-required-token-count",
    )


def _smt_baseline() -> MutationCase:
    return MutationCase(
        case_id="smt-baseline",
        surface=FuzzSurface.SMT_ENCODINGS,
        description="finite SMT contract with no counterexample in the domain",
        payload={"domain": (0, 1), "threshold": 2, "mode": "violation-search"},
    )


def _smt_satisfiable_violation() -> MutationCase:
    return MutationCase(
        case_id="smt-satisfiable-violation",
        surface=FuzzSurface.SMT_ENCODINGS,
        description="mutation widens the finite domain so Z3/enumeration extracts a violation witness",
        payload={"domain": (0, 3), "threshold": 2, "mode": "violation-search"},
        mutation="widen-domain",
    )


def _smt_unsat_core_conflict() -> MutationCase:
    return MutationCase(
        case_id="smt-unsat-core-conflict",
        surface=FuzzSurface.SMT_ENCODINGS,
        description="mutation creates contradictory finite constraints to exercise unsat/unknown reporting",
        payload={"domain": (1, 1), "threshold": 0, "mode": "contradiction"},
        mutation="contradictory-constraints",
    )


def _run_chat_template(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    observations: list[MutationObservation] = []
    try:
        parsed = parse_hf_chat_template_config(payload)
    except ChatTemplateParseError as exc:
        return (_observation("chat-template-parse-error", "error", str(exc)),)
    if parsed.unsupported_constructs:
        observations.extend(
            _observation("chat-template-unsupported-construct", "warning", item.reason, (("expression", item.expression),))
            for item in parsed.unsupported_constructs
        )
    execution = symbolically_execute_chat_template(parsed, bounds=ChatTemplateSymbolicBounds(max_paths=4))
    if execution.abstentions:
        observations.extend(
            _observation("chat-template-symbolic-abstention", "warning", item.reason, (("expression", item.expression),))
            for item in execution.abstentions
        )
    if not observations:
        observations.append(_observation("chat-template-supported", "info", f"{len(execution.paths)} symbolic paths"))
    return tuple(observations)


def _run_tokenizer(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    tokenizer = ByteLevelTokenizer(
        added_tokens=tuple(payload["added_tokens"]),
        special_tokens=dict(payload["special_tokens"]),
        normalization=tuple(payload["normalization"]),
    )
    observations: list[MutationObservation] = []
    for sample in payload["samples"]:
        sample_text = str(sample)
        encoded = tokenizer.encode(sample_text, add_special_tokens=False)
        round_trip = tokenizer.round_trip(sample_text)
        if round_trip.normalized_text != round_trip.input_text:
            observations.append(
                _observation(
                    "tokenizer-normalization-drift",
                    "warning",
                    "sample changes under tokenizer normalization",
                    (("input", round_trip.input_text), ("normalized", round_trip.normalized_text)),
                )
            )
        if any(token.special or token.added for token in encoded.tokens):
            observations.append(
                _observation(
                    "tokenizer-control-token-reachable",
                    "error",
                    "mutated user sample encodes as an added or special control token",
                    (("input", round_trip.input_text), ("token_ids", ",".join(str(item) for item in round_trip.token_ids))),
                )
            )
    if not observations:
        observations.append(_observation("tokenizer-round-trip-stable", "info", "all fuzz samples round-tripped exactly"))
    return tuple(observations)


def _run_stop_policy(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    stop_policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="fuzz-stops",
        location=ArtifactLocation(uri="memory://fuzz/stops"),
        stop_sequences=tuple(payload["stop_sequences"]),
        stop_token_ids=tuple(payload["stop_token_ids"]),
    )
    tokenizer = ByteLevelTokenizer(added_tokens=("</tool_call>",), special_tokens={"</tool_call>": 32002})
    token_report = analyze_stop_policy_tokenizer(stop_policy, tokenizer)
    overreach = analyze_stop_overreachability(stop_policy)
    observations: list[MutationObservation] = []
    observations.extend(
        _observation("stop-token-unreachable", "error", item.error or "stop token id is unreachable", (("token_id", str(item.token_id)),))
        for item in token_report.unreachable_token_ids
    )
    observations.extend(
        _observation("stop-prefix-collision", "warning", item.relation, (("shorter", item.shorter), ("longer", item.longer)))
        for item in token_report.collisions
    )
    observations.extend(
        _observation(
            "stop-overreachability",
            "error",
            f"stop {item.stop_sequence!r} fires in {item.region.name} before {item.resulting_structure}",
            (
                ("stop_sequence", item.stop_sequence),
                ("region", item.region.name),
                ("firing_point", item.firing_point),
                ("resulting_state", item.resulting_state),
            ),
        )
        for item in overreach.findings
    )
    if not observations:
        observations.append(_observation("stop-policy-stable", "info", "no stop collisions or overreach findings"))
    return tuple(observations)


def _run_schema(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    observations: list[MutationObservation] = []
    normalized = normalize_json_schema_mapping(payload)
    observations.extend(
        _observation(f"schema-{issue.code}", issue.severity, issue.message, (("path", ".".join(issue.path) or "<root>"),))
        for issue in normalized.issues
    )
    try:
        compiled = compile_json_schema_mapping(payload)
    except ValueError as exc:
        observations.append(_observation("schema-compile-error", "error", str(exc)))
    else:
        if compiled.witness.text is None:
            observations.append(_observation("schema-empty-language", "error", "compiled schema has no finite witness"))
    if isinstance(payload.get("enum"), list) and not payload["enum"]:
        observations.append(_observation("schema-empty-enum", "error", "enum has no legal values"))
    if not observations:
        observations.append(_observation("schema-supported", "info", "schema normalized and compiled"))
    return tuple(observations)


def _run_grammar(payload: object) -> tuple[MutationObservation, ...]:
    observations: list[MutationObservation] = []
    try:
        if isinstance(payload, dict):
            result = ingest_grammar_mapping(payload)
        else:
            result = ingest_grammar_text(str(payload))
    except GrammarIngestionError as exc:
        return (_observation("grammar-ingestion-error", "error", str(exc)),)
    observations.extend(
        _observation(f"grammar-{issue.code}", issue.severity, issue.message)
        for issue in result.issues
    )
    if result.start_symbol and result.rule_names and result.start_symbol not in result.rule_names:
        observations.append(
            _observation(
                "grammar-missing-start-rule",
                "error",
                "start symbol does not name a declared rule",
                (("start_symbol", result.start_symbol), ("rules", ",".join(result.rule_names))),
            )
        )
    if not observations:
        observations.append(_observation("grammar-supported", "info", "grammar ingested without unsupported features"))
    return tuple(observations)


def _run_tool_definition(payload: object) -> tuple[MutationObservation, ...]:
    observations: list[MutationObservation] = []
    try:
        result = ingest_tool_schema_mapping(payload)
    except ToolSchemaIngestionError as exc:
        return (_observation("tool-schema-ingestion-error", "error", str(exc)),)
    observations.extend(
        _observation(f"tool-schema-{issue.kind.value}", "warning", issue.message, (("path", ".".join(issue.path)),))
        for issue in result.issues
    )
    tool_artifact = ToolDefinitionArtifact(
        kind=ArtifactKind.TOOL_DEFINITION,
        name="fuzz-tools",
        location=ArtifactLocation(uri="memory://fuzz/tools"),
        provider=result.provider_family.value,
        tool_names=result.tool_names,
    )
    loaded = LoadedArtifact(
        artifact=tool_artifact,
        source_type="memory",
        pinned=True,
        resolved=True,
        metadata=result.to_metadata(),
    )
    provider = LoadedArtifact(
        artifact=ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name="active-provider",
            location=ArtifactLocation(uri="memory://fuzz/provider"),
            provider="openai",
        ),
        source_type="memory",
        pinned=True,
        resolved=True,
    )
    static_report = analyze_static_contracts(VerificationConfig(name="tool-fuzz"), (loaded, provider))
    observations.extend(_static_observations(static_report.findings))
    if not observations:
        observations.append(_observation("tool-schema-stable", "info", "tool schemas ingested without contract issues"))
    return tuple(observations)


def _run_truncation_config(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    segments = tuple(
        PromptSegment(
            name=f"segment-{index}",
            role="system" if index == 0 else "tool",
            required=True,
            token_count=tokens,
        )
        for index, tokens in enumerate(payload["segment_tokens"])
    )
    segment_artifact = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="fuzz-segments",
        location=ArtifactLocation(uri="memory://fuzz/segments"),
        segments=segments,
    )
    truncation_artifact = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="fuzz-truncation",
        location=ArtifactLocation(uri="memory://fuzz/truncation"),
        framework="langchain",
        strategy=TruncationStrategy.OLDEST_MESSAGE,
        max_context_tokens=int(payload["max_context_tokens"]),
        reserve_output_tokens=int(payload["reserve_output_tokens"]),
    )
    loaded = (
        _loaded(segment_artifact),
        _loaded(truncation_artifact),
    )
    report = analyze_static_contracts(VerificationConfig(name="truncation-fuzz"), loaded)
    observations = _static_observations(report.findings)
    return observations or (_observation("truncation-budget-stable", "info", "required segments fit modeled budget"),)


def _run_smt_encoding(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    low, high = payload["domain"]
    threshold = int(payload["threshold"])
    constraints = [NamedConstraint("length-exceeds-threshold", Gt(Var("length"), Value(threshold)))]
    if payload["mode"] == "contradiction":
        constraints.append(NamedConstraint("impossible-upper-bound", Gt(Value(threshold), Var("length"))))
    problem = FiniteContractProblem(
        name="mutation-fuzz-smt-contract",
        variables=(IntRangeDomain("length", int(low), int(high)),),
        constraints=tuple(constraints),
    )
    result = problem.solve(prefer_z3=True)
    if result.sat:
        return (
            _observation(
                "smt-counterexample",
                "error",
                "finite SMT encoding admits a counterexample",
                (("assignment", json.dumps(result.assignment, sort_keys=True)),),
            ),
        )
    if result.status is SolverStatus.UNSAT:
        return (_observation("smt-unsat-contract", "warning", "finite SMT encoding is unsatisfiable"),)
    return (_observation("smt-unknown-contract", "warning", "finite SMT encoding returned unknown"),)


def _static_observations(findings) -> tuple[MutationObservation, ...]:
    return tuple(
        _observation(
            finding.name,
            finding.severity,
            finding.message,
            tuple((key, value) for key, value in finding.evidence),
        )
        for finding in findings
        if finding.severity != "info"
    )


def _loaded(artifact) -> LoadedArtifact:
    return LoadedArtifact(artifact=artifact, source_type="memory", pinned=True, resolved=True)


def _observation(
    rule_id: str,
    severity: str,
    message: str,
    evidence: tuple[tuple[str, str], ...] = (),
) -> MutationObservation:
    return MutationObservation(rule_id=rule_id, severity=severity, message=message, evidence=evidence)
