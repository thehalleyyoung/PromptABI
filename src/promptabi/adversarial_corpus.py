"""Deterministic adversarial prompt-interface corpus generation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, ArtifactLocation, ProviderConfigArtifact
from .chat_templates import parse_hf_chat_template_config
from .loaders import ArtifactLoader
from .mutation_fuzzing import MutationObservation, _run_stop_policy, _run_tokenizer
from .provider_migration import analyze_provider_migration
from .role_boundaries import analyze_role_boundary_nonforgeability
from .tool_schemas import ToolSchemaIngestionError, ingest_tool_schema_mapping


ADVERSARIAL_CORPUS_VERSION = 1


class AdversarialSurface(StrEnum):
    """Adversarial input classes covered by the generated corpus."""

    ROLE_DELIMITERS = "role-delimiters"
    SPECIAL_TOKENS = "special-tokens"
    UNICODE_NORMALIZATION = "unicode-normalization"
    JSON_ESCAPING = "json-escaping"
    MARKDOWN_FENCES = "markdown-fences"
    XML_TOOL_TAGS = "xml-tool-tags"
    PROVIDER_ENVELOPES = "provider-envelopes"


REQUIRED_ADVERSARIAL_SURFACES = tuple(surface for surface in AdversarialSurface)


@dataclass(frozen=True, slots=True)
class AdversarialCorpusCase:
    """One generated adversarial payload with expected structural findings."""

    case_id: str
    surface: AdversarialSurface
    description: str
    payload: dict[str, object]
    expected_rule_ids: tuple[str, ...]

    @property
    def payload_sha256(self) -> str:
        return _stable_sha256(self.payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "surface": self.surface.value,
            "description": self.description,
            "payload": _json_safe(self.payload),
            "payload_sha256": self.payload_sha256,
            "expected_rule_ids": list(self.expected_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class AdversarialCorpusReplay:
    """Replay result for one generated adversarial case."""

    case: AdversarialCorpusCase
    observed_rule_ids: tuple[str, ...]
    observations: tuple[MutationObservation, ...]

    @property
    def passed(self) -> bool:
        return set(self.case.expected_rule_ids).issubset(self.observed_rule_ids)

    @property
    def missing_rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.case.expected_rule_ids).difference(self.observed_rule_ids)))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case.case_id,
            "surface": self.case.surface.value,
            "passed": self.passed,
            "expected_rule_ids": list(self.case.expected_rule_ids),
            "observed_rule_ids": list(self.observed_rule_ids),
            "missing_rule_ids": list(self.missing_rule_ids),
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(frozen=True, slots=True)
class AdversarialCorpusReport:
    """Generated adversarial corpus manifest plus replay evidence."""

    cases: tuple[AdversarialCorpusCase, ...]
    replays: tuple[AdversarialCorpusReplay, ...]

    @property
    def all_cases_passed(self) -> bool:
        return all(replay.passed for replay in self.replays)

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(surface.value for surface in REQUIRED_ADVERSARIAL_SURFACES)

    @property
    def manifest_sha256(self) -> str:
        payload = {
            "manifest_version": ADVERSARIAL_CORPUS_VERSION,
            "cases": [case.to_dict() for case in self.cases],
        }
        return _stable_sha256(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": ADVERSARIAL_CORPUS_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "case_count": len(self.cases),
            "surfaces": list(self.surfaces),
            "all_cases_passed": self.all_cases_passed,
            "cases": [case.to_dict() for case in self.cases],
            "replays": [replay.to_dict() for replay in self.replays],
        }


class AdversarialCorpusError(ValueError):
    """Raised when adversarial corpus generation or replay is invalid."""


def generate_adversarial_corpus() -> tuple[AdversarialCorpusCase, ...]:
    """Generate deterministic adversarial cases across all required surfaces."""

    cases = (
        _role_delimiter_case(),
        _special_token_case(),
        _unicode_normalization_case(),
        _json_escaping_case(),
        _markdown_fence_case(),
        _xml_tool_tag_case(),
        _provider_envelope_case(),
    )
    observed_surfaces = {case.surface for case in cases}
    missing = set(REQUIRED_ADVERSARIAL_SURFACES).difference(observed_surfaces)
    if missing:
        names = ", ".join(sorted(surface.value for surface in missing))
        raise AdversarialCorpusError(f"adversarial corpus is missing required surface(s): {names}")
    return cases


def replay_adversarial_corpus(
    cases: tuple[AdversarialCorpusCase, ...] | None = None,
) -> tuple[AdversarialCorpusReplay, ...]:
    """Replay generated adversarial cases against real PromptABI analyzers."""

    resolved_cases = cases or generate_adversarial_corpus()
    return tuple(_replay_case(case) for case in resolved_cases)


def build_adversarial_corpus_manifest() -> dict[str, object]:
    """Build a deterministic manifest with replay verdicts and content hashes."""

    cases = generate_adversarial_corpus()
    report = AdversarialCorpusReport(cases=cases, replays=replay_adversarial_corpus(cases))
    return report.to_dict()


def write_adversarial_corpus_manifest(path: str | Path) -> dict[str, object]:
    """Write the generated adversarial corpus manifest as deterministic JSON."""

    manifest = build_adversarial_corpus_manifest()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def render_adversarial_corpus_json(report: AdversarialCorpusReport | None = None) -> str:
    """Render the generated corpus report as deterministic JSON."""

    resolved = report or AdversarialCorpusReport(
        cases=generate_adversarial_corpus(),
        replays=replay_adversarial_corpus(),
    )
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"


def render_adversarial_corpus_text(report: AdversarialCorpusReport | None = None) -> str:
    """Render a concise generated-corpus replay summary."""

    resolved = report or AdversarialCorpusReport(
        cases=generate_adversarial_corpus(),
        replays=replay_adversarial_corpus(),
    )
    lines = [
        "PromptABI adversarial corpus",
        f"status: {'PASS' if resolved.all_cases_passed else 'FAIL'}",
        f"cases: {len(resolved.cases)}",
        f"surfaces: {', '.join(resolved.surfaces)}",
        f"manifest_sha256: {resolved.manifest_sha256}",
    ]
    for replay in resolved.replays:
        verdict = "PASS" if replay.passed else "FAIL"
        observed = ", ".join(replay.observed_rule_ids) or "<none>"
        lines.append(f"- {replay.case.case_id}: {verdict} ({observed})")
    return "\n".join(lines) + "\n"


def _role_delimiter_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="role-delimiter-chatml-assistant-forgery",
        surface=AdversarialSurface.ROLE_DELIMITERS,
        description="user content attempts to render a ChatML assistant header",
        payload={
            "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        expected_rule_ids=("role-boundary-nonforgeability",),
    )


def _special_token_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="special-token-added-assistant-reachable",
        surface=AdversarialSurface.SPECIAL_TOKENS,
        description="attacker text is an added special token under byte-level tokenization",
        payload={
            "samples": ("<|assistant|>", "<|im_start|>"),
            "added_tokens": ("<|assistant|>", "<|im_start|>"),
            "special_tokens": {"<|assistant|>": 32001, "<|im_start|>": 32002},
            "normalization": (),
        },
        expected_rule_ids=("tokenizer-control-token-reachable",),
    )


def _unicode_normalization_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="unicode-normalization-nfkc-confusable",
        surface=AdversarialSurface.UNICODE_NORMALIZATION,
        description="NFKC-sensitive Unicode text changes before tokenization",
        payload={
            "samples": ("Ａssistant ℌeader", "＜tool_call＞"),
            "added_tokens": (),
            "special_tokens": {},
            "normalization": ("nfkc",),
        },
        expected_rule_ids=("tokenizer-normalization-drift",),
    )


def _json_escaping_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="json-escaping-early-object-stop",
        surface=AdversarialSurface.JSON_ESCAPING,
        description="JSON string content can contain raw structural stop delimiters",
        payload={"stop_sequences": ("}", "\\\"", "\n}"), "stop_token_ids": ()},
        expected_rule_ids=("stop-overreachability",),
    )


def _markdown_fence_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="markdown-fence-tool-output-stop",
        surface=AdversarialSurface.MARKDOWN_FENCES,
        description="markdown code fences can terminate structured output regions early",
        payload={"stop_sequences": ("```", "```json"), "stop_token_ids": ()},
        expected_rule_ids=("stop-overreachability",),
    )


def _xml_tool_tag_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="xml-tool-tag-close-in-string",
        surface=AdversarialSurface.XML_TOOL_TAGS,
        description="XML-ish tool close tag is reachable inside string-encoded tool arguments",
        payload={
            "provider": "anthropic",
            "tools": [
                {
                    "name": "emit_xmlish_tool",
                    "input_schema": {"type": "string"},
                }
            ],
        },
        expected_rule_ids=("tool-schema-open-string-arguments",),
    )


def _provider_envelope_case() -> AdversarialCorpusCase:
    return AdversarialCorpusCase(
        case_id="provider-envelope-parallel-tool-drift",
        surface=AdversarialSurface.PROVIDER_ENVELOPES,
        description="provider migration changes parallel tool-call and response-format envelope assumptions",
        payload={
            "source": {
                "provider": "openai",
                "supports_parallel_tool_calls": True,
                "response_format": "json_schema",
                "tool_choice": "auto",
                "stop": ["</tool_call>"],
            },
            "target": {
                "provider": "anthropic",
                "supports_parallel_tool_calls": False,
                "response_format": "text",
                "tool_choice": "none",
                "stop": [],
            },
        },
        expected_rule_ids=("provider-migration",),
    )


def _replay_case(case: AdversarialCorpusCase) -> AdversarialCorpusReplay:
    runners = {
        AdversarialSurface.ROLE_DELIMITERS: _replay_role_delimiter,
        AdversarialSurface.SPECIAL_TOKENS: _run_tokenizer,
        AdversarialSurface.UNICODE_NORMALIZATION: _run_tokenizer,
        AdversarialSurface.JSON_ESCAPING: _run_stop_policy,
        AdversarialSurface.MARKDOWN_FENCES: _run_stop_policy,
        AdversarialSurface.XML_TOOL_TAGS: _replay_xml_tool_tag,
        AdversarialSurface.PROVIDER_ENVELOPES: _replay_provider_envelope,
    }
    observations = runners[case.surface](case.payload)
    observed = tuple(sorted({observation.rule_id for observation in observations if observation.severity != "info"}))
    return AdversarialCorpusReplay(case=case, observed_rule_ids=observed, observations=observations)


def _replay_role_delimiter(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    parsed = parse_hf_chat_template_config(payload)
    report = analyze_role_boundary_nonforgeability(parsed)
    return tuple(
        MutationObservation(
            rule_id="role-boundary-nonforgeability",
            severity="error",
            message=finding.boundary_description,
            evidence=(("field", finding.input_expression), ("marker", finding.marker)),
        )
        for finding in report.findings
    ) or (MutationObservation("role-boundary-stable", "info", "no forged role delimiters"),)


def _replay_xml_tool_tag(payload: object) -> tuple[MutationObservation, ...]:
    try:
        result = ingest_tool_schema_mapping(payload)
    except ToolSchemaIngestionError as exc:
        return (MutationObservation("tool-schema-ingestion-error", "error", str(exc)),)
    observations = [
        MutationObservation(
            rule_id=f"tool-schema-{issue.kind.value}",
            severity="warning",
            message=issue.message,
            evidence=(("path", ".".join(issue.path)),),
        )
        for issue in result.issues
    ]
    if _has_string_tool_arguments(payload):
        observations.append(
            MutationObservation(
                "tool-schema-open-string-arguments",
                "warning",
                "tool arguments are string-encoded, so XML-ish close tags remain reachable inside arguments",
                (("provider", str(result.provider_family.value)),),
            )
        )
    stop_observations = _run_stop_policy({"stop_sequences": ("</tool_call>",), "stop_token_ids": ()})
    observations.extend(stop_observations)
    return tuple(observations) or (MutationObservation("tool-schema-stable", "info", "tool schema is closed"),)


def _replay_provider_envelope(payload: object) -> tuple[MutationObservation, ...]:
    assert isinstance(payload, dict)
    with tempfile.TemporaryDirectory(prefix="promptabi-adversarial-provider-") as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source-provider.json"
        target_path = root / "target-provider.json"
        source_path.write_text(json.dumps(_provider_fixture_payload(payload["source"], target_name="target-provider"), indent=2), encoding="utf-8")
        target_path.write_text(json.dumps(_provider_fixture_payload(payload["target"]), indent=2), encoding="utf-8")
        loader = ArtifactLoader()
        loaded = (
            loader.load(
                ProviderConfigArtifact(
                    kind=ArtifactKind.PROVIDER_CONFIG,
                    name="source-provider",
                    location=ArtifactLocation(path=str(source_path)),
                    provider="openai",
                )
            ),
            loader.load(
                ProviderConfigArtifact(
                    kind=ArtifactKind.PROVIDER_CONFIG,
                    name="target-provider",
                    location=ArtifactLocation(path=str(target_path)),
                    provider="anthropic",
                )
            ),
        )
        report = analyze_provider_migration(loaded)
    return tuple(
        MutationObservation(
            rule_id="provider-migration",
            severity=finding.severity,
            message=finding.message,
            evidence=tuple((key, value) for key, value in finding.evidence),
        )
        for finding in report.findings
    ) or (MutationObservation("provider-migration-stable", "info", "provider envelopes are compatible"),)


def _provider_fixture_payload(value: object, *, target_name: str | None = None) -> dict[str, object]:
    assert isinstance(value, dict)
    provider = str(value["provider"])
    compatibility = {
        "provider_family": provider,
        "request": {"required_fields": ("messages", "tools", "response_format", "tool_choice")},
        "response": {"required_fields": ("content", "tool_calls")},
        "tools": {
            "argument_encoding": "json-object" if provider == "openai" else "content-block-json",
            "id_path": "tool_calls[].id" if provider == "openai" else "content[].id",
            "supports_parallel_tool_calls": bool(value["supports_parallel_tool_calls"]),
        },
        "streaming": {"emits_argument_fragments": provider == "openai"},
        "stops": {"sequences": tuple(value["stop"])},
        "limits": {"max_input_tokens": 128000 if provider == "openai" else 200000, "max_output_tokens": 4096},
        "structured_outputs": {"modes": (str(value["response_format"]),)},
        "errors": {"code_path": "error.code" if provider == "openai" else "error.type"},
    }
    payload: dict[str, object] = {
        "provider": provider,
        "request_shape": compatibility["request"],
        "response_shape": compatibility["response"],
        "streaming_deltas": compatibility["streaming"],
        "migration_compatibility": compatibility,
    }
    if target_name is not None:
        payload["provider_migration"] = {"targets": [target_name]}
    return payload


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _has_string_tool_arguments(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    return any(
        isinstance(tool, dict)
        and isinstance(schema := tool.get("input_schema"), dict)
        and schema.get("type") == "string"
        for tool in tools
    )


def normalize_adversarial_text(text: str, form: str = "NFKC") -> str:
    """Expose the exact Unicode normalization used by adversarial corpus cases."""

    return unicodedata.normalize(form, text)
