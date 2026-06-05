"""CPU-only differential fixtures for structured-output grammar semantics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .grammars import (
    GrammarDialect,
    GrammarIngestionError,
    GrammarIngestionResult,
    ingest_grammar_mapping,
    ingest_grammar_text,
)


class GrammarDifferentialStatus(StrEnum):
    """Outcome for one recorded backend-semantics fixture."""

    AGREEMENT = "agreement"
    MISMATCH = "mismatch"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class GrammarDifferentialSample:
    """One hand-labeled membership sample from a backend fixture."""

    text: str
    expected_accepts: bool

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "expected_accepts": self.expected_accepts}


@dataclass(frozen=True, slots=True)
class GrammarDifferentialObservation:
    """PromptABI's local membership result for one backend-labeled sample."""

    sample: GrammarDifferentialSample
    promptabi_accepts: bool | None
    reason: str | None = None

    @property
    def mismatch(self) -> bool:
        return self.promptabi_accepts is not None and self.promptabi_accepts != self.sample.expected_accepts

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "sample": self.sample.to_dict(),
            "promptabi_accepts": self.promptabi_accepts,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class GrammarDifferentialCaseReport:
    """Differential result for one backend fixture case."""

    case_id: str
    backend_family: str
    declared_type: str
    status: GrammarDifferentialStatus
    observations: tuple[GrammarDifferentialObservation, ...]
    assumptions: tuple[str, ...]
    features: tuple[str, ...] = ()
    terminals: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def mismatches(self) -> tuple[GrammarDifferentialObservation, ...]:
        return tuple(observation for observation in self.observations if observation.mismatch)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "case_id": self.case_id,
            "backend_family": self.backend_family,
            "declared_type": self.declared_type,
            "status": self.status.value,
            "assumptions": list(self.assumptions),
            "features": list(self.features),
            "terminals": list(self.terminals),
            "observations": [observation.to_dict() for observation in self.observations],
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class GrammarDifferentialReport:
    """Report for a recorded structured-output backend fixture corpus."""

    version: int
    cases: tuple[GrammarDifferentialCaseReport, ...]

    @property
    def mismatches(self) -> tuple[GrammarDifferentialCaseReport, ...]:
        return tuple(case for case in self.cases if case.status is GrammarDifferentialStatus.MISMATCH)

    @property
    def abstentions(self) -> tuple[GrammarDifferentialCaseReport, ...]:
        return tuple(case for case in self.cases if case.status is GrammarDifferentialStatus.ABSTAINED)

    @property
    def agreements(self) -> tuple[GrammarDifferentialCaseReport, ...]:
        return tuple(case for case in self.cases if case.status is GrammarDifferentialStatus.AGREEMENT)

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "cases": [case.to_dict() for case in self.cases]}


def analyze_grammar_differential_corpus(path: str | Path) -> GrammarDifferentialReport:
    """Compare PromptABI grammar semantics against hand-labeled backend fixtures.

    The corpus is intentionally offline: each sample is a recorded or reduced
    backend membership label, while PromptABI evaluates the same sample with the
    local supported-fragment implementation. Optional third-party backends are
    not required for normal CI; unsupported fragments become abstentions.
    """

    corpus_path = Path(path)
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GrammarIngestionError("grammar differential corpus root must be an object")
    return analyze_grammar_differential_mapping(raw)


def analyze_grammar_differential_mapping(raw: dict[str, Any]) -> GrammarDifferentialReport:
    """Analyze an already parsed grammar differential corpus."""

    version = raw.get("version")
    if not isinstance(version, int) or version <= 0:
        raise GrammarIngestionError("grammar differential corpus requires a positive integer version")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise GrammarIngestionError("grammar differential corpus requires a non-empty cases array")
    return GrammarDifferentialReport(
        version=version,
        cases=tuple(_analyze_case(index, case) for index, case in enumerate(cases)),
    )


def _analyze_case(index: int, raw_case: Any) -> GrammarDifferentialCaseReport:
    if not isinstance(raw_case, dict):
        raise GrammarIngestionError(f"grammar differential case {index} must be an object")
    case_id = _required_str(raw_case, "id")
    backend_family = _required_str(raw_case, "backend_family")
    declared_type = _required_str(raw_case, "declared_type")
    artifact = raw_case.get("artifact")
    samples = _samples(raw_case)

    try:
        ingestion = _ingest_artifact(artifact, declared_type=declared_type)
    except GrammarIngestionError as exc:
        return GrammarDifferentialCaseReport(
            case_id=case_id,
            backend_family=backend_family,
            declared_type=declared_type,
            status=GrammarDifferentialStatus.ABSTAINED,
            observations=(),
            assumptions=("recorded-backend-membership-labels", "promptabi-ingestion"),
            reason=str(exc),
        )

    observations: list[GrammarDifferentialObservation] = []
    for sample in samples:
        promptabi_accepts, reason = _accepts_sample(
            ingestion,
            artifact,
            sample.text,
            declared_type=declared_type,
        )
        observations.append(
            GrammarDifferentialObservation(
                sample=sample,
                promptabi_accepts=promptabi_accepts,
                reason=reason,
            )
        )

    if any(observation.promptabi_accepts is None for observation in observations):
        status = GrammarDifferentialStatus.ABSTAINED
        reason = next(
            observation.reason for observation in observations if observation.promptabi_accepts is None
        )
    elif any(observation.mismatch for observation in observations):
        status = GrammarDifferentialStatus.MISMATCH
        reason = "PromptABI local semantics disagreed with at least one recorded backend label"
    else:
        status = GrammarDifferentialStatus.AGREEMENT
        reason = None

    return GrammarDifferentialCaseReport(
        case_id=case_id,
        backend_family=backend_family,
        declared_type=declared_type,
        status=status,
        observations=tuple(observations),
        assumptions=(
            "recorded-backend-membership-labels",
            "full-string-membership",
            "cpu-only-supported-fragment",
        ),
        features=ingestion.features,
        terminals=ingestion.terminal_texts,
        reason=reason,
    )


def _ingest_artifact(artifact: Any, *, declared_type: str) -> GrammarIngestionResult:
    if isinstance(artifact, str):
        return ingest_grammar_text(artifact, declared_type=declared_type)
    if isinstance(artifact, dict):
        return ingest_grammar_mapping(artifact, declared_type=declared_type)
    raise GrammarIngestionError("grammar differential artifact must be a string or object")


def _samples(raw_case: dict[str, Any]) -> tuple[GrammarDifferentialSample, ...]:
    accepts = raw_case.get("accepts")
    rejects = raw_case.get("rejects")
    if not isinstance(accepts, list) or not accepts or not all(isinstance(item, str) for item in accepts):
        raise GrammarIngestionError(f"grammar differential case '{raw_case.get('id')}' requires accepted string samples")
    if not isinstance(rejects, list) or not rejects or not all(isinstance(item, str) for item in rejects):
        raise GrammarIngestionError(f"grammar differential case '{raw_case.get('id')}' requires rejected string samples")
    return tuple(
        [*(GrammarDifferentialSample(text=item, expected_accepts=True) for item in accepts)]
        + [*(GrammarDifferentialSample(text=item, expected_accepts=False) for item in rejects)]
    )


def _accepts_sample(
    ingestion: GrammarIngestionResult,
    artifact: Any,
    sample: str,
    *,
    declared_type: str,
) -> tuple[bool | None, str | None]:
    if not ingestion.supported_fragment:
        return None, f"grammar ingestion abstained with issues: {', '.join(issue.code for issue in ingestion.issues)}"

    dialect = ingestion.dialect
    if dialect is GrammarDialect.JSON_SCHEMA:
        if not isinstance(artifact, dict):
            return None, "JSON Schema semantics require an object artifact"
        return _json_schema_accepts(artifact, sample)
    if dialect is GrammarDialect.REGEX:
        pattern = artifact.get("regex") if isinstance(artifact, dict) else artifact
        if not isinstance(pattern, str):
            return None, "regex semantics require a string pattern"
        return re.fullmatch(pattern, sample) is not None, None
    if dialect is GrammarDialect.OUTLINES:
        return _outlines_accepts(artifact, sample)
    if dialect is GrammarDialect.LLGUIDANCE:
        return _llguidance_accepts(artifact, sample)
    if dialect in {GrammarDialect.EBNF, GrammarDialect.XGRAMMAR, GrammarDialect.PROMPTABI}:
        language, reason = _finite_rule_language(ingestion)
        if language is None:
            return None, reason
        return sample in language, None
    return None, f"no local differential semantics for declared type {declared_type!r}"


def _outlines_accepts(artifact: Any, sample: str) -> tuple[bool | None, str | None]:
    if not isinstance(artifact, dict):
        return None, "Outlines semantics require an object fixture"
    schema = artifact.get("json_schema", artifact.get("schema"))
    if isinstance(schema, dict):
        return _json_schema_accepts(schema, sample)
    if isinstance(artifact.get("regex"), str):
        return re.fullmatch(artifact["regex"], sample) is not None, None
    choices = artifact.get("choices", artifact.get("choice"))
    if isinstance(choices, list) and all(isinstance(item, str) for item in choices):
        return sample in choices, None
    return None, "Outlines fixture is outside JSON Schema, regex, or choices semantics"


def _llguidance_accepts(artifact: Any, sample: str) -> tuple[bool | None, str | None]:
    if not isinstance(artifact, dict):
        return None, "llguidance semantics require an object fixture"
    if isinstance(artifact.get("json_schema"), dict):
        return _json_schema_accepts(artifact["json_schema"], sample)
    if isinstance(artifact.get("regex"), str):
        return re.fullmatch(artifact["regex"], sample) is not None, None
    if isinstance(artifact.get("lark_grammar"), str) or isinstance(artifact.get("grammar"), str):
        ingestion = _ingest_artifact(artifact, declared_type="llguidance")
        language, reason = _finite_rule_language(_rules_with_start_alias(ingestion))
        if language is None:
            return None, reason
        return sample in language, None
    return None, "llguidance fixture is outside JSON Schema, regex, or finite grammar semantics"


def _json_schema_accepts(schema: dict[str, Any], sample: str) -> tuple[bool | None, str | None]:
    try:
        instance = json.loads(sample)
    except json.JSONDecodeError:
        return False, None
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return _json_schema_accepts_supported_subset(schema, instance), None
    try:
        jsonschema.Draft202012Validator(schema).validate(instance)
    except jsonschema.ValidationError:
        return False, None
    return True, None


def _json_schema_accepts_supported_subset(schema: dict[str, Any], instance: Any) -> bool:
    """Evaluate the small JSON Schema fragment used by offline conformance fixtures."""

    if "oneOf" in schema:
        options = schema["oneOf"]
        if not isinstance(options, list):
            return False
        return sum(
            1
            for option in options
            if isinstance(option, dict) and _json_schema_accepts_supported_subset(option, instance)
        ) == 1
    if "enum" in schema:
        enum_values = schema["enum"]
        return isinstance(enum_values, list) and instance in enum_values
    if "const" in schema and instance != schema["const"]:
        return False

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return any(
            _json_schema_accepts_supported_subset({**schema, "type": item}, instance)
            for item in schema_type
            if isinstance(item, str)
        )
    if isinstance(schema_type, str) and not _matches_json_type(instance, schema_type):
        return False

    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        if not isinstance(instance, dict):
            return False
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            return False
        if any(key not in instance for key in required):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in instance):
            return False
        for key, subschema in properties.items():
            if key in instance and isinstance(subschema, dict):
                if not _json_schema_accepts_supported_subset(subschema, instance[key]):
                    return False
        return True

    if schema_type == "array":
        if not isinstance(instance, list):
            return False
        items = schema.get("items")
        if isinstance(items, dict):
            return all(_json_schema_accepts_supported_subset(items, item) for item in instance)
        return True

    if schema_type == "string":
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            return False
    if schema_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            return False
    return True


def _matches_json_type(instance: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(instance, dict)
    if schema_type == "array":
        return isinstance(instance, list)
    if schema_type == "string":
        return isinstance(instance, str)
    if schema_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if schema_type == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if schema_type == "boolean":
        return isinstance(instance, bool)
    if schema_type == "null":
        return instance is None
    return False


_LITERAL_OR_REF_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'|([A-Za-z_][A-Za-z0-9_-]*)')


def _finite_rule_language(
    ingestion: GrammarIngestionResult,
    *,
    max_depth: int = 8,
    max_values: int = 64,
) -> tuple[frozenset[str] | None, str | None]:
    rules = {rule.name: rule.expression for rule in ingestion.rules}
    if not rules:
        return None, "finite grammar semantics require at least one rule"
    start = ingestion.start_symbol or ingestion.rules[0].name

    def expand(name: str, stack: tuple[str, ...]) -> frozenset[str] | None:
        if name in stack or len(stack) > max_depth:
            return None
        expression = rules.get(name)
        if expression is None:
            return None
        values: set[str] = set()
        for alternative in _split_top_level_alternatives(expression):
            parts = _parts(alternative)
            if parts is None:
                return None
            expanded_parts: list[frozenset[str]] = []
            for kind, value in parts:
                if kind == "literal":
                    expanded_parts.append(frozenset({value}))
                    continue
                child = expand(value, (*stack, name))
                if child is None:
                    return None
                expanded_parts.append(child)
            values.update(_concat(expanded_parts, max_values=max_values))
            if len(values) > max_values:
                return frozenset(sorted(values)[:max_values])
        return frozenset(values)

    language = expand(start, ())
    if language is None:
        return None, "grammar fixture uses recursion, regex terminals, undefined references, or unsupported operators"
    return language, None


def _rules_with_start_alias(ingestion: GrammarIngestionResult) -> GrammarIngestionResult:
    if not ingestion.rules:
        return ingestion
    rule_names = {rule.name for rule in ingestion.rules}
    if ingestion.start_symbol is not None and ingestion.start_symbol in rule_names:
        return ingestion
    first_rule = ingestion.rules[0]
    if first_rule.name in {"start", "root"}:
        return GrammarIngestionResult(
            dialect=ingestion.dialect,
            declared_type=ingestion.declared_type,
            start_symbol=first_rule.name,
            rules=ingestion.rules,
            terminals=ingestion.terminals,
            references=ingestion.references,
            features=ingestion.features,
            issues=ingestion.issues,
            source_spans=ingestion.source_spans,
        )
    return ingestion


def _split_top_level_alternatives(expression: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in expression.split("|") if part.strip())


def _parts(expression: str) -> tuple[tuple[str, str], ...] | None:
    parts: list[tuple[str, str]] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _LITERAL_OR_REF_RE.match(expression, position)
        if match is None:
            return None
        literal = match.group(1) if match.group(1) is not None else match.group(2)
        if literal is not None:
            parts.append(("literal", bytes(literal, "utf-8").decode("unicode_escape")))
        else:
            parts.append(("ref", match.group(3)))
        position = match.end()
    return tuple(parts)


def _concat(parts: list[frozenset[str]], *, max_values: int) -> frozenset[str]:
    values = {""}
    for part in parts:
        values = {left + right for left in values for right in part}
        if len(values) > max_values:
            values = set(sorted(values)[:max_values])
    return frozenset(values)


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise GrammarIngestionError(f"grammar differential case field '{key}' must be a non-empty string")
    return value
