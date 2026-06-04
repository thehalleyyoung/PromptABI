"""Structured-output grammar versus application-parser compatibility checks."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, GrammarArtifact, SchemaArtifact
from .grammar_differential import _accepts_sample, _finite_rule_language, _ingest_artifact
from .grammars import GrammarDialect, GrammarIngestionError, GrammarIngestionResult, ingest_grammar_file
from .json_schema import compile_json_schema_mapping
from .source import build_json_source_map


class ParserCompatibilityStatus(StrEnum):
    """Outcome for a bounded parser-compatibility replay."""

    AGREEMENT = "agreement"
    MISMATCH = "mismatch"
    ABSTAINED = "abstained"


class ParserCompatibilityDirection(StrEnum):
    """Which side accepts more than the other for one sample."""

    GRAMMAR_BROADER = "grammar-broader"
    PARSER_BROADER = "parser-broader"


@dataclass(frozen=True, slots=True)
class ParserCompatibilitySample:
    """One concrete structured-output string checked by both models."""

    text: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "source": self.source}


@dataclass(frozen=True, slots=True)
class ParserCompatibilityObservation:
    """Membership results for one concrete sample."""

    sample: ParserCompatibilitySample
    grammar_accepts: bool | None
    parser_accepts: bool | None
    reason: str | None = None

    @property
    def mismatch(self) -> bool:
        return self.grammar_accepts is not None and self.parser_accepts is not None and self.grammar_accepts != self.parser_accepts

    @property
    def direction(self) -> ParserCompatibilityDirection | None:
        if not self.mismatch:
            return None
        if self.grammar_accepts:
            return ParserCompatibilityDirection.GRAMMAR_BROADER
        return ParserCompatibilityDirection.PARSER_BROADER

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "sample": self.sample.to_dict(),
            "grammar_accepts": self.grammar_accepts,
            "parser_accepts": self.parser_accepts,
        }
        if self.direction is not None:
            data["direction"] = self.direction.value
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class ParserCompatibilityReport:
    """Bounded evidence comparing grammar membership to parser acceptance."""

    artifact_name: str
    grammar_kind: str
    parser_format: str
    status: ParserCompatibilityStatus
    observations: tuple[ParserCompatibilityObservation, ...]
    assumptions: tuple[str, ...]
    reason: str | None = None

    @property
    def mismatches(self) -> tuple[ParserCompatibilityObservation, ...]:
        return tuple(observation for observation in self.observations if observation.mismatch)

    @property
    def abstained(self) -> bool:
        return self.status is ParserCompatibilityStatus.ABSTAINED

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "artifact_name": self.artifact_name,
            "grammar_kind": self.grammar_kind,
            "parser_format": self.parser_format,
            "status": self.status.value,
            "assumptions": list(self.assumptions),
            "observations": [observation.to_dict() for observation in self.observations],
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


def analyze_parser_compatibility(
    artifact: SchemaArtifact | GrammarArtifact,
    *,
    max_samples: int = 64,
) -> ParserCompatibilityReport:
    """Compare bounded grammar evidence against an explicitly modeled parser.

    This is deliberately witness-based and heuristic: JSON Schema compilation in
    PromptABI currently yields representative examples, not a complete language.
    Non-JSON application parsers are declared fixture models, so their assumptions
    are carried in the report rather than promoted to sound equivalence claims.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if artifact.location.path is None:
        return _abstain(artifact, "unknown", "parser compatibility requires a local structured-output artifact")

    parser_format = _parser_format(artifact)
    if parser_format is None:
        return _abstain(
            artifact,
            "unknown",
            "non-schema parser compatibility requires metadata.parser_format, metadata.parser_kind, or metadata.application_parser",
        )

    try:
        grammar_model = _load_grammar_model(artifact)
    except (OSError, json.JSONDecodeError, GrammarIngestionError, ValueError) as exc:
        return _abstain(artifact, parser_format, f"could not load grammar model: {exc}")

    samples = _samples(artifact, grammar_model, max_samples=max_samples)
    if not samples:
        return _abstain(artifact, parser_format, "no bounded samples were available for parser compatibility replay")

    observations: list[ParserCompatibilityObservation] = []
    for sample in samples:
        grammar_accepts, grammar_reason = grammar_model.accepts(sample.text)
        parser_accepts, parser_reason = _parser_accepts(parser_format, sample.text, artifact)
        observations.append(
            ParserCompatibilityObservation(
                sample=sample,
                grammar_accepts=grammar_accepts,
                parser_accepts=parser_accepts,
                reason=grammar_reason or parser_reason,
            )
        )

    if any(observation.grammar_accepts is None or observation.parser_accepts is None for observation in observations):
        status = ParserCompatibilityStatus.ABSTAINED
        reason = next(
            observation.reason
            for observation in observations
            if observation.grammar_accepts is None or observation.parser_accepts is None
        )
    elif any(observation.mismatch for observation in observations):
        status = ParserCompatibilityStatus.MISMATCH
        reason = "bounded grammar/parser replay found at least one membership disagreement"
    else:
        status = ParserCompatibilityStatus.AGREEMENT
        reason = None

    return ParserCompatibilityReport(
        artifact_name=artifact.name,
        grammar_kind=grammar_model.kind,
        parser_format=parser_format,
        status=status,
        observations=tuple(observations),
        assumptions=_assumptions(parser_format),
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _GrammarModel:
    kind: str
    accepts: Any
    witnesses: tuple[str, ...]


def _load_grammar_model(artifact: SchemaArtifact | GrammarArtifact) -> _GrammarModel:
    path = Path(artifact.location.path or "")
    if artifact.kind is ArtifactKind.SCHEMA:
        raw, source_map = _load_json_object(path)
        compiled = compile_json_schema_mapping(raw, source_map=source_map)
        if not compiled.supported_fragment:
            reason = ", ".join(issue.code for issue in (*compiled.normalized.issues, *compiled.issues))
            raise ValueError(f"JSON Schema compiler abstained: {reason or 'unsupported fragment'}")
        return _GrammarModel(
            kind="json-schema",
            accepts=lambda text: _json_schema_accepts(raw, text),
            witnesses=(compiled.witness.text,),
        )

    if artifact.kind is not ArtifactKind.GRAMMAR:
        raise ValueError(f"expected schema or grammar artifact, got {artifact.kind.value}")
    ingestion = ingest_grammar_file(path, declared_type=artifact.grammar_type)
    raw_artifact = _raw_grammar_artifact(path, artifact.grammar_type, ingestion)
    return _GrammarModel(
        kind=ingestion.dialect.value,
        accepts=lambda text: _accepts_sample(ingestion, raw_artifact, text, declared_type=artifact.grammar_type),
        witnesses=_grammar_witnesses(ingestion, raw_artifact),
    )


def _load_json_object(path: Path) -> tuple[dict[str, Any], Any]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("structured-output JSON artifact root must be an object")
    return raw, build_json_source_map(text, path)


def _raw_grammar_artifact(path: Path, declared_type: str, ingestion: GrammarIngestionResult) -> Any:
    text = path.read_text(encoding="utf-8")
    if ingestion.dialect is GrammarDialect.REGEX:
        return text
    if ingestion.dialect in {GrammarDialect.EBNF, GrammarDialect.XGRAMMAR} and not text.lstrip().startswith("{"):
        return text
    return json.loads(text)


def _grammar_witnesses(ingestion: GrammarIngestionResult, raw_artifact: Any) -> tuple[str, ...]:
    if ingestion.dialect is GrammarDialect.JSON_SCHEMA and isinstance(raw_artifact, dict):
        try:
            compiled = compile_json_schema_mapping(raw_artifact)
        except ValueError:
            return ()
        return (compiled.witness.text,) if compiled.supported_fragment else ()
    if ingestion.dialect is GrammarDialect.REGEX:
        literals = tuple(terminal.text for terminal in ingestion.terminals if terminal.terminal_type == "literal")
        return literals[:8]
    if ingestion.dialect is GrammarDialect.OUTLINES and isinstance(raw_artifact, dict):
        choices = raw_artifact.get("choices", raw_artifact.get("choice"))
        if isinstance(choices, list) and all(isinstance(item, str) for item in choices):
            return tuple(choices[:8])
    if ingestion.dialect in {GrammarDialect.EBNF, GrammarDialect.XGRAMMAR, GrammarDialect.LLGUIDANCE, GrammarDialect.PROMPTABI}:
        language, _reason = _finite_rule_language(ingestion)
        if language is not None:
            return tuple(sorted(language)[:8])
    return ()


def _parser_format(artifact: SchemaArtifact | GrammarArtifact) -> str | None:
    metadata = dict(artifact.metadata)
    for key in ("parser_format", "parser_kind", "application_parser"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace("_", "-")
    if artifact.kind is ArtifactKind.SCHEMA and artifact.dialect.lower().replace("_", "-") in {
        "json-schema",
        "jsonschema",
        "json-schema-2020-12",
    }:
        return "json-schema"
    return None


def _samples(
    artifact: SchemaArtifact | GrammarArtifact,
    grammar_model: _GrammarModel,
    *,
    max_samples: int,
) -> tuple[ParserCompatibilitySample, ...]:
    samples: list[ParserCompatibilitySample] = []
    for witness in grammar_model.witnesses:
        samples.append(ParserCompatibilitySample(witness, "grammar-witness"))
    for text in _metadata_samples(artifact):
        samples.append(ParserCompatibilitySample(text, "metadata-sample"))
    if artifact.kind is ArtifactKind.SCHEMA:
        samples.extend(
            ParserCompatibilitySample(text, "json-parser-stress")
            for text in ('{}', '[]', 'null', '"string"', '{"unexpected":true}')
        )
    deduped: dict[str, ParserCompatibilitySample] = {}
    for sample in samples:
        deduped.setdefault(sample.text, sample)
    return tuple(deduped.values())[:max_samples]


def _metadata_samples(artifact: SchemaArtifact | GrammarArtifact) -> tuple[str, ...]:
    metadata = dict(artifact.metadata)
    raw = metadata.get("parser_compatibility", metadata.get("parser_compatibility_samples", metadata.get("samples")))
    values: list[str] = []
    if isinstance(raw, list):
        values.extend(item for item in raw if isinstance(item, str))
    elif isinstance(raw, dict):
        for key in ("samples", "accepts", "rejects"):
            item = raw.get(key)
            if isinstance(item, list):
                values.extend(text for text in item if isinstance(text, str))
    return tuple(values)


def _parser_accepts(
    parser_format: str,
    text: str,
    artifact: SchemaArtifact | GrammarArtifact,
) -> tuple[bool | None, str | None]:
    if parser_format in {"json", "json-parser", "json-loads"}:
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return False, None
        return True, None
    if parser_format in {"json-schema", "jsonschema", "json-schema-validator"}:
        if artifact.kind is not ArtifactKind.SCHEMA:
            return None, "json-schema parser compatibility requires a schema artifact"
        raw, _source_map = _load_json_object(Path(artifact.location.path or ""))
        return _json_schema_accepts(raw, text)
    if parser_format in {"xml-tool-call", "xml-tool", "xml"}:
        return _xml_tool_call_accepts(text), None
    if parser_format in {"markdown-fence", "markdown-code-fence", "fence"}:
        return _markdown_fence_accepts(text, artifact), None
    if parser_format in {"custom-delimited", "custom-delimiter", "delimiter"}:
        return _custom_delimited_accepts(text, artifact)
    return None, f"unsupported parser_format '{parser_format}'"


def _json_schema_accepts(schema: dict[str, Any], text: str) -> tuple[bool | None, str | None]:
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return None, "jsonschema is not installed, so JSON Schema parser compatibility abstained"
    try:
        instance = json.loads(text)
    except json.JSONDecodeError:
        return False, None
    try:
        jsonschema.Draft202012Validator(schema).validate(instance)
    except jsonschema.ValidationError:
        return False, None
    return True, None


def _xml_tool_call_accepts(text: str) -> bool:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return False
    if root.tag not in {"tool_call", "tool", "function_call"}:
        return False
    name = root.attrib.get("name") or root.attrib.get("tool") or root.attrib.get("function")
    if name is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", name):
        return False
    return bool((root.text or "").strip()) or bool(root.attrib)


_FENCE_RE = re.compile(r"\A```(?P<lang>[A-Za-z0-9_.+-]*)[ \t]*\n(?P<body>.*)\n```\s*\Z", re.DOTALL)


def _markdown_fence_accepts(text: str, artifact: SchemaArtifact | GrammarArtifact) -> bool:
    match = _FENCE_RE.fullmatch(text)
    if match is None:
        return False
    metadata = dict(artifact.metadata)
    language = metadata.get("fence_language")
    if isinstance(language, str) and language and match.group("lang") != language:
        return False
    payload_format = metadata.get("payload_format")
    if payload_format == "json":
        try:
            json.loads(match.group("body"))
        except json.JSONDecodeError:
            return False
    return True


def _custom_delimited_accepts(text: str, artifact: SchemaArtifact | GrammarArtifact) -> tuple[bool | None, str | None]:
    metadata = dict(artifact.metadata)
    start = metadata.get("start_delimiter", metadata.get("open_delimiter"))
    end = metadata.get("end_delimiter", metadata.get("close_delimiter"))
    if not isinstance(start, str) or not start or not isinstance(end, str) or not end:
        return None, "custom-delimited parser requires metadata.start_delimiter and metadata.end_delimiter"
    if not text.startswith(start) or not text.endswith(end) or len(text) <= len(start) + len(end):
        return False, None
    payload = text[len(start) : len(text) - len(end)]
    if metadata.get("payload_format") == "json":
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            return False, None
    return True, None


def _assumptions(parser_format: str) -> tuple[str, ...]:
    return (
        "bounded-witness-replay",
        "full-string-membership",
        f"declared-parser-format:{parser_format}",
        "heuristic-not-language-equivalence",
    )


def _abstain(
    artifact: SchemaArtifact | GrammarArtifact,
    parser_format: str,
    reason: str,
) -> ParserCompatibilityReport:
    grammar_kind = artifact.dialect if artifact.kind is ArtifactKind.SCHEMA else artifact.grammar_type
    return ParserCompatibilityReport(
        artifact_name=artifact.name,
        grammar_kind=grammar_kind,
        parser_format=parser_format,
        status=ParserCompatibilityStatus.ABSTAINED,
        observations=(),
        assumptions=_assumptions(parser_format),
        reason=reason,
    )
