"""Grammar ingestion for structured-output and constrained-decoding artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .diagnostics import SourceSpan
from .source import JsonSourceMap, build_json_source_map


class GrammarDialect(StrEnum):
    """Grammar families normalized by the ingestion layer."""

    JSON_SCHEMA = "json-schema"
    REGEX = "regex"
    EBNF = "ebnf"
    OUTLINES = "outlines"
    XGRAMMAR = "xgrammar"
    LLGUIDANCE = "llguidance"
    PROMPTABI = "promptabi"


@dataclass(frozen=True, slots=True)
class GrammarRule:
    """A normalized grammar rule declaration."""

    name: str
    expression: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("grammar rule name must be non-empty")


@dataclass(frozen=True, slots=True)
class GrammarTerminal:
    """A literal or regex terminal discovered during ingestion."""

    text: str
    terminal_type: str = "literal"
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.terminal_type:
            raise ValueError("grammar terminal type must be non-empty")


@dataclass(frozen=True, slots=True)
class GrammarIngestionIssue:
    """A non-fatal limitation or semantic issue found while ingesting a grammar."""

    code: str
    message: str
    severity: str = "warning"
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("grammar issues require a code and message")
        if self.severity not in {"warning", "abstention"}:
            raise ValueError("grammar issue severity must be warning or abstention")

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (("code", self.code), ("message", self.message), ("severity", self.severity))


@dataclass(frozen=True, slots=True)
class GrammarIngestionResult:
    """Typed summary of an ingested grammar artifact."""

    dialect: GrammarDialect
    declared_type: str
    start_symbol: str | None = None
    rules: tuple[GrammarRule, ...] = ()
    terminals: tuple[GrammarTerminal, ...] = ()
    references: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    issues: tuple[GrammarIngestionIssue, ...] = ()
    source_spans: tuple[tuple[str, SourceSpan], ...] = ()

    @property
    def supported_fragment(self) -> bool:
        return not any(issue.severity == "abstention" for issue in self.issues)

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(sorted(dict.fromkeys(rule.name for rule in self.rules)))

    @property
    def terminal_texts(self) -> tuple[str, ...]:
        return tuple(terminal.text for terminal in self.terminals if terminal.text)

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("declared_type", self.declared_type),
            ("dialect", self.dialect.value),
            ("features", self.features),
            ("issue_count", len(self.issues)),
            ("issue_codes", tuple(issue.code for issue in self.issues)),
            ("reference_count", len(self.references)),
            ("references", self.references),
            ("rule_count", len(self.rules)),
            ("rule_names", self.rule_names),
            ("start_symbol", self.start_symbol),
            ("supported_fragment", self.supported_fragment),
            ("terminal_count", len(self.terminals)),
            ("terminals", self.terminal_texts),
        )


class GrammarIngestionError(ValueError):
    """Raised when a grammar artifact cannot be ingested at all."""

    def __init__(self, message: str, *, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.span = span


def ingest_grammar_file(path: str | Path, *, declared_type: str = "promptabi") -> GrammarIngestionResult:
    """Ingest a local grammar file using its declared type plus deterministic heuristics."""

    grammar_path = Path(path)
    text = grammar_path.read_text(encoding="utf-8")
    return ingest_grammar_text(text, declared_type=declared_type, path=str(grammar_path))


def ingest_grammar_text(
    text: str,
    *,
    declared_type: str = "promptabi",
    path: str | None = None,
) -> GrammarIngestionResult:
    """Normalize JSON Schema, regex, EBNF, Outlines, xgrammar, llguidance, and PromptABI grammars."""

    normalized_type = _normalize_type(declared_type)
    if normalized_type is GrammarDialect.REGEX:
        return _ingest_regex(text, declared_type=declared_type, path=path)
    if normalized_type in {GrammarDialect.EBNF, GrammarDialect.XGRAMMAR} and not _looks_like_json(text):
        return _ingest_ebnf(text, dialect=normalized_type, declared_type=declared_type, path=path)

    raw, source_map = _parse_json_document(text, path)
    if not isinstance(raw, dict):
        raise GrammarIngestionError("grammar JSON root must be an object", span=source_map.span_for(()))
    return ingest_grammar_mapping(raw, declared_type=declared_type, source_map=source_map)


def ingest_grammar_mapping(
    raw: dict[str, Any],
    *,
    declared_type: str = "promptabi",
    source_map: JsonSourceMap | None = None,
) -> GrammarIngestionResult:
    """Normalize an already-parsed grammar object."""

    dialect = _detect_dialect(raw, declared_type)
    if dialect is GrammarDialect.JSON_SCHEMA:
        return _ingest_json_schema(raw, declared_type=declared_type, source_map=source_map)
    if dialect is GrammarDialect.OUTLINES:
        return _ingest_outlines(raw, declared_type=declared_type, source_map=source_map)
    if dialect is GrammarDialect.XGRAMMAR:
        return _ingest_xgrammar(raw, declared_type=declared_type, source_map=source_map)
    if dialect is GrammarDialect.LLGUIDANCE:
        return _ingest_llguidance(raw, declared_type=declared_type, source_map=source_map)
    if dialect is GrammarDialect.PROMPTABI:
        return _ingest_promptabi(raw, declared_type=declared_type, source_map=source_map)
    if dialect is GrammarDialect.REGEX:
        pattern = _required_str(raw, "regex", source_map)
        return _ingest_regex(pattern, declared_type=declared_type, path=source_map.path if source_map else None)
    raise GrammarIngestionError(f"unsupported grammar dialect: {dialect.value}")


def ingest_json_schema_mapping(
    raw: dict[str, Any],
    *,
    declared_type: str = "json-schema",
    source_map: JsonSourceMap | None = None,
) -> GrammarIngestionResult:
    """Ingest a JSON Schema artifact for later normalization and compilation steps."""

    return _ingest_json_schema(raw, declared_type=declared_type, source_map=source_map)


def _ingest_json_schema(
    raw: dict[str, Any],
    *,
    declared_type: str,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    supported_keywords = {
        "$schema",
        "$id",
        "additionalProperties",
        "const",
        "description",
        "enum",
        "format",
        "items",
        "maxItems",
        "maximum",
        "maxLength",
        "minItems",
        "minimum",
        "minLength",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
    unsupported = sorted(_json_schema_keywords(raw) - supported_keywords)
    issues = tuple(
        GrammarIngestionIssue(
            code="json-schema-unsupported-keyword",
            message=f"JSON Schema keyword '{keyword}' is outside the ingestion fragment",
            severity="abstention",
            span=_span(source_map, keyword),
        )
        for keyword in unsupported
    )
    properties = raw.get("properties", {})
    rules: list[GrammarRule] = [GrammarRule("schema", json.dumps(_shape_preview(raw), sort_keys=True), _span(source_map, ()))]
    if isinstance(properties, dict):
        for name, value in sorted(properties.items()):
            if isinstance(name, str):
                rules.append(
                    GrammarRule(
                        f"property:{name}",
                        json.dumps(_shape_preview(value), sort_keys=True),
                        _span(source_map, ("properties", name)),
                    )
                )
    terminals = _schema_terminals(raw, source_map)
    features = sorted(key for key in supported_keywords if key in _json_schema_keywords(raw))
    return GrammarIngestionResult(
        dialect=GrammarDialect.JSON_SCHEMA,
        declared_type=declared_type,
        start_symbol="schema",
        rules=tuple(rules),
        terminals=tuple(terminals),
        features=tuple(features),
        issues=issues,
        source_spans=_source_spans(source_map),
    )


def _ingest_regex(text: str, *, declared_type: str, path: str | None) -> GrammarIngestionResult:
    try:
        re.compile(text)
    except re.error as exc:
        raise GrammarIngestionError(f"regex grammar is not valid: {exc}") from exc
    issues = tuple(_regex_issues(text, path))
    terminals = tuple(GrammarTerminal(literal, "literal", _text_span(path)) for literal in _regex_literals(text))
    features = tuple(sorted(_regex_features(text)))
    return GrammarIngestionResult(
        dialect=GrammarDialect.REGEX,
        declared_type=declared_type,
        start_symbol="regex",
        rules=(GrammarRule("regex", text, _text_span(path)),),
        terminals=terminals,
        features=features,
        issues=issues,
        source_spans=(("regex", _text_span(path)),) if path else (),
    )


_EBNF_RULE_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\s*(?P<op>::=|=|:)\s*(?P<body>.*?)(?:;\s*)?$")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_-]*\b")
_TERMINAL_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"|/([^/\\]*(?:\\.[^/\\]*)*)/")


def _ingest_ebnf(
    text: str,
    *,
    dialect: GrammarDialect,
    declared_type: str,
    path: str | None,
) -> GrammarIngestionResult:
    rules: list[GrammarRule] = []
    terminals: list[GrammarTerminal] = []
    references: set[str] = set()
    spans: list[tuple[str, SourceSpan]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        match = _EBNF_RULE_RE.match(line)
        if match is None:
            raise GrammarIngestionError(
                f"EBNF grammar line {line_number} is not a supported rule declaration",
                span=_line_span(path, line_number, line),
            )
        name = match.group("name")
        body = match.group("body").strip()
        span = _line_span(path, line_number, line, start=match.start("name") + 1, end=match.end("name"))
        spans.append((f"rules.{name}", span))
        rules.append(GrammarRule(name, body, span))
        for terminal_match in _TERMINAL_RE.finditer(body):
            terminal_text = next(group for group in terminal_match.groups() if group is not None)
            terminal_kind = "regex" if terminal_match.group(3) is not None else "literal"
            terminals.append(GrammarTerminal(terminal_text, terminal_kind, span))
        without_terminals = _TERMINAL_RE.sub(" ", body)
        references.update(identifier for identifier in _IDENT_RE.findall(without_terminals) if identifier != name)
    if not rules:
        raise GrammarIngestionError("EBNF grammar must define at least one rule", span=_text_span(path))
    rule_names = {rule.name for rule in rules}
    undefined = tuple(sorted(references - rule_names))
    issues = tuple(
        GrammarIngestionIssue(
            code="grammar-undefined-reference",
            message=f"Rule reference '{name}' has no declaration in this artifact",
            severity="warning",
        )
        for name in undefined
    )
    return GrammarIngestionResult(
        dialect=dialect if dialect is GrammarDialect.XGRAMMAR else GrammarDialect.EBNF,
        declared_type=declared_type,
        start_symbol=rules[0].name,
        rules=tuple(rules),
        terminals=tuple(terminals),
        references=tuple(sorted(references)),
        features=("iso-ebnf", "w3c-ebnf"),
        issues=issues,
        source_spans=tuple(spans),
    )


def _ingest_outlines(
    raw: dict[str, Any],
    *,
    declared_type: str,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    if "json_schema" in raw or "schema" in raw:
        schema = raw.get("json_schema", raw.get("schema"))
        if not isinstance(schema, dict):
            raise GrammarIngestionError("Outlines schema grammar must contain an object schema", span=_span(source_map, "schema"))
        nested = _ingest_json_schema(schema, declared_type="json-schema", source_map=None)
        return _wrap_nested(GrammarDialect.OUTLINES, declared_type, "outlines", nested, source_map, ("json-schema",))
    if "regex" in raw:
        pattern = _required_str(raw, "regex", source_map)
        nested = _ingest_regex(pattern, declared_type="regex", path=source_map.path if source_map else None)
        return _wrap_nested(GrammarDialect.OUTLINES, declared_type, "outlines", nested, source_map, ("regex",))
    choices = raw.get("choices", raw.get("choice"))
    if isinstance(choices, list) and all(isinstance(item, str) and item for item in choices):
        terminals = tuple(GrammarTerminal(item, "choice", _span(source_map, "choices")) for item in choices)
        return GrammarIngestionResult(
            dialect=GrammarDialect.OUTLINES,
            declared_type=declared_type,
            start_symbol="choice",
            rules=(GrammarRule("choice", " | ".join(json.dumps(item) for item in choices), _span(source_map, "choices")),),
            terminals=terminals,
            features=("choice",),
            source_spans=_source_spans(source_map),
        )
    raise GrammarIngestionError("Outlines grammar must contain json_schema, schema, regex, choices, or choice")


def _ingest_xgrammar(
    raw: dict[str, Any],
    *,
    declared_type: str,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    grammar = raw.get("grammar", raw.get("bnf"))
    if isinstance(grammar, str):
        nested = _ingest_ebnf(grammar, dialect=GrammarDialect.XGRAMMAR, declared_type=declared_type, path=source_map.path if source_map else None)
        root = raw.get("root_rule") or raw.get("root")
        if isinstance(root, str) and root:
            nested = _replace_start(nested, root)
        return nested
    rules_obj = raw.get("rules")
    if isinstance(rules_obj, dict):
        return _rules_mapping_result(
            rules_obj,
            dialect=GrammarDialect.XGRAMMAR,
            declared_type=declared_type,
            start_symbol=_optional_start(raw, "root_rule", "root", "start"),
            source_map=source_map,
        )
    raise GrammarIngestionError("xgrammar grammar must contain a grammar string or rules object", span=_span(source_map, "grammar"))


def _ingest_llguidance(
    raw: dict[str, Any],
    *,
    declared_type: str,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    if isinstance(raw.get("json_schema"), dict):
        nested = _ingest_json_schema(raw["json_schema"], declared_type="json-schema", source_map=None)
        return _wrap_nested(GrammarDialect.LLGUIDANCE, declared_type, "llguidance", nested, source_map, ("json-schema",))
    if isinstance(raw.get("regex"), str):
        nested = _ingest_regex(raw["regex"], declared_type="regex", path=source_map.path if source_map else None)
        return _wrap_nested(GrammarDialect.LLGUIDANCE, declared_type, "llguidance", nested, source_map, ("regex",))
    grammar = raw.get("lark_grammar", raw.get("grammar"))
    if isinstance(grammar, str):
        nested = _ingest_ebnf(grammar, dialect=GrammarDialect.EBNF, declared_type=declared_type, path=source_map.path if source_map else None)
        return _wrap_nested(GrammarDialect.LLGUIDANCE, declared_type, "llguidance", nested, source_map, ("ebnf",))
    grammars = raw.get("grammars")
    if isinstance(grammars, dict):
        return _rules_mapping_result(
            grammars,
            dialect=GrammarDialect.LLGUIDANCE,
            declared_type=declared_type,
            start_symbol=_optional_start(raw, "start", "root"),
            source_map=source_map,
        )
    raise GrammarIngestionError("llguidance grammar must contain json_schema, regex, lark_grammar, grammar, or grammars")


def _ingest_promptabi(
    raw: dict[str, Any],
    *,
    declared_type: str,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    rules = raw.get("rules")
    if not isinstance(rules, dict):
        raise GrammarIngestionError("PromptABI grammar must contain a rules object", span=_span(source_map, "rules"))
    start = _optional_start(raw, "start", "start_symbol")
    result = _rules_mapping_result(
        rules,
        dialect=GrammarDialect.PROMPTABI,
        declared_type=declared_type,
        start_symbol=start,
        source_map=source_map,
    )
    terminals = raw.get("terminals")
    extra_terminals: tuple[GrammarTerminal, ...] = ()
    if isinstance(terminals, dict):
        extra_terminals = tuple(
            GrammarTerminal(str(value), name, _span(source_map, ("terminals", name)))
            for name, value in sorted(terminals.items())
            if isinstance(name, str) and isinstance(value, str)
        )
    issues = list(result.issues)
    if start is not None and start not in result.rule_names:
        issues.append(
            GrammarIngestionIssue(
                code="grammar-undefined-start",
                message=f"Start rule '{start}' has no declaration in this PromptABI grammar",
                severity="warning",
                span=_span(source_map, "start"),
            )
        )
    return GrammarIngestionResult(
        dialect=result.dialect,
        declared_type=result.declared_type,
        start_symbol=start or result.start_symbol,
        rules=result.rules,
        terminals=(*result.terminals, *extra_terminals),
        references=result.references,
        features=("promptabi-rules",),
        issues=tuple(issues),
        source_spans=result.source_spans or _source_spans(source_map),
    )


def _rules_mapping_result(
    rules_obj: dict[str, Any],
    *,
    dialect: GrammarDialect,
    declared_type: str,
    start_symbol: str | None,
    source_map: JsonSourceMap | None,
) -> GrammarIngestionResult:
    rules: list[GrammarRule] = []
    terminals: list[GrammarTerminal] = []
    references: set[str] = set()
    for name, expression in sorted(rules_obj.items()):
        if not isinstance(name, str) or not name:
            raise GrammarIngestionError("grammar rule names must be non-empty strings")
        expr_text = expression if isinstance(expression, str) else json.dumps(expression, sort_keys=True)
        rules.append(GrammarRule(name, expr_text, _span(source_map, ("rules", name))))
        for terminal_match in _TERMINAL_RE.finditer(expr_text):
            terminal_text = next(group for group in terminal_match.groups() if group is not None)
            terminals.append(GrammarTerminal(terminal_text, "literal", _span(source_map, ("rules", name))))
        references.update(identifier for identifier in _IDENT_RE.findall(_TERMINAL_RE.sub(" ", expr_text)) if identifier != name)
    if not rules:
        raise GrammarIngestionError("grammar rules object must not be empty", span=_span(source_map, "rules"))
    rule_names = {rule.name for rule in rules}
    undefined = tuple(sorted(references - rule_names))
    issues = tuple(
        GrammarIngestionIssue(
            code="grammar-undefined-reference",
            message=f"Rule reference '{name}' has no declaration in this artifact",
            severity="warning",
        )
        for name in undefined
    )
    return GrammarIngestionResult(
        dialect=dialect,
        declared_type=declared_type,
        start_symbol=start_symbol or rules[0].name,
        rules=tuple(rules),
        terminals=tuple(terminals),
        references=tuple(sorted(references)),
        features=("rules-object",),
        issues=issues,
        source_spans=_source_spans(source_map),
    )


def _wrap_nested(
    dialect: GrammarDialect,
    declared_type: str,
    start_symbol: str,
    nested: GrammarIngestionResult,
    source_map: JsonSourceMap | None,
    features: tuple[str, ...],
) -> GrammarIngestionResult:
    return GrammarIngestionResult(
        dialect=dialect,
        declared_type=declared_type,
        start_symbol=start_symbol,
        rules=nested.rules,
        terminals=nested.terminals,
        references=nested.references,
        features=features,
        issues=nested.issues,
        source_spans=_source_spans(source_map) or nested.source_spans,
    )


def _replace_start(result: GrammarIngestionResult, start_symbol: str) -> GrammarIngestionResult:
    return GrammarIngestionResult(
        dialect=result.dialect,
        declared_type=result.declared_type,
        start_symbol=start_symbol,
        rules=result.rules,
        terminals=result.terminals,
        references=result.references,
        features=result.features,
        issues=result.issues,
        source_spans=result.source_spans,
    )


def _parse_json_document(text: str, path: str | None) -> tuple[Any, JsonSourceMap]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GrammarIngestionError(
            f"grammar JSON is not valid at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            span=SourceSpan(path=path or "<grammar>", start_line=exc.lineno, start_column=exc.colno),
        ) from exc
    try:
        source_map = build_json_source_map(text, path or "<grammar>")
    except ValueError as exc:
        raise GrammarIngestionError(f"grammar JSON could not be source-mapped: {exc}") from exc
    return raw, source_map


def _detect_dialect(raw: dict[str, Any], declared_type: str) -> GrammarDialect:
    normalized = _normalize_type(declared_type)
    if normalized is not GrammarDialect.PROMPTABI:
        return normalized
    if "$schema" in raw or "properties" in raw or raw.get("type") in {"object", "array", "string", "number", "integer", "boolean", "null"}:
        return GrammarDialect.JSON_SCHEMA
    if "json_schema" in raw or "schema" in raw or "choices" in raw or "choice" in raw:
        return GrammarDialect.OUTLINES
    if "root_rule" in raw or "bnf" in raw:
        return GrammarDialect.XGRAMMAR
    if "llguidance" in raw or "grammars" in raw or "lark_grammar" in raw:
        return GrammarDialect.LLGUIDANCE
    if "regex" in raw:
        return GrammarDialect.REGEX
    return GrammarDialect.PROMPTABI


def _normalize_type(value: str) -> GrammarDialect:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "bnf": GrammarDialect.EBNF,
        "ebnf-like": GrammarDialect.EBNF,
        "jsonschema": GrammarDialect.JSON_SCHEMA,
        "json-schema-2020-12": GrammarDialect.JSON_SCHEMA,
        "lark": GrammarDialect.EBNF,
        "ll-guidance": GrammarDialect.LLGUIDANCE,
        "promptabi-grammar": GrammarDialect.PROMPTABI,
        "regular-expression": GrammarDialect.REGEX,
        "re": GrammarDialect.REGEX,
        "x-grammar": GrammarDialect.XGRAMMAR,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return GrammarDialect(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in GrammarDialect)
        raise GrammarIngestionError(f"unsupported grammar_type '{value}' (expected one of {allowed})") from exc


def _json_schema_keywords(value: Any, *, in_properties: bool = False) -> set[str]:
    keywords: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and not in_properties:
                keywords.add(key)
            keywords.update(_json_schema_keywords(child, in_properties=key == "properties"))
    elif isinstance(value, list):
        for child in value:
            keywords.update(_json_schema_keywords(child))
    return keywords


def _schema_terminals(value: Any, source_map: JsonSourceMap | None, path: tuple[str, ...] = ()) -> tuple[GrammarTerminal, ...]:
    terminals: list[GrammarTerminal] = []
    if isinstance(value, dict):
        if isinstance(value.get("const"), str):
            terminals.append(GrammarTerminal(value["const"], "const", _span(source_map, (*path, "const"))))
        enum_value = value.get("enum")
        if isinstance(enum_value, list):
            for index, item in enumerate(enum_value):
                if isinstance(item, str):
                    terminals.append(GrammarTerminal(item, "enum", _span(source_map, (*path, "enum", str(index)))))
        for key, child in value.items():
            if isinstance(key, str):
                terminals.extend(_schema_terminals(child, source_map, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            terminals.extend(_schema_terminals(child, source_map, (*path, str(index))))
    return tuple(terminals)


def _shape_preview(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    preview: dict[str, Any] = {}
    for key in ("type", "enum", "const", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum"):
        if key in value:
            preview[key] = value[key]
    if "properties" in value and isinstance(value["properties"], dict):
        preview["properties"] = sorted(value["properties"])
    if "items" in value:
        preview["items"] = _shape_preview(value["items"])
    return preview


def _regex_issues(pattern: str, path: str | None) -> tuple[GrammarIngestionIssue, ...]:
    unsupported = {
        "regex-backreference": re.search(r"\\[1-9]|\\g<", pattern),
        "regex-lookaround": re.search(r"\(\?[=!<]", pattern),
        "regex-conditional": re.search(r"\(\?\(", pattern),
        "regex-named-group": re.search(r"\(\?P<", pattern),
    }
    return tuple(
        GrammarIngestionIssue(
            code=code,
            message=f"Regex construct '{code.removeprefix('regex-')}' is outside the supported subset",
            severity="abstention",
            span=_text_span(path),
        )
        for code, match in sorted(unsupported.items())
        if match is not None
    )


def _regex_features(pattern: str) -> set[str]:
    features = {"literal"} if _regex_literals(pattern) else set()
    if "|" in pattern:
        features.add("alternation")
    if re.search(r"[*+?]|\{\d", pattern):
        features.add("quantifier")
    if "[" in pattern:
        features.add("character-class")
    if "(" in pattern:
        features.add("group")
    return features


def _regex_literals(pattern: str) -> tuple[str, ...]:
    literals: list[str] = []
    current: list[str] = []
    escaped = False
    metacharacters = set(".^$*+?{}[]\\|()")
    for char in pattern:
        if escaped:
            if char in "dDsSwWbBAZ":
                if current:
                    literals.append("".join(current))
                    current = []
            else:
                current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in metacharacters:
            if current:
                literals.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        literals.append("".join(current))
    return tuple(sorted(dict.fromkeys(item for item in literals if item)))


def _required_str(raw: dict[str, Any], key: str, source_map: JsonSourceMap | None) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise GrammarIngestionError(f"grammar field '{key}' must be a non-empty string", span=_span(source_map, key))
    return value


def _optional_start(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _source_spans(source_map: JsonSourceMap | None) -> tuple[tuple[str, SourceSpan], ...]:
    if source_map is None:
        return ()
    return source_map.prefixed(())


def _span(source_map: JsonSourceMap | None, path: str | tuple[str, ...]) -> SourceSpan | None:
    if source_map is None:
        return None
    if isinstance(path, str):
        path = (path,)
    return source_map.span_for(path) or source_map.key_span_for(path)


def _text_span(path: str | None) -> SourceSpan | None:
    if path is None:
        return None
    grammar_path = Path(path)
    try:
        lines = grammar_path.read_text(encoding="utf-8").splitlines() or [""]
    except OSError:
        lines = [""]
    return SourceSpan(path=path, start_line=1, start_column=1, end_line=len(lines), end_column=max(1, len(lines[-1])))


def _line_span(path: str | None, line_number: int, line: str, *, start: int = 1, end: int | None = None) -> SourceSpan | None:
    if path is None:
        return None
    return SourceSpan(
        path=path,
        start_line=line_number,
        start_column=start,
        end_line=line_number,
        end_column=end or max(1, len(line)),
    )
