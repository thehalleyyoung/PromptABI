"""JSON Schema normalization for PromptABI's supported structured-output subset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .diagnostics import SourceSpan
from .formal import AutomatonWitness, DeterministicFiniteAutomaton
from .source import JsonSourceMap

JsonPath = tuple[str, ...]
JsonScalar = str | int | float | bool | None


class JsonSchemaNodeKind(StrEnum):
    """Normalized node families used before grammar compilation."""

    ANY = "any"
    ARRAY = "array"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NULL = "null"
    NUMBER = "number"
    OBJECT = "object"
    STRING = "string"
    UNION = "union"
    INTERSECTION = "intersection"


@dataclass(frozen=True, slots=True)
class JsonSchemaIssue:
    """A non-fatal limitation found while normalizing JSON Schema."""

    code: str
    message: str
    severity: str = "abstention"
    path: JsonPath = ()
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("JSON Schema issues require a code and message")
        if self.severity not in {"warning", "abstention"}:
            raise ValueError("JSON Schema issue severity must be warning or abstention")

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("code", self.code),
            ("message", self.message),
            ("path", ".".join(self.path) or "<root>"),
            ("severity", self.severity),
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaProperty:
    """A normalized object property declaration."""

    name: str
    schema: "JsonSchemaNode"
    required: bool = False
    span: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class JsonSchemaNode:
    """A normalized JSON Schema node with explicit constraints and children."""

    kind: JsonSchemaNodeKind
    path: JsonPath = ()
    span: SourceSpan | None = None
    title: str | None = None
    description: str | None = None
    ref: str | None = None
    enum_values: tuple[str, ...] = ()
    const_value: str | None = None
    required: tuple[str, ...] = ()
    properties: tuple[JsonSchemaProperty, ...] = ()
    additional_properties: bool | "JsonSchemaNode" | None = None
    items: "JsonSchemaNode | None" = None
    min_items: int | None = None
    max_items: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: int | float | None = None
    exclusive_maximum: int | float | None = None
    multiple_of: int | float | None = None
    union_kind: str | None = None
    variants: tuple["JsonSchemaNode", ...] = ()

    @property
    def child_nodes(self) -> tuple["JsonSchemaNode", ...]:
        children = [property.schema for property in self.properties]
        if isinstance(self.additional_properties, JsonSchemaNode):
            children.append(self.additional_properties)
        if self.items is not None:
            children.append(self.items)
        children.extend(self.variants)
        return tuple(children)

    def walk(self) -> tuple["JsonSchemaNode", ...]:
        """Return this node and all descendants in deterministic preorder."""

        nodes = [self]
        for child in self.child_nodes:
            nodes.extend(child.walk())
        return tuple(nodes)


@dataclass(frozen=True, slots=True)
class JsonSchemaNormalizationResult:
    """Typed result of JSON Schema normalization."""

    root: JsonSchemaNode
    issues: tuple[JsonSchemaIssue, ...] = ()
    source_spans: tuple[tuple[str, SourceSpan], ...] = ()

    @property
    def supported_fragment(self) -> bool:
        return not any(issue.severity == "abstention" for issue in self.issues)

    @property
    def node_count(self) -> int:
        return len(self.root.walk())

    @property
    def max_depth(self) -> int:
        return _max_depth(self.root)

    @property
    def features(self) -> tuple[str, ...]:
        features: set[str] = set()
        for node in self.root.walk():
            features.add(node.kind.value)
            if node.properties:
                features.add("properties")
            if node.required:
                features.add("required")
            if node.additional_properties is not None:
                features.add("additionalProperties")
            if node.items is not None:
                features.add("items")
            if node.enum_values:
                features.add("enum")
            if node.const_value is not None:
                features.add("const")
            if node.union_kind is not None:
                features.add(node.union_kind)
            if any(
                value is not None
                for value in (
                    node.minimum,
                    node.maximum,
                    node.exclusive_minimum,
                    node.exclusive_maximum,
                    node.multiple_of,
                )
            ):
                features.add("numeric-constraints")
            if any(value is not None for value in (node.min_length, node.max_length, node.pattern, node.format)):
                features.add("string-constraints")
            if any(value is not None for value in (node.min_items, node.max_items)):
                features.add("array-constraints")
            if node.ref is not None:
                features.add("$ref")
        return tuple(sorted(features))

    @property
    def property_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for node in self.root.walk():
            for property in node.properties:
                paths.append(".".join((*node.path, "properties", property.name)))
        return tuple(sorted(dict.fromkeys(paths)))

    @property
    def required_property_count(self) -> int:
        return sum(len(node.required) for node in self.root.walk())

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("features", self.features),
            ("issue_count", len(self.issues)),
            ("issue_codes", tuple(issue.code for issue in self.issues)),
            ("max_depth", self.max_depth),
            ("node_count", self.node_count),
            ("property_paths", self.property_paths),
            ("required_property_count", self.required_property_count),
            ("root_kind", self.root.kind.value),
            ("supported_fragment", self.supported_fragment),
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaGrammarRule:
    """One compiled JSON Schema grammar rule with source provenance."""

    name: str
    expression: str
    path: JsonPath = ()
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.expression:
            raise ValueError("compiled JSON Schema grammar rules require a name and expression")


@dataclass(frozen=True, slots=True)
class JsonSchemaParserState:
    """A parser state reached while compiling the JSON grammar witness."""

    name: str
    path: JsonPath
    state_type: str
    accepts: bool
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.state_type:
            raise ValueError("compiled JSON Schema parser states require a name and type")

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("accepts", self.accepts),
            ("name", self.name),
            ("path", ".".join(self.path) or "<root>"),
            ("state_type", self.state_type),
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaGrammarIR:
    """Bounded grammar IR compiled from PromptABI's normalized JSON Schema subset."""

    start_symbol: str
    rules: tuple[JsonSchemaGrammarRule, ...]
    terminals: tuple[str, ...]
    parser_states: tuple[JsonSchemaParserState, ...]
    source_spans: tuple[tuple[str, SourceSpan], ...]
    automaton: DeterministicFiniteAutomaton

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.rules)

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("compiled_automaton_accept_count", len(self.automaton.accepts)),
            ("compiled_automaton_state_count", len(self.automaton.states)),
            ("compiled_parser_state_count", len(self.parser_states)),
            ("compiled_rule_count", len(self.rules)),
            ("compiled_rule_names", self.rule_names),
            ("compiled_start_symbol", self.start_symbol),
            ("compiled_terminal_count", len(self.terminals)),
            ("compiled_terminals", self.terminals),
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaWitness:
    """A concrete JSON value accepted by the compiled grammar and validator."""

    text: str
    value: JsonScalar | dict[str, Any] | list[Any]
    automaton_witness: AutomatonWitness
    validator: str
    validator_accepts: bool

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("compiled_witness_text", self.text),
            ("compiled_witness_validator", self.validator),
            ("compiled_witness_validator_accepts", self.validator_accepts),
            ("compiled_witness_state_path", self.automaton_witness.states),
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaCompilationResult:
    """JSON Schema compilation result with IR, witness, validator round-trip, and issues."""

    normalized: JsonSchemaNormalizationResult
    grammar: JsonSchemaGrammarIR
    witness: JsonSchemaWitness | None
    issues: tuple[JsonSchemaIssue, ...] = ()

    @property
    def supported_fragment(self) -> bool:
        return self.normalized.supported_fragment and not any(issue.severity == "abstention" for issue in self.issues)

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        witness_metadata = self.witness.to_metadata() if self.witness is not None else ()
        return (
            *self.grammar.to_metadata(),
            ("compiled_issue_codes", tuple(issue.code for issue in self.issues)),
            ("compiled_issue_count", len(self.issues)),
            ("compiled_supported_fragment", self.supported_fragment),
            *witness_metadata,
        )


SUPPORTED_JSON_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "default",
        "definitions",
        "description",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
)

SCHEMA_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


def normalize_json_schema_mapping(
    raw: dict[str, Any],
    *,
    source_map: JsonSourceMap | None = None,
    max_ref_depth: int = 4,
) -> JsonSchemaNormalizationResult:
    """Normalize a JSON Schema object into PromptABI's bounded supported subset."""

    if max_ref_depth < 1:
        raise ValueError("max_ref_depth must be positive")
    normalizer = _JsonSchemaNormalizer(raw=raw, source_map=source_map, max_ref_depth=max_ref_depth)
    root = normalizer.normalize(raw, (), ref_stack=())
    return JsonSchemaNormalizationResult(
        root=root,
        issues=tuple(normalizer.issues),
        source_spans=_source_spans(source_map),
    )


def compile_json_schema_mapping(
    raw: dict[str, Any],
    *,
    source_map: JsonSourceMap | None = None,
    max_ref_depth: int = 4,
) -> JsonSchemaCompilationResult:
    """Compile the supported JSON Schema subset to bounded grammar IR plus a witness.

    The compiler is intentionally finite: it creates a representative grammar and
    DFA over concrete JSON witnesses for the normalized subset, then round-trips
    the shortest witness through the real ``jsonschema`` Draft 2020-12 validator
    when that package is available.
    """

    normalized = normalize_json_schema_mapping(raw, source_map=source_map, max_ref_depth=max_ref_depth)
    compiler = _JsonSchemaCompiler(source_map=source_map)
    sample = compiler.compile_node(normalized.root, name="schema")
    words = (sample.text,) if sample.text is not None else ()
    automaton = DeterministicFiniteAutomaton.finite_language(words, name="json-schema-compiled-witnesses")
    grammar = JsonSchemaGrammarIR(
        start_symbol="schema",
        rules=tuple(compiler.rules),
        terminals=tuple(sorted(dict.fromkeys(compiler.terminals))),
        parser_states=tuple(compiler.parser_states),
        source_spans=normalized.source_spans,
        automaton=automaton,
    )
    witness = _compiled_witness(raw, automaton)
    issues = tuple(compiler.issues)
    if witness is None:
        issues = (
            *issues,
            JsonSchemaIssue(
                code="json-schema-compile-empty-language",
                message="The bounded JSON Schema compiler could not construct an accepted witness",
                severity="abstention",
            ),
        )
    elif not witness.validator_accepts:
        issues = (
            *issues,
            JsonSchemaIssue(
                code="json-schema-validator-round-trip-failed",
                message="The compiled JSON Schema witness was rejected by the real JSON Schema validator",
                severity="abstention",
            ),
        )
    return JsonSchemaCompilationResult(normalized=normalized, grammar=grammar, witness=witness, issues=issues)


@dataclass(slots=True)
class _JsonSchemaNormalizer:
    raw: dict[str, Any]
    source_map: JsonSourceMap | None
    max_ref_depth: int
    issues: list[JsonSchemaIssue]

    def __init__(self, raw: dict[str, Any], source_map: JsonSourceMap | None, max_ref_depth: int) -> None:
        self.raw = raw
        self.source_map = source_map
        self.max_ref_depth = max_ref_depth
        self.issues = []

    def normalize(self, schema: Any, path: JsonPath, *, ref_stack: tuple[str, ...]) -> JsonSchemaNode:
        if not isinstance(schema, dict):
            self._issue(
                "json-schema-invalid-node",
                "JSON Schema subschemas must be objects in the supported fragment",
                path,
            )
            return JsonSchemaNode(kind=JsonSchemaNodeKind.ANY, path=path, span=self._span(path))

        self._record_unsupported_keywords(schema, path)

        ref = schema.get("$ref")
        if isinstance(ref, str):
            resolved = self._resolve_ref(ref, path, ref_stack)
            if resolved is not None:
                target, target_path = resolved
                target_node = self.normalize(target, target_path, ref_stack=(*ref_stack, ref))
                return _copy_node_with_ref(target_node, path=path, span=self._span(path), ref=ref)
            return JsonSchemaNode(kind=JsonSchemaNodeKind.ANY, path=path, span=self._span(path), ref=ref)
        if ref is not None:
            self._issue("json-schema-invalid-ref", "JSON Schema $ref must be a string", (*path, "$ref"))

        variants = self._composition(schema, path, ref_stack)
        if variants is not None:
            union_kind, children = variants
            kind = JsonSchemaNodeKind.INTERSECTION if union_kind == "allOf" else JsonSchemaNodeKind.UNION
            return JsonSchemaNode(
                kind=kind,
                path=path,
                span=self._span(path),
                title=_optional_str(schema.get("title")),
                description=_optional_str(schema.get("description")),
                union_kind=union_kind,
                variants=children,
            )

        type_union = self._type_union(schema, path, ref_stack)
        if type_union is not None:
            return type_union

        kind = self._node_kind(schema, path)
        required = self._required(schema.get("required"), (*path, "required"))
        properties = self._properties(schema.get("properties"), (*path, "properties"), required, ref_stack)
        additional_properties = self._additional_properties(
            schema.get("additionalProperties"),
            (*path, "additionalProperties"),
            ref_stack,
        )
        items = self._items(schema.get("items"), (*path, "items"), ref_stack)
        enum_values = self._enum_values(schema.get("enum"), (*path, "enum"))
        const_value = self._const_value(schema, (*path, "const"))
        return JsonSchemaNode(
            kind=kind,
            path=path,
            span=self._span(path),
            title=_optional_str(schema.get("title")),
            description=_optional_str(schema.get("description")),
            enum_values=enum_values,
            const_value=const_value,
            required=required,
            properties=properties,
            additional_properties=additional_properties,
            items=items,
            min_items=self._nonnegative_int(schema.get("minItems"), (*path, "minItems")),
            max_items=self._nonnegative_int(schema.get("maxItems"), (*path, "maxItems")),
            min_length=self._nonnegative_int(schema.get("minLength"), (*path, "minLength")),
            max_length=self._nonnegative_int(schema.get("maxLength"), (*path, "maxLength")),
            pattern=_optional_str(schema.get("pattern")),
            format=_optional_str(schema.get("format")),
            minimum=self._number(schema.get("minimum"), (*path, "minimum")),
            maximum=self._number(schema.get("maximum"), (*path, "maximum")),
            exclusive_minimum=self._number(schema.get("exclusiveMinimum"), (*path, "exclusiveMinimum")),
            exclusive_maximum=self._number(schema.get("exclusiveMaximum"), (*path, "exclusiveMaximum")),
            multiple_of=self._positive_number(schema.get("multipleOf"), (*path, "multipleOf")),
        )

    def _type_union(
        self,
        schema: dict[str, Any],
        path: JsonPath,
        ref_stack: tuple[str, ...],
    ) -> JsonSchemaNode | None:
        raw_type = schema.get("type")
        if not isinstance(raw_type, list):
            return None
        variants: list[JsonSchemaNode] = []
        for index, item in enumerate(raw_type):
            if not isinstance(item, str) or item not in SCHEMA_TYPES:
                self._issue(
                    "json-schema-unsupported-type",
                    "JSON Schema type arrays must contain supported type strings",
                    (*path, "type", str(index)),
                )
                continue
            branch = dict(schema)
            branch["type"] = item
            variants.append(self.normalize(branch, (*path, "type", str(index)), ref_stack=ref_stack))
        if not variants:
            return JsonSchemaNode(kind=JsonSchemaNodeKind.ANY, path=path, span=self._span(path))
        if len(variants) == 1:
            return variants[0]
        return JsonSchemaNode(
            kind=JsonSchemaNodeKind.UNION,
            path=path,
            span=self._span(path),
            title=_optional_str(schema.get("title")),
            description=_optional_str(schema.get("description")),
            union_kind="type",
            variants=tuple(variants),
        )

    def _composition(
        self,
        schema: dict[str, Any],
        path: JsonPath,
        ref_stack: tuple[str, ...],
    ) -> tuple[str, tuple[JsonSchemaNode, ...]] | None:
        present = [key for key in ("anyOf", "oneOf", "allOf") if key in schema]
        if not present:
            return None
        if len(present) > 1:
            self._issue(
                "json-schema-multiple-composition-keywords",
                "Only one of anyOf, oneOf, or allOf is supported at one schema node",
                path,
            )
        key = present[0]
        value = schema.get(key)
        if not isinstance(value, list) or not value:
            self._issue(f"json-schema-invalid-{key}", f"JSON Schema {key} must be a non-empty array", (*path, key))
            return key, ()
        children = tuple(
            self.normalize(child, (*path, key, str(index)), ref_stack=ref_stack)
            for index, child in enumerate(value)
        )
        return key, children

    def _node_kind(self, schema: dict[str, Any], path: JsonPath) -> JsonSchemaNodeKind:
        raw_type = schema.get("type")
        if isinstance(raw_type, str):
            if raw_type in SCHEMA_TYPES:
                return JsonSchemaNodeKind(raw_type)
            self._issue("json-schema-unsupported-type", f"JSON Schema type '{raw_type}' is not supported", (*path, "type"))
            return JsonSchemaNodeKind.ANY
        if isinstance(raw_type, list):
            variants: list[JsonSchemaNode] = []
            for index, item in enumerate(raw_type):
                if not isinstance(item, str) or item not in SCHEMA_TYPES:
                    self._issue(
                        "json-schema-unsupported-type",
                        "JSON Schema type arrays must contain supported type strings",
                        (*path, "type", str(index)),
                    )
                    continue
                variants.append(JsonSchemaNode(kind=JsonSchemaNodeKind(item), path=(*path, "type", str(index)), span=self._span((*path, "type", str(index)))))
            if len(variants) == 1:
                return variants[0].kind
            if variants:
                return JsonSchemaNodeKind.UNION
            return JsonSchemaNodeKind.ANY
        if raw_type is not None:
            self._issue("json-schema-invalid-type", "JSON Schema type must be a string or array of strings", (*path, "type"))
        if "properties" in schema or "required" in schema or "additionalProperties" in schema:
            return JsonSchemaNodeKind.OBJECT
        if "items" in schema or "minItems" in schema or "maxItems" in schema:
            return JsonSchemaNodeKind.ARRAY
        if any(key in schema for key in ("minLength", "maxLength", "pattern", "format")):
            return JsonSchemaNodeKind.STRING
        if any(key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")):
            return JsonSchemaNodeKind.NUMBER
        return JsonSchemaNodeKind.ANY

    def _properties(
        self,
        value: Any,
        path: JsonPath,
        required: tuple[str, ...],
        ref_stack: tuple[str, ...],
    ) -> tuple[JsonSchemaProperty, ...]:
        if value is None:
            return ()
        if not isinstance(value, dict):
            self._issue("json-schema-invalid-properties", "JSON Schema properties must be an object", path)
            return ()
        properties: list[JsonSchemaProperty] = []
        for name, child in sorted(value.items()):
            if not isinstance(name, str):
                self._issue("json-schema-invalid-property-name", "JSON Schema property names must be strings", path)
                continue
            child_path = (*path, name)
            properties.append(
                JsonSchemaProperty(
                    name=name,
                    schema=self.normalize(child, child_path, ref_stack=ref_stack),
                    required=name in required,
                    span=self._span(child_path),
                )
            )
        return tuple(properties)

    def _additional_properties(
        self,
        value: Any,
        path: JsonPath,
        ref_stack: tuple[str, ...],
    ) -> bool | JsonSchemaNode | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return self.normalize(value, path, ref_stack=ref_stack)
        self._issue(
            "json-schema-invalid-additional-properties",
            "JSON Schema additionalProperties must be a boolean or schema object",
            path,
        )
        return None

    def _items(self, value: Any, path: JsonPath, ref_stack: tuple[str, ...]) -> JsonSchemaNode | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return self.normalize(value, path, ref_stack=ref_stack)
        if isinstance(value, list):
            self._issue(
                "json-schema-tuple-items",
                "Tuple-form array items are outside the supported JSON Schema fragment",
                path,
            )
            return None
        self._issue("json-schema-invalid-items", "JSON Schema items must be a schema object", path)
        return None

    def _required(self, value: Any, path: JsonPath) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            self._issue("json-schema-invalid-required", "JSON Schema required must be an array of strings", path)
            return ()
        return tuple(sorted(dict.fromkeys(value)))

    def _enum_values(self, value: Any, path: JsonPath) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or not value:
            self._issue("json-schema-invalid-enum", "JSON Schema enum must be a non-empty array", path)
            return ()
        return tuple(_canonical_json(item) for item in value)

    def _const_value(self, schema: dict[str, Any], path: JsonPath) -> str | None:
        if "const" not in schema:
            return None
        return _canonical_json(schema["const"])

    def _nonnegative_int(self, value: Any, path: JsonPath) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self._issue("json-schema-invalid-nonnegative-integer", "Constraint must be a non-negative integer", path)
            return None
        return value

    def _number(self, value: Any, path: JsonPath) -> int | float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._issue("json-schema-invalid-number", "Numeric constraint must be a number", path)
            return None
        return value

    def _positive_number(self, value: Any, path: JsonPath) -> int | float | None:
        number = self._number(value, path)
        if number is not None and number <= 0:
            self._issue("json-schema-invalid-positive-number", "Numeric constraint must be positive", path)
            return None
        return number

    def _resolve_ref(
        self,
        ref: str,
        path: JsonPath,
        ref_stack: tuple[str, ...],
    ) -> tuple[Any, JsonPath] | None:
        if not ref.startswith("#"):
            self._issue("json-schema-external-ref", "Only local JSON Schema $ref values are supported", (*path, "$ref"))
            return None
        if ref in ref_stack or len(ref_stack) >= self.max_ref_depth:
            self._issue(
                "json-schema-recursion-limit",
                f"JSON Schema reference expansion exceeded the bounded depth of {self.max_ref_depth}",
                (*path, "$ref"),
            )
            return None
        target_path = _decode_json_pointer(ref)
        if target_path is None:
            self._issue("json-schema-invalid-ref", "JSON Schema $ref must be a valid local JSON pointer", (*path, "$ref"))
            return None
        target: Any = self.raw
        for part in target_path:
            if not isinstance(target, dict) or part not in target:
                self._issue("json-schema-unresolved-ref", f"JSON Schema $ref target '{ref}' was not found", (*path, "$ref"))
                return None
            target = target[part]
        return target, target_path

    def _record_unsupported_keywords(self, schema: dict[str, Any], path: JsonPath) -> None:
        for key in schema:
            if key not in SUPPORTED_JSON_SCHEMA_KEYWORDS:
                self._issue(
                    "json-schema-unsupported-keyword",
                    f"JSON Schema keyword '{key}' is outside the supported normalization fragment",
                    (*path, key),
                )

    def _issue(self, code: str, message: str, path: JsonPath) -> None:
        self.issues.append(JsonSchemaIssue(code=code, message=message, path=path, span=self._span(path)))

    def _span(self, path: JsonPath) -> SourceSpan | None:
        if self.source_map is None:
            return None
        return self.source_map.span_for(path) or self.source_map.key_span_for(path)


@dataclass(frozen=True, slots=True)
class _CompiledSample:
    expression: str
    value: JsonScalar | dict[str, Any] | list[Any]

    @property
    def text(self) -> str:
        return _canonical_json(self.value)


@dataclass(slots=True)
class _JsonSchemaCompiler:
    source_map: JsonSourceMap | None
    rules: list[JsonSchemaGrammarRule]
    terminals: list[str]
    parser_states: list[JsonSchemaParserState]
    issues: list[JsonSchemaIssue]

    def __init__(self, source_map: JsonSourceMap | None) -> None:
        self.source_map = source_map
        self.rules = []
        self.terminals = []
        self.parser_states = []
        self.issues = []

    def compile_node(self, node: JsonSchemaNode, *, name: str) -> _CompiledSample:
        self.parser_states.append(
            JsonSchemaParserState(
                name=f"{name}:enter",
                path=node.path,
                state_type=f"{node.kind.value}:enter",
                accepts=False,
                span=node.span,
            )
        )
        sample = self._sample_for(node, name=name)
        self.rules.append(JsonSchemaGrammarRule(name=name, expression=sample.expression, path=node.path, span=node.span))
        self.parser_states.append(
            JsonSchemaParserState(
                name=f"{name}:accept",
                path=node.path,
                state_type=f"{node.kind.value}:accept",
                accepts=True,
                span=node.span,
            )
        )
        return sample

    def _sample_for(self, node: JsonSchemaNode, *, name: str) -> _CompiledSample:
        if node.const_value is not None:
            value = json.loads(node.const_value)
            self.terminals.append(node.const_value)
            return _CompiledSample(expression=f"const {node.const_value}", value=value)
        if node.enum_values:
            value = json.loads(node.enum_values[0])
            self.terminals.extend(node.enum_values)
            return _CompiledSample(expression="enum " + " | ".join(node.enum_values), value=value)
        if node.kind is JsonSchemaNodeKind.OBJECT:
            return self._object_sample(node, name=name)
        if node.kind is JsonSchemaNodeKind.ARRAY:
            return self._array_sample(node, name=name)
        if node.kind is JsonSchemaNodeKind.STRING:
            value = self._string_value(node)
            self.terminals.append(json.dumps(value))
            return _CompiledSample(expression=self._string_expression(node), value=value)
        if node.kind is JsonSchemaNodeKind.INTEGER:
            value = int(self._number_value(node, integer=True))
            self.terminals.append(str(value))
            return _CompiledSample(expression=self._number_expression(node, integer=True), value=value)
        if node.kind is JsonSchemaNodeKind.NUMBER:
            value = self._number_value(node, integer=False)
            self.terminals.append(_canonical_json(value))
            return _CompiledSample(expression=self._number_expression(node, integer=False), value=value)
        if node.kind is JsonSchemaNodeKind.BOOLEAN:
            self.terminals.extend(("true", "false"))
            return _CompiledSample(expression="true | false", value=True)
        if node.kind is JsonSchemaNodeKind.NULL:
            self.terminals.append("null")
            return _CompiledSample(expression="null", value=None)
        if node.kind is JsonSchemaNodeKind.UNION:
            if not node.variants:
                self._issue("json-schema-compile-empty-union", "Union schema has no variants to compile", node.path)
                return _CompiledSample(expression="any-json", value={})
            branch = self.compile_node(node.variants[0], name=f"{name}.variant0")
            return _CompiledSample(
                expression=f"{node.union_kind or 'union'}({', '.join(self._rule_name(name, index) for index, _ in enumerate(node.variants))})",
                value=branch.value,
            )
        if node.kind is JsonSchemaNodeKind.INTERSECTION:
            return self._intersection_sample(node, name=name)
        self._issue("json-schema-compile-any", "Unconstrained JSON Schema nodes compile to a bounded object witness", node.path)
        return _CompiledSample(expression="any-json", value={})

    def _object_sample(self, node: JsonSchemaNode, *, name: str) -> _CompiledSample:
        properties: dict[str, Any] = {}
        expressions: list[str] = []
        for property in node.properties:
            child_name = f"{name}.properties.{_safe_rule_part(property.name)}"
            child = self.compile_node(property.schema, name=child_name)
            modifier = "required" if property.required else "optional"
            expressions.append(f"{json.dumps(property.name)}:{child_name}<{modifier}>")
            if property.required:
                properties[property.name] = child.value
        if isinstance(node.additional_properties, JsonSchemaNode):
            child = self.compile_node(node.additional_properties, name=f"{name}.additionalProperties")
            expressions.append(f"<additionalProperties>:{child.expression}")
        elif node.additional_properties is True:
            expressions.append("<additionalProperties>:any-json")
        return _CompiledSample(expression="object{" + ",".join(expressions) + "}", value=properties)

    def _array_sample(self, node: JsonSchemaNode, *, name: str) -> _CompiledSample:
        min_items = node.min_items if node.min_items is not None else 0
        if node.max_items is not None and min_items > node.max_items:
            self._issue(
                "json-schema-compile-unsatisfiable-array-bounds",
                "Array minItems exceeds maxItems in the compiled schema",
                node.path,
            )
            min_items = node.max_items
        item = self.compile_node(node.items, name=f"{name}.items") if node.items is not None else _CompiledSample("any-json", {})
        return _CompiledSample(expression=f"array[{item.expression}]{{{min_items}, {node.max_items or '*'}}}", value=[item.value for _ in range(min_items)])

    def _intersection_sample(self, node: JsonSchemaNode, *, name: str) -> _CompiledSample:
        samples = [self.compile_node(variant, name=f"{name}.allOf{index}") for index, variant in enumerate(node.variants)]
        if all(isinstance(sample.value, dict) for sample in samples):
            merged: dict[str, Any] = {}
            for sample in samples:
                merged.update(sample.value)
            return _CompiledSample(expression="allOf(" + ",".join(sample.expression for sample in samples) + ")", value=merged)
        if samples:
            return _CompiledSample(expression="allOf(" + ",".join(sample.expression for sample in samples) + ")", value=samples[0].value)
        self._issue("json-schema-compile-empty-intersection", "Intersection schema has no variants to compile", node.path)
        return _CompiledSample(expression="allOf()", value={})

    def _string_value(self, node: JsonSchemaNode) -> str:
        min_length = node.min_length or 0
        max_length = node.max_length
        candidates = ("", "a", "aa", "safe", "value", "abc", "0")
        if min_length > 0:
            candidates = (*candidates, "a" * min_length)
        for candidate in candidates:
            if len(candidate) < min_length:
                continue
            if max_length is not None and len(candidate) > max_length:
                continue
            if node.pattern is not None and re.search(node.pattern, candidate) is None:
                continue
            return candidate
        self._issue(
            "json-schema-compile-pattern-witness",
            "The compiler could not synthesize a string satisfying the declared pattern and length bounds",
            node.path,
        )
        return "a" * min_length

    def _number_value(self, node: JsonSchemaNode, *, integer: bool) -> int | float:
        lower = 0
        if node.minimum is not None:
            lower = node.minimum
        if node.exclusive_minimum is not None:
            lower = node.exclusive_minimum + (1 if integer else 0.5)
        value: int | float = int(lower) if integer else lower
        if node.multiple_of is not None:
            multiple = int(node.multiple_of) if integer else node.multiple_of
            if multiple:
                quotient = value / multiple
                if quotient != int(quotient):
                    value = (int(quotient) + 1) * multiple
        upper = node.maximum if node.maximum is not None else node.exclusive_maximum
        if upper is not None and value > upper:
            self._issue(
                "json-schema-compile-unsatisfiable-number-bounds",
                "Numeric bounds did not admit the synthesized witness",
                node.path,
            )
        return value

    def _string_expression(self, node: JsonSchemaNode) -> str:
        bounds = []
        if node.min_length is not None:
            bounds.append(f"minLength={node.min_length}")
        if node.max_length is not None:
            bounds.append(f"maxLength={node.max_length}")
        if node.pattern is not None:
            bounds.append(f"pattern={node.pattern!r}")
        return "string" + ("[" + ",".join(bounds) + "]" if bounds else "")

    def _number_expression(self, node: JsonSchemaNode, *, integer: bool) -> str:
        bounds = []
        for key in ("minimum", "maximum", "exclusive_minimum", "exclusive_maximum", "multiple_of"):
            value = getattr(node, key)
            if value is not None:
                bounds.append(f"{key}={value}")
        return ("integer" if integer else "number") + ("[" + ",".join(bounds) + "]" if bounds else "")

    def _rule_name(self, name: str, index: int) -> str:
        return f"{name}.variant{index}"

    def _issue(self, code: str, message: str, path: JsonPath) -> None:
        self.issues.append(JsonSchemaIssue(code=code, message=message, severity="abstention", path=path, span=self._span(path)))

    def _span(self, path: JsonPath) -> SourceSpan | None:
        if self.source_map is None:
            return None
        return self.source_map.span_for(path) or self.source_map.key_span_for(path)


def _copy_node_with_ref(node: JsonSchemaNode, *, path: JsonPath, span: SourceSpan | None, ref: str) -> JsonSchemaNode:
    return JsonSchemaNode(
        kind=node.kind,
        path=path,
        span=span,
        title=node.title,
        description=node.description,
        ref=ref,
        enum_values=node.enum_values,
        const_value=node.const_value,
        required=node.required,
        properties=node.properties,
        additional_properties=node.additional_properties,
        items=node.items,
        min_items=node.min_items,
        max_items=node.max_items,
        min_length=node.min_length,
        max_length=node.max_length,
        pattern=node.pattern,
        format=node.format,
        minimum=node.minimum,
        maximum=node.maximum,
        exclusive_minimum=node.exclusive_minimum,
        exclusive_maximum=node.exclusive_maximum,
        multiple_of=node.multiple_of,
        union_kind=node.union_kind,
        variants=node.variants,
    )


def _decode_json_pointer(ref: str) -> JsonPath | None:
    if ref == "#":
        return ()
    if not ref.startswith("#/"):
        return None
    parts = []
    for part in ref[2:].split("/"):
        parts.append(part.replace("~1", "/").replace("~0", "~"))
    return tuple(parts)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _max_depth(node: JsonSchemaNode) -> int:
    if not node.child_nodes:
        return 1
    return 1 + max(_max_depth(child) for child in node.child_nodes)


def _source_spans(source_map: JsonSourceMap | None) -> tuple[tuple[str, SourceSpan], ...]:
    if source_map is None:
        return ()
    return source_map.prefixed(())


def _safe_rule_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return safe or "property"


def _compiled_witness(raw_schema: dict[str, Any], automaton: DeterministicFiniteAutomaton) -> JsonSchemaWitness | None:
    automaton_witness = automaton.shortest_witness()
    if automaton_witness is None:
        return None
    text = automaton_witness.text
    value = json.loads(text)
    validator, accepts = _validate_with_real_jsonschema(raw_schema, value)
    return JsonSchemaWitness(
        text=text,
        value=value,
        automaton_witness=automaton_witness,
        validator=validator,
        validator_accepts=accepts,
    )


def _validate_with_real_jsonschema(raw_schema: dict[str, Any], value: Any) -> tuple[str, bool]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return "jsonschema-unavailable", True
    validator = Draft202012Validator(raw_schema)
    return "jsonschema.Draft202012Validator", not any(validator.iter_errors(value))
