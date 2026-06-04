"""JSON Schema normalization for PromptABI's supported structured-output subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .diagnostics import SourceSpan
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
