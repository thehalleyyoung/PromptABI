"""Tokenizer x grammar bounded ambiguity checks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, GrammarArtifact, SchemaArtifact, TokenizerArtifact
from .grammar_emptiness import _BoundedGrammarProduct, _load_bounded_grammar
from .json_schema import JsonSchemaNode, JsonSchemaNodeKind, normalize_json_schema_mapping
from .tokenizers import EncodeResult, TokenizerAdapter, TokenizerBackend, TokenizerError


class GrammarTokenizerAmbiguityKind(StrEnum):
    """Ambiguity families found in the tokenizer x grammar product."""

    TOKEN_PATH_CONFLICT = "token-path-conflict"
    DECODED_TEXT_CONFLICT = "decoded-text-conflict"
    BYTE_ALIAS = "byte-alias"
    ADDED_TOKEN_ALIAS = "added-token-alias"


@dataclass(frozen=True, slots=True)
class GrammarTokenizerAmbiguityFinding:
    """One bounded ambiguity witness between grammar texts and token paths."""

    kind: GrammarTokenizerAmbiguityKind
    grammar_text: str
    other_grammar_text: str
    structured_value: str
    other_structured_value: str
    token_ids: tuple[int, ...]
    other_token_ids: tuple[int, ...]
    decoded_text: str
    other_decoded_text: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "grammar_text": self.grammar_text,
            "other_grammar_text": self.other_grammar_text,
            "structured_value": self.structured_value,
            "other_structured_value": self.other_structured_value,
            "token_ids": list(self.token_ids),
            "other_token_ids": list(self.other_token_ids),
            "decoded_text": self.decoded_text,
            "other_decoded_text": self.other_decoded_text,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GrammarTokenizerAmbiguityReport:
    """Bounded ambiguity analysis for one tokenizer x grammar pair."""

    tokenizer_name: str
    grammar_name: str
    grammar_kind: str
    tokenizer_backend: str
    assumptions: tuple[str, ...]
    findings: tuple[GrammarTokenizerAmbiguityFinding, ...] = ()
    abstained: bool = False
    reason: str | None = None
    checked_candidates: int = 0
    grammar_issue_codes: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "tokenizer_name": self.tokenizer_name,
            "grammar_name": self.grammar_name,
            "grammar_kind": self.grammar_kind,
            "tokenizer_backend": self.tokenizer_backend,
            "assumptions": list(self.assumptions),
            "abstained": self.abstained,
            "checked_candidates": self.checked_candidates,
            "grammar_issue_codes": list(self.grammar_issue_codes),
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class _Observation:
    text: str
    value_key: str
    encoded: EncodeResult
    decoded_text: str


def analyze_tokenizer_grammar_ambiguity(
    tokenizer_artifact: TokenizerArtifact,
    grammar_artifact: SchemaArtifact | GrammarArtifact,
    tokenizer: TokenizerAdapter,
    *,
    max_values: int = 16,
    max_variants_per_value: int = 8,
    max_findings: int = 16,
) -> GrammarTokenizerAmbiguityReport:
    """Find bounded tokenizer/grammar ambiguities over JSON Schema products.

    The check treats JSON lexical freedom as part of the grammar interface:
    whitespace, escaped strings, Unicode spellings, tokenizer normalization, and
    added-token shortcuts can all make multiple bytes or token paths correspond
    to the same or different structured values.  It reports concrete witnesses
    and abstains for grammar dialects that do not yet have a bounded JSON value
    generator.
    """

    if max_values <= 0:
        raise ValueError("max_values must be positive")
    if max_variants_per_value <= 0:
        raise ValueError("max_variants_per_value must be positive")
    if max_findings <= 0:
        raise ValueError("max_findings must be positive")

    try:
        product_info = _load_bounded_grammar(grammar_artifact)
        raw_schema = _load_json_schema(grammar_artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _abstained_report(
            tokenizer_artifact,
            grammar_artifact,
            tokenizer,
            reason=f"could not compile bounded ambiguity product: {exc}",
        )

    if not product_info.supported or product_info.kind != "json-schema":
        return _abstained_report(
            tokenizer_artifact,
            grammar_artifact,
            tokenizer,
            product=product_info,
            reason=product_info.reason or "grammar is outside the bounded JSON Schema ambiguity fragment",
        )
    if not _jsonschema_available():
        return _abstained_report(
            tokenizer_artifact,
            grammar_artifact,
            tokenizer,
            product=product_info,
            reason="jsonschema is required to filter bounded ambiguity candidates soundly",
        )

    observations = _observations(raw_schema, tokenizer, max_values=max_values, max_variants_per_value=max_variants_per_value)
    findings = _find_ambiguities(observations, tokenizer, max_findings=max_findings)
    return GrammarTokenizerAmbiguityReport(
        tokenizer_name=tokenizer_artifact.name,
        grammar_name=grammar_artifact.name,
        grammar_kind=product_info.kind,
        tokenizer_backend=tokenizer.backend.value,
        assumptions=(
            *product_info.assumptions,
            "json-lexical-variant-generation",
            "schema-validator-candidate-filter",
            "token-path-and-decoded-text-equivalence-classes",
        ),
        findings=tuple(findings),
        checked_candidates=len(observations),
        grammar_issue_codes=product_info.issue_codes,
    )


def _load_json_schema(artifact: SchemaArtifact | GrammarArtifact) -> dict[str, Any]:
    if artifact.kind is ArtifactKind.SCHEMA:
        if artifact.location.path is None:
            raise ValueError("schema artifacts require a local JSON file for ambiguity analysis")
        path = Path(artifact.location.path)
    elif artifact.kind is ArtifactKind.GRAMMAR and artifact.grammar_type.lower() in {
        "json-schema",
        "jsonschema",
        "json-schema-2020-12",
    }:
        if artifact.location.path is None:
            raise ValueError("JSON Schema grammar artifacts require a local file for ambiguity analysis")
        path = Path(artifact.location.path)
    else:
        raise ValueError("only local JSON Schema artifacts are supported by bounded ambiguity analysis")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON Schema root must be an object")
    return raw


def _observations(
    raw_schema: dict[str, Any],
    tokenizer: TokenizerAdapter,
    *,
    max_values: int,
    max_variants_per_value: int,
) -> tuple[_Observation, ...]:
    normalized = normalize_json_schema_mapping(raw_schema)
    values = _candidate_values(normalized.root, limit=max_values)
    observations: list[_Observation] = []
    seen_texts: set[str] = set()
    for value in values:
        for text in _json_text_variants(value, max_variants=max_variants_per_value):
            if text in seen_texts:
                continue
            seen_texts.add(text)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not _schema_accepts(raw_schema, parsed):
                continue
            encoded = tokenizer.encode(text, add_special_tokens=False)
            try:
                decoded = tokenizer.decode(encoded.token_ids).text
            except TokenizerError:
                continue
            if _structured_key_from_text(decoded, raw_schema) is None:
                continue
            observations.append(
                _Observation(
                    text=text,
                    value_key=_structured_key(parsed),
                    encoded=encoded,
                    decoded_text=decoded,
                )
            )
    return tuple(observations)


def _find_ambiguities(
    observations: tuple[_Observation, ...],
    tokenizer: TokenizerAdapter,
    *,
    max_findings: int,
) -> tuple[GrammarTokenizerAmbiguityFinding, ...]:
    findings: list[GrammarTokenizerAmbiguityFinding] = []
    findings.extend(_conflicts_by_token_path(observations))
    findings.extend(_conflicts_by_decoded_text(observations))
    findings.extend(_byte_aliases(observations))
    findings.extend(_added_token_aliases(observations, tokenizer))
    deduped: list[GrammarTokenizerAmbiguityFinding] = []
    seen: set[tuple[object, ...]] = set()
    for finding in sorted(
        findings,
        key=lambda item: (
            item.kind.value,
            item.grammar_text,
            item.other_grammar_text,
            item.token_ids,
            item.other_token_ids,
        ),
    ):
        key = (
            finding.kind,
            finding.grammar_text,
            finding.other_grammar_text,
            finding.structured_value,
            finding.other_structured_value,
            finding.token_ids,
            finding.other_token_ids,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
        if len(deduped) >= max_findings:
            break
    return tuple(deduped)


def _conflicts_by_token_path(observations: tuple[_Observation, ...]) -> tuple[GrammarTokenizerAmbiguityFinding, ...]:
    by_tokens: dict[tuple[int, ...], list[_Observation]] = defaultdict(list)
    for observation in observations:
        by_tokens[observation.encoded.token_ids].append(observation)
    findings: list[GrammarTokenizerAmbiguityFinding] = []
    for token_ids, group in by_tokens.items():
        first_by_value: dict[str, _Observation] = {}
        for observation in group:
            first_by_value.setdefault(observation.value_key, observation)
        if len(first_by_value) < 2:
            continue
        left, right = tuple(first_by_value.values())[:2]
        findings.append(
            _finding(
                GrammarTokenizerAmbiguityKind.TOKEN_PATH_CONFLICT,
                left,
                right,
                reason="distinct grammar values collapse to the same tokenizer path",
                token_ids=token_ids,
                other_token_ids=token_ids,
            )
        )
    return tuple(findings)


def _conflicts_by_decoded_text(observations: tuple[_Observation, ...]) -> tuple[GrammarTokenizerAmbiguityFinding, ...]:
    by_decoded: dict[str, list[_Observation]] = defaultdict(list)
    for observation in observations:
        by_decoded[observation.decoded_text].append(observation)
    findings: list[GrammarTokenizerAmbiguityFinding] = []
    for group in by_decoded.values():
        first_by_value: dict[str, _Observation] = {}
        for observation in group:
            first_by_value.setdefault(observation.value_key, observation)
        if len(first_by_value) < 2:
            continue
        left, right = tuple(first_by_value.values())[:2]
        findings.append(
            _finding(
                GrammarTokenizerAmbiguityKind.DECODED_TEXT_CONFLICT,
                left,
                right,
                reason="distinct grammar values collapse to the same decoded text after tokenizer normalization",
            )
        )
    return tuple(findings)


def _byte_aliases(observations: tuple[_Observation, ...]) -> tuple[GrammarTokenizerAmbiguityFinding, ...]:
    by_value: dict[str, list[_Observation]] = defaultdict(list)
    for observation in observations:
        by_value[observation.value_key].append(observation)
    findings: list[GrammarTokenizerAmbiguityFinding] = []
    for value_key, group in by_value.items():
        first_by_tokens: dict[tuple[int, ...], _Observation] = {}
        for observation in group:
            first_by_tokens.setdefault(observation.encoded.token_ids, observation)
        if len(first_by_tokens) < 2:
            continue
        left, right = tuple(first_by_tokens.values())[:2]
        findings.append(
            _finding(
                GrammarTokenizerAmbiguityKind.BYTE_ALIAS,
                left,
                right,
                structured_value=value_key,
                other_structured_value=value_key,
                reason="multiple JSON byte spellings parse to the same structured value with different token paths",
            )
        )
    return tuple(findings)


def _added_token_aliases(
    observations: tuple[_Observation, ...],
    tokenizer: TokenizerAdapter,
) -> tuple[GrammarTokenizerAmbiguityFinding, ...]:
    if tokenizer.backend is not TokenizerBackend.BYTE_LEVEL:
        return ()
    findings: list[GrammarTokenizerAmbiguityFinding] = []
    for observation in observations:
        if not any(token.added for token in observation.encoded.tokens):
            continue
        byte_ids = tuple(observation.encoded.normalized_text.encode("utf-8"))
        if byte_ids == observation.encoded.token_ids:
            continue
        try:
            byte_decoded = tokenizer.decode(byte_ids).text
        except TokenizerError:
            continue
        if byte_decoded != observation.decoded_text:
            continue
        findings.append(
            GrammarTokenizerAmbiguityFinding(
                kind=GrammarTokenizerAmbiguityKind.ADDED_TOKEN_ALIAS,
                grammar_text=observation.text,
                other_grammar_text=observation.text,
                structured_value=observation.value_key,
                other_structured_value=observation.value_key,
                token_ids=observation.encoded.token_ids,
                other_token_ids=byte_ids,
                decoded_text=observation.decoded_text,
                other_decoded_text=byte_decoded,
                reason="an added token and its byte-level spelling decode to the same grammar text",
            )
        )
    return tuple(findings)


def _finding(
    kind: GrammarTokenizerAmbiguityKind,
    left: _Observation,
    right: _Observation,
    *,
    reason: str,
    structured_value: str | None = None,
    other_structured_value: str | None = None,
    token_ids: tuple[int, ...] | None = None,
    other_token_ids: tuple[int, ...] | None = None,
) -> GrammarTokenizerAmbiguityFinding:
    return GrammarTokenizerAmbiguityFinding(
        kind=kind,
        grammar_text=left.text,
        other_grammar_text=right.text,
        structured_value=structured_value or left.value_key,
        other_structured_value=other_structured_value or right.value_key,
        token_ids=token_ids or left.encoded.token_ids,
        other_token_ids=other_token_ids or right.encoded.token_ids,
        decoded_text=left.decoded_text,
        other_decoded_text=right.decoded_text,
        reason=reason,
    )


def _candidate_values(node: JsonSchemaNode, *, limit: int) -> tuple[Any, ...]:
    values = tuple(_candidate_values_uncapped(node))
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _structured_key(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def _candidate_values_uncapped(node: JsonSchemaNode) -> tuple[Any, ...]:
    if node.const_value is not None:
        return (json.loads(node.const_value),)
    if node.enum_values:
        return tuple(json.loads(value) for value in node.enum_values)
    if node.kind is JsonSchemaNodeKind.BOOLEAN:
        return (True, False)
    if node.kind is JsonSchemaNodeKind.NULL:
        return (None,)
    if node.kind is JsonSchemaNodeKind.STRING:
        return (_string_value(node),)
    if node.kind is JsonSchemaNodeKind.INTEGER:
        return (0,)
    if node.kind is JsonSchemaNodeKind.NUMBER:
        return (0,)
    if node.kind is JsonSchemaNodeKind.ARRAY:
        min_items = node.min_items if node.min_items is not None else 0
        item_values = _candidate_values(node.items, limit=4) if node.items is not None else ({},)
        if min_items == 0:
            return ([], [item_values[0]])
        return ([item_values[0] for _ in range(min_items)],)
    if node.kind is JsonSchemaNodeKind.OBJECT:
        required_properties = tuple(property for property in node.properties if property.required)
        if not required_properties:
            return ({},)
        names = tuple(property.name for property in required_properties)
        value_sets = tuple(_candidate_values(property.schema, limit=4) for property in required_properties)
        values = []
        for combination in product(*value_sets):
            values.append(dict(zip(names, combination, strict=True)))
        return tuple(values)
    if node.kind is JsonSchemaNodeKind.UNION:
        values: list[Any] = []
        for variant in node.variants:
            values.extend(_candidate_values(variant, limit=4))
        return tuple(values)
    if node.kind is JsonSchemaNodeKind.INTERSECTION and node.variants:
        merged: dict[str, Any] = {}
        for variant in node.variants:
            first = _candidate_values(variant, limit=1)[0]
            if isinstance(first, dict):
                merged.update(first)
            else:
                return (first,)
        return (merged,)
    return ({},)


def _string_value(node: JsonSchemaNode) -> str:
    min_length = node.min_length or 0
    candidates = ("", "A", "é", "e\u0301", "safe", "value", "a" * min_length)
    for candidate in candidates:
        if len(candidate) < min_length:
            continue
        if node.max_length is not None and len(candidate) > node.max_length:
            continue
        return candidate
    return "a" * min_length


def _json_text_variants(value: Any, *, max_variants: int) -> tuple[str, ...]:
    compact_ascii = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    compact_unicode = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    spaced = json.dumps(value, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)
    variants = [
        compact_ascii,
        compact_unicode,
        spaced,
        f" {compact_unicode}\n",
        _ascii_string_escape_variant(value),
    ]
    deduped: list[str] = []
    for variant in variants:
        if variant is None or variant in deduped:
            continue
        deduped.append(variant)
        if len(deduped) >= max_variants:
            break
    return tuple(deduped)


def _ascii_string_escape_variant(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    escaped = "".join(f"\\u{ord(char):04x}" if char.isascii() and char.isalnum() else char for char in value)
    return f'"{escaped}"'


def _structured_key_from_text(text: str, raw_schema: dict[str, Any]) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not _schema_accepts(raw_schema, value):
        return None
    return _structured_key(value)


def _structured_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _schema_accepts(raw_schema: dict[str, Any], value: Any) -> bool:
    from jsonschema import Draft202012Validator
    return not any(Draft202012Validator(raw_schema).iter_errors(value))


def _jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def _abstained_report(
    tokenizer_artifact: TokenizerArtifact,
    grammar_artifact: SchemaArtifact | GrammarArtifact,
    tokenizer: TokenizerAdapter,
    *,
    reason: str,
    product: _BoundedGrammarProduct | None = None,
) -> GrammarTokenizerAmbiguityReport:
    return GrammarTokenizerAmbiguityReport(
        tokenizer_name=tokenizer_artifact.name,
        grammar_name=grammar_artifact.name,
        grammar_kind=product.kind if product is not None else grammar_artifact.kind.value,
        tokenizer_backend=tokenizer.backend.value,
        assumptions=product.assumptions if product is not None else ("tokenizer-encode-normalize-decode-product",),
        abstained=True,
        reason=reason,
        grammar_issue_codes=product.issue_codes if product is not None else (),
    )
