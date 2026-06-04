"""Tokenizer x grammar bounded emptiness checks."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, GrammarArtifact, SchemaArtifact, TokenizerArtifact
from .formal import AutomatonWitness, DeterministicFiniteAutomaton
from .grammars import GrammarDialect, GrammarIngestionError, ingest_grammar_file, ingest_grammar_mapping
from .json_schema import JsonSchemaCompilationResult, compile_json_schema_mapping
from .source import build_json_source_map
from .tokenizers import EncodeResult, TokenizerAdapter


class GrammarTokenizerEmptinessStatus(StrEnum):
    """Possible outcomes for the bounded tokenizer x grammar product."""

    SATISFIABLE = "satisfiable"
    EMPTY = "empty"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class GrammarTokenizerWitness:
    """A concrete token path that survives tokenizer assumptions and is grammar-accepted."""

    grammar_text: str
    normalized_text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str | None, ...]
    grammar_states: tuple[str, ...]

    @classmethod
    def from_encode(
        cls,
        *,
        grammar_text: str,
        encoded: EncodeResult,
        decoded_text: str,
        automaton_witness: AutomatonWitness,
    ) -> "GrammarTokenizerWitness":
        return cls(
            grammar_text=grammar_text,
            normalized_text=encoded.normalized_text,
            decoded_text=decoded_text,
            token_ids=encoded.token_ids,
            token_texts=encoded.token_texts,
            grammar_states=automaton_witness.states,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "grammar_text": self.grammar_text,
            "normalized_text": self.normalized_text,
            "decoded_text": self.decoded_text,
            "token_ids": list(self.token_ids),
            "token_texts": list(self.token_texts),
            "grammar_states": list(self.grammar_states),
        }


@dataclass(frozen=True, slots=True)
class GrammarTokenizerAttempt:
    """One candidate accepted by the grammar but rejected by tokenizer/backend assumptions."""

    grammar_text: str
    normalized_text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "grammar_text": self.grammar_text,
            "normalized_text": self.normalized_text,
            "decoded_text": self.decoded_text,
            "token_ids": list(self.token_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GrammarTokenizerEmptinessReport:
    """Bounded proof result for one tokenizer x grammar pair."""

    tokenizer_name: str
    grammar_name: str
    grammar_kind: str
    tokenizer_backend: str
    status: GrammarTokenizerEmptinessStatus
    assumptions: tuple[str, ...]
    witness: GrammarTokenizerWitness | None = None
    attempts: tuple[GrammarTokenizerAttempt, ...] = ()
    reason: str | None = None
    checked_candidates: int = 0
    grammar_state_count: int = 0
    grammar_accept_count: int = 0
    grammar_issue_codes: tuple[str, ...] = ()

    @property
    def satisfiable(self) -> bool:
        return self.status is GrammarTokenizerEmptinessStatus.SATISFIABLE

    @property
    def empty(self) -> bool:
        return self.status is GrammarTokenizerEmptinessStatus.EMPTY

    @property
    def abstained(self) -> bool:
        return self.status is GrammarTokenizerEmptinessStatus.ABSTAINED

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "tokenizer_name": self.tokenizer_name,
            "grammar_name": self.grammar_name,
            "grammar_kind": self.grammar_kind,
            "tokenizer_backend": self.tokenizer_backend,
            "status": self.status.value,
            "assumptions": list(self.assumptions),
            "checked_candidates": self.checked_candidates,
            "grammar_state_count": self.grammar_state_count,
            "grammar_accept_count": self.grammar_accept_count,
            "grammar_issue_codes": list(self.grammar_issue_codes),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }
        if self.reason is not None:
            data["reason"] = self.reason
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class _BoundedGrammarProduct:
    kind: str
    automaton: DeterministicFiniteAutomaton
    supported: bool
    assumptions: tuple[str, ...]
    issue_codes: tuple[str, ...] = ()
    reason: str | None = None


def analyze_tokenizer_grammar_emptiness(
    tokenizer_artifact: TokenizerArtifact,
    grammar_artifact: SchemaArtifact | GrammarArtifact,
    tokenizer: TokenizerAdapter,
    *,
    max_candidates: int = 32,
    max_depth: int = 256,
) -> GrammarTokenizerEmptinessReport:
    """Check whether a bounded grammar has any accepted token path under a tokenizer.

    The product is deliberately explicit about its backend assumptions: constrained
    decoding libraries commonly tokenize grammar literals or witnesses before
    decoding candidate token IDs.  PromptABI therefore proves emptiness over the
    finite language produced by the grammar compiler plus the selected tokenizer's
    encode-normalize-decode behavior, and abstains instead of guessing for grammar
    dialects without a bounded automaton.
    """

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    try:
        product = _load_bounded_grammar(grammar_artifact)
    except (OSError, json.JSONDecodeError, GrammarIngestionError, ValueError) as exc:
        return _abstained_report(
            tokenizer_artifact,
            grammar_artifact,
            tokenizer,
            reason=f"could not compile bounded grammar product: {exc}",
        )

    if not product.supported:
        return _abstained_report(
            tokenizer_artifact,
            grammar_artifact,
            tokenizer,
            product=product,
            reason=product.reason or "grammar is outside the supported bounded product fragment",
        )

    candidates = tuple(_accepted_witnesses(product.automaton, max_candidates=max_candidates, max_depth=max_depth))
    if not candidates:
        return GrammarTokenizerEmptinessReport(
            tokenizer_name=tokenizer_artifact.name,
            grammar_name=grammar_artifact.name,
            grammar_kind=product.kind,
            tokenizer_backend=tokenizer.backend.value,
            status=GrammarTokenizerEmptinessStatus.EMPTY,
            assumptions=product.assumptions,
            reason="bounded grammar automaton has no accepting path",
            checked_candidates=0,
            grammar_state_count=len(product.automaton.states),
            grammar_accept_count=len(product.automaton.accepts),
            grammar_issue_codes=product.issue_codes,
        )

    attempts: list[GrammarTokenizerAttempt] = []
    for witness in candidates:
        encoded = tokenizer.encode(witness.text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.token_ids).text
        if product.automaton.accepts_text(decoded):
            return GrammarTokenizerEmptinessReport(
                tokenizer_name=tokenizer_artifact.name,
                grammar_name=grammar_artifact.name,
                grammar_kind=product.kind,
                tokenizer_backend=tokenizer.backend.value,
                status=GrammarTokenizerEmptinessStatus.SATISFIABLE,
                assumptions=product.assumptions,
                witness=GrammarTokenizerWitness.from_encode(
                    grammar_text=witness.text,
                    encoded=encoded,
                    decoded_text=decoded,
                    automaton_witness=witness,
                ),
                checked_candidates=len(attempts) + 1,
                grammar_state_count=len(product.automaton.states),
                grammar_accept_count=len(product.automaton.accepts),
                grammar_issue_codes=product.issue_codes,
            )
        reason = (
            "tokenizer normalization changed the grammar witness outside the accepted language"
            if encoded.normalized_text != witness.text
            else "decoded token path is not accepted by the grammar automaton"
        )
        attempts.append(
            GrammarTokenizerAttempt(
                grammar_text=witness.text,
                normalized_text=encoded.normalized_text,
                decoded_text=decoded,
                token_ids=encoded.token_ids,
                reason=reason,
            )
        )

    return GrammarTokenizerEmptinessReport(
        tokenizer_name=tokenizer_artifact.name,
        grammar_name=grammar_artifact.name,
        grammar_kind=product.kind,
        tokenizer_backend=tokenizer.backend.value,
        status=GrammarTokenizerEmptinessStatus.EMPTY,
        assumptions=product.assumptions,
        attempts=tuple(attempts),
        reason="no bounded grammar witness survived the tokenizer encode/decode product",
        checked_candidates=len(attempts),
        grammar_state_count=len(product.automaton.states),
        grammar_accept_count=len(product.automaton.accepts),
        grammar_issue_codes=product.issue_codes,
    )


def _load_bounded_grammar(artifact: SchemaArtifact | GrammarArtifact) -> _BoundedGrammarProduct:
    if artifact.kind is ArtifactKind.SCHEMA:
        if artifact.location.path is None:
            return _unsupported_product(artifact, "schema artifacts require a local JSON file for bounded grammar compilation")
        raw, source_map = _load_json_object(Path(artifact.location.path))
        return _json_schema_product(compile_json_schema_mapping(raw, source_map=source_map))

    if artifact.kind is not ArtifactKind.GRAMMAR:
        raise ValueError(f"expected schema or grammar artifact, got {artifact.kind.value}")
    if artifact.location.path is None:
        return _unsupported_product(artifact, "grammar artifacts require a local file for bounded grammar compilation")
    path = Path(artifact.location.path)
    if artifact.grammar_type.lower() in {"json-schema", "jsonschema", "json-schema-2020-12"}:
        raw, source_map = _load_json_object(path)
        return _json_schema_product(compile_json_schema_mapping(raw, source_map=source_map))
    result = ingest_grammar_file(path, declared_type=artifact.grammar_type)
    if result.dialect is GrammarDialect.JSON_SCHEMA:
        raw, source_map = _load_json_object(path)
        return _json_schema_product(compile_json_schema_mapping(raw, source_map=source_map))
    return _unsupported_product(
        artifact,
        f"grammar dialect '{result.dialect.value}' does not yet compile to a bounded DFA product",
        issue_codes=tuple(issue.code for issue in result.issues),
    )


def _load_json_object(path: Path) -> tuple[dict[str, Any], Any]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("JSON grammar root must be an object")
    return raw, build_json_source_map(text, path)


def _json_schema_product(result: JsonSchemaCompilationResult) -> _BoundedGrammarProduct:
    issue_codes = tuple(issue.code for issue in (*result.normalized.issues, *result.issues))
    reason = None
    if not result.supported_fragment:
        reason = "JSON Schema compiler abstained: " + ", ".join(issue_codes or ("unsupported fragment",))
    return _BoundedGrammarProduct(
        kind="json-schema",
        automaton=result.grammar.automaton,
        supported=result.supported_fragment,
        assumptions=(
            "json-schema-supported-subset",
            "bounded-compiled-witness-dfa",
            "tokenizer-encode-normalize-decode-product",
        ),
        issue_codes=issue_codes,
        reason=reason,
    )


def _unsupported_product(
    artifact: SchemaArtifact | GrammarArtifact,
    reason: str,
    *,
    issue_codes: tuple[str, ...] = (),
) -> _BoundedGrammarProduct:
    return _BoundedGrammarProduct(
        kind=artifact.kind.value,
        automaton=DeterministicFiniteAutomaton.finite_language((), name="unsupported-grammar-product"),
        supported=False,
        assumptions=("bounded-compiled-witness-dfa", "tokenizer-encode-normalize-decode-product"),
        issue_codes=issue_codes,
        reason=reason,
    )


def _abstained_report(
    tokenizer_artifact: TokenizerArtifact,
    grammar_artifact: SchemaArtifact | GrammarArtifact,
    tokenizer: TokenizerAdapter,
    *,
    product: _BoundedGrammarProduct | None = None,
    reason: str,
) -> GrammarTokenizerEmptinessReport:
    return GrammarTokenizerEmptinessReport(
        tokenizer_name=tokenizer_artifact.name,
        grammar_name=grammar_artifact.name,
        grammar_kind=product.kind if product is not None else grammar_artifact.kind.value,
        tokenizer_backend=tokenizer.backend.value,
        status=GrammarTokenizerEmptinessStatus.ABSTAINED,
        assumptions=product.assumptions if product is not None else ("tokenizer-encode-normalize-decode-product",),
        reason=reason,
        grammar_state_count=len(product.automaton.states) if product is not None else 0,
        grammar_accept_count=len(product.automaton.accepts) if product is not None else 0,
        grammar_issue_codes=product.issue_codes if product is not None else (),
    )


def _accepted_witnesses(
    automaton: DeterministicFiniteAutomaton,
    *,
    max_candidates: int,
    max_depth: int,
) -> tuple[AutomatonWitness, ...]:
    witnesses: list[AutomatonWitness] = []
    queue = deque([(automaton.start, (), (automaton.start,))])
    seen = {(automaton.start, (), 0)}
    while queue and len(witnesses) < max_candidates:
        state, symbols, states = queue.popleft()
        if state in automaton.accepts:
            witnesses.append(AutomatonWitness(symbols=symbols, states=states))
        if len(symbols) >= max_depth:
            continue
        for symbol in automaton.alphabet:
            target = automaton.step(state, symbol)
            if target is None:
                continue
            next_symbols = (*symbols, symbol)
            key = (target, next_symbols, len(next_symbols))
            if key in seen:
                continue
            seen.add(key)
            queue.append((target, next_symbols, (*states, target)))
    return tuple(witnesses)
