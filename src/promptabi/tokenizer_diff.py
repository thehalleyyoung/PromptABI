"""Differential harness for tokenizer adapter validation.

The harness is intentionally backend-neutral: tests or corpus fixtures build an
expectation from the real tokenizer library, then PromptABI compares its stable
abstraction against that oracle across encode, decode, flags, spans, and
normalization metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .tokenizers import TokenizerAdapter


@dataclass(frozen=True, slots=True)
class TokenizerExpectation:
    """Expected behavior collected from a real tokenizer implementation."""

    token_ids: tuple[int, ...]
    decoded_text: str
    token_texts: tuple[str | None, ...] | None = None
    normalized_text: str | None = None
    special_token_ids: frozenset[int] = frozenset()
    added_token_ids: frozenset[int] = frozenset()
    normalization_steps: tuple[str, ...] | None = None
    byte_spans_required: bool = False
    round_trip_normalized: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(int(token_id) for token_id in self.token_ids))
        if self.token_texts is not None:
            object.__setattr__(self, "token_texts", tuple(self.token_texts))
        object.__setattr__(self, "special_token_ids", frozenset(int(token_id) for token_id in self.special_token_ids))
        object.__setattr__(self, "added_token_ids", frozenset(int(token_id) for token_id in self.added_token_ids))
        if self.normalization_steps is not None:
            object.__setattr__(self, "normalization_steps", tuple(self.normalization_steps))


@dataclass(frozen=True, slots=True)
class TokenizerDifferentialCase:
    """One tokenizer input plus the oracle behavior expected for it."""

    name: str
    text: str
    expectation: TokenizerExpectation
    add_special_tokens: bool = False
    skip_special_tokens: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("differential case name must be non-empty")


@dataclass(frozen=True, slots=True)
class TokenizerDifferentialMismatch:
    """A single field where PromptABI diverged from the oracle."""

    case_name: str
    field: str
    expected: object
    actual: object

    def to_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "field": self.field,
            "expected": _stable_value(self.expected),
            "actual": _stable_value(self.actual),
        }


@dataclass(frozen=True, slots=True)
class TokenizerDifferentialReport:
    """Stable result for one adapter against many differential cases."""

    backend: str
    cases_run: int
    mismatches: tuple[TokenizerDifferentialMismatch, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def assert_ok(self) -> None:
        if not self.ok:
            summary = "; ".join(
                f"{mismatch.case_name}.{mismatch.field}: expected {mismatch.expected!r}, got {mismatch.actual!r}"
                for mismatch in self.mismatches
            )
            raise AssertionError(f"tokenizer differential mismatch for {self.backend}: {summary}")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "cases_run": self.cases_run,
            "ok": self.ok,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }


def run_tokenizer_differential(
    adapter: TokenizerAdapter,
    cases: Sequence[TokenizerDifferentialCase],
) -> TokenizerDifferentialReport:
    """Compare a PromptABI tokenizer adapter with real-library expectations."""

    mismatches: list[TokenizerDifferentialMismatch] = []
    for case in cases:
        expected = case.expectation
        encoded = adapter.encode(case.text, add_special_tokens=case.add_special_tokens)
        decoded = adapter.decode(encoded.token_ids, skip_special_tokens=case.skip_special_tokens)
        round_trip = adapter.round_trip(case.text, add_special_tokens=case.add_special_tokens)

        _compare(mismatches, case.name, "token_ids", expected.token_ids, encoded.token_ids)
        _compare(mismatches, case.name, "decoded_text", expected.decoded_text, decoded.text)
        if expected.token_texts is not None:
            _compare(mismatches, case.name, "token_texts", expected.token_texts, encoded.token_texts)
        if expected.normalized_text is not None:
            _compare(mismatches, case.name, "normalized_text", expected.normalized_text, encoded.normalized_text)
        if expected.normalization_steps is not None:
            _compare(
                mismatches,
                case.name,
                "normalization_steps",
                expected.normalization_steps,
                encoded.normalization_steps,
            )
        if expected.round_trip_normalized is not None:
            _compare(
                mismatches,
                case.name,
                "round_trip_normalized",
                expected.round_trip_normalized,
                round_trip.normalized_match,
            )

        actual_special = frozenset(token.token_id for token in encoded.tokens if token.special)
        actual_added = frozenset(token.token_id for token in encoded.tokens if token.added)
        _compare(mismatches, case.name, "special_token_ids", expected.special_token_ids, actual_special)
        _compare(mismatches, case.name, "added_token_ids", expected.added_token_ids, actual_added)

        if expected.byte_spans_required:
            missing_spans = tuple(token.token_id for token in encoded.tokens if token.byte_span is None)
            _compare(mismatches, case.name, "missing_byte_spans", (), missing_spans)

    return TokenizerDifferentialReport(
        backend=adapter.backend.value,
        cases_run=len(cases),
        mismatches=tuple(mismatches),
    )


def _compare(
    mismatches: list[TokenizerDifferentialMismatch],
    case_name: str,
    field: str,
    expected: object,
    actual: object,
) -> None:
    if expected != actual:
        mismatches.append(
            TokenizerDifferentialMismatch(
                case_name=case_name,
                field=field,
                expected=expected,
                actual=actual,
            )
        )


def _stable_value(value: object) -> object:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    return value
