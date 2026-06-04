"""Stop-policy analysis against concrete tokenizer behavior."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import StopPolicyArtifact
from .tokenizers import EncodeResult, TokenizerAdapter


@dataclass(frozen=True, slots=True)
class StopSequenceAnalysis:
    """Tokenizer-backed alignment data for one configured stop string."""

    stop_sequence: str
    normalized_sequence: str
    utf8_bytes: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_texts: tuple[str | None, ...]
    byte_spans: tuple[tuple[int, int] | None, ...]
    decoded_text: str
    exact_round_trip: bool
    normalized_round_trip: bool
    normalization_steps: tuple[str, ...] = ()
    special_token_ids: tuple[int, ...] = ()
    added_token_ids: tuple[int, ...] = ()

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def multi_token(self) -> bool:
        return self.token_count > 1

    @property
    def normalization_changed(self) -> bool:
        return self.normalized_sequence != self.stop_sequence

    @property
    def has_special_interaction(self) -> bool:
        return bool(self.special_token_ids or self.added_token_ids)

    def token_summary(self) -> str:
        texts = tuple("<none>" if text is None else text for text in self.token_texts)
        return f"ids={self.token_ids}; texts={texts}; byte_spans={self.byte_spans}"


@dataclass(frozen=True, slots=True)
class StopTokenIdAnalysis:
    """Tokenizer-backed reachability data for one configured stop token id."""

    token_id: int
    decodable: bool
    decoded_text: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StopCollision:
    """A proper prefix/suffix or normalization collision between stops."""

    level: str
    relation: str
    shorter: str
    longer: str
    witness: str


@dataclass(frozen=True, slots=True)
class StopPolicyTokenizerAnalysisReport:
    """Analysis of one stop policy under one tokenizer."""

    tokenizer_backend: str
    sequences: tuple[StopSequenceAnalysis, ...]
    token_ids: tuple[StopTokenIdAnalysis, ...]
    collisions: tuple[StopCollision, ...] = ()
    normalization_collisions: tuple[StopCollision, ...] = ()

    @property
    def unreachable_token_ids(self) -> tuple[StopTokenIdAnalysis, ...]:
        return tuple(item for item in self.token_ids if not item.decodable)

    @property
    def special_interactions(self) -> tuple[StopSequenceAnalysis, ...]:
        return tuple(item for item in self.sequences if item.has_special_interaction)

    @property
    def lossy_or_normalizing_sequences(self) -> tuple[StopSequenceAnalysis, ...]:
        return tuple(
            item
            for item in self.sequences
            if item.normalization_changed or not item.normalized_round_trip
        )

    @property
    def multi_token_sequences(self) -> tuple[StopSequenceAnalysis, ...]:
        return tuple(item for item in self.sequences if item.multi_token)


def analyze_stop_policy_tokenizer(
    stop_policy: StopPolicyArtifact,
    tokenizer: TokenizerAdapter,
) -> StopPolicyTokenizerAnalysisReport:
    """Analyze configured stop strings and token ids under a tokenizer.

    The analysis deliberately avoids claiming that a string stop is unreachable
    merely because it does not round-trip through an encoder. Provider stop
    strings are commonly matched against decoded text streams. The sound
    unreachable finding here is narrower: a configured stop token id that the
    selected tokenizer cannot decode at all.
    """

    sequences = tuple(_analyze_sequence(sequence, tokenizer) for sequence in stop_policy.stop_sequences)
    token_ids = tuple(_analyze_token_id(token_id, tokenizer) for token_id in stop_policy.stop_token_ids)
    return StopPolicyTokenizerAnalysisReport(
        tokenizer_backend=tokenizer.backend.value,
        sequences=sequences,
        token_ids=token_ids,
        collisions=_prefix_suffix_collisions(sequences),
        normalization_collisions=_normalization_collisions(sequences),
    )


def _analyze_sequence(sequence: str, tokenizer: TokenizerAdapter) -> StopSequenceAnalysis:
    encoded = tokenizer.encode(sequence, add_special_tokens=False)
    decoded = tokenizer.decode(encoded.token_ids).text
    return StopSequenceAnalysis(
        stop_sequence=sequence,
        normalized_sequence=encoded.normalized_text,
        utf8_bytes=tuple(sequence.encode("utf-8")),
        token_ids=encoded.token_ids,
        token_texts=encoded.token_texts,
        byte_spans=tuple(token.byte_span for token in encoded.tokens),
        decoded_text=decoded,
        exact_round_trip=decoded == sequence,
        normalized_round_trip=decoded == encoded.normalized_text,
        normalization_steps=encoded.normalization_steps,
        special_token_ids=_token_ids_with_flag(encoded, "special"),
        added_token_ids=_token_ids_with_flag(encoded, "added"),
    )


def _analyze_token_id(token_id: int, tokenizer: TokenizerAdapter) -> StopTokenIdAnalysis:
    try:
        decoded = tokenizer.decode((token_id,)).text
    except Exception as exc:  # noqa: BLE001 - converted to deterministic analysis data.
        return StopTokenIdAnalysis(token_id=token_id, decodable=False, error=f"{type(exc).__name__}: {exc}")
    return StopTokenIdAnalysis(token_id=token_id, decodable=True, decoded_text=decoded)


def _token_ids_with_flag(encoded: EncodeResult, flag_name: str) -> tuple[int, ...]:
    return tuple(
        token.token_id
        for token in encoded.tokens
        if bool(getattr(token, flag_name))
    )


def _prefix_suffix_collisions(sequences: tuple[StopSequenceAnalysis, ...]) -> tuple[StopCollision, ...]:
    collisions: list[StopCollision] = []
    for left_index, left in enumerate(sequences):
        for right in sequences[left_index + 1 :]:
            collisions.extend(_collisions_for_pair(left, right))
    return tuple(sorted(collisions, key=lambda item: (item.level, item.relation, item.shorter, item.longer)))


def _collisions_for_pair(left: StopSequenceAnalysis, right: StopSequenceAnalysis) -> tuple[StopCollision, ...]:
    collisions: list[StopCollision] = []
    for level, left_value, right_value in (
        ("string", left.stop_sequence, right.stop_sequence),
        ("byte", left.utf8_bytes, right.utf8_bytes),
        ("token", left.token_ids, right.token_ids),
    ):
        collisions.extend(_proper_prefix_suffix(level, left.stop_sequence, left_value, right.stop_sequence, right_value))
    return tuple(collisions)


def _proper_prefix_suffix(
    level: str,
    left_name: str,
    left_value,
    right_name: str,
    right_value,
) -> tuple[StopCollision, ...]:
    collisions: list[StopCollision] = []
    if len(left_value) < len(right_value):
        collisions.extend(_ordered_prefix_suffix(level, left_name, left_value, right_name, right_value))
    elif len(right_value) < len(left_value):
        collisions.extend(_ordered_prefix_suffix(level, right_name, right_value, left_name, left_value))
    return tuple(collisions)


def _ordered_prefix_suffix(
    level: str,
    shorter_name: str,
    shorter_value,
    longer_name: str,
    longer_value,
) -> tuple[StopCollision, ...]:
    collisions: list[StopCollision] = []
    if _starts_with(longer_value, shorter_value):
        collisions.append(
            StopCollision(
                level=level,
                relation="prefix",
                shorter=shorter_name,
                longer=longer_name,
                witness=repr(shorter_value),
            )
        )
    if _ends_with(longer_value, shorter_value):
        collisions.append(
            StopCollision(
                level=level,
                relation="suffix",
                shorter=shorter_name,
                longer=longer_name,
                witness=repr(shorter_value),
            )
        )
    return tuple(collisions)


def _starts_with(value, prefix) -> bool:
    return value[: len(prefix)] == prefix


def _ends_with(value, suffix) -> bool:
    return value[-len(suffix) :] == suffix


def _normalization_collisions(sequences: tuple[StopSequenceAnalysis, ...]) -> tuple[StopCollision, ...]:
    collisions: list[StopCollision] = []
    for key_name, key_fn in (
        ("normalized-string", lambda item: item.normalized_sequence),
        ("decoded-text", lambda item: item.decoded_text),
        ("token", lambda item: item.token_ids),
    ):
        buckets: dict[object, list[str]] = {}
        for sequence in sequences:
            key = key_fn(sequence)
            buckets.setdefault(key, []).append(sequence.stop_sequence)
        for key, names in buckets.items():
            unique = tuple(sorted(dict.fromkeys(names)))
            if len(unique) < 2:
                continue
            collisions.append(
                StopCollision(
                    level=key_name,
                    relation="same-surface",
                    shorter=unique[0],
                    longer=unique[-1],
                    witness=repr(key),
                )
            )
    return tuple(sorted(collisions, key=lambda item: (item.level, item.shorter, item.longer)))
