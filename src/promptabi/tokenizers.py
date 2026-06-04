"""Tokenizer abstraction layer for PromptABI checks.

The adapters in this module intentionally expose only the deterministic pieces
that structural checks need: token ids, token surfaces, byte spans, special and
added-token flags, normalization metadata, and decode round-trip results.
"""

from __future__ import annotations

import importlib
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Sequence

from .artifacts import ArtifactKind, TokenizerArtifact


class TokenizerError(ValueError):
    """Raised when a tokenizer artifact cannot be adapted deterministically."""


class TokenizerBackend(StrEnum):
    """Tokenizer backend families surfaced through the common abstraction."""

    BYTE_LEVEL = "byte-level"
    HUGGINGFACE_TOKENIZERS = "huggingface-tokenizers"
    TIKTOKEN = "tiktoken"
    SENTENCEPIECE = "sentencepiece"


class NormalizationRule(StrEnum):
    """Portable normalization rules PromptABI can model without a backend."""

    IDENTITY = "identity"
    NFC = "nfc"
    NFKC = "nfkc"
    LOWERCASE = "lowercase"
    STRIP = "strip"


@dataclass(frozen=True, slots=True)
class EncodedToken:
    """One token with enough metadata for later structural witnesses."""

    token_id: int
    text: str | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    special: bool = False
    added: bool = False

    @property
    def byte_span(self) -> tuple[int, int] | None:
        if self.byte_start is None or self.byte_end is None:
            return None
        return (self.byte_start, self.byte_end)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.token_id}
        if self.text is not None:
            data["text"] = self.text
        if self.byte_span is not None:
            data["byte_span"] = list(self.byte_span)
        if self.special:
            data["special"] = True
        if self.added:
            data["added"] = True
        return data


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """Stable encode output shared by all tokenizer backends."""

    backend: TokenizerBackend
    input_text: str
    normalized_text: str
    tokens: tuple[EncodedToken, ...]
    normalization_steps: tuple[str, ...] = ()

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token_id for token in self.tokens)

    @property
    def token_texts(self) -> tuple[str | None, ...]:
        return tuple(token.text for token in self.tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "normalization_steps": list(self.normalization_steps),
            "tokens": [token.to_dict() for token in self.tokens],
        }


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Stable decode output shared by all tokenizer backends."""

    backend: TokenizerBackend
    token_ids: tuple[int, ...]
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "token_ids": list(self.token_ids),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """Encode/decode metadata used by differential and drift checks."""

    backend: TokenizerBackend
    input_text: str
    normalized_text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    normalized_match: bool
    exact_match: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "decoded_text": self.decoded_text,
            "token_ids": list(self.token_ids),
            "normalized_match": self.normalized_match,
            "exact_match": self.exact_match,
        }


class TokenizerAdapter(ABC):
    """Common interface implemented by all tokenizer backends."""

    backend: TokenizerBackend

    @abstractmethod
    def encode(self, text: str, *, add_special_tokens: bool = False) -> EncodeResult:
        """Encode text into token ids and structural metadata."""

    @abstractmethod
    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> DecodeResult:
        """Decode token ids back into text."""

    def round_trip(self, text: str, *, add_special_tokens: bool = False) -> RoundTripResult:
        encoded = self.encode(text, add_special_tokens=add_special_tokens)
        decoded = self.decode(encoded.token_ids)
        return RoundTripResult(
            backend=self.backend,
            input_text=text,
            normalized_text=encoded.normalized_text,
            decoded_text=decoded.text,
            token_ids=encoded.token_ids,
            normalized_match=decoded.text == encoded.normalized_text,
            exact_match=decoded.text == text,
        )


class ByteLevelTokenizer(TokenizerAdapter):
    """A deterministic byte alphabet tokenizer with greedy added-token handling."""

    backend = TokenizerBackend.BYTE_LEVEL

    def __init__(
        self,
        *,
        added_tokens: Iterable[str] = (),
        special_tokens: dict[str, int] | None = None,
        normalization: Iterable[NormalizationRule | str] = (),
    ) -> None:
        self._normalization = tuple(NormalizationRule(rule) for rule in normalization)
        self._normalization_cache: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._special_token_to_id = dict(special_tokens or {})
        next_id = 256
        added: dict[str, int] = {}
        for token in sorted(dict.fromkeys(added_tokens), key=lambda item: (-len(item), item)):
            if not token:
                raise TokenizerError("added tokens must be non-empty")
            token_id = self._special_token_to_id.get(token)
            if token_id is None:
                while next_id in self._special_token_to_id.values() or next_id in added.values():
                    next_id += 1
                token_id = next_id
                next_id += 1
            added[token] = token_id
        self._added_token_to_id = added
        self._id_to_added_token = {token_id: token for token, token_id in added.items()}
        self._id_to_special_token = {token_id: token for token, token_id in self._special_token_to_id.items()}

    @property
    def added_tokens(self) -> tuple[str, ...]:
        return tuple(sorted(self._added_token_to_id))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> EncodeResult:
        del add_special_tokens
        normalized, steps = self._normalize(text)
        char_to_byte = _char_to_byte_offsets(normalized)
        encoded: list[EncodedToken] = []
        index = 0
        while index < len(normalized):
            matched = self._match_added_token(normalized, index)
            if matched is not None:
                token_text, token_id = matched
                end = index + len(token_text)
                encoded.append(
                    EncodedToken(
                        token_id=token_id,
                        text=token_text,
                        byte_start=char_to_byte[index],
                        byte_end=char_to_byte[end],
                        special=token_text in self._special_token_to_id,
                        added=True,
                    )
                )
                index = end
                continue

            char = normalized[index]
            start = char_to_byte[index]
            for offset, byte_value in enumerate(char.encode("utf-8")):
                encoded.append(
                    EncodedToken(
                        token_id=byte_value,
                        text=f"0x{byte_value:02x}",
                        byte_start=start + offset,
                        byte_end=start + offset + 1,
                    )
                )
            index += 1
        return EncodeResult(
            backend=self.backend,
            input_text=text,
            normalized_text=normalized,
            tokens=tuple(encoded),
            normalization_steps=steps,
        )

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> DecodeResult:
        chunks: list[str] = []
        pending = bytearray()

        def flush_bytes() -> None:
            if pending:
                chunks.append(bytes(pending).decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in token_ids:
            if token_id in self._id_to_special_token:
                flush_bytes()
                if not skip_special_tokens:
                    chunks.append(self._id_to_special_token[token_id])
                continue
            if token_id in self._id_to_added_token:
                flush_bytes()
                chunks.append(self._id_to_added_token[token_id])
                continue
            if 0 <= token_id <= 255:
                pending.append(token_id)
                continue
            raise TokenizerError(f"byte-level tokenizer cannot decode unknown token id {token_id}")
        flush_bytes()
        return DecodeResult(backend=self.backend, token_ids=tuple(token_ids), text="".join(chunks))

    def _match_added_token(self, text: str, index: int) -> tuple[str, int] | None:
        for token, token_id in self._added_token_to_id.items():
            if text.startswith(token, index):
                return token, token_id
        return None

    def _normalize(self, text: str) -> tuple[str, tuple[str, ...]]:
        cached = self._normalization_cache.get(text)
        if cached is not None:
            return cached
        normalized = apply_normalization(text, self._normalization)
        if len(self._normalization_cache) >= 256:
            self._normalization_cache.pop(next(iter(self._normalization_cache)))
        self._normalization_cache[text] = normalized
        return normalized


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    """Adapter for local ``tokenizers`` JSON files and tokenizer directories."""

    backend = TokenizerBackend.HUGGINGFACE_TOKENIZERS

    def __init__(self, tokenizer, *, added_tokens: Iterable[str] = ()) -> None:
        self._tokenizer = tokenizer
        self._declared_added_tokens = frozenset(added_tokens)

    @classmethod
    def from_file(cls, path: str | Path, *, added_tokens: Iterable[str] = ()) -> "HuggingFaceTokenizerAdapter":
        tokenizers = _import_optional("tokenizers", "Install tokenizers to load Hugging Face tokenizer.json files.")
        return cls(tokenizers.Tokenizer.from_file(str(path)), added_tokens=added_tokens)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> EncodeResult:
        normalized, steps = self._normalize(text)
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        char_to_byte = _char_to_byte_offsets(text)
        special_mask = tuple(getattr(encoded, "special_tokens_mask", ()) or ())
        added_token_ids = self._added_token_ids()
        tokens: list[EncodedToken] = []
        for index, token_id in enumerate(encoded.ids):
            char_start, char_end = encoded.offsets[index]
            byte_start = char_to_byte[char_start] if 0 <= char_start < len(char_to_byte) else None
            byte_end = char_to_byte[char_end] if 0 <= char_end < len(char_to_byte) else None
            token_text = encoded.tokens[index]
            special = bool(special_mask[index]) if index < len(special_mask) else False
            tokens.append(
                EncodedToken(
                    token_id=token_id,
                    text=token_text,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    special=special,
                    added=not special and (token_id in added_token_ids or token_text in self._declared_added_tokens),
                )
            )
        return EncodeResult(
            backend=self.backend,
            input_text=text,
            normalized_text=normalized,
            tokens=tuple(tokens),
            normalization_steps=steps,
        )

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> DecodeResult:
        text = self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
        return DecodeResult(backend=self.backend, token_ids=tuple(token_ids), text=text)

    def _normalize(self, text: str) -> tuple[str, tuple[str, ...]]:
        normalizer = getattr(self._tokenizer, "normalizer", None)
        if normalizer is None:
            return text, ()
        normalize = getattr(normalizer, "normalize_str", None)
        if normalize is None:
            return text, (repr(normalizer),)
        normalized = normalize(text)
        step = repr(normalizer)
        return normalized, (step,) if normalized != text or step else ()

    def _added_token_ids(self) -> frozenset[int]:
        decoder = getattr(self._tokenizer, "get_added_tokens_decoder", None)
        if decoder is None:
            return frozenset()
        return frozenset(int(token_id) for token_id in decoder().keys())


class TiktokenAdapter(TokenizerAdapter):
    """Adapter for installed ``tiktoken`` encodings such as ``cl100k_base``."""

    backend = TokenizerBackend.TIKTOKEN

    def __init__(self, encoding) -> None:
        self._encoding = encoding
        self._special_id_to_text = {token_id: text for text, token_id in encoding._special_tokens.items()}

    @classmethod
    def from_encoding_name(cls, name: str) -> "TiktokenAdapter":
        tiktoken = _import_optional("tiktoken", "Install tiktoken to use tiktoken tokenizer adapters.")
        return cls(tiktoken.get_encoding(name))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> EncodeResult:
        allowed_special = "all" if add_special_tokens else set()
        ids = tuple(self._encoding.encode(text, allowed_special=allowed_special, disallowed_special=()))
        tokens: list[EncodedToken] = []
        byte_cursor = 0
        for token_id in ids:
            special_text = self._special_id_to_text.get(token_id)
            if special_text is not None:
                token_bytes = special_text.encode("utf-8")
                token_text = special_text
                special = True
            else:
                token_bytes = self._encoding.decode_single_token_bytes(token_id)
                token_text = token_bytes.decode("utf-8", errors="replace")
                special = False
            byte_start = byte_cursor
            byte_end = byte_cursor + len(token_bytes)
            byte_cursor = byte_end
            tokens.append(
                EncodedToken(
                    token_id=token_id,
                    text=token_text,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    special=special,
                    added=special,
                )
            )
        return EncodeResult(
            backend=self.backend,
            input_text=text,
            normalized_text=text,
            tokens=tuple(tokens),
        )

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> DecodeResult:
        if skip_special_tokens:
            token_ids = tuple(token_id for token_id in token_ids if token_id not in self._special_id_to_text)
        return DecodeResult(
            backend=self.backend,
            token_ids=tuple(token_ids),
            text=self._encoding.decode(list(token_ids)),
        )


class SentencePieceAdapter(TokenizerAdapter):
    """Adapter for local SentencePiece ``.model`` files."""

    backend = TokenizerBackend.SENTENCEPIECE

    def __init__(self, processor) -> None:
        self._processor = processor

    @classmethod
    def from_file(cls, path: str | Path) -> "SentencePieceAdapter":
        sentencepiece = _import_optional(
            "sentencepiece",
            "Install sentencepiece to load SentencePiece .model tokenizer artifacts.",
        )
        processor = sentencepiece.SentencePieceProcessor()
        if not processor.Load(str(path)):
            raise TokenizerError(f"could not load SentencePiece model: {path}")
        return cls(processor)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> EncodeResult:
        del add_special_tokens
        ids = tuple(int(token_id) for token_id in self._processor.EncodeAsIds(text))
        pieces = tuple(str(piece) for piece in self._processor.EncodeAsPieces(text))
        spans = _piece_byte_spans(text, pieces)
        tokens = tuple(
            EncodedToken(
                token_id=token_id,
                text=pieces[index],
                byte_start=spans[index][0] if spans[index] is not None else None,
                byte_end=spans[index][1] if spans[index] is not None else None,
                special=self._processor.IsControl(token_id) or self._processor.IsUnknown(token_id),
            )
            for index, token_id in enumerate(ids)
        )
        return EncodeResult(
            backend=self.backend,
            input_text=text,
            normalized_text=text,
            tokens=tokens,
        )

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> DecodeResult:
        if skip_special_tokens:
            token_ids = tuple(
                token_id
                for token_id in token_ids
                if not self._processor.IsControl(int(token_id)) and not self._processor.IsUnknown(int(token_id))
            )
        return DecodeResult(
            backend=self.backend,
            token_ids=tuple(int(token_id) for token_id in token_ids),
            text=self._processor.DecodeIds(list(token_ids)),
        )


def load_tokenizer(artifact: TokenizerArtifact) -> TokenizerAdapter:
    """Load a local tokenizer artifact into the common adapter interface."""

    if artifact.kind is not ArtifactKind.TOKENIZER:
        raise TokenizerError(f"expected tokenizer artifact, got {artifact.kind.value}")
    family = (artifact.family or "").lower()
    metadata = dict(artifact.metadata)
    if family in {"byte", "byte-level", "byte-bpe", "byte-level-bpe"}:
        normalization = metadata.get("normalization", ())
        if isinstance(normalization, str):
            normalization = (normalization,)
        if not isinstance(normalization, tuple | list):
            raise TokenizerError("tokenizer metadata 'normalization' must be a string or list of strings")
        return ByteLevelTokenizer(added_tokens=artifact.added_tokens, normalization=normalization)
    if family == "tiktoken":
        encoding_name = metadata.get("encoding") or metadata.get("encoding_name")
        if not isinstance(encoding_name, str) or not encoding_name:
            raise TokenizerError("tiktoken tokenizer artifacts must set metadata.encoding")
        return TiktokenAdapter.from_encoding_name(encoding_name)

    if artifact.location.path is None:
        raise TokenizerError("tokenizer loading currently requires a local path unless family is byte-level or tiktoken")
    path = Path(artifact.location.path)
    if path.is_dir():
        tokenizer_json = path / "tokenizer.json"
        sentencepiece_model = path / "tokenizer.model"
        if tokenizer_json.is_file():
            return HuggingFaceTokenizerAdapter.from_file(tokenizer_json, added_tokens=artifact.added_tokens)
        if sentencepiece_model.is_file():
            return SentencePieceAdapter.from_file(sentencepiece_model)
        raise TokenizerError(f"tokenizer directory lacks tokenizer.json or tokenizer.model: {path}")
    if path.suffix == ".model" or family == "sentencepiece":
        return SentencePieceAdapter.from_file(path)
    if path.suffix == ".json" or family in {"huggingface", "tokenizers", "hf"}:
        return HuggingFaceTokenizerAdapter.from_file(path, added_tokens=artifact.added_tokens)
    raise TokenizerError(f"unsupported tokenizer artifact path: {path}")


def apply_normalization(
    text: str,
    rules: Iterable[NormalizationRule | str],
) -> tuple[str, tuple[str, ...]]:
    """Apply portable normalization rules and report the rules that ran."""

    normalized = text
    steps: list[str] = []
    for raw_rule in rules:
        rule = NormalizationRule(raw_rule)
        if rule is NormalizationRule.IDENTITY:
            steps.append(rule.value)
            continue
        if rule is NormalizationRule.NFC:
            normalized = unicodedata.normalize("NFC", normalized)
        elif rule is NormalizationRule.NFKC:
            normalized = unicodedata.normalize("NFKC", normalized)
        elif rule is NormalizationRule.LOWERCASE:
            normalized = normalized.lower()
        elif rule is NormalizationRule.STRIP:
            normalized = normalized.strip()
        steps.append(rule.value)
    return normalized, tuple(steps)


def _char_to_byte_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return tuple(offsets)


def _piece_byte_spans(text: str, pieces: Sequence[str]) -> tuple[tuple[int, int] | None, ...]:
    encoded_text = text.encode("utf-8")
    cursor = 0
    spans: list[tuple[int, int] | None] = []
    for piece in pieces:
        surface = piece.replace("▁", " ")
        if not surface:
            spans.append(None)
            continue
        piece_bytes = surface.encode("utf-8")
        index = encoded_text.find(piece_bytes, cursor)
        if index < 0:
            stripped = surface.lstrip()
            piece_bytes = stripped.encode("utf-8")
            index = encoded_text.find(piece_bytes, cursor) if stripped else -1
        if index < 0:
            spans.append(None)
            continue
        end = index + len(piece_bytes)
        spans.append((index, end))
        cursor = end
    return tuple(spans)


def _import_optional(module: str, install_hint: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise TokenizerError(install_hint) from exc
