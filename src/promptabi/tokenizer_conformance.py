"""Maintained tokenizer-family conformance suite replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .tokenizer_diff import (
    TokenizerDifferentialCase,
    TokenizerDifferentialReport,
    TokenizerExpectation,
    run_tokenizer_differential,
)
from .tokenizers import (
    ByteLevelTokenizer,
    HuggingFaceTokenizerAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenizerAdapter,
    TokenizerError,
)


TOKENIZER_CONFORMANCE_VERSION = 1
DEFAULT_TOKENIZER_CONFORMANCE_SUITE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "tokenizer_conformance" / "suite.json"
)
REQUIRED_TOKENIZER_FAMILIES = ("bpe", "unigram", "byte-fallback")
REQUIRED_TOKENIZER_FEATURES = (
    "added-tokens",
    "normalized-spaces",
    "special-token-splitting",
    "detokenization",
)


class TokenizerConformanceError(ValueError):
    """Raised when a tokenizer-family conformance suite is malformed."""


@dataclass(frozen=True, slots=True)
class TokenizerConformanceCaseReport:
    """Replay result for one tokenizer-family conformance case."""

    case_id: str
    family: str
    backend: str
    features: tuple[str, ...]
    differential_report: TokenizerDifferentialReport

    @property
    def sample_count(self) -> int:
        return self.differential_report.cases_run

    @property
    def passed(self) -> bool:
        return self.sample_count > 0 and self.differential_report.ok

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "backend": self.backend,
            "features": list(self.features),
            "sample_count": self.sample_count,
            "passed": self.passed,
            "differential": self.differential_report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TokenizerFamilyCoverage:
    """Coverage and replay status for one tokenizer family."""

    family: str
    case_ids: tuple[str, ...]
    backends: tuple[str, ...]
    features: tuple[str, ...]
    sample_count: int
    passed_cases: int
    mismatches: int

    @property
    def passed(self) -> bool:
        return bool(self.case_ids) and self.sample_count > 0 and self.mismatches == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "passed": self.passed,
            "case_ids": list(self.case_ids),
            "backends": list(self.backends),
            "features": list(self.features),
            "sample_count": self.sample_count,
            "passed_cases": self.passed_cases,
            "mismatches": self.mismatches,
        }


@dataclass(frozen=True, slots=True)
class TokenizerConformanceReport:
    """Release-grade replay report for tokenizer-family conformance suites."""

    suite_version: int
    cases: tuple[TokenizerConformanceCaseReport, ...]
    family_coverage: tuple[TokenizerFamilyCoverage, ...]
    required_families: tuple[str, ...] = REQUIRED_TOKENIZER_FAMILIES
    required_features: tuple[str, ...] = REQUIRED_TOKENIZER_FEATURES

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def sample_count(self) -> int:
        return sum(case.sample_count for case in self.cases)

    @property
    def all_cases_passed(self) -> bool:
        return (
            self.case_count > 0
            and all(case.passed for case in self.cases)
            and not self.missing_families
            and not self.missing_features
            and all(coverage.passed for coverage in self.family_coverage)
        )

    @property
    def missing_families(self) -> tuple[str, ...]:
        observed = {coverage.family for coverage in self.family_coverage if coverage.case_ids}
        return tuple(family for family in self.required_families if family not in observed)

    @property
    def missing_features(self) -> tuple[str, ...]:
        observed = {feature for case in self.cases for feature in case.features}
        return tuple(feature for feature in self.required_features if feature not in observed)

    @property
    def manifest_sha256(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_version": TOKENIZER_CONFORMANCE_VERSION,
            "suite_version": self.suite_version,
            "case_count": self.case_count,
            "sample_count": self.sample_count,
            "all_cases_passed": self.all_cases_passed,
            "required_families": list(self.required_families),
            "required_features": list(self.required_features),
            "missing_families": list(self.missing_families),
            "missing_features": list(self.missing_features),
            "family_coverage": [coverage.to_dict() for coverage in self.family_coverage],
            "cases": [case.to_dict() for case in self.cases],
        }
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def build_tokenizer_conformance_report(path: str | Path | None = None) -> TokenizerConformanceReport:
    """Replay the maintained tokenizer-family suite against real tokenizer code."""

    suite_path = Path(path) if path is not None else DEFAULT_TOKENIZER_CONFORMANCE_SUITE
    raw = _load_suite(suite_path)
    version = raw.get("version")
    if not isinstance(version, int) or version <= 0:
        raise TokenizerConformanceError("tokenizer conformance suite requires a positive integer version")
    cases_data = raw.get("cases")
    if not isinstance(cases_data, list) or not cases_data:
        raise TokenizerConformanceError("tokenizer conformance suite requires a non-empty cases array")
    cases = tuple(_replay_case(_mapping(case, "case")) for case in cases_data)
    return TokenizerConformanceReport(
        suite_version=version,
        cases=cases,
        family_coverage=_family_coverage(cases),
    )


def write_tokenizer_conformance_manifest(path: str | Path, *, suite_path: str | Path | None = None) -> dict[str, object]:
    """Write the conformance report manifest as deterministic JSON."""

    report = build_tokenizer_conformance_report(suite_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_tokenizer_conformance_json(report), encoding="utf-8")
    return report.to_dict()


def render_tokenizer_conformance_json(report: TokenizerConformanceReport | None = None) -> str:
    """Render tokenizer conformance as deterministic JSON."""

    resolved = report or build_tokenizer_conformance_report()
    return json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"


def render_tokenizer_conformance_text(report: TokenizerConformanceReport | None = None) -> str:
    """Render a concise tokenizer-family conformance replay summary."""

    resolved = report or build_tokenizer_conformance_report()
    lines = [
        "PromptABI tokenizer family conformance",
        f"status: {'PASS' if resolved.all_cases_passed else 'FAIL'}",
        f"cases: {resolved.case_count}",
        f"samples: {resolved.sample_count}",
        f"required families: {', '.join(resolved.required_families)}",
        f"required features: {', '.join(resolved.required_features)}",
        f"manifest_sha256: {resolved.manifest_sha256}",
    ]
    for coverage in resolved.family_coverage:
        status = "PASS" if coverage.passed else "FAIL"
        lines.append(
            f"- {coverage.family}: {status} "
            f"({len(coverage.case_ids)} case(s), {coverage.sample_count} sample(s), "
            f"{coverage.mismatches} mismatch(es); backends: {', '.join(coverage.backends)})"
        )
    if resolved.missing_families:
        lines.append(f"missing families: {', '.join(resolved.missing_families)}")
    if resolved.missing_features:
        lines.append(f"missing features: {', '.join(resolved.missing_features)}")
    return "\n".join(lines) + "\n"


def _load_suite(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TokenizerConformanceError(f"cannot read tokenizer conformance suite {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TokenizerConformanceError(f"tokenizer conformance suite is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TokenizerConformanceError("tokenizer conformance suite root must be an object")
    return raw


def _replay_case(data: dict[str, Any]) -> TokenizerConformanceCaseReport:
    case_id = _required_str(data, "id")
    family = _required_str(data, "family")
    backend = _required_str(data, "backend")
    features = tuple(sorted(_str_sequence(data.get("features", ()), "features")))
    if not features:
        raise TokenizerConformanceError(f"{case_id}: features must be non-empty")
    sample_data = data.get("samples")
    if not isinstance(sample_data, list) or not sample_data:
        raise TokenizerConformanceError(f"{case_id}: samples must be a non-empty array")
    try:
        adapter, cases = _adapter_and_cases(data, sample_data)
    except (ImportError, TokenizerError, OSError, ValueError) as exc:
        raise TokenizerConformanceError(f"{case_id}: cannot replay {backend}: {exc}") from exc
    report = run_tokenizer_differential(adapter, cases)
    return TokenizerConformanceCaseReport(
        case_id=case_id,
        family=family,
        backend=backend,
        features=features,
        differential_report=report,
    )


def _adapter_and_cases(
    data: dict[str, Any],
    sample_data: list[Any],
) -> tuple[TokenizerAdapter, tuple[TokenizerDifferentialCase, ...]]:
    backend = _required_str(data, "backend")
    if backend == "byte-level":
        adapter = ByteLevelTokenizer(
            added_tokens=_str_sequence(data.get("added_tokens", ()), "added_tokens"),
            special_tokens=_int_mapping(data.get("special_tokens", {}), "special_tokens"),
            normalization=_str_sequence(data.get("normalization", ()), "normalization"),
        )
        return adapter, tuple(_byte_level_case(sample) for sample in sample_data)
    if backend == "huggingface-byte-bpe":
        return _huggingface_bpe_adapter_and_cases(data, sample_data)
    if backend == "sentencepiece-unigram":
        return _sentencepiece_adapter_and_cases(data, sample_data)
    if backend == "tiktoken":
        return _tiktoken_adapter_and_cases(data, sample_data)
    raise TokenizerConformanceError(f"unsupported tokenizer conformance backend: {backend}")


def _huggingface_bpe_adapter_and_cases(
    data: dict[str, Any],
    sample_data: list[Any],
) -> tuple[TokenizerAdapter, tuple[TokenizerDifferentialCase, ...]]:
    tokenizers = __import__("tokenizers")
    from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers

    tokenizer = tokenizers.Tokenizer(models.BPE(unk_token=_str(data.get("unk_token"), "[UNK]")))
    normalizer_names = _str_sequence(data.get("normalization", ()), "normalization")
    normalizer_steps = []
    for name in normalizer_names:
        if name == "nfc":
            normalizer_steps.append(normalizers.NFC())
        elif name == "nfkc":
            normalizer_steps.append(normalizers.NFKC())
        elif name == "lowercase":
            normalizer_steps.append(normalizers.Lowercase())
        elif name == "strip":
            normalizer_steps.append(normalizers.Strip())
        else:
            raise TokenizerConformanceError(f"unsupported Hugging Face normalizer: {name}")
    if normalizer_steps:
        tokenizer.normalizer = normalizers.Sequence(normalizer_steps)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=bool(data.get("add_prefix_space", False)))
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = _str_sequence(data.get("special_tokens", ()), "special_tokens")
    trainer = trainers.BpeTrainer(
        vocab_size=int(data.get("vocab_size", 128)),
        special_tokens=list(special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_str_sequence(data.get("training_corpus", ()), "training_corpus"), trainer=trainer)
    added_tokens = _str_sequence(data.get("added_tokens", ()), "added_tokens")
    if added_tokens:
        tokenizer.add_tokens(list(added_tokens))
    post_processor = data.get("post_processor")
    if post_processor is not None:
        post = _mapping(post_processor, "post_processor")
        special_pairs = [(token, tokenizer.token_to_id(token)) for token in _str_sequence(post.get("special_tokens", ()), "special_tokens")]
        tokenizer.post_processor = processors.TemplateProcessing(
            single=_required_str(post, "single"),
            special_tokens=special_pairs,
        )
    adapter = HuggingFaceTokenizerAdapter(tokenizer, added_tokens=added_tokens)
    cases = tuple(_runtime_case(sample, _hf_expectation(tokenizer, sample), added_tokens=added_tokens) for sample in sample_data)
    return adapter, cases


def _sentencepiece_adapter_and_cases(
    data: dict[str, Any],
    sample_data: list[Any],
) -> tuple[TokenizerAdapter, tuple[TokenizerDifferentialCase, ...]]:
    sentencepiece = __import__("sentencepiece")
    corpus = "\n".join(_str_sequence(data.get("training_corpus", ()), "training_corpus"))
    if not corpus:
        raise TokenizerConformanceError("sentencepiece-unigram cases require training_corpus")
    with tempfile.TemporaryDirectory(prefix="promptabi-spm-") as temp_dir:
        temp = Path(temp_dir)
        corpus_path = temp / "corpus.txt"
        model_prefix = temp / "spm"
        corpus_path.write_text(corpus, encoding="utf-8")
        sentencepiece.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=int(data.get("vocab_size", 64)),
            model_type="unigram",
            character_coverage=float(data.get("character_coverage", 1.0)),
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
            input_sentence_size=0,
            num_threads=1,
            minloglevel=2,
        )
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_prefix.with_suffix(".model")))
        adapter = SentencePieceAdapter(processor)
        cases = tuple(_runtime_case(sample, _sentencepiece_expectation(processor, sample)) for sample in sample_data)
        return adapter, cases


def _tiktoken_adapter_and_cases(
    data: dict[str, Any],
    sample_data: list[Any],
) -> tuple[TokenizerAdapter, tuple[TokenizerDifferentialCase, ...]]:
    tiktoken = __import__("tiktoken")
    encoding = tiktoken.get_encoding(_str(data.get("encoding"), "cl100k_base"))
    adapter = TiktokenAdapter(encoding)
    cases = tuple(_runtime_case(sample, _tiktoken_expectation(encoding, sample)) for sample in sample_data)
    return adapter, cases


def _byte_level_case(sample: Any) -> TokenizerDifferentialCase:
    data = _mapping(sample, "sample")
    expectation_data = _mapping(data.get("expectation"), "expectation")
    return TokenizerDifferentialCase(
        name=_required_str(data, "name"),
        text=_sample_text(data),
        add_special_tokens=bool(data.get("add_special_tokens", False)),
        skip_special_tokens=bool(data.get("skip_special_tokens", False)),
        expectation=_expectation_from_mapping(expectation_data),
    )


def _runtime_case(sample: Any, expectation: TokenizerExpectation, *, added_tokens: Sequence[str] = ()) -> TokenizerDifferentialCase:
    data = _mapping(sample, "sample")
    return TokenizerDifferentialCase(
        name=_required_str(data, "name"),
        text=_sample_text(data),
        add_special_tokens=bool(data.get("add_special_tokens", False)),
        skip_special_tokens=bool(data.get("skip_special_tokens", False)),
        expectation=_with_added_token_overrides(expectation, data, added_tokens),
    )


def _hf_expectation(tokenizer: Any, sample: Any) -> TokenizerExpectation:
    data = _mapping(sample, "sample")
    text = _sample_text(data)
    encoded = tokenizer.encode(text, add_special_tokens=bool(data.get("add_special_tokens", False)))
    normalizer = getattr(tokenizer, "normalizer", None)
    normalized = normalizer.normalize_str(text) if normalizer is not None else text
    special_ids = frozenset(
        int(token_id)
        for token_id, special in zip(encoded.ids, encoded.special_tokens_mask, strict=True)
        if special
    )
    return TokenizerExpectation(
        token_ids=tuple(encoded.ids),
        token_texts=tuple(encoded.tokens),
        decoded_text=tokenizer.decode(encoded.ids, skip_special_tokens=bool(data.get("skip_special_tokens", False))),
        normalized_text=normalized,
        special_token_ids=special_ids,
        byte_spans_required=bool(data.get("byte_spans_required", False)),
        round_trip_normalized=data.get("round_trip_normalized"),
    )


def _sentencepiece_expectation(processor: Any, sample: Any) -> TokenizerExpectation:
    data = _mapping(sample, "sample")
    text = _sample_text(data)
    ids = tuple(int(token_id) for token_id in processor.EncodeAsIds(text))
    if bool(data.get("skip_special_tokens", False)):
        decode_ids = tuple(token_id for token_id in ids if not processor.IsControl(token_id) and not processor.IsUnknown(token_id))
    else:
        decode_ids = ids
    return TokenizerExpectation(
        token_ids=ids,
        token_texts=tuple(str(piece) for piece in processor.EncodeAsPieces(text)),
        decoded_text=processor.DecodeIds(list(decode_ids)),
        special_token_ids=frozenset(token_id for token_id in ids if processor.IsControl(token_id) or processor.IsUnknown(token_id)),
        byte_spans_required=bool(data.get("byte_spans_required", False)),
        round_trip_normalized=data.get("round_trip_normalized"),
    )


def _tiktoken_expectation(encoding: Any, sample: Any) -> TokenizerExpectation:
    data = _mapping(sample, "sample")
    text = _sample_text(data)
    allowed_special = "all" if bool(data.get("add_special_tokens", False)) else set()
    ids = tuple(encoding.encode(text, allowed_special=allowed_special, disallowed_special=()))
    special_map = getattr(encoding, "_special_tokens", {})
    special_ids = frozenset(int(token_id) for token, token_id in special_map.items() if token in text and token_id in ids)
    if bool(data.get("skip_special_tokens", False)):
        decode_ids = tuple(token_id for token_id in ids if token_id not in special_ids)
    else:
        decode_ids = ids
    return TokenizerExpectation(
        token_ids=ids,
        decoded_text=encoding.decode(list(decode_ids)),
        special_token_ids=special_ids,
        added_token_ids=special_ids,
        byte_spans_required=bool(data.get("byte_spans_required", False)),
        round_trip_normalized=data.get("round_trip_normalized"),
    )


def _with_added_token_overrides(
    expectation: TokenizerExpectation,
    sample: dict[str, Any],
    added_tokens: Sequence[str],
) -> TokenizerExpectation:
    explicit = sample.get("expected_added_token_ids")
    if explicit is not None:
        added_ids = frozenset(int(token_id) for token_id in _int_sequence(explicit, "expected_added_token_ids"))
    elif added_tokens:
        added_ids = frozenset(
            token_id
            for token_id, token_text in zip(expectation.token_ids, expectation.token_texts or (), strict=True)
            if token_text in set(added_tokens)
        )
    else:
        added_ids = expectation.added_token_ids
    return TokenizerExpectation(
        token_ids=expectation.token_ids,
        decoded_text=expectation.decoded_text,
        token_texts=expectation.token_texts,
        normalized_text=expectation.normalized_text,
        special_token_ids=expectation.special_token_ids,
        added_token_ids=added_ids,
        normalization_steps=expectation.normalization_steps,
        byte_spans_required=expectation.byte_spans_required,
        round_trip_normalized=expectation.round_trip_normalized,
    )


def _expectation_from_mapping(data: dict[str, Any]) -> TokenizerExpectation:
    return TokenizerExpectation(
        token_ids=tuple(_int_sequence(data.get("token_ids"), "token_ids")),
        token_texts=tuple(data["token_texts"]) if "token_texts" in data else None,
        decoded_text=_required_str(data, "decoded_text"),
        normalized_text=str(data["normalized_text"]) if "normalized_text" in data else None,
        special_token_ids=frozenset(_int_sequence(data.get("special_token_ids", ()), "special_token_ids")),
        added_token_ids=frozenset(_int_sequence(data.get("added_token_ids", ()), "added_token_ids")),
        normalization_steps=tuple(_str_sequence(data["normalization_steps"], "normalization_steps"))
        if "normalization_steps" in data
        else None,
        byte_spans_required=bool(data.get("byte_spans_required", False)),
        round_trip_normalized=data.get("round_trip_normalized"),
    )


def _family_coverage(cases: tuple[TokenizerConformanceCaseReport, ...]) -> tuple[TokenizerFamilyCoverage, ...]:
    by_family: dict[str, list[TokenizerConformanceCaseReport]] = {family: [] for family in REQUIRED_TOKENIZER_FAMILIES}
    for case in cases:
        by_family.setdefault(case.family, []).append(case)
    return tuple(
        _coverage_for_family(family, tuple(by_family.get(family, ())))
        for family in sorted(by_family, key=_family_sort_key)
    )


def _coverage_for_family(
    family: str,
    cases: tuple[TokenizerConformanceCaseReport, ...],
) -> TokenizerFamilyCoverage:
    return TokenizerFamilyCoverage(
        family=family,
        case_ids=tuple(case.case_id for case in cases),
        backends=tuple(sorted({case.backend for case in cases})),
        features=tuple(sorted({feature for case in cases for feature in case.features})),
        sample_count=sum(case.sample_count for case in cases),
        passed_cases=sum(1 for case in cases if case.passed),
        mismatches=sum(len(case.differential_report.mismatches) for case in cases),
    )


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return REQUIRED_TOKENIZER_FAMILIES.index(family), family
    except ValueError:
        return len(REQUIRED_TOKENIZER_FAMILIES), family


def _sample_text(data: dict[str, Any]) -> str:
    if "text_bytes_hex" in data:
        return bytes.fromhex(_required_str(data, "text_bytes_hex")).decode("utf-8", errors="surrogateescape")
    return _required_str(data, "text")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TokenizerConformanceError(f"{label} must be an object")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise TokenizerConformanceError(f"{key} must be a non-empty string")
    return value


def _str(value: Any, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise TokenizerConformanceError("expected a non-empty string")
    return value


def _str_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TokenizerConformanceError(f"{label} must be an array")
    if not all(isinstance(item, str) for item in value):
        raise TokenizerConformanceError(f"{label} must contain only strings")
    return tuple(value)


def _int_sequence(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise TokenizerConformanceError(f"{label} must be an array")
    if not all(isinstance(item, int) for item in value):
        raise TokenizerConformanceError(f"{label} must contain only integers")
    return tuple(value)


def _int_mapping(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TokenizerConformanceError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int):
            raise TokenizerConformanceError(f"{label} must map strings to integers")
        result[key] = item
    return result
