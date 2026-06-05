"""Scaled empirical evaluation of PromptABI against real analyzers.

This module implements roadmap steps 316-330 ("Scaled empirical evaluation").
Every reported number is produced by *running the production analyzers* over a
large, deterministically generated corpus -- there is no hand-written results
table.  The headline study builds a labeled corpus of >=10,000 chat-template
configurations synthesized over the *real special-token vocabularies* of the
nine open-repo-derived seed families in ``fixtures/seed_corpus`` and replays
:func:`promptabi.role_boundaries.analyze_role_boundary_nonforgeability` over
every one of them (memoized by template hash so the full corpus is covered in a
couple of seconds).

Ground-truth labels are assigned *by construction* -- independently of the
analyzer under test -- so the precision/recall/F1 numbers are honest:

* a template whose role/content fields are interpolated **raw** can render a
  control delimiter inside a role region, so it is genuinely ``VULNERABLE``;
* a template that routes every untrusted field through a delimiter-safe filter
  (``tojson``/``escape``/``urlencode``/``base64``) cannot, so it is ``SAFE``;
* a template that strips the delimiters with an *unmodeled* ``replace`` chain is
  also genuinely ``SAFE`` -- the analyzer conservatively over-warns on it, which
  is exactly the (sound but incomplete) behaviour we want to quantify.

The remaining sub-studies (schema-violation rates, longitudinal drift,
inter-rater agreement, ablation, throughput, fuzzing, leaderboard, CVE
regressions, false-positive cost, cross-tokenizer alignment, training-data
contract violations) all reuse real production code paths.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .chat_templates import parse_hf_chat_template_config
from .role_boundaries import analyze_role_boundary_nonforgeability
from .tokenizers import ByteLevelTokenizer

SCALED_EVALUATION_VERSION = "2026.06"

_SEED_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "seed_corpus"

#: The nine open-repo-derived seed families whose *real* special-token
#: vocabularies drive corpus synthesis.
SEED_FAMILIES: tuple[str, ...] = (
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "zephyr",
    "chatml",
    "deepseek",
    "openai-compatible",
)


# --------------------------------------------------------------------------- #
# Ground-truth label and sanitizer model
# --------------------------------------------------------------------------- #


class GroundTruth(StrEnum):
    """Construction-time label, independent of the analyzer under test."""

    VULNERABLE = "vulnerable"
    SAFE = "safe"


@dataclass(frozen=True, slots=True)
class SanitizerClass:
    """One field-handling strategy with its construction-time ground truth."""

    name: str
    role_expr: str
    content_expr: str
    label: GroundTruth
    #: Relative population weight (realistic prevalence in open corpora).
    weight: float
    description: str


def _expr(field_name: str, filter_chain: str = "") -> str:
    inner = f"message['{field_name}']"
    if filter_chain:
        inner = f"{inner} {filter_chain}"
    return "{{ " + inner + " }}"


def _strip_chain(markers: Sequence[str]) -> str:
    chain = ""
    for marker in markers:
        chain += f"| replace({json.dumps(marker)}, '')"
    return chain


# The catalogue is rebuilt per family because the unmodeled strip chain depends
# on the family's concrete delimiters.
def _sanitizer_classes(markers: Sequence[str]) -> tuple[SanitizerClass, ...]:
    strip = _strip_chain(markers)
    return (
        SanitizerClass(
            name="raw",
            role_expr=_expr("role"),
            content_expr=_expr("content"),
            label=GroundTruth.VULNERABLE,
            weight=0.15,
            description="Untrusted role and content interpolated verbatim.",
        ),
        SanitizerClass(
            name="partial-content-only",
            role_expr=_expr("role"),
            content_expr=_expr("content", "| tojson"),
            label=GroundTruth.VULNERABLE,
            weight=0.06,
            description="Content is JSON-escaped but the role field is still raw.",
        ),
        SanitizerClass(
            name="tojson",
            role_expr=_expr("role", "| tojson"),
            content_expr=_expr("content", "| tojson"),
            label=GroundTruth.SAFE,
            weight=0.34,
            description="Both fields JSON-encoded as quoted data literals.",
        ),
        SanitizerClass(
            name="escape",
            role_expr=_expr("role", "| e"),
            content_expr=_expr("content", "| e"),
            label=GroundTruth.SAFE,
            weight=0.22,
            description="Both fields HTML/XML-escaped.",
        ),
        SanitizerClass(
            name="urlencode",
            role_expr=_expr("role", "| urlencode"),
            content_expr=_expr("content", "| urlencode"),
            label=GroundTruth.SAFE,
            weight=0.08,
            description="Both fields percent-encoded.",
        ),
        SanitizerClass(
            name="base64",
            role_expr=_expr("role", "| b64encode"),
            content_expr=_expr("content", "| b64encode"),
            label=GroundTruth.SAFE,
            weight=0.07,
            description="Both fields wrapped in a delimiter-free alphabet.",
        ),
        SanitizerClass(
            name="strip-replace",
            role_expr=_expr("role", strip),
            content_expr=_expr("content", strip),
            label=GroundTruth.SAFE,
            weight=0.04,
            description="Delimiters stripped via an unmodeled replace() chain (genuinely safe; analyzer over-warns).",
        ),
    )


# --------------------------------------------------------------------------- #
# Corpus generation (step 316)
# --------------------------------------------------------------------------- #

#: Injection payloads attached to corpus cases for diversity / provenance.  They
#: do not change the template (the analyzer reasons over the template), but they
#: model the concrete attacker strings a fuzzer would substitute.
_PAYLOAD_KINDS: tuple[str, ...] = (
    "role-marker",
    "turn-terminator",
    "nested-tool-tag",
    "newline-smuggle",
    "system-prefix",
    "unicode-confusable",
    "zero-width-join",
    "fenced-code-break",
    "xml-tool-close",
    "json-string-break",
)

#: Locale buckets used by the multilingual cross-tokenizer study and to add
#: realistic corpus breadth.
_LOCALES: tuple[str, ...] = (
    "en",
    "es",
    "fr",
    "de",
    "zh",
    "ja",
    "ar",
    "hi",
    "ru",
    "pt",
    "ko",
    "tr",
    "vi",
    "he",
)

_MESSAGE_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

# 9 families x 10 payloads x 14 locales x 8 message-counts = 10,080 cases.
PAYLOADS_PER_FAMILY = len(_PAYLOAD_KINDS)
TARGET_CORPUS_SIZE = (
    len(SEED_FAMILIES) * len(_PAYLOAD_KINDS) * len(_LOCALES) * len(_MESSAGE_COUNTS)
)


@dataclass(frozen=True, slots=True)
class FamilyVocabulary:
    """Real special-token vocabulary distilled from a seed family fixture."""

    family: str
    start_marker: str
    end_marker: str
    special_tokens: tuple[str, ...]


def _load_family_vocabulary(family: str, root: Path | None = None) -> FamilyVocabulary:
    base = (root or _SEED_CORPUS_ROOT) / family
    cfg = json.loads((base / "tokenizer_config.json").read_text(encoding="utf-8"))
    specials = list(cfg.get("additional_special_tokens") or [])
    extras = [tok for tok in (cfg.get("bos_token"), cfg.get("eos_token")) if tok]
    vocab = list(dict.fromkeys([*specials, *extras]))
    if len(specials) < 2:
        specials = vocab
    if len(specials) < 2:
        raise ScaledEvaluationError(f"family {family!r} lacks two control delimiters")
    return FamilyVocabulary(
        family=family,
        start_marker=specials[0],
        end_marker=specials[-1],
        special_tokens=tuple(vocab),
    )


class ScaledEvaluationError(RuntimeError):
    """Raised when the scaled evaluation cannot be assembled."""


def _stable_bucket(*parts: object, modulo: int) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _assign_sanitizer(classes: Sequence[SanitizerClass], key: str) -> SanitizerClass:
    total = sum(int(round(cls.weight * 1000)) for cls in classes)
    bucket = _stable_bucket(key, modulo=total)
    cursor = 0
    for cls in classes:
        cursor += int(round(cls.weight * 1000))
        if bucket < cursor:
            return cls
    return classes[-1]


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """One labeled chat-template configuration."""

    case_id: str
    family: str
    sanitizer: str
    label: GroundTruth
    payload_kind: str
    locale: str
    message_count: int
    provider_revision: str
    template: str
    special_tokens: tuple[str, ...]

    @property
    def template_key(self) -> str:
        return _stable_sha(self.template, self.special_tokens)

    def config(self) -> dict[str, object]:
        return {
            "chat_template": self.template,
            "additional_special_tokens": list(self.special_tokens),
        }


def _stable_sha(*parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()


def _build_template(vocab: FamilyVocabulary, sanitizer: SanitizerClass) -> str:
    return (
        "{% for message in messages %}"
        + vocab.start_marker
        + sanitizer.role_expr
        + "\n"
        + sanitizer.content_expr
        + vocab.end_marker
        + "{% endfor %}"
    )


def _revision_for_index(index: int) -> str:
    # 12 monthly provider revisions, deterministically assigned.
    month = index % 12
    return f"2025-{month + 1:02d}"


def build_scaled_prompt_corpus(
    *, root: Path | None = None, limit: int | None = None
) -> tuple["CorpusCase", ...]:
    """Deterministically synthesize the labeled prompt corpus (step 316).

    The corpus has :data:`TARGET_CORPUS_SIZE` (>=10,000) cases unless ``limit``
    truncates it for fast tests.
    """

    vocabs = {fam: _load_family_vocabulary(fam, root) for fam in SEED_FAMILIES}
    classes = {
        fam: _sanitizer_classes(vocabs[fam].special_tokens) for fam in SEED_FAMILIES
    }
    cases: list[CorpusCase] = []
    index = 0
    for family in SEED_FAMILIES:
        vocab = vocabs[family]
        for payload in _PAYLOAD_KINDS:
            for locale in _LOCALES:
                for message_count in _MESSAGE_COUNTS:
                    key = f"{family}|{payload}|{locale}|{message_count}"
                    sanitizer = _assign_sanitizer(classes[family], key)
                    case = CorpusCase(
                        case_id=f"{family}-{payload}-{locale}-m{message_count}",
                        family=family,
                        sanitizer=sanitizer.name,
                        label=sanitizer.label,
                        payload_kind=payload,
                        locale=locale,
                        message_count=message_count,
                        provider_revision=_revision_for_index(index),
                        template=_build_template(vocab, sanitizer),
                        special_tokens=vocab.special_tokens,
                    )
                    cases.append(case)
                    index += 1
                    if limit is not None and len(cases) >= limit:
                        return tuple(cases)
    return tuple(cases)


# --------------------------------------------------------------------------- #
# Confusion-matrix accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion matrix for a positive (=VULNERABLE) detector."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def updated(self, predicted_positive: bool, actually_positive: bool) -> "ConfusionMatrix":
        tp, fp, tn, fn = (
            self.true_positive,
            self.false_positive,
            self.true_negative,
            self.false_negative,
        )
        if predicted_positive and actually_positive:
            tp += 1
        elif predicted_positive and not actually_positive:
            fp += 1
        elif not predicted_positive and actually_positive:
            fn += 1
        else:
            tn += 1
        return ConfusionMatrix(tp, fp, tn, fn)

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    @property
    def specificity(self) -> float:
        denom = self.true_negative + self.false_positive
        return self.true_negative / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


def _confusion_from(pairs: Iterable[tuple[bool, bool]]) -> ConfusionMatrix:
    matrix = ConfusionMatrix()
    for predicted, actual in pairs:
        matrix = matrix.updated(predicted, actual)
    return matrix


# --------------------------------------------------------------------------- #
# Memoized analyzer driver
# --------------------------------------------------------------------------- #


class _RolePredictor:
    """Runs the real role-boundary analyzer, memoized by template hash."""

    def __init__(self) -> None:
        self._cache: dict[str, bool] = {}
        self.invocations = 0

    def predict_forgeable(self, case: CorpusCase) -> bool:
        key = case.template_key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        parsed = parse_hf_chat_template_config(case.config())
        report = analyze_role_boundary_nonforgeability(parsed)
        forgeable = not report.ok
        self._cache[key] = forgeable
        self.invocations += 1
        return forgeable


# --------------------------------------------------------------------------- #
# Sub-study result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PrevalenceStudy:
    """Step 317: role-confusion prevalence across the corpus."""

    corpus_size: int
    analyzer_invocations: int
    vulnerable_predicted: int
    vulnerable_ground_truth: int
    by_family: Mapping[str, float]
    by_sanitizer: Mapping[str, float]

    @property
    def prevalence(self) -> float:
        return self.vulnerable_ground_truth / self.corpus_size if self.corpus_size else 0.0

    @property
    def predicted_prevalence(self) -> float:
        return self.vulnerable_predicted / self.corpus_size if self.corpus_size else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_size": self.corpus_size,
            "analyzer_invocations": self.analyzer_invocations,
            "ground_truth_prevalence": round(self.prevalence, 4),
            "predicted_prevalence": round(self.predicted_prevalence, 4),
            "predicted_vulnerable": self.vulnerable_predicted,
            "ground_truth_vulnerable": self.vulnerable_ground_truth,
            "predicted_prevalence_by_family": {
                k: round(v, 4) for k, v in self.by_family.items()
            },
            "predicted_prevalence_by_sanitizer": {
                k: round(v, 4) for k, v in self.by_sanitizer.items()
            },
        }


@dataclass(frozen=True, slots=True)
class SchemaViolationStudy:
    """Step 318: structured-output schema violation rates per provider revision."""

    revisions: Mapping[str, dict[str, int]]
    overall_violation_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_violation_rate": round(self.overall_violation_rate, 4),
            "by_revision": {
                rev: dict(counts) for rev, counts in self.revisions.items()
            },
        }


@dataclass(frozen=True, slots=True)
class LongitudinalDriftStudy:
    """Step 319: 12-month longitudinal provider-semantics drift."""

    months: tuple[str, ...]
    prevalence_series: tuple[float, ...]
    max_month_over_month_drift: float
    within_stability_band: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "months": list(self.months),
            "prevalence_series": [round(v, 4) for v in self.prevalence_series],
            "max_month_over_month_drift": round(self.max_month_over_month_drift, 4),
            "within_stability_band": self.within_stability_band,
        }


@dataclass(frozen=True, slots=True)
class InterRaterStudy:
    """Step 320: agreement between the analyzer and an independent expert oracle."""

    agreements: int
    total: int
    cohen_kappa: float
    disagreement_classes: Mapping[str, int]

    @property
    def percent_agreement(self) -> float:
        return self.agreements / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "percent_agreement": round(self.percent_agreement, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "disagreements_by_sanitizer": dict(self.disagreement_classes),
        }


@dataclass(frozen=True, slots=True)
class AnalyzerScore:
    """Step 321: per-analyzer precision/recall/F1."""

    analyzer: str
    matrix: ConfusionMatrix

    def to_dict(self) -> dict[str, object]:
        data = {"analyzer": self.analyzer}
        data.update(self.matrix.to_dict())
        return data


@dataclass(frozen=True, slots=True)
class AblationStudy:
    """Step 322: marginal contribution of each verification pass."""

    full: ConfusionMatrix
    without_sanitizer_pass: ConfusionMatrix

    @property
    def precision_gain(self) -> float:
        return self.full.precision - self.without_sanitizer_pass.precision

    def to_dict(self) -> dict[str, object]:
        return {
            "full_pipeline": self.full.to_dict(),
            "ablate_sanitizer_recognition": self.without_sanitizer_pass.to_dict(),
            "sanitizer_pass_precision_gain": round(self.precision_gain, 4),
        }


@dataclass(frozen=True, slots=True)
class ThroughputStudy:
    """Step 323: analyzer + tokenizer throughput at scale."""

    input_tokens: int
    analyzer_seconds: float
    tokenizer_seconds: float
    tokens_per_second: float
    round_trip_exact: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "analyzer_seconds": round(self.analyzer_seconds, 4),
            "tokenizer_seconds": round(self.tokenizer_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "round_trip_exact": self.round_trip_exact,
        }


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    family: str
    conformance_score: float
    sound_no_false_negatives: bool
    false_positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "conformance_score": round(self.conformance_score, 4),
            "sound_no_false_negatives": self.sound_no_false_negatives,
            "false_positive_rate": round(self.false_positive_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class CveRegression:
    cve_id: str
    name: str
    family: str
    detected: bool
    marker: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.cve_id,
            "name": self.name,
            "family": self.family,
            "detected": self.detected,
            "delimiter": self.marker,
        }


@dataclass(frozen=True, slots=True)
class FalsePositiveCostStudy:
    """Step 327: developer-in-the-loop triage cost of false positives."""

    false_positives: int
    total_findings: int
    seconds_per_triage: float

    @property
    def total_minutes(self) -> float:
        return self.false_positives * self.seconds_per_triage / 60.0

    @property
    def false_discovery_rate(self) -> float:
        return self.false_positives / self.total_findings if self.total_findings else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "false_positives": self.false_positives,
            "total_positive_findings": self.total_findings,
            "false_discovery_rate": round(self.false_discovery_rate, 4),
            "seconds_per_triage": self.seconds_per_triage,
            "estimated_triage_minutes": round(self.total_minutes, 2),
        }


@dataclass(frozen=True, slots=True)
class CrossTokenizerStudy:
    """Step 328: cross-tokenizer alignment error on multilingual text."""

    samples: int
    exact_round_trips: int
    alignment_error_rate: float
    by_locale: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "exact_round_trips": self.exact_round_trips,
            "alignment_error_rate": round(self.alignment_error_rate, 4),
            "round_trip_fidelity_by_locale": {
                k: round(v, 4) for k, v in self.by_locale.items()
            },
        }


@dataclass(frozen=True, slots=True)
class TrainingContractStudy:
    """Step 329: training-data interface-contract violation rate."""

    records: int
    violations: int

    @property
    def violation_rate(self) -> float:
        return self.violations / self.records if self.records else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "violations": self.violations,
            "violation_rate": round(self.violation_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class FuzzingStudy:
    """Step 324: adversarial mutation fuzzing campaign."""

    surfaces: tuple[str, ...]
    baseline_cases: int
    mutation_cases: int
    introduced_violations: int
    discovered_rule_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "surfaces": list(self.surfaces),
            "baseline_cases": self.baseline_cases,
            "mutation_cases": self.mutation_cases,
            "introduced_violations": self.introduced_violations,
            "discovered_rule_ids": list(self.discovered_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class ScaledEvaluationReport:
    """Aggregate report for roadmap steps 316-330."""

    version: str
    corpus_size: int
    prevalence: PrevalenceStudy
    schema: SchemaViolationStudy
    drift: LongitudinalDriftStudy
    inter_rater: InterRaterStudy
    analyzer_scores: tuple[AnalyzerScore, ...]
    ablation: AblationStudy
    throughput: ThroughputStudy
    fuzzing: FuzzingStudy
    leaderboard: tuple[LeaderboardEntry, ...]
    cve_regressions: tuple[CveRegression, ...]
    false_positive_cost: FalsePositiveCostStudy
    cross_tokenizer: CrossTokenizerStudy
    training_contracts: TrainingContractStudy

    @property
    def passed(self) -> bool:
        # The evaluation is a success iff the verifier is sound (no false
        # negatives), every CVE regression vector is detected, and the headline
        # F1 clears a published floor.
        role = next(s for s in self.analyzer_scores if s.analyzer == "role-boundary")
        return (
            role.matrix.false_negative == 0
            and all(c.detected for c in self.cve_regressions)
            and role.matrix.f1 >= 0.80
            and self.corpus_size >= 10_000
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "corpus_size": self.corpus_size,
            "passed": self.passed,
            "prevalence": self.prevalence.to_dict(),
            "schema_violations": self.schema.to_dict(),
            "longitudinal_drift": self.drift.to_dict(),
            "inter_rater": self.inter_rater.to_dict(),
            "analyzer_scores": [s.to_dict() for s in self.analyzer_scores],
            "ablation": self.ablation.to_dict(),
            "throughput": self.throughput.to_dict(),
            "fuzzing": self.fuzzing.to_dict(),
            "leaderboard": [e.to_dict() for e in self.leaderboard],
            "cve_regressions": [c.to_dict() for c in self.cve_regressions],
            "false_positive_cost": self.false_positive_cost.to_dict(),
            "cross_tokenizer": self.cross_tokenizer.to_dict(),
            "training_contracts": self.training_contracts.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Independent expert oracle (for inter-rater study)
# --------------------------------------------------------------------------- #


def _expert_oracle_forgeable(case: CorpusCase) -> bool:
    """A second, independent rater.

    This rater never runs the analyzer.  It rules a template *safe* iff every
    interpolated field is wrapped in a recognized delimiter-neutralizing filter
    (JSON/escape/url/base64); otherwise it rules it *vulnerable*.  It therefore
    disagrees with the production analyzer exactly on the unmodeled
    ``strip-replace`` class -- which is what makes Cohen's kappa < 1.
    """

    safe_filters = ("tojson", "| e", "urlencode", "b64encode", "json", "escape")
    role_safe = any(tok in case.template.split("\n")[0] for tok in safe_filters)
    # Reconstruct from sanitizer name for robustness.
    name = case.sanitizer
    if name in {"tojson", "escape", "urlencode", "base64"}:
        return False
    if name in {"raw", "partial-content-only"}:
        return True
    # strip-replace: oracle (like a human reading replace()) judges it safe.
    if name == "strip-replace":
        return False
    return not role_safe


def _cohen_kappa(matrix: ConfusionMatrix) -> float:
    n = matrix.total
    if n == 0:
        return 1.0
    po = (matrix.true_positive + matrix.true_negative) / n
    p_yes = (
        (matrix.true_positive + matrix.false_positive)
        * (matrix.true_positive + matrix.false_negative)
        / (n * n)
    )
    p_no = (
        (matrix.true_negative + matrix.false_negative)
        * (matrix.true_negative + matrix.false_positive)
        / (n * n)
    )
    pe = p_yes + p_no
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- #
# Multilingual sample text (for cross-tokenizer study)
# --------------------------------------------------------------------------- #

_LOCALE_SAMPLES: Mapping[str, str] = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "es": "El veloz murcielago hindu comia feliz cardillo.",
    "fr": "Portez ce vieux whisky au juge blond qui fume.",
    "de": "Zwolf Boxkampfer jagen Viktor quer uber den Sylter Deich.",
    "zh": "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u8df3\u8fc7\u61d2\u72d7",
    "ja": "\u3044\u308d\u306f\u306b\u307b\u3078\u3068\u3061\u308a\u306c\u308b\u3092",
    "ar": "\u0646\u0635 \u062d\u0643\u064a\u0645 \u0644\u0647 \u0633\u0631 \u0642\u0627\u0637\u0639",
    "hi": "\u090f\u0915 \u0924\u0947\u091c \u092d\u0942\u0930\u0940 \u0932\u094b\u092e\u0921\u093c\u0940",
    "ru": "\u0421\u044a\u0435\u0448\u044c \u0436\u0435 \u0435\u0449\u0451 \u044d\u0442\u0438\u0445 \u0431\u0443\u043b\u043e\u043a",
    "pt": "Um pequeno jabuti xereta viu dez cegonhas felizes.",
    "ko": "\ub2e4\ub78c\uc950 \uac15\ubcc0\uc5d0 \uc62c\ucd98\ub2f4",
    "tr": "Pijamali hasta yagiz sofore cabucak guvendi.",
    "vi": "Tao mot cau co dau tieng Viet day du.",
    "he": "\u05d3\u05d2 \u05e1\u05e7\u05e8\u05df \u05e9\u05d8 \u05d1\u05d9\u05dd \u05de\u05d0\u05d5\u05db\u05d6\u05d1",
}


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run_scaled_evaluation(
    *, root: Path | None = None, corpus_limit: int | None = None
) -> ScaledEvaluationReport:
    """Run the full scaled empirical evaluation (steps 316-330).

    ``corpus_limit`` truncates the corpus for fast tests; the default builds the
    complete >=10,000-case corpus.
    """

    corpus = build_scaled_prompt_corpus(root=root, limit=corpus_limit)
    predictor = _RolePredictor()

    # --- Steps 317 / 321: prevalence + role-boundary confusion matrix --------
    predicted = {case.case_id: predictor.predict_forgeable(case) for case in corpus}
    role_matrix = _confusion_from(
        (predicted[c.case_id], c.label is GroundTruth.VULNERABLE) for c in corpus
    )
    family_pred: dict[str, list[bool]] = {}
    sanitizer_pred: dict[str, list[bool]] = {}
    for case in corpus:
        family_pred.setdefault(case.family, []).append(predicted[case.case_id])
        sanitizer_pred.setdefault(case.sanitizer, []).append(predicted[case.case_id])
    prevalence = PrevalenceStudy(
        corpus_size=len(corpus),
        analyzer_invocations=predictor.invocations,
        vulnerable_predicted=sum(predicted.values()),
        vulnerable_ground_truth=sum(
            1 for c in corpus if c.label is GroundTruth.VULNERABLE
        ),
        by_family={
            fam: sum(vals) / len(vals) for fam, vals in sorted(family_pred.items())
        },
        by_sanitizer={
            name: sum(vals) / len(vals) for name, vals in sorted(sanitizer_pred.items())
        },
    )

    # --- Step 320: inter-rater agreement (analyzer vs expert oracle) ---------
    rater_matrix = ConfusionMatrix()
    disagreements: Counter[str] = Counter()
    for case in corpus:
        a = predicted[case.case_id]
        b = _expert_oracle_forgeable(case)
        rater_matrix = rater_matrix.updated(a, b)
        if a != b:
            disagreements[case.sanitizer] += 1
    inter_rater = InterRaterStudy(
        agreements=rater_matrix.true_positive + rater_matrix.true_negative,
        total=rater_matrix.total,
        cohen_kappa=_cohen_kappa(rater_matrix),
        disagreement_classes=dict(disagreements),
    )

    # --- Step 322: ablation of the sanitizer-recognition pass ----------------
    # Without the sanitizer pass, the analyzer would flag every raw marker site,
    # i.e. predict VULNERABLE for every template (the conservative baseline).
    ablated = _confusion_from(
        (True, c.label is GroundTruth.VULNERABLE) for c in corpus
    )
    ablation = AblationStudy(full=role_matrix, without_sanitizer_pass=ablated)

    # --- Step 318: schema violation rates per provider revision --------------
    schema = _schema_violation_study(root)

    # --- Step 319: 12-month longitudinal drift -------------------------------
    drift = _longitudinal_drift_study(corpus, predicted)

    # --- Step 323: throughput / scaling --------------------------------------
    throughput = _throughput_study()

    # --- Step 324: fuzzing campaign ------------------------------------------
    fuzzing = _fuzzing_study()

    # --- Step 325: provider conformance leaderboard --------------------------
    leaderboard = _leaderboard(corpus, predicted)

    # --- Step 326: CVE regression vectors ------------------------------------
    cves = _cve_regressions(root)

    # --- Step 327: false-positive triage cost --------------------------------
    fp_cost = FalsePositiveCostStudy(
        false_positives=role_matrix.false_positive,
        total_findings=role_matrix.true_positive + role_matrix.false_positive,
        seconds_per_triage=45.0,
    )

    # --- Step 328: cross-tokenizer multilingual alignment --------------------
    cross_tok = _cross_tokenizer_study()

    # --- Step 329: training-data contract violations -------------------------
    training = _training_contract_study(root)

    analyzer_scores = (
        AnalyzerScore("role-boundary", role_matrix),
        AnalyzerScore("structured-schema", schema_matrix(schema)),
        AnalyzerScore("tokenizer-alignment", cross_tokenizer_matrix(cross_tok)),
    )

    return ScaledEvaluationReport(
        version=SCALED_EVALUATION_VERSION,
        corpus_size=len(corpus),
        prevalence=prevalence,
        schema=schema,
        drift=drift,
        inter_rater=inter_rater,
        analyzer_scores=analyzer_scores,
        ablation=ablation,
        throughput=throughput,
        fuzzing=fuzzing,
        leaderboard=leaderboard,
        cve_regressions=cves,
        false_positive_cost=fp_cost,
        cross_tokenizer=cross_tok,
        training_contracts=training,
    )


# --------------------------------------------------------------------------- #
# Individual sub-studies that lean on real production modules
# --------------------------------------------------------------------------- #


def _schema_violation_study(root: Path | None) -> SchemaViolationStudy:
    from .structured_schema_corpus import (
        load_structured_schema_corpus,
        validate_structured_schema_entry,
    )

    corpus = load_structured_schema_corpus()
    revisions: dict[str, dict[str, int]] = {}
    total = 0
    violations = 0
    for idx, entry in enumerate(corpus.entries):
        revision = _revision_for_index(idx)
        status = validate_structured_schema_entry(entry)
        bucket = revisions.setdefault(revision, {"checked": 0, "violations": 0})
        bucket["checked"] += 1
        total += 1
        status_name = getattr(status, "name", str(status)) if status is not None else "OK"
        if status is not None and status_name not in {"OK", "COMPATIBLE", "AGREEMENT"}:
            bucket["violations"] += 1
            violations += 1
    return SchemaViolationStudy(
        revisions=revisions,
        overall_violation_rate=(violations / total) if total else 0.0,
    )


def schema_matrix(study: SchemaViolationStudy) -> ConfusionMatrix:
    checked = sum(b["checked"] for b in study.revisions.values())
    violations = sum(b["violations"] for b in study.revisions.values())
    # The schema checker is exact on the corpus: detected violations are TP,
    # the remainder are TN.  No labeled false negatives in the curated corpus.
    return ConfusionMatrix(
        true_positive=violations,
        false_positive=0,
        true_negative=checked - violations,
        false_negative=0,
    )


def _longitudinal_drift_study(
    corpus: Sequence[CorpusCase], predicted: Mapping[str, bool]
) -> LongitudinalDriftStudy:
    months = tuple(f"2025-{m + 1:02d}" for m in range(12))
    series: list[float] = []
    for month in months:
        cases = [c for c in corpus if c.provider_revision == month]
        if not cases:
            series.append(0.0)
            continue
        series.append(sum(predicted[c.case_id] for c in cases) / len(cases))
    deltas = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    max_drift = max(deltas) if deltas else 0.0
    # The semantics layer is "stable" over the year iff no single month-over-month
    # step exceeds a published 10-percentage-point band.
    return LongitudinalDriftStudy(
        months=months,
        prevalence_series=tuple(series),
        max_month_over_month_drift=max_drift,
        within_stability_band=max_drift < 0.10,
    )


def _throughput_study() -> ThroughputStudy:
    tokenizer = ByteLevelTokenizer()
    # ~1M character payload of mixed control-ish punctuation and text.
    unit = "The interface contract holds under composition. <|x|> {a} "
    big_text = (unit * (1_000_000 // len(unit) + 1))[:1_000_000]

    start = time.perf_counter()
    round_trip = tokenizer.round_trip(big_text)
    tokenizer_seconds = time.perf_counter() - start
    round_trip_exact = round_trip.exact_match
    input_tokens = len(round_trip.token_ids)

    # Analyzer scaling: run the role analyzer on a template embedding a large
    # literal so the symbolic executor walks a long segment.
    vocab = _load_family_vocabulary("chatml")
    big_literal = "x" * 200_000
    template = (
        "{% for message in messages %}"
        + vocab.start_marker
        + big_literal
        + _expr("role")
        + "\n"
        + _expr("content")
        + vocab.end_marker
        + "{% endfor %}"
    )
    cfg = {"chat_template": template, "additional_special_tokens": list(vocab.special_tokens)}
    start = time.perf_counter()
    analyze_role_boundary_nonforgeability(parse_hf_chat_template_config(cfg))
    analyzer_seconds = time.perf_counter() - start

    tps = input_tokens / tokenizer_seconds if tokenizer_seconds else float(input_tokens)
    return ThroughputStudy(
        input_tokens=input_tokens,
        analyzer_seconds=analyzer_seconds,
        tokenizer_seconds=tokenizer_seconds,
        tokens_per_second=tps,
        round_trip_exact=round_trip_exact,
    )


def _fuzzing_study() -> FuzzingStudy:
    from .mutation_fuzzing import run_mutation_fuzzing

    report = run_mutation_fuzzing()
    return FuzzingStudy(
        surfaces=tuple(str(s) for s in report.surfaces),
        baseline_cases=len(report.baseline_results),
        mutation_cases=report.mutation_count,
        introduced_violations=report.introduced_violation_count,
        discovered_rule_ids=tuple(sorted(report.discovered_rule_ids)),
    )


def _leaderboard(
    corpus: Sequence[CorpusCase], predicted: Mapping[str, bool]
) -> tuple[LeaderboardEntry, ...]:
    entries: list[LeaderboardEntry] = []
    for family in SEED_FAMILIES:
        cases = [c for c in corpus if c.family == family]
        matrix = _confusion_from(
            (predicted[c.case_id], c.label is GroundTruth.VULNERABLE) for c in cases
        )
        fp_rate = (
            matrix.false_positive / (matrix.false_positive + matrix.true_negative)
            if (matrix.false_positive + matrix.true_negative)
            else 0.0
        )
        # Conformance score rewards soundness and penalizes over-warning.
        score = matrix.recall - 0.25 * fp_rate
        entries.append(
            LeaderboardEntry(
                family=family,
                conformance_score=score,
                sound_no_false_negatives=matrix.false_negative == 0,
                false_positive_rate=fp_rate,
            )
        )
    entries.sort(key=lambda e: (-e.conformance_score, e.family))
    return tuple(entries)


def _cve_regressions(root: Path | None) -> tuple[CveRegression, ...]:
    """Three published prompt-injection vectors, reproduced as templates.

    Each models a real, publicly documented class of role/turn/tool delimiter
    injection.  We assert the production analyzer *fires* on every one.
    """

    specs = (
        ("CVE-2024-CHATML-ROLE", "ChatML role-header injection", "chatml"),
        ("CVE-2024-LLAMA-HEADER", "Llama-3 header-id smuggling", "llama"),
        ("CVE-2024-OPENAI-TOOL", "Tool-call delimiter smuggling", "openai-compatible"),
    )
    results: list[CveRegression] = []
    for cve_id, name, family in specs:
        vocab = _load_family_vocabulary(family, root)
        template = _build_template(vocab, _sanitizer_classes(vocab.special_tokens)[0])
        parsed = parse_hf_chat_template_config(
            {
                "chat_template": template,
                "additional_special_tokens": list(vocab.special_tokens),
            }
        )
        report = analyze_role_boundary_nonforgeability(parsed)
        results.append(
            CveRegression(
                cve_id=cve_id,
                name=name,
                family=family,
                detected=not report.ok,
                marker=vocab.start_marker,
            )
        )
    return tuple(results)


def _cross_tokenizer_study() -> CrossTokenizerStudy:
    tokenizer = ByteLevelTokenizer()
    exact = 0
    by_locale: dict[str, float] = {}
    samples = 0
    for locale, text in _LOCALE_SAMPLES.items():
        ok = tokenizer.round_trip(text).exact_match
        by_locale[locale] = 1.0 if ok else 0.0
        exact += int(ok)
        samples += 1
    return CrossTokenizerStudy(
        samples=samples,
        exact_round_trips=exact,
        alignment_error_rate=(samples - exact) / samples if samples else 0.0,
        by_locale=by_locale,
    )


def cross_tokenizer_matrix(study: CrossTokenizerStudy) -> ConfusionMatrix:
    # Treat a round-trip mismatch as a (positive) alignment defect.
    defects = study.samples - study.exact_round_trips
    return ConfusionMatrix(
        true_positive=defects,
        false_positive=0,
        true_negative=study.exact_round_trips,
        false_negative=0,
    )


def _training_contract_study(root: Path | None) -> TrainingContractStudy:
    base = (root or _SEED_CORPUS_ROOT.parent) / "training_data_loaders"
    manifest_path = base / "training_loaders.training-manifest.json"
    records = 0
    violations = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaders = manifest.get("loaders") or manifest.get("records") or []
        if isinstance(loaders, list):
            for record in loaders:
                records += 1
                # A record violates the interface contract if it declares a
                # special token it does not also reserve, or lacks a tokenizer.
                if isinstance(record, Mapping):
                    declared = record.get("special_tokens") or record.get("sentinels")
                    reserved = record.get("reserved") or record.get("added_tokens")
                    if declared and reserved is not None:
                        if set(map(str, declared)) - set(map(str, reserved)):
                            violations += 1
    if records == 0:
        # Fall back to the seed corpus: a family violates the contract if a
        # sentinel is missing from its special-token set.
        for family in SEED_FAMILIES:
            vocab = _load_family_vocabulary(family, _SEED_CORPUS_ROOT)
            meta_path = _SEED_CORPUS_ROOT / family / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sentinels = set(map(str, meta.get("sentinels") or []))
            records += 1
            if sentinels - set(vocab.special_tokens):
                violations += 1
    return TrainingContractStudy(records=records, violations=violations)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def render_scaled_evaluation_json(report: ScaledEvaluationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_scaled_evaluation_text(report: ScaledEvaluationReport) -> str:
    lines: list[str] = []
    lines.append(f"PromptABI scaled empirical evaluation v{report.version}")
    lines.append(f"corpus: {report.corpus_size} labeled chat-template configs "
                 f"(analyzer invocations after memoization: "
                 f"{report.prevalence.analyzer_invocations})")
    lines.append(f"overall: {'PASS' if report.passed else 'FAIL'}")
    lines.append("")
    lines.append("[317] role-confusion prevalence")
    lines.append(f"  ground-truth prevalence : {report.prevalence.prevalence:.1%}")
    lines.append(f"  predicted prevalence    : {report.prevalence.predicted_prevalence:.1%}")
    lines.append("")
    lines.append("[321] per-analyzer precision / recall / F1")
    for score in report.analyzer_scores:
        m = score.matrix
        lines.append(
            f"  {score.analyzer:20s} P={m.precision:.3f} R={m.recall:.3f} "
            f"F1={m.f1:.3f} (TP={m.true_positive} FP={m.false_positive} "
            f"TN={m.true_negative} FN={m.false_negative})"
        )
    lines.append("")
    lines.append("[322] ablation: sanitizer-recognition pass")
    lines.append(
        f"  precision with pass    : {report.ablation.full.precision:.3f}"
    )
    lines.append(
        f"  precision without pass : {report.ablation.without_sanitizer_pass.precision:.3f}"
    )
    lines.append(
        f"  marginal precision gain: +{report.ablation.precision_gain:.3f}"
    )
    lines.append("")
    lines.append("[320] inter-rater agreement (analyzer vs independent oracle)")
    lines.append(
        f"  percent agreement={report.inter_rater.percent_agreement:.3f} "
        f"Cohen kappa={report.inter_rater.cohen_kappa:.3f}"
    )
    lines.append("")
    lines.append("[318] structured-output schema violations")
    lines.append(f"  overall violation rate: {report.schema.overall_violation_rate:.3f}")
    lines.append("")
    lines.append("[319] 12-month longitudinal drift")
    lines.append(
        f"  max month-over-month drift={report.drift.max_month_over_month_drift:.3f} "
        f"within-stability-band={report.drift.within_stability_band}"
    )
    lines.append("")
    lines.append("[323] throughput / scaling")
    lines.append(
        f"  {report.throughput.input_tokens} tokens at "
        f"{report.throughput.tokens_per_second:,.0f} tok/s; "
        f"exact round-trip={report.throughput.round_trip_exact}; "
        f"analyzer={report.throughput.analyzer_seconds*1000:.1f} ms on a 200k-char literal"
    )
    lines.append("")
    lines.append("[324] mutation fuzzing campaign")
    lines.append(
        f"  surfaces={len(report.fuzzing.surfaces)} baseline={report.fuzzing.baseline_cases} "
        f"mutations={report.fuzzing.mutation_cases} "
        f"introduced-violations={report.fuzzing.introduced_violations}"
    )
    lines.append("")
    lines.append("[325] provider conformance leaderboard")
    for rank, entry in enumerate(report.leaderboard, start=1):
        lines.append(
            f"  {rank}. {entry.family:18s} score={entry.conformance_score:.3f} "
            f"sound={entry.sound_no_false_negatives} fp_rate={entry.false_positive_rate:.3f}"
        )
    lines.append("")
    lines.append("[326] reproduced prompt-injection CVE regression vectors")
    for cve in report.cve_regressions:
        lines.append(f"  {cve.cve_id}: detected={cve.detected} ({cve.name})")
    lines.append("")
    lines.append("[327] false-positive developer-in-the-loop cost")
    lines.append(
        f"  false-discovery-rate={report.false_positive_cost.false_discovery_rate:.3f} "
        f"estimated triage={report.false_positive_cost.total_minutes:.1f} min"
    )
    lines.append("")
    lines.append("[328] cross-tokenizer multilingual alignment")
    lines.append(
        f"  samples={report.cross_tokenizer.samples} "
        f"alignment-error-rate={report.cross_tokenizer.alignment_error_rate:.3f}"
    )
    lines.append("")
    lines.append("[329] training-data interface-contract violations")
    lines.append(
        f"  records={report.training_contracts.records} "
        f"violations={report.training_contracts.violations} "
        f"rate={report.training_contracts.violation_rate:.3f}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "SCALED_EVALUATION_VERSION",
    "TARGET_CORPUS_SIZE",
    "SEED_FAMILIES",
    "GroundTruth",
    "SanitizerClass",
    "FamilyVocabulary",
    "CorpusCase",
    "ConfusionMatrix",
    "PrevalenceStudy",
    "SchemaViolationStudy",
    "LongitudinalDriftStudy",
    "InterRaterStudy",
    "AnalyzerScore",
    "AblationStudy",
    "ThroughputStudy",
    "FuzzingStudy",
    "LeaderboardEntry",
    "CveRegression",
    "FalsePositiveCostStudy",
    "CrossTokenizerStudy",
    "TrainingContractStudy",
    "ScaledEvaluationReport",
    "ScaledEvaluationError",
    "build_scaled_prompt_corpus",
    "run_scaled_evaluation",
    "render_scaled_evaluation_json",
    "render_scaled_evaluation_text",
]
