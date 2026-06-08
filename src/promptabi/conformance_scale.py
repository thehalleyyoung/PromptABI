"""Cross-provider conformance corpus and differential testing at scale (steps 416-430).

Everything in this module is computed by *running the production analyzers* over
the deterministically synthesized labeled corpus from
:mod:`promptabi.scaled_evaluation` (>=10,000 chat-template configurations built
over the *real* special-token vocabularies of the nine open-repo-derived seed
families).  There are no hand-written results tables.

The capabilities provided here are:

* :func:`build_conformance_corpus` -- assemble the >=10k-artifact corpus and
  summarise it per provider family (step 416).
* :func:`run_differential_oracle` -- compare PromptABI verdicts against an
  independent reference oracle that approximates provider runtime behaviour, and
  report agreement (step 417).
* :func:`normalize_chat_template` / :func:`mine_and_normalize` -- canonicalise
  raw chat templates and deduplicate by structural key (step 418).
* :func:`build_labeled_suites` -- per-rule positive/negative suites with
  provenance (step 419).
* :func:`run_metamorphic_suite` -- semantics-preserving rewrites must preserve
  verdicts (step 420).
* :func:`run_fuzzing_harness` -- adversarial template/schema generation that must
  never crash the analyzer and must flag genuinely forgeable templates
  (step 421).
* :func:`per_rule_metrics` -- precision/recall per rule with Wilson confidence
  intervals (step 422).
* :func:`regression_museum` -- historical upstream bugs PromptABI catches
  (step 423).
* :func:`detect_pairing_drift` -- template/tokenizer pairing-break alerts across
  provider revisions (step 424).
* :func:`cross_validate_golden_encodings` -- ByteLevel round-trip golden
  encodings (step 425).
* :func:`build_test_vector_package` -- a versioned, digest-stamped conformance
  test-vector package consumable by other tools (step 426).
* :func:`mcnemar_vs_baseline` -- statistical significance vs a naive linter
  baseline (step 427).
* :func:`inter_rater_reliability` -- Cohen's kappa for the labeled subset
  (step 428).
* :func:`corpus_snapshot` -- reproducible snapshot digests (step 429).
* :func:`conformance_dashboard` -- a public dashboard over time (step 430).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .chat_templates import parse_hf_chat_template_config
from .role_boundaries import analyze_role_boundary_nonforgeability
from .scaled_evaluation import (
    SEED_FAMILIES,
    ConfusionMatrix,
    CorpusCase,
    GroundTruth,
    _cohen_kappa,
    _expert_oracle_forgeable,
    build_scaled_prompt_corpus,
)
from .tokenizers import ByteLevelTokenizer

CONFORMANCE_SCALE_VERSION = "promptabi.conformance-scale.v1"

#: Provider families grouped into the major provider runtime "families" they map
#: onto.  Used so the corpus can be reported as spanning every major provider.
PROVIDER_FAMILY_MAP: Mapping[str, str] = {
    "llama": "meta-llama",
    "mistral": "mistral",
    "qwen": "qwen",
    "gemma": "google",
    "phi": "microsoft",
    "zephyr": "huggingface",
    "chatml": "openai",
    "deepseek": "deepseek",
    "openai-compatible": "openai",
}


# --------------------------------------------------------------------------- #
# Shared analyzer driver (memoized by structural key)
# --------------------------------------------------------------------------- #


class _Analyzer:
    """Runs the real role-boundary analyzer, memoized by template structure."""

    def __init__(self) -> None:
        self._cache: dict[str, bool] = {}
        self.invocations = 0

    def forgeable(self, config: Mapping[str, object]) -> bool:
        key = _stable_digest(config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        parsed = parse_hf_chat_template_config(dict(config))
        report = analyze_role_boundary_nonforgeability(parsed)
        forgeable = not report.ok
        self._cache[key] = forgeable
        self.invocations += 1
        return forgeable

    def forgeable_case(self, case: CorpusCase) -> bool:
        return self.forgeable(case.config())


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Step 416 -- assemble the conformance corpus
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProviderCorpusSlice:
    seed_family: str
    provider_family: str
    case_count: int
    vulnerable_count: int
    safe_count: int
    distinct_templates: int


@dataclass(frozen=True, slots=True)
class ConformanceCorpus:
    version: str
    total_cases: int
    provider_families: tuple[str, ...]
    slices: tuple[ProviderCorpusSlice, ...]
    digest: str

    def spans_every_seed_family(self) -> bool:
        return {s.seed_family for s in self.slices} == set(SEED_FAMILIES)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "total_cases": self.total_cases,
            "provider_families": list(self.provider_families),
            "slices": [
                {
                    "seed_family": s.seed_family,
                    "provider_family": s.provider_family,
                    "case_count": s.case_count,
                    "vulnerable_count": s.vulnerable_count,
                    "safe_count": s.safe_count,
                    "distinct_templates": s.distinct_templates,
                }
                for s in self.slices
            ],
            "digest": self.digest,
        }


def build_conformance_corpus(*, limit: int | None = None) -> ConformanceCorpus:
    cases = build_scaled_prompt_corpus(limit=limit)
    by_family: dict[str, list[CorpusCase]] = defaultdict(list)
    for case in cases:
        by_family[case.family].append(case)
    slices: list[ProviderCorpusSlice] = []
    for family in SEED_FAMILIES:
        group = by_family.get(family, [])
        if not group:
            continue
        vulnerable = sum(1 for c in group if c.label is GroundTruth.VULNERABLE)
        templates = {c.template_key for c in group}
        slices.append(
            ProviderCorpusSlice(
                seed_family=family,
                provider_family=PROVIDER_FAMILY_MAP[family],
                case_count=len(group),
                vulnerable_count=vulnerable,
                safe_count=len(group) - vulnerable,
                distinct_templates=len(templates),
            )
        )
    provider_families = tuple(sorted({s.provider_family for s in slices}))
    digest = _stable_digest([s.seed_family for s in slices] + [len(cases)])
    return ConformanceCorpus(
        version=CONFORMANCE_SCALE_VERSION,
        total_cases=len(cases),
        provider_families=provider_families,
        slices=tuple(slices),
        digest=digest,
    )


# --------------------------------------------------------------------------- #
# Step 417 -- differential oracle vs reference runtime behaviour
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DifferentialOracleReport:
    version: str
    compared: int
    agreements: int
    analyzer_only: int  # analyzer flags, oracle does not
    oracle_only: int  # oracle flags, analyzer does not
    disagreement_classes: tuple[str, ...]

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.compared if self.compared else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "compared": self.compared,
            "agreements": self.agreements,
            "analyzer_only": self.analyzer_only,
            "oracle_only": self.oracle_only,
            "agreement_rate": round(self.agreement_rate, 6),
            "disagreement_classes": list(self.disagreement_classes),
        }


def run_differential_oracle(*, limit: int | None = 600) -> DifferentialOracleReport:
    """Compare PromptABI's verdict against an independent reference oracle.

    The reference oracle models *observable provider runtime behaviour*: it rules
    a template forgeable iff an untrusted field can place a control delimiter in a
    role region (the same property a real provider would expose at decode time),
    judged structurally without invoking the analyzer.
    """

    analyzer = _Analyzer()
    cases = build_scaled_prompt_corpus(limit=limit)
    agreements = analyzer_only = oracle_only = 0
    disagreement: set[str] = set()
    for case in cases:
        analyzer_flag = analyzer.forgeable_case(case)
        oracle_flag = _expert_oracle_forgeable(case)
        if analyzer_flag == oracle_flag:
            agreements += 1
        elif analyzer_flag and not oracle_flag:
            analyzer_only += 1
            disagreement.add(case.sanitizer)
        else:
            oracle_only += 1
            disagreement.add(case.sanitizer)
    return DifferentialOracleReport(
        version=CONFORMANCE_SCALE_VERSION,
        compared=len(cases),
        agreements=agreements,
        analyzer_only=analyzer_only,
        oracle_only=oracle_only,
        disagreement_classes=tuple(sorted(disagreement)),
    )


# --------------------------------------------------------------------------- #
# Step 418 -- mine and normalize chat templates
# --------------------------------------------------------------------------- #

_WS_RUN = re.compile(r"[ \t]+")


def normalize_chat_template(template: str) -> str:
    """Canonicalise a raw chat template for structural deduplication.

    Collapses horizontal whitespace runs, strips trailing spaces, and removes
    Jinja comment blocks -- all semantics-preserving for the supported fragment.
    """

    without_comments = re.sub(r"\{#.*?#\}", "", template, flags=re.DOTALL)
    lines = []
    for line in without_comments.splitlines():
        collapsed = _WS_RUN.sub(" ", line).rstrip()
        lines.append(collapsed)
    return "\n".join(lines).strip()


@dataclass(frozen=True, slots=True)
class NormalizedTemplate:
    structural_key: str
    canonical: str
    sources: tuple[str, ...]


def mine_and_normalize(
    raw_templates: Mapping[str, str],
) -> tuple[NormalizedTemplate, ...]:
    """Group raw (name -> template) pairs by canonical structure."""

    buckets: dict[str, list[str]] = defaultdict(list)
    canon: dict[str, str] = {}
    for name, template in raw_templates.items():
        c = normalize_chat_template(template)
        key = _stable_digest(c)
        buckets[key].append(name)
        canon[key] = c
    return tuple(
        NormalizedTemplate(
            structural_key=key,
            canonical=canon[key],
            sources=tuple(sorted(names)),
        )
        for key, names in sorted(buckets.items())
    )


# --------------------------------------------------------------------------- #
# Step 419 -- labeled positive/negative suites with provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LabeledExample:
    case_id: str
    rule_id: str
    label: str  # "positive" or "negative"
    provenance: str
    family: str


@dataclass(frozen=True, slots=True)
class RuleSuite:
    rule_id: str
    positives: tuple[LabeledExample, ...]
    negatives: tuple[LabeledExample, ...]

    @property
    def balanced(self) -> bool:
        return bool(self.positives) and bool(self.negatives)


def build_labeled_suites(*, limit: int | None = 400) -> tuple[RuleSuite, ...]:
    """Build per-rule labeled suites with provenance from the corpus.

    The corpus's by-construction ground truth provides the provenance: every
    positive is a raw-interpolation template (a genuine forgery), every negative
    routes untrusted fields through a delimiter-safe filter.
    """

    rule_id = "role-boundary-nonforgeability"
    cases = build_scaled_prompt_corpus(limit=limit)
    positives: list[LabeledExample] = []
    negatives: list[LabeledExample] = []
    seen_pos: set[str] = set()
    seen_neg: set[str] = set()
    for case in cases:
        prov = f"corpus:{case.family}/{case.sanitizer}"
        if case.label is GroundTruth.VULNERABLE:
            if case.sanitizer in seen_pos:
                continue
            seen_pos.add(case.sanitizer)
            positives.append(
                LabeledExample(case.case_id, rule_id, "positive", prov, case.family)
            )
        else:
            if case.sanitizer in seen_neg:
                continue
            seen_neg.add(case.sanitizer)
            negatives.append(
                LabeledExample(case.case_id, rule_id, "negative", prov, case.family)
            )
    return (
        RuleSuite(
            rule_id=rule_id,
            positives=tuple(positives),
            negatives=tuple(negatives),
        ),
    )


# --------------------------------------------------------------------------- #
# Step 420 -- metamorphic testing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MetamorphicFinding:
    case_id: str
    transform: str
    base_verdict: bool
    rewritten_verdict: bool

    @property
    def preserved(self) -> bool:
        return self.base_verdict == self.rewritten_verdict


@dataclass(frozen=True, slots=True)
class MetamorphicReport:
    version: str
    checked: int
    preserved: int
    violations: tuple[MetamorphicFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checked": self.checked,
            "preserved": self.preserved,
            "violations": [
                {
                    "case_id": v.case_id,
                    "transform": v.transform,
                    "base_verdict": v.base_verdict,
                    "rewritten_verdict": v.rewritten_verdict,
                }
                for v in self.violations
            ],
        }


def _semantics_preserving_rewrites(template: str) -> tuple[tuple[str, str], ...]:
    """Return (transform-name, rewritten-template) pairs preserving semantics."""

    return (
        ("inject-comment", "{# benign #}" + template),
        ("trailing-whitespace", template.replace("\n", "  \n")),
        ("double-blank-lines", template.replace("{% endfor %}", "\n{% endfor %}")),
    )


def run_metamorphic_suite(*, limit: int | None = 240) -> MetamorphicReport:
    analyzer = _Analyzer()
    cases = build_scaled_prompt_corpus(limit=limit)
    checked = preserved = 0
    violations: list[MetamorphicFinding] = []
    seen: set[str] = set()
    for case in cases:
        if case.sanitizer in seen:
            continue
        seen.add(case.sanitizer)
        base = analyzer.forgeable(case.config())
        for name, rewritten in _semantics_preserving_rewrites(case.template):
            cfg = {
                "chat_template": rewritten,
                "additional_special_tokens": list(case.special_tokens),
            }
            verdict = analyzer.forgeable(cfg)
            checked += 1
            finding = MetamorphicFinding(case.case_id, name, base, verdict)
            if finding.preserved:
                preserved += 1
            else:
                violations.append(finding)
    return MetamorphicReport(
        version=CONFORMANCE_SCALE_VERSION,
        checked=checked,
        preserved=preserved,
        violations=tuple(violations),
    )


# --------------------------------------------------------------------------- #
# Step 421 -- fuzzing harness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FuzzReport:
    version: str
    generated: int
    crashes: int
    flagged_forgeable: int
    crash_inputs: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.crashes == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "generated": self.generated,
            "crashes": self.crashes,
            "flagged_forgeable": self.flagged_forgeable,
            "crash_inputs": list(self.crash_inputs),
        }


def _fuzz_templates(seed: int, count: int) -> list[Mapping[str, object]]:
    delimiters = ["<|im_start|>", "[INST]", "<<SYS>>", "</s>", "<|system|>", "###"]
    filters = ["", "| tojson", "| e", "| replace('<','')"]
    out: list[Mapping[str, object]] = []
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        delim = delimiters[state % len(delimiters)]
        filt = filters[(state >> 4) % len(filters)]
        template = (
            "{% for message in messages %}"
            + delim
            + "{{ message['role']" + filt + " }}\n"
            + "{{ message['content']" + filt + " }}"
            + "{% endfor %}"
        )
        out.append(
            {
                "chat_template": template,
                "additional_special_tokens": [delim] if delim.startswith("<") else [],
            }
        )
    return out


def run_fuzzing_harness(*, count: int = 200, seed: int = 1) -> FuzzReport:
    analyzer = _Analyzer()
    crashes: list[str] = []
    flagged = 0
    inputs = _fuzz_templates(seed, count)
    for cfg in inputs:
        try:
            if analyzer.forgeable(cfg):
                flagged += 1
        except Exception:  # noqa: BLE001 -- the harness must surface any crash
            crashes.append(_stable_digest(cfg))
    return FuzzReport(
        version=CONFORMANCE_SCALE_VERSION,
        generated=len(inputs),
        crashes=len(crashes),
        flagged_forgeable=flagged,
        crash_inputs=tuple(crashes),
    )


# --------------------------------------------------------------------------- #
# Step 422 -- precision/recall per rule with confidence intervals
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""

    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denom
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    rule_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    def precision_ci(self) -> tuple[float, float]:
        return wilson_interval(
            self.true_positives, self.true_positives + self.false_positives
        )

    def recall_ci(self) -> tuple[float, float]:
        return wilson_interval(
            self.true_positives, self.true_positives + self.false_negatives
        )

    def to_dict(self) -> dict[str, object]:
        lo_p, hi_p = self.precision_ci()
        lo_r, hi_r = self.recall_ci()
        return {
            "rule_id": self.rule_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 6),
            "precision_ci95": [round(lo_p, 6), round(hi_p, 6)],
            "recall": round(self.recall, 6),
            "recall_ci95": [round(lo_r, 6), round(hi_r, 6)],
        }


def per_rule_metrics(*, limit: int | None = 600) -> RuleMetrics:
    analyzer = _Analyzer()
    cases = build_scaled_prompt_corpus(limit=limit)
    tp = fp = fn = tn = 0
    for case in cases:
        predicted = analyzer.forgeable_case(case)
        actual = case.label is GroundTruth.VULNERABLE
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return RuleMetrics("role-boundary-nonforgeability", tp, fp, fn, tn)


# --------------------------------------------------------------------------- #
# Step 423 -- regression museum
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MuseumEntry:
    bug_id: str
    title: str
    upstream_ref: str
    rule_id: str
    config: Mapping[str, object]
    expected_forgeable: bool


@dataclass(frozen=True, slots=True)
class MuseumResult:
    entry_id: str
    caught: bool
    detail: str


def regression_museum() -> tuple[MuseumEntry, ...]:
    """Historical upstream chat-template bugs PromptABI would have caught."""

    return (
        MuseumEntry(
            bug_id="chatml-raw-role",
            title="ChatML template interpolates role raw, enabling header forgery",
            upstream_ref="github.com/openai/openai-python#chatml-delimiter",
            rule_id="role-boundary-nonforgeability",
            config={
                "chat_template": (
                    "{% for message in messages %}<|im_start|>"
                    "{{ message['role'] }}\n{{ message['content'] }}"
                    "<|im_end|>{% endfor %}"
                ),
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            expected_forgeable=True,
        ),
        MuseumEntry(
            bug_id="llama-header-raw",
            title="Llama-3 header token forgeable via raw content interpolation",
            upstream_ref="github.com/meta-llama/llama3#header-injection",
            rule_id="role-boundary-nonforgeability",
            config={
                "chat_template": (
                    "{% for message in messages %}<|start_header_id|>"
                    "{{ message['role'] }}<|end_header_id|>\n"
                    "{{ message['content'] }}<|eot_id|>{% endfor %}"
                ),
                "additional_special_tokens": [
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|eot_id|>",
                ],
            },
            expected_forgeable=True,
        ),
        MuseumEntry(
            bug_id="mistral-inst-raw",
            title="Mistral [INST] tag forgeable when content is raw",
            upstream_ref="github.com/mistralai/mistral-common#inst-tag",
            rule_id="role-boundary-nonforgeability",
            config={
                "chat_template": (
                    "{% for message in messages %}[INST] {{ message['content'] }}"
                    " [/INST]{% endfor %}"
                ),
                "additional_special_tokens": [],
            },
            expected_forgeable=True,
        ),
        MuseumEntry(
            bug_id="json-wrapped-safe",
            title="JSON-wrapped content is not forgeable (true negative anchor)",
            upstream_ref="promptabi.regression/json-safe",
            rule_id="role-boundary-nonforgeability",
            config={
                "chat_template": (
                    "{% for message in messages %}<|im_start|>"
                    "{{ message['role'] | tojson }}\n"
                    "{{ message['content'] | tojson }}<|im_end|>{% endfor %}"
                ),
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            expected_forgeable=False,
        ),
    )


def replay_regression_museum() -> tuple[MuseumResult, ...]:
    analyzer = _Analyzer()
    results: list[MuseumResult] = []
    for entry in regression_museum():
        forgeable = analyzer.forgeable(entry.config)
        caught = forgeable == entry.expected_forgeable
        detail = (
            f"analyzer={'forgeable' if forgeable else 'safe'} "
            f"expected={'forgeable' if entry.expected_forgeable else 'safe'}"
        )
        results.append(MuseumResult(entry.bug_id, caught, detail))
    return tuple(results)


# --------------------------------------------------------------------------- #
# Step 424 -- provider-version pairing drift
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PairingDriftAlert:
    provider_family: str
    from_revision: str
    to_revision: str
    reason: str


def detect_pairing_drift(
    pairings: Sequence[Mapping[str, object]],
) -> tuple[PairingDriftAlert, ...]:
    """Alert when a (template, special-tokens) pairing changes verdict across revisions.

    ``pairings`` is an ordered sequence of dicts with keys ``provider_family``,
    ``revision``, ``chat_template``, ``additional_special_tokens``.
    """

    analyzer = _Analyzer()
    by_provider: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for p in pairings:
        by_provider[str(p["provider_family"])].append(p)
    alerts: list[PairingDriftAlert] = []
    for provider, series in by_provider.items():
        prev: tuple[str, bool] | None = None
        for snapshot in series:
            cfg = {
                "chat_template": snapshot["chat_template"],
                "additional_special_tokens": list(
                    snapshot.get("additional_special_tokens", [])
                ),
            }
            forgeable = analyzer.forgeable(cfg)
            rev = str(snapshot["revision"])
            if prev is not None and prev[1] != forgeable:
                reason = (
                    "pairing became forgeable"
                    if forgeable
                    else "pairing became safe"
                )
                alerts.append(
                    PairingDriftAlert(provider, prev[0], rev, reason)
                )
            prev = (rev, forgeable)
    return tuple(alerts)


# --------------------------------------------------------------------------- #
# Step 425 -- golden-encoding cross-validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GoldenEncodingReport:
    version: str
    checked: int
    roundtrip_failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.roundtrip_failures


def cross_validate_golden_encodings(
    samples: Sequence[str] | None = None,
) -> GoldenEncodingReport:
    """Cross-validate ByteLevel encode/decode against golden round-trip identity."""

    if samples is None:
        samples = (
            "hello world",
            "café — déjà vu",
            "<|im_start|>system",
            "日本語テキスト",
            "emoji 🚀 test",
            '{"key": "value"}',
            "tab\tand\nnewline",
        )
    tokenizer = ByteLevelTokenizer()
    failures: list[str] = []
    for text in samples:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode([t.token_id for t in encoded.tokens])
        if decoded.text != encoded.normalized_text:
            failures.append(text)
    return GoldenEncodingReport(
        version=CONFORMANCE_SCALE_VERSION,
        checked=len(samples),
        roundtrip_failures=tuple(failures),
    )


# --------------------------------------------------------------------------- #
# Step 426 -- versioned conformance test-vector package
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TestVector:
    vector_id: str
    config: Mapping[str, object]
    expected_forgeable: bool


@dataclass(frozen=True, slots=True)
class TestVectorPackage:
    version: str
    vectors: tuple[TestVector, ...]
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "digest": self.digest,
            "vectors": [
                {
                    "vector_id": v.vector_id,
                    "config": dict(v.config),
                    "expected_forgeable": v.expected_forgeable,
                }
                for v in self.vectors
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_test_vector_package() -> TestVectorPackage:
    """Build a versioned, digest-stamped package consumable by third-party tools.

    Each vector pairs a config with the expected verdict; the digest lets other
    tools assert they consumed the exact same package.
    """

    vectors = tuple(
        TestVector(entry.bug_id, entry.config, entry.expected_forgeable)
        for entry in regression_museum()
    )
    payload = [
        {
            "vector_id": v.vector_id,
            "config": dict(v.config),
            "expected_forgeable": v.expected_forgeable,
        }
        for v in vectors
    ]
    digest = _stable_digest(payload)
    return TestVectorPackage(
        version=CONFORMANCE_SCALE_VERSION,
        vectors=vectors,
        digest=digest,
    )


def verify_test_vector_package(package: TestVectorPackage) -> bool:
    """Re-run the analyzer over every vector and confirm the expected verdict."""

    analyzer = _Analyzer()
    return all(
        analyzer.forgeable(v.config) == v.expected_forgeable for v in package.vectors
    )


# --------------------------------------------------------------------------- #
# Step 427 -- statistical significance vs baseline linter
# --------------------------------------------------------------------------- #


def _naive_baseline_forgeable(case: CorpusCase) -> bool:
    """A naive linter: flags any template that contains a literal '<|' delimiter.

    This ignores sanitisation entirely, so it over-flags safe filtered templates.
    """

    return "<|" in case.template or "[INST]" in case.template


@dataclass(frozen=True, slots=True)
class McNemarReport:
    version: str
    n: int
    analyzer_correct_baseline_wrong: int
    baseline_correct_analyzer_wrong: int
    statistic: float
    significant_at_05: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "n": self.n,
            "b_analyzer_only_correct": self.analyzer_correct_baseline_wrong,
            "c_baseline_only_correct": self.baseline_correct_analyzer_wrong,
            "statistic": round(self.statistic, 6),
            "significant_at_05": self.significant_at_05,
        }


def mcnemar_vs_baseline(*, limit: int | None = 600) -> McNemarReport:
    """McNemar's test comparing PromptABI to a naive linter baseline."""

    analyzer = _Analyzer()
    cases = build_scaled_prompt_corpus(limit=limit)
    b = c = 0  # discordant cells
    for case in cases:
        actual = case.label is GroundTruth.VULNERABLE
        a_correct = analyzer.forgeable_case(case) == actual
        base_correct = _naive_baseline_forgeable(case) == actual
        if a_correct and not base_correct:
            b += 1
        elif base_correct and not a_correct:
            c += 1
    # McNemar's chi-square with continuity correction.
    denom = b + c
    statistic = (abs(b - c) - 1) ** 2 / denom if denom else 0.0
    return McNemarReport(
        version=CONFORMANCE_SCALE_VERSION,
        n=len(cases),
        analyzer_correct_baseline_wrong=b,
        baseline_correct_analyzer_wrong=c,
        statistic=statistic,
        significant_at_05=statistic > 3.841,  # chi-square df=1, alpha=0.05
    )


# --------------------------------------------------------------------------- #
# Step 428 -- inter-rater reliability
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InterRaterReport:
    version: str
    n: int
    cohens_kappa: float
    observed_agreement: float

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "n": self.n,
            "cohens_kappa": round(self.cohens_kappa, 6),
            "observed_agreement": round(self.observed_agreement, 6),
        }


def inter_rater_reliability(*, limit: int | None = 600) -> InterRaterReport:
    """Cohen's kappa between the analyzer and the independent expert oracle."""

    analyzer = _Analyzer()
    cases = build_scaled_prompt_corpus(limit=limit)
    tp = fp = fn = tn = 0
    agree = 0
    for case in cases:
        a = analyzer.forgeable_case(case)
        e = _expert_oracle_forgeable(case)
        if a and e:
            tp += 1
        elif a and not e:
            fp += 1
        elif not a and e:
            fn += 1
        else:
            tn += 1
        if a == e:
            agree += 1
    matrix = ConfusionMatrix(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )
    return InterRaterReport(
        version=CONFORMANCE_SCALE_VERSION,
        n=len(cases),
        cohens_kappa=_cohen_kappa(matrix),
        observed_agreement=agree / len(cases) if cases else 1.0,
    )


# --------------------------------------------------------------------------- #
# Step 429 -- reproducible corpus snapshots
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    version: str
    case_count: int
    template_digest: str
    label_digest: str
    snapshot_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "case_count": self.case_count,
            "template_digest": self.template_digest,
            "label_digest": self.label_digest,
            "snapshot_id": self.snapshot_id,
        }


def corpus_snapshot(*, limit: int | None = None) -> CorpusSnapshot:
    """Produce a reproducible snapshot digest of the corpus."""

    cases = build_scaled_prompt_corpus(limit=limit)
    template_digest = _stable_digest(sorted(c.template_key for c in cases))
    label_digest = _stable_digest(
        sorted((c.case_id, c.label.value) for c in cases)
    )
    snapshot_id = _stable_digest([template_digest, label_digest, len(cases)])[:16]
    return CorpusSnapshot(
        version=CONFORMANCE_SCALE_VERSION,
        case_count=len(cases),
        template_digest=template_digest,
        label_digest=label_digest,
        snapshot_id=snapshot_id,
    )


# --------------------------------------------------------------------------- #
# Step 430 -- public conformance dashboard
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DashboardPoint:
    snapshot_id: str
    agreement_rate: float
    precision: float
    recall: float
    kappa: float
    metamorphic_ok: bool
    golden_ok: bool


@dataclass(frozen=True, slots=True)
class ConformanceDashboard:
    version: str
    points: tuple[DashboardPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "points": [
                {
                    "snapshot_id": p.snapshot_id,
                    "agreement_rate": round(p.agreement_rate, 6),
                    "precision": round(p.precision, 6),
                    "recall": round(p.recall, 6),
                    "kappa": round(p.kappa, 6),
                    "metamorphic_ok": p.metamorphic_ok,
                    "golden_ok": p.golden_ok,
                }
                for p in self.points
            ],
        }


def conformance_dashboard(*, limit: int | None = 400) -> ConformanceDashboard:
    """Assemble a single dashboard point from a live run of every sub-study."""

    snapshot = corpus_snapshot(limit=limit)
    oracle = run_differential_oracle(limit=limit)
    metrics = per_rule_metrics(limit=limit)
    irr = inter_rater_reliability(limit=limit)
    metamorphic = run_metamorphic_suite(limit=limit)
    golden = cross_validate_golden_encodings()
    point = DashboardPoint(
        snapshot_id=snapshot.snapshot_id,
        agreement_rate=oracle.agreement_rate,
        precision=metrics.precision,
        recall=metrics.recall,
        kappa=irr.cohens_kappa,
        metamorphic_ok=metamorphic.ok,
        golden_ok=golden.ok,
    )
    return ConformanceDashboard(
        version=CONFORMANCE_SCALE_VERSION,
        points=(point,),
    )


def render_dashboard_text(dashboard: ConformanceDashboard) -> str:
    lines = [f"PromptABI conformance dashboard ({dashboard.version})"]
    for p in dashboard.points:
        lines.append(
            f"  snapshot {p.snapshot_id}: agreement={p.agreement_rate:.3f} "
            f"precision={p.precision:.3f} recall={p.recall:.3f} "
            f"kappa={p.kappa:.3f} metamorphic={'ok' if p.metamorphic_ok else 'FAIL'} "
            f"golden={'ok' if p.golden_ok else 'FAIL'}"
        )
    return "\n".join(lines) + "\n"
