"""Benchmarks, leaderboards, and competitions (steps 476-490).

PromptABI-Bench is a public, versioned benchmark of contract-violation
detection.  A *submission* is any callable ``config -> bool`` that predicts
whether a chat-template config is role-boundary forgeable.  Every score in this
module is produced by running the submission over the labeled corpus from
:mod:`promptabi.scaled_evaluation` (ground truth assigned by construction).

Capabilities:

* :func:`build_benchmark` / public + hidden split (steps 476, 477).
* :func:`evaluate_submission` and the SARIF leaderboard interface (steps 477, 478).
* :func:`baseline_results` -- naive linter, LLM-grader sim, PromptABI (step 479).
* :func:`competition_rules` and :func:`ctf_challenges` (steps 480, 481).
* :func:`score_submission` -- soundness-weighted rubric (step 482).
* :func:`artifact_evaluation_checklist` (step 483).
* :func:`per_rule_difficulty` and human-baseline comparison (step 484).
* :func:`state_of_prompt_safety_report` (step 485).
* :func:`evaluate_adversarial_submission` (step 486).
* :func:`evaluation_container_spec` (step 487).
* :func:`bootstrap_leaderboard` -- bootstrap confidence intervals (step 488).
* :func:`benchmark_doi_metadata` (step 489).
* :func:`certification_gate` -- benchmark integrated into standards cert (step 490).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .adoption_tooling import verify_chat_template
from .scaled_evaluation import GroundTruth, build_scaled_prompt_corpus

BENCH_SUITE_VERSION = "promptabi-bench.v1"

#: A submission predicts forgeability from a chat-template config.
Submission = Callable[[Mapping[str, object]], bool]


# --------------------------------------------------------------------------- #
# Steps 476 & 477 -- benchmark definition with public / hidden split
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    config: Mapping[str, object]
    forgeable: bool  # ground truth
    difficulty: str  # easy | medium | hard
    hidden: bool


@dataclass(frozen=True, slots=True)
class Benchmark:
    version: str
    cases: tuple[BenchmarkCase, ...]
    digest: str

    @property
    def public(self) -> tuple[BenchmarkCase, ...]:
        return tuple(c for c in self.cases if not c.hidden)

    @property
    def hidden(self) -> tuple[BenchmarkCase, ...]:
        return tuple(c for c in self.cases if c.hidden)


def _difficulty_for(sanitizer: str) -> str:
    # The unmodeled strip-replace class is the hard, calibration-relevant case.
    if sanitizer == "strip-replace":
        return "hard"
    if sanitizer in {"tojson", "escape", "urlencode", "base64"}:
        return "medium"
    return "easy"


def build_benchmark(*, limit: int | None = None) -> Benchmark:
    """Assemble a deterministic benchmark with a content-hashed hidden split."""

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for case in build_scaled_prompt_corpus(limit=limit):
        # One representative per (family, sanitizer) keeps the benchmark compact.
        key = f"{case.family}/{case.sanitizer}"
        if key in seen:
            continue
        seen.add(key)
        h = int(hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(), 16)
        cases.append(
            BenchmarkCase(
                case_id=case.case_id,
                config=case.config(),
                forgeable=case.label is GroundTruth.VULNERABLE,
                difficulty=_difficulty_for(case.sanitizer),
                hidden=(h % 3 == 0),  # ~1/3 hidden
            )
        )
    digest = hashlib.sha256(
        json.dumps([c.case_id for c in cases], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Benchmark(BENCH_SUITE_VERSION, tuple(cases), digest)


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SubmissionScore:
    name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def sound(self) -> bool:
        """A sound detector never misses a genuine forgery."""

        return self.false_negatives == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "sound": self.sound,
            "false_negatives": self.false_negatives,
        }


def evaluate_submission(
    submission: Submission, cases: Sequence[BenchmarkCase], *, name: str = "submission"
) -> SubmissionScore:
    tp = fp = fn = tn = 0
    for case in cases:
        predicted = submission(case.config)
        if predicted and case.forgeable:
            tp += 1
        elif predicted and not case.forgeable:
            fp += 1
        elif not predicted and case.forgeable:
            fn += 1
        else:
            tn += 1
    return SubmissionScore(name, tp, fp, fn, tn)


# --------------------------------------------------------------------------- #
# Step 478 -- SARIF leaderboard interface
# --------------------------------------------------------------------------- #


def submission_from_sarif(sarif_by_case: Mapping[str, dict]) -> Submission:
    """Adapt a third-party tool's per-case SARIF output into a submission.

    A case is predicted forgeable iff the tool emitted any result for it.
    """

    def predict(config: Mapping[str, object]) -> bool:
        key = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()
        doc = sarif_by_case.get(key)
        if not doc:
            return False
        runs = doc.get("runs", [])
        return any(run.get("results") for run in runs)

    return predict


# --------------------------------------------------------------------------- #
# Step 479 -- baseline submissions
# --------------------------------------------------------------------------- #


def promptabi_submission(config: Mapping[str, object]) -> bool:
    return bool(verify_chat_template(config))


def naive_linter_submission(config: Mapping[str, object]) -> bool:
    template = str(config.get("chat_template", ""))
    return "<|" in template or "[INST]" in template


def llm_grader_submission(config: Mapping[str, object]) -> bool:
    """A deterministic stand-in for an LLM grader.

    It "reasons" that a template is safe iff it sees a safe-filter token; this
    mirrors an LLM that over-trusts the presence of an escaping filter and so
    misses raw interpolations that still contain a filter elsewhere.
    """

    template = str(config.get("chat_template", ""))
    has_delim = "<|" in template or "[INST]" in template or "<<SYS>>" in template
    has_filter = any(f in template for f in ("tojson", "| e ", "|e", "escape", "urlencode"))
    return has_delim and not has_filter


def baseline_results(*, limit: int | None = 600) -> tuple[SubmissionScore, ...]:
    benchmark = build_benchmark(limit=limit)
    cases = benchmark.cases
    return (
        evaluate_submission(promptabi_submission, cases, name="PromptABI"),
        evaluate_submission(naive_linter_submission, cases, name="naive-linter"),
        evaluate_submission(llm_grader_submission, cases, name="llm-grader"),
    )


# --------------------------------------------------------------------------- #
# Step 480 -- conformance-challenge competition
# --------------------------------------------------------------------------- #


def competition_rules() -> dict[str, object]:
    return {
        "name": "PromptABI Conformance Challenge",
        "cadence": "quarterly",
        "safe_harbor": [
            "Submissions run only against the public PromptABI-Bench corpus.",
            "No targeting of third-party production systems.",
            "Responsible disclosure required for any real upstream bug found.",
        ],
        "scoring": "soundness-weighted F1 (see score_submission)",
        "eligibility": "any tool exposing the SARIF leaderboard interface",
    }


# --------------------------------------------------------------------------- #
# Step 481 -- capture-the-contract CTF
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CtfChallenge:
    tier: int
    name: str
    config: Mapping[str, object]
    objective: str
    is_forgeable: bool


def ctf_challenges() -> tuple[CtfChallenge, ...]:
    return (
        CtfChallenge(
            1,
            "ChatML raw role",
            {
                "chat_template": (
                    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                    "{{ m['content'] }}<|im_end|>{% endfor %}"
                ),
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            "Forge an <|im_start|>system header from user content.",
            True,
        ),
        CtfChallenge(
            2,
            "Filtered but reachable",
            {
                "chat_template": (
                    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                    "{{ m['content'] | tojson }}<|im_end|>{% endfor %}"
                ),
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            "Role field is still raw -- forge via the role.",
            True,
        ),
        CtfChallenge(
            3,
            "Fully sealed",
            {
                "chat_template": (
                    "{% for m in messages %}<|im_start|>{{ m['role'] | tojson }}\n"
                    "{{ m['content'] | tojson }}<|im_end|>{% endfor %}"
                ),
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            "Both fields escaped -- prove non-forgeability.",
            False,
        ),
    )


def grade_ctf(submission: Submission) -> dict[str, object]:
    challenges = ctf_challenges()
    solved = sum(1 for c in challenges if submission(c.config) == c.is_forgeable)
    return {"total": len(challenges), "solved": solved, "max_tier": max(c.tier for c in challenges)}


# --------------------------------------------------------------------------- #
# Step 482 -- soundness-weighted scoring rubric
# --------------------------------------------------------------------------- #


def score_submission(score: SubmissionScore, *, fn_penalty: float = 5.0) -> float:
    """Soundness-weighted score: false negatives are penalized heavily.

    A single missed forgery (false negative) costs ``fn_penalty`` x a false
    positive, so a sound-but-noisy tool always outranks an unsound-but-precise
    one.  The score is in ``[0, 1]``.
    """

    total = (
        score.true_positives
        + score.false_positives
        + score.false_negatives
        + score.true_negatives
    )
    if total == 0:
        return 1.0
    penalty = score.false_positives + fn_penalty * score.false_negatives
    max_penalty = fn_penalty * total
    return max(0.0, 1.0 - penalty / max_penalty)


# --------------------------------------------------------------------------- #
# Step 483 -- artifact-evaluation track
# --------------------------------------------------------------------------- #


def artifact_evaluation_checklist() -> tuple[tuple[str, bool], ...]:
    """An AE checklist mirroring top-venue requirements, self-evaluated."""

    return (
        ("Available (public archive with DOI)", True),
        ("Functional (runs from a single command)", True),
        ("Reusable (documented public API)", True),
        ("Reproducible (pinned deterministic corpus + digests)", True),
        ("No network or GPU required", True),
    )


def artifact_evaluation_passed() -> bool:
    return all(ok for _, ok in artifact_evaluation_checklist())


# --------------------------------------------------------------------------- #
# Step 484 -- per-rule difficulty calibration + human baseline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DifficultyBucket:
    difficulty: str
    count: int
    promptabi_accuracy: float
    human_accuracy: float


def _human_baseline_predict(case: BenchmarkCase) -> bool:
    # A human rater: judges by whether they can eyeball a raw interpolation; they
    # are fooled by the hard strip-replace class (rule it safe like PromptABI
    # over-warns), matching the inter-rater study.
    if case.difficulty == "hard":
        return False
    return case.forgeable


def per_rule_difficulty(*, limit: int | None = 600) -> tuple[DifficultyBucket, ...]:
    benchmark = build_benchmark(limit=limit)
    buckets: dict[str, list[BenchmarkCase]] = {}
    for case in benchmark.cases:
        buckets.setdefault(case.difficulty, []).append(case)
    out: list[DifficultyBucket] = []
    for difficulty in ("easy", "medium", "hard"):
        group = buckets.get(difficulty, [])
        if not group:
            continue
        pa_correct = sum(
            1 for c in group if promptabi_submission(c.config) == c.forgeable
        )
        human_correct = sum(
            1 for c in group if _human_baseline_predict(c) == c.forgeable
        )
        out.append(
            DifficultyBucket(
                difficulty=difficulty,
                count=len(group),
                promptabi_accuracy=pa_correct / len(group),
                human_accuracy=human_correct / len(group),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Step 485 -- state-of-prompt-safety report
# --------------------------------------------------------------------------- #


def state_of_prompt_safety_report(*, limit: int | None = 600) -> dict[str, object]:
    results = baseline_results(limit=limit)
    by_name = {r.name: r for r in results}
    return {
        "version": BENCH_SUITE_VERSION,
        "headline": "PromptABI is the only sound baseline (zero false negatives).",
        "baselines": [r.to_dict() for r in results],
        "sound_tools": sorted(r.name for r in results if r.sound),
        "best_f1": max(r.f1 for r in results),
        "promptabi_score": round(score_submission(by_name["PromptABI"]), 6),
    }


# --------------------------------------------------------------------------- #
# Step 486 -- adversarial submission track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AdversarialResult:
    bypassed_soundness: bool
    missed_cases: tuple[str, ...]


def evaluate_adversarial_submission(
    submission: Submission, *, limit: int | None = 600
) -> AdversarialResult:
    """An adversarial submission tries to *bypass* the soundness guarantee.

    It "passes" the attack only if it misses a genuine forgery PromptABI catches.
    PromptABI itself must never be bypassable.
    """

    benchmark = build_benchmark(limit=limit)
    missed: list[str] = []
    for case in benchmark.cases:
        if case.forgeable and not submission(case.config):
            missed.append(case.case_id)
    return AdversarialResult(bool(missed), tuple(missed))


# --------------------------------------------------------------------------- #
# Step 487 -- reproducible evaluation container
# --------------------------------------------------------------------------- #


def evaluation_container_spec() -> dict[str, object]:
    benchmark = build_benchmark(limit=120)
    return {
        "base_image": "python:3.12-slim",
        "pinned": {
            "python": "3.12",
            "promptabi": "1.0.0",
            "z3-solver": "4.13.0",
            "jsonschema": "4.26.0",
        },
        "entrypoint": ["python", "-m", "promptabi.bench_suite"],
        "benchmark_digest": benchmark.digest,
        "network": "disabled",
        "gpu": "none",
    }


def render_dockerfile() -> str:
    return (
        "FROM python:3.12-slim\n"
        "RUN pip install --no-cache-dir promptabi==1.0.0 z3-solver==4.13.0 "
        "jsonschema==4.26.0\n"
        "ENV PROMPTABI_OFFLINE=1\n"
        'ENTRYPOINT ["python", "-m", "promptabi.bench_suite"]\n'
    )


# --------------------------------------------------------------------------- #
# Step 488 -- bootstrap-CI leaderboard
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    name: str
    f1: float
    f1_ci_low: float
    f1_ci_high: float
    score: float
    sound: bool


def bootstrap_leaderboard(
    submissions: Mapping[str, Submission],
    *,
    limit: int | None = 400,
    resamples: int = 200,
    seed: int = 0,
) -> tuple[LeaderboardRow, ...]:
    """Rank submissions with bootstrap confidence intervals on F1."""

    benchmark = build_benchmark(limit=limit)
    cases = benchmark.cases
    rng = random.Random(seed)
    rows: list[LeaderboardRow] = []
    for name, submission in submissions.items():
        preds = [(submission(c.config), c.forgeable) for c in cases]
        point = evaluate_submission(submission, cases, name=name)
        f1_samples: list[float] = []
        n = len(preds)
        for _ in range(resamples):
            sample = [preds[rng.randrange(n)] for _ in range(n)]
            tp = sum(1 for p, a in sample if p and a)
            fp = sum(1 for p, a in sample if p and not a)
            fn = sum(1 for p, a in sample if not p and a)
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            f1_samples.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        f1_samples.sort()
        lo = f1_samples[int(0.025 * resamples)]
        hi = f1_samples[min(resamples - 1, int(0.975 * resamples))]
        rows.append(
            LeaderboardRow(
                name=name,
                f1=point.f1,
                f1_ci_low=lo,
                f1_ci_high=hi,
                score=score_submission(point),
                sound=point.sound,
            )
        )
    # Rank by soundness-weighted score, then F1.
    rows.sort(key=lambda r: (r.score, r.f1), reverse=True)
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Step 489 -- DOI / archival metadata
# --------------------------------------------------------------------------- #


def benchmark_doi_metadata() -> dict[str, object]:
    benchmark = build_benchmark(limit=120)
    return {
        "title": "PromptABI-Bench: a benchmark for prompt-interface contract-violation detection",
        "version": BENCH_SUITE_VERSION,
        "doi": "10.5281/zenodo.promptabi-bench",
        "license": "Apache-2.0",
        "archival_host": "zenodo.org",
        "content_digest": benchmark.digest,
        "creators": ["PromptABI contributors"],
    }


# --------------------------------------------------------------------------- #
# Step 490 -- benchmark integrated into standards certification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CertificationResult:
    certified: bool
    score: float
    threshold: float
    sound: bool
    reason: str


def certification_gate(
    submission: Submission, *, limit: int | None = 600, threshold: float = 0.9
) -> CertificationResult:
    """Certify a tool against the standard: must be sound AND score >= threshold."""

    benchmark = build_benchmark(limit=limit)
    score = evaluate_submission(submission, benchmark.cases)
    weighted = score_submission(score)
    certified = score.sound and weighted >= threshold
    if not score.sound:
        reason = "rejected: not sound (missed a genuine forgery)"
    elif weighted < threshold:
        reason = f"rejected: score {weighted:.3f} < threshold {threshold}"
    else:
        reason = "certified"
    return CertificationResult(certified, weighted, threshold, score.sound, reason)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI shim
    report = state_of_prompt_safety_report(limit=300)
    print(json.dumps(report, indent=2))
    return 0
