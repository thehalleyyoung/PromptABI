"""Training-contract benchmark suites (step 279).

A benchmark suite is a curated set of *scenarios*, each pairing a concrete input
with the verdict PromptABI's training-contract analyzers should return.  Running
the suite drives the real analyzers (loss-mask certification, target-span
survival, preference symmetry, dataset transforms, benchmark leakage) and scores
how many scenarios produced the expected verdict.  Because every scenario calls
production code, the suite doubles as a regression benchmark and as evidence that
the analyzers behave as documented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .benchmark_leakage import (
    BenchmarkItem,
    SyntheticExample,
    detect_leakage,
)
from .dataset_transforms import (
    DatasetContract,
    InterfaceFootprint,
    TransformPass,
    verify_pipeline,
)
from .loss_mask_certification import LoaderMask, TargetSpec, certify_loss_masks
from .preference_symmetry import PreferencePair, Turn, verify_preference_pair
from .target_spans import Span, TruncationSide, verify_target_spans

TRAINING_BENCHMARK_VERSION = "promptabi.training-benchmark.v1"


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    category: str
    evaluate: Callable[[], bool]
    expected_pass: bool


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    name: str
    category: str
    expected_pass: bool
    actual_pass: bool
    correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "expected_pass": self.expected_pass,
            "actual_pass": self.actual_pass,
            "correct": self.correct,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    version: str
    correct: int
    total: int
    outcomes: tuple[ScenarioOutcome, ...] = field(default=())

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def run_suite(scenarios: tuple[Scenario, ...]) -> BenchmarkReport:
    outcomes: list[ScenarioOutcome] = []
    for sc in scenarios:
        actual = bool(sc.evaluate())
        outcomes.append(
            ScenarioOutcome(
                name=sc.name,
                category=sc.category,
                expected_pass=sc.expected_pass,
                actual_pass=actual,
                correct=(actual == sc.expected_pass),
            )
        )
    correct = sum(1 for o in outcomes if o.correct)
    return BenchmarkReport(
        version=TRAINING_BENCHMARK_VERSION,
        correct=correct,
        total=len(outcomes),
        outcomes=tuple(outcomes),
    )


def default_training_suite() -> tuple[Scenario, ...]:
    """A built-in suite exercising each training-contract analyzer both ways."""

    def good_loss_mask() -> bool:
        spec = TargetSpec(length=4, target_positions=frozenset({2, 3}))
        loaders = (LoaderMask("a", (0, 0, 1, 1)), LoaderMask("b", (0, 0, 1, 1)))
        return certify_loss_masks(spec, loaders).certified

    def bad_loss_mask() -> bool:
        spec = TargetSpec(length=4, target_positions=frozenset({2, 3}))
        loaders = (LoaderMask("a", (1, 0, 1, 1)),)  # supervises prompt pos 0
        return certify_loss_masks(spec, loaders).certified

    def good_target_span() -> bool:
        return verify_target_spans(
            10, 8, TruncationSide.RIGHT, (Span("resp", 2, 6),)
        ).preserved

    def bad_target_span() -> bool:
        return verify_target_spans(
            10, 4, TruncationSide.RIGHT, (Span("resp", 6, 9),)
        ).preserved

    def good_preference() -> bool:
        chosen = (Turn("user", "hi"), Turn("assistant", "great answer"))
        rejected = (Turn("user", "hi"), Turn("assistant", "bad answer"))
        return verify_preference_pair(PreferencePair(chosen, rejected)).symmetric

    def bad_preference() -> bool:
        chosen = (Turn("user", "hi"), Turn("assistant", "x"))
        rejected = (Turn("user", "DIFFERENT"), Turn("assistant", "y"))
        return verify_preference_pair(PreferencePair(chosen, rejected)).symmetric

    def good_pipeline() -> bool:
        contract = DatasetContract(frozenset({"system", "user"}), frozenset({"<eos>"}))
        init = InterfaceFootprint(frozenset({"system", "user"}), frozenset({"<eos>"}), True)
        passes = (TransformPass("noop", lambda fp: fp),)
        return verify_pipeline(contract, init, passes).preserved

    def bad_pipeline() -> bool:
        contract = DatasetContract(frozenset({"system", "user"}), frozenset({"<eos>"}))
        init = InterfaceFootprint(frozenset({"system", "user"}), frozenset({"<eos>"}), True)
        passes = (
            TransformPass(
                "inject",
                lambda fp: InterfaceFootprint(fp.roles | {"tool"}, fp.tokens, fp.has_eos),
            ),
        )
        return verify_pipeline(contract, init, passes).preserved

    def clean_synth() -> bool:
        return detect_leakage(
            (SyntheticExample("s1", "the sky is mostly blue today"),),
            (BenchmarkItem("b1", "what is the capital of france"),),
        ).clean

    def leaked_synth() -> bool:
        return detect_leakage(
            (SyntheticExample("s1", "answer: what is the capital of france"),),
            (BenchmarkItem("b1", "what is the capital of france"),),
        ).clean

    return (
        Scenario("loss-mask-good", "loss-mask", good_loss_mask, True),
        Scenario("loss-mask-bad", "loss-mask", bad_loss_mask, False),
        Scenario("target-span-good", "target-span", good_target_span, True),
        Scenario("target-span-bad", "target-span", bad_target_span, False),
        Scenario("preference-good", "preference", good_preference, True),
        Scenario("preference-bad", "preference", bad_preference, False),
        Scenario("pipeline-good", "transforms", good_pipeline, True),
        Scenario("pipeline-bad", "transforms", bad_pipeline, False),
        Scenario("leakage-clean", "leakage", clean_synth, True),
        Scenario("leakage-contaminated", "leakage", leaked_synth, False),
    )


def render_benchmark_text(report: BenchmarkReport) -> str:
    lines = [
        f"PromptABI training-contract benchmark ({report.version})",
        f"accuracy: {report.correct}/{report.total} ({report.accuracy:.0%})",
    ]
    for o in report.outcomes:
        mark = "ok" if o.correct else "MISS"
        lines.append(f"  [{mark}] {o.name} ({o.category})")
    return "\n".join(lines) + "\n"
