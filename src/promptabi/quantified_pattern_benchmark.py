"""Benchmark quantified-pattern approximations (step 228).

PromptABI's solver fragment is quantifier-free, yet many prompt-interface
obligations are naturally *universally quantified*: "for every retrieved chunk
its length fits the window", "for every message the role header is well formed",
"for every tool call the arguments parse".  PromptABI discharges those
obligations by expanding the quantifier over the finite, bounded index set --
but on large index sets it is tempting to *approximate* the expansion by
sampling a subset of indices.

A sampled approximation is **fast** but potentially **unsound**: if the one
index that violates the obligation is not in the sample, the approximation
reports "safe" while the exact expansion finds the counterexample.  This module
benchmarks that trade-off.  For a corpus of quantified patterns it runs the
exact finite expansion and a sampling approximation, compares verdicts, measures
the speedup, and -- crucially -- flags every approximation that is *unsound*
(claims the obligation holds when the exact expansion refutes it).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Sequence

from .formal import (
    FiniteContractProblem,
    Gt,
    IntRangeDomain,
    NamedConstraint,
    Or,
    SolverStatus,
    Value,
    Var,
)

QUANTIFIED_PATTERN_BENCHMARK_VERSION = "promptabi.quantified-pattern-benchmark.v1"


class QuantifiedVerdict(StrEnum):
    HOLDS = "holds"
    VIOLATED = "violated"


@dataclass(frozen=True, slots=True)
class QuantifiedPattern:
    """A universally-quantified obligation over a finite, bounded index set.

    Each index ``i`` contributes a bounded integer variable ``0 <= v_i <= dom_i``
    and the obligation ``v_i <= cap``.  The obligation is *violated* iff some
    ``dom_i > cap``; the index set is otherwise symmetric, so a sampling
    approximation is unsound exactly when it omits a violating index.
    """

    name: str
    domains: tuple[int, ...]
    cap: int

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("a quantified pattern needs at least one index")
        if self.cap < 0:
            raise ValueError("cap must be non-negative")

    def indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.domains)))

    def counterexample_problem(self, included: Sequence[int]) -> FiniteContractProblem:
        included = tuple(included)
        if not included:
            raise ValueError("at least one index must be included")
        variables = tuple(
            IntRangeDomain(name=f"v{i}", minimum=0, maximum=self.domains[i]) for i in included
        )
        violators = Or(*(Gt(Var(f"v{i}"), Value(self.cap)) for i in included))
        constraint = NamedConstraint(name="some-index-violates", expression=violators)
        return FiniteContractProblem(
            variables=variables,
            constraints=(constraint,),
            name=f"{self.name}:{'-'.join(str(i) for i in included)}",
        )


SamplingStrategy = Callable[[QuantifiedPattern], tuple[int, ...]]


def first_k_strategy(k: int) -> SamplingStrategy:
    """Sample the first ``k`` indices (a common cheap approximation)."""

    if k <= 0:
        raise ValueError("k must be positive")

    def sample(pattern: QuantifiedPattern) -> tuple[int, ...]:
        return pattern.indices()[:k]

    return sample


def _verdict(problem: FiniteContractProblem) -> tuple[QuantifiedVerdict, float]:
    start = time.perf_counter()
    result = problem.solve(prefer_z3=True)
    elapsed = time.perf_counter() - start
    verdict = QuantifiedVerdict.VIOLATED if result.status is SolverStatus.SAT else QuantifiedVerdict.HOLDS
    return verdict, elapsed


@dataclass(frozen=True, slots=True)
class QuantifiedPatternResult:
    pattern: str
    exact_verdict: str
    approx_verdict: str
    exact_indices: int
    approx_indices: int
    exact_seconds: float
    approx_seconds: float
    sound: bool
    precise: bool

    @property
    def speedup(self) -> float:
        if self.approx_seconds <= 0:
            return float("inf")
        return self.exact_seconds / self.approx_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern": self.pattern,
            "exact_verdict": self.exact_verdict,
            "approx_verdict": self.approx_verdict,
            "exact_indices": self.exact_indices,
            "approx_indices": self.approx_indices,
            "exact_seconds": round(self.exact_seconds, 6),
            "approx_seconds": round(self.approx_seconds, 6),
            "speedup": round(self.speedup, 3) if self.speedup != float("inf") else None,
            "sound": self.sound,
            "precise": self.precise,
        }


@dataclass(frozen=True, slots=True)
class QuantifiedPatternBenchmark:
    version: str
    results: tuple[QuantifiedPatternResult, ...] = field(default=())

    @property
    def unsound(self) -> tuple[QuantifiedPatternResult, ...]:
        return tuple(result for result in self.results if not result.sound)

    @property
    def all_sound(self) -> bool:
        return all(result.sound for result in self.results)

    @property
    def precision_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for result in self.results if result.precise) / len(self.results)

    @property
    def mean_speedup(self) -> float:
        finite = [r.speedup for r in self.results if r.speedup != float("inf")]
        return sum(finite) / len(finite) if finite else float("inf")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "all_sound": self.all_sound,
            "precision_rate": round(self.precision_rate, 4),
            "unsound_count": len(self.unsound),
            "results": [result.to_dict() for result in self.results],
        }


def benchmark_quantified_patterns(
    patterns: Sequence[QuantifiedPattern],
    strategy: SamplingStrategy,
) -> QuantifiedPatternBenchmark:
    """Compare exact expansion against a sampling approximation for each pattern."""

    results: list[QuantifiedPatternResult] = []
    for pattern in patterns:
        all_indices = pattern.indices()
        sampled = strategy(pattern)
        if not sampled:
            raise ValueError(f"strategy returned an empty sample for {pattern.name!r}")
        exact_verdict, exact_seconds = _verdict(pattern.counterexample_problem(all_indices))
        approx_verdict, approx_seconds = _verdict(pattern.counterexample_problem(sampled))
        # Unsound iff exact found a violation that the approximation missed.
        sound = not (
            exact_verdict is QuantifiedVerdict.VIOLATED
            and approx_verdict is QuantifiedVerdict.HOLDS
        )
        precise = exact_verdict == approx_verdict
        results.append(
            QuantifiedPatternResult(
                pattern=pattern.name,
                exact_verdict=exact_verdict.value,
                approx_verdict=approx_verdict.value,
                exact_indices=len(all_indices),
                approx_indices=len(sampled),
                exact_seconds=exact_seconds,
                approx_seconds=approx_seconds,
                sound=sound,
                precise=precise,
            )
        )
    return QuantifiedPatternBenchmark(
        version=QUANTIFIED_PATTERN_BENCHMARK_VERSION,
        results=tuple(results),
    )


def render_quantified_pattern_json(benchmark: QuantifiedPatternBenchmark) -> str:
    return json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n"


def render_quantified_pattern_text(benchmark: QuantifiedPatternBenchmark) -> str:
    lines = [
        f"PromptABI quantified-pattern benchmark ({benchmark.version})",
        f"all sound: {benchmark.all_sound}",
        f"precision rate: {benchmark.precision_rate:.2%}",
        f"unsound approximations: {len(benchmark.unsound)}",
    ]
    for result in benchmark.results:
        flag = "sound" if result.sound else "UNSOUND"
        lines.append(
            f"  {result.pattern}: exact={result.exact_verdict} approx={result.approx_verdict}"
            f" ({result.approx_indices}/{result.exact_indices} idx) [{flag}]"
        )
    return "\n".join(lines) + "\n"
