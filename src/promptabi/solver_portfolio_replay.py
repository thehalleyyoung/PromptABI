"""Add solver portfolio replay metadata (step 225).

A *solver portfolio* runs the same finite-contract lemma through several solver
strategies -- for PromptABI that is at minimum the Z3-backed SMT path and the
deterministic finite-enumeration fallback -- and records, for every strategy,
which backend actually answered, the verdict it produced, and how much of the
finite budget it explored.  The portfolio then emits **replay metadata**: the
normalized solver-query fingerprint, the solver-version fingerprints, and the
strategy that "won" (the first conclusive answer).  A maintainer can ship that
metadata in a bug report and re-run :func:`replay_solver_portfolio` to reproduce
the exact verdict without shipping prompts, datasets, or credentials.

The module proves three properties a trustworthy portfolio must have:

* **Cross-strategy agreement** -- every strategy that returns a conclusive
  verdict (``sat``/``unsat``) must agree on the conclusion.  A disagreement means
  one backend is unsound for this fragment and is reported as a finding.
* **Determinism** -- replaying a recorded portfolio reproduces the recorded
  winning verdict.  Otherwise the replay metadata is worthless.
* **Conclusiveness (optional)** -- at least one strategy answered conclusively,
  so the portfolio is not vacuously "agreeing" by all abstaining.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .formal import (
    FiniteContractProblem,
    SolverBackend,
    SolverBudgetOutcome,
    SolverConclusion,
    SolverResult,
    SolverStatus,
)
from .token_budget_arithmetic import TokenBudgetContract, compile_token_budget_problem

SOLVER_PORTFOLIO_REPLAY_VERSION = "promptabi.solver-portfolio-replay.v1"


class SolverPortfolioFindingKind(StrEnum):
    PORTFOLIO_DISAGREEMENT = "portfolio-disagreement"
    NONDETERMINISTIC_REPLAY = "nondeterministic-replay"
    NO_CONCLUSIVE_STRATEGY = "no-conclusive-strategy"


@dataclass(frozen=True, slots=True)
class SolverStrategy:
    """One named solver configuration in a portfolio."""

    name: str
    prefer_z3: bool = True
    max_assignments: int | None = None
    timeout_seconds: float | None = None

    def solve(self, problem: FiniteContractProblem) -> SolverResult:
        return problem.solve(
            prefer_z3=self.prefer_z3,
            max_assignments=self.max_assignments,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prefer_z3": self.prefer_z3,
            "max_assignments": self.max_assignments,
            "timeout_seconds": self.timeout_seconds,
        }


def default_portfolio() -> tuple[SolverStrategy, ...]:
    """The default PromptABI portfolio: SMT first, bounded finite enumeration second.

    The finite-enumeration fallback carries a finite assignment budget so that it
    stays fast on contracts with wide integer domains: it returns a *bounded*
    (inconclusive) verdict instead of exhaustively enumerating millions of
    assignments, leaving the Z3 path to deliver the conclusive answer.
    """

    return (
        SolverStrategy(name="z3-smt", prefer_z3=True),
        SolverStrategy(name="finite-enumeration", prefer_z3=False, max_assignments=20000),
    )


@dataclass(frozen=True, slots=True)
class PortfolioAttempt:
    strategy: str
    backend: str
    status: str
    conclusion: str
    budget_outcome: str
    checked_assignments: int
    elapsed_microseconds: int
    conclusive: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "backend": self.backend,
            "status": self.status,
            "conclusion": self.conclusion,
            "budget_outcome": self.budget_outcome,
            "checked_assignments": self.checked_assignments,
            "elapsed_microseconds": self.elapsed_microseconds,
            "conclusive": self.conclusive,
        }


@dataclass(frozen=True, slots=True)
class SolverPortfolioFinding:
    kind: SolverPortfolioFindingKind
    lemma: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "lemma": self.lemma, "message": self.message}


@dataclass(frozen=True, slots=True)
class SolverPortfolioReplayMetadata:
    """Self-contained metadata to deterministically replay a portfolio result."""

    lemma: str
    query_key: str
    solver_version_fingerprints: tuple[tuple[str, str], ...]
    winning_strategy: str | None
    winning_status: str
    winning_conclusion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lemma": self.lemma,
            "query_key": self.query_key,
            "solver_version_fingerprints": [list(item) for item in self.solver_version_fingerprints],
            "winning_strategy": self.winning_strategy,
            "winning_status": self.winning_status,
            "winning_conclusion": self.winning_conclusion,
        }


@dataclass(frozen=True, slots=True)
class SolverPortfolioRecord:
    lemma: str
    attempts: tuple[PortfolioAttempt, ...]
    metadata: SolverPortfolioReplayMetadata
    agreement_ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "lemma": self.lemma,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "metadata": self.metadata.to_dict(),
            "agreement_ok": self.agreement_ok,
        }


@dataclass(frozen=True, slots=True)
class SolverPortfolioReport:
    version: str
    records: tuple[SolverPortfolioRecord, ...] = field(default=())
    findings: tuple[SolverPortfolioFinding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ok": self.ok,
            "records": [record.to_dict() for record in self.records],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _winning_result(results: Sequence[tuple[SolverStrategy, SolverResult]]) -> tuple[SolverStrategy, SolverResult] | None:
    for strategy, result in results:
        if result.status is not SolverStatus.UNKNOWN:
            return strategy, result
    return None


def run_solver_portfolio(
    problem: FiniteContractProblem,
    *,
    portfolio: Sequence[SolverStrategy] | None = None,
    require_conclusive: bool = True,
) -> SolverPortfolioRecord:
    """Run one lemma through a portfolio and build its replay record."""

    strategies = tuple(portfolio) if portfolio is not None else default_portfolio()
    if not strategies:
        raise ValueError("a solver portfolio must contain at least one strategy")

    raw: list[tuple[SolverStrategy, SolverResult]] = []
    attempts: list[PortfolioAttempt] = []
    for strategy in strategies:
        start = time.perf_counter_ns()
        result = strategy.solve(problem)
        elapsed = max(0, (time.perf_counter_ns() - start) // 1000)
        raw.append((strategy, result))
        attempts.append(
            PortfolioAttempt(
                strategy=strategy.name,
                backend=result.backend.value,
                status=result.status.value,
                conclusion=result.conclusion.value,
                budget_outcome=result.budget_outcome.value,
                checked_assignments=result.checked_assignments,
                elapsed_microseconds=elapsed,
                conclusive=result.status is not SolverStatus.UNKNOWN,
            )
        )

    winner = _winning_result(raw)
    winning_strategy = winner[0].name if winner is not None else None
    winning_result = winner[1] if winner is not None else None

    conclusive = [(s, r) for s, r in raw if r.status is not SolverStatus.UNKNOWN]
    conclusions = {r.conclusion for _, r in conclusive}
    agreement_ok = len(conclusions) <= 1

    # Build replay metadata from the deterministic query fingerprint of the
    # winning strategy (or the first strategy if nothing was conclusive).
    fingerprint_strategy = winner[0] if winner is not None else strategies[0]
    query = problem.normalized_solver_query(
        prefer_z3=fingerprint_strategy.prefer_z3,
        max_assignments=fingerprint_strategy.max_assignments,
        timeout_seconds=fingerprint_strategy.timeout_seconds,
    )
    query_key = problem.solver_query_key(
        prefer_z3=fingerprint_strategy.prefer_z3,
        max_assignments=fingerprint_strategy.max_assignments,
        timeout_seconds=fingerprint_strategy.timeout_seconds,
    )
    metadata = SolverPortfolioReplayMetadata(
        lemma=problem.name,
        query_key=query_key,
        solver_version_fingerprints=tuple(
            (str(name), str(version)) for name, version in query["solver_version_fingerprints"]
        ),
        winning_strategy=winning_strategy,
        winning_status=(winning_result.status.value if winning_result is not None else SolverStatus.UNKNOWN.value),
        winning_conclusion=(
            winning_result.conclusion.value if winning_result is not None else SolverConclusion.ABSTENTION.value
        ),
    )

    return SolverPortfolioRecord(
        lemma=problem.name,
        attempts=tuple(attempts),
        metadata=metadata,
        agreement_ok=agreement_ok,
    )


def replay_solver_portfolio(
    problem: FiniteContractProblem,
    metadata: SolverPortfolioReplayMetadata,
    *,
    portfolio: Sequence[SolverStrategy] | None = None,
) -> bool:
    """Replay a portfolio and confirm it reproduces the recorded verdict."""

    record = run_solver_portfolio(problem, portfolio=portfolio, require_conclusive=False)
    return (
        record.metadata.winning_status == metadata.winning_status
        and record.metadata.winning_conclusion == metadata.winning_conclusion
        and record.metadata.query_key == metadata.query_key
    )


def verify_solver_portfolio(
    problems: Sequence[tuple[str, FiniteContractProblem]],
    *,
    portfolio: Sequence[SolverStrategy] | None = None,
    require_conclusive: bool = True,
) -> SolverPortfolioReport:
    """Run a portfolio over many lemmas and prove agreement + replay determinism."""

    records: list[SolverPortfolioRecord] = []
    findings: list[SolverPortfolioFinding] = []
    for name, problem in problems:
        record = run_solver_portfolio(problem, portfolio=portfolio, require_conclusive=require_conclusive)
        records.append(record)
        if not record.agreement_ok:
            findings.append(
                SolverPortfolioFinding(
                    kind=SolverPortfolioFindingKind.PORTFOLIO_DISAGREEMENT,
                    lemma=name,
                    message="solver strategies returned conflicting conclusive verdicts",
                )
            )
        if require_conclusive and record.metadata.winning_strategy is None:
            findings.append(
                SolverPortfolioFinding(
                    kind=SolverPortfolioFindingKind.NO_CONCLUSIVE_STRATEGY,
                    lemma=name,
                    message="no strategy in the portfolio produced a conclusive verdict",
                )
            )
        if not replay_solver_portfolio(problem, record.metadata, portfolio=portfolio):
            findings.append(
                SolverPortfolioFinding(
                    kind=SolverPortfolioFindingKind.NONDETERMINISTIC_REPLAY,
                    lemma=name,
                    message="replaying the portfolio did not reproduce the recorded verdict",
                )
            )
    return SolverPortfolioReport(
        version=SOLVER_PORTFOLIO_REPLAY_VERSION,
        records=tuple(records),
        findings=tuple(findings),
    )


def token_budget_portfolio_problems(
    contract: TokenBudgetContract,
) -> tuple[tuple[str, FiniteContractProblem], ...]:
    """Compile every guarantee of a token-budget contract into a named lemma."""

    problems: list[tuple[str, FiniteContractProblem]] = []
    for guarantee in contract.guarantees:
        problems.append(
            (f"{contract.name}:{guarantee.name}", compile_token_budget_problem(contract, guarantee))
        )
    return tuple(problems)


def render_solver_portfolio_json(report: SolverPortfolioReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_solver_portfolio_text(report: SolverPortfolioReport) -> str:
    lines = [
        f"PromptABI solver portfolio replay ({report.version})",
        f"status: {'OK' if report.ok else 'VIOLATED'}",
        f"lemmas: {len(report.records)}",
    ]
    for record in report.records:
        winner = record.metadata.winning_strategy or "<none>"
        lines.append(
            f"  {record.lemma}: {record.metadata.winning_status} via {winner}"
            f" [{'agree' if record.agreement_ok else 'DISAGREE'}]"
        )
        for attempt in record.attempts:
            lines.append(
                f"      {attempt.strategy}: {attempt.status} ({attempt.backend},"
                f" {attempt.checked_assignments} checked)"
            )
    for finding in report.findings:
        lines.append(f"  ! {finding.kind.value} [{finding.lemma}]: {finding.message}")
    return "\n".join(lines) + "\n"
