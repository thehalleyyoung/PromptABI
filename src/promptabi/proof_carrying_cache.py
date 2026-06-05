"""Add proof-carrying solver cache entries (step 232).

A plain solver cache asks consumers to *trust* that a stored verdict is correct.
A **proof-carrying** cache stores, alongside each verdict, a checkable proof
object and re-validates it on every hit -- so a corrupted, stale, or tampered
entry is caught instead of silently trusted:

* a ``SAT`` verdict carries its **model**; validation re-evaluates every
  constraint against the model with no solver call at all (linear, deterministic);
* an ``UNSAT`` verdict carries its **unsat core**; validation rebuilds the
  core-only sub-problem and confirms it is still unsatisfiable -- a strictly
  smaller, cheaper obligation than the original solve.

Any entry whose proof fails validation is rejected as ``tampered`` and the cache
falls back to a fresh solve, guaranteeing that a proof-carrying hit is never
weaker than solving from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from .formal import (
    FiniteContractProblem,
    SolverResult,
    SolverStatus,
)

PROOF_CARRYING_CACHE_VERSION = "promptabi.proof-carrying-cache.v1"


class ProofKind(StrEnum):
    MODEL = "model"
    UNSAT_CORE = "unsat-core"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ProofCarryingEntry:
    cache_key: str
    status: str
    proof_kind: ProofKind
    model: Mapping[str, object] | None = None
    unsat_core: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "cache_key": self.cache_key,
            "status": self.status,
            "proof_kind": self.proof_kind.value,
        }
        if self.model is not None:
            data["model"] = dict(sorted(self.model.items()))
        if self.unsat_core:
            data["unsat_core"] = list(self.unsat_core)
        return data


def _model_satisfies(problem: FiniteContractProblem, model: Mapping[str, object]) -> bool:
    declared = {variable.name for variable in problem.variables}
    if set(model) != declared:
        return False
    try:
        return all(bool(constraint.expression.evaluate(model)) for constraint in problem.constraints)
    except (TypeError, ValueError, AttributeError):
        return False


def _core_is_unsat(problem: FiniteContractProblem, core: tuple[str, ...], *, prefer_z3: bool) -> bool:
    by_name = {constraint.name: constraint for constraint in problem.constraints}
    selected = tuple(by_name[name] for name in core if name in by_name)
    if not selected:
        return False
    sub_problem = FiniteContractProblem(
        variables=tuple(problem.variables),
        constraints=selected,
        name=f"{problem.name}:core",
    )
    return sub_problem.solve(prefer_z3=prefer_z3).status is SolverStatus.UNSAT


def build_proof(result: SolverResult, cache_key: str) -> ProofCarryingEntry:
    """Extract a checkable proof object from a solver result."""

    if result.status is SolverStatus.SAT and result.assignment is not None:
        return ProofCarryingEntry(
            cache_key=cache_key,
            status=result.status.value,
            proof_kind=ProofKind.MODEL,
            model=dict(result.assignment),
        )
    if result.status is SolverStatus.UNSAT:
        return ProofCarryingEntry(
            cache_key=cache_key,
            status=result.status.value,
            proof_kind=ProofKind.UNSAT_CORE,
            unsat_core=tuple(result.unsat_core),
        )
    return ProofCarryingEntry(
        cache_key=cache_key,
        status=result.status.value,
        proof_kind=ProofKind.NONE,
    )


def validate_proof(
    problem: FiniteContractProblem,
    entry: ProofCarryingEntry,
    *,
    prefer_z3: bool = True,
) -> bool:
    """Independently re-check a proof-carrying entry against the problem."""

    if entry.proof_kind is ProofKind.MODEL:
        return entry.model is not None and _model_satisfies(problem, entry.model)
    if entry.proof_kind is ProofKind.UNSAT_CORE:
        return _core_is_unsat(problem, entry.unsat_core, prefer_z3=prefer_z3)
    # An abstention proof carries no obligation to re-check.
    return entry.status == SolverStatus.UNKNOWN.value


@dataclass(slots=True)
class ProofCarryingCache:
    """A solver cache whose hits are independently re-validated against proofs."""

    prefer_z3: bool = True
    entries: dict[str, ProofCarryingEntry] = field(default_factory=dict)
    results: dict[str, SolverResult] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    rejected: int = 0

    def solve(self, problem: FiniteContractProblem) -> tuple[SolverResult, bool]:
        """Return ``(result, validated_hit)``; rejected entries trigger a fresh solve."""

        key = problem.solver_query_key(prefer_z3=self.prefer_z3)
        entry = self.entries.get(key)
        if entry is not None and validate_proof(problem, entry, prefer_z3=self.prefer_z3):
            self.hits += 1
            return self.results[key].with_cache_metadata(cache_key=key, cache_hit=True), True
        if entry is not None:
            # Present but failed validation -> tampered/stale; discard.
            self.rejected += 1
            del self.entries[key]
            self.results.pop(key, None)
        self.misses += 1
        result = problem.solve(prefer_z3=self.prefer_z3)
        stored = result.with_cache_metadata(cache_key=key, cache_hit=False)
        self.entries[key] = build_proof(stored, key)
        self.results[key] = stored
        return stored, False

    def inject(self, problem: FiniteContractProblem, entry: ProofCarryingEntry, result: SolverResult) -> None:
        """Insert an entry directly (used to model corruption/tampering in tests)."""

        key = problem.solver_query_key(prefer_z3=self.prefer_z3)
        self.entries[key] = entry
        self.results[key] = result

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PROOF_CARRYING_CACHE_VERSION,
            "hits": self.hits,
            "misses": self.misses,
            "rejected": self.rejected,
            "entries": {key: entry.to_dict() for key, entry in sorted(self.entries.items())},
        }


def render_proof_carrying_cache_json(cache: ProofCarryingCache) -> str:
    return json.dumps(cache.to_dict(), indent=2, sort_keys=True) + "\n"
