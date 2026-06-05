"""Cache solver lemmas by normalized artifact products (step 224).

Verification re-solves the *same* finite-contract lemmas over and over: across
incremental runs, across artifacts that differ only in key order or whitespace,
and across contracts whose constraints are emitted in a different order.  The
finite/SMT solver already supports a deterministic query cache
(:class:`promptabi.formal.SolverQueryCache`); this module raises that to the
level of **artifact products** and proves the two properties a lemma cache must
have to be trustworthy:

* **Soundness** -- a cached verdict is identical to a fresh, cache-bypassing
  solve.  Caching must never change ``proven``/``refuted``/``abstained``.
* **Reuse (non-vacuous)** -- two lemmas whose *normalized artifact product*
  (the order-independent, formatting-independent content hash of the source
  artifacts) and whose normalized solver query coincide share a single cache
  entry, so the cache actually saves work.

A :class:`SolverLemma` couples a :class:`~promptabi.formal.FiniteContractProblem`
with the artifacts it was derived from.  :func:`verify_solver_lemma_cache`
solves every lemma twice (cached and fresh), checks soundness, and reports how
many solves were served from the cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence

from .formal import FiniteContractProblem, SolverQueryCache, SolverResult
from .token_budget_arithmetic import TokenBudgetContract, compile_token_budget_problem

SOLVER_LEMMA_CACHE_VERSION = "promptabi.solver-lemma-cache.v1"


class SolverLemmaCacheFindingKind(StrEnum):
    UNSOUND_CACHE_HIT = "unsound-cache-hit"
    NO_REUSE = "no-reuse"


def canonical_artifact_hash(artifact: Mapping[str, object]) -> str:
    """Content hash of an artifact, invariant to key order and whitespace."""

    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_artifact_product(artifacts: Sequence[Mapping[str, object]]) -> str:
    """Order-independent content hash of a product of artifacts."""

    member_hashes = sorted(canonical_artifact_hash(artifact) for artifact in artifacts)
    digest = hashlib.sha256("|".join(member_hashes).encode("utf-8")).hexdigest()
    return digest


@dataclass(frozen=True, slots=True)
class SolverLemma:
    """A solver lemma plus the artifact product it was derived from."""

    name: str
    problem: FiniteContractProblem
    product: tuple[Mapping[str, object], ...]

    def product_key(self) -> str:
        return normalize_artifact_product(self.product)

    def artifact_hashes(self) -> dict[str, str]:
        return {"product": self.product_key()}


@dataclass(frozen=True, slots=True)
class SolverLemmaCacheFinding:
    kind: SolverLemmaCacheFindingKind
    lemma: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "lemma": self.lemma, "message": self.message}


@dataclass(frozen=True, slots=True)
class SolverLemmaRecord:
    lemma: str
    product_key: str
    cache_key: str
    status: str
    cache_hit: bool
    sound: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "lemma": self.lemma,
            "product_key": self.product_key[:16],
            "cache_key": self.cache_key[:16],
            "status": self.status,
            "cache_hit": self.cache_hit,
            "sound": self.sound,
        }


@dataclass(frozen=True, slots=True)
class SolverLemmaCacheReport:
    version: str
    records: tuple[SolverLemmaRecord, ...] = field(default=())
    findings: tuple[SolverLemmaCacheFinding, ...] = field(default=())
    hits: int = 0
    misses: int = 0

    @property
    def soundness_ok(self) -> bool:
        return all(record.sound for record in self.records)

    @property
    def reuse_demonstrated(self) -> bool:
        return self.hits > 0

    @property
    def ok(self) -> bool:
        return self.soundness_ok and not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ok": self.ok,
            "soundness_ok": self.soundness_ok,
            "reuse_demonstrated": self.reuse_demonstrated,
            "hits": self.hits,
            "misses": self.misses,
            "records": [record.to_dict() for record in self.records],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def verify_solver_lemma_cache(
    lemmas: Sequence[SolverLemma],
    *,
    prefer_z3: bool = True,
    require_reuse: bool = True,
) -> SolverLemmaCacheReport:
    """Prove the lemma cache is sound and (optionally) demonstrably reused."""

    cache = SolverQueryCache()
    records: list[SolverLemmaRecord] = []
    findings: list[SolverLemmaCacheFinding] = []

    for lemma in lemmas:
        fresh = lemma.problem.solve(prefer_z3=prefer_z3, artifact_hashes=lemma.artifact_hashes())
        cached = cache.solve(lemma.problem, prefer_z3=prefer_z3, artifact_hashes=lemma.artifact_hashes())
        sound = _verdicts_match(fresh, cached)
        if not sound:
            findings.append(
                SolverLemmaCacheFinding(
                    kind=SolverLemmaCacheFindingKind.UNSOUND_CACHE_HIT,
                    lemma=lemma.name,
                    message=(
                        f"cached verdict {cached.status.value!r} differs from fresh "
                        f"verdict {fresh.status.value!r}"
                    ),
                )
            )
        records.append(
            SolverLemmaRecord(
                lemma=lemma.name,
                product_key=lemma.product_key(),
                cache_key=cached.cache_key or "",
                status=cached.status.value,
                cache_hit=cached.cache_hit,
                sound=sound,
            )
        )

    if require_reuse and cache.hits == 0 and lemmas:
        findings.append(
            SolverLemmaCacheFinding(
                kind=SolverLemmaCacheFindingKind.NO_REUSE,
                lemma="<suite>",
                message="no lemma was served from the cache; normalization did not enable reuse",
            )
        )

    return SolverLemmaCacheReport(
        version=SOLVER_LEMMA_CACHE_VERSION,
        records=tuple(records),
        findings=tuple(findings),
        hits=cache.hits,
        misses=cache.misses,
    )


def _verdicts_match(left: SolverResult, right: SolverResult) -> bool:
    return (
        left.status is right.status
        and left.conclusion is right.conclusion
        and left.budget_outcome is right.budget_outcome
    )


def _reordered_contract(contract: TokenBudgetContract) -> TokenBudgetContract:
    """A semantically-identical contract with assumptions in reverse order."""

    from dataclasses import replace

    return replace(contract, assumptions=tuple(reversed(contract.assumptions)))


def token_budget_lemma_suite(contract: TokenBudgetContract) -> tuple[SolverLemma, ...]:
    """Build a lemma suite that exercises normalization-keyed reuse.

    For every guarantee the suite emits the lemma twice: once from ``contract``
    and once from a constraint-reordered, re-serialized equivalent.  Both map to
    the same normalized artifact product and solver query, so the second solve
    must be a cache hit.
    """

    reordered = _reordered_contract(contract)
    # canonical product dict reused for both passes (formatting-independent)
    product = (contract.to_dict(),)
    reformatted_product = (json.loads(json.dumps(contract.to_dict(), sort_keys=True)),)

    lemmas: list[SolverLemma] = []
    for guarantee in contract.guarantees:
        lemmas.append(
            SolverLemma(
                name=f"{contract.name}:{guarantee.name}",
                problem=compile_token_budget_problem(contract, guarantee),
                product=product,
            )
        )
    for guarantee in reordered.guarantees:
        lemmas.append(
            SolverLemma(
                name=f"{contract.name}:{guarantee.name}:reordered",
                problem=compile_token_budget_problem(reordered, guarantee),
                product=reformatted_product,
            )
        )
    return tuple(lemmas)


def render_solver_lemma_cache_json(report: SolverLemmaCacheReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_solver_lemma_cache_text(report: SolverLemmaCacheReport) -> str:
    lines = [
        f"PromptABI solver-lemma cache ({report.version})",
        f"status: {'OK' if report.ok else 'VIOLATED'}",
        f"soundness: {'ok' if report.soundness_ok else 'broken'}",
        f"cache: {report.hits} hit(s), {report.misses} miss(es)",
        f"lemmas: {len(report.records)}",
    ]
    for record in report.records:
        marker = "hit" if record.cache_hit else "miss"
        sound = "sound" if record.sound else "UNSOUND"
        lines.append(f"  {record.lemma}: {record.status} [{marker}, {sound}]")
    for finding in report.findings:
        lines.append(f"  ! {finding.kind.value}: {finding.message}")
    return "\n".join(lines) + "\n"
