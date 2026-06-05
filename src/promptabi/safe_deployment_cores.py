"""Minimal unsat-core certificates for deployment-safe PromptABI obligations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .formal import FiniteContractProblem, NamedConstraint, SolverBackend, SolverBudgetOutcome, SolverStatus
from .smt_benchmarks import SmtBenchmarkCase, SmtBenchmarkError, load_smt_benchmark_suite


SAFE_DEPLOYMENT_CORE_MANIFEST_VERSION = "promptabi.safe-deployment-cores.v1"


class SafeDeploymentCoreError(ValueError):
    """Raised when minimal deployment cores cannot be derived soundly."""


@dataclass(frozen=True, slots=True)
class MinimalUnsatCoreCertificate:
    """A replay-derived irreducible unsat core for one safe deployment obligation."""

    case_id: str
    display_name: str
    failure_class: str
    source: str
    backend: SolverBackend
    core: tuple[str, ...]
    removed_constraints: tuple[str, ...]
    checked_assignments: int
    solver_budget_outcome: SolverBudgetOutcome
    query_key: str
    proof_hash: str

    @property
    def minimal(self) -> bool:
        return bool(self.core)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "display_name": self.display_name,
            "failure_class": self.failure_class,
            "source": self.source,
            "backend": self.backend.value,
            "core": list(self.core),
            "removed_constraints": list(self.removed_constraints),
            "checked_assignments": self.checked_assignments,
            "solver_budget_outcome": self.solver_budget_outcome.value,
            "minimal": self.minimal,
            "query_key": self.query_key,
            "proof_hash": self.proof_hash,
        }


@dataclass(frozen=True, slots=True)
class SafeDeploymentCoreReport:
    """Deterministic proof bundle for safe deployment obligations."""

    source: str
    certificates: tuple[MinimalUnsatCoreCertificate, ...]
    manifest_sha256: str

    @property
    def ok(self) -> bool:
        return bool(self.certificates) and all(certificate.minimal for certificate in self.certificates)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": SAFE_DEPLOYMENT_CORE_MANIFEST_VERSION,
            "promptabi_version": __version__,
            "source": self.source,
            "ok": self.ok,
            "certificate_count": len(self.certificates),
            "certificates": [certificate.to_dict() for certificate in self.certificates],
            "manifest_sha256": self.manifest_sha256,
        }


def derive_safe_deployment_cores(path: str | Path | None = None) -> SafeDeploymentCoreReport:
    """Derive minimal unsat cores from replayed safe-deployment SMT obligations."""

    suite = load_smt_benchmark_suite(path)
    certificates = tuple(
        _derive_case_certificate(case)
        for case in suite.cases
        if case.category == "unsatisfiable" or "unsat-core" in case.labels
    )
    if not certificates:
        raise SafeDeploymentCoreError("SMT benchmark contains no unsatisfiable deployment obligations")
    payload = {
        "manifest_version": SAFE_DEPLOYMENT_CORE_MANIFEST_VERSION,
        "promptabi_version": __version__,
        "source": str(suite.path),
        "ok": all(certificate.minimal for certificate in certificates),
        "certificate_count": len(certificates),
        "certificates": [certificate.to_dict() for certificate in certificates],
    }
    return SafeDeploymentCoreReport(
        source=str(suite.path),
        certificates=certificates,
        manifest_sha256=_stable_json_hash(payload),
    )


def render_safe_deployment_cores_json(report: SafeDeploymentCoreReport) -> str:
    """Render minimal unsat-core certificates as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_safe_deployment_cores_text(report: SafeDeploymentCoreReport) -> str:
    """Render a compact release-engineering summary."""

    lines = [
        "PromptABI safe deployment unsat cores",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"source: {report.source}",
        f"certificates: {len(report.certificates)}",
    ]
    for certificate in report.certificates:
        lines.append(f"- {certificate.case_id}: {', '.join(certificate.core)}")
        if certificate.removed_constraints:
            lines.append(f"  removed: {', '.join(certificate.removed_constraints)}")
        lines.append(f"  proof_hash: {certificate.proof_hash}")
    return "\n".join(lines) + "\n"


def _derive_case_certificate(case: SmtBenchmarkCase) -> MinimalUnsatCoreCertificate:
    replay = case.replay_file()
    replay_report = replay.replay()
    if not replay_report.ok or replay_report.actual.status is not SolverStatus.UNSAT:
        raise SafeDeploymentCoreError(f"case {case.case_id!r} did not replay as unsat")

    candidate_names = replay_report.actual.unsat_core or tuple(constraint.name for constraint in case.problem.constraints)
    candidate_constraints = tuple(
        constraint
        for constraint in case.problem.constraints
        if constraint.name in set(candidate_names)
    )
    if not candidate_constraints:
        raise SafeDeploymentCoreError(f"case {case.case_id!r} produced an empty unsat core")
    minimized = _minimize_constraints(
        case.problem,
        candidate_constraints,
        prefer_z3=bool(case.options.get("prefer_z3", True)),
    )
    minimized_problem = _problem_with_constraints(case.problem, minimized)
    minimized_result = minimized_problem.solve(prefer_z3=bool(case.options.get("prefer_z3", True)))
    if minimized_result.status is not SolverStatus.UNSAT:
        raise SafeDeploymentCoreError(f"case {case.case_id!r} minimized core no longer proves unsat")
    removed = tuple(
        constraint.name
        for constraint in case.problem.constraints
        if constraint.name not in {item.name for item in minimized}
    )
    proof_payload = {
        "case_id": case.case_id,
        "problem": minimized_problem.to_dict(),
        "solver": minimized_result.to_dict(),
        "artifact_hashes": dict(sorted(case.artifact_hashes.items())),
        "supported_fragment_metadata": case.supported_fragment_metadata,
    }
    return MinimalUnsatCoreCertificate(
        case_id=case.case_id,
        display_name=case.display_name,
        failure_class=case.failure_class,
        source=case.source,
        backend=minimized_result.backend,
        core=tuple(constraint.name for constraint in minimized),
        removed_constraints=removed,
        checked_assignments=minimized_result.checked_assignments,
        solver_budget_outcome=minimized_result.budget_outcome,
        query_key=minimized_problem.solver_query_key(prefer_z3=bool(case.options.get("prefer_z3", True))),
        proof_hash=_stable_json_hash(proof_payload),
    )


def _minimize_constraints(
    problem: FiniteContractProblem,
    constraints: tuple[NamedConstraint, ...],
    *,
    prefer_z3: bool,
) -> tuple[NamedConstraint, ...]:
    remaining = list(constraints)
    changed = True
    while changed and len(remaining) > 1:
        changed = False
        for constraint in tuple(remaining):
            candidate = tuple(item for item in remaining if item is not constraint)
            result = _problem_with_constraints(problem, candidate).solve(prefer_z3=prefer_z3)
            if result.status is SolverStatus.UNSAT:
                remaining = list(candidate)
                changed = True
                break
    for constraint in tuple(remaining):
        candidate = tuple(item for item in remaining if item is not constraint)
        if candidate and _problem_with_constraints(problem, candidate).solve(prefer_z3=prefer_z3).status is SolverStatus.UNSAT:
            raise SafeDeploymentCoreError(f"constraint {constraint.name!r} is removable from purported minimal core")
    return tuple(remaining)


def _problem_with_constraints(
    problem: FiniteContractProblem,
    constraints: tuple[NamedConstraint, ...],
) -> FiniteContractProblem:
    return FiniteContractProblem(
        name=f"{problem.name}-minimal-unsat-core",
        variables=problem.variables,
        constraints=constraints,
    )


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
