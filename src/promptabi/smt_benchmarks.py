"""Minimized SMT benchmark corpus for prompt-interface contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formal import FiniteContractProblem, SolverConclusion, SolverReplayFile, SolverStatus


DEFAULT_SMT_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "smt_benchmarks" / "benchmark.json"
SMT_BENCHMARK_MANIFEST_VERSION = 1
REQUIRED_SMT_BENCHMARK_CATEGORIES = frozenset(
    {
        "satisfiable",
        "unsatisfiable",
        "timeout-prone",
        "unsupported",
    }
)


class SmtBenchmarkError(ValueError):
    """Raised when the minimized SMT benchmark corpus is malformed or fails replay."""


@dataclass(frozen=True, slots=True)
class SmtBenchmarkExpectation:
    """Expected replay outcome for one benchmark case."""

    status: SolverStatus
    conclusion: SolverConclusion
    witness_variables: tuple[str, ...] = ()
    unsat_core: tuple[str, ...] = ()
    minimum_checked_assignments: int | None = None
    reason_contains: str | None = None


@dataclass(frozen=True, slots=True)
class SmtBenchmarkCase:
    """One minimized SMT obligation reduced from a prompt-interface failure."""

    case_id: str
    category: str
    display_name: str
    source: str
    failure_class: str
    labels: tuple[str, ...]
    problem: FiniteContractProblem
    options: dict[str, object]
    supported_fragment_metadata: dict[str, object]
    artifact_hashes: dict[str, str]
    expected: SmtBenchmarkExpectation

    def replay_file(self) -> SolverReplayFile:
        return SolverReplayFile.from_problem(
            self.problem,
            replay_id=self.case_id,
            prefer_z3=bool(self.options.get("prefer_z3", True)),
            max_assignments=_optional_int(self.options.get("max_assignments")),
            timeout_seconds=_optional_float(self.options.get("timeout_seconds")),
            artifact_hashes=self.artifact_hashes,
            supported_fragment_metadata=self.supported_fragment_metadata,
        )


@dataclass(frozen=True, slots=True)
class SmtBenchmarkResult:
    """Replay result for one minimized SMT benchmark case."""

    case: SmtBenchmarkCase
    status: SolverStatus
    conclusion: SolverConclusion
    passed: bool
    query_key: str
    checked_assignments: int
    backend: str
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case.case_id,
            "category": self.case.category,
            "display_name": self.case.display_name,
            "source": self.case.source,
            "failure_class": self.case.failure_class,
            "labels": list(self.case.labels),
            "status": self.status.value,
            "conclusion": self.conclusion.value,
            "backend": self.backend,
            "checked_assignments": self.checked_assignments,
            "query_key": self.query_key,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class SmtBenchmarkSuite:
    """A deterministic collection of minimized SMT obligations."""

    path: Path
    methodology: str
    cases: tuple[SmtBenchmarkCase, ...]

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({case.category for case in self.cases}))

    def replay(self) -> tuple[SmtBenchmarkResult, ...]:
        return tuple(_replay_case(case) for case in self.cases)

    def manifest(self) -> dict[str, object]:
        results = self.replay()
        manifest: dict[str, object] = {
            "manifest_version": SMT_BENCHMARK_MANIFEST_VERSION,
            "methodology": self.methodology,
            "path": str(self.path),
            "case_count": len(results),
            "categories": list(self.categories),
            "required_categories": sorted(REQUIRED_SMT_BENCHMARK_CATEGORIES),
            "all_cases_passed": all(result.passed for result in results),
            "entries": [result.to_dict() for result in results],
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_smt_benchmark_suite(path: str | Path | None = None) -> SmtBenchmarkSuite:
    """Load and validate the minimized SMT benchmark corpus."""

    benchmark_path = Path(path) if path is not None else DEFAULT_SMT_BENCHMARK_PATH
    payload = _read_json_object(benchmark_path)
    if payload.get("manifest_version") != SMT_BENCHMARK_MANIFEST_VERSION:
        raise SmtBenchmarkError(f"{benchmark_path} has unsupported SMT benchmark manifest_version")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        raise SmtBenchmarkError(f"{benchmark_path} field 'methodology' must be a non-empty string")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SmtBenchmarkError(f"{benchmark_path} field 'cases' must be a non-empty list")
    cases = tuple(sorted((_case_from_mapping(benchmark_path, item) for item in raw_cases), key=lambda case: case.case_id))
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise SmtBenchmarkError(f"{benchmark_path} contains duplicate SMT benchmark case ids")
    categories = {case.category for case in cases}
    missing = REQUIRED_SMT_BENCHMARK_CATEGORIES.difference(categories)
    if missing:
        raise SmtBenchmarkError("SMT benchmark is missing required categories: " + ", ".join(sorted(missing)))
    return SmtBenchmarkSuite(path=benchmark_path, methodology=methodology, cases=cases)


def build_smt_benchmark_manifest(path: str | Path | None = None) -> dict[str, object]:
    """Validate, replay, and return a deterministic SMT benchmark manifest."""

    manifest = load_smt_benchmark_suite(path).manifest()
    if not manifest["all_cases_passed"]:
        failed = ", ".join(str(entry["id"]) for entry in manifest["entries"] if not entry["passed"])  # type: ignore[index]
        raise SmtBenchmarkError(f"SMT benchmark replay failed: {failed}")
    return manifest


def write_smt_benchmark_manifest(output: str | Path, *, path: str | Path | None = None) -> dict[str, object]:
    """Write the deterministic SMT benchmark manifest."""

    manifest = build_smt_benchmark_manifest(path)
    Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def render_smt_benchmark_text(manifest: dict[str, object]) -> str:
    """Render a concise terminal summary of the SMT benchmark manifest."""

    lines = [
        "PromptABI SMT benchmark corpus",
        f"status: {'PASS' if manifest.get('all_cases_passed') else 'FAIL'}",
        f"cases: {manifest.get('case_count', 0)}",
        "categories: " + ", ".join(str(category) for category in manifest.get("categories", ())),
    ]
    for entry in manifest.get("entries", ()):
        if not isinstance(entry, dict):
            continue
        lines.append(
            "- {id}: {status}/{conclusion} via {backend} ({checked} checked)".format(
                id=entry.get("id"),
                status=entry.get("status"),
                conclusion=entry.get("conclusion"),
                backend=entry.get("backend"),
                checked=entry.get("checked_assignments"),
            )
        )
        for failure in entry.get("failures", ()):
            lines.append(f"  failure: {failure}")
    return "\n".join(lines) + "\n"


def _case_from_mapping(path: Path, raw: object) -> SmtBenchmarkCase:
    if not isinstance(raw, dict):
        raise SmtBenchmarkError(f"{path} cases must be JSON objects")
    case_id = _required_string(raw, path, "id")
    category = _required_string(raw, path, "category")
    if category not in REQUIRED_SMT_BENCHMARK_CATEGORIES:
        raise SmtBenchmarkError(f"{path} case {case_id!r} has unsupported category {category!r}")
    expected = _expectation_from_mapping(path, case_id, _required_mapping(raw.get("expected"), "expected"))
    problem = FiniteContractProblem.from_dict(_required_mapping(raw.get("problem"), "problem"))
    labels = _string_tuple(raw.get("labels", ()), path, case_id, "labels")
    options = dict(_required_mapping(raw.get("options", {}), "options"))
    supported = dict(_required_mapping(raw.get("supported_fragment_metadata", {}), "supported_fragment_metadata"))
    hashes = _required_mapping(raw.get("artifact_hashes", {}), "artifact_hashes")
    return SmtBenchmarkCase(
        case_id=case_id,
        category=category,
        display_name=_required_string(raw, path, "display_name"),
        source=_required_string(raw, path, "source"),
        failure_class=_required_string(raw, path, "failure_class"),
        labels=labels,
        problem=problem,
        options=options,
        supported_fragment_metadata=supported,
        artifact_hashes={str(key): str(value) for key, value in hashes.items()},
        expected=expected,
    )


def _expectation_from_mapping(path: Path, case_id: str, raw: dict[str, object]) -> SmtBenchmarkExpectation:
    try:
        status = SolverStatus(str(raw["status"]))
        conclusion = SolverConclusion(str(raw["conclusion"]))
    except (KeyError, ValueError) as exc:
        raise SmtBenchmarkError(f"{path} case {case_id!r} has invalid expected solver status/conclusion") from exc
    return SmtBenchmarkExpectation(
        status=status,
        conclusion=conclusion,
        witness_variables=_string_tuple(raw.get("witness_variables", ()), path, case_id, "witness_variables"),
        unsat_core=_string_tuple(raw.get("unsat_core", ()), path, case_id, "unsat_core"),
        minimum_checked_assignments=_optional_int(raw.get("minimum_checked_assignments")),
        reason_contains=str(raw["reason_contains"]) if raw.get("reason_contains") is not None else None,
    )


def _replay_case(case: SmtBenchmarkCase) -> SmtBenchmarkResult:
    replay = case.replay_file()
    report = replay.replay()
    actual = report.actual
    failures: list[str] = []
    if actual.status is not case.expected.status:
        failures.append(f"expected status {case.expected.status.value}, got {actual.status.value}")
    if actual.conclusion is not case.expected.conclusion:
        failures.append(f"expected conclusion {case.expected.conclusion.value}, got {actual.conclusion.value}")
    if case.expected.witness_variables:
        assignment = actual.assignment or {}
        missing = sorted(set(case.expected.witness_variables).difference(assignment))
        if missing:
            failures.append("missing witness variable(s): " + ", ".join(missing))
    if case.expected.unsat_core:
        missing_core = sorted(set(case.expected.unsat_core).difference(actual.unsat_core))
        if missing_core:
            failures.append("missing unsat core constraint(s): " + ", ".join(missing_core))
    if (
        case.expected.minimum_checked_assignments is not None
        and actual.checked_assignments < case.expected.minimum_checked_assignments
    ):
        failures.append(
            "checked assignments "
            f"{actual.checked_assignments} below expected minimum {case.expected.minimum_checked_assignments}"
        )
    if case.expected.reason_contains is not None and case.expected.reason_contains not in (actual.reason or ""):
        failures.append(f"solver reason did not contain {case.expected.reason_contains!r}")
    if not report.ok:
        failures.append("solver replay status or stored witness validation failed")
    return SmtBenchmarkResult(
        case=case,
        status=actual.status,
        conclusion=actual.conclusion,
        passed=not failures,
        query_key=report.query_key,
        checked_assignments=actual.checked_assignments,
        backend=actual.backend.value,
        failures=tuple(failures),
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SmtBenchmarkError(f"cannot read SMT benchmark {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SmtBenchmarkError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmtBenchmarkError(f"{path} must contain a JSON object")
    return payload


def _required_string(raw: dict[str, object], path: Path, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SmtBenchmarkError(f"{path} field {key!r} must be a non-empty string")
    return value


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SmtBenchmarkError(f"{name} must be an object")
    return value


def _string_tuple(value: object, path: Path, case_id: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise SmtBenchmarkError(f"{path} case {case_id!r} field {field!r} must be a sequence")
    strings = tuple(str(item) for item in value)
    if any(not item for item in strings):
        raise SmtBenchmarkError(f"{path} case {case_id!r} field {field!r} must not contain empty strings")
    return strings


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
