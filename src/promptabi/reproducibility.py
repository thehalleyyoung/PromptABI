"""Paper reproducibility package generation for PromptABI."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .benchmarks import benchmark_cases, repo_root, run_benchmarks
from .evaluation import run_evaluation
from .provider_fixture_packs import DEFAULT_PROVIDER_FIXTURE_PACK_ROOT, build_provider_fixture_pack_manifest
from .real_bug_benchmarks import DEFAULT_REAL_BUG_BENCHMARK_PATH, build_real_bug_benchmark_manifest
from .seed_corpus import DEFAULT_SEED_CORPUS_ROOT, build_seed_corpus_manifest
from .structured_schema_corpus import DEFAULT_STRUCTURED_SCHEMA_CORPUS_ROOT, build_structured_schema_corpus_manifest


REPRODUCIBILITY_PACKAGE_VERSION = 1
DEFAULT_REPRODUCIBILITY_OUTPUT_DIR = Path("paper_artifact")
DEFAULT_EVALUATION_CORPUS_PATH = repo_root() / "fixtures" / "evaluation" / "labeled_corpus.json"
PACKAGE_FILENAMES = (
    "manifest.json",
    "fixture_hashes.json",
    "expected_tables.json",
    "environment.json",
    "reproduction_commands.sh",
)


class ReproducibilityPackageError(ValueError):
    """Raised when the paper reproducibility package cannot be built or written safely."""


@dataclass(frozen=True, slots=True)
class ReproducibilityInputs:
    """Concrete paths used to freeze a paper artifact package."""

    repository_root: Path = repo_root()
    seed_corpus_root: Path = DEFAULT_SEED_CORPUS_ROOT
    structured_schema_root: Path = DEFAULT_STRUCTURED_SCHEMA_CORPUS_ROOT
    provider_fixture_root: Path = DEFAULT_PROVIDER_FIXTURE_PACK_ROOT
    real_bug_benchmark_path: Path = DEFAULT_REAL_BUG_BENCHMARK_PATH
    evaluation_corpus_path: Path = DEFAULT_EVALUATION_CORPUS_PATH


@dataclass(frozen=True, slots=True)
class ReproducibilityPackage:
    """In-memory representation of the complete paper reproducibility package."""

    manifest: dict[str, object]
    fixture_hashes: dict[str, object]
    expected_tables: dict[str, object]
    environment: dict[str, object]
    reproduction_commands: str

    def file_payloads(self) -> dict[str, str]:
        return {
            "manifest.json": _json_dump(self.manifest),
            "fixture_hashes.json": _json_dump(self.fixture_hashes),
            "expected_tables.json": _json_dump(self.expected_tables),
            "environment.json": _json_dump(self.environment),
            "reproduction_commands.sh": self.reproduction_commands,
        }


def build_reproducibility_package(
    *,
    inputs: ReproducibilityInputs | None = None,
    benchmark_iterations: int = 1,
) -> ReproducibilityPackage:
    """Build the deterministic paper artifact package against real repository code paths."""

    if benchmark_iterations <= 0:
        raise ReproducibilityPackageError("benchmark_iterations must be positive")
    resolved = inputs or ReproducibilityInputs()
    fixture_hashes = _fixture_hashes(resolved)
    expected_tables = _expected_tables(resolved, benchmark_iterations=benchmark_iterations)
    environment = _environment(resolved)
    commands = _reproduction_commands(resolved, benchmark_iterations=benchmark_iterations)
    manifest = _manifest(
        resolved,
        fixture_hashes=fixture_hashes,
        expected_tables=expected_tables,
        environment=environment,
        benchmark_iterations=benchmark_iterations,
    )
    return ReproducibilityPackage(
        manifest=manifest,
        fixture_hashes=fixture_hashes,
        expected_tables=expected_tables,
        environment=environment,
        reproduction_commands=commands,
    )


def write_reproducibility_package(
    output_dir: str | Path = DEFAULT_REPRODUCIBILITY_OUTPUT_DIR,
    *,
    inputs: ReproducibilityInputs | None = None,
    benchmark_iterations: int = 1,
    force: bool = False,
) -> ReproducibilityPackage:
    """Write the paper artifact package to a directory, refusing unsafe overwrites by default."""

    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise ReproducibilityPackageError(f"output path exists and is not a directory: {destination}")
        existing = {path.name for path in destination.iterdir() if not path.name.startswith(".")}
        unexpected = existing.difference(PACKAGE_FILENAMES)
        if existing and (unexpected or not force):
            detail = ", ".join(sorted(existing))
            raise ReproducibilityPackageError(
                f"output directory is not empty: {destination} ({detail}); pass --force to overwrite package files"
            )
    destination.mkdir(parents=True, exist_ok=True)
    package = build_reproducibility_package(inputs=inputs, benchmark_iterations=benchmark_iterations)
    for name, payload in package.file_payloads().items():
        path = destination / name
        path.write_text(payload, encoding="utf-8")
        if name.endswith(".sh"):
            path.chmod(0o755)
    return package


def _manifest(
    inputs: ReproducibilityInputs,
    *,
    fixture_hashes: dict[str, object],
    expected_tables: dict[str, object],
    environment: dict[str, object],
    benchmark_iterations: int,
) -> dict[str, object]:
    artifact = {
        "manifest_version": REPRODUCIBILITY_PACKAGE_VERSION,
        "promptabi_version": __version__,
        "repository_root": ".",
        "purpose": "PromptABI paper artifact: frozen corpora, fixture hashes, solver pins, expected tables, and regeneration commands.",
        "package_files": list(PACKAGE_FILENAMES),
        "frozen_corpora": {
            "seed_corpus": _rel(inputs.seed_corpus_root, inputs.repository_root),
            "structured_schemas": _rel(inputs.structured_schema_root, inputs.repository_root),
            "provider_fixture_packs": _rel(inputs.provider_fixture_root, inputs.repository_root),
            "real_bug_benchmark": _rel(inputs.real_bug_benchmark_path, inputs.repository_root),
            "evaluation_corpus": _rel(inputs.evaluation_corpus_path, inputs.repository_root),
        },
        "benchmark_iterations": benchmark_iterations,
        "summary": {
            "fixture_file_count": fixture_hashes["summary"]["file_count"],  # type: ignore[index]
            "fixture_tree_sha256": fixture_hashes["summary"]["tree_sha256"],  # type: ignore[index]
            "evaluation_case_count": expected_tables["evaluation"]["case_count"],  # type: ignore[index]
            "evaluation_passed": expected_tables["evaluation"]["passed"],  # type: ignore[index]
            "benchmark_count": len(expected_tables["benchmarks"]),  # type: ignore[arg-type]
            "z3_available": environment["solver"]["z3_available"],  # type: ignore[index]
        },
    }
    artifact["artifact_sha256"] = _stable_json_hash(
        {
            "manifest_without_hash": artifact,
            "fixture_hashes_sha256": _stable_json_hash(fixture_hashes),
            "expected_tables_sha256": _stable_json_hash(expected_tables),
            "environment_sha256": _stable_json_hash(environment),
        }
    )
    return artifact


def _fixture_hashes(inputs: ReproducibilityInputs) -> dict[str, object]:
    corpora = {
        "seed_corpus": _corpus_hashes(inputs.seed_corpus_root, inputs.repository_root),
        "structured_schemas": _corpus_hashes(inputs.structured_schema_root, inputs.repository_root),
        "provider_fixture_packs": _corpus_hashes(inputs.provider_fixture_root, inputs.repository_root),
        "real_bug_benchmark": _file_hashes((inputs.real_bug_benchmark_path,), inputs.repository_root),
        "evaluation_corpus": _file_hashes((inputs.evaluation_corpus_path,), inputs.repository_root),
    }
    files = [entry for corpus in corpora.values() for entry in corpus["files"]]  # type: ignore[index]
    return {
        "manifest_version": REPRODUCIBILITY_PACKAGE_VERSION,
        "corpora": corpora,
        "summary": {
            "file_count": len(files),
            "tree_sha256": _stable_json_hash(files),
        },
    }


def _expected_tables(inputs: ReproducibilityInputs, *, benchmark_iterations: int) -> dict[str, object]:
    seed_manifest = build_seed_corpus_manifest(inputs.seed_corpus_root)
    structured_manifest = build_structured_schema_corpus_manifest(inputs.structured_schema_root)
    provider_manifest = build_provider_fixture_pack_manifest(inputs.provider_fixture_root)
    real_bug_manifest = build_real_bug_benchmark_manifest(inputs.real_bug_benchmark_path)
    evaluation = run_evaluation(inputs.evaluation_corpus_path).to_dict()
    benchmarks = run_benchmarks(("all",), iterations=benchmark_iterations, root=inputs.repository_root)
    return {
        "manifest_version": REPRODUCIBILITY_PACKAGE_VERSION,
        "corpus_manifests": {
            "seed_corpus": {
                "entry_count": seed_manifest["entry_count"],
                "families": seed_manifest["families"],
                "manifest_sha256": _normalized_manifest_hash(seed_manifest),
            },
            "structured_schemas": {
                "entry_count": structured_manifest["entry_count"],
                "source_categories": structured_manifest["source_categories"],
                "manifest_sha256": _normalized_manifest_hash(structured_manifest),
            },
            "provider_fixture_packs": {
                "entry_count": provider_manifest["entry_count"],
                "provider_families": provider_manifest["provider_families"],
                "manifest_sha256": _normalized_manifest_hash(provider_manifest),
            },
            "real_bug_benchmark": {
                "case_count": real_bug_manifest["case_count"],
                "categories": real_bug_manifest["categories"],
                "all_cases_passed": real_bug_manifest["all_cases_passed"],
                "manifest_sha256": _normalized_manifest_hash(real_bug_manifest),
            },
        },
        "evaluation": {
            "case_count": evaluation["case_count"],
            "passed": evaluation["passed"],
            "precision": evaluation["score"]["precision"],  # type: ignore[index]
            "recall": evaluation["score"]["recall"],  # type: ignore[index]
            "f1": evaluation["score"]["f1"],  # type: ignore[index]
            "abstention_rate": evaluation["abstention_rate"],
            "witness_quality_score": evaluation["witness_quality_score"],
            "solver_result_quality": evaluation["solver_result_quality"],
            "differential_agreement_rate": evaluation["differential_agreement_rate"],
        },
        "benchmarks": [
            {
                "benchmark": result.name,
                "iterations": result.iterations,
                "metrics": _stable_benchmark_metrics(result.metrics),
            }
            for result in benchmarks
        ],
    }


def _environment(inputs: ReproducibilityInputs) -> dict[str, object]:
    pyproject = _read_pyproject(inputs.repository_root)
    z3_version = _z3_version()
    dependencies = list(pyproject.get("project", {}).get("dependencies", []))  # type: ignore[union-attr]
    optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})  # type: ignore[union-attr]
    return {
        "manifest_version": REPRODUCIBILITY_PACKAGE_VERSION,
        "python": {
            "requires": pyproject.get("project", {}).get("requires-python"),  # type: ignore[union-attr]
            "observed": platform.python_version(),
        },
        "package": {
            "name": pyproject.get("project", {}).get("name"),  # type: ignore[union-attr]
            "version": __version__,
            "dependencies": dependencies,
            "optional_dependencies": optional_dependencies,
        },
        "solver": {
            "z3_available": z3_version is not None,
            "z3_version": z3_version,
            "reproduction_pin": f"z3-solver=={z3_version}" if z3_version is not None else None,
            "fallback": "finite exhaustive enumeration for bounded contracts when z3-solver is absent",
        },
        "benchmark_cases": [case.name for case in benchmark_cases()],
    }


def _reproduction_commands(inputs: ReproducibilityInputs, *, benchmark_iterations: int) -> str:
    z3_pin = _z3_version()
    install_solver = f"python -m pip install z3-solver=={_shell(z3_pin)}\n" if z3_pin else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
        "python -m pip install -e .\n"
        f"{install_solver}"
        "python -m promptabi.benchmarks all --iterations "
        f"{benchmark_iterations} --repo-root . > ${'{'}PROMPTABI_BENCHMARK_JSON:-benchmark-results.json{'}'}\n"
        "promptabi corpus manifest --output seed-corpus-manifest.json\n"
        "promptabi corpus structured-schema-manifest --output structured-schema-manifest.json\n"
        "promptabi corpus provider-fixture-manifest --output provider-fixture-manifest.json\n"
        "promptabi corpus real-bug-benchmark --output real-bug-benchmark-manifest.json\n"
        "promptabi corpus evaluation --format json --output evaluation-report.json\n"
        f"promptabi paper reproducibility --output-dir paper_artifact --benchmark-iterations {benchmark_iterations} --force\n"
    )


def _corpus_hashes(root: Path, repo: Path) -> dict[str, object]:
    if not root.is_dir():
        raise ReproducibilityPackageError(f"corpus root does not exist: {root}")
    files = tuple(path for path in sorted(root.rglob("*")) if path.is_file())
    return _file_hashes(files, repo)


def _file_hashes(paths: tuple[Path, ...], repo: Path) -> dict[str, object]:
    entries = []
    for path in paths:
        if not path.is_file():
            raise ReproducibilityPackageError(f"fixture file does not exist: {path}")
        entries.append(
            {
                "path": _rel(path, repo),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {"file_count": len(entries), "tree_sha256": _stable_json_hash(entries), "files": entries}


def _read_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ReproducibilityPackageError(f"{path} did not parse as a TOML table")
    return data


def _z3_version() -> str | None:
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError:
        return None
    version = getattr(z3, "get_version_string", None)
    if callable(version):
        return str(version())
    return getattr(z3, "__version__", "installed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _normalized_manifest_hash(value: dict[str, object]) -> str:
    return _stable_json_hash({key: item for key, item in value.items() if key not in {"manifest_sha256", "root", "path"}})


def _stable_benchmark_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in sorted(metrics.items())
        if "second" not in key and "per_second" not in key
    }


def _json_dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _shell(value: str) -> str:
    return shlex.quote(value)
