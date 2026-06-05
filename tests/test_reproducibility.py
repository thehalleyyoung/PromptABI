import json
import shutil
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.reproducibility import (
    ReproducibilityInputs,
    ReproducibilityPackageError,
    build_reproducibility_package,
    write_reproducibility_package,
)


def test_reproducibility_package_freezes_real_corpora_and_stable_expected_tables() -> None:
    package = build_reproducibility_package(benchmark_iterations=1)

    assert package.manifest["summary"]["fixture_file_count"] > 20
    assert package.fixture_hashes["summary"]["tree_sha256"]
    assert package.expected_tables["evaluation"]["passed"] is True
    assert package.expected_tables["evaluation"]["precision"] == 1.0
    assert package.expected_tables["corpus_manifests"]["real_bug_benchmark"]["all_cases_passed"] is True
    assert package.expected_tables["mutation_fuzzing"]["mutation_count"] == 16
    assert "smt-counterexample" in package.expected_tables["mutation_fuzzing"]["discovered_rule_ids"]

    benchmark_names = {row["benchmark"] for row in package.expected_tables["benchmarks"]}
    assert benchmark_names == {
        "tokenizer-analysis",
        "template-symbolic-execution",
        "grammar-emptiness",
        "stop-checks",
        "z3-static-contracts",
        "budget-checks",
        "corpus-wide-verification",
        "cache-cold-warm",
    }
    assert all("seconds" not in row and "runs_per_second" not in row for row in package.expected_tables["benchmarks"])
    assert all("second" not in key for row in package.expected_tables["benchmarks"] for key in row["metrics"])
    assert "python -m pip install -e ." in package.reproduction_commands
    assert "promptabi corpus evaluation --format json" in package.reproduction_commands
    assert "promptabi fuzz mutations --format json" in package.reproduction_commands
    assert package.environment["solver"]["z3_available"] in {True, False}
    if package.environment["solver"]["z3_available"]:
        assert str(package.environment["solver"]["reproduction_pin"]).startswith("z3-solver==")

    for payload in package.file_payloads().values():
        if payload.lstrip().startswith("{"):
            json.loads(payload)


def test_reproducibility_writer_and_cli_create_roundtrippable_package(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "artifact"
    package = write_reproducibility_package(output_dir, benchmark_iterations=1)

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "environment.json",
        "expected_tables.json",
        "fixture_hashes.json",
        "manifest.json",
        "reproduction_commands.sh",
    ]
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["artifact_sha256"]
    assert package.manifest["package_files"]

    try:
        write_reproducibility_package(output_dir, benchmark_iterations=1)
    except ReproducibilityPackageError as exc:
        assert "pass --force" in str(exc)
    else:
        raise AssertionError("expected existing reproducibility package to require force")

    cli_dir = tmp_path / "cli-artifact"
    exit_code = main(["paper", "reproducibility", "--output-dir", str(cli_dir), "--benchmark-iterations", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "wrote paper reproducibility package" in captured.out
    assert json.loads((cli_dir / "expected_tables.json").read_text(encoding="utf-8"))["evaluation"]["passed"] is True


def test_reproducibility_hashes_change_when_frozen_fixture_copy_changes(tmp_path: Path) -> None:
    repo = Path.cwd()
    copied_seed = tmp_path / "seed_corpus"
    shutil.copytree(repo / "fixtures" / "seed_corpus", copied_seed)
    inputs = ReproducibilityInputs(repository_root=repo, seed_corpus_root=copied_seed)

    before = build_reproducibility_package(inputs=inputs, benchmark_iterations=1).fixture_hashes["corpora"]["seed_corpus"][
        "tree_sha256"
    ]
    metadata = copied_seed / "llama" / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["reproducibility_notes"] = "mutated in a tmp-path test copy"
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    after = build_reproducibility_package(inputs=inputs, benchmark_iterations=1).fixture_hashes["corpora"]["seed_corpus"][
        "tree_sha256"
    ]

    assert before != after


def test_public_api_builds_reproducibility_package() -> None:
    package = promptabi.create_reproducibility_package(benchmark_iterations=1)

    assert isinstance(package, promptabi.ReproducibilityPackage)
    assert package.expected_tables["evaluation"]["case_count"] >= 11
