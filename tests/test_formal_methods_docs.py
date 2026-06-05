import re
from pathlib import Path

from promptabi import build_soundness_audit_report, proof_sketches
from promptabi.cli import main
from promptabi.formal import (
    BoundedStringDomain,
    Contains,
    DeterministicFiniteAutomaton,
    FiniteContractProblem,
    SolverStatus,
    Value,
    Var,
    NamedConstraint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "formal-methods.md"


def test_formal_methods_guide_is_linked_and_decision_oriented() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    checks = (REPO_ROOT / "docs" / "checks.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Formal methods guide: formal-methods.md" in mkdocs
    assert "[Formal methods decision guide](formal-methods.md)" in checks
    assert "[formal methods guide](docs/formal-methods.md)" in readme
    for phrase in (
        "Choosing automata, Z3, composition, or abstention",
        "Finite automata",
        "Finite-state transducers",
        "Z3-backed finite SMT",
        "Composed products",
        "Abstention",
    ):
        assert phrase in guide


def test_formal_methods_guide_references_real_files_and_commands(capsys) -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    referenced_paths = sorted(set(re.findall(r"(?:src|fixtures|examples)/[-_./A-Za-z0-9]+", guide)))

    assert "src/promptabi/formal.py" in referenced_paths
    assert "fixtures/solver_replays/role-region-forgery.solver-replay.json" in referenced_paths
    for relative_path in referenced_paths:
        assert (REPO_ROOT / relative_path).exists(), relative_path

    for command in (
        ["proofs", "--format", "json"],
        ["soundness-audit", "--rule", "static-contracts", "--format", "json"],
        ["solver", "replay", "fixtures/solver_replays/role-region-forgery.solver-replay.json", "--format", "json"],
    ):
        exit_code = main(command)
        captured = capsys.readouterr()
        assert exit_code == 0, command
        assert "Traceback" not in captured.err
        assert captured.out.startswith("{")


def test_formal_methods_automata_and_smt_examples_execute_against_real_code() -> None:
    stop = DeterministicFiniteAutomaton.literal("</s>", alphabet=set("</s>abc"))
    prefixes = DeterministicFiniteAutomaton.prefix_closed_literal("</s>", alphabet=set("</s>abc"))
    witness = stop.intersect(prefixes).shortest_witness()

    assert witness is not None
    assert witness.text == "</s>"

    problem = FiniteContractProblem(
        name="delimiter-forgery",
        variables=(BoundedStringDomain("content", tuple("<a>bc"), min_length=0, max_length=3),),
        constraints=(NamedConstraint("contains-marker", Contains(Var("content"), Value("<a>"))),),
    )
    result = problem.solve(prefer_z3=False)

    assert result.status is SolverStatus.SAT
    assert result.assignment == {"content": "<a>"}


def test_formal_methods_public_evidence_surfaces_are_populated() -> None:
    proof_report = proof_sketches()
    audit_report = build_soundness_audit_report(rule="static-contracts")

    assert proof_report.passed
    assert {sketch.property_id for sketch in proof_report.sketches} >= {
        "role-boundary-nonforgeability",
        "z3-backed-finite-contract",
    }
    assert audit_report.passed
    assert audit_report.audits[0].proof_obligations
    assert audit_report.audits[0].differential_evidence
