import json
import shutil
from pathlib import Path

import promptabi
from promptabi import TraceEvidenceKind, build_theorem_traceability_report, render_theorem_traceability_text
from promptabi.cli import main
from promptabi.release import build_release_readiness_report


def test_theorem_traceability_covers_every_core_proof_claim() -> None:
    report = build_theorem_traceability_report()

    assert report.passed
    assert {trace.property_id for trace in report.traces} == {
        "role-boundary-nonforgeability",
        "stop-overreachability",
        "grammar-tokenizer-emptiness",
        "must-survive-budget",
        "z3-backed-finite-contract",
        "incremental-cache-soundness",
    }
    for trace in report.traces:
        assert {item.kind for item in trace.evidence} >= set(TraceEvidenceKind)
        assert not trace.missing_kinds
        assert not trace.failures


def test_theorem_traceability_renderers_and_public_api_are_stable(capsys) -> None:
    report = build_theorem_traceability_report()
    text = render_theorem_traceability_text(report)
    payload = json.loads(promptabi.theorem_traceability(output_format="json"))

    assert "PromptABI theorem traceability" in text
    assert "role-boundary-nonforgeability: PASS" in text
    assert payload["passed"] is True
    assert payload["theorem_count"] == 6

    exit_code = main(["proofs", "--traceability", "--format", "json"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["traces"] == payload["traces"]


def test_theorem_traceability_fails_closed_for_stale_evidence_copy(tmp_path: Path) -> None:
    repo = Path.cwd()
    copied = tmp_path / "repo"
    shutil.copytree(
        repo,
        copied,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc", ".venv"),
    )
    target = copied / "tests" / "test_proof_sketches.py"
    target.write_text(target.read_text(encoding="utf-8").replace("test_stop_proof_validates_prefix_split_for_real_overreach_witness", "renamed_stop_test"), encoding="utf-8")

    report = build_theorem_traceability_report(copied)
    by_id = {trace.property_id: trace for trace in report.traces}

    assert not report.passed
    assert not by_id["stop-overreachability"].passed
    assert any("missing symbol" in failure for failure in by_id["stop-overreachability"].failures)


def test_release_readiness_includes_theorem_traceability_gate() -> None:
    report = build_release_readiness_report(expected_version="1.0.0")
    by_name = {check.name: check for check in report.checks}

    assert by_name["theorem-traceability"].passed
    assert by_name["theorem-traceability"].evidence
