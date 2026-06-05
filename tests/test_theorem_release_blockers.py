import json

from promptabi import (
    REQUIRED_RELEASE_THEOREMS,
    TheoremReleaseBlockerKind,
    derive_theorem_release_blockers,
    render_theorem_release_blockers_text,
)
from promptabi.cli import main
from promptabi.release import build_release_readiness_report


def test_live_repository_has_no_theorem_release_blockers() -> None:
    gate = derive_theorem_release_blockers(release_version="1.0.0")

    assert gate.release_allowed
    assert gate.blockers == ()
    assert gate.proven_count == gate.traced_count
    assert set(gate.required_theorems) == set(REQUIRED_RELEASE_THEOREMS)
    assert "RELEASE-ALLOWED" in render_theorem_release_blockers_text(gate)


def test_missing_required_theorem_blocks_release() -> None:
    gate = derive_theorem_release_blockers(
        required_theorems=(*REQUIRED_RELEASE_THEOREMS, "theorem-that-does-not-exist"),
    )

    assert not gate.release_allowed
    blocker = next(
        b for b in gate.blockers if b.property_id == "theorem-that-does-not-exist"
    )
    assert blocker.kind is TheoremReleaseBlockerKind.MISSING_REQUIRED_THEOREM
    assert blocker.witness.minimal_fixes


def test_release_readiness_reports_release_blockers_in_evidence() -> None:
    report = build_release_readiness_report(expected_version="1.0.0")
    check = next(c for c in report.checks if c.name == "theorem-traceability")
    evidence = dict(check.evidence)

    assert check.passed
    assert evidence["release_blockers"] == []
    assert set(evidence["required_theorems"]) == set(REQUIRED_RELEASE_THEOREMS)


def test_release_blockers_cli(capsys) -> None:
    exit_code = main(["release", "blockers", "--format", "json", "--release-version", "1.0.0"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["release_allowed"] is True
    assert payload["manifest_version"] == "promptabi.theorem-release-blockers.v1"
    assert payload["proven_count"] == payload["traced_count"]
