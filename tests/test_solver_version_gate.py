"""Tests for solver-version compatibility gates (step 230)."""

from __future__ import annotations

import json

from promptabi.solver_version_gate import (
    SolverVersionDriftKind,
    SolverVersionGate,
    SolverVersionPolicy,
    capture_solver_fingerprints,
    evaluate_solver_version_gate,
    render_solver_version_gate_json,
    render_solver_version_gate_text,
)


def _policy() -> SolverVersionPolicy:
    return SolverVersionPolicy(
        pinned=frozenset({"z3", "promptabi-finite-contract-solver"}),
        flexible=frozenset({"note"}),
    )


def test_capture_fingerprints_includes_core_solver() -> None:
    fingerprints = capture_solver_fingerprints()
    assert "promptabi-finite-contract-solver" in fingerprints


def test_identical_fingerprints_are_compatible() -> None:
    baseline = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2-query-cache"}
    report = evaluate_solver_version_gate(baseline, dict(baseline), _policy())
    assert report.compatible
    assert report.drifts == ()


def test_pinned_change_requires_reverification() -> None:
    baseline = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2-query-cache"}
    current = {"z3": "4.13.0", "promptabi-finite-contract-solver": "v2-query-cache"}
    report = evaluate_solver_version_gate(baseline, current, _policy())
    assert not report.compatible
    assert report.requires_reverification
    assert any(d.kind is SolverVersionDriftKind.PINNED_CHANGED for d in report.drifts)
    assert report.blocking_drifts


def test_flexible_change_is_tolerated() -> None:
    baseline = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2", "note": "a"}
    current = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2", "note": "b"}
    report = evaluate_solver_version_gate(baseline, current, _policy())
    assert report.compatible
    assert any(d.kind is SolverVersionDriftKind.FLEXIBLE_CHANGED for d in report.drifts)


def test_missing_pinned_component_blocks() -> None:
    baseline = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2"}
    current = {"promptabi-finite-contract-solver": "v2"}
    report = evaluate_solver_version_gate(baseline, current, _policy())
    assert not report.compatible
    assert any(d.kind is SolverVersionDriftKind.PINNED_MISSING for d in report.drifts)


def test_new_pinned_component_blocks() -> None:
    baseline = {"promptabi-finite-contract-solver": "v2"}
    current = {"promptabi-finite-contract-solver": "v2", "z3": "4.13.0"}
    report = evaluate_solver_version_gate(baseline, current, _policy())
    assert not report.compatible
    assert any(d.kind is SolverVersionDriftKind.NEW_COMPONENT for d in report.drifts)


def test_gate_object_and_render() -> None:
    baseline = {"z3": "4.12.1", "promptabi-finite-contract-solver": "v2"}
    gate = SolverVersionGate(baseline=baseline, policy=_policy())
    report = gate.evaluate({"z3": "4.99.9", "promptabi-finite-contract-solver": "v2"})
    payload = json.loads(render_solver_version_gate_json(report))
    assert payload["compatible"] is False
    assert "solver-version gate" in render_solver_version_gate_text(report)
