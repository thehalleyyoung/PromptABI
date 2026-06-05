from promptabi.provider_benchmark_drift import (
    ConformanceRun,
    DriftFindingKind,
    build_drift_dashboard,
    render_drift_dashboard_text,
)


def test_no_drift_when_stable():
    runs = (
        ConformanceRun("r1", 10, 10),
        ConformanceRun("r2", 10, 10),
    )
    dash = build_drift_dashboard(runs)
    assert not dash.alarm
    assert dash.latest_pass_rate == 1.0


def test_regression_detected():
    runs = (
        ConformanceRun("r1", 10, 10),
        ConformanceRun("r2", 10, 8, failing_vector_ids=frozenset({"v1", "v2"})),
    )
    dash = build_drift_dashboard(runs)
    kinds = {f.kind for f in dash.findings}
    assert DriftFindingKind.REGRESSION in kinds
    assert DriftFindingKind.PASS_RATE_DROP in kinds
    assert dash.alarm


def test_recovery_detected():
    runs = (
        ConformanceRun("r1", 10, 8, failing_vector_ids=frozenset({"v1", "v2"})),
        ConformanceRun("r2", 10, 10),
    )
    dash = build_drift_dashboard(runs)
    kinds = {f.kind for f in dash.findings}
    assert DriftFindingKind.RECOVERED in kinds


def test_empty_runs():
    dash = build_drift_dashboard(())
    assert not dash.alarm
    assert dash.latest_pass_rate == 0.0


def test_render_smoke():
    out = render_drift_dashboard_text(
        build_drift_dashboard((ConformanceRun("r1", 5, 5),))
    )
    assert out.endswith("\n")
