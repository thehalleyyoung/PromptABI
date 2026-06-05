from promptabi.annual_provider_report import (
    RevisionRecord,
    compile_annual_report,
    render_annual_report_text,
)


def test_net_change_and_counts():
    records = (
        RevisionRecord("r1", 0, 0.80),
        RevisionRecord("r2", 1, 0.90),
        RevisionRecord("r3", 2, 0.95),
    )
    report = compile_annual_report(year=2024, provider="acme", records=records)
    assert report.revisions_shipped == 3
    assert abs(report.net_pass_rate_change - 0.15) < 1e-9


def test_regression_and_recovery_tracked():
    records = (
        RevisionRecord("r1", 0, 0.9, failing_obligations=frozenset()),
        RevisionRecord("r2", 1, 0.8, failing_obligations=frozenset({"stop"})),
        RevisionRecord("r3", 2, 0.9, failing_obligations=frozenset()),
    )
    report = compile_annual_report(year=2024, provider="acme", records=records)
    assert "stop" in report.regressed_obligations
    assert "stop" in report.recovered_obligations


def test_longest_standing_failure():
    records = (
        RevisionRecord("r1", 0, 0.9, failing_obligations=frozenset({"tool"})),
        RevisionRecord("r2", 1, 0.9, failing_obligations=frozenset({"tool"})),
        RevisionRecord("r3", 2, 0.9, failing_obligations=frozenset({"tool"})),
    )
    report = compile_annual_report(year=2024, provider="acme", records=records)
    assert report.longest_standing_failure == "tool"
    assert report.longest_standing_span == 3


def test_empty_records():
    report = compile_annual_report(year=2024, provider="acme", records=())
    assert report.revisions_shipped == 0


def test_render_smoke():
    records = (RevisionRecord("r1", 0, 0.9),)
    out = render_annual_report_text(
        compile_annual_report(year=2024, provider="acme", records=records)
    )
    assert out.endswith("\n")
