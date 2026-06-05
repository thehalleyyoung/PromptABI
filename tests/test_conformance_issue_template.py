from promptabi.conformance_issue_template import (
    ConformanceFailure,
    IssueSeverity,
    dedup_issues,
    render_issue_template,
    render_issue_template_text,
)


def _failure(vector_id="v1", obligation="stop-terminates"):
    return ConformanceFailure(
        vector_id=vector_id,
        obligation=obligation,
        expected="output stops at stop sequence",
        observed="output continued past stop",
        severity=IssueSeverity.BLOCKER,
    )


def test_template_fields_and_fingerprint_stable():
    t1 = render_issue_template(_failure())
    t2 = render_issue_template(_failure())
    assert t1.fingerprint == t2.fingerprint
    assert t1.severity == IssueSeverity.BLOCKER
    assert "stop-terminates" in t1.title
    assert "Reproduce" in t1.body


def test_dedup_collapses_identical():
    failures = (_failure(), _failure())
    assert len(dedup_issues(failures)) == 1


def test_dedup_keeps_distinct():
    failures = (_failure(vector_id="v1"), _failure(vector_id="v2"))
    assert len(dedup_issues(failures)) == 2


def test_render_smoke():
    out = render_issue_template_text(render_issue_template(_failure()))
    assert out.endswith("\n")
