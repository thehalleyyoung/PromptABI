from promptabi.context_window_semantics import (
    ContextFindingKind,
    ContextWindowSemantics,
    RequestPlan,
    RevisionChangeKind,
    check_request_fits,
    diff_revisions,
    render_fit_text,
)


def _sem(**kw) -> ContextWindowSemantics:
    base = dict(
        provider="acme",
        revision="2024-01",
        max_total_tokens=8000,
        max_output_tokens=4000,
        shared_budget=True,
        template_token_overhead=10,
    )
    base.update(kw)
    return ContextWindowSemantics(**base)


def test_request_fits_shared_budget():
    result = check_request_fits(_sem(), RequestPlan(1000, 500))
    assert result.fits


def test_shared_budget_overflow():
    result = check_request_fits(_sem(), RequestPlan(7000, 2000))
    kinds = {f.kind for f in result.findings}
    assert ContextFindingKind.SHARED_BUDGET_OVERFLOW in kinds
    assert not result.fits


def test_output_cap_exceeded():
    result = check_request_fits(_sem(), RequestPlan(10, 5000))
    kinds = {f.kind for f in result.findings}
    assert ContextFindingKind.EXCEEDS_OUTPUT in kinds


def test_non_shared_total_overflow():
    sem = _sem(shared_budget=False, max_total_tokens=1000)
    result = check_request_fits(sem, RequestPlan(2000, 10))
    kinds = {f.kind for f in result.findings}
    assert ContextFindingKind.EXCEEDS_TOTAL in kinds


def test_diff_detects_shrink_and_budget_change():
    old = _sem()
    new = _sem(max_total_tokens=4000, max_output_tokens=2000, shared_budget=False)
    kinds = {c.kind for c in diff_revisions(old, new)}
    assert RevisionChangeKind.TOTAL_SHRUNK in kinds
    assert RevisionChangeKind.OUTPUT_SHRUNK in kinds
    assert RevisionChangeKind.BUDGET_MODEL_CHANGED in kinds


def test_render_smoke():
    out = render_fit_text(check_request_fits(_sem(), RequestPlan(1, 1)))
    assert out.endswith("\n")
