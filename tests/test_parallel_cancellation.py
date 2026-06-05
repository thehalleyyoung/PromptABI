from promptabi.parallel_cancellation import (
    CallDisposition,
    CancellationFindingKind,
    EmittedCall,
    render_cancellation_text,
    verify_cancellation,
)


def test_all_accounted_consistent():
    emitted = (EmittedCall("a", "f"), EmittedCall("b", "g"))
    disp = (CallDisposition("a", "answered"), CallDisposition("b", "cancelled"))
    result = verify_cancellation(emitted, disp, frozenset({"b"}))
    assert result.consistent


def test_unaccounted_call():
    emitted = (EmittedCall("a", "f"), EmittedCall("b", "g"))
    disp = (CallDisposition("a", "answered"),)
    result = verify_cancellation(emitted, disp, frozenset())
    kinds = {f.kind for f in result.findings}
    assert CancellationFindingKind.UNACCOUNTED_CALL in kinds


def test_result_for_cancelled():
    emitted = (EmittedCall("a", "f"),)
    disp = (CallDisposition("a", "answered"),)
    result = verify_cancellation(emitted, disp, frozenset({"a"}))
    kinds = {f.kind for f in result.findings}
    assert CancellationFindingKind.RESULT_FOR_CANCELLED in kinds


def test_double_accounted():
    emitted = (EmittedCall("a", "f"),)
    disp = (CallDisposition("a", "answered"), CallDisposition("a", "cancelled"))
    result = verify_cancellation(emitted, disp, frozenset())
    kinds = {f.kind for f in result.findings}
    assert CancellationFindingKind.DOUBLE_ACCOUNTED in kinds


def test_unknown_call_referenced():
    emitted = (EmittedCall("a", "f"),)
    disp = (CallDisposition("a", "answered"), CallDisposition("z", "answered"))
    result = verify_cancellation(emitted, disp, frozenset())
    kinds = {f.kind for f in result.findings}
    assert CancellationFindingKind.UNKNOWN_CALL_REFERENCED in kinds


def test_render_smoke():
    out = render_cancellation_text(verify_cancellation((), (), frozenset()))
    assert out.endswith("\n")
