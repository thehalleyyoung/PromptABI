from promptabi.refusal_envelope import (
    RefusalClass,
    RefusalFindingKind,
    StructuredResponse,
    classify_refusal,
    render_refusal_text,
)


def test_valid_data():
    resp = StructuredResponse(
        finish_reason="stop",
        parsed={"city": "Paris"},
        refusal=None,
        raw_content='{"city": "Paris"}',
    )
    result = classify_refusal(resp)
    assert result.classification == RefusalClass.VALID_DATA
    assert result.safe_to_parse


def test_well_formed_refusal_via_flag():
    resp = StructuredResponse(
        finish_reason="content_filter",
        parsed=None,
        refusal="cannot comply",
        raw_content="",
    )
    result = classify_refusal(resp)
    assert result.classification == RefusalClass.WELL_FORMED_REFUSAL
    assert not result.safe_to_parse


def test_ambiguous_refusal_in_data_channel():
    resp = StructuredResponse(
        finish_reason="stop",
        parsed=None,
        refusal=None,
        raw_content="I can't help with that.",
    )
    result = classify_refusal(resp)
    assert result.classification == RefusalClass.AMBIGUOUS_REFUSAL
    kinds = {f.kind for f in result.findings}
    assert RefusalFindingKind.REFUSAL_IN_DATA_CHANNEL in kinds
    assert RefusalFindingKind.MISSING_REFUSAL_FLAG in kinds


def test_data_and_refusal_conflict():
    resp = StructuredResponse(
        finish_reason="content_filter",
        parsed={"x": 1},
        refusal="no",
        raw_content="{}",
    )
    result = classify_refusal(resp)
    assert result.classification == RefusalClass.AMBIGUOUS_REFUSAL
    kinds = {f.kind for f in result.findings}
    assert RefusalFindingKind.DATA_AND_REFUSAL in kinds


def test_render_smoke():
    resp = StructuredResponse("stop", {"x": 1}, None, "{}")
    out = render_refusal_text(classify_refusal(resp))
    assert out.endswith("\n")
