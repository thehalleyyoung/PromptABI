from promptabi.error_envelope import (
    ErrorEnvelopeFindingKind,
    ProviderErrorResponse,
    certify_error_envelope,
    render_error_envelope_text,
)


def test_conformant_envelope():
    body = {"error": {"message": "bad", "type": "invalid_request_error"}}
    result = certify_error_envelope(ProviderErrorResponse(400, body))
    assert result.conformant


def test_non_object_body():
    result = certify_error_envelope(ProviderErrorResponse(500, "boom"))
    kinds = {f.kind for f in result.findings}
    assert ErrorEnvelopeFindingKind.NOT_AN_OBJECT in kinds


def test_missing_error_key():
    result = certify_error_envelope(ProviderErrorResponse(400, {"message": "x"}))
    kinds = {f.kind for f in result.findings}
    assert ErrorEnvelopeFindingKind.MISSING_ERROR_KEY in kinds


def test_missing_field():
    body = {"error": {"type": "rate_limit_error"}}
    result = certify_error_envelope(ProviderErrorResponse(429, body))
    kinds = {f.kind for f in result.findings}
    assert ErrorEnvelopeFindingKind.MISSING_FIELD in kinds


def test_status_type_mismatch():
    body = {"error": {"message": "x", "type": "server_error"}}
    result = certify_error_envelope(ProviderErrorResponse(400, body))
    kinds = {f.kind for f in result.findings}
    assert ErrorEnvelopeFindingKind.STATUS_TYPE_MISMATCH in kinds


def test_render_smoke():
    out = render_error_envelope_text(
        certify_error_envelope(ProviderErrorResponse(500, "x"))
    )
    assert out.endswith("\n")
