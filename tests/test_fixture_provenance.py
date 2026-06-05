from promptabi.fixture_provenance import (
    ProvenanceFindingKind,
    sign_fixture,
    verify_fixture,
    render_provenance_text,
)


def _attest(payload, secret=b"k"):
    return sign_fixture(
        fixture_id="f1",
        provider="acme",
        provider_revision="2024-01",
        captured_by="alice",
        captured_at="2024-01-02",
        payload=payload,
        secret=secret,
    )


def test_valid_attestation_trusted():
    payload = {"a": 1, "b": [1, 2]}
    att = _attest(payload)
    result = verify_fixture(att, payload, known_signers={"alice": b"k"})
    assert result.trusted


def test_tampered_payload_detected():
    att = _attest({"a": 1})
    result = verify_fixture(att, {"a": 2}, known_signers={"alice": b"k"})
    kinds = {f.kind for f in result.findings}
    assert ProvenanceFindingKind.PAYLOAD_TAMPERED in kinds


def test_unknown_signer():
    att = _attest({"a": 1})
    result = verify_fixture(att, {"a": 1}, known_signers={"bob": b"k"})
    kinds = {f.kind for f in result.findings}
    assert ProvenanceFindingKind.UNKNOWN_SIGNER in kinds


def test_bad_signature_wrong_secret():
    att = _attest({"a": 1})
    result = verify_fixture(att, {"a": 1}, known_signers={"alice": b"other"})
    kinds = {f.kind for f in result.findings}
    assert ProvenanceFindingKind.BAD_SIGNATURE in kinds


def test_stale_revision():
    att = _attest({"a": 1})
    result = verify_fixture(
        att,
        {"a": 1},
        known_signers={"alice": b"k"},
        current_revisions={"acme": "2024-06"},
    )
    kinds = {f.kind for f in result.findings}
    assert ProvenanceFindingKind.STALE_REVISION in kinds


def test_render_smoke():
    att = _attest({"a": 1})
    out = render_provenance_text(
        verify_fixture(att, {"a": 1}, known_signers={"alice": b"k"})
    )
    assert out.endswith("\n")
