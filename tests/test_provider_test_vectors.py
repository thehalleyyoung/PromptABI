from promptabi.provider_test_vectors import (
    Obligation,
    ProviderTestVector,
    TestVectorError,
    VectorMessage,
    load_test_vector,
    render_test_vector_text,
)


def _raw() -> dict[str, object]:
    return {
        "vector_id": "v1",
        "messages": [{"role": "user", "content": "hi"}],
        "obligations": ["stop-terminates", "role-non-forgeable"],
        "params": {"temperature": 0},
        "expected_features": {"max_tokens": 5},
    }


def test_load_roundtrip_and_digest_stable():
    vec = load_test_vector(_raw())
    assert vec.vector_id == "v1"
    assert Obligation.STOP_TERMINATES in vec.obligations
    again = load_test_vector(_raw())
    assert vec.digest() == again.digest()
    assert vec.digest().startswith("sha256:")


def test_to_dict_roundtrips_through_loader():
    vec = load_test_vector(_raw())
    assert load_test_vector(vec.to_dict()).digest() == vec.digest()


def test_empty_messages_rejected():
    raw = _raw()
    raw["messages"] = []
    try:
        load_test_vector(raw)
    except TestVectorError:
        pass
    else:
        raise AssertionError("expected TestVectorError")


def test_render_smoke():
    vec = ProviderTestVector(
        vector_id="v2",
        messages=(VectorMessage("user", "hello"),),
        obligations=(Obligation.TOOL_CALL_WELL_FORMED,),
    )
    out = render_test_vector_text(vec)
    assert "v2" in out
    assert out.endswith("\n")
