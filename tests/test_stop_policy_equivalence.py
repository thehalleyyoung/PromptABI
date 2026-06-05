from promptabi.stop_policy_equivalence import (
    StopKind,
    StopPolicy,
    certify_stop_equivalence,
    evaluate_stop_policy,
    render_equivalence_text,
)


def test_string_list_cuts_at_first_match():
    policy = StopPolicy("a", StopKind.STRING_LIST, stops=("\n\n", "END"))
    ev = evaluate_stop_policy(policy, "hello END world")
    assert ev.cut_index == len("hello ")
    assert ev.triggered_by == "END"


def test_equivalent_policies():
    p1 = StopPolicy("a", StopKind.STRING_LIST, stops=("END",))
    p2 = StopPolicy("b", StopKind.SINGLE_STRING, stops=("END",))
    result = certify_stop_equivalence((p1, p2), "foo END bar")
    assert result.equivalent
    assert result.divergences == ()


def test_divergent_policies():
    p1 = StopPolicy("a", StopKind.STRING_LIST, stops=("END",))
    p2 = StopPolicy("b", StopKind.MAX_TOKENS, max_tokens=1)
    result = certify_stop_equivalence((p1, p2), "foobar END xxxx")
    assert not result.equivalent
    assert result.divergences


def test_max_tokens_no_cut_when_short():
    policy = StopPolicy("a", StopKind.MAX_TOKENS, max_tokens=100)
    ev = evaluate_stop_policy(policy, "short")
    assert ev.cut_index == len("short")
    assert ev.triggered_by is None


def test_render_smoke():
    p1 = StopPolicy("a", StopKind.STRING_LIST, stops=("END",))
    out = render_equivalence_text(certify_stop_equivalence((p1,), "x END"))
    assert out.endswith("\n")
