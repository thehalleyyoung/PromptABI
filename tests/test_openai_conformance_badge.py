from promptabi.openai_conformance_badge import (
    ConformanceState,
    evaluate_server,
    render_server_badge_text,
)


def test_full_conformance_green():
    caps = {
        "chat_completions": True,
        "tool_calls": True,
        "json_mode": True,
        "streaming": True,
        "stop_sequences": True,
        "logprobs": True,
    }
    badge = evaluate_server("srv", caps, caps)
    assert badge.conformant == badge.total
    assert badge.color == "green"
    assert all(r.state == ConformanceState.CONFORMANT for r in badge.reports)


def test_declared_but_missing_is_partial():
    declared = {"chat_completions": True, "tool_calls": True}
    observed = {"chat_completions": True, "tool_calls": False}
    badge = evaluate_server("srv", declared, observed)
    tool = next(r for r in badge.reports if r.capability == "tool_calls")
    assert tool.state == ConformanceState.PARTIAL
    assert badge.color in {"yellow", "red"}


def test_shields_endpoint_and_render():
    badge = evaluate_server("srv", {}, {})
    ep = badge.to_shields_endpoint()
    assert ep["label"] == "openai-compat"
    assert "/" in str(ep["message"])
    out = render_server_badge_text(badge)
    assert out.endswith("\n")
