from promptabi.capability_negotiation import (
    Capability,
    Fallback,
    NegotiationOutcome,
    ProviderCapabilities,
    negotiate,
    render_negotiation_text,
)


def test_all_native_satisfied():
    caps = ProviderCapabilities("acme", frozenset({"json_schema", "streaming"}))
    contract = negotiate((Capability("json_schema"), Capability("streaming")), caps)
    assert contract.satisfiable
    assert not contract.used_fallback


def test_fallback_used():
    caps = ProviderCapabilities(
        "acme",
        frozenset({"streaming"}),
        fallbacks=(Fallback("json_schema", "json_mode+validation"),),
    )
    contract = negotiate((Capability("json_schema"),), caps)
    assert contract.satisfiable
    assert contract.used_fallback
    assert contract.items[0].outcome == NegotiationOutcome.SATISFIED_WITH_FALLBACK


def test_unmet_requirement():
    caps = ProviderCapabilities("acme", frozenset())
    contract = negotiate((Capability("parallel_tools"),), caps)
    assert not contract.satisfiable
    assert contract.items[0].outcome == NegotiationOutcome.UNMET


def test_render_smoke():
    caps = ProviderCapabilities("acme", frozenset({"x"}))
    out = render_negotiation_text(negotiate((Capability("x"),), caps))
    assert out.endswith("\n")
