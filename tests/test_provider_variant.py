from promptabi.provider_variant import (
    ProviderVariant,
    VariantFindingKind,
    VariantRequirement,
    check_variant,
    render_variant_text,
)


def _variant(**kw) -> ProviderVariant:
    base = dict(
        provider="acme",
        variant="eu-vpc",
        region="eu-west-1",
        features=frozenset({"json_mode", "tool_calls"}),
        params=frozenset({"temperature", "max_tokens"}),
        data_residency="eu",
    )
    base.update(kw)
    return ProviderVariant(**base)


def _req(**kw) -> VariantRequirement:
    base = dict(
        required_features=frozenset({"json_mode"}),
        required_params=frozenset({"temperature"}),
        allowed_residencies=frozenset({"eu"}),
    )
    base.update(kw)
    return VariantRequirement(**base)


def test_compatible_variant():
    assert check_variant(_variant(), _req()).compatible


def test_missing_feature():
    result = check_variant(_variant(), _req(required_features=frozenset({"vision"})))
    kinds = {f.kind for f in result.findings}
    assert VariantFindingKind.MISSING_FEATURE in kinds


def test_residency_violation():
    result = check_variant(_variant(), _req(allowed_residencies=frozenset({"us"})))
    kinds = {f.kind for f in result.findings}
    assert VariantFindingKind.RESIDENCY_VIOLATION in kinds


def test_unsupported_param():
    result = check_variant(_variant(), _req(required_params=frozenset({"logit_bias"})))
    kinds = {f.kind for f in result.findings}
    assert VariantFindingKind.UNSUPPORTED_PARAM in kinds


def test_render_smoke():
    out = render_variant_text(check_variant(_variant(), _req()))
    assert out.endswith("\n")
