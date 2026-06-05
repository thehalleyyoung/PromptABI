from promptabi.grammar_backends import (
    BackendFindingKind,
    GrammarBackend,
    GrammarFeature,
    GrammarSpec,
    check_backend,
    render_backend_text,
)


def test_compatible_when_features_supported():
    backend = GrammarBackend(
        "llguidance",
        frozenset({GrammarFeature.REGEX, GrammarFeature.JSON_SCHEMA}),
    )
    spec = GrammarSpec("json", frozenset({GrammarFeature.JSON_SCHEMA}))
    result = check_backend(spec, backend)
    assert result.compatible
    assert result.findings == ()


def test_incompatible_reports_feature_and_alternative():
    backend = GrammarBackend("simple", frozenset({GrammarFeature.REGEX}))
    alt = GrammarBackend("cfg", frozenset({GrammarFeature.RECURSION}))
    spec = GrammarSpec("nested", frozenset({GrammarFeature.RECURSION}))
    result = check_backend(spec, backend, alternatives=(alt,))
    assert not result.compatible
    f = result.findings[0]
    assert f.kind == BackendFindingKind.UNSUPPORTED_FEATURE
    assert f.feature == GrammarFeature.RECURSION
    assert f.alternative == "cfg"


def test_render_smoke():
    backend = GrammarBackend("b", frozenset())
    spec = GrammarSpec("g", frozenset({GrammarFeature.REGEX}))
    out = render_backend_text(check_backend(spec, backend))
    assert out.endswith("\n")
