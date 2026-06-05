"""Tests for dataset transforms as contract-preserving passes (step 262)."""

from __future__ import annotations

from promptabi.dataset_transforms import (
    DatasetContract,
    InterfaceFootprint,
    TransformPass,
    TransformViolationKind,
    render_pipeline_text,
    verify_pipeline,
)

CONTRACT = DatasetContract(
    allowed_roles=frozenset({"system", "user", "assistant"}),
    required_tokens=frozenset({"<eos>"}),
)
INIT = InterfaceFootprint(
    roles=frozenset({"system", "user", "assistant"}),
    tokens=frozenset({"<eos>", "<bos>"}),
    has_eos=True,
)


def test_identity_pipeline_preserves() -> None:
    passes = (TransformPass("noop", lambda fp: fp),)
    assert verify_pipeline(CONTRACT, INIT, passes).preserved


def test_new_role_breaks() -> None:
    bad = TransformPass(
        "inject",
        lambda fp: InterfaceFootprint(fp.roles | {"developer"}, fp.tokens, fp.has_eos),
    )
    result = verify_pipeline(CONTRACT, INIT, (bad,))
    assert not result.preserved
    assert any(
        v.kind is TransformViolationKind.NEW_ROLE_INTRODUCED for v in result.violations
    )


def test_dropping_required_token_breaks() -> None:
    bad = TransformPass(
        "strip",
        lambda fp: InterfaceFootprint(fp.roles, frozenset(), fp.has_eos),
    )
    result = verify_pipeline(CONTRACT, INIT, (bad,))
    assert any(
        v.kind is TransformViolationKind.REQUIRED_TOKEN_DROPPED
        for v in result.violations
    )


def test_eos_strip_breaks() -> None:
    bad = TransformPass(
        "noeos",
        lambda fp: InterfaceFootprint(fp.roles, fp.tokens, False),
    )
    result = verify_pipeline(CONTRACT, INIT, (bad,))
    assert any(v.kind is TransformViolationKind.EOS_STRIPPED for v in result.violations)


def test_reports_earliest_breaking_pass() -> None:
    p1 = TransformPass("ok", lambda fp: fp)
    p2 = TransformPass(
        "bad",
        lambda fp: InterfaceFootprint(fp.roles | {"x"}, fp.tokens, fp.has_eos),
    )
    p3 = TransformPass("never", lambda fp: fp)
    result = verify_pipeline(CONTRACT, INIT, (p1, p2, p3))
    assert {v.pass_name for v in result.violations} == {"bad"}


def test_render_text_smoke() -> None:
    result = verify_pipeline(CONTRACT, INIT, (TransformPass("noop", lambda fp: fp),))
    assert "pipeline" in render_pipeline_text(result)
