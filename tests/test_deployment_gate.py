from promptabi.deployment_gate import (
    ConformanceEvidence,
    GateBlockKind,
    GateDecision,
    GatePolicy,
    evaluate_gate,
    render_gate_text,
)


def _evidence(**kw) -> ConformanceEvidence:
    base = dict(
        provider="acme",
        revision="2024-06",
        pass_rate=0.99,
        failed_obligations=frozenset(),
        max_open_severity="minor",
    )
    base.update(kw)
    return ConformanceEvidence(**base)


def _policy(**kw) -> GatePolicy:
    base = dict(
        environment="prod",
        min_pass_rate=0.95,
        required_obligations=frozenset({"stop-terminates"}),
        max_allowed_severity="major",
        denied_revisions=frozenset(),
    )
    base.update(kw)
    return GatePolicy(**base)


def test_promote_when_all_pass():
    result = evaluate_gate(_evidence(), _policy())
    assert result.decision == GateDecision.PROMOTE


def test_block_low_pass_rate():
    result = evaluate_gate(_evidence(pass_rate=0.5), _policy())
    kinds = {b.kind for b in result.blocks}
    assert GateBlockKind.PASS_RATE_TOO_LOW in kinds
    assert result.decision == GateDecision.BLOCK


def test_block_required_obligation_failed():
    result = evaluate_gate(
        _evidence(failed_obligations=frozenset({"stop-terminates"})), _policy()
    )
    kinds = {b.kind for b in result.blocks}
    assert GateBlockKind.REQUIRED_OBLIGATION_FAILED in kinds


def test_block_severity_exceeded():
    result = evaluate_gate(_evidence(max_open_severity="blocker"), _policy())
    kinds = {b.kind for b in result.blocks}
    assert GateBlockKind.SEVERITY_EXCEEDED in kinds


def test_block_denied_revision():
    result = evaluate_gate(
        _evidence(), _policy(denied_revisions=frozenset({"2024-06"}))
    )
    kinds = {b.kind for b in result.blocks}
    assert GateBlockKind.REVISION_DENIED in kinds


def test_render_smoke():
    out = render_gate_text(evaluate_gate(_evidence(), _policy()))
    assert out.endswith("\n")
