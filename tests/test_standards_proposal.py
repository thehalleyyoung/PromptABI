from promptabi.standards_proposal import (
    ClauseLevel,
    default_proposal,
    render_adoption_report_text,
    score_adoption,
)


def test_default_proposal_has_clauses():
    proposal = default_proposal()
    assert proposal.clauses
    must = [c for c in proposal.clauses if c.level == ClauseLevel.MUST]
    assert must


def test_full_compliance():
    proposal = default_proposal()
    report = score_adoption(proposal, "acme", proposal.clause_ids())
    assert report.compliant
    assert report.adoption_score == 1.0
    assert report.unmet_must == ()


def test_partial_compliance_unmet_must():
    proposal = default_proposal()
    # satisfy everything except one MUST clause
    must_ids = [c.clause_id for c in proposal.clauses if c.level == ClauseLevel.MUST]
    satisfied = proposal.clause_ids() - {must_ids[0]}
    report = score_adoption(proposal, "acme", frozenset(satisfied))
    assert not report.compliant
    assert must_ids[0] in report.unmet_must
    assert 0.0 < report.adoption_score < 1.0


def test_render_smoke():
    proposal = default_proposal()
    out = render_adoption_report_text(
        score_adoption(proposal, "acme", proposal.clause_ids())
    )
    assert out.endswith("\n")
