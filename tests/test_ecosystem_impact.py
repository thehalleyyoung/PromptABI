"""Tests for ecosystem_impact (steps 491-500)."""

from __future__ import annotations

from promptabi.ecosystem_impact import (
    DIAGNOSTIC_MESSAGE_KEYS,
    DisclosureState,
    Marketplace,
    aggregate_impact,
    available_locales,
    coordinate_cve,
    curriculum_labs,
    cut_release,
    governance_model,
    grade_lab,
    milestone_report,
    open_disclosure,
    record_case_study,
    reproduce_case_study,
    steering_committee,
    technical_roadmap,
    translate,
    verify_catalog,
    verify_release,
)


# --- 491 marketplace ---------------------------------------------------------


def test_marketplace_signs_and_verifies():
    mkt = Marketplace(signing_key="k")
    p = mkt.publish(name="my-check", version="1.0", rule_ids=["x"], certified=True)
    assert mkt.verify(p)
    assert mkt.installable(p)


def test_marketplace_rejects_uncertified():
    mkt = Marketplace(signing_key="k")
    p = mkt.publish(name="x", version="1", rule_ids=[], certified=False)
    assert mkt.verify(p)
    assert not mkt.installable(p)


def test_marketplace_detects_tamper():
    mkt = Marketplace(signing_key="k")
    p = mkt.publish(name="x", version="1", rule_ids=["a"], certified=True)
    import dataclasses

    tampered = dataclasses.replace(p, certified=True, name="evil")
    assert not mkt.verify(tampered)


# --- 492 disclosure / CVE ----------------------------------------------------


def test_disclosure_lifecycle():
    d = open_disclosure(advisory_id="PA-1", component="core", severity="high")
    assert d.state == DisclosureState.REPORTED
    d2 = coordinate_cve(d, cve_id="CVE-2026-0001")
    assert d2.cve_id == "CVE-2026-0001"
    assert d2.state == DisclosureState.COORDINATED


def test_disclosure_publish_is_terminal():
    d = open_disclosure(advisory_id="PA-2", component="cli", severity="low")
    for _ in range(10):
        d = d.advance()
    assert d.state == DisclosureState.PUBLISHED


# --- 493 curriculum ----------------------------------------------------------


def test_labs_have_known_answers():
    labs = curriculum_labs()
    assert len(labs) >= 2
    forgeable = next(l for l in labs if l.expected_forgeable)
    sealed = next(l for l in labs if not l.expected_forgeable)
    assert grade_lab(forgeable, True)
    assert not grade_lab(forgeable, False)
    assert grade_lab(sealed, False)


def test_lab_grading_against_real_analyzer():
    from promptabi.adoption_tooling import verify_chat_template

    for lab in curriculum_labs():
        findings = verify_chat_template(dict(lab.starter_config))
        forgeable = any(f.forgeable for f in findings)
        assert forgeable == lab.expected_forgeable


# --- 494 case studies --------------------------------------------------------


def test_case_study_reproducible():
    log = [{"config": "a", "forgeable": True}, {"config": "b", "forgeable": False}]
    cs = record_case_study(adopter="acme", verification_log=log)
    assert cs.bugs_caught == 1
    assert cs.configs_verified == 2
    assert reproduce_case_study(cs, log)


def test_case_study_detects_divergence():
    log = [{"config": "a", "forgeable": True}]
    cs = record_case_study(adopter="acme", verification_log=log)
    assert not reproduce_case_study(cs, [{"config": "a", "forgeable": False}])


# --- 495/496 governance ------------------------------------------------------


def test_steering_committee_and_roadmap():
    sc = steering_committee()
    assert sc["seats"] == 7
    rm = technical_roadmap()
    assert all(year >= 2026 for year, _ in rm)


def test_governance_neutrality():
    gov = governance_model()
    assert "neutrality_guarantee" in gov


# --- 497 i18n catalogs -------------------------------------------------------


def test_all_catalogs_complete():
    for locale in available_locales():
        report = verify_catalog(locale)
        assert report.complete, f"{locale} missing {report.missing_keys}"


def test_catalog_covers_all_keys():
    assert len(DIAGNOSTIC_MESSAGE_KEYS) >= 5
    for key in DIAGNOSTIC_MESSAGE_KEYS:
        assert translate(key, locale="es") != ""
        assert translate(key, locale="ja") != ""


def test_translate_falls_back_to_english():
    assert translate("role-boundary-forgeable", locale="xx").startswith("A role")


# --- 498 releases ------------------------------------------------------------


def test_release_signed_and_verifiable():
    r = cut_release(version="1.1.0", quarter="2027Q1", signing_key="sk")
    assert verify_release(r, signing_key="sk")
    assert not verify_release(r, signing_key="wrong")
    assert any(c["component"] == "z3-solver" for c in r.sbom)


# --- 499 impact --------------------------------------------------------------


def test_aggregate_impact():
    log1 = [{"forgeable": True}, {"forgeable": True}, {"forgeable": False}]
    log2 = [{"forgeable": True}]
    cs1 = record_case_study(adopter="a", verification_log=log1)
    cs2 = record_case_study(adopter="b", verification_log=log2)
    report = aggregate_impact([cs1, cs2])
    assert report.bugs_prevented == 3
    assert report.adopters == 2
    assert report.configs_verified == 4


# --- 500 milestones ----------------------------------------------------------


def test_milestone_report_achieved():
    report = milestone_report(stars=1200, best_paper=True)
    assert report.all_achieved
    d = report.to_dict()
    assert d["all_achieved"] is True


def test_milestone_report_incomplete():
    report = milestone_report(stars=10, best_paper=False)
    assert not report.all_achieved
