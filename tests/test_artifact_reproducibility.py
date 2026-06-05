"""Tests for the reproducibility and artifact-quality suite (roadmap steps 376-390)."""

from __future__ import annotations

import json

import pytest

from promptabi import api
from promptabi.artifact_reproducibility import (
    ARTIFACT_REPRODUCIBILITY_VERSION,
    SUPPORTED_PYTHON_VERSIONS,
    ArtifactReproducibilityReport,
    artifact_evaluation_badges,
    bibliography_bibtex,
    ci_python_matrix_yaml,
    citation_cff,
    continuous_benchmark,
    cross_environment_consistency,
    data_statement_markdown,
    detect_flakiness,
    hermetic_environment_manifest,
    make_reproduce_target,
    property_based_role_soundness,
    render_artifact_reproducibility_json,
    render_artifact_reproducibility_text,
    reproducibility_checklist,
    run_artifact_reproducibility_suite,
    run_mutation_testing,
    signed_release_metadata,
    verify_experiment_reproducibility,
    write_reproducibility_artifacts,
    zenodo_metadata,
)


def test_hermetic_environment_zero_dependencies():
    manifest = hermetic_environment_manifest()
    assert manifest["runtime_dependencies"] == []
    assert manifest["cpu_only"] is True
    assert manifest["network_free"] is True
    assert manifest["python_requires"].startswith(">=3.")


def test_artifact_badges_are_svg():
    badges = artifact_evaluation_badges()
    assert set(badges) == {"available", "functional", "reproduced"}
    for svg in badges.values():
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "artifact" in svg


def test_experiment_reproducibility_is_bit_for_bit():
    check = verify_experiment_reproducibility(scaled_limit=63)
    assert check.reproduced is True
    assert len(check.digests) == 3


def test_zenodo_metadata_has_archival_fields():
    meta = zenodo_metadata()
    assert meta["upload_type"] == "software"
    assert meta["title"]
    assert isinstance(meta.get("keywords"), list)


def test_property_based_role_soundness_holds():
    result = property_based_role_soundness(trials=80, seed=1)
    assert result.trials == 80
    assert result.holds is True
    assert result.falsifying_examples == ()


def test_property_based_is_seed_deterministic():
    a = property_based_role_soundness(trials=40, seed=7)
    b = property_based_role_soundness(trials=40, seed=7)
    assert a.to_dict() == b.to_dict()


def test_mutation_testing_kills_every_mutant():
    result = run_mutation_testing()
    assert result.total > 0
    assert result.killed == result.total
    assert result.score == pytest.approx(1.0)


def test_continuous_benchmark_no_regression():
    alarm = continuous_benchmark(scaled_limit=63)
    assert alarm.regression is False
    assert alarm.golden_digest == alarm.current_digest


def test_make_reproduce_target_present():
    makefile = make_reproduce_target()
    assert "reproduce:" in makefile
    assert "promptabi reproduce" in makefile


def test_cross_environment_consistency():
    result = cross_environment_consistency(scaled_limit=63)
    assert result["consistent"] is True
    assert result["profile_a"] != result["profile_b"]


def test_data_statement_has_content():
    md = data_statement_markdown()
    assert "data" in md.lower()
    assert len(md) > 100


def test_signed_release_semver_and_signature():
    meta = signed_release_metadata()
    assert meta["semver_valid"] is True
    assert len(meta["signature"]) == 64
    assert meta["version"] == ARTIFACT_REPRODUCIBILITY_VERSION


def test_citation_cff_and_bibtex():
    cff = citation_cff()
    assert cff.startswith("cff-version")
    bib = bibliography_bibtex()
    assert bib.strip().startswith("@software")
    assert ARTIFACT_REPRODUCIBILITY_VERSION in bib


def test_ci_matrix_covers_all_python_versions():
    matrix = ci_python_matrix_yaml()
    for version in SUPPORTED_PYTHON_VERSIONS:
        assert version in matrix


def test_flakiness_detector_is_deterministic():
    report = detect_flakiness(runs=3, scaled_limit=42)
    assert report.deterministic is True
    assert report.quarantined == ()


def test_reproducibility_checklist_all_satisfied():
    checklist = reproducibility_checklist()
    assert checklist
    assert all(checklist.values())


def test_suite_passes_all_fifteen_steps():
    report = run_artifact_reproducibility_suite(scaled_limit=42)
    assert isinstance(report, ArtifactReproducibilityReport)
    assert [s.step for s in report.steps] == list(range(376, 391))
    assert report.passed is True
    assert all(s.ok for s in report.steps)


def test_renderers_round_trip():
    report = run_artifact_reproducibility_suite(scaled_limit=42)
    text = render_artifact_reproducibility_text(report)
    assert "PASS" in text
    payload = json.loads(render_artifact_reproducibility_json(report))
    assert payload["version"] == ARTIFACT_REPRODUCIBILITY_VERSION
    assert len(payload["steps"]) == 15


def test_public_api_entrypoint():
    report = api.artifact_reproducibility()
    assert isinstance(report, ArtifactReproducibilityReport)
    text = api.artifact_reproducibility(output_format="text")
    assert isinstance(text, str) and "reproducibility" in text.lower()
    with pytest.raises(ValueError):
        api.artifact_reproducibility(output_format="yaml")


def test_write_reproducibility_artifacts(tmp_path):
    written = write_reproducibility_artifacts(tmp_path)
    assert "CITATION.cff" in written
    assert ".zenodo.json" in written
    assert (tmp_path / "CITATION.cff").read_text().startswith("cff-version")
    json.loads((tmp_path / ".zenodo.json").read_text())
