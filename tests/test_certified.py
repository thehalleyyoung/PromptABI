import json

import pytest

import promptabi
from promptabi.cli import main
from promptabi.certified import (
    CERTIFIED_KERNEL_VERSION,
    KernelOutcome,
    ProofCertificate,
    ProofClaim,
    ProofKernel,
    all_certified_theorems,
    attach_certificate,
    benchmark_kernel,
    certified_check_families,
    check_family_boundaries,
    extracted_kernel_source,
    formal_semantics_report,
    gate_checks_by_certification,
    role_boundary_soundness_theorem,
    run_certified_verification,
    token_budget_arithmetic_theorem,
    trusted_computing_base_audit,
)


def test_all_certified_theorems_pass_independent_kernel() -> None:
    report = run_certified_verification()
    assert report.passed
    assert report.theorem_count == 7
    assert report.checked_states > 1000
    families = report.certified_families()
    assert set(families) == {
        "role-boundary",
        "tokenizer",
        "stop-policy",
        "token-budget",
        "template-abstract-interpretation",
        "json-schema",
        "multi-agent-handoffs",
    }
    for result in report.results:
        assert result.passed
        assert result.verdict.outcome in {KernelOutcome.PROVED_SAFE, KernelOutcome.WITNESSED}


def test_kernel_reparses_from_serialized_form_only() -> None:
    # Round-trip every certificate through JSON; the kernel must still verify it.
    kernel = ProofKernel()
    for theorem in all_certified_theorems():
        blob = json.dumps(theorem.certificate.to_dict())
        restored = ProofCertificate.from_dict(json.loads(blob))
        verdict = kernel.verify(restored)
        assert verdict.verified, theorem.theorem_id


def test_kernel_refutes_tampered_no_counterexample_obligation() -> None:
    # Drop the sanitizer constraint so a forgery becomes reachable; the kernel
    # must catch that the "no counterexample" claim is now false.
    theorem = role_boundary_soundness_theorem()
    obligation = theorem.certificate.obligation
    tampered_constraints = [
        c for c in obligation["constraints"] if c["name"] != "sanitizer-removes-delimiter"  # type: ignore[index]
    ]
    tampered_obligation = dict(obligation)
    tampered_obligation["constraints"] = tampered_constraints
    bad = ProofCertificate(
        theorem_id="thm-tampered",
        title="tampered",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=tampered_obligation,
    )
    verdict = ProofKernel().verify(bad)
    assert not verdict.verified
    assert verdict.outcome is KernelOutcome.REFUTED
    assert "counterexample" in verdict.reason


def test_kernel_rejects_invalid_unsat_core_minimality() -> None:
    theorem = token_budget_arithmetic_theorem()
    # Add a redundant third constraint to the core that is not needed for unsat.
    obligation = dict(theorem.certificate.obligation)
    constraints = list(obligation["constraints"])  # type: ignore[arg-type]
    constraints.append({"name": "trivially-true", "expression": {"ge": [{"var": "budget"}, {"value": 0}]}})
    obligation["constraints"] = constraints
    bad = ProofCertificate(
        theorem_id="thm-bad-core",
        title="bad core",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation,
        unsat_core=("checker-accepts", "segment-overflows", "trivially-true"),
    )
    verdict = ProofKernel().verify(bad)
    assert not verdict.verified
    assert "not minimal" in verdict.reason


def test_kernel_validates_and_refutes_witnesses() -> None:
    obligation = {
        "name": "demo",
        "variables": [{"name": "x", "type": "int-range", "minimum": 0, "maximum": 5}],
        "constraints": [{"name": "ge3", "expression": {"ge": [{"var": "x"}, {"value": 3}]}}],
    }
    good = ProofCertificate(
        theorem_id="w-good",
        title="witness",
        claim=ProofClaim.WITNESS,
        obligation=obligation,
        witness={"x": 4},
    )
    bad = ProofCertificate(
        theorem_id="w-bad",
        title="witness",
        claim=ProofClaim.WITNESS,
        obligation=obligation,
        witness={"x": 1},
    )
    kernel = ProofKernel()
    assert kernel.verify(good).outcome is KernelOutcome.WITNESSED
    refuted = kernel.verify(bad)
    assert not refuted.verified
    assert refuted.outcome is KernelOutcome.REFUTED


def test_kernel_abstains_above_state_bound() -> None:
    obligation = {
        "name": "huge",
        "variables": [
            {"name": "a", "type": "int-range", "minimum": 0, "maximum": 99},
            {"name": "b", "type": "int-range", "minimum": 0, "maximum": 99},
        ],
        "constraints": [{"name": "c", "expression": {"gt": [{"var": "a"}, {"value": 1000}]}}],
    }
    cert = ProofCertificate(
        theorem_id="huge",
        title="huge",
        claim=ProofClaim.NO_COUNTEREXAMPLE,
        obligation=obligation,
    )
    verdict = ProofKernel(max_model_states=100).verify(cert)
    assert verdict.outcome is KernelOutcome.ABSTAINED
    assert not verdict.verified


def test_proof_carrying_diagnostic_round_trips_and_verifies() -> None:
    theorem = role_boundary_soundness_theorem()
    diag = attach_certificate("PROMPTABI-ROLE-FORGE", "role boundary is non-forgeable", theorem)
    payload = diag.to_dict()
    assert payload["rule_id"] == "PROMPTABI-ROLE-FORGE"
    assert payload["certificate_digest"] == theorem.certificate.digest()
    assert diag.verify().verified


def test_check_family_boundaries_reference_existing_theorems() -> None:
    theorem_ids = {theorem.theorem_id for theorem in all_certified_theorems()}
    boundaries = check_family_boundaries()
    assert len(boundaries) == 7
    for boundary in boundaries:
        assert boundary.sound is True
        assert boundary.abstains_outside_fragment is True
        assert boundary.theorem_id in theorem_ids


def test_certified_gating_filters_unproven_families() -> None:
    certified = certified_check_families()
    gate = gate_checks_by_certification(["role-boundary", "tokenizer", "made-up-family"])
    assert "role-boundary" in gate["allowed"]
    assert gate["gated_out"] == ["made-up-family"]
    assert set(gate["certified"]) == set(certified)


def test_extracted_kernels_are_emitted_and_self_consistent() -> None:
    ocaml = extracted_kernel_source("ocaml")
    rust = extracted_kernel_source("rust")
    assert "check_no_counterexample" in ocaml
    assert "check_no_counterexample" in rust
    with pytest.raises(ValueError):
        extracted_kernel_source("haskell")


def test_benchmark_kernel_reports_every_theorem() -> None:
    results = benchmark_kernel(repeats=2)
    assert {r.theorem_id for r in results} == {t.theorem_id for t in all_certified_theorems()}
    for result in results:
        assert result.python_seconds >= 0.0
        assert result.checked_states >= 1


def test_tcb_audit_and_report_are_honest() -> None:
    audit = trusted_computing_base_audit()
    assert "the production analyzer and all rule implementations" in audit["explicitly_untrusted"]
    assert audit["abstention_boundary_states"] > 0
    report = formal_semantics_report()
    assert "Technical Report" in report
    assert "thm-role-boundary-soundness" in report
    assert "Trusted computing base" in report


def test_public_api_and_cli_surface() -> None:
    payload = json.loads(promptabi.certified_verification(output_format="json"))
    assert payload["passed"] is True
    assert payload["kernel_version"] == CERTIFIED_KERNEL_VERSION

    exit_code = main(["certify", "--format", "json"])
    assert exit_code == 0


def test_cli_report_and_audit_and_extract(capsys) -> None:
    assert main(["certify", "--report"]) == 0
    assert "Technical Report" in capsys.readouterr().out
    assert main(["certify", "--tcb-audit"]) == 0
    assert "tcb-audit" in capsys.readouterr().out
    assert main(["certify", "--extract", "rust"]) == 0
    assert "check_no_counterexample" in capsys.readouterr().out
