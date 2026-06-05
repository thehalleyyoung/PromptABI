import json

from promptabi.cli import main
from promptabi.safe_deployment_cores import (
    derive_safe_deployment_cores,
    render_safe_deployment_cores_json,
    render_safe_deployment_cores_text,
)


def test_safe_deployment_cores_are_minimal_against_real_smt_benchmark() -> None:
    report = derive_safe_deployment_cores()
    certificate = report.certificates[0]

    assert report.ok is True
    assert certificate.case_id == "budget-unsat-survival"
    assert certificate.core == ("required-tokens-exceed-input-budget",)
    assert certificate.removed_constraints == ()
    assert len(certificate.proof_hash) == 64
    assert len(report.manifest_sha256) == 64


def test_safe_deployment_cores_render_json_and_text() -> None:
    report = derive_safe_deployment_cores()
    payload = json.loads(render_safe_deployment_cores_json(report))
    text = render_safe_deployment_cores_text(report)

    assert payload["manifest_version"] == "promptabi.safe-deployment-cores.v1"
    assert payload["ok"] is True
    assert payload["certificate_count"] == 1
    assert payload["certificates"][0]["minimal"] is True
    assert "PromptABI safe deployment unsat cores" in text
    assert "budget-unsat-survival" in text


def test_solver_minimal_unsat_cores_cli(capsys) -> None:
    exit_code = main(["solver", "minimal-unsat-cores", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["certificates"][0]["core"] == ["required-tokens-exceed-input-budget"]
