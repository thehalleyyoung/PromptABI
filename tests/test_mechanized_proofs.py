import json

import promptabi
from promptabi.cli import main
from promptabi.mechanized_proofs import (
    MECHANIZED_PROOF_EXPERIMENT_VERSION,
    render_mechanized_proof_experiments_text,
    run_mechanized_proof_experiments,
)


def test_mechanized_proof_experiments_pass_and_cover_core_fragments() -> None:
    report = run_mechanized_proof_experiments()
    by_id = {experiment.experiment_id: experiment for experiment in report.experiments}

    assert report.passed
    assert report.experiment_count == 4
    assert report.check_count >= 20
    assert set(by_id) == {
        "dfa-de-morgan-bounded",
        "dfa-minimization-language-preservation",
        "finite-contract-sat-assignment",
        "finite-contract-unsat-core",
    }
    assert by_id["finite-contract-sat-assignment"].to_dict()["artifacts"]["assignment"]["enabled"] is True
    assert by_id["finite-contract-unsat-core"].to_dict()["artifacts"]["unsat_core"] == [
        "safe-required",
        "unsafe-required",
    ]
    assert any(
        check.name == "bounded-language-equivalence"
        for check in by_id["dfa-de-morgan-bounded"].checks
    )


def test_mechanized_proof_experiment_renderers_and_public_api_are_stable() -> None:
    report = run_mechanized_proof_experiments()
    text = render_mechanized_proof_experiments_text(report)
    payload = json.loads(promptabi.mechanized_proof_experiments(output_format="json"))

    assert f"PromptABI mechanized proof experiments ({MECHANIZED_PROOF_EXPERIMENT_VERSION})" in text
    assert "dfa-de-morgan-bounded: PASS" in text
    assert payload["passed"] is True
    assert payload["experiment_count"] == report.experiment_count
    assert {item["experiment_id"] for item in payload["experiments"]} == {
        experiment.experiment_id for experiment in report.experiments
    }


def test_mechanized_proof_experiment_cli_outputs_json(capsys) -> None:
    exit_code = main(["proofs", "--experiments", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert payload["version"] == MECHANIZED_PROOF_EXPERIMENT_VERSION
