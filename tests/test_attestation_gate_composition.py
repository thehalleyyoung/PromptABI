import json

from promptabi import (
    AttestationGateDecision,
    AttestationGateFindingKind,
    compose_attestation_gate_from_config,
    render_attestation_gate_text,
)
from promptabi.cli import main


CONFIG = "examples/attestation-gate/promptabi.json"


def test_attestation_matching_gate_is_admitted() -> None:
    report = compose_attestation_gate_from_config(
        CONFIG,
        attestation_key="shared-release-key",
        gate_key="shared-release-key",
    )

    assert report.admitted
    assert report.decision is AttestationGateDecision.ADMIT
    assert report.findings == ()
    assert report.attestation_bundle_hash == report.gate_bundle_hash
    assert "findings: none" in render_attestation_gate_text(report)


def test_signing_key_mismatch_denies_admission_with_witness() -> None:
    report = compose_attestation_gate_from_config(
        CONFIG,
        attestation_key="running-key",
        gate_key="gate-trusts-other-key",
    )

    assert not report.admitted
    kinds = {finding.kind for finding in report.findings}
    assert AttestationGateFindingKind.SIGNING_KEY_MISMATCH in kinds
    mismatch = next(f for f in report.findings if f.kind == AttestationGateFindingKind.SIGNING_KEY_MISMATCH)
    assert mismatch.witness.artifacts  # replayable witness references both artifacts
    assert mismatch.expected != mismatch.actual


def test_attestation_gate_cli_admits_and_denies(tmp_path, capsys) -> None:
    admit_code = main(
        [
            "attestation-gate",
            "--config",
            CONFIG,
            "--attestation-key",
            "k",
            "--gate-key",
            "k",
            "--format",
            "json",
        ]
    )
    admit_payload = json.loads(capsys.readouterr().out)
    assert admit_code == 0
    assert admit_payload["decision"] == "admit"
    assert admit_payload["version"] == "promptabi.attestation-gate.v1"

    deny_code = main(
        [
            "attestation-gate",
            "--config",
            CONFIG,
            "--attestation-key",
            "k1",
            "--gate-key",
            "k2",
            "--format",
            "json",
        ]
    )
    deny_payload = json.loads(capsys.readouterr().out)
    assert deny_code == 1
    assert deny_payload["decision"] == "deny"
    assert any(f["kind"] == "signing-key-mismatch" for f in deny_payload["findings"])
