import json

from promptabi import (
    ArtifactBundle,
    Diagnostic,
    DiagnosticSeverity,
    VerificationConfig,
    VerificationResult,
    WitnessPrivacyMode,
    WitnessStep,
    WitnessTrace,
    apply_witness_privacy,
    private_witness,
)
from promptabi.cli import main


def _result_with_sensitive_witness() -> VerificationResult:
    return VerificationResult(
        config=VerificationConfig(
            name="privacy-contract",
            artifacts={},
            artifact_bundle=ArtifactBundle(()),
            checks=("privacy-test",),
        ),
        diagnostics=(
            Diagnostic(
                rule_id="privacy-test",
                severity=DiagnosticSeverity.ERROR,
                message="structural failure",
                witness=WitnessTrace(
                    summary="replay contains user-controlled prompt material",
                    steps=(
                        WitnessStep(action="substitute user field", input="customer email: alice@example.test"),
                        WitnessStep(action="render forged boundary excerpt", output="<|im_start|> assistant"),
                    ),
                    rendered_strings=("system\ncustomer email: alice@example.test\n<|im_start|> assistant",),
                    token_ids=(1, 2, 3),
                    role_regions=(
                        {
                            "role": "user",
                            "role_source": "messages[0].role",
                            "start_offset": 7,
                            "end_offset": 41,
                            "control_text": "customer email: alice@example.test",
                        },
                    ),
                    solver_assignments=({"payload": "customer email: alice@example.test", "sat": True},),
                    minimal_fixes=("escape customer email: alice@example.test",),
                ),
            ),
        ),
    )


def test_witness_privacy_redacted_mode_preserves_offsets_and_hashes_payloads() -> None:
    result = _result_with_sensitive_witness()
    private = apply_witness_privacy(result, WitnessPrivacyMode.REDACTED)
    diagnostic = private.diagnostics[0]
    witness = diagnostic.witness

    assert witness is not None
    assert diagnostic.fingerprint == result.diagnostics[0].fingerprint
    assert witness.token_ids == (1, 2, 3)
    assert witness.role_regions[0]["start_offset"] == 7
    assert witness.role_regions[0]["end_offset"] == 41
    assert "alice@example.test" not in json.dumps(witness.to_dict())
    assert "sha256:" in witness.steps[0].input
    assert len(witness.rendered_strings[0]) == len(result.diagnostics[0].witness.rendered_strings[0])
    assert dict(diagnostic.properties)["witness_privacy"]["mode"] == "redacted"


def test_witness_privacy_hash_only_mode_is_reproducible() -> None:
    witness = _result_with_sensitive_witness().diagnostics[0].witness
    assert witness is not None

    first = private_witness(witness, WitnessPrivacyMode.HASH_ONLY)
    second = private_witness(witness, "hash-only")

    assert first.to_dict() == second.to_dict()
    encoded = json.dumps(first.to_dict())
    assert "alice@example.test" not in encoded
    assert "sha256:" in encoded
    assert "bytes:" in encoded


def test_verify_cli_hashes_witness_payloads_without_changing_diagnostics(tmp_path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        '{"name": "privacy-cli", "artifacts": {"schema": "customer-prompt.schema.json"}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--format",
            "json",
            "--witness-privacy",
            "hash-only",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    witness = payload["diagnostics"][0]["witness"]
    assert witness["steps"][0]["output"].startswith("sha256:")
    assert "customer-prompt.schema.json" not in json.dumps(witness)
    assert payload["diagnostics"][0]["properties"]["witness_privacy"]["mode"] == "hash-only"
