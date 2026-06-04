import json
from datetime import date
from pathlib import Path

from promptabi.cli import main
from promptabi.config import load_config
from promptabi.diagnostics import Diagnostic, DiagnosticSeverity
from promptabi.policies import VerificationPolicy, apply_policy_diagnostics
from promptabi.session import VerificationSession


def test_policy_file_suppresses_accepted_risk_without_weakening_other_output(tmp_path: Path, capsys) -> None:
    config = tmp_path / "promptabi.json"
    policy = tmp_path / "promptabi.policy.json"
    config.write_text(
        json.dumps(
            {
                "name": "suppressed-missing-artifact",
                "artifacts": {"schema": "missing.schema.json"},
            }
        ),
        encoding="utf-8",
    )
    raw_result = VerificationSession(load_config(config)).collect_diagnostics()
    missing = next(diagnostic for diagnostic in raw_result if diagnostic.rule_id == "artifact-missing")
    policy.write_text(
        json.dumps(
            {
                "require_justification": True,
                "require_expiration": True,
                "suppressions": [
                    {
                        "rule_id": "artifact-missing",
                        "fingerprint": missing.fingerprint,
                        "justification": "Fixture intentionally models a known absent upstream schema.",
                        "accepted_risk": "CI may proceed because this repo-only fixture is not deployed.",
                        "owner": "platform",
                        "expires_on": "2999-01-01",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        json.dumps(
            {
                "name": "suppressed-missing-artifact",
                "artifacts": {"schema": "missing.schema.json"},
                "policy_files": [policy.name],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]] == [
        "diagnostic-suppressed",
        "repository-skeleton",
    ]
    suppressed = payload["diagnostics"][0]
    assert suppressed["properties"]["original_rule_id"] == "artifact-missing"
    assert suppressed["properties"]["original_fingerprint"] == missing.fingerprint
    assert suppressed["properties"]["accepted_risk"].startswith("CI may proceed")
    assert payload["ok"] is True


def test_policy_rejects_expired_or_unjustified_suppressions_and_keeps_ci_strict(
    tmp_path: Path,
    capsys,
) -> None:
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "invalid-suppression",
                "artifacts": {"schema": "missing.schema.json"},
                "suppressions": [
                    {
                        "rule_id": "artifact-missing",
                        "artifact": "schema",
                        "justification": "",
                        "expires_on": "2000-01-01",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]]
    assert exit_code == 1
    assert "artifact-missing" in rule_ids
    assert "policy-suppression-invalid" in rule_ids
    invalid = next(diagnostic for diagnostic in payload["diagnostics"] if diagnostic["rule_id"] == "policy-suppression-invalid")
    assert "justification is required" in invalid["message"]
    assert "in the past" in invalid["message"]


def test_policy_threshold_turns_unsuppressed_warnings_into_failures() -> None:
    warning = Diagnostic(
        rule_id="artifact-unpinned",
        severity=DiagnosticSeverity.WARNING,
        message="artifact is not pinned",
    )
    policy = VerificationPolicy(severity_threshold=DiagnosticSeverity.WARNING)

    diagnostics = apply_policy_diagnostics((warning,), policy, today=date(2026, 1, 1))

    assert [diagnostic.rule_id for diagnostic in diagnostics] == [
        "policy-threshold-violation",
        "artifact-unpinned",
    ]
    threshold = diagnostics[0]
    assert threshold.severity is DiagnosticSeverity.ERROR
    assert dict(threshold.properties)["matched_fingerprints"] == [warning.fingerprint]
