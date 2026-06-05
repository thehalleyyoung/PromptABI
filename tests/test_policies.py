import json
from datetime import date
from hashlib import sha256
from pathlib import Path

from promptabi.cli import main
from promptabi.config import load_config
from promptabi.diagnostics import CheckMode, Diagnostic, DiagnosticSeverity
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


def test_org_policy_pack_verifies_required_checks_solver_privacy_and_approved_fixtures(tmp_path: Path) -> None:
    fixture = tmp_path / "provider-fixture.json"
    fixture.write_text('{"provider":"openai","request":{"model":"gpt-test"}}', encoding="utf-8")
    fixture_digest = sha256(fixture.read_bytes()).hexdigest()
    policy_pack = tmp_path / "org.policy.json"
    policy_pack.write_text(
        json.dumps(
            {
                "severity_threshold": "error",
                "required_checks": ["repository-skeleton", "enterprise-readiness"],
                "supported_fragments": {"repository-skeleton": ["heuristic"]},
                "max_solver_timeout_ms": 1000,
                "require_strict_no_network": True,
                "forbid_local_usage_summary": True,
                "approved_provider_fixtures": [{"sha256": fixture_digest}],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "org-pack-ok",
                "checks": ["repository-skeleton", "enterprise-readiness"],
                "enterprise": {
                    "strict_no_network": True,
                    "policy_packs": [{"name": "org", "path": policy_pack.name}],
                    "internal_provider_fixtures": [
                        {"name": "openai-internal", "path": fixture.name, "sha256": fixture_digest}
                    ],
                    "solver_sandbox": {
                        "enabled": True,
                        "timeout_ms": 500,
                        "max_memory_mb": 256,
                        "allow_network": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config_path).run()

    assert result.ok is True
    assert "policy-pack-verified" in {diagnostic.rule_id for diagnostic in result.diagnostics}
    assert load_config(config_path).policy.org_policy.required_checks == ("enterprise-readiness", "repository-skeleton")


def test_org_policy_pack_reports_missing_checks_network_timeout_fragment_and_fixture_violations(tmp_path: Path) -> None:
    fixture = tmp_path / "provider-fixture.json"
    fixture.write_text('{"provider":"openai","request":{"model":"gpt-test"}}', encoding="utf-8")
    approved_digest = "0" * 64
    policy_pack = tmp_path / "org.policy.json"
    policy_pack.write_text(
        json.dumps(
            {
                "required_checks": ["enterprise-readiness", "static-contracts"],
                "supported_fragments": {"repository-skeleton": ["sound"]},
                "max_solver_timeout_ms": 100,
                "require_strict_no_network": True,
                "approved_provider_fixture_sha256": [approved_digest],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "org-pack-bad",
                "checks": ["repository-skeleton"],
                "enterprise": {
                    "strict_no_network": False,
                    "policy_packs": [{"name": "org", "path": policy_pack.name}],
                    "internal_provider_fixtures": [{"name": "openai-internal", "path": fixture.name}],
                    "solver_sandbox": {"enabled": True, "timeout_ms": 250, "max_memory_mb": 256},
                },
            }
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config_path).run()
    rule_ids = {diagnostic.rule_id for diagnostic in result.diagnostics}

    assert result.ok is False
    assert "policy-pack-required-check-missing" in rule_ids
    assert "policy-pack-supported-fragment-violation" in rule_ids
    assert "policy-pack-strict-no-network-required" in rule_ids
    assert "policy-pack-solver-timeout-exceeded" in rule_ids
    assert "policy-pack-provider-fixture-unpinned" in rule_ids


def test_org_policy_pack_merging_keeps_stricter_layered_constraints(tmp_path: Path) -> None:
    first = tmp_path / "first.policy.json"
    first.write_text(
        json.dumps(
            {
                "required_checks": ["enterprise-readiness"],
                "supported_fragments": {"repository-skeleton": ["heuristic", "sound"]},
                "max_solver_timeout_ms": 1000,
            }
        ),
        encoding="utf-8",
    )
    second = tmp_path / "second.policy.json"
    second.write_text(
        json.dumps(
            {
                "required_checks": ["static-contracts"],
                "supported_fragments": {"repository-skeleton": ["sound"]},
                "max_solver_timeout_ms": 250,
                "require_strict_no_network": True,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "org-pack-merge",
                "enterprise": {
                    "policy_packs": [
                        {"name": "first", "path": first.name},
                        {"name": "second", "path": second.name},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    org_policy = load_config(config_path).policy.org_policy

    assert org_policy.required_checks == ("enterprise-readiness", "static-contracts")
    assert org_policy.max_solver_timeout_ms == 250
    assert org_policy.require_strict_no_network is True
    assert dict(org_policy.supported_fragments)["repository-skeleton"] == (CheckMode.SOUND,)


def test_org_policy_pack_privacy_rule_blocks_cli_local_summary(tmp_path: Path, capsys) -> None:
    policy_pack = tmp_path / "org.policy.json"
    policy_pack.write_text('{"forbid_local_usage_summary": true}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "org-pack-privacy",
                "enterprise": {"policy_packs": [{"name": "org", "path": policy_pack.name}]},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--local-summary", str(tmp_path / "usage.jsonl")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "forbids --local-summary" in captured.err
    assert not (tmp_path / "usage.jsonl").exists()
