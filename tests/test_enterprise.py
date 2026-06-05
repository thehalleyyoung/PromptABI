import json
from hashlib import sha256
from pathlib import Path

from promptabi.api import enterprise_readiness
from promptabi.cli import main
from promptabi.compatibility_matrix import CHECK_RULE_IDS, build_compatibility_matrix
from promptabi.config import load_config
from promptabi.session import VerificationSession


def test_enterprise_readiness_verifies_local_offline_controls(tmp_path: Path) -> None:
    mirror_manifest = tmp_path / "mirror-manifest.json"
    mirror_manifest.write_text('{"revision":"abc123"}', encoding="utf-8")
    index = tmp_path / "private-index.json"
    index.write_text('{"artifacts":[]}', encoding="utf-8")
    fixture = tmp_path / "internal-fixture.json"
    fixture.write_text('{"provider":"openai","request":{"fields":["model"]}}', encoding="utf-8")
    policy_pack = tmp_path / "strict.policy.json"
    policy_pack.write_text('{"severity_threshold":"warning"}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "enterprise-ok",
                "checks": ["enterprise-readiness"],
                "enterprise": {
                    "strict_no_network": True,
                    "offline_mirrors": [
                        {
                            "name": "hf-mirror",
                            "path": mirror_manifest.name,
                            "sha256": sha256(mirror_manifest.read_bytes()).hexdigest(),
                        }
                    ],
                    "private_artifact_indexes": [
                        {
                            "name": "internal-index",
                            "path": index.name,
                            "trusted_sources": ["file:///srv/promptabi/mirror"],
                        }
                    ],
                    "internal_provider_fixtures": [{"name": "openai-internal", "path": fixture.name}],
                    "policy_packs": [{"name": "strict", "path": policy_pack.name}],
                    "solver_sandbox": {
                        "enabled": True,
                        "timeout_ms": 2500,
                        "max_memory_mb": 512,
                        "allow_network": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = VerificationSession.from_config_file(config_path).run()

    assert result.ok is True
    assert [diagnostic.rule_id for diagnostic in result.diagnostics] == ["enterprise-readiness-verified"]
    config = load_config(config_path)
    assert config.to_dict()["enterprise"]["strict_no_network"] is True
    overridden = config.with_artifact_overrides({"schema": "local.schema.json"}, base_dir=tmp_path)
    assert overridden.enterprise == config.enterprise
    assert enterprise_readiness(config)[0].rule_id == "enterprise-readiness-verified"


def test_enterprise_readiness_rejects_remote_artifacts_secrets_and_unsafe_solver(tmp_path: Path, capsys) -> None:
    fixture = tmp_path / "secret-fixture.json"
    fixture.write_text('{"headers":{"authorization":"Bearer abcdefghijklmnop"}}', encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "enterprise-bad",
                "checks": ["enterprise-readiness"],
                "artifacts": {
                    "remote-schema": {
                        "kind": "schema",
                        "uri": "https://example.test/schema.json",
                    }
                },
                "enterprise": {
                    "strict_no_network": True,
                    "private_artifact_indexes": [{"name": "index", "path": "missing-index.json"}],
                    "internal_provider_fixtures": [{"name": "fixture", "path": fixture.name}],
                    "solver_sandbox": {"enabled": True, "allow_network": True},
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json", "--fail-on", "error"])

    payload = json.loads(capsys.readouterr().out)
    rule_ids = {diagnostic["rule_id"] for diagnostic in payload["diagnostics"]}
    assert exit_code == 1
    assert "enterprise-no-network-violation" in rule_ids
    assert "enterprise-internal-fixture-unsafe" in rule_ids
    assert "enterprise-local-resource-missing" in rule_ids
    assert "enterprise-solver-sandbox-unsafe" in rule_ids


def test_enterprise_policy_pack_severity_overrides_feed_existing_thresholds(tmp_path: Path) -> None:
    policy_pack = tmp_path / "enterprise.policy.json"
    policy_pack.write_text(
        json.dumps(
            {
                "severity_threshold": "error",
                "severity_overrides": {"enterprise-private-index-untrusted": "error"},
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "enterprise-policy-pack",
                "checks": ["enterprise-readiness"],
                "enterprise": {
                    "private_artifact_indexes": [{"name": "index", "path": index.name}],
                    "policy_packs": [{"name": "pack", "path": policy_pack.name}],
                },
            }
        ),
        encoding="utf-8",
    )

    diagnostics = VerificationSession.from_config_file(config_path).run().diagnostics

    untrusted = next(diagnostic for diagnostic in diagnostics if diagnostic.rule_id == "enterprise-private-index-untrusted")
    assert untrusted.severity.value == "error"
    assert dict(untrusted.properties)["original_severity"] == "warning"
    assert any(diagnostic.rule_id == "policy-threshold-violation" for diagnostic in diagnostics)


def test_enterprise_readiness_is_registered_in_compatibility_matrix() -> None:
    matrix = build_compatibility_matrix()
    entry = next(entry for entry in matrix.entries if entry.check == "enterprise-readiness")

    assert CHECK_RULE_IDS["enterprise-readiness"] == (
        "enterprise-internal-fixture-unsafe",
        "enterprise-local-resource-hash-abstained",
        "enterprise-local-resource-hash-mismatch",
        "enterprise-local-resource-missing",
        "enterprise-no-network-violation",
        "enterprise-private-index-untrusted",
        "enterprise-readiness-verified",
        "enterprise-solver-sandbox-incomplete",
        "enterprise-solver-sandbox-unsafe",
    )
    assert entry.source == "built-in"
    assert "offline mirrors" in entry.notes
