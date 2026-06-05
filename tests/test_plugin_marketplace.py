import json
from pathlib import Path

from promptabi import (
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    PLUGIN_CERTIFICATION_SECRET,
    PluginMarketplaceVerificationStatus,
    PluginRegistry,
    build_plugin_marketplace_index,
    plugin_marketplace_index,
    render_plugin_marketplace_json,
    render_plugin_marketplace_text,
)
from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_marketplace_indexes_capabilities_fragments_privacy_and_compatibility() -> None:
    registry = PluginRegistry()
    registry.register_capability(
        "grammar-backend",
        "json-schema-grammar",
        plugin="tests.marketplace.safe",
        version="2.4.0",
        modes=(CheckMode.SOUND, CheckMode.BOUNDED),
        properties={
            "grammar_type": "json-schema",
            "checks": ("grammar-tokenizer-emptiness", "parser-compatibility"),
            "network": "never",
        },
    )

    index = build_plugin_marketplace_index(registry)
    payload = json.loads(render_plugin_marketplace_json(index))
    text = render_plugin_marketplace_text(index)
    package = next(item for item in payload["packages"] if item["name"] == "tests.marketplace.safe")

    assert index.ok
    assert package["verification"]["status"] == "pass"
    assert package["guarantee_modes"] == ["bounded", "sound"]
    assert package["compatibility"]["promptabi"]["min_version"] == "1.0.0"
    assert package["compatibility"]["python"] == ">=3.11"
    assert package["compatibility"]["plugin_versions"] == ["2.4.0"]
    assert package["privacy"]["network_access"] == "never"
    assert package["privacy"]["witness_privacy"] == "hash-only-certified"
    assert "grammar_type:json-schema" in package["supported_fragments"]
    assert "checks:parser-compatibility" in package["supported_fragments"]
    assert "tests.marketplace.safe" in text


def test_plugin_marketplace_attributes_privacy_failures_to_registered_check() -> None:
    registry = PluginRegistry()

    def leaking_check(context):
        yield Diagnostic(
            rule_id="marketplace-leak",
            severity=DiagnosticSeverity.INFO,
            message=f"leaked {PLUGIN_CERTIFICATION_SECRET} from {context.config.name}",
        )

    registry.register_check(
        "marketplace-leak-check",
        leaking_check,
        modes=(CheckMode.HEURISTIC,),
        plugin="tests.marketplace.leaky",
    )

    index = build_plugin_marketplace_index(registry)
    package = next(package for package in index.packages if package.name == "tests.marketplace.leaky")

    assert not index.ok
    assert package.verification_status is PluginMarketplaceVerificationStatus.FAIL
    assert package.privacy.witness_privacy == "failed"
    assert any("secret" in finding for finding in package.privacy.certification_findings)


def test_plugin_marketplace_api_renders_json_for_first_party_plugins() -> None:
    payload = json.loads(plugin_marketplace_index(output_format="json"))
    names = {package["name"] for package in payload["packages"]}

    assert payload["ok"] is True
    assert "promptabi.first_party.z3" in names
    z3 = next(package for package in payload["packages"] if package["name"] == "promptabi.first_party.z3")
    assert "capability:solver-encoding" in z3["supported_fragments"]
    assert "z3-backed-smt" in z3["guarantee_modes"]


def test_plugin_marketplace_example_fixture_matches_live_index() -> None:
    fixture = (REPO_ROOT / "examples/plugin-marketplace/index.json").read_text(encoding="utf-8")

    assert json.loads(fixture) == json.loads(plugin_marketplace_index(output_format="json"))


def test_cli_plugins_marketplace_reports_leaky_plugin_failure(tmp_path, monkeypatch, capsys) -> None:
    module = tmp_path / "promptabi_marketplace_leaky.py"
    module.write_text(
        """
from promptabi import CheckMode, Diagnostic, DiagnosticSeverity, PLUGIN_CERTIFICATION_SECRET


def register_promptabi_plugin(registry):
    def check(context):
        yield Diagnostic(
            rule_id="cli-marketplace-leak",
            severity=DiagnosticSeverity.INFO,
            message=f"leaked {PLUGIN_CERTIFICATION_SECRET}",
        )

    registry.register_check(
        "cli-marketplace-leak",
        check,
        modes=(CheckMode.HEURISTIC,),
        plugin="cli.marketplace.leaky",
    )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = main(["plugins", "marketplace", "--plugin", "promptabi_marketplace_leaky", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    leaky = next(package for package in payload["packages"] if package["name"] == "cli.marketplace.leaky")
    assert exit_code == 1
    assert payload["ok"] is False
    assert leaky["verification"]["status"] == "fail"
    assert leaky["privacy"]["witness_privacy"] == "failed"
    assert captured.err == ""
