import json

from promptabi import (
    ArtifactKind,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    PluginRegistry,
    VerificationConfig,
    VerificationSession,
    build_compatibility_matrix,
    compatibility_matrix,
    render_compatibility_matrix,
)
from promptabi.cli import main
from promptabi.compatibility_matrix import CHECK_RULE_IDS
from promptabi.session import CHECK_MODE_CATALOG


def test_compatibility_matrix_covers_every_default_session_check() -> None:
    matrix = build_compatibility_matrix()
    checks = {entry.check for entry in matrix.entries}
    session_checks = set(VerificationSession(VerificationConfig(name="matrix", checks=())).checks)

    assert session_checks.issubset(checks)
    assert {mode for entry in matrix.entries for mode in entry.modes} == set(CheckMode)


def test_compatibility_matrix_aggregates_rule_modes_by_check_name() -> None:
    matrix = build_compatibility_matrix(include_plugins=False)
    by_check = {entry.check: entry for entry in matrix.entries}

    expected_stop_modes = _union_modes(CHECK_RULE_IDS["stop-tokenizer-analysis"])
    assert by_check["stop-tokenizer-analysis"].modes == expected_stop_modes
    assert CheckMode.SOUND in by_check["stop-tokenizer-analysis"].modes
    assert CheckMode.HEURISTIC in by_check["stop-tokenizer-analysis"].modes
    assert CheckMode.ABSTAINING in by_check["stop-tokenizer-analysis"].modes
    assert by_check["static-contracts"].modes == _union_modes(CHECK_RULE_IDS["static-contracts"])


def test_compatibility_matrix_documents_training_static_contract_coverage() -> None:
    matrix = compatibility_matrix(include_plugins=False)
    by_check = {entry.check: entry for entry in matrix.entries}
    static_surfaces = {surface.key: surface for surface in by_check["static-contracts"].surfaces}

    assert static_surfaces["training:supervised-jsonl"].artifact_kind is ArtifactKind.TRAINING_MANIFEST
    assert "target-role alignment" in static_surfaces["training:supervised-jsonl"].notes
    assert static_surfaces["training:loss-masks"].status == "covered"
    assert "loss-mask contract" in static_surfaces["training:loss-masks"].notes
    assert static_surfaces["training:packed-datasets"].status == "bounded"
    assert "preserved packing boundaries" in static_surfaces["training:packed-datasets"].notes
    assert static_surfaces["training:preference-pairs"].status == "bounded"
    assert "chosen/rejected pairs" in static_surfaces["training:preference-pairs"].notes


def test_compatibility_matrix_includes_plugin_checks_and_can_exclude_them() -> None:
    registry = PluginRegistry()

    def plugin_check(context):
        del context
        yield Diagnostic(
            rule_id="plugin-provider-contract",
            severity=DiagnosticSeverity.INFO,
            message="plugin provider contract ran",
        )

    registry.register_check(
        "plugin-provider-contract",
        plugin_check,
        artifact_kinds=(ArtifactKind.PROVIDER_CONFIG,),
        modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        plugin="tests",
    )

    with_plugins = build_compatibility_matrix(plugin_registry=registry)
    without_plugins = build_compatibility_matrix(plugin_registry=registry, include_plugins=False)
    plugin_entry = next(entry for entry in with_plugins.entries if entry.check == "plugin-provider-contract")

    assert plugin_entry.source == "plugin"
    assert plugin_entry.modes == (CheckMode.COMPLETE, CheckMode.SOUND)
    assert plugin_entry.artifact_kinds == (ArtifactKind.PROVIDER_CONFIG,)
    assert "plugin-provider-contract" not in {entry.check for entry in without_plugins.entries}


def test_render_compatibility_matrix_api_and_cli(capsys) -> None:
    text = render_compatibility_matrix(output_format="text", include_plugins=False)
    assert text.startswith("PromptABI compatibility matrix")
    assert "static-contracts [abstaining,bounded,sound,z3-backed-smt]" in text

    exit_code = main(["matrix", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["mode_descriptions"]["z3-backed-smt"].startswith("The check lowers")
    assert any(entry["check"] == "role-boundary-nonforgeability" for entry in payload["entries"])
    assert any(surface["axis"] == "provider" and surface["name"] == "openai" for surface in payload["surfaces"])
    assert captured.err == ""


def _union_modes(rule_ids: tuple[str, ...]) -> tuple[CheckMode, ...]:
    modes = {mode for rule_id in rule_ids for mode in CHECK_MODE_CATALOG[rule_id]}
    return tuple(sorted(modes, key=lambda mode: mode.value))
