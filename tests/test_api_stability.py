import json
import warnings

from promptabi import (
    ApiCompatibilityIssueKind,
    ApiStability,
    ApiSymbol,
    DeprecatedApi,
    PluginRegistry,
    PublicApiManifest,
    build_public_api_manifest,
    compare_public_api_manifests,
    deprecated_api,
    public_api_reference,
    render_public_api_manifest_markdown,
)
from promptabi.cli import main


def test_public_api_manifest_marks_downstream_plugin_surface_stable() -> None:
    manifest = build_public_api_manifest()
    symbols = manifest.symbol_map()

    required_plugin_symbols = {
        "ArtifactLoadHook",
        "CheckCallable",
        "CheckContext",
        "Diagnostic",
        "DiagnosticRenderer",
        "LoadedArtifact",
        "PluginRegistry",
        "PluginCapabilityKind",
        "VerificationConfig",
        "VerificationSession",
        "run_verification",
    }

    assert required_plugin_symbols.issubset(symbols)
    assert all(symbols[name].stability is ApiStability.STABLE for name in required_plugin_symbols)
    assert symbols["PluginRegistry"].kind == "class"
    assert "register_check" in dir(PluginRegistry)
    assert manifest.policy_version


def test_public_api_reference_renders_generated_markdown_and_json() -> None:
    markdown = public_api_reference(output_format="markdown")
    payload = json.loads(public_api_reference(output_format="json"))

    assert "# PromptABI public API" in markdown
    assert "Stable embedding and plugin surface" in markdown
    assert "`PluginRegistry`" in markdown
    assert payload["module"] == "promptabi"
    assert any(symbol["name"] == "run_verification" for symbol in payload["symbols"])


def test_public_api_compatibility_detects_removed_stable_symbols() -> None:
    current = build_public_api_manifest()
    stable_symbol = current.symbol_map()["PluginRegistry"]
    baseline = PublicApiManifest(
        policy_version=current.policy_version,
        module=current.module,
        stability_policy=current.stability_policy,
        symbols=(
            stable_symbol,
            ApiSymbol(
                name="RemovedStableHook",
                module="promptabi",
                kind="function",
                stability=ApiStability.STABLE,
                signature="(config)",
            ),
        ),
    )

    issues = compare_public_api_manifests(baseline, current)

    assert issues
    assert issues[0].kind is ApiCompatibilityIssueKind.REMOVED_STABLE_SYMBOL
    assert issues[0].symbol == "RemovedStableHook"


def test_deprecated_api_decorator_warns_and_enters_manifest_metadata() -> None:
    @deprecated_api(
        since="0.9",
        replacement="promptabi.new_symbol",
        remove_in="2.0",
        reason="unit-test coverage",
    )
    def old_symbol(value: str) -> str:
        return value.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_symbol("ok") == "OK"

    assert caught
    metadata = getattr(old_symbol, "__promptabi_deprecated__")
    assert isinstance(metadata, DeprecatedApi)
    assert metadata.to_dict()["replacement"] == "promptabi.new_symbol"


def test_cli_api_docs_generates_markdown_and_json(tmp_path, capsys) -> None:
    output = tmp_path / "public-api.md"
    exit_code = main(["api-docs", "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert "PromptABI public API" in output.read_text(encoding="utf-8")

    exit_code = main(["api-docs", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["policy_version"]
    assert captured.err == ""


def test_downstream_plugin_author_can_import_only_stable_symbols() -> None:
    namespace: dict[str, object] = {}
    exec(
        """
from promptabi import (
    ArtifactKind,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    PluginRegistry,
)

def install(registry: PluginRegistry) -> None:
    def check(context):
        yield Diagnostic(
            rule_id="stable-api-plugin-check",
            severity=DiagnosticSeverity.INFO,
            message=f"checked {context.config.name}",
            check_modes=(CheckMode.HEURISTIC,),
        )

    registry.register_check(
        "stable-api-plugin-check",
        check,
        artifact_kinds=(ArtifactKind.SCHEMA,),
        modes=(CheckMode.HEURISTIC,),
        plugin="stable-api-test",
    )
""",
        namespace,
    )

    registry = PluginRegistry()
    namespace["install"](registry)

    assert "stable-api-plugin-check" in registry.checks
    markdown = render_public_api_manifest_markdown(build_public_api_manifest())
    assert "`PluginRegistry`" in markdown
