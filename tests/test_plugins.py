import json

from promptabi import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    ProviderConfigArtifact,
    LoadedArtifact,
    PluginCapabilityKind,
    PluginRegistry,
    SchemaArtifact,
    VerificationConfig,
    VerificationSession,
    create_first_party_plugin_registry,
    render_result,
)
from promptabi.cli import main


def test_plugin_registry_extends_loaders_checks_renderers_and_capabilities() -> None:
    registry = PluginRegistry()

    def load_plugin_schema(artifact):
        if artifact.location.uri != "plugin://schema":
            return None
        return LoadedArtifact(
            artifact=artifact,
            source_type="plugin-memory-schema",
            pinned=True,
            resolved=True,
            metadata=(("plugin", "unit-test"),),
        )

    def check_plugin_schema(context):
        loaded = context.artifact("schema")
        yield Diagnostic(
            rule_id="plugin-schema-loaded",
            severity=DiagnosticSeverity.INFO,
            message=f"{loaded.source_type} handled {loaded.artifact.name}",
        )

    registry.register_artifact_loader(
        "memory-schema-loader",
        load_plugin_schema,
        artifact_kinds=(ArtifactKind.SCHEMA,),
        uri_schemes=("plugin",),
        plugin="tests",
    )
    registry.register_check(
        "plugin-schema-check",
        check_plugin_schema,
        artifact_kinds=(ArtifactKind.SCHEMA,),
        modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        plugin="tests",
    )
    registry.register_capability(
        PluginCapabilityKind.GRAMMAR_BACKEND,
        "toy-grammar",
        plugin="tests",
        modes=(CheckMode.BOUNDED,),
        properties={"dialect": "unit-test"},
    )
    registry.register_renderer(
        "summary",
        lambda result: f"{result.config.name}:{len(result.diagnostics)}:{result.diagnostics[0].rule_id}\n",
        media_type="text/plain",
        plugin="tests",
    )
    config = VerificationConfig(
        name="plugin-api",
        checks=("plugin-schema-check",),
        artifact_bundle=ArtifactBundle(
            (
                SchemaArtifact(
                    kind=ArtifactKind.SCHEMA,
                    name="schema",
                    location=ArtifactLocation(uri="plugin://schema"),
                ),
            )
        ),
    )

    session = VerificationSession(config, plugin_registry=registry)
    loaded = session.load_artifacts()
    result = session.run()

    assert loaded[0].source_type == "plugin-memory-schema"
    assert result.ok
    assert result.diagnostics[0].rule_id == "plugin-schema-loaded"
    assert result.diagnostics[0].check_modes == (CheckMode.COMPLETE, CheckMode.SOUND)
    assert render_result(result, output_format="summary", plugin_registry=registry) == "plugin-api:1:plugin-schema-loaded\n"
    assert {
        capability.kind
        for capability in registry.capabilities
    } >= {
        PluginCapabilityKind.ARTIFACT_LOADER,
        PluginCapabilityKind.CHECK,
        PluginCapabilityKind.GRAMMAR_BACKEND,
        PluginCapabilityKind.DIAGNOSTIC_RENDERER,
    }


def test_cli_loads_plugin_module_for_checks_and_renderers(tmp_path, monkeypatch, capsys) -> None:
    plugin_module = tmp_path / "promptabi_unit_plugin.py"
    plugin_module.write_text(
        """
from promptabi import CheckMode, Diagnostic, DiagnosticSeverity


def register_promptabi_plugin(registry):
    def check(context):
        yield Diagnostic(
            rule_id="cli-plugin-check",
            severity=DiagnosticSeverity.INFO,
            message=f"plugin saw {context.config.name}",
        )

    registry.register_check(
        "cli-plugin-check",
        check,
        modes=(CheckMode.HEURISTIC,),
        plugin="cli-test",
    )
    registry.register_renderer(
        "one-line",
        lambda result: f"{result.config.name}|{result.diagnostics[0].rule_id}|{result.diagnostics[0].check_modes[0].value}\\n",
        plugin="cli-test",
    )
""",
        encoding="utf-8",
    )
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps({"name": "cli-plugin", "checks": ["cli-plugin-check"], "artifacts": {}}),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--plugin",
            "promptabi_unit_plugin",
            "--format",
            "one-line",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "cli-plugin|cli-plugin-check|heuristic\n"
    assert captured.err == ""


def test_first_party_plugin_pack_registers_required_families() -> None:
    registry = create_first_party_plugin_registry()

    by_name = {capability.name: capability for capability in registry.capabilities}
    families = {
        capability.plugin.removeprefix("promptabi.first_party.")
        for capability in registry.capabilities
        if capability.plugin.startswith("promptabi.first_party")
    }

    assert {
        "huggingface",
        "openai-compatible",
        "vllm",
        "llama.cpp",
        "langchain",
        "llamaindex",
        "outlines",
        "xgrammar",
        "llguidance",
        "pydantic",
        "mcp",
        "z3",
    }.issubset(families)
    assert by_name["z3-finite-contract-encoding"].kind is PluginCapabilityKind.SOLVER_ENCODING
    assert by_name["outlines-grammar-backend"].kind is PluginCapabilityKind.GRAMMAR_BACKEND
    assert "first-party-plugin-coverage" in registry.checks
    assert any(loader.name == "first-party-reference-loader" for loader in registry.artifact_loaders)


def test_default_session_loads_first_party_reference_and_runs_coverage() -> None:
    config = VerificationConfig(
        name="first-party-openai",
        checks=("first-party-plugin-coverage",),
        artifact_bundle=ArtifactBundle(
            (
                ProviderConfigArtifact(
                    kind=ArtifactKind.PROVIDER_CONFIG,
                    name="openai-contract",
                    location=ArtifactLocation(uri="openai://chat-completions?version=2024-06"),
                    provider="openai",
                    api_family="openai-compatible",
                ),
            )
        ),
    )

    session = VerificationSession(config)
    loaded = session.load_artifacts()
    result = session.run()

    assert loaded[0].source_type == "first-party-openai-compatible-reference"
    assert loaded[0].pinned is True
    assert ("first_party_plugin", "openai-compatible") in loaded[0].metadata
    assert result.ok
    assert result.diagnostics[0].rule_id == "first-party-plugin-coverage"
    assert "openai-compatible" in dict(result.diagnostics[0].properties)["loaded_reference_families"]


def test_cli_lists_first_party_plugin_capabilities(capsys) -> None:
    exit_code = main(["plugins", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {capability["name"] for capability in payload["capabilities"]}

    assert exit_code == 0
    assert "huggingface-artifact-reference" in names
    assert "mcp-tool-contract" in names
    assert "z3-finite-contract-encoding" in names
    assert captured.err == ""
