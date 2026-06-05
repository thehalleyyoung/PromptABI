import json

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    LoadedArtifact,
    PLUGIN_CERTIFICATION_SECRET,
    PluginRegistry,
    SchemaArtifact,
    certify_plugin_registry,
    plugin_certification,
    render_plugin_certification_json,
    render_plugin_certification_text,
)
from promptabi.cli import main


def test_plugin_certification_passes_for_privacy_safe_plugin(capsys) -> None:
    registry = PluginRegistry()

    def load_schema(artifact):
        return LoadedArtifact(
            artifact=artifact,
            source_type="certified-loader",
            pinned=True,
            resolved=True,
            metadata=(("privacy", "metadata-only"),),
        )

    def check(context):
        yield Diagnostic(
            rule_id="certified-check",
            severity=DiagnosticSeverity.INFO,
            message=f"checked {context.config.name.split(PLUGIN_CERTIFICATION_SECRET)[0]}",
        )

    registry.register_artifact_loader(
        "certified-loader",
        load_schema,
        artifact_kinds=(ArtifactKind.SCHEMA,),
        uri_schemes=("certified",),
        plugin="tests.certified",
    )
    registry.register_check(
        "certified-check",
        check,
        modes=(CheckMode.SOUND, CheckMode.COMPLETE),
        plugin="tests.certified",
    )
    registry.register_renderer(
        "certified-renderer",
        lambda result: json.dumps(result.to_dict(), sort_keys=True),
        plugin="tests.certified",
    )

    report = certify_plugin_registry(registry)
    payload = json.loads(render_plugin_certification_json(report))
    text = render_plugin_certification_text(report)

    assert report.ok
    assert payload["ok"] is True
    assert payload["summary"]["failed"] == 0
    assert "certified-check" in text
    assert PLUGIN_CERTIFICATION_SECRET not in json.dumps(payload)

    api_payload = plugin_certification(registry, output_format="json")
    assert isinstance(api_payload, str)
    assert json.loads(api_payload)["ok"] is True


def test_plugin_certification_fails_when_check_leaks_private_witness() -> None:
    registry = PluginRegistry()

    def leaking_check(context):
        yield Diagnostic(
            rule_id="leaking-check",
            severity=DiagnosticSeverity.INFO,
            message=f"leaked {context.config.name}",
        )

    registry.register_check(
        "leaking-check",
        leaking_check,
        modes=(CheckMode.HEURISTIC,),
        plugin="tests.leaking",
    )

    report = certify_plugin_registry(registry)
    failures = [case for case in report.cases if case.status.value == "fail"]

    assert not report.ok
    assert any(case.name == "leaking-check" and "leaked secret" in case.message for case in failures)


def test_plugin_certification_fails_when_renderer_leaks_hash_only_payload() -> None:
    registry = PluginRegistry()
    registry.register_renderer(
        "leaking-renderer",
        lambda result: PLUGIN_CERTIFICATION_SECRET,
        plugin="tests.leaking",
    )

    report = certify_plugin_registry(registry)

    assert not report.ok
    assert any(
        case.name == "leaking-renderer" and case.surface == "renderer" and case.status.value == "fail"
        for case in report.cases
    )


def test_plugin_certification_exercises_loader_contracts() -> None:
    registry = PluginRegistry()

    def load_wrong_artifact(artifact):
        return LoadedArtifact(
            artifact=SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name="different",
                location=ArtifactLocation(uri="wrong://artifact"),
            ),
            source_type="wrong-loader",
            pinned=True,
            resolved=True,
        )

    registry.register_artifact_loader(
        "wrong-loader",
        load_wrong_artifact,
        artifact_kinds=(ArtifactKind.SCHEMA,),
        uri_schemes=("wrong",),
        plugin="tests.wrong",
    )

    report = certify_plugin_registry(registry)

    assert not report.ok
    assert any(case.name == "wrong-loader" and "different artifact" in case.message for case in report.cases)


def test_cli_plugins_certify_reports_failures(tmp_path, monkeypatch, capsys) -> None:
    module = tmp_path / "promptabi_leaky_plugin.py"
    module.write_text(
        """
from promptabi import Diagnostic, DiagnosticSeverity, PLUGIN_CERTIFICATION_SECRET


def register_promptabi_plugin(registry):
    def check(context):
        yield Diagnostic(
            rule_id="cli-leaky-check",
            severity=DiagnosticSeverity.INFO,
            message=f"leaked {PLUGIN_CERTIFICATION_SECRET}",
        )

    registry.register_check("cli-leaky-check", check, modes=("heuristic",), plugin="cli-leaky")
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = main(["plugins", "certify", "--plugin", "promptabi_leaky_plugin", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert any(case["name"] == "cli-leaky-check" for case in payload["cases"])
    assert captured.err == ""
