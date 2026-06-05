"""Tests for the developer experience and ecosystem surfaces (steps 346-360)."""

from __future__ import annotations

import ast
import json

import pytest

from promptabi import devex_ecosystem as api_entry
from promptabi.devex_ecosystem import (
    FRAMEWORK_INTEGRATIONS,
    SUPPORTED_LOCALES,
    SUPPORTED_SDK_LANGUAGES,
    DiagnosticSeverity,
    PolicyProfile,
    apply_profile,
    build_lsp_diagnostics,
    builtin_profiles,
    coverage_badge_svg,
    explain_demo_finding,
    framework_integration_adapters,
    generate_type_stub,
    language_server_initialize_result,
    load_demo_diagnostics,
    localize_finding,
    quickstart_find_bug,
    ranked_autofixes,
    render_devex_ecosystem_json,
    render_devex_ecosystem_text,
    result_json_schemas,
    run_devex_ecosystem,
    run_third_party_plugin,
    sdk_reader_sources,
    tutorial_notebook,
    validate_against_diagnostic_schema,
    vscode_extension_files,
    wasm_playground_html,
)


@pytest.fixture(scope="module")
def diagnostics():
    return load_demo_diagnostics()


@pytest.fixture(scope="module")
def report():
    return run_devex_ecosystem()


def test_demo_corpus_has_real_error(diagnostics):
    assert any(d.severity is DiagnosticSeverity.ERROR for d in diagnostics)


# --- 346 LSP ---------------------------------------------------------------- #


def test_lsp_publish_diagnostics_shape(diagnostics):
    note = build_lsp_diagnostics(diagnostics, uri="file:///x.json")
    assert note["method"] == "textDocument/publishDiagnostics"
    items = note["params"]["diagnostics"]
    assert len(items) == len(diagnostics)
    first = items[0]
    assert first["source"] == "promptabi"
    assert set(first["range"]) == {"start", "end"}
    assert first["severity"] in (1, 2, 3)


def test_language_server_capabilities():
    init = language_server_initialize_result()
    assert init["capabilities"]["diagnosticProvider"]["identifier"] == "promptabi"


# --- 347 VS Code ------------------------------------------------------------ #


def test_vscode_extension_package_json_valid():
    files = vscode_extension_files()
    pkg = json.loads(files["package.json"])
    assert pkg["name"] == "promptabi"
    assert any(c["command"] == "promptabi.verify" for c in pkg["contributes"]["commands"])
    assert "activate" in files["extension.js"]


# --- 348 SDK readers -------------------------------------------------------- #


def test_sdk_readers_cover_four_languages():
    sdk = sdk_reader_sources()
    assert set(sdk) == set(SUPPORTED_SDK_LANGUAGES)
    # Python reader must be importable and round-trip a real diagnostic.
    ns: dict = {}
    exec(compile(sdk["python"], "<py>", "exec"), ns)  # noqa: S102
    payload = load_demo_diagnostics()[0].to_dict()
    obj = ns["Diagnostic"].from_dict(payload)
    assert obj.rule_id == payload["rule_id"]
    # Other languages contain the declared fields and balanced braces.
    for lang in ("typescript", "go", "rust"):
        src = sdk[lang]
        assert "rule_id" in src and "fingerprint" in src
        assert src.count("{") == src.count("}")


# --- 349 explain ------------------------------------------------------------ #


def test_explain_demo_renders_text():
    text = explain_demo_finding()
    assert text.strip()
    assert "rag" in text.lower() or "citation" in text.lower()


# --- 350 autofix ------------------------------------------------------------ #


def test_ranked_autofixes_nonempty(diagnostics):
    fixes = ranked_autofixes(diagnostics)
    assert fixes
    assert all(hasattr(f, "text") for f in fixes)


# --- 351 JSON Schema -------------------------------------------------------- #


def test_json_schemas_are_well_formed(diagnostics):
    schemas = result_json_schemas()
    assert set(schemas) == {"SourceSpan", "Diagnostic", "VerificationResult"}
    for schema in schemas.values():
        assert schema["$schema"].endswith("schema")
        assert schema["type"] == "object"
    error = next(d for d in diagnostics if d.severity is DiagnosticSeverity.ERROR)
    assert validate_against_diagnostic_schema(error)


# --- 352 playground --------------------------------------------------------- #


def test_wasm_playground_html():
    html = wasm_playground_html()
    assert html.startswith("<!doctype html>")
    assert "pyodide" in html and "promptabi" in html


# --- 353 quickstart --------------------------------------------------------- #


def test_quickstart_finds_real_bug_fast():
    qs = quickstart_find_bug()
    assert qs.found_bug
    assert qs.severity == "error"
    assert qs.suggestion
    assert qs.elapsed_seconds < 300


# --- 354 plugin API --------------------------------------------------------- #


def test_third_party_plugin_runs_through_scheduler():
    result = run_third_party_plugin()
    assert result.registered
    assert result.plugin_finding_present
    assert "thirdparty-banned-phrase" in result.emitted_rule_ids


# --- 355 frameworks --------------------------------------------------------- #


def test_framework_adapters_cover_all_and_parse_python():
    adapters = framework_integration_adapters()
    assert set(adapters) == set(FRAMEWORK_INTEGRATIONS)
    for src in adapters.values():
        ast.parse(src)  # all three are valid Python shims
        assert "promptabi" in src


# --- 356 type stub ---------------------------------------------------------- #


def test_type_stub_parses():
    stub = generate_type_stub()
    ast.parse(stub)
    assert "def run_devex_ecosystem" in stub


# --- 357 i18n --------------------------------------------------------------- #


def test_localization_translates_known_keys(diagnostics):
    target = next(
        d for d in diagnostics
        if d.localization_key == "promptabi.diagnostic.rag.citation.loss"
    )
    en = localize_finding(target, "en")
    ja = localize_finding(target, "ja")
    assert en != ja
    assert len(SUPPORTED_LOCALES) == 5
    with pytest.raises(ValueError):
        localize_finding(target, "xx")


# --- 358 notebook ----------------------------------------------------------- #


def test_tutorial_notebook_is_valid_nbformat():
    nb = tutorial_notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
    # Serializable as a real .ipynb document.
    json.dumps(nb)


# --- 359 profiles ----------------------------------------------------------- #


def test_policy_profile_overrides_and_gate(diagnostics):
    lenient = builtin_profiles()["lenient"]
    res = apply_profile(diagnostics, lenient)
    # artifact-unpinned is downgraded to info under the lenient profile.
    assert "artifact-unpinned" not in res.blocking_rule_ids

    strict = PolicyProfile(name="strict", fail_on=DiagnosticSeverity.WARNING)
    res_strict = apply_profile(diagnostics, strict)
    # Errors still block under strict.
    assert res_strict.blocking_rule_ids
    assert not res_strict.gate_passed

    disabling = PolicyProfile(name="d", disabled_rules=("rag-citation-loss",))
    res_disabled = apply_profile(diagnostics, disabling)
    assert all(d.rule_id != "rag-citation-loss" for d in res_disabled.kept)


# --- 360 badge -------------------------------------------------------------- #


def test_coverage_badge_svg():
    svg = coverage_badge_svg(96.0)
    assert svg.startswith("<svg")
    assert "96%" in svg
    assert "#4c1" in svg  # green for >=90
    assert coverage_badge_svg(50.0).count("#e05d44") == 1


# --- aggregate + API -------------------------------------------------------- #


def test_report_passes_all_fifteen_steps(report):
    assert report.passed
    assert len(report.steps) == 15
    assert {s.step for s in report.steps} == set(range(346, 361))
    text = render_devex_ecosystem_text(report)
    for step in range(346, 361):
        assert f"[{step}]" in text
    decoded = json.loads(render_devex_ecosystem_json(report))
    assert decoded["passed"] is True


def test_public_api_entrypoint():
    text = api_entry(output_format="text")
    assert "developer experience" in text.lower()
    obj = api_entry()
    assert obj.passed
    with pytest.raises(ValueError):
        api_entry(output_format="xml")


def test_run_is_deterministic():
    assert run_devex_ecosystem().to_dict() == run_devex_ecosystem().to_dict()
