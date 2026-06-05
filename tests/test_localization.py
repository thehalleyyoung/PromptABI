import json

import promptabi
from promptabi import CheckMode, Diagnostic, DiagnosticSeverity
from promptabi.cli import main
from promptabi.localization import LocalizationError, render_localized_message


def test_diagnostic_can_carry_localization_metadata_without_changing_message() -> None:
    diagnostic = Diagnostic(
        rule_id="artifact-missing",
        severity=DiagnosticSeverity.ERROR,
        message="artifact {name} does not exist",
        message_id="promptabi.diagnostic.artifact_missing",
        message_args=(("name", "schema"),),
        check_modes=(CheckMode.SOUND,),
    )

    payload = diagnostic.to_dict()
    catalog = promptabi.diagnostic_message_catalog((diagnostic,))

    assert diagnostic.localization_key == "promptabi.diagnostic.artifact_missing"
    assert payload["message"] == "artifact {name} does not exist"
    assert payload["localization"] == {
        "message_id": "promptabi.diagnostic.artifact_missing",
        "default_locale": "en",
        "message_args": {"name": "schema"},
    }
    assert catalog[0].to_dict() == {
        "message_id": "promptabi.diagnostic.artifact_missing",
        "locale": "en",
        "default_message": "artifact {name} does not exist",
        "rule_ids": ["artifact-missing"],
        "severities": ["error"],
        "placeholders": ["name"],
    }
    assert render_localized_message(catalog[0].default_message, {"name": "schema"}) == "artifact schema does not exist"


def test_localization_catalog_rejects_inconsistent_placeholders() -> None:
    diagnostic = Diagnostic(
        rule_id="demo",
        severity=DiagnosticSeverity.WARNING,
        message="field {name} failed",
        message_id="promptabi.diagnostic.demo",
        message_args=(("unused", "value"),),
    )

    try:
        promptabi.build_diagnostic_catalog((diagnostic,))
    except LocalizationError as exc:
        assert "inconsistent placeholders" in str(exc)
    else:
        raise AssertionError("expected placeholder mismatch to fail")


def test_diagnostics_catalog_cli_uses_real_verification_output(capsys) -> None:
    exit_code = main(["diagnostics", "catalog", "--config", "examples/minimal/promptabi.json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["locale"] == "en"
    assert payload["entries"][0]["message_id"] == "promptabi.diagnostic.repository.skeleton"
    assert payload["entries"][0]["rule_ids"] == ["repository-skeleton"]
    assert payload["entries"][0]["severities"] == ["info"]
