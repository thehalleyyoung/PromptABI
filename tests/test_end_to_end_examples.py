import json
from pathlib import Path

from promptabi.cli import main


EXAMPLE_ROOT = Path("examples/end-to-end")

SCENARIOS = {
    "tool-calling": {"bug_rules": {"tool-serialization"}, "fixed_rules": set()},
    "structured-output": {
        "bug_rules": {"parser-compatibility-mismatch"},
        "fixed_rules": {"parser-compatibility-agreement"},
    },
    "rag-truncation": {"bug_rules": {"rag-citation-loss", "rag-payload-truncation"}, "fixed_rules": set()},
    "provider-migration": {"bug_rules": {"provider-migration"}, "fixed_rules": set()},
    "training-alignment": {
        "bug_rules": {"static-contract-violation"},
        "fixed_rules": {"static-contract-proved"},
    },
}


def _verify(config: Path, capsys) -> tuple[int, dict[str, object]]:
    exit_code = main(["verify", "--config", str(config), "--format", "json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def _rule_ids(payload: dict[str, object], *, severity: str | None = None) -> set[str]:
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, list)
    result = set()
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, dict)
        if severity is None or diagnostic["severity"] == severity:
            result.add(str(diagnostic["rule_id"]))
    return result


def test_end_to_end_buggy_examples_are_caught_by_intended_checkers(capsys) -> None:
    for scenario, expectation in SCENARIOS.items():
        config = EXAMPLE_ROOT / scenario / "buggy.promptabi.json"
        exit_code, payload = _verify(config, capsys)

        error_rules = _rule_ids(payload, severity="error")
        assert exit_code == 1, scenario
        assert expectation["bug_rules"] <= error_rules, (scenario, error_rules)
        assert "check-unknown" not in error_rules


def test_end_to_end_fixed_examples_verify_without_errors(capsys) -> None:
    for scenario, expectation in SCENARIOS.items():
        config = EXAMPLE_ROOT / scenario / "fixed.promptabi.json"
        exit_code, payload = _verify(config, capsys)

        error_rules = _rule_ids(payload, severity="error")
        all_rules = _rule_ids(payload)
        assert exit_code == 0, (scenario, error_rules)
        assert not error_rules, (scenario, error_rules)
        assert "check-unknown" not in all_rules
        assert expectation["fixed_rules"] <= all_rules, (scenario, all_rules)


def test_end_to_end_examples_document_every_contract_pair() -> None:
    overview = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")

    for scenario in SCENARIOS:
        directory = EXAMPLE_ROOT / scenario
        assert (directory / "README.md").is_file(), scenario
        assert (directory / "buggy_app.py").is_file(), scenario
        assert (directory / "fixed_app.py").is_file(), scenario
        assert (directory / "buggy.promptabi.json").is_file(), scenario
        assert (directory / "fixed.promptabi.json").is_file(), scenario
        assert f"`{scenario}`" in overview

