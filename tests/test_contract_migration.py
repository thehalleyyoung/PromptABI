import json

from promptabi.cli import main
from promptabi.contract_migration import migrate_static_contract_text, render_contract_migration_text


LEGACY_CONTRACT = """\
contract promptabi.contract/v0

rule boundary kind llm-app level warning artifacts chat-template,stop-policy:
  roles system,user,assistant
  requires_regions assistant
  forbidden_tokens "<|im_start|>"
  requires_schema tool_call fields name,arguments
  stop_policy json sequences "</json>","```" inside json-string
  assert required_prompt_tokens <= input_budget_tokens
"""


def test_contract_migration_rewrites_legacy_syntax_and_preserves_obligations() -> None:
    report = migrate_static_contract_text(LEGACY_CONTRACT, name="legacy", path="legacy.pabi")

    assert report.changed is True
    assert report.error_count == 0
    assert "contract promptabi.contract/v1" in report.migrated_text
    assert "rule boundary type llm-app severity warning applies_to chat-template,stop-policy:" in report.migrated_text
    assert "allowed_roles assistant,system,user" in report.migrated_text
    assert "required_regions assistant" in report.migrated_text
    assert 'forbid_delimiters "<|im_start|>"' in report.migrated_text
    assert "schema tool_call requires arguments,name" in report.migrated_text
    assert 'stop json stops "</json>","```" forbid_inside json-string' in report.migrated_text
    assert "invariant migrated-1: required_prompt_tokens <= input_budget_tokens" in report.migrated_text
    assert {edit.backend for edit in report.edits} >= {"automata", "solver", "automata+solver"}

    rendered = render_contract_migration_text(report)
    assert "PromptABI contract migration: legacy" in rendered
    assert "behavior:" in rendered
    assert "SMT witnesses" in rendered


def test_contract_migration_names_multiple_legacy_assertions_stably() -> None:
    report = migrate_static_contract_text(
        """\
contract promptabi.contract/v0

rule budgets kind budget level error:
  assert required_prompt_tokens <= input_budget_tokens
  roles system,user
  assert reserved_tool_tokens < context_window_tokens
""",
        name="assertions",
    )

    assert "invariant migrated-1: required_prompt_tokens <= input_budget_tokens" in report.migrated_text
    assert "invariant migrated-2: reserved_tool_tokens < context_window_tokens" in report.migrated_text


def test_contract_migration_cli_check_write_and_json_report(tmp_path, capsys) -> None:
    contract_path = tmp_path / "legacy.pabi"
    report_path = tmp_path / "migration.json"
    contract_path.write_text(LEGACY_CONTRACT, encoding="utf-8")

    exit_code = main(["contract", "migrate", str(contract_path), "--check"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "needs contract migration" in captured.err

    exit_code = main(
        [
            "contract",
            "migrate",
            str(contract_path),
            "--format",
            "json",
            "--output",
            str(report_path),
            "--write",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["changed"] is True
    assert payload["edit_count"] >= 8
    assert "migrated_text" in payload
    assert contract_path.read_text(encoding="utf-8") == payload["migrated_text"]

    exit_code = main(["contract", "migrate", str(contract_path), "--check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
