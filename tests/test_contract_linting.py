import json

from promptabi.cli import main
from promptabi.contract_language import parse_static_contract_text
from promptabi.contract_linting import lint_static_contract, render_contract_lint_text


def test_contract_lint_flags_impossible_vacuous_unsupported_and_contradictory_rules() -> None:
    contract = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule impossible type future-check severity error applies_to neural-weights:
  allowed_roles assistant
  required_regions system
  invariant required_prompt_tokens <= 10
  invariant required_prompt_tokens > 10

rule vacuous type interface severity warning:
  description "No finite obligation"
""",
        name="linted",
        path="linted.pabi",
    )

    report = lint_static_contract(contract)

    codes = {finding.code for finding in report.findings}
    assert {
        "impossible-rule",
        "vacuous-guarantee",
        "unsupported-fragment",
        "contradictory-policy",
    } <= codes
    assert report.error_count == 2
    assert report.warning_count == 3
    text = render_contract_lint_text(report)
    assert "PromptABI contract lint: linted" in text
    assert "linted.pabi:3:1" in text


def test_contract_lint_cli_reports_broad_suppressions_from_real_policy_file(tmp_path, capsys) -> None:
    contract_path = tmp_path / "safe.pabi"
    contract_path.write_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity error applies_to chat-template:
  allowed_roles assistant,user
  required_regions assistant
""",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "suppressions": [
                    {
                        "rule_id": "promptabi.*",
                        "justification": "temporary migration window",
                        "artifact": "boundary",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "contract",
            "lint",
            str(contract_path),
            "--policy-file",
            str(policy_path),
            "--format",
            "json",
            "--fail-on",
            "warning",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["warning_count"] == 1
    assert payload["findings"][0]["code"] == "overly-broad-suppression"
    assert payload["findings"][0]["span"]["path"] == str(policy_path)


def test_contract_lint_cli_passes_clean_contract(capsys) -> None:
    exit_code = main(["contract", "lint", "examples/static-contract-language/app.pabi"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: PASS" in captured.out
    assert captured.err == ""
