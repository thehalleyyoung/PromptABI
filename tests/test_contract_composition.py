import json

from promptabi.cli import main
from promptabi.contract_composition import ContractLayer, compose_static_contracts
from promptabi.contract_language import parse_static_contract_text


def test_contract_composition_merges_layers_without_weakening_policy() -> None:
    org = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity error applies_to chat-template,tool-definition:
  description "Org floor"
  allowed_roles system,user,assistant,tool
  required_regions system
  forbid_delimiters "<|im_start|>"
  schema tool_call requires name
  stop json stops "</json>" forbid_inside json-string
  invariant required_prompt_tokens <= input_budget_tokens
  assume prompt-pack requires tokenizer.control-token-stable
  guarantee tokenizer provides tokenizer.control-token-stable
""",
        name="org",
        path="org.pabi",
    )
    app = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity warning applies_to chat-template:
  description "App specialization"
  allowed_roles system,user,assistant
  required_regions assistant
  forbid_delimiters "<tool_call>"
  schema tool_call requires arguments
  stop json stops "```" forbid_inside json-string
  assume app-config requires schema.tool-call-json
  guarantee schema provides schema.tool-call-json
""",
        name="app",
        path="app.pabi",
    )

    result = compose_static_contracts(
        (
            (ContractLayer.ORGANIZATION_POLICY, org),
            (ContractLayer.APP_CONFIG, app),
        ),
        name="merged",
    )

    rule = result.artifact.rules[0]
    assert rule.severity == "error"
    assert rule.description == "App specialization"
    assert rule.applies_to == ("chat-template",)
    assert rule.allowed_roles == ("assistant", "system", "user")
    assert rule.required_regions == ("assistant", "system")
    assert rule.forbidden_delimiters == ("<tool_call>", "<|im_start|>")
    assert rule.schema_obligations[0].requires == ("arguments", "name")
    assert rule.stop_policies[0].stops == ("</json>", "```")
    assert rule.assumptions[0].requires == ("schema.tool-call-json",)
    assert rule.assumptions[1].requires == ("tokenizer.control-token-stable",)
    assert rule.guarantees[0].provides == ("schema.tool-call-json",)
    assert rule.guarantees[1].provides == ("tokenizer.control-token-stable",)
    assert [conflict.field for conflict in result.conflicts] == ["severity"]
    assert "app-config attempts to weaken severity" in result.conflicts[0].message


def test_contract_composition_allows_specific_layers_to_strengthen_severity() -> None:
    org = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity warning:
  allowed_roles user,assistant
""",
        name="org",
    )
    app = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity error:
  allowed_roles assistant
""",
        name="app",
    )

    result = compose_static_contracts(
        (
            (ContractLayer.ORGANIZATION_POLICY, org),
            (ContractLayer.APP_CONFIG, app),
        )
    )

    assert result.ok
    assert result.artifact.rules[0].severity == "error"
    assert result.artifact.rules[0].allowed_roles == ("assistant",)


def test_contract_composition_reports_same_layer_and_empty_role_intersection() -> None:
    pack_a = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule shared type interface severity warning:
  allowed_roles assistant
""",
        name="pack-a",
    )
    pack_b = parse_static_contract_text(
        """\
contract promptabi.contract/v1

rule shared type interface severity warning:
  allowed_roles tool
""",
        name="pack-b",
    )

    result = compose_static_contracts(
        (
            (ContractLayer.PROMPT_PACK, pack_a),
            (ContractLayer.PROMPT_PACK, pack_b),
        )
    )

    fields = {conflict.field for conflict in result.conflicts}
    assert "same-layer" in fields
    assert "allowed_roles" in fields
    assert result.artifact.rules[0].allowed_roles == ()


def test_contract_compose_cli_uses_real_pabi_files_and_source_spans(tmp_path, capsys) -> None:
    org_path = tmp_path / "org.pabi"
    app_path = tmp_path / "app.pabi"
    org_path.write_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity error:
  allowed_roles system,user,assistant
""",
        encoding="utf-8",
    )
    app_path.write_text(
        """\
contract promptabi.contract/v1

rule boundary type interface severity warning:
  allowed_roles user,assistant
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "contract",
            "compose",
            "--contract",
            f"organization-policy={org_path}",
            "--contract",
            f"app-config={app_path}",
            "--format",
            "json",
            "--fail-on-conflict",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["conflicts"][0]["field"] == "severity"
    assert payload["conflicts"][0]["left_span"]["path"] == str(org_path)
    assert payload["contract"]["rules"][0]["allowed_roles"] == ["assistant", "user"]

    exit_code = main(
        [
            "contract",
            "compose",
            "--contract",
            f"organization-policy={org_path}",
            "--contract",
            f"app-config={app_path}",
            "--format",
            "pabi",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rule boundary type interface severity error:" in captured.out
    assert "allowed_roles assistant,user" in captured.out
