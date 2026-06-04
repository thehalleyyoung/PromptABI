import json
from pathlib import Path

from promptabi import ArtifactKind, ArtifactLocation, SchemaArtifact, StopPolicyArtifact
from promptabi.cli import main
from promptabi.stop_overreachability import analyze_stop_overreachability


def test_stop_overreachability_builds_schema_string_witness(tmp_path: Path) -> None:
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["decision", "comment"],
                "properties": {
                    "decision": {"type": "string", "enum": ["call_tool"]},
                    "comment": {"type": "string"},
                },
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="tool-stop",
        location=ArtifactLocation(uri="memory://stop"),
        stop_sequences=("</tool_call>",),
    )
    schema = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="answer-schema",
        location=ArtifactLocation(path=str(schema_path)),
    )

    report = analyze_stop_overreachability(policy, (schema,))

    content = [finding for finding in report.content_findings if finding.region.kind == "json-schema-string"]
    assert content
    finding = content[0]
    parsed = json.loads(finding.valid_output)
    assert parsed["decision"] == "call_tool"
    assert parsed["comment"] == "</tool_call>"
    assert finding.truncated_prefix == finding.valid_output[: finding.firing_offset]
    assert finding.resulting_state == "inside JSON string value at $.comment"
    assert report.abstentions == ()


def test_stop_overreachability_offsets_use_serialized_output_for_escaped_stops() -> None:
    policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="quote-stop",
        location=ArtifactLocation(uri="memory://stop"),
        stop_sequences=('"',),
    )

    report = analyze_stop_overreachability(policy)

    structural = [finding for finding in report.structural_findings if finding.region.kind == "json"]
    assert structural
    finding = structural[0]
    assert finding.valid_output[finding.firing_offset] == '"'
    assert finding.valid_output_prefix == '{"'
    assert finding.resulting_state.startswith("inside nested JSON")


def test_stop_overreachability_cli_reports_tool_argument_and_provider_envelope(
    tmp_path: Path,
    capsys,
) -> None:
    stop_path = tmp_path / "stops.json"
    tools_path = tmp_path / "tools.json"
    config_path = tmp_path / "promptabi.json"
    stop_path.write_text('{"stop": ["</tool_call>"]}', encoding="utf-8")
    tools_path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "refund_user",
                        "parameters": {
                            "type": "object",
                            "required": ["user_id", "reason"],
                            "properties": {
                                "user_id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "name": "stop-overreach-fixture",
                "checks": ["stop-overreachability"],
                "artifacts": {
                    "stops": {
                        "kind": "stop-policy",
                        "path": str(stop_path),
                    },
                    "tools": {
                        "kind": "tool-definition",
                        "path": str(tools_path),
                        "provider": "openai",
                        "tool_names": ["refund_user"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = {diagnostic["rule_id"] for diagnostic in payload["diagnostics"]}
    assert exit_code == 1
    assert "stop-overreach-content" in rule_ids
    assert "stop-overreach-structural" in rule_ids
    content = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "stop-overreach-content"
    ][0]
    assert content["check_modes"] == ["bounded", "sound"]
    assert "refund_user" in json.dumps(content["witness"])
    assert any(
        step["action"] == "record parser state at truncation"
        for step in content["witness"]["steps"]
    )
    assert captured.err == ""
