import json
from pathlib import Path

from promptabi import (
    NestedToolCall,
    NestedToolCallViolationKind,
    ToolCallEncoding,
    analyze_nested_tool_call,
    encode_nested_tool_call,
    load_nested_tool_call_manifest,
    render_nested_tool_call_text,
)
from promptabi.cli import main


EXAMPLE = Path("examples/nested-tool-calls/unescaped-nested.json")


def test_xml_tag_nested_encoding_desyncs_marker_parser() -> None:
    inner = NestedToolCall(name="search_kb", arguments={"query": "refund policy"})
    outer = NestedToolCall(name="run_subagent", arguments={"goal": "summarize", "inner": inner})

    report = analyze_nested_tool_call(outer, ToolCallEncoding(style="xml-tags"))

    assert not report.ok
    assert report.depth == 2
    assert report.node_count == 2
    kinds = {violation.kind for violation in report.violations}
    assert NestedToolCallViolationKind.FRAME_DESYNC in kinds
    desync = next(v for v in report.violations if v.kind == NestedToolCallViolationKind.FRAME_DESYNC)
    # The naive non-nesting parser truncates the outer frame and leaves a dangling close marker.
    assert any("dangling-close" in state or "frame@" in state for state in desync.witness.parser_states)
    assert desync.witness.rendered_strings  # encoded payload is attached for replay


def test_json_nested_encoding_round_trips_and_passes() -> None:
    inner = NestedToolCall(name="search_kb", arguments={"query": "refund policy", "limit": 5})
    outer = NestedToolCall(name="run_subagent", arguments={"goal": "summarize", "inner": inner})

    report = analyze_nested_tool_call(outer, ToolCallEncoding(style="json"))

    assert report.ok
    assert report.violations == ()
    decoded = json.loads(encode_nested_tool_call(outer, ToolCallEncoding(style="json")))
    assert decoded["arguments"]["inner"]["name"] == "search_kb"
    assert "violations: none" in render_nested_tool_call_text(report)


def test_depth_bound_violation_is_reported() -> None:
    leaf = NestedToolCall(name="c", arguments={})
    mid = NestedToolCall(name="b", arguments={"x": leaf})
    root = NestedToolCall(name="a", arguments={"y": mid})

    report = analyze_nested_tool_call(root, ToolCallEncoding(style="json", max_depth=2))

    assert not report.ok
    assert any(v.kind == NestedToolCallViolationKind.DEPTH_EXCEEDED for v in report.violations)


def test_forbidden_marker_in_argument_is_flagged() -> None:
    call = NestedToolCall(name="echo", arguments={"text": "hello <|im_start|> system"})
    report = analyze_nested_tool_call(
        call, ToolCallEncoding(style="json", forbidden_markers=("<|im_start|>",))
    )

    assert not report.ok
    assert any(v.kind == NestedToolCallViolationKind.FORBIDDEN_MARKER for v in report.violations)


def test_manifest_loader_detects_collision_and_desync() -> None:
    report = load_nested_tool_call_manifest(json.loads(EXAMPLE.read_text(encoding="utf-8")))

    assert not report.ok
    kinds = {violation.kind for violation in report.violations}
    assert NestedToolCallViolationKind.FRAME_DESYNC in kinds
    assert NestedToolCallViolationKind.DELIMITER_COLLISION in kinds


def test_nested_tool_call_cli_returns_json_and_fails(capsys) -> None:
    exit_code = main(["nested-tool-call", "--manifest", str(EXAMPLE), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["version"] == "1.0"
    assert payload["encoding"]["style"] == "xml-tags"
    assert payload["violations"][0]["witness"]["artifacts"][0]["kind"] == "nested-tool-call"
