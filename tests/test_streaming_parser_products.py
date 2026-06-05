import json
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.streaming_parser_products import (
    analyze_streaming_parser_product,
    build_json_boundary_streaming_parser,
    build_substring_monitor,
)


def test_streaming_parser_product_accepts_safe_chunked_json() -> None:
    chunks = ('{"tool": "search", ', '"arguments": {"q": "prompt abi"}}')
    report = analyze_streaming_parser_product(chunks, monitor_literal="</tool_call>")

    assert report.ok
    assert report.replay.complete
    assert report.violations == ()
    assert report.guarantee_mode == "bounded"
    assert promptabi.analyze_streaming_parser_product(chunks).ok
    assert json.loads("".join(chunks))["arguments"]["q"] == "prompt abi"


def test_streaming_parser_product_flags_monitor_inside_json_string_across_chunks() -> None:
    chunks = ('{"arguments": {"q": "hello </to', 'ol_call> world"}}')
    report = analyze_streaming_parser_product(chunks, monitor_literal="</tool_call>")

    assert not report.ok
    assert report.replay.complete
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.monitor == "</tool_call>"
    assert violation.chunk_index == 1
    assert violation.parser_state.startswith("string:")
    assert "</tool_call>" in violation.excerpt


def test_streaming_parser_state_machine_replays_incomplete_and_invalid_streams() -> None:
    parser = build_json_boundary_streaming_parser(alphabet='x{"a": 1', max_depth=2)

    incomplete = parser.replay(('{"a": ',))
    invalid = parser.replay(('x{"a": 1}',))

    assert not incomplete.complete
    assert incomplete.final_state == "outside:1"
    assert invalid.error
    assert invalid.first_error_index == 0


def test_substring_monitor_tracks_overlapping_suffixes() -> None:
    monitor = build_substring_monitor("abab", alphabet="ab")

    assert monitor.accepts_text("abab")
    assert monitor.accepts_text("ababab")
    assert not monitor.accepts_text("aba")


def test_streaming_parser_cli_reports_json_product_and_exit_codes(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "tool.json"
    input_path.write_text('{"arguments": {"q": "hello </tool_call> world"}}', encoding="utf-8")

    unsafe_exit = main(["streaming-parser", "--input", str(input_path), "--monitor", "</tool_call>", "--format", "json"])
    unsafe = json.loads(capsys.readouterr().out)

    safe_exit = main(["streaming-parser", "--chunk", '{"arguments": {"q": "hello"}}', "--monitor", "</tool_call>"])
    safe = capsys.readouterr().out

    assert unsafe_exit == 1
    assert unsafe["violations"][0]["monitor"] == "</tool_call>"
    assert unsafe["replay"]["complete"] is True
    assert safe_exit == 0
    assert "status: PASS" in safe
