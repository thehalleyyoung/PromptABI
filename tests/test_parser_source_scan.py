"""Tests for the real-world parser-source interface-safety scanner."""

from __future__ import annotations

from promptabi.parser_source_scan import (
    PARSER_SOURCE_SCAN_VERSION,
    scan_parser_source,
    render_parser_source_report_json,
    render_parser_source_report_text,
)

BUGGY_SPLIT = '''
class ToolParser:
    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text):
        buffer = current_text
        tool_body, _, rest = buffer.partition("</tool_call>")
        return tool_body
'''

BUGGY_REGEX = '''
import re
class OlmoLike:
    def extract_tool_calls(self, model_output, request):
        match = re.search(r"<function_calls>(.*?)</function_calls>", model_output, re.DOTALL)
        return match.group(1) if match else None
'''

SAFE_RAW_DECODE = '''
import json
class SafeParser:
    def extract_tool_calls(self, model_output, request):
        decoder = json.JSONDecoder()
        idx = model_output.index("<tool_call>") + len("<tool_call>")
        obj, end = decoder.raw_decode(model_output, idx)
        return obj
'''

# Reasoning chain-of-thought boundaries are a different, weaker class and must
# not be reported by the tool-call scanner.
REASONING_THINK = '''
import re
class ReasoningParser:
    def strip_thinking(self, text):
        return text.rsplit("</think>")[-1]
    def find_reasoning(self, model_output):
        return re.finditer(r"<think>(.*?)</think>", model_output)
'''


def test_scanner_flags_naive_closing_sentinel_split_over_buffer() -> None:
    report = scan_parser_source(BUGGY_SPLIT, path="buggy_split.py")

    assert report.version == PARSER_SOURCE_SCAN_VERSION
    assert not report.ok
    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.rule_id == "naive-sentinel-split-over-buffer"
    assert candidate.bug_class == "parser-boundary-confusion"
    assert candidate.sentinel == "</tool_call>"
    # A concrete streaming-parser witness places the sentinel inside a JSON string.
    assert candidate.guarantee == "bounded"
    assert candidate.witness is not None
    assert candidate.witness.monitor == "</tool_call>"
    assert candidate.witness.protected_state.startswith("string")


def test_scanner_flags_greedy_tool_call_capture_regex_with_witness() -> None:
    report = scan_parser_source(BUGGY_REGEX, path="olmo_like.py")

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.rule_id == "greedy-tool-call-capture-regex"
    assert candidate.bug_class == "tool-call-corruption"
    assert candidate.sentinel == "</function_calls>"
    assert candidate.witness is not None


def test_scanner_does_not_flag_incremental_json_decoder() -> None:
    report = scan_parser_source(SAFE_RAW_DECODE, path="safe.py")

    assert report.ok
    assert report.candidates == ()


def test_scanner_ignores_reasoning_chain_of_thought_boundaries() -> None:
    report = scan_parser_source(REASONING_THINK, path="reasoning.py")

    assert report.ok
    assert report.candidates == ()


def test_scanner_reports_syntax_errors_without_crashing() -> None:
    report = scan_parser_source("def broken(:\n    pass\n", path="broken.py")

    assert report.parse_error is not None
    assert report.candidates == ()
    assert not report.ok


def test_renderers_emit_text_and_json() -> None:
    report = scan_parser_source(BUGGY_REGEX, path="olmo_like.py")

    text = render_parser_source_report_text(report)
    assert "CANDIDATE greedy-tool-call-capture-regex" in text
    assert "suggestion:" in text

    payload = render_parser_source_report_json(report)
    assert '"rule_id": "greedy-tool-call-capture-regex"' in payload
    assert '"witness"' in payload
