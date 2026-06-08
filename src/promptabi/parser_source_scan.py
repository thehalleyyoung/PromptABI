"""Static interface-safety scanner for real-world parser source.

PromptABI's chat-template, stop, and grammar analyzers reason about *artifacts*.
This module adds a complementary capability: scanning the *source code* of the
streaming tool-call / reasoning parsers that real serving stacks (vLLM,
llama.cpp bindings, SGLang, TGI adapters, ...) run, to surface concrete
candidate interface-safety bugs that can then be triaged and reported.

The flagship bug class is **parser boundary confusion in tool-call streaming
parsers**: code that locates a tool-call boundary by naively searching for a
closing sentinel (``</tool_call>``, ``</function>``, ``[/TOOL_CALLS]`` ...) over
a buffer that *also* contains the model's JSON ``arguments``. Because tool
arguments are attacker-influenced (they routinely echo retrieved web/RAG/API
text) and a JSON string can legally contain the literal sentinel, a naive
``buffer.split("</tool_call>")`` / ``re.search(r"<tool_call>(.*?)</tool_call>")``
mis-terminates the call inside attacker data -> truncated / corrupted tool call.

Honesty contract (mirrors ``upstream_bug_campaign``):

* Every finding is a **candidate** with guarantee mode ``heuristic`` -- a
  structural smell, never a silently "confirmed" bug.
* When the implicated sentinel can be shown, by PromptABI's own streaming JSON
  boundary parser, to occur *inside a protected JSON-string state*, the
  candidate carries a concrete, replayable witness (guarantee ``bounded``).
* The scanner only inspects source you point it at; it makes no network calls.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .streaming_parser_products import analyze_streaming_parser_product

PARSER_SOURCE_SCAN_VERSION = 1

# Bug classes this scanner is allowed to assign (subset of the campaign taxonomy).
_BUG_CLASSES = frozenset(
    {
        "parser-boundary-confusion",
        "tool-call-corruption",
    }
)

# String/byte methods that locate or split on a substring boundary.
_BOUNDARY_METHODS = frozenset(
    {"split", "rsplit", "find", "rfind", "index", "rindex", "partition", "rpartition"}
)

# ``re`` functions that scan a string for a pattern.
_RE_FUNCTIONS = frozenset(
    {"split", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn"}
)

# Identifiers whose presence in a function marks it as accumulating streamed
# model output / tool arguments (attacker-influenced data).
_STREAM_CONTEXT_HINTS = (
    "arg",
    "buffer",
    "buf",
    "delta",
    "stream",
    "accumulat",
    "current_text",
    "previous_text",
    "tool_call",
    "toolcall",
    "function_call",
)

# Names that strongly suggest the value being searched is a model-output buffer.
_BUFFER_NAME_HINTS = (
    "text",
    "buffer",
    "buf",
    "content",
    "output",
    "delta",
    "chunk",
    "response",
    "stream",
    "current",
    "previous",
    "accumulat",
)

_KNOWN_TOOL_SENTINELS = frozenset(
    {
        "</tool_call>",
        "<tool_call>",
        "</function>",
        "</function_calls>",
        "</tool_response>",
        "</tools>",
        "[/TOOL_CALLS]",
        "[TOOL_CALLS]",
        "</tool>",
        "<|tool_call|>",
    }
)

# Reasoning chain-of-thought delimiters. Their content is model-authored, not
# attacker-supplied tool data, so boundary confusion here is a *different*,
# weaker class (reasoning-boundary) handled elsewhere; exclude from the
# tool-call detectors to keep this scan high-precision.
_REASONING_SENTINELS = frozenset(
    {
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
        "<reasoning>",
        "</reasoning>",
        "<reason>",
        "</reason>",
    }
)

_CLOSING_TAG_RE = re.compile(r"</[A-Za-z0-9_\-]+>")
_BRACKET_SENTINEL_RE = re.compile(r"\[/?[A-Z_]+\]")
# Non-greedy/greedy capture sitting between two angle/bracket sentinels.
_CAPTURE_BETWEEN_SENTINELS_RE = re.compile(
    r"(</?[A-Za-z0-9_\-]+>|\[/?[A-Z_]+\])\s*\(\s*\.[*+]\??\s*\)\s*(</?[A-Za-z0-9_\-]+>|\[/?[A-Z_]+\])"
)


def _looks_like_tool_sentinel(literal: str) -> bool:
    if literal in _KNOWN_TOOL_SENTINELS:
        return True
    low = literal.lower()
    looks_structural = bool(_CLOSING_TAG_RE.search(literal) or _BRACKET_SENTINEL_RE.search(literal))
    mentions_tool = any(word in low for word in ("tool", "function", "call"))
    return looks_structural and mentions_tool


_KNOWN_CLOSING_SENTINELS = frozenset(
    {"</tool_call>", "</function>", "</function_calls>", "</tool_response>", "</tools>", "</tool>", "[/TOOL_CALLS]"}
)


def _looks_like_closing_sentinel(literal: str) -> bool:
    """A *closing* boundary marker.

    Searching a buffer for the *opening* marker (``<tool_call>``) and then JSON-decoding
    is the safe idiom; the dangerous idiom is locating the *closing* boundary by literal
    search over a buffer that includes the arguments object. Restricting the split/find
    detector to closing sentinels keeps it high-precision.
    """

    if literal in _REASONING_SENTINELS:
        return False
    if literal in _KNOWN_CLOSING_SENTINELS:
        return True
    low = literal.lower()
    is_close_tag = bool(re.fullmatch(r"</[A-Za-z0-9_\-]+>", literal)) or bool(
        re.fullmatch(r"\[/[A-Z_]+\]", literal)
    )
    mentions_tool = any(word in low for word in ("tool", "function", "call"))
    return is_close_tag and mentions_tool


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """1-indexed source location of a candidate."""

    line: int
    column: int
    end_line: int
    end_column: int

    def to_dict(self) -> dict[str, int]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    def __str__(self) -> str:
        return f"{self.line}:{self.column}-{self.end_line}:{self.end_column}"


@dataclass(frozen=True, slots=True)
class ParserSourceWitness:
    """A concrete, replayable streaming-parser witness for a candidate."""

    sentinel: str
    chunks: tuple[str, ...]
    monitor: str
    protected_state: str
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sentinel": self.sentinel,
            "chunks": list(self.chunks),
            "monitor": self.monitor,
            "protected_state": self.protected_state,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class ParserSourceCandidate:
    """A heuristic interface-safety candidate found in parser source."""

    rule_id: str
    bug_class: str
    guarantee: str  # "heuristic" or "bounded" (when a witness is attached)
    symbol: str
    sentinel: str | None
    span: SourceSpan
    evidence: str
    suggestion: str
    witness: ParserSourceWitness | None = None

    def __post_init__(self) -> None:
        if self.bug_class not in _BUG_CLASSES:
            raise ValueError(f"unsupported bug_class: {self.bug_class}")
        if self.guarantee not in {"heuristic", "bounded"}:
            raise ValueError(f"unsupported guarantee: {self.guarantee}")
        if not self.rule_id or not self.symbol or not self.evidence:
            raise ValueError("rule_id, symbol and evidence must be non-empty")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rule_id": self.rule_id,
            "bug_class": self.bug_class,
            "guarantee": self.guarantee,
            "symbol": self.symbol,
            "sentinel": self.sentinel,
            "span": self.span.to_dict(),
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class ParserSourceReport:
    """Result of scanning one parser source file."""

    version: int
    path: str
    functions_scanned: int
    candidates: tuple[ParserSourceCandidate, ...] = ()
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.candidates and self.parse_error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "path": self.path,
            "ok": self.ok,
            "functions_scanned": self.functions_scanned,
            "parse_error": self.parse_error,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class _FunctionScanner(ast.NodeVisitor):
    """Collect sentinel-boundary candidates within a single function body."""

    def __init__(self, symbol: str, source_lines: Sequence[str]) -> None:
        self.symbol = symbol
        self.source_lines = source_lines
        self.candidates: list[ParserSourceCandidate] = []
        self._context_names: set[str] = set()
        self._stream_context = False

    def scan(self, node: ast.AST) -> list[ParserSourceCandidate]:
        # First pass: does this function look like it accumulates streamed output?
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                self._context_names.add(sub.id.lower())
            elif isinstance(sub, ast.Attribute):
                self._context_names.add(sub.attr.lower())
        joined = " ".join(self._context_names)
        self._stream_context = any(hint in joined for hint in _STREAM_CONTEXT_HINTS)
        self.visit(node)
        return self.candidates

    # -- string boundary methods: buffer.split("</tool_call>") etc. -------------
    def visit_Call(self, node: ast.Call) -> None:
        self._check_string_boundary_call(node)
        self._check_re_call(node)
        self.generic_visit(node)

    def _receiver_is_buffer(self, receiver: ast.expr) -> bool:
        name = _receiver_name(receiver)
        if name is None:
            # Computed receiver (e.g. a slice / call result) — treat as buffer-ish
            # only when stream context is present, to stay high-precision.
            return self._stream_context
        low = name.lower()
        return any(hint in low for hint in _BUFFER_NAME_HINTS)

    def _check_string_boundary_call(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _BOUNDARY_METHODS:
            return
        literal = _first_str_arg(node)
        if literal is None or not _looks_like_closing_sentinel(literal):
            return
        if not self._receiver_is_buffer(func.value):
            return
        receiver = _receiver_name(func.value) or "<expr>"
        self._add(
            rule_id="naive-sentinel-split-over-buffer",
            bug_class="parser-boundary-confusion",
            sentinel=literal,
            node=node,
            evidence=(
                f"{receiver}.{func.attr}({literal!r}) locates a tool-call boundary by "
                f"scanning a model-output buffer for a sentinel that tool JSON arguments "
                f"can themselves contain"
            ),
            suggestion=(
                "Delimit tool calls with an incremental JSON decoder (json.JSONDecoder.raw_decode "
                "or a balanced-brace scan) instead of splitting on the literal sentinel, or only "
                "search the region *outside* the arguments object."
            ),
        )

    def _check_re_call(self, node: ast.Call) -> None:
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _RE_FUNCTIONS):
            return
        if not _is_re_module(func.value):
            return
        pattern = _first_str_arg(node)
        if pattern is None:
            return
        if not _CAPTURE_BETWEEN_SENTINELS_RE.search(pattern):
            return
        sentinel = _closing_sentinel_in(pattern)
        if sentinel is None or sentinel in _REASONING_SENTINELS:
            # Reasoning chain-of-thought capture is a separate, weaker class.
            return
        self._add(
            rule_id="greedy-tool-call-capture-regex",
            bug_class="tool-call-corruption",
            sentinel=sentinel,
            node=node,
            evidence=(
                f"re.{func.attr}({pattern!r}) captures tool-call arguments between sentinels with a "
                f"'.*' group; the closing sentinel inside attacker-influenced arguments terminates the "
                f"capture early"
            ),
            suggestion=(
                "Parse the arguments object as JSON rather than capturing it with a regex, so an "
                "embedded sentinel inside a string value cannot truncate the tool call."
            ),
        )

    def _add(
        self,
        *,
        rule_id: str,
        bug_class: str,
        sentinel: str | None,
        node: ast.AST,
        evidence: str,
        suggestion: str,
    ) -> None:
        span = SourceSpan(
            line=getattr(node, "lineno", 1),
            column=getattr(node, "col_offset", 0) + 1,
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            end_column=getattr(node, "end_col_offset", 0) + 1,
        )
        witness = _build_witness(sentinel) if sentinel else None
        self.candidates.append(
            ParserSourceCandidate(
                rule_id=rule_id,
                bug_class=bug_class,
                guarantee="bounded" if witness is not None else "heuristic",
                symbol=self.symbol,
                sentinel=sentinel,
                span=span,
                evidence=evidence,
                suggestion=suggestion,
                witness=witness,
            )
        )


def _receiver_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _first_str_arg(node: ast.Call) -> str | None:
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        # Only the first positional argument matters for these APIs.
        return None
    return None


def _is_re_module(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"re", "regex"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"re", "regex"}
    return False


def _closing_sentinel_in(pattern: str) -> str | None:
    matches = _CLOSING_TAG_RE.findall(pattern)
    if matches:
        return matches[-1]
    bracket = _BRACKET_SENTINEL_RE.findall(pattern)
    if bracket:
        return bracket[-1]
    return None


def _build_witness(sentinel: str) -> ParserSourceWitness | None:
    """Show the sentinel can appear inside a protected JSON-string (arguments) state.

    Constructs a realistic tool-call arguments payload whose string value embeds
    the sentinel, then uses PromptABI's streaming JSON boundary parser to confirm
    the sentinel is matched while the parser is inside a protected string state --
    i.e. exactly where a naive split would wrongly terminate the call.
    """

    # Only angle/bracket-style closing sentinels make sense as a literal monitor.
    if not (_CLOSING_TAG_RE.search(sentinel) or _BRACKET_SENTINEL_RE.search(sentinel)):
        return None
    payload = json.dumps({"arguments": {"query": f"see {sentinel} for details"}})
    try:
        report = analyze_streaming_parser_product([payload], monitor_literal=sentinel)
    except Exception:  # noqa: BLE001 - witness generation is best-effort
        return None
    if not report.violations:
        return None
    violation = report.violations[0]
    return ParserSourceWitness(
        sentinel=sentinel,
        chunks=(payload,),
        monitor=sentinel,
        protected_state=str(violation.parser_state),
        excerpt=violation.excerpt,
    )


def scan_parser_source(source: str, *, path: str = "<source>") -> ParserSourceReport:
    """Scan Python parser source for tool-call boundary-confusion candidates."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ParserSourceReport(
            version=PARSER_SOURCE_SCAN_VERSION,
            path=path,
            functions_scanned=0,
            parse_error=f"{exc.msg} at line {exc.lineno}",
        )

    source_lines = source.splitlines()
    candidates: list[ParserSourceCandidate] = []
    functions_scanned = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions_scanned += 1
            scanner = _FunctionScanner(node.name, source_lines)
            candidates.extend(scanner.scan(node))

    candidates.sort(key=lambda c: (c.span.line, c.span.column, c.rule_id))
    return ParserSourceReport(
        version=PARSER_SOURCE_SCAN_VERSION,
        path=path,
        functions_scanned=functions_scanned,
        candidates=tuple(candidates),
    )


def scan_parser_source_file(path: str | Path) -> ParserSourceReport:
    """Scan a Python parser source file on disk."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return scan_parser_source(text, path=str(file_path))


def render_parser_source_report_json(report: ParserSourceReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_parser_source_report_text(report: ParserSourceReport) -> str:
    lines = [f"PromptABI parser-source scan: {report.path}"]
    if report.parse_error is not None:
        lines.append(f"  parse-error: {report.parse_error}")
        return "\n".join(lines) + "\n"
    lines.append(
        f"  functions scanned: {report.functions_scanned}  candidates: {len(report.candidates)}"
    )
    if not report.candidates:
        lines.append("  no tool-call boundary-confusion candidates found")
        return "\n".join(lines) + "\n"
    for candidate in report.candidates:
        lines.append(
            f"  CANDIDATE {candidate.rule_id} [{candidate.bug_class}, {candidate.guarantee}]"
        )
        lines.append(f"    in {candidate.symbol} at {candidate.span}")
        lines.append(f"    {candidate.evidence}")
        if candidate.witness is not None:
            lines.append(
                f"    witness: monitor {candidate.witness.monitor!r} matches inside protected "
                f"state {candidate.witness.protected_state} | excerpt {candidate.witness.excerpt!r}"
            )
        lines.append(f"    suggestion: {candidate.suggestion}")
    lines.append(
        "  note: candidates are structural smells (heuristic); confirm against the real "
        "parser code path and check for an existing upstream issue before reporting."
    )
    return "\n".join(lines) + "\n"
