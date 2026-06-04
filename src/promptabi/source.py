"""Source-map helpers for local PromptABI artifacts."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import SourceSpan


JsonPath = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JsonSourceMap:
    """One-based source spans for JSON values and object keys."""

    path: str
    spans: dict[JsonPath, SourceSpan]

    def span_for(self, path: JsonPath) -> SourceSpan | None:
        return self.spans.get(path)

    def key_span_for(self, path: JsonPath) -> SourceSpan | None:
        return self.spans.get((*path, "@key"))

    def prefixed(self, prefix: JsonPath) -> tuple[tuple[str, SourceSpan], ...]:
        """Return deterministic named spans below ``prefix``."""

        items: list[tuple[str, SourceSpan]] = []
        for path, span in self.spans.items():
            if path[: len(prefix)] != prefix or path == prefix:
                continue
            suffix = ".".join(path[len(prefix) :])
            items.append((suffix, span))
        return tuple(sorted(items, key=lambda item: item[0]))


def build_json_source_map(text: str, path: str | Path) -> JsonSourceMap:
    """Parse JSON text and return spans for every value and object property key."""

    parser = _JsonSpanParser(text=text, path=str(path))
    return parser.parse()


@dataclass(slots=True)
class _JsonSpanParser:
    text: str
    path: str
    index: int = field(init=False, default=0)
    line_starts: tuple[int, ...] = field(init=False)
    spans: dict[JsonPath, SourceSpan] = field(init=False)

    def __post_init__(self) -> None:
        self.line_starts = _line_starts(self.text)
        self.spans: dict[JsonPath, SourceSpan] = {}

    def parse(self) -> JsonSourceMap:
        self._parse_value(())
        self._skip_ws()
        if self.index != len(self.text):
            raise ValueError("unexpected trailing JSON text")
        return JsonSourceMap(path=self.path, spans=dict(self.spans))

    def _parse_value(self, path: JsonPath) -> None:
        self._skip_ws()
        start = self.index
        char = self._peek()
        if char == "{":
            self._parse_object(path)
        elif char == "[":
            self._parse_array(path)
        elif char == '"':
            self._parse_string()
        elif char == "-" or char.isdigit():
            self._parse_number()
        elif self.text.startswith("true", self.index):
            self.index += 4
        elif self.text.startswith("false", self.index):
            self.index += 5
        elif self.text.startswith("null", self.index):
            self.index += 4
        else:
            raise ValueError(f"invalid JSON value at offset {self.index}")
        self.spans[path] = self._span(start, self.index)

    def _parse_object(self, path: JsonPath) -> None:
        self._expect("{")
        self._skip_ws()
        if self._peek() == "}":
            self.index += 1
            return
        while True:
            self._skip_ws()
            key_start = self.index
            key = self._parse_string()
            key_end = self.index
            key_path = (*path, key)
            self.spans[(*key_path, "@key")] = self._span(key_start, key_end)
            self._skip_ws()
            self._expect(":")
            self._parse_value(key_path)
            self._skip_ws()
            char = self._peek()
            if char == "}":
                self.index += 1
                return
            self._expect(",")

    def _parse_array(self, path: JsonPath) -> None:
        self._expect("[")
        self._skip_ws()
        if self._peek() == "]":
            self.index += 1
            return
        item_index = 0
        while True:
            self._parse_value((*path, str(item_index)))
            item_index += 1
            self._skip_ws()
            char = self._peek()
            if char == "]":
                self.index += 1
                return
            self._expect(",")

    def _parse_string(self) -> str:
        start = self.index
        self._expect('"')
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                return json.loads(self.text[start : self.index])
        raise ValueError("unterminated JSON string")

    def _parse_number(self) -> None:
        start = self.index
        allowed = set("0123456789+-.eE")
        while self.index < len(self.text) and self.text[self.index] in allowed:
            self.index += 1
        json.loads(self.text[start : self.index])

    def _skip_ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _peek(self) -> str:
        if self.index >= len(self.text):
            raise ValueError("unexpected end of JSON")
        return self.text[self.index]

    def _expect(self, expected: str) -> None:
        if self._peek() != expected:
            raise ValueError(f"expected {expected!r} at offset {self.index}")
        self.index += 1

    def _span(self, start: int, end: int) -> SourceSpan:
        end = max(start, end - 1)
        start_line, start_column = _line_column(self.line_starts, start)
        end_line, end_column = _line_column(self.line_starts, end)
        return SourceSpan(
            path=self.path,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
        )


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return tuple(starts)


def _line_column(line_starts: tuple[int, ...], offset: int) -> tuple[int, int]:
    line_index = bisect_right(line_starts, offset) - 1
    return line_index + 1, offset - line_starts[line_index] + 1
