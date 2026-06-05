"""Verification of nested tool-call encodings against framing/round-trip soundness.

Many provider and framework tool-calling stacks serialize a tool call by wrapping
a JSON object in literal text markers (for example ``<tool_call> ... </tool_call>``)
and then *re-parse* the stream by splitting on those markers.  When a tool call's
arguments themselves carry an encoded tool call (a *nested* tool call), a naive
marker-based parser sees the inner markers as top-level frame boundaries and
desynchronizes -- a real, exploitable class of tool-call injection.

This module models a bounded nested tool-call tree, encodes it under a declared
encoding scheme, then *replays* a stream parser over the encoded bytes to prove
whether the encoding preserves the original framing and round-trips.  Every
finding ships a replayable :class:`WitnessTrace` whose ``parser_states`` show the
exact open/close events that desynchronized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace


NESTED_TOOL_CALL_VERSION = "1.0"


class NestedToolCallError(ValueError):
    """Raised when a nested tool-call manifest or encoding is structurally invalid."""


class NestedToolCallViolationKind(StrEnum):
    """Concrete, finitely-checkable nested tool-call encoding violations."""

    DEPTH_EXCEEDED = "depth-exceeded"
    DELIMITER_COLLISION = "delimiter-collision"
    FRAME_DESYNC = "frame-desync"
    ROUND_TRIP_MISMATCH = "round-trip-mismatch"
    FORBIDDEN_MARKER = "forbidden-marker"


_SUPPORTED_STYLES = ("xml-tags", "json")


@dataclass(frozen=True, slots=True)
class NestedToolCall:
    """A bounded tool call whose argument values may themselves be tool calls."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise NestedToolCallError("tool call name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise NestedToolCallError("tool call arguments must be an object")
        object.__setattr__(self, "arguments", dict(self.arguments))

    @property
    def depth(self) -> int:
        """Maximum nesting depth of this call (a leaf call has depth 1)."""

        nested = [value for value in self.arguments.values() if isinstance(value, NestedToolCall)]
        nested += [
            item
            for value in self.arguments.values()
            if isinstance(value, list | tuple)
            for item in value
            if isinstance(item, NestedToolCall)
        ]
        if not nested:
            return 1
        return 1 + max(child.depth for child in nested)

    def node_count(self) -> int:
        """Total number of tool-call nodes in the tree rooted at this call."""

        total = 1
        for value in self.arguments.values():
            if isinstance(value, NestedToolCall):
                total += value.node_count()
            elif isinstance(value, list | tuple):
                total += sum(item.node_count() for item in value if isinstance(item, NestedToolCall))
        return total


@dataclass(frozen=True, slots=True)
class ToolCallEncoding:
    """A declared tool-call serialization scheme PromptABI can check for soundness."""

    style: str = "xml-tags"
    open_marker: str = "<tool_call>"
    close_marker: str = "</tool_call>"
    max_depth: int = 4
    forbidden_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.style not in _SUPPORTED_STYLES:
            raise NestedToolCallError(
                f"unsupported encoding style {self.style!r}; expected one of {', '.join(_SUPPORTED_STYLES)}"
            )
        if self.max_depth < 1:
            raise NestedToolCallError("max_depth must be at least 1")
        if self.style == "xml-tags":
            if not self.open_marker or not self.close_marker:
                raise NestedToolCallError("xml-tags encoding requires non-empty open/close markers")
            if self.open_marker == self.close_marker:
                raise NestedToolCallError("open and close markers must differ")
        object.__setattr__(self, "forbidden_markers", tuple(dict.fromkeys(self.forbidden_markers)))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"style": self.style, "max_depth": self.max_depth}
        if self.style == "xml-tags":
            data["open_marker"] = self.open_marker
            data["close_marker"] = self.close_marker
        if self.forbidden_markers:
            data["forbidden_markers"] = list(self.forbidden_markers)
        return data


@dataclass(frozen=True, slots=True)
class NestedToolCallViolation:
    """One nested tool-call encoding violation with a replayable witness."""

    kind: NestedToolCallViolationKind
    message: str
    path: str
    witness: WitnessTrace
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "path": self.path,
            "suggestion": self.suggestion,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NestedToolCallReport:
    """Verification result for a nested tool-call encoding."""

    name: str
    encoding: ToolCallEncoding
    encoded: str
    depth: int
    node_count: int
    violations: tuple[NestedToolCallViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "encoded": self.encoded,
            "encoding": self.encoding.to_dict(),
            "name": self.name,
            "node_count": self.node_count,
            "ok": self.ok,
            "version": NESTED_TOOL_CALL_VERSION,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def encode_nested_tool_call(call: NestedToolCall, encoding: ToolCallEncoding) -> str:
    """Encode a nested tool call under the declared scheme.

    For ``json`` the call is a nested JSON object that round-trips.  For
    ``xml-tags`` the call is wrapped in literal markers and nested calls are
    embedded *as their own encoded marker strings* -- faithfully reproducing the
    naive serializers whose markers collide across nesting depth.
    """

    if encoding.style == "json":
        return json.dumps(_to_json_tree(call), sort_keys=True)
    return _encode_xml(call, encoding)


def analyze_nested_tool_call(
    call: NestedToolCall,
    encoding: ToolCallEncoding | None = None,
    *,
    name: str = "nested-tool-call",
    manifest_path: str = "memory://nested-tool-calls",
) -> NestedToolCallReport:
    """Encode and replay-parse a nested tool call, emitting concrete witnesses."""

    encoding = encoding or ToolCallEncoding()
    encoded = encode_nested_tool_call(call, encoding)
    depth = call.depth
    node_count = call.node_count()
    artifact = ArtifactRef(kind="nested-tool-call", name=name, path=manifest_path)
    violations: list[NestedToolCallViolation] = []

    if depth > encoding.max_depth:
        violations.append(
            _violation(
                NestedToolCallViolationKind.DEPTH_EXCEEDED,
                f"nesting depth {depth} exceeds declared bound {encoding.max_depth}",
                path="$",
                artifact=artifact,
                parser_states=(f"depth={depth}", f"bound={encoding.max_depth}"),
                step_output=f"depth {depth} > {encoding.max_depth}",
                suggestion="Flatten the tool-call tree or raise max_depth to the verified bound.",
            )
        )

    _collect_marker_violations(call, encoding, "$", artifact, violations)

    if encoding.style == "xml-tags":
        frame_states, frames, dangling = _replay_naive_parser(encoded, encoding)
        naive_sound = len(frames) == 1 and dangling == 0 and _json_parses(frames[0])
        if node_count > 1 and not naive_sound:
            violations.append(
                _violation(
                    NestedToolCallViolationKind.FRAME_DESYNC,
                    (
                        f"a non-nesting marker parser recovered {len(frames)} frame(s) and {dangling} dangling "
                        f"close marker(s) from a single {node_count}-node nested call; inner markers are not escaped"
                    ),
                    path="$",
                    artifact=artifact,
                    parser_states=frame_states or ("no-frames",),
                    step_output=f"frames={len(frames)} dangling={dangling} expected=1/0",
                    suggestion=(
                        "Escape nested markers, length-prefix frames, or use the json encoding so a "
                        "stream parser cannot mistake inner tool calls for top-level frames."
                    ),
                    rendered=encoded,
                )
            )

    # Round-trip soundness: a correct encoding must decode back to the original tree.
    decoded = _safe_decode(encoded, encoding)
    if decoded != _to_json_tree(call):
        violations.append(
            _violation(
                NestedToolCallViolationKind.ROUND_TRIP_MISMATCH,
                "decoding the encoded tool call did not reconstruct the original nested arguments",
                path="$",
                artifact=artifact,
                parser_states=("encode", "decode", "compare"),
                step_output="decoded != original",
                suggestion="Use a self-delimiting encoding (json) whose decoder is the inverse of the encoder.",
                rendered=encoded,
            )
        )

    return NestedToolCallReport(
        name=name,
        encoding=encoding,
        encoded=encoded,
        depth=depth,
        node_count=node_count,
        violations=tuple(violations),
    )


def load_nested_tool_call_manifest(data: dict[str, Any]) -> NestedToolCallReport:
    """Build and analyze a nested tool call from a plain JSON manifest object."""

    if not isinstance(data, dict):
        raise NestedToolCallError("nested tool-call manifest root must be a JSON object")
    name = data.get("name", "nested-tool-call")
    if not isinstance(name, str) or not name:
        raise NestedToolCallError("manifest name must be a non-empty string")
    call = _parse_call(data.get("call"))
    encoding = _parse_encoding(data.get("encoding"))
    return analyze_nested_tool_call(call, encoding, name=name)


def render_nested_tool_call_json(report: NestedToolCallReport) -> str:
    """Render a nested tool-call report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_nested_tool_call_text(report: NestedToolCallReport) -> str:
    """Render a nested tool-call report for CLI users."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        "PromptABI nested tool-call encoding",
        f"name: {report.name}",
        f"style: {report.encoding.style}",
        f"status: {status}",
        f"depth: {report.depth}",
        f"nodes: {report.node_count}",
    ]
    if report.ok:
        lines.append("violations: none")
        return "\n".join(lines) + "\n"
    lines.append(f"violations: {len(report.violations)}")
    for violation in report.violations:
        lines.append(f"ERROR {violation.kind.value} [{violation.path}]: {violation.message}")
        lines.append(f"  witness: {violation.witness.summary}")
        if violation.witness.parser_states:
            lines.append(f"  parser-states: {' -> '.join(violation.witness.parser_states)}")
        lines.append(f"  suggestion: {violation.suggestion}")
    return "\n".join(lines) + "\n"


# --- encoding helpers -------------------------------------------------------


def _to_json_tree(call: NestedToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": _json_value(call.arguments)}


def _json_value(value: Any) -> Any:
    if isinstance(value, NestedToolCall):
        return _to_json_tree(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _encode_xml(call: NestedToolCall, encoding: ToolCallEncoding) -> str:
    arguments = {key: _encode_xml_value(value, encoding) for key, value in call.arguments.items()}
    body = json.dumps({"name": call.name, "arguments": arguments}, sort_keys=True)
    return f"{encoding.open_marker}{body}{encoding.close_marker}"


def _encode_xml_value(value: Any, encoding: ToolCallEncoding) -> Any:
    if isinstance(value, NestedToolCall):
        return _encode_xml(value, encoding)
    if isinstance(value, dict):
        return {str(key): _encode_xml_value(item, encoding) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_encode_xml_value(item, encoding) for item in value]
    return value


def _replay_naive_parser(
    encoded: str, encoding: ToolCallEncoding
) -> tuple[tuple[str, ...], list[str], int]:
    """Replay a non-nesting marker parser, the kind real tool-call stacks ship.

    The parser pairs each open marker with the *immediately following* close
    marker -- it is not nesting aware.  Nested, unescaped tool calls therefore
    truncate the outer frame at the inner close marker and leave the outer close
    marker dangling.  Returns the state log, recovered frame contents, and the
    count of dangling (unmatched) close markers.
    """

    states: list[str] = []
    frames: list[str] = []
    open_len = len(encoding.open_marker)
    close_len = len(encoding.close_marker)
    index = 0
    length = len(encoded)
    while index < length:
        open_at = encoded.find(encoding.open_marker, index)
        if open_at == -1:
            dangling = encoded.count(encoding.close_marker, index)
            if dangling:
                states.append(f"dangling-close x{dangling}")
            return tuple(states), frames, dangling
        close_at = encoded.find(encoding.close_marker, open_at + open_len)
        if close_at == -1:
            states.append(f"open@{open_at}->unterminated")
            return tuple(states), frames, 0
        frames.append(encoded[open_at + open_len : close_at])
        states.append(f"frame@{open_at}:{close_at}")
        index = close_at + close_len
    return tuple(states), frames, 0


def _json_parses(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _collect_marker_violations(
    call: NestedToolCall,
    encoding: ToolCallEncoding,
    path: str,
    artifact: ArtifactRef,
    violations: list[NestedToolCallViolation],
) -> None:
    markers = set(encoding.forbidden_markers)
    if encoding.style == "xml-tags":
        markers |= {encoding.open_marker, encoding.close_marker}
    for key, value in call.arguments.items():
        child_path = f"{path}.{key}"
        if isinstance(value, NestedToolCall):
            _collect_marker_violations(value, encoding, child_path, artifact, violations)
            continue
        if isinstance(value, list | tuple):
            for offset, item in enumerate(value):
                if isinstance(item, NestedToolCall):
                    _collect_marker_violations(item, encoding, f"{child_path}[{offset}]", artifact, violations)
                elif isinstance(item, str):
                    _check_string_markers(item, markers, encoding, f"{child_path}[{offset}]", artifact, violations)
            continue
        if isinstance(value, str):
            _check_string_markers(value, markers, encoding, child_path, artifact, violations)


def _check_string_markers(
    text: str,
    markers: set[str],
    encoding: ToolCallEncoding,
    path: str,
    artifact: ArtifactRef,
    violations: list[NestedToolCallViolation],
) -> None:
    for marker in sorted(markers):
        if marker and marker in text:
            kind = (
                NestedToolCallViolationKind.DELIMITER_COLLISION
                if marker in {encoding.open_marker, encoding.close_marker}
                else NestedToolCallViolationKind.FORBIDDEN_MARKER
            )
            violations.append(
                _violation(
                    kind,
                    f"argument string at {path} contains control marker {marker!r}",
                    path=path,
                    artifact=artifact,
                    parser_states=(f"scan {path}", f"hit {marker}"),
                    step_output=f"{path} contains {marker}",
                    suggestion="Escape or json-encode argument strings before embedding them in a tool-call frame.",
                    rendered=text,
                )
            )
            return


def _safe_decode(encoded: str, encoding: ToolCallEncoding) -> Any:
    if encoding.style == "json":
        try:
            return json.loads(encoded)
        except json.JSONDecodeError:
            return None
    # The naive xml-tags scheme is decoded by stripping the outermost markers and
    # json-parsing.  Nested markers left inside argument strings break this.
    if not (encoded.startswith(encoding.open_marker) and encoded.endswith(encoding.close_marker)):
        return None
    inner = encoded[len(encoding.open_marker) : len(encoded) - len(encoding.close_marker)]
    try:
        parsed = json.loads(inner)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _decode_xml_tree(parsed, encoding)


def _decode_xml_tree(parsed: dict[str, Any], encoding: ToolCallEncoding) -> Any:
    arguments = parsed.get("arguments", {})
    decoded_args = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            decoded_args[key] = _decode_xml_value(value, encoding)
    return {"name": parsed.get("name"), "arguments": decoded_args}


def _decode_xml_value(value: Any, encoding: ToolCallEncoding) -> Any:
    if isinstance(value, str) and value.startswith(encoding.open_marker) and value.endswith(encoding.close_marker):
        decoded = _safe_decode(value, encoding)
        return decoded if decoded is not None else value
    if isinstance(value, dict):
        return {key: _decode_xml_value(item, encoding) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_xml_value(item, encoding) for item in value]
    return value


def _violation(
    kind: NestedToolCallViolationKind,
    message: str,
    *,
    path: str,
    artifact: ArtifactRef,
    parser_states: tuple[str, ...],
    step_output: str,
    suggestion: str,
    rendered: str | None = None,
) -> NestedToolCallViolation:
    return NestedToolCallViolation(
        kind=kind,
        message=message,
        path=path,
        witness=WitnessTrace(
            summary=f"{path} violates {kind.value}",
            steps=(
                WitnessStep(action="encode nested tool call", input=path, output=kind.value),
                WitnessStep(action="replay encoding contract", input="parser", output=step_output),
                WitnessStep(action="emit minimal encoding fix", input=kind.value, output=suggestion),
            ),
            artifacts=(artifact,),
            parser_states=parser_states,
            rendered_strings=(rendered,) if rendered else (),
            minimal_fixes=(suggestion,),
        ),
        suggestion=suggestion,
    )


# --- manifest parsing -------------------------------------------------------


def _parse_call(raw: object) -> NestedToolCall:
    if not isinstance(raw, dict):
        raise NestedToolCallError("manifest 'call' must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise NestedToolCallError("tool call name must be a non-empty string")
    arguments_raw = raw.get("arguments", {})
    if not isinstance(arguments_raw, dict):
        raise NestedToolCallError("tool call arguments must be an object")
    return NestedToolCall(name=name, arguments={key: _parse_value(value) for key, value in arguments_raw.items()})


def _parse_value(raw: object) -> Any:
    if isinstance(raw, dict) and "__tool_call__" in raw:
        return _parse_call(raw["__tool_call__"])
    if isinstance(raw, dict):
        return {key: _parse_value(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return [_parse_value(value) for value in raw]
    return raw


def _parse_encoding(raw: object) -> ToolCallEncoding:
    if raw is None:
        return ToolCallEncoding()
    if not isinstance(raw, dict):
        raise NestedToolCallError("manifest 'encoding' must be an object")
    return ToolCallEncoding(
        style=str(raw.get("style", "xml-tags")),
        open_marker=str(raw.get("open_marker", "<tool_call>")),
        close_marker=str(raw.get("close_marker", "</tool_call>")),
        max_depth=int(raw.get("max_depth", 4)),
        forbidden_markers=tuple(raw.get("forbidden_markers", ()) or ()),
    )
