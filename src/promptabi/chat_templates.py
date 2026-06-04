"""Hugging Face chat-template parsing for tokenizer_config.json artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import SourceSpan
from .source import JsonSourceMap, build_json_source_map


_JINJA_TOKEN_RE = re.compile(r"({[#%{]-?.*?-?[#%}]})", re.DOTALL)
_STRING_RE = re.compile(r"""(['"])(.*?)(?<!\\)\1""", re.DOTALL)
_MESSAGE_FIELD_RE = re.compile(
    r"""message(?:\[['"](?P<bracket>[^'"]+)['"]\]|\.(?P<dot>[A-Za-z_][A-Za-z0-9_]*))"""
)
_TOOL_FIELD_RE = re.compile(
    r"""tool(?:\[['"](?P<bracket>[^'"]+)['"]\]|\.(?P<dot>[A-Za-z_][A-Za-z0-9_]*))"""
)
_SPECIAL_TOKEN_KEYS = {
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
}
_SUPPORTED_TAG_PREFIXES = (
    "if ",
    "elif ",
    "else",
    "endif",
    "for ",
    "endfor",
    "set ",
)
_UNSUPPORTED_TAG_PREFIXES = (
    "macro ",
    "endmacro",
    "call ",
    "endcall",
    "filter ",
    "endfilter",
    "block ",
    "endblock",
    "extends ",
    "include ",
    "import ",
    "from ",
    "with ",
    "endwith",
    "raw",
    "endraw",
)
_UNSUPPORTED_GLOBALS = ("raise_exception(", "strftime_now(", "cycler(", "joiner(", "namespace(")


@dataclass(frozen=True, slots=True)
class ChatTemplateSpecialToken:
    """A special-token declaration extracted from tokenizer metadata."""

    name: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "text": self.text}


@dataclass(frozen=True, slots=True)
class ChatTemplateFieldUse:
    """A message/tool field reference found in a Jinja expression."""

    owner: str
    field: str
    expression: str

    def to_dict(self) -> dict[str, str]:
        return {"owner": self.owner, "field": self.field, "expression": self.expression}


@dataclass(frozen=True, slots=True)
class ChatTemplateLoop:
    """A supported loop over messages or tools."""

    variable: str
    iterable: str
    expression: str

    def to_dict(self) -> dict[str, str]:
        return {"variable": self.variable, "iterable": self.iterable, "expression": self.expression}


@dataclass(frozen=True, slots=True)
class ChatTemplateCondition:
    """A branch condition relevant to later symbolic execution."""

    expression: str
    role: str | None = None
    uses_generation_prompt: bool = False

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "expression": self.expression,
            "uses_generation_prompt": self.uses_generation_prompt,
        }
        if self.role is not None:
            data["role"] = self.role
        return data


@dataclass(frozen=True, slots=True)
class ChatTemplateUnsupportedConstruct:
    """A construct outside PromptABI's current sound chat-template fragment."""

    kind: str
    expression: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "expression": self.expression, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ChatTemplateParseResult:
    """A deterministic summary of a Hugging Face chat template."""

    template_source: str
    source_span: SourceSpan | None = None
    template_format: str = "jinja"
    special_tokens: tuple[ChatTemplateSpecialToken, ...] = ()
    message_fields: tuple[ChatTemplateFieldUse, ...] = ()
    tool_fields: tuple[ChatTemplateFieldUse, ...] = ()
    loops: tuple[ChatTemplateLoop, ...] = ()
    conditions: tuple[ChatTemplateCondition, ...] = ()
    filters: tuple[str, ...] = ()
    constants: tuple[str, ...] = ()
    role_assumptions: tuple[str, ...] = ()
    generation_prompt_excerpts: tuple[str, ...] = ()
    uses_generation_prompt: bool = False
    uses_tools: bool = False
    uses_whitespace_control: bool = False
    unsupported_constructs: tuple[ChatTemplateUnsupportedConstruct, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.unsupported_constructs

    def to_dict(self) -> dict[str, object]:
        return {
            "template_format": self.template_format,
            "template_length": len(self.template_source),
            "source_span": self.source_span.to_dict() if self.source_span is not None else None,
            "special_tokens": [token.to_dict() for token in self.special_tokens],
            "message_fields": [field.to_dict() for field in self.message_fields],
            "tool_fields": [field.to_dict() for field in self.tool_fields],
            "loops": [loop.to_dict() for loop in self.loops],
            "conditions": [condition.to_dict() for condition in self.conditions],
            "filters": list(self.filters),
            "constants": list(self.constants),
            "role_assumptions": list(self.role_assumptions),
            "generation_prompt_excerpts": list(self.generation_prompt_excerpts),
            "uses_generation_prompt": self.uses_generation_prompt,
            "uses_tools": self.uses_tools,
            "uses_whitespace_control": self.uses_whitespace_control,
            "supported": self.supported,
            "unsupported_constructs": [
                unsupported.to_dict() for unsupported in self.unsupported_constructs
            ],
        }


@dataclass(frozen=True, slots=True)
class ChatTemplateSymbolicBounds:
    """Finite limits for bounded chat-template symbolic execution."""

    max_messages: int = 2
    max_tools: int = 1
    max_loop_iterations: int = 2
    max_paths: int = 128

    def __post_init__(self) -> None:
        for name in ("max_messages", "max_tools", "max_loop_iterations", "max_paths"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_paths == 0:
            raise ValueError("max_paths must be positive")

    def limit_for(self, iterable: str) -> int:
        if iterable == "messages":
            return min(self.max_messages, self.max_loop_iterations)
        if iterable == "tools":
            return min(self.max_tools, self.max_loop_iterations)
        return self.max_loop_iterations

    def to_dict(self) -> dict[str, int]:
        return {
            "max_messages": self.max_messages,
            "max_tools": self.max_tools,
            "max_loop_iterations": self.max_loop_iterations,
            "max_paths": self.max_paths,
        }


@dataclass(frozen=True, slots=True)
class ChatTemplateSymbolicSegment:
    """One literal or symbolic output segment produced by a bounded template path."""

    kind: str
    value: str
    expression: str | None = None
    filters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"literal", "variable", "constant", "unknown"}:
            raise ValueError(f"unsupported symbolic segment kind: {self.kind}")
        if not self.value:
            raise ValueError("symbolic segment value must be non-empty")
        object.__setattr__(self, "filters", tuple(self.filters))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind, "value": self.value}
        if self.expression is not None:
            data["expression"] = self.expression
        if self.filters:
            data["filters"] = list(self.filters)
        return data


@dataclass(frozen=True, slots=True)
class ChatTemplateSymbolicPath:
    """A single bounded control-flow path through a chat template."""

    conditions: tuple[str, ...] = ()
    segments: tuple[ChatTemplateSymbolicSegment, ...] = ()
    loop_iterations: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "loop_iterations", tuple(self.loop_iterations))

    @property
    def rendered_pattern(self) -> str:
        return "".join(segment.value for segment in self.segments)

    def to_dict(self) -> dict[str, object]:
        return {
            "conditions": list(self.conditions),
            "segments": [segment.to_dict() for segment in self.segments],
            "loop_iterations": [
                {"iterable": iterable, "count": count}
                for iterable, count in self.loop_iterations
            ],
            "rendered_pattern": self.rendered_pattern,
        }


@dataclass(frozen=True, slots=True)
class ChatTemplateSymbolicAbstention:
    """Reason the symbolic executor declined to prove a fragment precisely."""

    kind: str
    expression: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "expression": self.expression, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ChatTemplateSymbolicExecution:
    """Bounded symbolic execution result for a supported HF/Jinja chat template."""

    bounds: ChatTemplateSymbolicBounds
    paths: tuple[ChatTemplateSymbolicPath, ...] = ()
    abstentions: tuple[ChatTemplateSymbolicAbstention, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))
        unique = {(item.kind, item.expression, item.reason): item for item in self.abstentions}
        object.__setattr__(self, "abstentions", tuple(unique[key] for key in sorted(unique)))

    @property
    def supported(self) -> bool:
        return not self.abstentions

    def to_dict(self) -> dict[str, object]:
        return {
            "bounds": self.bounds.to_dict(),
            "supported": self.supported,
            "path_count": len(self.paths),
            "paths": [path.to_dict() for path in self.paths],
            "abstentions": [abstention.to_dict() for abstention in self.abstentions],
        }


class ChatTemplateParseError(ValueError):
    """Raised when tokenizer_config.json cannot yield a chat template."""


def parse_hf_tokenizer_config_chat_template(
    path: str | Path,
) -> ChatTemplateParseResult:
    """Parse a Hugging Face tokenizer_config.json chat_template from disk."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChatTemplateParseError(
            f"tokenizer_config.json is not valid JSON at {config_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise ChatTemplateParseError(f"tokenizer_config.json must contain an object: {config_path}")
    try:
        source_map = build_json_source_map(text, config_path)
    except ValueError as exc:
        raise ChatTemplateParseError(f"tokenizer_config.json source map failed: {exc}") from exc
    return parse_hf_chat_template_config(raw, source_map=source_map)


def parse_hf_chat_template_config(
    config: dict[str, object],
    *,
    source_map: JsonSourceMap | None = None,
) -> ChatTemplateParseResult:
    """Extract a conservative, deterministic summary from HF tokenizer metadata."""

    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ChatTemplateParseError("tokenizer_config.json must define a non-empty string chat_template")

    special_tokens = _extract_special_tokens(config)
    tokens = tuple(_iter_jinja_tokens(template))
    expression_bodies = tuple(body for kind, body, _raw in tokens if kind == "expression")
    tag_bodies = tuple(body for kind, body, _raw in tokens if kind == "tag")
    all_bodies = expression_bodies + tag_bodies

    message_fields = _field_uses("message", all_bodies, _MESSAGE_FIELD_RE)
    tool_fields = _field_uses("tool", all_bodies, _TOOL_FIELD_RE)
    loops = _loops(tag_bodies)
    conditions = _conditions(tag_bodies)
    role_assumptions = _role_assumptions(template, conditions)
    unsupported = _unsupported_constructs(expression_bodies, tag_bodies)
    constants = _constants(template, config, special_tokens)
    filters = _filters(expression_bodies)
    generation_prompt_excerpts = _generation_prompt_excerpts(template)
    uses_tools = bool(tool_fields) or "tools" in template

    return ChatTemplateParseResult(
        template_source=template,
        source_span=source_map.span_for(("chat_template",)) if source_map is not None else None,
        special_tokens=special_tokens,
        message_fields=message_fields,
        tool_fields=tool_fields,
        loops=loops,
        conditions=conditions,
        filters=filters,
        constants=constants,
        role_assumptions=role_assumptions,
        generation_prompt_excerpts=generation_prompt_excerpts,
        uses_generation_prompt="add_generation_prompt" in template,
        uses_tools=uses_tools,
        uses_whitespace_control=any(raw.startswith(("{%-", "{{-", "{#-")) or raw.endswith(("-}", "-}}", "-#}")) for _kind, _body, raw in tokens),
        unsupported_constructs=unsupported,
    )


def symbolically_execute_chat_template(
    parsed: ChatTemplateParseResult,
    *,
    bounds: ChatTemplateSymbolicBounds | None = None,
) -> ChatTemplateSymbolicExecution:
    """Boundedly execute the supported HF/Jinja chat-template fragment.

    The executor mirrors the Hugging Face chat-template Jinja environment's
    structural whitespace behavior (`trim_blocks=True`, `lstrip_blocks=True`)
    and returns symbolic output paths rather than invoking Jinja. It abstains
    when the template uses unsupported constructs or exceeds finite bounds.
    """

    active_bounds = bounds or ChatTemplateSymbolicBounds()
    abstentions = [
        ChatTemplateSymbolicAbstention(
            kind=item.kind,
            expression=item.expression,
            reason=item.reason,
        )
        for item in parsed.unsupported_constructs
    ]
    segments = _lex_symbolic_segments(parsed.template_source)
    parser = _SymbolicParser(segments)
    nodes = parser.parse()
    abstentions.extend(parser.abstentions)
    if parser.abstentions:
        return ChatTemplateSymbolicExecution(bounds=active_bounds, paths=(), abstentions=tuple(abstentions))
    executor = _SymbolicExecutor(parsed, active_bounds)
    paths = executor.execute(nodes)
    abstentions.extend(executor.abstentions)
    return ChatTemplateSymbolicExecution(bounds=active_bounds, paths=paths, abstentions=tuple(abstentions))


def _extract_special_tokens(config: dict[str, object]) -> tuple[ChatTemplateSpecialToken, ...]:
    tokens: list[ChatTemplateSpecialToken] = []
    for key, value in sorted(config.items()):
        if key in _SPECIAL_TOKEN_KEYS or key.endswith("_token"):
            text = _special_token_text(value)
            if text:
                tokens.append(ChatTemplateSpecialToken(name=key, text=text))

    additional = config.get("additional_special_tokens")
    if isinstance(additional, list):
        for index, value in enumerate(additional):
            text = _special_token_text(value)
            if text:
                tokens.append(ChatTemplateSpecialToken(name=f"additional_special_tokens.{index}", text=text))

    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for token_id, value in sorted(decoder.items(), key=lambda item: str(item[0])):
            text = _special_token_text(value)
            if text:
                tokens.append(ChatTemplateSpecialToken(name=f"added_tokens_decoder.{token_id}", text=text))

    deduped = {(token.name, token.text): token for token in tokens}
    return tuple(deduped[key] for key in sorted(deduped))


def _special_token_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _iter_jinja_tokens(template: str) -> tuple[tuple[str, str, str], ...]:
    tokens: list[tuple[str, str, str]] = []
    for match in _JINJA_TOKEN_RE.finditer(template):
        raw = match.group(1)
        if raw.startswith("{{"):
            kind = "expression"
            body = raw[2:-2]
        elif raw.startswith("{%"):
            kind = "tag"
            body = raw[2:-2]
        else:
            kind = "comment"
            body = raw[2:-2]
        tokens.append((kind, body.strip().strip("-").strip(), raw))
    return tuple(tokens)


def _field_uses(
    owner: str,
    expressions: tuple[str, ...],
    pattern: re.Pattern[str],
) -> tuple[ChatTemplateFieldUse, ...]:
    uses: dict[tuple[str, str, str], ChatTemplateFieldUse] = {}
    for expression in expressions:
        for match in pattern.finditer(expression):
            field = match.group("bracket") or match.group("dot")
            use = ChatTemplateFieldUse(owner=owner, field=field, expression=expression)
            uses[(owner, field, expression)] = use
    return tuple(uses[key] for key in sorted(uses))


def _loops(tag_bodies: tuple[str, ...]) -> tuple[ChatTemplateLoop, ...]:
    loops: dict[tuple[str, str, str], ChatTemplateLoop] = {}
    for body in tag_bodies:
        if not body.startswith("for "):
            continue
        match = re.fullmatch(r"for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([A-Za-z_][A-Za-z0-9_\.]*)", body)
        if match:
            loop = ChatTemplateLoop(variable=match.group(1), iterable=match.group(2), expression=body)
            loops[(loop.variable, loop.iterable, loop.expression)] = loop
    return tuple(loops[key] for key in sorted(loops))


def _conditions(tag_bodies: tuple[str, ...]) -> tuple[ChatTemplateCondition, ...]:
    conditions: dict[tuple[str, str | None, bool], ChatTemplateCondition] = {}
    for body in tag_bodies:
        if body.startswith("if "):
            expression = body[3:].strip()
        elif body.startswith("elif "):
            expression = body[5:].strip()
        else:
            continue
        role = _role_literal(expression)
        condition = ChatTemplateCondition(
            expression=expression,
            role=role,
            uses_generation_prompt="add_generation_prompt" in expression,
        )
        conditions[(condition.expression, condition.role, condition.uses_generation_prompt)] = condition
    return tuple(conditions[key] for key in sorted(conditions))


def _role_literal(expression: str) -> str | None:
    match = re.search(
        r"""message(?:\[['"]role['"]\]|\.role)\s*(?:==|!=)\s*(['"])(?P<role>[^'"]+)\1""",
        expression,
    )
    return match.group("role") if match else None


def _role_assumptions(
    template: str,
    conditions: tuple[ChatTemplateCondition, ...],
) -> tuple[str, ...]:
    roles = {condition.role for condition in conditions if condition.role is not None}
    for match in re.finditer(r"""(?:^|[^A-Za-z_])(['"])(system|user|assistant|tool|developer|function)\1""", template):
        roles.add(match.group(2))
    if "add_generation_prompt" in template and re.search(r"assistant|model|Response", template):
        roles.add("assistant")
    return tuple(sorted(role for role in roles if role is not None))


def _unsupported_constructs(
    expression_bodies: tuple[str, ...],
    tag_bodies: tuple[str, ...],
) -> tuple[ChatTemplateUnsupportedConstruct, ...]:
    unsupported: list[ChatTemplateUnsupportedConstruct] = []
    for body in tag_bodies:
        if body.startswith(_UNSUPPORTED_TAG_PREFIXES):
            unsupported.append(
                ChatTemplateUnsupportedConstruct(
                    kind="tag",
                    expression=body,
                    reason="Jinja control tag is outside PromptABI's supported symbolic fragment.",
                )
            )
            continue
        if not body.startswith(_SUPPORTED_TAG_PREFIXES):
            unsupported.append(
                ChatTemplateUnsupportedConstruct(
                    kind="tag",
                    expression=body,
                    reason="Jinja control tag is not recognized by the bounded parser.",
                )
            )
            continue
        if body.startswith("for ") and body not in {loop.expression for loop in _loops((body,))}:
            unsupported.append(
                ChatTemplateUnsupportedConstruct(
                    kind="loop",
                    expression=body,
                    reason="Only simple 'for variable in iterable' loops are currently modeled.",
                )
            )
    for body in expression_bodies + tag_bodies:
        for global_name in _UNSUPPORTED_GLOBALS:
            if global_name in body:
                unsupported.append(
                    ChatTemplateUnsupportedConstruct(
                        kind="global",
                        expression=body,
                        reason=f"Template global {global_name[:-1]!r} requires runtime behavior.",
                    )
                )
    unique = {(item.kind, item.expression, item.reason): item for item in unsupported}
    return tuple(unique[key] for key in sorted(unique))


def _filters(expression_bodies: tuple[str, ...]) -> tuple[str, ...]:
    filters: set[str] = set()
    for body in expression_bodies:
        for match in re.finditer(r"\|\s*([A-Za-z_][A-Za-z0-9_]*)", body):
            filters.add(match.group(1))
    return tuple(sorted(filters))


def _constants(
    template: str,
    config: dict[str, object],
    special_tokens: tuple[ChatTemplateSpecialToken, ...],
) -> tuple[str, ...]:
    constants = {match.group(2) for match in _STRING_RE.finditer(template) if match.group(2)}
    constants.update(token.text for token in special_tokens)
    for key in ("chat_template",):
        value = config.get(key)
        if isinstance(value, str):
            for token in special_tokens:
                if token.text in value:
                    constants.add(token.text)
    return tuple(sorted(constants))


def _generation_prompt_excerpts(template: str) -> tuple[str, ...]:
    excerpts: list[str] = []
    pattern = re.compile(
        r"{%-?\s*if\s+add_generation_prompt\s*-?%}(.*?){%-?\s*endif\s*-?%}",
        re.DOTALL,
    )
    for match in pattern.finditer(template):
        excerpt = _strip_jinja(match.group(1)).strip()
        if excerpt:
            excerpts.append(excerpt)
    return tuple(sorted(dict.fromkeys(excerpts)))


def _strip_jinja(text: str) -> str:
    return _JINJA_TOKEN_RE.sub("", text)


@dataclass(frozen=True, slots=True)
class _SymbolicSegment:
    kind: str
    body: str
    raw: str = ""


@dataclass(frozen=True, slots=True)
class _LiteralNode:
    text: str


@dataclass(frozen=True, slots=True)
class _ExpressionNode:
    expression: str


@dataclass(frozen=True, slots=True)
class _SetNode:
    name: str
    expression: str


@dataclass(frozen=True, slots=True)
class _ForNode:
    variable: str
    iterable: str
    body: tuple["_Node", ...]


@dataclass(frozen=True, slots=True)
class _IfBranch:
    condition: str | None
    body: tuple["_Node", ...]


@dataclass(frozen=True, slots=True)
class _IfNode:
    branches: tuple[_IfBranch, ...]


_Node = _LiteralNode | _ExpressionNode | _SetNode | _ForNode | _IfNode


@dataclass(frozen=True, slots=True)
class _PathState:
    conditions: tuple[str, ...] = ()
    segments: tuple[ChatTemplateSymbolicSegment, ...] = ()
    loop_iterations: tuple[tuple[str, int], ...] = ()
    bindings: tuple[tuple[str, ChatTemplateSymbolicSegment], ...] = ()

    def bind(self, name: str, value: ChatTemplateSymbolicSegment) -> "_PathState":
        bindings = {key: segment for key, segment in self.bindings}
        bindings[name] = value
        return _PathState(
            conditions=self.conditions,
            segments=self.segments,
            loop_iterations=self.loop_iterations,
            bindings=tuple(sorted(bindings.items())),
        )

    def with_condition(self, condition: str) -> "_PathState":
        return _PathState(
            conditions=self.conditions + (condition,),
            segments=self.segments,
            loop_iterations=self.loop_iterations,
            bindings=self.bindings,
        )

    def with_loop_iterations(self, iterable: str, count: int) -> "_PathState":
        return _PathState(
            conditions=self.conditions,
            segments=self.segments,
            loop_iterations=self.loop_iterations + ((iterable, count),),
            bindings=self.bindings,
        )

    def with_segment(self, segment: ChatTemplateSymbolicSegment) -> "_PathState":
        return _PathState(
            conditions=self.conditions,
            segments=self.segments + (segment,),
            loop_iterations=self.loop_iterations,
            bindings=self.bindings,
        )

    def to_public(self) -> ChatTemplateSymbolicPath:
        return ChatTemplateSymbolicPath(
            conditions=self.conditions,
            segments=self.segments,
            loop_iterations=self.loop_iterations,
        )


def _lex_symbolic_segments(template: str) -> tuple[_SymbolicSegment, ...]:
    segments: list[_SymbolicSegment] = []
    position = 0
    trim_leading_whitespace = False
    trim_one_newline = False
    for match in _JINJA_TOKEN_RE.finditer(template):
        raw = match.group(1)
        literal = template[position : match.start()]
        if trim_leading_whitespace:
            literal = literal.lstrip()
        elif trim_one_newline:
            literal = re.sub(r"^\r?\n", "", literal, count=1)

        kind, body, left_trim, right_trim = _symbolic_token_parts(raw)
        if kind == "tag":
            literal = _apply_lstrip_blocks(literal)
        if left_trim:
            literal = literal.rstrip()
        if literal:
            segments.append(_SymbolicSegment(kind="literal", body=literal))
        if kind != "comment":
            segments.append(_SymbolicSegment(kind=kind, body=body, raw=raw))
        position = match.end()
        trim_leading_whitespace = right_trim
        trim_one_newline = kind == "tag" and not right_trim

    literal = template[position:]
    if trim_leading_whitespace:
        literal = literal.lstrip()
    elif trim_one_newline:
        literal = re.sub(r"^\r?\n", "", literal, count=1)
    if literal:
        segments.append(_SymbolicSegment(kind="literal", body=literal))
    return tuple(segments)


def _symbolic_token_parts(raw: str) -> tuple[str, str, bool, bool]:
    if raw.startswith("{{"):
        kind = "expression"
        body = raw[2:-2]
        left_trim = raw.startswith("{{-")
        right_trim = raw.endswith("-}}")
    elif raw.startswith("{%"):
        kind = "tag"
        body = raw[2:-2]
        left_trim = raw.startswith("{%-")
        right_trim = raw.endswith("-%}")
    else:
        kind = "comment"
        body = raw[2:-2]
        left_trim = raw.startswith("{#-")
        right_trim = raw.endswith("-#}")
    return kind, body.strip().strip("-").strip(), left_trim, right_trim


def _apply_lstrip_blocks(literal: str) -> str:
    newline = max(literal.rfind("\n"), literal.rfind("\r"))
    prefix = literal[newline + 1 :]
    if prefix and all(character in " \t" for character in prefix):
        return literal[: newline + 1]
    return literal


class _SymbolicParser:
    def __init__(self, segments: tuple[_SymbolicSegment, ...]) -> None:
        self.segments = segments
        self.abstentions: list[ChatTemplateSymbolicAbstention] = []

    def parse(self) -> tuple[_Node, ...]:
        nodes, stop_tag, _index = self._parse_block(0, ())
        if stop_tag is not None:
            self._abstain("tag", stop_tag, "Jinja block terminator does not have a matching opener.")
        return nodes

    def _parse_block(
        self,
        index: int,
        stop_prefixes: tuple[str, ...],
    ) -> tuple[tuple[_Node, ...], str | None, int]:
        nodes: list[_Node] = []
        while index < len(self.segments):
            segment = self.segments[index]
            if segment.kind == "literal":
                nodes.append(_LiteralNode(segment.body))
                index += 1
                continue
            if segment.kind == "expression":
                nodes.append(_ExpressionNode(segment.body))
                index += 1
                continue
            if segment.kind != "tag":
                index += 1
                continue
            body = segment.body
            if stop_prefixes and body.startswith(stop_prefixes):
                return tuple(nodes), body, index
            if body.startswith("for "):
                node, index = self._parse_for(body, index)
                if node is not None:
                    nodes.append(node)
                continue
            if body.startswith("if "):
                node, index = self._parse_if(body, index)
                if node is not None:
                    nodes.append(node)
                continue
            if body.startswith("set "):
                node = self._parse_set(body)
                if node is not None:
                    nodes.append(node)
                index += 1
                continue
            if body in {"endif", "endfor"} or body.startswith(("elif ", "else")):
                return tuple(nodes), body, index
            self._abstain("tag", body, "Jinja tag is outside the bounded symbolic executor fragment.")
            index += 1
        return tuple(nodes), None, index

    def _parse_for(self, body: str, index: int) -> tuple[_ForNode | None, int]:
        match = re.fullmatch(r"for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([A-Za-z_][A-Za-z0-9_\.]*)", body)
        if not match:
            self._abstain("loop", body, "Only simple 'for variable in iterable' loops are modeled.")
            return None, index + 1
        child_nodes, stop_tag, child_index = self._parse_block(index + 1, ("endfor",))
        if stop_tag != "endfor":
            self._abstain("loop", body, "For loop is missing a matching endfor tag.")
            return None, child_index
        return _ForNode(variable=match.group(1), iterable=match.group(2), body=child_nodes), child_index + 1

    def _parse_if(self, body: str, index: int) -> tuple[_IfNode | None, int]:
        branches: list[_IfBranch] = []
        condition = body[3:].strip()
        next_index = index + 1
        saw_else = False
        while True:
            child_nodes, stop_tag, child_index = self._parse_block(next_index, ("elif ", "else", "endif"))
            branches.append(_IfBranch(condition=condition, body=child_nodes))
            if stop_tag == "endif":
                if not saw_else:
                    branches.append(_IfBranch(condition=None, body=()))
                return _IfNode(branches=tuple(branches)), child_index + 1
            if stop_tag is None:
                self._abstain("condition", body, "If block is missing a matching endif tag.")
                return None, child_index
            if stop_tag.startswith("elif "):
                if saw_else:
                    self._abstain("condition", stop_tag, "Elif tag cannot follow an else tag.")
                    return None, child_index + 1
                condition = stop_tag[5:].strip()
                next_index = child_index + 1
                continue
            if stop_tag == "else":
                saw_else = True
                condition = None
                next_index = child_index + 1
                continue
            self._abstain("condition", stop_tag, "Unsupported if-block terminator.")
            return None, child_index + 1

    def _parse_set(self, body: str) -> _SetNode | None:
        match = re.fullmatch(r"set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", body)
        if not match:
            self._abstain("set", body, "Only simple scalar set assignments are modeled.")
            return None
        return _SetNode(name=match.group(1), expression=match.group(2).strip())

    def _abstain(self, kind: str, expression: str, reason: str) -> None:
        self.abstentions.append(ChatTemplateSymbolicAbstention(kind=kind, expression=expression, reason=reason))


class _SymbolicExecutor:
    def __init__(self, parsed: ChatTemplateParseResult, bounds: ChatTemplateSymbolicBounds) -> None:
        self.parsed = parsed
        self.bounds = bounds
        self.abstentions: list[ChatTemplateSymbolicAbstention] = []
        self.special_tokens = {token.name: token.text for token in parsed.special_tokens}
        self.special_tokens.update(_canonical_special_token_names(parsed.special_tokens))

    def execute(self, nodes: tuple[_Node, ...]) -> tuple[ChatTemplateSymbolicPath, ...]:
        states = self._execute_nodes(nodes, (_PathState(),), {})
        return tuple(state.to_public() for state in states[: self.bounds.max_paths])

    def _execute_nodes(
        self,
        nodes: tuple[_Node, ...],
        states: tuple[_PathState, ...],
        environment: dict[str, str],
    ) -> tuple[_PathState, ...]:
        active = states
        local_environment = dict(environment)
        for node in nodes:
            if not active:
                return ()
            if isinstance(node, _LiteralNode):
                segment = ChatTemplateSymbolicSegment(kind="literal", value=node.text)
                active = tuple(state.with_segment(segment) for state in active)
            elif isinstance(node, _ExpressionNode):
                active = self._execute_expression(node.expression, active, local_environment)
            elif isinstance(node, _SetNode):
                active = self._execute_set(node, active, local_environment)
            elif isinstance(node, _ForNode):
                active = self._execute_for(node, active, local_environment)
            elif isinstance(node, _IfNode):
                active = self._execute_if(node, active, local_environment)
            active = self._enforce_path_bound(active)
        return active

    def _execute_expression(
        self,
        expression: str,
        states: tuple[_PathState, ...],
        environment: dict[str, str],
    ) -> tuple[_PathState, ...]:
        return tuple(
            state.with_segment(self._expression_segment(expression, environment, state.bindings))
            for state in states
        )

    def _execute_set(
        self,
        node: _SetNode,
        states: tuple[_PathState, ...],
        environment: dict[str, str],
    ) -> tuple[_PathState, ...]:
        return tuple(
            state.bind(node.name, self._expression_segment(node.expression, environment, state.bindings))
            for state in states
        )

    def _execute_for(
        self,
        node: _ForNode,
        states: tuple[_PathState, ...],
        environment: dict[str, str],
    ) -> tuple[_PathState, ...]:
        if node.iterable not in {"messages", "tools"}:
            self._abstain("loop", node.iterable, "Only loops over messages and tools are symbolically bounded.")
            return ()
        all_states: list[_PathState] = []
        for state in states:
            for count in range(self.bounds.limit_for(node.iterable) + 1):
                repeated_states = (state.with_loop_iterations(node.iterable, count),)
                for iteration in range(count):
                    child_environment = dict(environment)
                    child_environment[node.variable] = f"{node.iterable}[{iteration}]"
                    child_environment["loop"] = f"{node.iterable}.loop[{iteration}]"
                    child_environment["loop.last"] = "true" if iteration == count - 1 else "false"
                    repeated_states = self._execute_nodes(node.body, repeated_states, child_environment)
                    if not repeated_states:
                        break
                all_states.extend(repeated_states)
                if len(all_states) > self.bounds.max_paths:
                    return self._enforce_path_bound(tuple(all_states))
        return tuple(all_states)

    def _execute_if(
        self,
        node: _IfNode,
        states: tuple[_PathState, ...],
        environment: dict[str, str],
    ) -> tuple[_PathState, ...]:
        all_states: list[_PathState] = []
        prior_conditions: list[str] = []
        for branch in node.branches:
            condition = branch.condition
            branch_label = "else" if condition is None else self._condition_label(condition, environment)
            if condition is None and prior_conditions:
                branch_label = "else after " + " and ".join(f"not({item})" for item in prior_conditions)
            branch_states = tuple(state.with_condition(branch_label) for state in states)
            all_states.extend(self._execute_nodes(branch.body, branch_states, environment))
            if condition is not None:
                prior_conditions.append(self._condition_label(condition, environment))
            if len(all_states) > self.bounds.max_paths:
                return self._enforce_path_bound(tuple(all_states))
        return tuple(all_states)

    def _expression_segment(
        self,
        expression: str,
        environment: dict[str, str],
        bindings: tuple[tuple[str, ChatTemplateSymbolicSegment], ...],
    ) -> ChatTemplateSymbolicSegment:
        base, filters = _split_filters(expression)
        bound_segment = dict(bindings).get(base)
        if bound_segment is not None:
            if filters:
                return ChatTemplateSymbolicSegment(
                    kind=bound_segment.kind,
                    value=bound_segment.value,
                    expression=expression,
                    filters=bound_segment.filters + filters,
                )
            return bound_segment
        literal = _string_literal(base)
        if literal is not None:
            return ChatTemplateSymbolicSegment(kind="literal", value=literal, expression=expression, filters=filters)
        if base in {"true", "True", "false", "False"}:
            return ChatTemplateSymbolicSegment(kind="constant", value=base.lower(), expression=expression, filters=filters)
        if base in self.special_tokens:
            return ChatTemplateSymbolicSegment(kind="constant", value=self.special_tokens[base], expression=expression, filters=filters)
        if base in environment:
            value = environment[base]
            kind = "constant" if value in {"true", "false"} else "variable"
            rendered = value if kind == "constant" else "{" + value + "}"
            return ChatTemplateSymbolicSegment(kind=kind, value=rendered, expression=expression, filters=filters)
        if base == "add_generation_prompt":
            return ChatTemplateSymbolicSegment(
                kind="variable",
                value="{add_generation_prompt}",
                expression=expression,
                filters=filters,
            )
        if base == "loop.last":
            return ChatTemplateSymbolicSegment(kind="variable", value="{loop.last}", expression=expression, filters=filters)
        bound = self._bound_name(base, environment)
        if bound is not None:
            return ChatTemplateSymbolicSegment(kind="variable", value="{" + bound + "}", expression=expression, filters=filters)
        self._abstain("expression", expression, "Expression is outside the bounded symbolic evaluator fragment.")
        return ChatTemplateSymbolicSegment(kind="unknown", value="{" + expression + "}", expression=expression, filters=filters)

    def _bound_name(self, expression: str, environment: dict[str, str]) -> str | None:
        path = _variable_path(expression)
        if path is None:
            return None
        root, fields = path
        if root in environment:
            return ".".join((environment[root],) + fields)
        return ".".join((root,) + fields)

    def _condition_label(self, condition: str, environment: dict[str, str]) -> str:
        label = condition
        for variable, replacement in sorted(environment.items(), key=lambda item: len(item[0]), reverse=True):
            label = re.sub(rf"\b{re.escape(variable)}\b", replacement, label)
        return label

    def _enforce_path_bound(self, states: tuple[_PathState, ...]) -> tuple[_PathState, ...]:
        if len(states) <= self.bounds.max_paths:
            return states
        self._abstain(
            "bounds",
            f"{len(states)} paths",
            f"Symbolic execution exceeded max_paths={self.bounds.max_paths}.",
        )
        return states[: self.bounds.max_paths]

    def _abstain(self, kind: str, expression: str, reason: str) -> None:
        self.abstentions.append(ChatTemplateSymbolicAbstention(kind=kind, expression=expression, reason=reason))


def _canonical_special_token_names(tokens: tuple[ChatTemplateSpecialToken, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        if token.name.startswith("added_tokens_decoder.") or token.name.startswith("additional_special_tokens."):
            continue
        result[token.name] = token.text
    return result


def _split_filters(expression: str) -> tuple[str, tuple[str, ...]]:
    parts = [part.strip() for part in expression.split("|")]
    base = parts[0]
    filters = tuple(part.split("(", 1)[0].strip() for part in parts[1:] if part.strip())
    return base, filters


def _string_literal(expression: str) -> str | None:
    match = re.fullmatch(r"""(['"])(.*)\1""", expression, re.DOTALL)
    if not match:
        return None
    return bytes(match.group(2), "utf-8").decode("unicode_escape")


def _variable_path(expression: str) -> tuple[str, tuple[str, ...]] | None:
    root_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", expression)
    if root_match is None:
        return None
    root = root_match.group(1)
    position = root_match.end()
    fields: list[str] = []
    while position < len(expression):
        dot = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)", expression[position:])
        if dot is not None:
            fields.append(dot.group(1))
            position += dot.end()
            continue
        bracket = re.match(r"""\[['"]([^'"]+)['"]\]""", expression[position:])
        if bracket is not None:
            fields.append(bracket.group(1))
            position += bracket.end()
            continue
        return None
    if not fields and root not in {"loop"}:
        return None
    return root, tuple(fields)
