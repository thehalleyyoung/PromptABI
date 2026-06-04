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
