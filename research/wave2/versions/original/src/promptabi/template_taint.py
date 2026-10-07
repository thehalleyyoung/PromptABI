"""Whole-template taint analysis of Hugging Face chat templates.

The bounded symbolic executor in :mod:`promptabi.chat_templates` produces exact
witness paths but supports only a fragment of Jinja.  Production chat templates
on the Hugging Face Hub routinely use ``namespace()`` objects, reversed loops,
macros, string methods, and ``raise_exception``.  This module works on the real
Jinja2 AST (``jinja2.Environment.parse``;
nothing is rendered or executed) and computing flow-insensitive heuristic *may-taint*:

* sources are fields of the conversation the caller supplies:
  ``messages[*].<field>``, ``tools``, ``documents``;
* taint propagates through assignment (including ``namespace`` attributes and
  ``{% set %}`` blocks), loops, slicing, string methods, concatenation,
  formatting, conditional expressions, macro calls, and filters;
  direct non-text outputs are excluded from sinks;
* sinks are ``{{ ... }}`` outputs.

A *finding* pairs a tainted sink with a structural marker the template itself
emits (a literal control string or a declared special token) that the tainted
value may contain. Replacement preserves taint conservatively. Jinja's HTML
escaping filters applied directly to a field neutralize markers that contain ``<``, ``>``, ``&``, or
quotes.  ``tojson`` is *not* a sanitizer under the Hugging Face environment,
whose ``tojson`` override calls ``json.dumps(..., ensure_ascii=False)`` and
leaves ``<|...|>`` sequences intact.

Each sink also records the role guards that dominate it (``message.role ==
'assistant'`` and similar), so a finding can be attributed to the roles whose
content reaches it.  The analysis computes heuristic may-taint over the AST and is not
path-sensitive; :mod:`promptabi.hf_tokenizer_composition` and the dynamic
oracle in ``study/hub_scan/oracle.py`` decide which findings materialize at the
token level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .role_boundaries import _extract_marker_candidates

ATTACKER_ROLES = frozenset({"user", "tool", "ipython", "function", "observation"})

_NON_TEXT_FILTERS = frozenset(
    {"length", "count", "int", "float", "bool", "abs", "round", "first_is_none", "wordcount", "sum"}
)
_ESCAPE_FILTERS = frozenset({"e", "escape", "forceescape", "urlencode", "xmlattr"})
_PRESERVING_METHODS = frozenset(
    {
        "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "lower", "upper", "title", "capitalize",
        "replace", "format", "join", "get", "items", "values", "keys", "removeprefix", "removesuffix", "partition",
        "rpartition", "casefold", "swapcase", "center", "ljust", "rjust", "zfill", "expandtabs",
    }
)


@dataclass(frozen=True, slots=True)
class Taint:
    """A set of source labels plus the markers removed by ``replace`` on every flow."""

    labels: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()
    escaped: bool = False

    def __bool__(self) -> bool:
        return bool(self.labels)

    def union(self, other: "Taint") -> "Taint":
        if not other.labels:
            return self
        if not self.labels:
            return other
        return Taint(self.labels | other.labels, self.removed & other.removed, self.escaped and other.escaped)

    def with_field(self, name: str) -> "Taint":
        labels = set()
        for label in self.labels:
            if label in {"messages", "messages[*]"} or label.endswith("[]"):
                base = label[:-2] if label.endswith("[]") else "messages"
                labels.add(f"{base}[*].{name}" if base == "messages" else f"{base}.{name}")
            else:
                labels.add(label)
        return Taint(frozenset(labels), self.removed, self.escaped)

    def element(self) -> "Taint":
        labels = set()
        for label in self.labels:
            labels.add("messages[*]" if label == "messages" else label)
        return Taint(frozenset(labels), self.removed, self.escaped)


EMPTY = Taint()


@dataclass(frozen=True, slots=True)
class TaintSink:
    lineno: int
    labels: tuple[str, ...]
    removed: tuple[str, ...]
    escaped: bool
    role_guard: tuple[str, ...] | None
    excluded_roles: tuple[str, ...]
    expression: str

    def reachable_by(self, role: str) -> bool:
        if role in self.excluded_roles:
            return False
        return self.role_guard is None or role in self.role_guard

    def to_dict(self) -> dict[str, object]:
        return {
            "lineno": self.lineno,
            "labels": list(self.labels),
            "removed": list(self.removed),
            "escaped": self.escaped,
            "role_guard": list(self.role_guard) if self.role_guard is not None else None,
            "excluded_roles": list(self.excluded_roles),
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class TaintFinding:
    field: str
    marker: str
    marker_source: str
    attacker_roles: tuple[str, ...]
    lineno: int
    expression: str

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "marker": self.marker,
            "marker_source": self.marker_source,
            "attacker_roles": list(self.attacker_roles),
            "lineno": self.lineno,
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class TemplateTaintReport:
    parsed: bool
    error: str
    sinks: tuple[TaintSink, ...]
    markers: tuple[tuple[str, str], ...]
    findings: tuple[TaintFinding, ...]
    other_role_findings: tuple[TaintFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "parsed": self.parsed,
            "error": self.error,
            "sink_count": len(self.sinks),
            "markers": [list(m) for m in self.markers],
            "findings": [f.to_dict() for f in self.findings],
            "other_role_findings": [f.to_dict() for f in self.other_role_findings],
        }


def _environment():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _Generation(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.Scope(body).set_lineno(lineno)

    return ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[_Generation, jinja2.ext.loopcontrols]
    )


class _Analyzer:
    def __init__(self) -> None:
        from jinja2 import nodes

        self.nodes = nodes
        self.env: dict[str, Taint] = {}
        self.macros: dict[str, object] = {}
        self.sinks: list[TaintSink] = []
        self.literals: list[str] = []
        self.guards: list[tuple[frozenset[str] | None, frozenset[str]]] = []
        self.changed = False
        self.macro_depth = 0

    # -- environment -------------------------------------------------------
    def bind(self, name: str, taint: Taint) -> None:
        old = self.env.get(name, EMPTY)
        new = old.union(taint)
        if new != old:
            self.env[name] = new
            self.changed = True

    # -- expressions -------------------------------------------------------
    def expr(self, node) -> Taint:
        n = self.nodes
        if node is None:
            return EMPTY
        if isinstance(node, n.Name):
            if node.name in {"messages", "loop_messages", "conversation"} and node.name not in self.env:
                return Taint(frozenset({"messages"}))
            if node.name in {"tools", "documents", "tool_calls"} and node.name not in self.env:
                return Taint(frozenset({f"{node.name}[]"}))
            return self.env.get(node.name, EMPTY)
        if isinstance(node, n.Const):
            return EMPTY
        if isinstance(node, n.Getattr):
            base = node.node
            if isinstance(base, n.Name) and f"{base.name}.{node.attr}" in self.env:
                return self.env[f"{base.name}.{node.attr}"]
            return self.expr(base).with_field(node.attr)
        if isinstance(node, n.Getitem):
            base = self.expr(node.node)
            arg = node.arg
            if isinstance(arg, n.Const) and isinstance(arg.value, str):
                return base.with_field(arg.value)
            if isinstance(arg, n.Slice):
                return base
            return base.element()
        if isinstance(node, n.Filter):
            inner = self.expr(node.node)
            whole_field_escape = False
            args = EMPTY
            for a in list(node.args) + [kw.value for kw in node.kwargs]:
                args = args.union(self.expr(a))
            if node.name in _NON_TEXT_FILTERS and inner:
                inner = Taint(inner.labels, frozenset(), False)
            if node.name == "replace" and inner:
                inner = Taint(inner.labels, frozenset(), False)
            if inner:
                whole_field_escape = node.name in _ESCAPE_FILTERS and isinstance(node.node, (n.Getattr, n.Getitem))
                inner = Taint(inner.labels, inner.removed, whole_field_escape)
            if node.name in {"map", "select", "reject", "selectattr", "rejectattr", "list", "reverse", "sort", "unique", "batch", "slice", "groupby", "dictsort"}:
                return inner.union(args)
            combined = inner.union(args)
            return Taint(combined.labels, combined.removed, inner.escaped and not bool(args))
        if isinstance(node, n.Call):
            target = node.node
            args = EMPTY
            for a in list(node.args) + [kw.value for kw in node.kwargs]:
                args = args.union(self.expr(a))
            if node.dyn_args is not None:
                args = args.union(self.expr(node.dyn_args))
            if isinstance(target, n.Getattr):
                recv = self.expr(target.node)
                if target.attr == "replace" and recv:
                    recv = Taint(recv.labels, frozenset(), False)
                # str methods and unknown methods both preserve taint (may-taint).
                combined = recv.union(args)
                return Taint(combined.labels, frozenset(), False)
            if isinstance(target, n.Name):
                if target.name == "namespace":
                    return EMPTY
                if target.name in self.macros:
                    return self.call_macro(target.name, node, capture=True)
            return args
        if isinstance(node, (n.Concat,)):
            out = EMPTY
            for x in node.nodes:
                out = out.union(self.expr(x))
            return Taint(out.labels, frozenset(), False)
        if isinstance(node, n.CondExpr):
            return self.expr(node.expr1).union(self.expr(node.expr2))
        if isinstance(node, (n.Add, n.Mod, n.Mul)):
            combined = self.expr(node.left).union(self.expr(node.right))
            return Taint(combined.labels, frozenset(), False)
        if isinstance(node, (n.List, n.Tuple)):
            out = EMPTY
            for x in node.items:
                out = out.union(self.expr(x))
            return out
        if isinstance(node, n.Dict):
            out = EMPTY
            for pair in node.items:
                out = out.union(self.expr(pair.value))
            return out
        if isinstance(node, (n.Compare, n.Test, n.Not, n.And, n.Or, n.Sub, n.Div, n.FloorDiv, n.Pow, n.Neg, n.Pos)):
            return EMPTY
        if isinstance(node, n.Slice):
            return EMPTY
        return EMPTY

    # -- role guards ---------------------------------------------------------
    def _role_ref(self, node) -> bool:
        n = self.nodes
        if isinstance(node, n.Getattr) and node.attr == "role":
            return True
        if isinstance(node, n.Getitem) and isinstance(node.arg, n.Const) and node.arg.value == "role":
            return True
        return False

    def role_guard(self, test) -> tuple[frozenset[str] | None, frozenset[str]]:
        """Return (positive roles, excluded roles) implied when ``test`` is true."""
        n = self.nodes
        if isinstance(test, n.And):
            lp, le = self.role_guard(test.left)
            rp, re_ = self.role_guard(test.right)
            pos = lp if rp is None else (rp if lp is None else lp & rp)
            return pos, le | re_
        if isinstance(test, n.Compare) and len(test.ops) == 1 and self._role_ref(test.expr):
            op = test.ops[0]
            if isinstance(op.expr, n.Const) and isinstance(op.expr.value, str):
                if op.op == "eq":
                    return frozenset({op.expr.value}), frozenset()
                if op.op == "ne":
                    return None, frozenset({op.expr.value})
            if op.op == "in" and isinstance(op.expr, (n.List, n.Tuple)):
                vals = {x.value for x in op.expr.items if isinstance(x, n.Const)}
                if len(vals) == len(op.expr.items):
                    return frozenset(vals), frozenset()
        if isinstance(test, n.Or):
            lp, _ = self.role_guard(test.left)
            rp, _ = self.role_guard(test.right)
            if lp is not None and rp is not None:
                return lp | rp, frozenset()
        return None, frozenset()

    def current_guard(self) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
        pos: frozenset[str] | None = None
        excl: frozenset[str] = frozenset()
        for p, e in self.guards:
            if p is not None:
                pos = p if pos is None else pos & p
            excl = excl | e
        return (tuple(sorted(pos)) if pos is not None else None), tuple(sorted(excl))

    # -- statements ----------------------------------------------------------
    def body(self, stmts, capture: list | None = None) -> Taint:
        out = EMPTY
        for s in stmts:
            out = out.union(self.stmt(s, capture))
        return out

    def output_expr(self, node, capture: list | None) -> Taint:
        """Handle one node inside ``{{ }}``; record literals and sinks."""
        n = self.nodes
        if isinstance(node, n.TemplateData):
            self.literals.append(node.data)
            return EMPTY
        if isinstance(node, n.Call) and isinstance(node.node, n.Name) and node.node.name in self.macros:
            return self.call_macro(node.node.name, node, capture=capture is not None, sink_capture=capture)
        self._collect_consts(node)
        t = self.expr(node)
        direct_numeric = isinstance(node, n.Filter) and node.name in _NON_TEXT_FILTERS
        if t and capture is None and not direct_numeric:
            pos, excl = self.current_guard()
            self.sinks.append(
                TaintSink(
                    lineno=getattr(node, "lineno", 0) or 0,
                    labels=tuple(sorted(t.labels)),
                    removed=tuple(sorted(t.removed)),
                    escaped=t.escaped,
                    role_guard=pos,
                    excluded_roles=excl,
                    expression=type(node).__name__,
                )
            )
        return t

    def _collect_consts(self, node) -> None:
        """Record string constants that the output expression can emit.

        Constants used only in comparisons, tests, filter/method arguments, or
        ``raise_exception`` messages are not emitted and are not markers.
        """
        n = self.nodes
        if isinstance(node, n.Const):
            if isinstance(node.value, str):
                self.literals.append(node.value)
        elif isinstance(node, n.Concat):
            for x in node.nodes:
                self._collect_consts(x)
        elif isinstance(node, (n.Add, n.Mod)):
            self._collect_consts(node.left)
            if isinstance(node, n.Add):
                self._collect_consts(node.right)
        elif isinstance(node, n.CondExpr):
            self._collect_consts(node.expr1)
            if node.expr2 is not None:
                self._collect_consts(node.expr2)
        elif isinstance(node, n.Filter):
            self._collect_consts(node.node)
            if node.name == "join":
                for a in node.args:
                    self._collect_consts(a)
        elif isinstance(node, n.Call) and isinstance(node.node, n.Getattr):
            self._collect_consts(node.node.node)
        elif isinstance(node, (n.List, n.Tuple)):
            for x in node.items:
                self._collect_consts(x)

    def call_macro(self, name: str, call, capture: bool, sink_capture: list | None = None) -> Taint:
        macro = self.macros[name]
        if self.macro_depth > 4:
            out = EMPTY
            for a in call.args:
                out = out.union(self.expr(a))
            return out
        saved = dict(self.env)
        for i, arg in enumerate(macro.args):
            t = self.expr(call.args[i]) if i < len(call.args) else EMPTY
            for kw in call.kwargs:
                if kw.key == arg.name:
                    t = t.union(self.expr(kw.value))
            self.env[arg.name] = t
        self.macro_depth += 1
        try:
            result = self.body(macro.body, sink_capture)
        finally:
            self.macro_depth -= 1
            for k in list(self.env):
                if k not in saved:
                    del self.env[k]
                else:
                    self.env[k] = saved[k]
        return result

    def stmt(self, s, capture: list | None) -> Taint:
        n = self.nodes
        if isinstance(s, n.Output):
            out = EMPTY
            for x in s.nodes:
                out = out.union(self.output_expr(x, capture))
            return out
        if isinstance(s, n.Assign):
            t = self.expr(s.node)
            self._assign(s.target, t)
            return EMPTY
        if isinstance(s, n.AssignBlock):
            t = self.body(s.body, capture=[])
            self._assign(s.target, t)
            return EMPTY
        if isinstance(s, n.For):
            it = self.expr(s.iter).element()
            self._assign(s.target, it)
            out = self.body(s.body, capture)
            out = out.union(self.body(s.else_ or [], capture))
            return out
        if isinstance(s, n.If):
            guard = self.role_guard(s.test)
            self.guards.append(guard)
            out = self.body(s.body, capture)
            self.guards.pop()
            neg_excl = guard[0] if guard[0] is not None and len(guard[0]) == 1 and not guard[1] else frozenset()
            self.guards.append((None, frozenset(neg_excl)))
            for elif_ in s.elif_:
                g2 = self.role_guard(elif_.test)
                self.guards.append(g2)
                out = out.union(self.body(elif_.body, capture))
                self.guards.pop()
            out = out.union(self.body(s.else_ or [], capture))
            self.guards.pop()
            return out
        if isinstance(s, n.Macro):
            self.macros[s.name] = s
            return EMPTY
        if isinstance(s, (n.Scope, n.ScopedEvalContextModifier, n.FilterBlock, n.With)):
            if isinstance(s, n.With):
                for target, value in zip(s.targets, s.values):
                    self._assign(target, self.expr(value))
            return self.body(s.body, capture)
        if isinstance(s, n.CallBlock):
            return self.body(s.body, capture)
        if isinstance(s, n.ExprStmt):
            self.expr(s.node)
            return EMPTY
        return EMPTY

    def _assign(self, target, taint: Taint) -> None:
        n = self.nodes
        if isinstance(target, n.Name):
            self.bind(target.name, taint)
        elif isinstance(target, n.NSRef):
            self.bind(f"{target.name}.{target.attr}", taint)
        elif isinstance(target, n.Tuple):
            for item in target.items:
                self._assign(item, taint)


def _special_token_texts(config: dict[str, object]) -> list[str]:
    out: list[str] = []
    for key, value in config.items():
        if key.endswith("_token") or key == "additional_special_tokens":
            values = value if isinstance(value, list) else [value]
            for v in values:
                text = v.get("content") if isinstance(v, dict) else v
                if isinstance(text, str) and text:
                    out.append(text)
    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for v in decoder.values():
            if isinstance(v, dict) and v.get("special") and isinstance(v.get("content"), str):
                out.append(v["content"])
    return out


def _marker_catalog(literals: list[str], special_tokens: list[str]) -> tuple[tuple[str, str], ...]:
    markers: dict[str, str] = {}
    joined = "".join(literals)
    for tok in special_tokens:
        if len(tok.strip()) >= 3 and tok in joined:
            markers.setdefault(tok, "special-token-emitted")
    for lit in literals:
        for cand in _extract_marker_candidates(lit):
            markers.setdefault(cand, "template-literal")
    for tok in special_tokens:
        if len(tok.strip()) >= 3:
            markers.setdefault(tok, "special-token-declared")
    return tuple(sorted(markers.items()))


def _neutralized(marker: str, sink: TaintSink) -> bool:
    if marker in sink.removed or any(r and r in marker for r in sink.removed if len(r) >= 3):
        return True
    if sink.escaped and re.search(r"[<>&'\"]", marker):
        return True
    return False


def _content_field(label: str) -> str | None:
    if label.startswith("messages[*]"):
        return label
    if label.startswith(("tools", "documents")):
        return label
    return None


def analyze_template_taint(template: str, config: dict[str, object] | None = None) -> TemplateTaintReport:
    """Run the whole-template taint analysis on a chat template string."""

    config = config or {}
    try:
        ast = _environment().parse(template)
    except Exception as exc:  # jinja2.TemplateSyntaxError and friends
        return TemplateTaintReport(False, f"{type(exc).__name__}: {exc}", (), (), ())
    analyzer = _Analyzer()
    # Iterate to a fixed point so taint that flows backwards through loops
    # (namespace attributes, re-assigned variables) reaches every sink.
    for _ in range(8):
        analyzer.changed = False
        analyzer.sinks = []
        analyzer.literals = []
        analyzer.body(ast.body)
        if not analyzer.changed:
            break
    markers = _marker_catalog(analyzer.literals, _special_token_texts(config))
    findings: dict[tuple[str, str], TaintFinding] = {}
    other: dict[tuple[str, str], TaintFinding] = {}
    for sink in analyzer.sinks:
        roles = tuple(sorted(r for r in ATTACKER_ROLES if sink.reachable_by(r)))
        for label in sink.labels:
            field_name = _content_field(label)
            if field_name is None or field_name.endswith(".role") or field_name == "messages[*]":
                continue
            for marker, source in markers:
                if _neutralized(marker, sink):
                    continue
                finding = TaintFinding(field_name, marker, source, roles, sink.lineno, sink.expression)
                bucket = findings if roles or field_name.startswith(("tools", "documents")) else other
                bucket.setdefault((field_name, marker), finding)
    return TemplateTaintReport(
        parsed=True,
        error="",
        sinks=tuple(analyzer.sinks),
        markers=markers,
        findings=tuple(findings[k] for k in sorted(findings)),
        other_role_findings=tuple(other[k] for k in sorted(other) if k not in findings),
    )


__all__ = [
    "ATTACKER_ROLES",
    "TaintFinding",
    "TaintSink",
    "TemplateTaintReport",
    "analyze_template_taint",
]

