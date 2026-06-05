"""Abstract interpretation of supported chat-template dialects.

The bounded symbolic executor in :mod:`promptabi.chat_templates` enumerates a
finite number of concrete render paths (for example two messages).  Abstract
interpretation complements it by proving properties that hold for *any* number of
messages or tools: it lifts each template into a *count multiplicity* lattice and
computes, for every special control marker, an over-approximation of how many
times it can be emitted across all executions -- including unbounded loops, which
are summarized with a Kleene-style closure rather than unrolled.

This lets PromptABI prove structural invariants such as "every message frame is
balanced (one open marker per close marker) regardless of conversation length"
and "the assistant generation header is emitted at most once", which a bounded
unrolling can only ever sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .chat_templates import (
    ChatTemplateParseResult,
    _ExpressionNode,
    _ForNode,
    _IfNode,
    _LiteralNode,
    _lex_symbolic_segments,
    _SetNode,
    _SymbolicParser,
    parse_hf_tokenizer_config_chat_template,
)
from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace


TEMPLATE_ABSTRACT_INTERPRETATION_VERSION = "promptabi.template-abstract-interpretation.v1"


class TemplateAbstractInterpretationError(ValueError):
    """Raised when a template cannot be abstractly interpreted."""


class TemplateInvariantKind(StrEnum):
    """Structural invariants the abstract interpreter can refute."""

    MARKER_IMBALANCE = "marker-imbalance"
    GENERATION_PROMPT_UNBOUNDED = "generation-prompt-unbounded"


@dataclass(frozen=True, slots=True)
class AbstractCount:
    """A count abstraction: an interval over the naturals with optional infinity.

    ``upper is None`` denotes an unbounded number of emissions (a loop closure).
    """

    lower: int
    upper: int | None

    @property
    def bounded(self) -> bool:
        return self.upper is not None

    def join(self, other: "AbstractCount") -> "AbstractCount":
        upper = None if self.upper is None or other.upper is None else max(self.upper, other.upper)
        return AbstractCount(min(self.lower, other.lower), upper)

    def add(self, other: "AbstractCount") -> "AbstractCount":
        upper = None if self.upper is None or other.upper is None else self.upper + other.upper
        return AbstractCount(self.lower + other.lower, upper)

    def star(self) -> "AbstractCount":
        """Closure under 0..n loop iterations."""

        if self.upper == 0:
            return AbstractCount(0, 0)
        return AbstractCount(0, None)

    def to_dict(self) -> dict[str, object]:
        return {"lower": self.lower, "upper": self.upper}

    def describe(self) -> str:
        return f"[{self.lower}, {'inf' if self.upper is None else self.upper}]"


_ZERO = AbstractCount(0, 0)


@dataclass(frozen=True, slots=True)
class TemplateInvariantViolation:
    """One refuted template invariant with a replayable witness."""

    kind: TemplateInvariantKind
    message: str
    marker: str
    witness: WitnessTrace
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "marker": self.marker,
            "message": self.message,
            "suggestion": self.suggestion,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TemplateAbstractInterpretation:
    """Result of abstractly interpreting a chat-template dialect."""

    name: str
    supported: bool
    marker_counts: tuple[tuple[str, AbstractCount], ...]
    marker_pairs: tuple[tuple[str, str], ...]
    violations: tuple[TemplateInvariantViolation, ...]
    abstentions: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.supported and not self.violations

    def count_for(self, marker: str) -> AbstractCount:
        for key, value in self.marker_counts:
            if key == marker:
                return value
        return _ZERO

    def to_dict(self) -> dict[str, object]:
        return {
            "abstentions": list(self.abstentions),
            "marker_counts": {marker: count.to_dict() for marker, count in self.marker_counts},
            "marker_pairs": [list(pair) for pair in self.marker_pairs],
            "name": self.name,
            "ok": self.ok,
            "supported": self.supported,
            "version": TEMPLATE_ABSTRACT_INTERPRETATION_VERSION,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def interpret_chat_template(
    parsed: ChatTemplateParseResult,
    *,
    name: str = "chat-template",
) -> TemplateAbstractInterpretation:
    """Abstractly interpret a parsed chat template over the count lattice."""

    artifact = ArtifactRef(kind="chat-template", name=name, path="memory://chat-template")
    if parsed.unsupported_constructs:
        abstentions = tuple(item.expression for item in parsed.unsupported_constructs)
        return TemplateAbstractInterpretation(
            name=name,
            supported=False,
            marker_counts=(),
            marker_pairs=(),
            violations=(),
            abstentions=abstentions,
        )

    segments = _lex_symbolic_segments(parsed.template_source)
    parser = _SymbolicParser(segments)
    nodes = parser.parse()
    if parser.abstentions:
        abstentions = tuple(item.expression for item in parser.abstentions)
        return TemplateAbstractInterpretation(
            name=name,
            supported=False,
            marker_counts=(),
            marker_pairs=(),
            violations=(),
            abstentions=abstentions,
        )

    markers = _collect_markers(parsed)
    pairs = _marker_pairs(tuple(dict.fromkeys(token.text for token in parsed.special_tokens if token.text)))
    interpreter = _AbstractInterpreter(markers, pairs, parsed, artifact)
    total = interpreter.interpret(nodes)

    violations = list(interpreter.violations)
    for excerpt in parsed.generation_prompt_excerpts:
        count = total.get(excerpt, _ZERO)
        if not count.bounded:
            violations.append(
                _violation(
                    TemplateInvariantKind.GENERATION_PROMPT_UNBOUNDED,
                    f"generation-prompt marker {excerpt!r} can be emitted unboundedly ({count.describe()})",
                    excerpt,
                    artifact,
                    step_in=f"abstract count {excerpt}",
                    step_out=count.describe(),
                    suggestion="Emit the assistant generation header outside message loops, guarded once.",
                )
            )

    marker_counts = tuple(sorted(((marker, total.get(marker, _ZERO)) for marker in markers), key=lambda item: item[0]))
    return TemplateAbstractInterpretation(
        name=name,
        supported=True,
        marker_counts=marker_counts,
        marker_pairs=pairs,
        violations=tuple(violations),
        abstentions=(),
    )


def interpret_chat_template_file(path: str | Path, *, name: str | None = None) -> TemplateAbstractInterpretation:
    """Parse a Hugging Face tokenizer_config.json and abstractly interpret it."""

    parsed = parse_hf_tokenizer_config_chat_template(path)
    return interpret_chat_template(parsed, name=name or Path(path).parent.name or "chat-template")


def render_template_abstract_interpretation_json(report: TemplateAbstractInterpretation) -> str:
    """Render an abstract interpretation as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_template_abstract_interpretation_text(report: TemplateAbstractInterpretation) -> str:
    """Render an abstract interpretation for CLI users."""

    lines = [
        "PromptABI chat-template abstract interpretation",
        f"name: {report.name}",
        f"supported: {report.supported}",
        f"status: {'PROVEN' if report.ok else 'REFUTED' if report.supported else 'ABSTAINED'}",
    ]
    if not report.supported:
        lines.append(f"abstentions: {len(report.abstentions)}")
        for abstention in report.abstentions:
            lines.append(f"  - {abstention}")
        return "\n".join(lines) + "\n"
    lines.append("marker multiplicities:")
    for marker, count in report.marker_counts:
        lines.append(f"  {marker} -> {count.describe()}")
    if report.ok:
        lines.append("invariants: all proven (frames balanced, generation prompt bounded)")
        return "\n".join(lines) + "\n"
    lines.append(f"violations: {len(report.violations)}")
    for violation in report.violations:
        lines.append(f"REFUTED {violation.kind.value} [{violation.marker}]: {violation.message}")
        lines.append(f"  suggestion: {violation.suggestion}")
    return "\n".join(lines) + "\n"


class _AbstractInterpreter:
    def __init__(
        self,
        markers: tuple[str, ...],
        pairs: tuple[tuple[str, str], ...],
        parsed: ChatTemplateParseResult,
        artifact: ArtifactRef,
    ) -> None:
        self.markers = markers
        self.pairs = pairs
        self.parsed = parsed
        self.artifact = artifact
        self.violations: list[TemplateInvariantViolation] = []
        self._loop_index = 0

    def interpret(self, nodes: tuple[object, ...]) -> dict[str, AbstractCount]:
        counts: dict[str, AbstractCount] = {marker: _ZERO for marker in self.markers}
        for node in nodes:
            self._step(node, counts)
        return counts

    def _step(self, node: object, counts: dict[str, AbstractCount]) -> None:
        if isinstance(node, _LiteralNode):
            for marker in self.markers:
                occurrences = _count_occurrences(node.text, marker)
                counts[marker] = counts[marker].add(AbstractCount(occurrences, occurrences))
        elif isinstance(node, _ExpressionNode | _SetNode):
            return
        elif isinstance(node, _ForNode):
            body_counts = self.interpret(node.body)
            self._check_balance(node, body_counts)
            for marker in self.markers:
                counts[marker] = counts[marker].add(body_counts[marker].star())
        elif isinstance(node, _IfNode):
            branch_counts = [self.interpret(branch.body) for branch in node.branches]
            for marker in self.markers:
                joined = _ZERO
                for branch in branch_counts:
                    joined = joined.join(branch[marker])
                counts[marker] = counts[marker].add(joined)

    def _check_balance(self, node: _ForNode, body_counts: dict[str, AbstractCount]) -> None:
        self._loop_index += 1
        for open_marker, close_marker in self.pairs:
            open_count = body_counts.get(open_marker, _ZERO)
            close_count = body_counts.get(close_marker, _ZERO)
            if open_count != close_count:
                self.violations.append(
                    _violation(
                        TemplateInvariantKind.MARKER_IMBALANCE,
                        (
                            f"loop over '{node.iterable}' emits {open_marker!r} {open_count.describe()} times but "
                            f"{close_marker!r} {close_count.describe()} times per iteration; frames desync at scale"
                        ),
                        f"{open_marker}|{close_marker}",
                        self.artifact,
                        step_in=f"loop#{self._loop_index} over {node.iterable}",
                        step_out=f"{open_marker}={open_count.describe()} {close_marker}={close_count.describe()}",
                        suggestion="Emit exactly one open and one close marker per message frame inside the loop body.",
                    )
                )


def _collect_markers(parsed: ChatTemplateParseResult) -> tuple[str, ...]:
    markers: list[str] = []
    for token in parsed.special_tokens:
        if token.text and token.text not in markers:
            markers.append(token.text)
    for excerpt in parsed.generation_prompt_excerpts:
        if excerpt and excerpt not in markers:
            markers.append(excerpt)
    return tuple(markers)


def _marker_pairs(markers: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for marker in markers:
        lowered = marker.lower()
        if "start" in lowered or "begin" in lowered:
            close = _matching_close(marker, markers)
            if close is not None:
                pairs.append((marker, close))
    return tuple(pairs)


def _matching_close(open_marker: str, markers: tuple[str, ...]) -> str | None:
    lowered = open_marker.lower()
    for candidate in markers:
        candidate_lower = candidate.lower()
        if "end" not in candidate_lower:
            continue
        if lowered.replace("start", "end").replace("begin", "end") == candidate_lower:
            return candidate
    for candidate in markers:
        if "end" in candidate.lower():
            return candidate
    return None


def _count_occurrences(text: str, marker: str) -> int:
    if not marker:
        return 0
    return text.count(marker)


def _violation(
    kind: TemplateInvariantKind,
    message: str,
    marker: str,
    artifact: ArtifactRef,
    *,
    step_in: str,
    step_out: str,
    suggestion: str,
) -> TemplateInvariantViolation:
    return TemplateInvariantViolation(
        kind=kind,
        message=message,
        marker=marker,
        witness=WitnessTrace(
            summary=f"abstract interpretation refutes invariant: {kind.value}",
            steps=(
                WitnessStep(action="lift template into count lattice", input=step_in, output=step_out),
                WitnessStep(action="emit minimal template fix", input=kind.value, output=suggestion),
            ),
            artifacts=(artifact,),
            minimal_fixes=(suggestion,),
        ),
        suggestion=suggestion,
    )
