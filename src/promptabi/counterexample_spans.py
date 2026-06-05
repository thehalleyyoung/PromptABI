"""Connect SMT counterexamples to source spans (step 234).

A solver counterexample is an assignment like ``{"reserved_completion": 256,
"context": 900}``.  On its own it tells a developer *what* values break the
contract but not *where* in their artifacts those values come from.  This module
closes that loop: it maps each counterexample variable back to the artifact and
one-based source span that introduced it, so a finding can point at
``tokenizer_config.json:12`` instead of an opaque variable name.

The mapping is provided as :class:`VariableProvenance` records (variable name ->
:class:`~promptabi.diagnostics.ArtifactRef` + :class:`~promptabi.diagnostics.SourceSpan`).
:func:`annotate_counterexample` joins a :class:`~promptabi.formal.SolverResult`
against that provenance and reports both the located variables and any variable
that lacks provenance (so missing source maps are visible rather than silently
dropped).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .diagnostics import ArtifactRef, SourceSpan
from .formal import SolverResult, SolverStatus

COUNTEREXAMPLE_SPANS_VERSION = "promptabi.counterexample-spans.v1"


class CounterexampleSpanError(ValueError):
    """Raised when a result has no counterexample to annotate."""


@dataclass(frozen=True, slots=True)
class VariableProvenance:
    variable: str
    artifact: ArtifactRef
    span: SourceSpan
    snippet: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "variable": self.variable,
            "artifact": self.artifact.to_dict(),
            "span": self.span.to_dict(),
        }
        if self.snippet is not None:
            data["snippet"] = self.snippet
        return data


@dataclass(frozen=True, slots=True)
class LocatedAssignment:
    variable: str
    value: object
    artifact: ArtifactRef
    span: SourceSpan
    snippet: str | None = None

    def location(self) -> str:
        path = self.span.path
        return f"{path}:{self.span.start_line}:{self.span.start_column}"

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "variable": self.variable,
            "value": self.value,
            "artifact": self.artifact.to_dict(),
            "span": self.span.to_dict(),
            "location": self.location(),
        }
        if self.snippet is not None:
            data["snippet"] = self.snippet
        return data


@dataclass(frozen=True, slots=True)
class AnnotatedCounterexample:
    version: str
    located: tuple[LocatedAssignment, ...]
    unmapped: tuple[str, ...] = field(default=())

    @property
    def fully_located(self) -> bool:
        return not self.unmapped

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fully_located": self.fully_located,
            "located": [item.to_dict() for item in self.located],
            "unmapped": list(self.unmapped),
        }


def annotate_counterexample(
    result: SolverResult,
    provenance: Sequence[VariableProvenance] | Mapping[str, VariableProvenance],
) -> AnnotatedCounterexample:
    """Join a SAT counterexample assignment against variable provenance."""

    if result.status is not SolverStatus.SAT or result.assignment is None:
        raise CounterexampleSpanError("solver result carries no SAT counterexample to annotate")

    if isinstance(provenance, Mapping):
        index = dict(provenance)
    else:
        index = {record.variable: record for record in provenance}

    located: list[LocatedAssignment] = []
    unmapped: list[str] = []
    for variable, value in sorted(result.assignment.items()):
        record = index.get(variable)
        if record is None:
            unmapped.append(variable)
            continue
        located.append(
            LocatedAssignment(
                variable=variable,
                value=value,
                artifact=record.artifact,
                span=record.span,
                snippet=record.snippet,
            )
        )
    return AnnotatedCounterexample(
        version=COUNTEREXAMPLE_SPANS_VERSION,
        located=tuple(located),
        unmapped=tuple(unmapped),
    )


def render_annotated_counterexample_json(annotated: AnnotatedCounterexample) -> str:
    return json.dumps(annotated.to_dict(), indent=2, sort_keys=True) + "\n"


def render_annotated_counterexample_text(annotated: AnnotatedCounterexample) -> str:
    lines = [
        f"PromptABI counterexample source map ({annotated.version})",
        f"fully located: {annotated.fully_located}",
    ]
    for item in annotated.located:
        snippet = f"  {item.snippet}" if item.snippet else ""
        lines.append(f"  {item.variable} = {item.value!r} @ {item.location()}{snippet}")
    for variable in annotated.unmapped:
        lines.append(f"  {variable} = <unmapped: no source provenance>")
    return "\n".join(lines) + "\n"
