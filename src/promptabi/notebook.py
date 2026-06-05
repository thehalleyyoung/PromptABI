"""Notebook-friendly visualizations for PromptABI's core abstractions."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .artifacts import StopPolicyArtifact
from .budgets import TokenBudgetReport, TokenBudgetVisualization, build_token_budget_visualization
from .chat_templates import (
    ChatTemplateParseResult,
    ChatTemplateRenderError,
    render_chat_template_supported_fragment,
    symbolically_execute_chat_template,
)
from .diagnostics import CheckMode
from .formal import DeterministicFiniteAutomaton, FiniteContractProblem
from .stop_analysis import analyze_stop_policy_tokenizer
from .tokenizers import TokenizerAdapter


@dataclass(frozen=True, slots=True)
class NotebookSection:
    """One deterministic section inside a notebook visualization."""

    title: str
    rows: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("notebook section title must be non-empty")
        object.__setattr__(self, "rows", tuple((key, value) for key, value in self.rows))
        if any(not key for key, _value in self.rows):
            raise ValueError("notebook section row keys must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"title": self.title, "rows": [{"label": key, "value": value} for key, value in self.rows]}


@dataclass(frozen=True, slots=True)
class NotebookVisualization:
    """A dependency-free rich repr object for Jupyter, terminals, and tests."""

    title: str
    mode: CheckMode
    summary: str
    sections: tuple[NotebookSection, ...]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("notebook visualization title must be non-empty")
        if not self.summary:
            raise ValueError("notebook visualization summary must be non-empty")
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "payload", dict(sorted(self.payload.items())))

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "mode": self.mode.value,
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections],
            "payload": dict(self.payload),
        }

    def render_text(self) -> str:
        lines = [f"{self.title} [{self.mode.label}]", self.summary]
        for section in self.sections:
            lines.append(f"{section.title}:")
            for key, value in section.rows:
                lines.append(f"  {key}: {_display_value(value)}")
        return "\n".join(lines) + "\n"

    def render_html(self) -> str:
        sections = "\n".join(_render_section_html(section) for section in self.sections)
        payload = html.escape(json.dumps(self.payload, indent=2, sort_keys=True), quote=True)
        return "\n".join(
            (
                '<div class="promptabi-notebook">',
                "<style>",
                _NOTEBOOK_CSS,
                "</style>",
                f"<h3>{html.escape(self.title, quote=True)}</h3>",
                '<div class="promptabi-mode">'
                f"{html.escape(self.mode.label, quote=True)}"
                "</div>",
                f"<p>{html.escape(self.summary, quote=True)}</p>",
                sections,
                "<details><summary>Machine-readable payload</summary>",
                f"<pre>{payload}</pre>",
                "</details>",
                "</div>",
            )
        )

    def _repr_html_(self) -> str:
        return self.render_html()

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict[str, str]:
        del include, exclude
        return {"text/plain": self.render_text(), "text/html": self.render_html()}

    def __str__(self) -> str:
        return self.render_text()


def visualize_tokenization(
    text: str,
    tokenizer: TokenizerAdapter,
    *,
    add_special_tokens: bool = False,
) -> NotebookVisualization:
    """Visualize tokenizer normalization, token ids, surfaces, and byte spans."""

    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    rows = tuple(
        (
            str(index),
            {
                "id": token.token_id,
                "text": token.text,
                "byte_span": token.byte_span,
                "special": token.special,
                "added": token.added,
            },
        )
        for index, token in enumerate(encoded.tokens)
    )
    return NotebookVisualization(
        title="Tokenization",
        mode=CheckMode.BOUNDED,
        summary=f"{encoded.backend.value} encoded {len(text)} chars into {len(encoded.tokens)} token(s).",
        sections=(
            NotebookSection(
                "Input",
                (
                    ("text", text),
                    ("normalized", encoded.normalized_text),
                    ("normalization", encoded.normalization_steps or ("identity",)),
                ),
            ),
            NotebookSection("Tokens", rows or (("tokens", "<none>"),)),
        ),
        payload=encoded.to_dict(),
    )


def visualize_template_rendering(
    parsed: ChatTemplateParseResult,
    messages: Sequence[Mapping[str, object]],
    *,
    add_generation_prompt: bool = False,
    tools: Sequence[Mapping[str, object]] | None = None,
    variables: Mapping[str, object] | None = None,
) -> NotebookVisualization:
    """Visualize concrete rendering plus bounded symbolic template paths."""

    execution = symbolically_execute_chat_template(parsed)
    render_error: str | None = None
    try:
        rendered = render_chat_template_supported_fragment(
            parsed,
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            variables=variables,
        )
    except ChatTemplateRenderError as exc:
        rendered = ""
        render_error = str(exc)
    mode = CheckMode.ABSTAINING if render_error or execution.abstentions else CheckMode.BOUNDED
    path_rows = tuple(
        (
            str(index),
            {
                "conditions": path.conditions,
                "loop_iterations": path.loop_iterations,
                "rendered_pattern": path.rendered_pattern,
            },
        )
        for index, path in enumerate(execution.paths)
    )
    abstention_rows = tuple((item.kind, item.reason) for item in execution.abstentions)
    sections = [
        NotebookSection(
            "Concrete render",
            (
                ("rendered", rendered if render_error is None else f"abstained: {render_error}"),
                ("message_count", len(messages)),
                ("tool_count", len(tools or ())),
                ("add_generation_prompt", add_generation_prompt),
            ),
        ),
        NotebookSection("Symbolic paths", path_rows or (("paths", "<none>"),)),
    ]
    if abstention_rows:
        sections.append(NotebookSection("Abstentions", abstention_rows))
    payload = {
        "parse": parsed.to_dict(),
        "rendered": rendered,
        "render_error": render_error,
        "symbolic_execution": execution.to_dict(),
    }
    return NotebookVisualization(
        title="Template rendering",
        mode=mode,
        summary=f"{len(execution.paths)} bounded path(s), {len(execution.abstentions)} abstention(s).",
        sections=tuple(sections),
        payload=payload,
    )


def visualize_stop_reachability(
    stop_policy: StopPolicyArtifact,
    tokenizer: TokenizerAdapter,
) -> NotebookVisualization:
    """Visualize stop strings and stop token ids against concrete tokenizer behavior."""

    report = analyze_stop_policy_tokenizer(stop_policy, tokenizer)
    sequence_rows = tuple(
        (
            sequence.stop_sequence,
            {
                "token_ids": sequence.token_ids,
                "decoded_text": sequence.decoded_text,
                "exact_round_trip": sequence.exact_round_trip,
                "normalized_round_trip": sequence.normalized_round_trip,
                "special_token_ids": sequence.special_token_ids,
                "added_token_ids": sequence.added_token_ids,
            },
        )
        for sequence in report.sequences
    )
    token_id_rows = tuple(
        (
            str(item.token_id),
            {"decodable": item.decodable, "decoded_text": item.decoded_text, "error": item.error},
        )
        for item in report.token_ids
    )
    payload = {
        "tokenizer_backend": report.tokenizer_backend,
        "sequences": [_stop_sequence_to_dict(sequence) for sequence in report.sequences],
        "token_ids": [_stop_token_id_to_dict(item) for item in report.token_ids],
        "collisions": [_stop_collision_to_dict(item) for item in report.collisions],
        "normalization_collisions": [_stop_collision_to_dict(item) for item in report.normalization_collisions],
    }
    return NotebookVisualization(
        title="Stop reachability",
        mode=CheckMode.SOUND,
        summary=(
            f"{len(report.sequences)} string stop(s), {len(report.token_ids)} token-id stop(s), "
            f"{len(report.unreachable_token_ids)} unreachable token id(s)."
        ),
        sections=(
            NotebookSection("Stop strings", sequence_rows or (("sequences", "<none>"),)),
            NotebookSection("Stop token ids", token_id_rows or (("token_ids", "<none>"),)),
            NotebookSection(
                "Collisions",
                tuple((item.level, f"{item.relation}: {item.shorter!r} vs {item.longer!r}") for item in report.collisions)
                or (("collisions", "<none>"),),
            ),
        ),
        payload=payload,
    )


def visualize_grammar_product(
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    *,
    max_depth: int | None = None,
) -> NotebookVisualization:
    """Visualize a lazy DFA product/intersection witness for grammar fragments."""

    result = left.intersection_witness(right, max_depth=max_depth)
    payload = {
        "left": left.to_dict(),
        "right": right.to_dict(),
        "intersection": result.to_dict(),
        "max_depth": max_depth,
    }
    witness = result.witness.to_dict() if result.witness is not None else None
    return NotebookVisualization(
        title="Grammar product",
        mode=CheckMode.COMPLETE if max_depth is None else CheckMode.BOUNDED,
        summary="intersection witness found" if result.found else "no intersection witness found",
        sections=(
            NotebookSection(
                "Product search",
                (
                    ("left", left.name),
                    ("right", right.name),
                    ("found", result.found),
                    ("explored_states", result.explored_states),
                    ("explored_transitions", result.explored_transitions),
                    ("max_depth", max_depth if max_depth is not None else "unbounded"),
                ),
            ),
            NotebookSection("Witness", tuple((key, value) for key, value in (witness or {"witness": "<none>"}).items())),
        ),
        payload=payload,
    )


def visualize_smt_constraints(
    problem: FiniteContractProblem,
    *,
    prefer_z3: bool = True,
    max_assignments: int | None = None,
    timeout_seconds: float | None = None,
) -> NotebookVisualization:
    """Visualize finite contract domains, constraints, and solver conclusion."""

    result = problem.solve(
        prefer_z3=prefer_z3,
        max_assignments=max_assignments,
        timeout_seconds=timeout_seconds,
    )
    mode = CheckMode.Z3_BACKED_SMT if result.backend.value == "z3" else CheckMode.BOUNDED
    if result.reason is not None:
        mode = CheckMode.ABSTAINING
    payload = {"problem": problem.to_dict(), "result": result.to_dict()}
    return NotebookVisualization(
        title="SMT constraints",
        mode=mode,
        summary=f"{result.status.value} via {result.backend.value}: {result.conclusion.value}.",
        sections=(
            NotebookSection(
                "Problem",
                (
                    ("name", problem.name),
                    ("variables", len(problem.variables)),
                    ("constraints", len(problem.constraints)),
                ),
            ),
            NotebookSection(
                "Solver result",
                tuple((key, value) for key, value in result.to_dict().items()),
            ),
        ),
        payload=payload,
    )


def visualize_truncation(
    budget: TokenBudgetReport | TokenBudgetVisualization,
) -> NotebookVisualization:
    """Visualize token-budget arithmetic and truncation survival decisions."""

    visualization = budget if isinstance(budget, TokenBudgetVisualization) else build_token_budget_visualization(budget)
    if visualization is None:
        return NotebookVisualization(
            title="Truncation behavior",
            mode=CheckMode.ABSTAINING,
            summary="token-budget visualization is unavailable because no reservation was modeled.",
            sections=(NotebookSection("Truncation", (("status", "abstained"),)),),
            payload={"visualization": None},
        )
    return NotebookVisualization(
        title="Truncation behavior",
        mode=CheckMode.BOUNDED,
        summary=(
            f"{visualization.framework}:{visualization.strategy} budget keeps "
            f"{len([row for row in visualization.rows if row.status == 'kept'])} segment(s) "
            f"and drops {len(visualization.dropped_fields)}."
        ),
        sections=(
            NotebookSection(
                "Budget",
                (
                    ("source", visualization.budget_source),
                    ("context", visualization.max_context_tokens),
                    ("reserved", visualization.reserved_total),
                    ("input_budget", visualization.input_budget_tokens),
                    ("overflow", visualization.overflow_tokens if visualization.overflow_tokens is not None else "unknown"),
                    ("must_survive", visualization.must_survive_status),
                ),
            ),
            NotebookSection(
                "Segments",
                tuple((row.name, row.to_dict()) for row in visualization.rows) or (("segments", "<none>"),),
            ),
        ),
        payload=visualization.to_dict(),
    )


def render_notebook_visualization_text(visualization: NotebookVisualization) -> str:
    """Render a notebook visualization as deterministic plain text."""

    return visualization.render_text()


def render_notebook_visualization_html(visualization: NotebookVisualization) -> str:
    """Render a notebook visualization as self-contained escaped HTML."""

    return visualization.render_html()


def _display_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), sort_keys=True)


def _render_section_html(section: NotebookSection) -> str:
    rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(key, quote=True)}</th>"
        f"<td><code>{html.escape(_display_value(value), quote=True)}</code></td>"
        "</tr>"
        for key, value in section.rows
    )
    return "\n".join(
        (
            "<section>",
            f"<h4>{html.escape(section.title, quote=True)}</h4>",
            "<table>",
            "<tbody>",
            rows,
            "</tbody>",
            "</table>",
            "</section>",
        )
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_jsonable(item) for item in value), key=repr)
    return value


def _stop_sequence_to_dict(sequence) -> dict[str, object]:
    return {
        "stop_sequence": sequence.stop_sequence,
        "normalized_sequence": sequence.normalized_sequence,
        "utf8_bytes": list(sequence.utf8_bytes),
        "token_ids": list(sequence.token_ids),
        "token_texts": list(sequence.token_texts),
        "byte_spans": [list(span) if span is not None else None for span in sequence.byte_spans],
        "decoded_text": sequence.decoded_text,
        "exact_round_trip": sequence.exact_round_trip,
        "normalized_round_trip": sequence.normalized_round_trip,
        "normalization_steps": list(sequence.normalization_steps),
        "special_token_ids": list(sequence.special_token_ids),
        "added_token_ids": list(sequence.added_token_ids),
    }


def _stop_token_id_to_dict(item) -> dict[str, object]:
    data: dict[str, object] = {"token_id": item.token_id, "decodable": item.decodable}
    if item.decoded_text is not None:
        data["decoded_text"] = item.decoded_text
    if item.error is not None:
        data["error"] = item.error
    return data


def _stop_collision_to_dict(item) -> dict[str, object]:
    return {
        "level": item.level,
        "relation": item.relation,
        "shorter": item.shorter,
        "longer": item.longer,
        "witness": item.witness,
    }


_NOTEBOOK_CSS = """
.promptabi-notebook {
  border: 1px solid #d0d7de;
  border-radius: 8px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 0.75rem;
}
.promptabi-notebook h3, .promptabi-notebook h4 { margin: 0.4rem 0; }
.promptabi-notebook table { border-collapse: collapse; width: 100%; }
.promptabi-notebook th { text-align: left; width: 14rem; }
.promptabi-notebook th, .promptabi-notebook td { border-top: 1px solid #d8dee4; padding: 0.35rem; vertical-align: top; }
.promptabi-notebook code, .promptabi-notebook pre { white-space: pre-wrap; word-break: break-word; }
.promptabi-mode { display: inline-block; background: #ddf4ff; border-radius: 999px; padding: 0.1rem 0.5rem; }
""".strip()
