"""Conformance badges for OpenAI-compatible servers (step 281).

Many inference servers (vLLM, TGI, llama.cpp server, LiteLLM) advertise an
"OpenAI-compatible" API.  This module evaluates such a server's *declared*
capabilities and observed responses against the OpenAI chat-completions contract
and emits a conformance badge: which capabilities (chat, tool calls, JSON mode,
streaming, logprobs, stop sequences) are conformant, partial, or missing.  The
badge is a shareable, shields.io-style artifact a server README can publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

OPENAI_CONFORMANCE_VERSION = "promptabi.openai-conformance.v1"

_CAPABILITIES = (
    "chat_completions",
    "tool_calls",
    "json_mode",
    "streaming",
    "stop_sequences",
    "logprobs",
)


class ConformanceState(StrEnum):
    CONFORMANT = "conformant"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    capability: str
    state: ConformanceState
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "state": self.state.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ServerConformanceBadge:
    version: str
    server: str
    conformant: int
    total: int
    reports: tuple[CapabilityReport, ...] = field(default=())

    @property
    def color(self) -> str:
        if self.conformant == self.total:
            return "green"
        if self.conformant == 0:
            return "red"
        return "yellow"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "server": self.server,
            "conformant": self.conformant,
            "total": self.total,
            "color": self.color,
            "reports": [r.to_dict() for r in self.reports],
        }

    def to_shields_endpoint(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "label": "openai-compat",
            "message": f"{self.conformant}/{self.total}",
            "color": self.color,
        }


def evaluate_server(
    server: str,
    declared: dict[str, bool],
    observed: dict[str, bool],
) -> ServerConformanceBadge:
    reports: list[CapabilityReport] = []
    conformant = 0
    for cap in _CAPABILITIES:
        d = declared.get(cap, False)
        o = observed.get(cap, False)
        if d and o:
            state = ConformanceState.CONFORMANT
            conformant += 1
        elif d and not o:
            state = ConformanceState.PARTIAL
        elif not d and o:
            # Works but undeclared -> partial (undocumented behavior).
            state = ConformanceState.PARTIAL
        else:
            state = ConformanceState.MISSING
        detail = f"declared={d} observed={o}"
        reports.append(CapabilityReport(cap, state, detail))

    return ServerConformanceBadge(
        version=OPENAI_CONFORMANCE_VERSION,
        server=server,
        conformant=conformant,
        total=len(_CAPABILITIES),
        reports=tuple(reports),
    )


def render_server_badge_text(badge: ServerConformanceBadge) -> str:
    lines = [
        f"PromptABI OpenAI-compat conformance ({badge.version})",
        f"server: {badge.server}",
        f"status: {badge.color.upper()} {badge.conformant}/{badge.total}",
    ]
    for r in badge.reports:
        lines.append(f"  [{r.state.value}] {r.capability}: {r.detail}")
    return "\n".join(lines) + "\n"
