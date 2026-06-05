"""Package-level differential provider fixtures for prompt packs (step 250).

A reusable prompt pack ships *example* prompts (a system message, a few turns,
maybe a tool list).  The pack claims to behave identically on every provider it
declares support for.  This module turns that claim into a *differential test*:
each declared provider supplies a tiny, deterministic rendering fixture, the pack
example is rendered through every one, and any structural divergence -- a role
header that appears on one provider but not another, a special token that leaks,
a differing turn count -- is reported as a finding.

The rendering model here is intentionally small and self-contained (no network,
no model weights): a :class:`ProviderRenderFixture` describes how a provider wraps
roles and joins turns.  The point is to *prove* that a pack either renders to the
same role/segment structure everywhere, or to pinpoint exactly where it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROMPT_PACK_DIFFERENTIAL_VERSION = "promptabi.prompt-pack-differential.v1"


class DifferentialFindingKind(StrEnum):
    ROLE_SET_DIVERGENCE = "role-set-divergence"
    TURN_COUNT_DIVERGENCE = "turn-count-divergence"
    SPECIAL_TOKEN_LEAK = "special-token-leak"
    EMPTY_RENDER = "empty-render"


@dataclass(frozen=True, slots=True)
class PackMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class PackExample:
    name: str
    messages: tuple[PackMessage, ...]


@dataclass(frozen=True, slots=True)
class ProviderRenderFixture:
    """Deterministic description of how one provider renders a turn."""

    provider: str
    role_prefixes: dict[str, str]
    role_suffix: str = "\n"
    control_tokens: tuple[str, ...] = ()

    def render(self, example: PackExample) -> "RenderedExample":
        segments: list[tuple[str, str]] = []
        for msg in example.messages:
            prefix = self.role_prefixes.get(msg.role, f"<<{msg.role}>>")
            segments.append((msg.role, prefix + msg.content + self.role_suffix))
        text = "".join(seg for _, seg in segments)
        return RenderedExample(self.provider, tuple(segments), text)


@dataclass(frozen=True, slots=True)
class RenderedExample:
    provider: str
    segments: tuple[tuple[str, str], ...]
    text: str

    def role_sequence(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.segments)


@dataclass(frozen=True, slots=True)
class DifferentialFinding:
    kind: DifferentialFindingKind
    detail: str
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "detail": self.detail,
            "providers": list(self.providers),
        }


@dataclass(frozen=True, slots=True)
class DifferentialResult:
    version: str
    example: str
    consistent: bool
    findings: tuple[DifferentialFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "example": self.example,
            "consistent": self.consistent,
            "findings": [f.to_dict() for f in self.findings],
        }


def run_differential(
    example: PackExample,
    fixtures: tuple[ProviderRenderFixture, ...],
) -> DifferentialResult:
    """Render ``example`` through every fixture and report divergences."""

    if not fixtures:
        raise ValueError("at least one provider fixture is required")

    rendered = {fx.provider: fx.render(example) for fx in fixtures}
    findings: list[DifferentialFinding] = []

    # Empty renders.
    for provider, r in rendered.items():
        if not r.text.strip():
            findings.append(
                DifferentialFinding(
                    DifferentialFindingKind.EMPTY_RENDER,
                    f"{provider} produced an empty render",
                    (provider,),
                )
            )

    # Role sequence divergence.
    role_seqs = {p: r.role_sequence() for p, r in rendered.items()}
    distinct_role_seqs = set(role_seqs.values())
    if len(distinct_role_seqs) > 1:
        findings.append(
            DifferentialFinding(
                DifferentialFindingKind.ROLE_SET_DIVERGENCE,
                "providers disagree on rendered role sequence: "
                + "; ".join(f"{p}={list(s)}" for p, s in sorted(role_seqs.items())),
                tuple(sorted(role_seqs)),
            )
        )

    # Turn count divergence (number of segments).
    turn_counts = {p: len(r.segments) for p, r in rendered.items()}
    if len(set(turn_counts.values())) > 1:
        findings.append(
            DifferentialFinding(
                DifferentialFindingKind.TURN_COUNT_DIVERGENCE,
                "providers disagree on turn count: "
                + ", ".join(f"{p}={c}" for p, c in sorted(turn_counts.items())),
                tuple(sorted(turn_counts)),
            )
        )

    # Control-token leakage: a provider's control token appears inside
    # user-authored content on another provider's render.
    user_text = "".join(
        m.content for m in example.messages if m.role in ("user", "tool")
    )
    for fx in fixtures:
        for token in fx.control_tokens:
            if token and token in user_text:
                findings.append(
                    DifferentialFinding(
                        DifferentialFindingKind.SPECIAL_TOKEN_LEAK,
                        f"control token {token!r} (declared by {fx.provider}) "
                        "appears verbatim in user-authored example content",
                        (fx.provider,),
                    )
                )

    return DifferentialResult(
        version=PROMPT_PACK_DIFFERENTIAL_VERSION,
        example=example.name,
        consistent=not findings,
        findings=tuple(findings),
    )


def render_differential_text(result: DifferentialResult) -> str:
    lines = [
        f"PromptABI prompt-pack differential ({result.version})",
        f"example: {result.example}",
        f"result: {'CONSISTENT' if result.consistent else 'DIVERGENT'}",
    ]
    for finding in result.findings:
        provs = f" [{', '.join(finding.providers)}]" if finding.providers else ""
        lines.append(f"  ! {finding.kind.value}{provs}: {finding.detail}")
    return "\n".join(lines) + "\n"
