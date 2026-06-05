"""Provider capability negotiation contracts (step 291).

Before sending a request, a client should *negotiate*: given the features a
request needs (e.g. JSON-schema structured output, parallel tool calls, streaming
+ tool calls together), find whether a provider supports them, and if not,
whether a declared fallback restores correctness.  This module performs that
negotiation and emits a contract: the resolved capability set, the fallbacks
applied, and any unmet requirement that must abort the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

CAPABILITY_NEGOTIATION_VERSION = "promptabi.capability-negotiation.v1"


class NegotiationOutcome(StrEnum):
    SATISFIED = "satisfied"
    SATISFIED_WITH_FALLBACK = "satisfied-with-fallback"
    UNMET = "unmet"


@dataclass(frozen=True, slots=True)
class Capability:
    name: str


@dataclass(frozen=True, slots=True)
class Fallback:
    """If ``capability`` is unsupported, ``provides`` can stand in for it."""

    capability: str
    provides_via: str


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    supported: frozenset[str]
    fallbacks: tuple[Fallback, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class NegotiatedCapability:
    name: str
    outcome: NegotiationOutcome
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class NegotiationContract:
    version: str
    provider: str
    items: tuple[NegotiatedCapability, ...]

    @property
    def satisfiable(self) -> bool:
        return all(i.outcome != NegotiationOutcome.UNMET for i in self.items)

    @property
    def used_fallback(self) -> bool:
        return any(
            i.outcome == NegotiationOutcome.SATISFIED_WITH_FALLBACK
            for i in self.items
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "provider": self.provider,
            "satisfiable": self.satisfiable,
            "used_fallback": self.used_fallback,
            "items": [i.to_dict() for i in self.items],
        }


def negotiate(
    required: tuple[Capability, ...], caps: ProviderCapabilities
) -> NegotiationContract:
    fallback_by_cap = {fb.capability: fb for fb in caps.fallbacks}
    items: list[NegotiatedCapability] = []

    for req in required:
        if req.name in caps.supported:
            items.append(
                NegotiatedCapability(
                    req.name,
                    NegotiationOutcome.SATISFIED,
                    "natively supported",
                )
            )
        elif req.name in fallback_by_cap:
            fb = fallback_by_cap[req.name]
            items.append(
                NegotiatedCapability(
                    req.name,
                    NegotiationOutcome.SATISFIED_WITH_FALLBACK,
                    f"via {fb.provides_via}",
                )
            )
        else:
            items.append(
                NegotiatedCapability(
                    req.name,
                    NegotiationOutcome.UNMET,
                    "no native support and no fallback declared",
                )
            )

    return NegotiationContract(
        version=CAPABILITY_NEGOTIATION_VERSION,
        provider=caps.provider,
        items=tuple(items),
    )


def render_negotiation_text(contract: NegotiationContract) -> str:
    lines = [
        f"PromptABI capability negotiation ({contract.version})",
        f"provider: {contract.provider}",
        f"satisfiable: {contract.satisfiable} "
        f"(fallback={contract.used_fallback})",
    ]
    for item in contract.items:
        lines.append(f"  {item.outcome.value}: {item.name} -- {item.detail}")
    return "\n".join(lines) + "\n"
