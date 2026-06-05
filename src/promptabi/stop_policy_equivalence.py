"""Stop-policy equivalence across providers (step 293).

The same logical stop condition is expressed differently per provider: a string
list, a single string, a regex, or a max-tokens cap.  Two providers are
*stop-equivalent* for a candidate output if they would truncate it at the same
position.  This module evaluates each provider's stop policy against a candidate
continuation, computes the cut index, and reports divergences so a prompt that
relies on stop behavior can be certified portable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

STOP_POLICY_EQUIVALENCE_VERSION = "promptabi.stop-policy-equivalence.v1"


class StopKind(StrEnum):
    STRING_LIST = "string-list"
    SINGLE_STRING = "single-string"
    MAX_TOKENS = "max-tokens"


@dataclass(frozen=True, slots=True)
class StopPolicy:
    provider: str
    kind: StopKind
    stops: tuple[str, ...] = field(default=())
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StopEvaluation:
    provider: str
    cut_index: int  # len of retained text; len(candidate) means "no stop hit"
    triggered_by: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "cut_index": self.cut_index,
            "triggered_by": self.triggered_by,
        }


def _first_stop_index(text: str, stops: tuple[str, ...]) -> tuple[int, str | None]:
    best = len(text)
    trigger: str | None = None
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < best:
            best = idx
            trigger = s
    return best, trigger


def evaluate_stop_policy(
    policy: StopPolicy, candidate: str, *, approx_chars_per_token: int = 4
) -> StopEvaluation:
    if policy.kind in (StopKind.STRING_LIST, StopKind.SINGLE_STRING):
        cut, trigger = _first_stop_index(candidate, policy.stops)
        return StopEvaluation(policy.provider, cut, trigger)
    if policy.kind == StopKind.MAX_TOKENS:
        if policy.max_tokens is None:
            return StopEvaluation(policy.provider, len(candidate), None)
        cap = policy.max_tokens * approx_chars_per_token
        if cap < len(candidate):
            return StopEvaluation(policy.provider, cap, f"max_tokens={policy.max_tokens}")
        return StopEvaluation(policy.provider, len(candidate), None)
    return StopEvaluation(policy.provider, len(candidate), None)


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    version: str
    equivalent: bool
    evaluations: tuple[StopEvaluation, ...]
    divergences: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "equivalent": self.equivalent,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "divergences": list(self.divergences),
        }


def certify_stop_equivalence(
    policies: tuple[StopPolicy, ...], candidate: str
) -> EquivalenceResult:
    evals = tuple(evaluate_stop_policy(p, candidate) for p in policies)
    cut_indices = {e.cut_index for e in evals}
    divergences: list[str] = []
    if len(cut_indices) > 1:
        base = evals[0]
        for e in evals[1:]:
            if e.cut_index != base.cut_index:
                divergences.append(
                    f"{base.provider} cuts at {base.cut_index} but "
                    f"{e.provider} cuts at {e.cut_index}"
                )
    return EquivalenceResult(
        version=STOP_POLICY_EQUIVALENCE_VERSION,
        equivalent=len(cut_indices) == 1,
        evaluations=evals,
        divergences=tuple(divergences),
    )


def render_equivalence_text(result: EquivalenceResult) -> str:
    lines = [
        f"PromptABI stop-policy equivalence ({result.version})",
        f"result: {'EQUIVALENT' if result.equivalent else 'DIVERGENT'}",
    ]
    for e in result.evaluations:
        lines.append(f"  {e.provider}: cut@{e.cut_index} by {e.triggered_by!r}")
    for d in result.divergences:
        lines.append(f"  ! {d}")
    return "\n".join(lines) + "\n"
