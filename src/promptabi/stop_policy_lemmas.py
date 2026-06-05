"""String-prefix and suffix lemmas for stop policies (step 221).

Stop sequences are the contractual boundary between a model's output and the
host application.  When one stop sequence is a *prefix* (or, for streaming
detectors, a *suffix* or *border*) of another, the longer sequence can become
unreachable or the streaming detector can mis-fire.  Those are real,
ship-blocking defects that are easy to introduce when prompt packs merge stop
policies from several providers.

This module proves a handful of decidable lemmas about a stop policy and backs
every verdict with the repository's own deterministic finite automata
(:class:`promptabi.formal.DeterministicFiniteAutomaton`) so the verdict is not
merely a Python ``str.startswith`` call but an automaton-certified fact:

* ``non-empty`` -- no stop sequence is empty (an empty stop fires immediately).
* ``prefix-free`` -- no stop is a proper prefix of another; if violated the
  longer sequence is *shadowed* (unreachable) because the prefix fires first.
* ``substring-free`` -- no stop occurs as an internal substring of another (the
  general shadowing condition that subsumes ``prefix-free``).
* ``suffix-free`` -- no stop is a proper suffix of another, which matters for
  right-anchored / streaming detectors and de-duplication.
* ``border-free`` -- no stop has a non-trivial border (a proper prefix that is
  also a proper suffix); borders force streaming detectors to buffer and can
  cause double-firing.

Each lemma yields a :class:`StopPolicyLemmaResult` with concrete witnesses so a
reviewer can see exactly which pair (and which shadowed output) violates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .formal import DeterministicFiniteAutomaton

STOP_POLICY_LEMMAS_VERSION = "promptabi.stop-policy-lemmas.v1"


class StopPolicyLemmaKind(StrEnum):
    """The decidable lemmas this module proves about a stop policy."""

    NON_EMPTY = "non-empty"
    PREFIX_FREE = "prefix-free"
    SUBSTRING_FREE = "substring-free"
    SUFFIX_FREE = "suffix-free"
    BORDER_FREE = "border-free"


class StopPolicyLemmaStatus(StrEnum):
    PROVEN = "proven"
    REFUTED = "refuted"


@dataclass(frozen=True, slots=True)
class StopPolicyLemmaWitness:
    """A concrete counterexample to a refuted lemma."""

    shadowed: str
    shadowing: str
    relation: str
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "shadowed": self.shadowed,
            "shadowing": self.shadowing,
            "relation": self.relation,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class StopPolicyLemmaResult:
    """The verdict for one lemma over a stop policy."""

    kind: StopPolicyLemmaKind
    status: StopPolicyLemmaStatus
    witnesses: tuple[StopPolicyLemmaWitness, ...] = ()
    certified_by_automaton: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "certified_by_automaton": self.certified_by_automaton,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
        }


@dataclass(frozen=True, slots=True)
class StopPolicyLemmaReport:
    """Verification report for every lemma over a stop policy."""

    version: str
    policy_name: str
    stop_sequences: tuple[str, ...]
    results: tuple[StopPolicyLemmaResult, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return all(result.status is StopPolicyLemmaStatus.PROVEN for result in self.results)

    @property
    def refuted(self) -> tuple[StopPolicyLemmaResult, ...]:
        return tuple(r for r in self.results if r.status is StopPolicyLemmaStatus.REFUTED)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policy_name": self.policy_name,
            "stop_sequences": list(self.stop_sequences),
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def _alphabet(stop_sequences: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({char for sequence in stop_sequences for char in sequence}))


def _automaton_is_prefix(shorter: str, longer: str, alphabet: tuple[str, ...]) -> bool:
    """Certify ``shorter`` is a prefix of ``longer`` with a prefix-closed DFA."""

    if not shorter:
        return True
    prefixes = DeterministicFiniteAutomaton.prefix_closed_literal(longer, alphabet=alphabet)
    # The prefix-closed automaton of ``longer`` accepts a string iff it is a
    # prefix of ``longer``; we additionally require the literal automaton of the
    # candidate to accept it so the witness is exact.
    literal = DeterministicFiniteAutomaton.literal(shorter, alphabet=alphabet)
    return prefixes.accepts_text(shorter) and literal.accepts_text(shorter)


def _automaton_substring_at(needle: str, haystack: str, start: int, alphabet: tuple[str, ...]) -> bool:
    """Certify ``haystack[start:start+len(needle)] == needle`` via a literal DFA."""

    window = haystack[start : start + len(needle)]
    if len(window) != len(needle):
        return False
    return DeterministicFiniteAutomaton.literal(needle, alphabet=alphabet).accepts_text(window)


def _borders(sequence: str) -> tuple[str, ...]:
    """Return the non-trivial borders (proper prefix == proper suffix)."""

    found: list[str] = []
    for length in range(1, len(sequence)):
        if sequence[:length] == sequence[-length:]:
            found.append(sequence[:length])
    return tuple(found)


def verify_stop_policy_lemmas(
    stop_sequences: Mapping[str, object] | tuple[str, ...] | list[str],
    *,
    policy_name: str = "stop-policy",
) -> StopPolicyLemmaReport:
    """Prove the stop-policy lemmas over ``stop_sequences``.

    ``stop_sequences`` may be a sequence of strings or a parsed stop-policy
    mapping with a ``stop_sequences`` key (and optional ``name``).
    """

    if isinstance(stop_sequences, Mapping):
        policy_name = str(stop_sequences.get("name", policy_name))
        raw = stop_sequences.get("stop_sequences", [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("stop policy 'stop_sequences' must be a list of strings")
        sequences = tuple(raw)
    else:
        sequences = tuple(stop_sequences)

    alphabet = _alphabet(sequences)
    results: list[StopPolicyLemmaResult] = []

    # non-empty
    empty_witnesses = tuple(
        StopPolicyLemmaWitness(
            shadowed=sequence,
            shadowing="",
            relation="empty",
            explanation="an empty stop sequence fires before any token is emitted",
        )
        for sequence in sequences
        if sequence == ""
    )
    results.append(
        StopPolicyLemmaResult(
            kind=StopPolicyLemmaKind.NON_EMPTY,
            status=StopPolicyLemmaStatus.REFUTED if empty_witnesses else StopPolicyLemmaStatus.PROVEN,
            witnesses=empty_witnesses,
            certified_by_automaton=False,
        )
    )

    nonempty = tuple(sequence for sequence in sequences if sequence)

    # prefix-free
    prefix_witnesses: list[StopPolicyLemmaWitness] = []
    for shorter in nonempty:
        for longer in nonempty:
            if shorter is longer or len(shorter) >= len(longer):
                continue
            if _automaton_is_prefix(shorter, longer, alphabet):
                prefix_witnesses.append(
                    StopPolicyLemmaWitness(
                        shadowed=longer,
                        shadowing=shorter,
                        relation="proper-prefix",
                        explanation=(
                            f"emitting toward {longer!r} fires the prefix {shorter!r} first, "
                            f"so {longer!r} is unreachable"
                        ),
                    )
                )
    results.append(
        StopPolicyLemmaResult(
            kind=StopPolicyLemmaKind.PREFIX_FREE,
            status=StopPolicyLemmaStatus.REFUTED if prefix_witnesses else StopPolicyLemmaStatus.PROVEN,
            witnesses=tuple(prefix_witnesses),
        )
    )

    # substring-free (general shadowing)
    substring_witnesses: list[StopPolicyLemmaWitness] = []
    for needle in nonempty:
        for haystack in nonempty:
            if needle is haystack or len(needle) >= len(haystack):
                continue
            positions = [
                start
                for start in range(0, len(haystack) - len(needle) + 1)
                if _automaton_substring_at(needle, haystack, start, alphabet)
            ]
            if positions:
                substring_witnesses.append(
                    StopPolicyLemmaWitness(
                        shadowed=haystack,
                        shadowing=needle,
                        relation=f"substring@{positions[0]}",
                        explanation=(
                            f"{needle!r} occurs inside {haystack!r}; the generator stops at "
                            f"{needle!r} before completing {haystack!r}"
                        ),
                    )
                )
    results.append(
        StopPolicyLemmaResult(
            kind=StopPolicyLemmaKind.SUBSTRING_FREE,
            status=StopPolicyLemmaStatus.REFUTED if substring_witnesses else StopPolicyLemmaStatus.PROVEN,
            witnesses=tuple(substring_witnesses),
        )
    )

    # suffix-free
    suffix_witnesses: list[StopPolicyLemmaWitness] = []
    for shorter in nonempty:
        for longer in nonempty:
            if shorter is longer or len(shorter) >= len(longer):
                continue
            # ``shorter`` is a suffix of ``longer`` iff it is a prefix of the
            # reversed literal; certify with the same automaton machinery.
            if _automaton_is_prefix(shorter[::-1], longer[::-1], _alphabet((shorter[::-1], longer[::-1]))):
                suffix_witnesses.append(
                    StopPolicyLemmaWitness(
                        shadowed=longer,
                        shadowing=shorter,
                        relation="proper-suffix",
                        explanation=(
                            f"{shorter!r} is a suffix of {longer!r}; right-anchored streaming "
                            f"detectors cannot distinguish the two boundaries"
                        ),
                    )
                )
    results.append(
        StopPolicyLemmaResult(
            kind=StopPolicyLemmaKind.SUFFIX_FREE,
            status=StopPolicyLemmaStatus.REFUTED if suffix_witnesses else StopPolicyLemmaStatus.PROVEN,
            witnesses=tuple(suffix_witnesses),
        )
    )

    # border-free (self-overlap)
    border_witnesses: list[StopPolicyLemmaWitness] = []
    for sequence in nonempty:
        for border in _borders(sequence):
            border_witnesses.append(
                StopPolicyLemmaWitness(
                    shadowed=sequence,
                    shadowing=border,
                    relation="border",
                    explanation=(
                        f"{sequence!r} has border {border!r} (a proper prefix that is also a "
                        f"proper suffix); streaming detectors must buffer to avoid double-firing"
                    ),
                )
            )
    results.append(
        StopPolicyLemmaResult(
            kind=StopPolicyLemmaKind.BORDER_FREE,
            status=StopPolicyLemmaStatus.REFUTED if border_witnesses else StopPolicyLemmaStatus.PROVEN,
            witnesses=tuple(border_witnesses),
            certified_by_automaton=False,
        )
    )

    return StopPolicyLemmaReport(
        version=STOP_POLICY_LEMMAS_VERSION,
        policy_name=policy_name,
        stop_sequences=sequences,
        results=tuple(results),
    )


def load_stop_policy(path: str) -> dict[str, object]:
    """Load a stop-policy JSON document (``{"name": ..., "stop_sequences": [...]}``)."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("stop policy document must be a JSON object")
    return dict(data)


def render_stop_policy_lemmas_json(report: StopPolicyLemmaReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_stop_policy_lemmas_text(report: StopPolicyLemmaReport) -> str:
    lines = [
        f"PromptABI stop-policy lemmas '{report.policy_name}' ({report.version})",
        f"status: {'PROVEN' if report.ok else 'VIOLATED'}",
        f"stop sequences: {len(report.stop_sequences)}",
    ]
    for result in report.results:
        lines.append("")
        suffix = " (automaton-certified)" if result.certified_by_automaton else ""
        lines.append(f"{result.kind.value}: {result.status.value}{suffix}")
        for witness in result.witnesses:
            lines.append(f"  - {witness.explanation}")
    return "\n".join(lines) + "\n"
