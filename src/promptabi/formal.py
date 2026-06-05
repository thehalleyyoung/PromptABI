"""Finite automata and finite-domain SMT primitives for PromptABI.

The classes in this module are intentionally small, deterministic, and CPU-only.
They are the executable core used by early checkers before richer template,
grammar, and tokenizer products exist.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any, Protocol


Symbol = str
State = str
Assignment = Mapping[str, object]


class AutomatonError(ValueError):
    """Raised when an automaton definition violates PromptABI invariants."""


@dataclass(frozen=True, slots=True)
class AutomatonWitness:
    """A shortest path through an automaton product."""

    symbols: tuple[Symbol, ...]
    states: tuple[State, ...]

    @property
    def text(self) -> str:
        return "".join(self.symbols)

    def to_dict(self) -> dict[str, object]:
        return {"symbols": list(self.symbols), "text": self.text, "states": list(self.states)}


@dataclass(frozen=True, slots=True)
class AutomatonSearchResult:
    """Result and cost counters for a lazy automata-product search."""

    witness: AutomatonWitness | None
    explored_states: int
    explored_transitions: int
    alphabet_symbols: int
    representative_symbols: int

    @property
    def found(self) -> bool:
        return self.witness is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "found": self.found,
            "witness": self.witness.to_dict() if self.witness is not None else None,
            "explored_states": self.explored_states,
            "explored_transitions": self.explored_transitions,
            "alphabet_symbols": self.alphabet_symbols,
            "representative_symbols": self.representative_symbols,
        }


@dataclass(frozen=True, slots=True)
class TransducerLabel:
    """One input/output edge label in a finite-state transducer path."""

    input_symbol: Symbol | None
    output_symbol: Symbol | None

    def __post_init__(self) -> None:
        if self.input_symbol == "":
            raise AutomatonError("transducer input labels must be non-empty or epsilon")
        if self.output_symbol == "":
            raise AutomatonError("transducer output labels must be non-empty or epsilon")

    def to_dict(self) -> dict[str, object]:
        return {"input": self.input_symbol, "output": self.output_symbol}


@dataclass(frozen=True, slots=True)
class TransducerTransition:
    """A directed finite-state transducer transition."""

    source: State
    label: TransducerLabel
    target: State

    def to_dict(self) -> dict[str, object]:
        return {"from": self.source, "label": self.label.to_dict(), "to": self.target}


@dataclass(frozen=True, slots=True)
class TransducerWitness:
    """A shortest accepted input/output pair with the path that produced it."""

    input_symbols: tuple[Symbol, ...]
    output_symbols: tuple[Symbol, ...]
    states: tuple[State, ...]
    labels: tuple[TransducerLabel, ...]

    @property
    def input_text(self) -> str:
        return "".join(self.input_symbols)

    @property
    def output_text(self) -> str:
        return "".join(self.output_symbols)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_symbols": list(self.input_symbols),
            "input_text": self.input_text,
            "output_symbols": list(self.output_symbols),
            "output_text": self.output_text,
            "states": list(self.states),
            "labels": [label.to_dict() for label in self.labels],
        }


@dataclass(frozen=True, slots=True)
class DeterministicFiniteAutomaton:
    """A partial deterministic finite automaton over a finite symbol alphabet."""

    states: frozenset[State]
    alphabet: tuple[Symbol, ...]
    start: State
    accepts: frozenset[State]
    transitions: Mapping[tuple[State, Symbol], State] = field(default_factory=dict)
    name: str = "dfa"

    def __post_init__(self) -> None:
        if not self.states:
            raise AutomatonError("automata must define at least one state")
        if self.start not in self.states:
            raise AutomatonError("automaton start state must be declared")
        if not self.accepts <= self.states:
            raise AutomatonError("accepting states must be declared states")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise AutomatonError("automaton alphabet must not contain duplicates")
        if any(symbol == "" for symbol in self.alphabet):
            raise AutomatonError("automaton symbols must be non-empty")
        alphabet = tuple(sorted(self.alphabet))
        normalized_transitions = dict(self.transitions)
        for (source, symbol), target in normalized_transitions.items():
            if source not in self.states:
                raise AutomatonError(f"transition source is not declared: {source}")
            if target not in self.states:
                raise AutomatonError(f"transition target is not declared: {target}")
            if symbol not in alphabet:
                raise AutomatonError(f"transition symbol is not in alphabet: {symbol}")
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "transitions", normalized_transitions)

    @classmethod
    def literal(cls, literal: str, *, alphabet: Iterable[Symbol] | None = None, name: str | None = None) -> "DeterministicFiniteAutomaton":
        """Accept exactly one literal string."""

        symbols = tuple(literal)
        effective_alphabet = tuple(sorted(set(symbols).union(alphabet or ())))
        states = frozenset(str(index) for index in range(len(symbols) + 1))
        transitions = {
            (str(index), symbol): str(index + 1)
            for index, symbol in enumerate(symbols)
        }
        return cls(
            states=states,
            alphabet=effective_alphabet,
            start="0",
            accepts=frozenset({str(len(symbols))}),
            transitions=transitions,
            name=name or f"literal({literal!r})",
        )

    @classmethod
    def prefix_closed_literal(cls, literal: str, *, alphabet: Iterable[Symbol] | None = None, name: str | None = None) -> "DeterministicFiniteAutomaton":
        """Accept every prefix of a literal, including the full literal."""

        automaton = cls.literal(literal, alphabet=alphabet, name=name or f"prefixes({literal!r})")
        return cls(
            states=automaton.states,
            alphabet=automaton.alphabet,
            start=automaton.start,
            accepts=frozenset(automaton.states),
            transitions=automaton.transitions,
            name=automaton.name,
        )

    @classmethod
    def finite_language(
        cls,
        words: Iterable[Sequence[Symbol] | str],
        *,
        alphabet: Iterable[Symbol] | None = None,
        name: str = "finite-language",
    ) -> "DeterministicFiniteAutomaton":
        """Build a trie automaton that accepts exactly the provided finite words."""

        root = "q0"
        states = {root}
        accepts: set[State] = set()
        transitions: dict[tuple[State, Symbol], State] = {}
        symbols = set(alphabet or ())
        next_id = 1
        for word in words:
            current = root
            for symbol in tuple(word):
                symbols.add(symbol)
                key = (current, symbol)
                if key not in transitions:
                    target = f"q{next_id}"
                    next_id += 1
                    transitions[key] = target
                    states.add(target)
                current = transitions[key]
            accepts.add(current)
        return cls(
            states=frozenset(states),
            alphabet=tuple(symbols),
            start=root,
            accepts=frozenset(accepts),
            transitions=transitions,
            name=name,
        )

    def step(self, state: State, symbol: Symbol) -> State | None:
        return self.transitions.get((state, symbol))

    def accepts_symbols(self, symbols: Iterable[Symbol]) -> bool:
        state: State | None = self.start
        for symbol in symbols:
            if state is None:
                return False
            state = self.step(state, symbol)
        return state in self.accepts if state is not None else False

    def accepts_text(self, text: str) -> bool:
        return self.accepts_symbols(tuple(text))

    def shortest_witness(self, *, max_depth: int | None = None) -> AutomatonWitness | None:
        """Return the shortest accepted word, if one is reachable."""

        return self._shortest_path(lambda state: state in self.accepts, max_depth=max_depth)

    def intersection_witness(
        self,
        other: "DeterministicFiniteAutomaton",
        *,
        max_depth: int | None = None,
        compress_alphabet: bool = True,
    ) -> AutomatonSearchResult:
        """Search the intersection lazily without materializing the product DFA.

        This method is intentionally scoped to intersection: a missing transition
        on either side rejects that symbol, so no implicit sink is needed. Union
        and difference still use the eager completed product to preserve sink
        semantics.
        """

        alphabet = tuple(sorted(set(self.alphabet).union(other.alphabet)))
        start = (self.start, other.start)
        queue = deque([(start, 0)])
        seen = {start}
        predecessors: dict[tuple[State, State], tuple[tuple[State, State], Symbol]] = {}
        explored_states = 0
        explored_transitions = 0
        representative_symbols = 0
        while queue:
            (left_state, right_state), depth = queue.popleft()
            explored_states += 1
            if left_state in self.accepts and right_state in other.accepts:
                return AutomatonSearchResult(
                    witness=_reconstruct_product_witness(start, (left_state, right_state), predecessors),
                    explored_states=explored_states,
                    explored_transitions=explored_transitions,
                    alphabet_symbols=len(alphabet),
                    representative_symbols=representative_symbols,
                )
            if max_depth is not None and depth >= max_depth:
                continue
            moves = _intersection_moves(self, other, left_state, right_state, alphabet, compress_alphabet=compress_alphabet)
            representative_symbols += len(moves)
            for symbol, next_left, next_right in moves:
                explored_transitions += 1
                target = (next_left, next_right)
                if target in seen:
                    continue
                seen.add(target)
                predecessors[target] = ((left_state, right_state), symbol)
                queue.append((target, depth + 1))
        return AutomatonSearchResult(
            witness=None,
            explored_states=explored_states,
            explored_transitions=explored_transitions,
            alphabet_symbols=len(alphabet),
            representative_symbols=representative_symbols,
        )

    def reachable_states(self) -> frozenset[State]:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            for symbol in self.alphabet:
                target = self.step(state, symbol)
                if target is not None and target not in seen:
                    seen.add(target)
                    queue.append(target)
        return frozenset(seen)

    def complete(
        self,
        *,
        alphabet: Iterable[Symbol] | None = None,
        sink: State = "__sink__",
    ) -> "DeterministicFiniteAutomaton":
        """Return an equivalent total DFA by routing missing transitions to a sink."""

        effective_alphabet = tuple(sorted(set(self.alphabet).union(alphabet or ())))
        states = set(self.states)
        transitions = dict(self.transitions)
        needs_sink = any((state, symbol) not in transitions for state in self.states for symbol in effective_alphabet)
        if needs_sink:
            while sink in states:
                sink = f"_{sink}"
            states.add(sink)
            for symbol in effective_alphabet:
                transitions[(sink, symbol)] = sink
        for state in tuple(states):
            for symbol in effective_alphabet:
                transitions.setdefault((state, symbol), sink if needs_sink else state)
        return DeterministicFiniteAutomaton(
            states=frozenset(states),
            alphabet=effective_alphabet,
            start=self.start,
            accepts=self.accepts,
            transitions=transitions,
            name=f"{self.name}-complete",
        )

    def minimize(self, *, complete: bool = True, name: str | None = None) -> "DeterministicFiniteAutomaton":
        """Return an equivalent DFA with reachable partition-equivalent states merged."""

        if complete:
            return self.complete().minimize(complete=False, name=name or f"{self.name}-min")

        reachable = self.reachable_states()
        accepting = frozenset(self.accepts.intersection(reachable))
        rejecting = frozenset(reachable - accepting)
        partitions = tuple(part for part in (accepting, rejecting) if part)
        if not partitions:
            partitions = (frozenset({self.start}),)

        changed = True
        while changed:
            changed = False
            state_to_part = {
                state: index
                for index, partition in enumerate(partitions)
                for state in partition
            }
            refined: list[frozenset[State]] = []
            for partition in partitions:
                buckets: dict[tuple[int | None, ...], set[State]] = {}
                for state in partition:
                    signature = tuple(
                        state_to_part.get(self.step(state, symbol))
                        for symbol in self.alphabet
                    )
                    buckets.setdefault(signature, set()).add(state)
                refined.extend(frozenset(bucket) for _, bucket in sorted(buckets.items(), key=lambda item: (item[0], sorted(item[1]))))
                changed = changed or len(buckets) > 1
            partitions = tuple(refined)

        state_names = {
            partition: f"m{index}"
            for index, partition in enumerate(sorted(partitions, key=lambda part: sorted(part)[0]))
        }
        representative_to_partition = {
            state: partition
            for partition in partitions
            for state in partition
        }
        transitions: dict[tuple[State, Symbol], State] = {}
        for partition, source_name in state_names.items():
            representative = sorted(partition)[0]
            for symbol in self.alphabet:
                target = self.step(representative, symbol)
                if target is None or target not in representative_to_partition:
                    continue
                transitions[(source_name, symbol)] = state_names[representative_to_partition[target]]
        return DeterministicFiniteAutomaton(
            states=frozenset(state_names.values()),
            alphabet=self.alphabet,
            start=state_names[representative_to_partition[self.start]],
            accepts=frozenset(
                state_names[partition]
                for partition in partitions
                if partition.intersection(self.accepts)
            ),
            transitions=transitions,
            name=name or f"{self.name}-min",
        )

    def complement(self) -> "DeterministicFiniteAutomaton":
        complete = self.complete()
        return DeterministicFiniteAutomaton(
            states=complete.states,
            alphabet=complete.alphabet,
            start=complete.start,
            accepts=complete.states - complete.accepts,
            transitions=complete.transitions,
            name=f"not({self.name})",
        )

    def intersect(self, other: "DeterministicFiniteAutomaton", *, name: str | None = None) -> "DeterministicFiniteAutomaton":
        return _product(self, other, lambda left, right: left and right, name=name or f"({self.name}&{other.name})")

    def union(self, other: "DeterministicFiniteAutomaton", *, name: str | None = None) -> "DeterministicFiniteAutomaton":
        return _product(self, other, lambda left, right: left or right, name=name or f"({self.name}|{other.name})")

    def difference(self, other: "DeterministicFiniteAutomaton", *, name: str | None = None) -> "DeterministicFiniteAutomaton":
        return _product(self, other, lambda left, right: left and not right, name=name or f"({self.name}-{other.name})")

    def is_empty(self) -> bool:
        return self.shortest_witness() is None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "states": sorted(self.states),
            "alphabet": list(self.alphabet),
            "start": self.start,
            "accepts": sorted(self.accepts),
            "transitions": [
                {"from": source, "symbol": symbol, "to": target}
                for (source, symbol), target in sorted(self.transitions.items())
            ],
        }

    def _shortest_path(
        self,
        predicate: Callable[[State], bool],
        *,
        max_depth: int | None,
    ) -> AutomatonWitness | None:
        queue = deque([(self.start, (), (self.start,))])
        seen = {self.start}
        while queue:
            state, symbols, states = queue.popleft()
            if predicate(state):
                return AutomatonWitness(symbols=symbols, states=states)
            if max_depth is not None and len(symbols) >= max_depth:
                continue
            for symbol in self.alphabet:
                target = self.step(state, symbol)
                if target is None:
                    continue
                if target in seen:
                    continue
                seen.add(target)
                queue.append((target, symbols + (symbol,), states + (target,)))
        return None


@dataclass(frozen=True, slots=True)
class FiniteStateTransducer:
    """A finite relation between input and output symbol streams.

    Epsilon is represented as ``None`` on either side of a transition. That is
    enough for render-like insertion/deletion, tokenize-like segmentation, and
    decode-like normalization products while keeping witnesses finite and stable.
    """

    states: frozenset[State]
    input_alphabet: tuple[Symbol, ...]
    output_alphabet: tuple[Symbol, ...]
    start: State
    accepts: frozenset[State]
    transitions: tuple[TransducerTransition, ...] = ()
    name: str = "fst"
    approximation: str = "exact"

    def __post_init__(self) -> None:
        if not self.states:
            raise AutomatonError("transducers must define at least one state")
        if self.start not in self.states:
            raise AutomatonError("transducer start state must be declared")
        if not self.accepts <= self.states:
            raise AutomatonError("transducer accepting states must be declared states")
        if len(set(self.input_alphabet)) != len(self.input_alphabet):
            raise AutomatonError("transducer input alphabet must not contain duplicates")
        if len(set(self.output_alphabet)) != len(self.output_alphabet):
            raise AutomatonError("transducer output alphabet must not contain duplicates")
        if any(symbol == "" for symbol in self.input_alphabet + self.output_alphabet):
            raise AutomatonError("transducer symbols must be non-empty")
        input_alphabet = tuple(sorted(self.input_alphabet))
        output_alphabet = tuple(sorted(self.output_alphabet))
        normalized: list[TransducerTransition] = []
        seen: set[tuple[State, Symbol | None, Symbol | None, State]] = set()
        for transition in self.transitions:
            if transition.source not in self.states:
                raise AutomatonError(f"transducer transition source is not declared: {transition.source}")
            if transition.target not in self.states:
                raise AutomatonError(f"transducer transition target is not declared: {transition.target}")
            if transition.label.input_symbol is not None and transition.label.input_symbol not in input_alphabet:
                raise AutomatonError(f"transducer input symbol is not in alphabet: {transition.label.input_symbol}")
            if transition.label.output_symbol is not None and transition.label.output_symbol not in output_alphabet:
                raise AutomatonError(f"transducer output symbol is not in alphabet: {transition.label.output_symbol}")
            key = (transition.source, transition.label.input_symbol, transition.label.output_symbol, transition.target)
            if key not in seen:
                normalized.append(transition)
                seen.add(key)
        normalized.sort(key=lambda item: (item.source, item.label.input_symbol or "", item.label.output_symbol or "", item.target))
        object.__setattr__(self, "input_alphabet", input_alphabet)
        object.__setattr__(self, "output_alphabet", output_alphabet)
        object.__setattr__(self, "transitions", tuple(normalized))

    @classmethod
    def literal_mapping(cls, input_text: str, output_text: str, *, name: str | None = None) -> "FiniteStateTransducer":
        """Accept exactly one string-to-string mapping."""

        return cls.finite_relation(((input_text, output_text),), name=name or f"mapping({input_text!r}->{output_text!r})")

    @classmethod
    def identity(cls, alphabet: Iterable[Symbol], *, name: str = "identity") -> "FiniteStateTransducer":
        symbols = tuple(sorted(set(alphabet)))
        transitions = tuple(
            TransducerTransition("q0", TransducerLabel(symbol, symbol), "q0")
            for symbol in symbols
        )
        return cls(
            states=frozenset({"q0"}),
            input_alphabet=symbols,
            output_alphabet=symbols,
            start="q0",
            accepts=frozenset({"q0"}),
            transitions=transitions,
            name=name,
        )

    @classmethod
    def finite_relation(
        cls,
        pairs: Iterable[tuple[Sequence[Symbol] | str, Sequence[Symbol] | str]],
        *,
        name: str = "finite-relation",
    ) -> "FiniteStateTransducer":
        """Build a transducer that accepts exactly the provided finite pairs."""

        root = "q0"
        states = {root}
        accepts: set[State] = set()
        transitions: list[TransducerTransition] = []
        input_alphabet: set[Symbol] = set()
        output_alphabet: set[Symbol] = set()
        next_id = 1
        for input_word, output_word in pairs:
            current = root
            input_symbols = tuple(input_word)
            output_symbols = tuple(output_word)
            length = max(len(input_symbols), len(output_symbols))
            for index in range(length):
                input_symbol = input_symbols[index] if index < len(input_symbols) else None
                output_symbol = output_symbols[index] if index < len(output_symbols) else None
                if input_symbol is not None:
                    input_alphabet.add(input_symbol)
                if output_symbol is not None:
                    output_alphabet.add(output_symbol)
                target = f"q{next_id}"
                next_id += 1
                transitions.append(TransducerTransition(current, TransducerLabel(input_symbol, output_symbol), target))
                states.add(target)
                current = target
            accepts.add(current)
        return cls(
            states=frozenset(states),
            input_alphabet=tuple(input_alphabet),
            output_alphabet=tuple(output_alphabet),
            start=root,
            accepts=frozenset(accepts),
            transitions=tuple(transitions),
            name=name,
        )

    @classmethod
    def from_independent_languages(
        cls,
        input_language: DeterministicFiniteAutomaton,
        output_language: DeterministicFiniteAutomaton,
        *,
        name: str = "projection-overapproximation",
    ) -> "FiniteStateTransducer":
        """Over-approximate a relation as every accepted input paired with every accepted output."""

        states = {_pair(input_state, output_state) for input_state in input_language.states for output_state in output_language.states}
        transitions: list[TransducerTransition] = []
        for input_state in input_language.states:
            for output_state in output_language.states:
                source = _pair(input_state, output_state)
                for symbol in input_language.alphabet:
                    target_input = input_language.step(input_state, symbol)
                    if target_input is not None:
                        transitions.append(TransducerTransition(source, TransducerLabel(symbol, None), _pair(target_input, output_state)))
                for symbol in output_language.alphabet:
                    target_output = output_language.step(output_state, symbol)
                    if target_output is not None:
                        transitions.append(TransducerTransition(source, TransducerLabel(None, symbol), _pair(input_state, target_output)))
        accepts = {
            _pair(input_state, output_state)
            for input_state in input_language.accepts
            for output_state in output_language.accepts
        }
        return cls(
            states=frozenset(states),
            input_alphabet=input_language.alphabet,
            output_alphabet=output_language.alphabet,
            start=_pair(input_language.start, output_language.start),
            accepts=frozenset(accepts),
            transitions=tuple(transitions),
            name=name,
            approximation="overapproximation",
        )

    def accepts_pair(self, input_symbols: Iterable[Symbol] | str, output_symbols: Iterable[Symbol] | str) -> bool:
        input_tuple = tuple(input_symbols)
        output_tuple = tuple(output_symbols)
        queue = deque([(self.start, 0, 0)])
        seen = {(self.start, 0, 0)}
        by_source = self._transitions_by_source()
        while queue:
            state, input_index, output_index = queue.popleft()
            if state in self.accepts and input_index == len(input_tuple) and output_index == len(output_tuple):
                return True
            for transition in by_source.get(state, ()):
                next_input = input_index
                next_output = output_index
                if transition.label.input_symbol is not None:
                    if input_index >= len(input_tuple) or input_tuple[input_index] != transition.label.input_symbol:
                        continue
                    next_input += 1
                if transition.label.output_symbol is not None:
                    if output_index >= len(output_tuple) or output_tuple[output_index] != transition.label.output_symbol:
                        continue
                    next_output += 1
                item = (transition.target, next_input, next_output)
                if item not in seen:
                    seen.add(item)
                    queue.append(item)
        return False

    def shortest_witness(self, *, max_depth: int | None = None) -> TransducerWitness | None:
        queue = deque([(self.start, (), (), (self.start,), ())])
        seen = {self.start}
        by_source = self._transitions_by_source()
        while queue:
            state, input_symbols, output_symbols, states, labels = queue.popleft()
            if state in self.accepts:
                return TransducerWitness(
                    input_symbols=input_symbols,
                    output_symbols=output_symbols,
                    states=states,
                    labels=labels,
                )
            if max_depth is not None and len(labels) >= max_depth:
                continue
            for transition in by_source.get(state, ()):
                if transition.target in seen:
                    continue
                seen.add(transition.target)
                queue.append(
                    (
                        transition.target,
                        input_symbols + (() if transition.label.input_symbol is None else (transition.label.input_symbol,)),
                        output_symbols + (() if transition.label.output_symbol is None else (transition.label.output_symbol,)),
                        states + (transition.target,),
                        labels + (transition.label,),
                    )
                )
        return None

    def project_input(self, *, name: str | None = None) -> DeterministicFiniteAutomaton:
        return self._project(side="input", name=name or f"input({self.name})")

    def project_output(self, *, name: str | None = None) -> DeterministicFiniteAutomaton:
        return self._project(side="output", name=name or f"output({self.name})")

    def overapproximate_by_projections(self, *, name: str | None = None) -> "FiniteStateTransducer":
        return self.from_independent_languages(
            self.project_input(),
            self.project_output(),
            name=name or f"overapprox({self.name})",
        )

    def compose(self, other: "FiniteStateTransducer", *, name: str | None = None) -> "FiniteStateTransducer":
        """Compose this relation with another relation over the shared middle alphabet."""

        start = _pair(self.start, other.start)
        states = {start}
        accepts: set[State] = set()
        transitions: list[TransducerTransition] = []
        queue = deque([(self.start, other.start)])
        left_by_source = self._transitions_by_source()
        right_by_source = other._transitions_by_source()
        right_by_state_and_input = {
            state: _transducer_transitions_by_input(items)
            for state, items in right_by_source.items()
        }
        right_epsilons = {
            state: tuple(transition for transition in items if transition.label.input_symbol is None)
            for state, items in right_by_source.items()
        }
        while queue:
            left_state, right_state = queue.popleft()
            product_state = _pair(left_state, right_state)
            if left_state in self.accepts and right_state in other.accepts:
                accepts.add(product_state)

            candidates: list[tuple[TransducerLabel, State, State]] = []
            right_index = right_by_state_and_input.get(right_state, {})
            for left_transition in left_by_source.get(left_state, ()):
                if left_transition.label.output_symbol is None:
                    candidates.append((TransducerLabel(left_transition.label.input_symbol, None), left_transition.target, right_state))
                if left_transition.label.output_symbol is not None:
                    for right_transition in right_index.get(left_transition.label.output_symbol, ()):
                        candidates.append(
                            (
                                TransducerLabel(left_transition.label.input_symbol, right_transition.label.output_symbol),
                                left_transition.target,
                                right_transition.target,
                            )
                        )
            for right_transition in right_epsilons.get(right_state, ()):
                candidates.append((TransducerLabel(None, right_transition.label.output_symbol), left_state, right_transition.target))

            for label, next_left, next_right in candidates:
                target = _pair(next_left, next_right)
                transitions.append(TransducerTransition(product_state, label, target))
                if target not in states:
                    states.add(target)
                    queue.append((next_left, next_right))

        approximation = "exact" if self.approximation == other.approximation == "exact" else "overapproximation"
        return FiniteStateTransducer(
            states=frozenset(states),
            input_alphabet=self.input_alphabet,
            output_alphabet=other.output_alphabet,
            start=start,
            accepts=frozenset(accepts),
            transitions=tuple(transitions),
            name=name or f"({self.name};{other.name})",
            approximation=approximation,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "approximation": self.approximation,
            "states": sorted(self.states),
            "input_alphabet": list(self.input_alphabet),
            "output_alphabet": list(self.output_alphabet),
            "start": self.start,
            "accepts": sorted(self.accepts),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def _project(self, *, side: str, name: str) -> DeterministicFiniteAutomaton:
        alphabet = self.input_alphabet if side == "input" else self.output_alphabet
        by_source = self._transitions_by_source()
        start_set = frozenset(self._epsilon_closure({self.start}, side=side, by_source=by_source))
        state_names = {start_set: _set_state_name(start_set)}
        queue = deque([start_set])
        transitions: dict[tuple[State, Symbol], State] = {}
        accepts: set[State] = set()
        while queue:
            subset = queue.popleft()
            subset_name = state_names[subset]
            if self.accepts.intersection(subset):
                accepts.add(subset_name)
            for symbol in alphabet:
                targets: set[State] = set()
                for state in subset:
                    for transition in by_source.get(state, ()):
                        label_symbol = transition.label.input_symbol if side == "input" else transition.label.output_symbol
                        if label_symbol == symbol:
                            targets.add(transition.target)
                if not targets:
                    continue
                closed = frozenset(self._epsilon_closure(targets, side=side, by_source=by_source))
                if closed not in state_names:
                    state_names[closed] = _set_state_name(closed)
                    queue.append(closed)
                transitions[(subset_name, symbol)] = state_names[closed]
        return DeterministicFiniteAutomaton(
            states=frozenset(state_names.values()),
            alphabet=alphabet,
            start=state_names[start_set],
            accepts=frozenset(accepts),
            transitions=transitions,
            name=name,
        )

    def _epsilon_closure(
        self,
        states: Iterable[State],
        *,
        side: str,
        by_source: Mapping[State, tuple[TransducerTransition, ...]],
    ) -> set[State]:
        seen = set(states)
        queue = deque(seen)
        while queue:
            state = queue.popleft()
            for transition in by_source.get(state, ()):
                label_symbol = transition.label.input_symbol if side == "input" else transition.label.output_symbol
                if label_symbol is None and transition.target not in seen:
                    seen.add(transition.target)
                    queue.append(transition.target)
        return seen

    def _transitions_by_source(self) -> dict[State, tuple[TransducerTransition, ...]]:
        grouped: dict[State, list[TransducerTransition]] = {}
        for transition in self.transitions:
            grouped.setdefault(transition.source, []).append(transition)
        return {state: tuple(items) for state, items in grouped.items()}


def _product(
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    accept: Callable[[bool, bool], bool],
    *,
    name: str,
) -> DeterministicFiniteAutomaton:
    alphabet = tuple(sorted(set(left.alphabet).union(right.alphabet)))
    left_complete = left.complete(alphabet=alphabet).minimize(complete=False)
    right_complete = right.complete(alphabet=alphabet).minimize(complete=False)
    start = _pair(left_complete.start, right_complete.start)
    states = {start}
    accepts: set[State] = set()
    transitions: dict[tuple[State, Symbol], State] = {}
    queue = deque([(left_complete.start, right_complete.start)])
    while queue:
        left_state, right_state = queue.popleft()
        product_state = _pair(left_state, right_state)
        if accept(left_state in left_complete.accepts, right_state in right_complete.accepts):
            accepts.add(product_state)
        for symbol in alphabet:
            next_left = left_complete.step(left_state, symbol)
            next_right = right_complete.step(right_state, symbol)
            if next_left is None or next_right is None:
                raise AutomatonError("completed product automata must be total")
            target = _pair(next_left, next_right)
            transitions[(product_state, symbol)] = target
            if target not in states:
                states.add(target)
                queue.append((next_left, next_right))
    return DeterministicFiniteAutomaton(
        states=frozenset(states),
        alphabet=alphabet,
        start=start,
        accepts=frozenset(accepts),
        transitions=transitions,
        name=name,
    )


def _pair(left: State, right: State) -> State:
    return f"{left}\u241f{right}"


def _intersection_moves(
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    left_state: State,
    right_state: State,
    alphabet: Sequence[Symbol],
    *,
    compress_alphabet: bool,
) -> tuple[tuple[Symbol, State, State], ...]:
    moves: dict[tuple[State, State], Symbol] = {}
    uncompressed: list[tuple[Symbol, State, State]] = []
    for symbol in alphabet:
        next_left = left.step(left_state, symbol)
        next_right = right.step(right_state, symbol)
        if next_left is None or next_right is None:
            continue
        if not compress_alphabet:
            uncompressed.append((symbol, next_left, next_right))
            continue
        moves.setdefault((next_left, next_right), symbol)
    if not compress_alphabet:
        return tuple(uncompressed)
    return tuple((symbol, next_left, next_right) for (next_left, next_right), symbol in sorted(moves.items(), key=lambda item: item[1]))


def _reconstruct_product_witness(
    start: tuple[State, State],
    accept: tuple[State, State],
    predecessors: Mapping[tuple[State, State], tuple[tuple[State, State], Symbol]],
) -> AutomatonWitness:
    symbols: list[Symbol] = []
    state_pairs: list[tuple[State, State]] = [accept]
    current = accept
    while current != start:
        previous, symbol = predecessors[current]
        symbols.append(symbol)
        current = previous
        state_pairs.append(current)
    symbols.reverse()
    state_pairs.reverse()
    return AutomatonWitness(
        symbols=tuple(symbols),
        states=tuple(_pair(left, right) for left, right in state_pairs),
    )


def _set_state_name(states: Iterable[State]) -> State:
    return "{" + ",".join(sorted(states)) + "}"


def _transducer_transitions_by_input(
    transitions: Sequence[TransducerTransition],
) -> dict[Symbol, tuple[TransducerTransition, ...]]:
    grouped: dict[Symbol, list[TransducerTransition]] = {}
    for transition in transitions:
        if transition.label.input_symbol is not None:
            grouped.setdefault(transition.label.input_symbol, []).append(transition)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


class SolverStatus(StrEnum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class SolverBackend(StrEnum):
    Z3 = "z3"
    FINITE_ENUMERATION = "finite-enumeration"


class SolverConclusion(StrEnum):
    COUNTEREXAMPLE = "concrete-counterexample"
    UNSAT_CORE_PROOF = "unsat-core-proof"
    ABSTENTION = "abstention"


class SolverBudgetOutcome(StrEnum):
    """How completely the solver explored its declared finite/SMT budget."""

    PROVED = "proved"
    BOUNDED = "bounded"
    TIMED_OUT = "timed-out"
    APPROXIMATED = "approximated"
    ABSTAINED = "abstained"


class Expression(Protocol):
    def evaluate(self, assignment: Assignment) -> bool | int | str:
        ...

    def to_z3(self, context: "_Z3Context") -> Any:
        ...

    def to_dict(self) -> dict[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class Var:
    name: str

    def evaluate(self, assignment: Assignment) -> object:
        return assignment[self.name]

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.variables[self.name]

    def to_dict(self) -> dict[str, object]:
        return {"var": self.name}


@dataclass(frozen=True, slots=True)
class Value:
    value: bool | int | str

    def evaluate(self, assignment: Assignment) -> bool | int | str:
        return self.value

    def to_z3(self, context: "_Z3Context") -> Any:
        z3 = context.z3
        if isinstance(self.value, bool):
            return z3.BoolVal(self.value)
        if isinstance(self.value, int):
            return z3.IntVal(self.value)
        return z3.StringVal(self.value)

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value}


@dataclass(frozen=True, slots=True)
class Eq:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        return self.left.evaluate(assignment) == self.right.evaluate(assignment)

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) == self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"eq": [self.left.to_dict(), self.right.to_dict()]}


@dataclass(frozen=True, slots=True)
class Ne:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        return self.left.evaluate(assignment) != self.right.evaluate(assignment)

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) != self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"ne": [self.left.to_dict(), self.right.to_dict()]}


def _finite_int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class Le:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        left = _finite_int_value(self.left.evaluate(assignment))
        right = _finite_int_value(self.right.evaluate(assignment))
        if left is None or right is None:
            raise TypeError("numeric comparison received a non-integer finite value")
        return left <= right

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) <= self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"le": [self.left.to_dict(), self.right.to_dict()]}


@dataclass(frozen=True, slots=True)
class Lt:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        left = _finite_int_value(self.left.evaluate(assignment))
        right = _finite_int_value(self.right.evaluate(assignment))
        if left is None or right is None:
            raise TypeError("numeric comparison received a non-integer finite value")
        return left < right

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) < self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"lt": [self.left.to_dict(), self.right.to_dict()]}


@dataclass(frozen=True, slots=True)
class Ge:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        left = _finite_int_value(self.left.evaluate(assignment))
        right = _finite_int_value(self.right.evaluate(assignment))
        if left is None or right is None:
            raise TypeError("numeric comparison received a non-integer finite value")
        return left >= right

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) >= self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"ge": [self.left.to_dict(), self.right.to_dict()]}


@dataclass(frozen=True, slots=True)
class Gt:
    left: Expression
    right: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        left = _finite_int_value(self.left.evaluate(assignment))
        right = _finite_int_value(self.right.evaluate(assignment))
        if left is None or right is None:
            raise TypeError("numeric comparison received a non-integer finite value")
        return left > right

    def to_z3(self, context: "_Z3Context") -> Any:
        return self.left.to_z3(context) > self.right.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"gt": [self.left.to_dict(), self.right.to_dict()]}


@dataclass(frozen=True, slots=True)
class Sum:
    """Integer sum of finite expressions (linear arithmetic over the solver)."""

    terms: tuple[Expression, ...]

    def __init__(self, *terms: Expression) -> None:
        object.__setattr__(self, "terms", tuple(terms))

    def evaluate(self, assignment: Assignment) -> int:
        total = 0
        for term in self.terms:
            value = _finite_int_value(term.evaluate(assignment))
            if value is None:
                raise TypeError("sum received a non-integer finite value")
            total += value
        return total

    def to_z3(self, context: "_Z3Context") -> Any:
        operands = [term.to_z3(context) for term in self.terms]
        if not operands:
            return context.z3.IntVal(0)
        accumulator = operands[0]
        for operand in operands[1:]:
            accumulator = accumulator + operand
        return accumulator

    def to_dict(self) -> dict[str, object]:
        return {"sum": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True, slots=True)
class Mul:
    """Multiply a finite expression by an integer coefficient."""

    coefficient: int
    term: Expression

    def evaluate(self, assignment: Assignment) -> int:
        value = _finite_int_value(self.term.evaluate(assignment))
        if value is None:
            raise TypeError("scaled term received a non-integer finite value")
        return self.coefficient * value

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.IntVal(self.coefficient) * self.term.to_z3(context)

    def to_dict(self) -> dict[str, object]:
        return {"mul": [self.coefficient, self.term.to_dict()]}


@dataclass(frozen=True, slots=True)
class And:
    terms: tuple[Expression, ...]

    def __init__(self, *terms: Expression) -> None:
        object.__setattr__(self, "terms", tuple(terms))

    def evaluate(self, assignment: Assignment) -> bool:
        return all(bool(term.evaluate(assignment)) for term in self.terms)

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.And(*(term.to_z3(context) for term in self.terms))

    def to_dict(self) -> dict[str, object]:
        return {"and": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True, slots=True)
class Or:
    terms: tuple[Expression, ...]

    def __init__(self, *terms: Expression) -> None:
        object.__setattr__(self, "terms", tuple(terms))

    def evaluate(self, assignment: Assignment) -> bool:
        return any(bool(term.evaluate(assignment)) for term in self.terms)

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.Or(*(term.to_z3(context) for term in self.terms))

    def to_dict(self) -> dict[str, object]:
        return {"or": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True, slots=True)
class Not:
    term: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        return not bool(self.term.evaluate(assignment))

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.Not(self.term.to_z3(context))

    def to_dict(self) -> dict[str, object]:
        return {"not": self.term.to_dict()}


@dataclass(frozen=True, slots=True)
class Implies:
    condition: Expression
    consequence: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        return (not bool(self.condition.evaluate(assignment))) or bool(self.consequence.evaluate(assignment))

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.Implies(self.condition.to_z3(context), self.consequence.to_z3(context))

    def to_dict(self) -> dict[str, object]:
        return {"implies": [self.condition.to_dict(), self.consequence.to_dict()]}


@dataclass(frozen=True, slots=True)
class InSet:
    term: Expression
    values: frozenset[bool | int | str]

    def __init__(self, term: Expression, values: Iterable[bool | int | str]) -> None:
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "values", frozenset(values))

    def evaluate(self, assignment: Assignment) -> bool:
        return self.term.evaluate(assignment) in self.values

    def to_z3(self, context: "_Z3Context") -> Any:
        z3 = context.z3
        term = self.term.to_z3(context)
        return z3.Or(*(term == Value(value).to_z3(context) for value in sorted(self.values, key=repr)))

    def to_dict(self) -> dict[str, object]:
        return {"in": [self.term.to_dict(), sorted(self.values, key=repr)]}


@dataclass(frozen=True, slots=True)
class Length:
    term: Expression

    def evaluate(self, assignment: Assignment) -> int:
        return len(str(self.term.evaluate(assignment)))

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.Length(self.term.to_z3(context))

    def to_dict(self) -> dict[str, object]:
        return {"length": self.term.to_dict()}


@dataclass(frozen=True, slots=True)
class Contains:
    haystack: Expression
    needle: Expression

    def evaluate(self, assignment: Assignment) -> bool:
        return str(self.needle.evaluate(assignment)) in str(self.haystack.evaluate(assignment))

    def to_z3(self, context: "_Z3Context") -> Any:
        return context.z3.Contains(self.haystack.to_z3(context), self.needle.to_z3(context))

    def to_dict(self) -> dict[str, object]:
        return {"contains": [self.haystack.to_dict(), self.needle.to_dict()]}


@dataclass(frozen=True, slots=True)
class VariableDomain:
    name: str

    def values(self) -> tuple[object, ...]:
        raise NotImplementedError

    def z3_variable(self, context: "_Z3Context") -> Any:
        raise NotImplementedError

    def z3_domain_constraint(self, variable: Any, context: "_Z3Context") -> Any:
        raise NotImplementedError

    def z3_value_to_python(self, value: Any) -> object:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BoolDomain(VariableDomain):
    def values(self) -> tuple[bool, ...]:
        return (False, True)

    def z3_variable(self, context: "_Z3Context") -> Any:
        return context.z3.Bool(self.name)

    def z3_domain_constraint(self, variable: Any, context: "_Z3Context") -> Any:
        return context.z3.BoolVal(True)

    def z3_value_to_python(self, value: Any) -> bool:
        return str(value) == "True"

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": "bool"}


@dataclass(frozen=True, slots=True)
class EnumDomain(VariableDomain):
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("enum domains must contain at least one member")
        if any(not member for member in self.members):
            raise ValueError("enum members must be non-empty")
        object.__setattr__(self, "members", tuple(sorted(dict.fromkeys(self.members))))

    def values(self) -> tuple[str, ...]:
        return self.members

    def z3_variable(self, context: "_Z3Context") -> Any:
        return context.z3.String(self.name)

    def z3_domain_constraint(self, variable: Any, context: "_Z3Context") -> Any:
        return context.z3.Or(*(variable == context.z3.StringVal(member) for member in self.members))

    def z3_value_to_python(self, value: Any) -> str:
        return value.as_string()

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": "enum", "members": list(self.members)}


@dataclass(frozen=True, slots=True)
class IntRangeDomain(VariableDomain):
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("integer domain minimum must be <= maximum")

    def values(self) -> tuple[int, ...]:
        return tuple(range(self.minimum, self.maximum + 1))

    def z3_variable(self, context: "_Z3Context") -> Any:
        return context.z3.Int(self.name)

    def z3_domain_constraint(self, variable: Any, context: "_Z3Context") -> Any:
        return context.z3.And(variable >= self.minimum, variable <= self.maximum)

    def z3_value_to_python(self, value: Any) -> int:
        return value.as_long()

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": "int-range", "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class BoundedStringDomain(VariableDomain):
    alphabet: tuple[str, ...]
    min_length: int = 0
    max_length: int = 8

    def __post_init__(self) -> None:
        if self.min_length < 0 or self.max_length < self.min_length:
            raise ValueError("bounded string lengths must satisfy 0 <= min_length <= max_length")
        if not self.alphabet:
            raise ValueError("bounded string domains must contain at least one symbol")
        if any(len(symbol) != 1 for symbol in self.alphabet):
            raise ValueError("bounded string symbols must be single-character strings")
        object.__setattr__(self, "alphabet", tuple(sorted(dict.fromkeys(self.alphabet))))

    def values(self) -> tuple[str, ...]:
        strings: list[str] = []
        for length in range(self.min_length, self.max_length + 1):
            strings.extend("".join(chars) for chars in product(self.alphabet, repeat=length))
        return tuple(strings)

    def z3_variable(self, context: "_Z3Context") -> Any:
        return context.z3.String(self.name)

    def z3_domain_constraint(self, variable: Any, context: "_Z3Context") -> Any:
        z3 = context.z3
        length = z3.Length(variable)
        range_constraint = z3.And(length >= self.min_length, length <= self.max_length)
        if not self.alphabet:
            return z3.And(range_constraint, variable == z3.StringVal(""))
        char_constraints = []
        for index in range(self.max_length):
            char = z3.SubString(variable, index, 1)
            char_constraints.append(
                z3.Implies(
                    length > index,
                    z3.Or(*(char == z3.StringVal(symbol) for symbol in self.alphabet)),
                )
            )
        return z3.And(range_constraint, *char_constraints)

    def z3_value_to_python(self, value: Any) -> str:
        return value.as_string()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": "bounded-string",
            "alphabet": list(self.alphabet),
            "min_length": self.min_length,
            "max_length": self.max_length,
        }


@dataclass(frozen=True, slots=True)
class NamedConstraint:
    name: str
    expression: Expression

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint names must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "expression": self.expression.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NamedConstraint":
        return cls(
            name=str(data["name"]),
            expression=_expression_from_dict(_require_mapping(data.get("expression"), "constraint expression")),
        )


@dataclass(frozen=True, slots=True)
class FiniteContractProblem:
    """A finite symbolic contract over Bool, enum, int, and bounded string domains."""

    variables: tuple[VariableDomain, ...]
    constraints: tuple[NamedConstraint, ...] = ()
    name: str = "finite-contract"

    def __post_init__(self) -> None:
        names = [variable.name for variable in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("finite contract variables must have unique names")
        object.__setattr__(self, "variables", tuple(sorted(self.variables, key=lambda variable: variable.name)))
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def solve(
        self,
        *,
        prefer_z3: bool = True,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
        query_cache: "SolverQueryCache | None" = None,
        artifact_hashes: Mapping[str, str] | None = None,
        supported_fragment_metadata: Mapping[str, object] | None = None,
    ) -> "SolverResult":
        if max_assignments is not None and max_assignments <= 0:
            raise ValueError("max_assignments must be positive when provided")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        cache_key: str | None = None
        if query_cache is not None:
            cache_key = self.solver_query_key(
                prefer_z3=prefer_z3,
                max_assignments=max_assignments,
                timeout_seconds=timeout_seconds,
                artifact_hashes=artifact_hashes,
                supported_fragment_metadata=supported_fragment_metadata,
            )
            cached = query_cache.get(cache_key)
            if cached is not None:
                return cached
        if prefer_z3:
            z3_result = self._solve_with_z3(timeout_seconds=timeout_seconds)
            if z3_result is not None:
                return query_cache.put(cache_key, z3_result) if query_cache is not None and cache_key is not None else z3_result
        result = self._solve_by_enumeration(max_assignments=max_assignments, timeout_seconds=timeout_seconds)
        if (
            prefer_z3
            and result.status is SolverStatus.UNKNOWN
            and result.reason is not None
            and "unsupported solver fragment" in result.reason
        ):
            result = replace(result, backend=SolverBackend.Z3)
        return query_cache.put(cache_key, result) if query_cache is not None and cache_key is not None else result

    def normalized_solver_query(
        self,
        *,
        prefer_z3: bool = True,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        supported_fragment_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return the deterministic payload used to fingerprint a solver query."""

        return {
            "schema": "promptabi.solver-query.v1",
            "problem": {
                "name": self.name,
                "variables": [_stable_json_value(variable.to_dict()) for variable in self.variables],
                "constraints": [
                    _stable_json_value(constraint.to_dict())
                    for constraint in sorted(self.constraints, key=lambda constraint: constraint.name)
                ],
                "z3_formulas": _normalized_z3_formulas(self) if prefer_z3 else (),
            },
            "options": {
                "prefer_z3": prefer_z3,
                "max_assignments": max_assignments,
                "timeout_milliseconds": None if timeout_seconds is None else max(1, int(timeout_seconds * 1000)),
            },
            "artifact_hashes": tuple(sorted((artifact_hashes or {}).items())),
            "supported_fragment_metadata": _stable_json_value(supported_fragment_metadata or {}),
            "solver_version_fingerprints": _solver_version_fingerprints(prefer_z3=prefer_z3),
        }

    def solver_query_key(
        self,
        *,
        prefer_z3: bool = True,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        supported_fragment_metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Hash normalized formula, artifact, fragment, and solver-version inputs."""

        payload = self.normalized_solver_query(
            prefer_z3=prefer_z3,
            max_assignments=max_assignments,
            timeout_seconds=timeout_seconds,
            artifact_hashes=artifact_hashes,
            supported_fragment_metadata=supported_fragment_metadata,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _solve_with_z3(self, *, timeout_seconds: float | None = None) -> "SolverResult | None":
        try:
            import z3  # type: ignore[import-not-found]
        except ImportError:
            return None

        try:
            context = _Z3Context(z3=z3, variables={})
            solver = z3.Solver()
            if timeout_seconds is not None:
                solver.set(timeout=max(1, int(timeout_seconds * 1000)))
            domain_constraints: list[Any] = []
            for variable in self.variables:
                z3_variable = variable.z3_variable(context)
                context.variables[variable.name] = z3_variable
                domain_constraint = variable.z3_domain_constraint(z3_variable, context)
                domain_constraints.append(domain_constraint)
                solver.add(domain_constraint)
            tracked_constraints: list[tuple[str, str, Any]] = []
            for index, constraint in enumerate(self.constraints):
                tracker = f"constraint_{index}_{constraint.name}"
                expression = constraint.expression.to_z3(context)
                tracked_constraints.append((tracker, constraint.name, expression))
                solver.assert_and_track(expression, tracker)
            status = solver.check()
            if status == z3.sat:
                model = solver.model()
                assignment = {
                    variable.name: variable.z3_value_to_python(model.eval(context.variables[variable.name], model_completion=True))
                    for variable in self.variables
                }
                return SolverResult(
                    status=SolverStatus.SAT,
                    backend=SolverBackend.Z3,
                    assignment=assignment,
                    checked_assignments=1,
                    budget_outcome=SolverBudgetOutcome.PROVED,
                    budget_reason="Z3 produced a satisfying model within the configured solver budget.",
                )
            if status == z3.unsat:
                core_trackers = tuple(str(item) for item in solver.unsat_core())
                core = _minimize_z3_unsat_core(
                    z3,
                    domain_constraints,
                    tracked_constraints,
                    core_trackers,
                )
                return SolverResult(
                    status=SolverStatus.UNSAT,
                    backend=SolverBackend.Z3,
                    unsat_core=core,
                    checked_assignments=0,
                    budget_outcome=SolverBudgetOutcome.PROVED,
                    budget_reason="Z3 proved unsatisfiability within the configured solver budget.",
                )
            reason = str(solver.reason_unknown() or "Z3 returned unknown for the supported fragment.")
            outcome = (
                SolverBudgetOutcome.TIMED_OUT
                if "timeout" in reason.lower()
                else SolverBudgetOutcome.ABSTAINED
            )
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                backend=SolverBackend.Z3,
                checked_assignments=0,
                reason=reason,
                budget_outcome=outcome,
                budget_reason=reason,
            )
        except (TypeError, ValueError, AttributeError, z3.Z3Exception) as exc:
            reason = f"unsupported solver fragment: {exc}"
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                backend=SolverBackend.Z3,
                checked_assignments=0,
                reason=reason,
                budget_outcome=SolverBudgetOutcome.ABSTAINED,
                budget_reason=reason,
            )

    def _solve_by_enumeration(
        self,
        *,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
    ) -> "SolverResult":
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        constraint_variables = tuple((constraint, _expression_variables(constraint.expression)) for constraint in self.constraints)
        checked = 0
        hit_assignment_cap = False
        hit_deadline = False
        unsupported_reason: str | None = None

        def timed_out() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def search(index: int, assignment: dict[str, object], assigned: frozenset[str]) -> tuple[str, dict[str, object] | None]:
            nonlocal checked, hit_assignment_cap, hit_deadline, unsupported_reason
            if timed_out():
                hit_deadline = True
                return "unknown", None
            if index == len(self.variables):
                checked += 1
                if max_assignments is not None and checked > max_assignments:
                    hit_assignment_cap = True
                    return "unknown", None
                try:
                    satisfied = all(bool(constraint.expression.evaluate(assignment)) for constraint in self.constraints)
                except (TypeError, ValueError, AttributeError) as exc:
                    unsupported_reason = f"unsupported solver fragment: {exc}"
                    return "unknown", None
                if satisfied:
                    return "sat", dict(assignment)
                return "unsat", None
            for constraint, variables in constraint_variables:
                if variables <= assigned:
                    try:
                        if not bool(constraint.expression.evaluate(assignment)):
                            return "pruned", None
                    except (TypeError, ValueError, AttributeError) as exc:
                        unsupported_reason = f"unsupported solver fragment: {exc}"
                        return "unknown", None
            variable = self.variables[index]
            for value in variable.values():
                assignment[variable.name] = value
                status, model = search(index + 1, assignment, assigned | {variable.name})
                assignment.pop(variable.name, None)
                if status in {"sat", "unknown"}:
                    return status, model
            return "unsat", None

        status, assignment = search(0, {}, frozenset())
        if status == "sat" and assignment is not None:
            return SolverResult(
                status=SolverStatus.SAT,
                backend=SolverBackend.FINITE_ENUMERATION,
                assignment=assignment,
                checked_assignments=checked,
                budget_outcome=SolverBudgetOutcome.PROVED,
                budget_reason="Finite enumeration found a satisfying assignment within the configured solver budget.",
            )
        if status == "unknown":
            if hit_deadline:
                outcome = SolverBudgetOutcome.TIMED_OUT
                reason = "finite enumeration timed out before exhausting the search domain"
            elif hit_assignment_cap:
                outcome = SolverBudgetOutcome.BOUNDED
                reason = "finite enumeration stopped at the configured assignment budget"
            elif unsupported_reason is not None:
                outcome = SolverBudgetOutcome.ABSTAINED
                reason = unsupported_reason
            else:
                outcome = SolverBudgetOutcome.ABSTAINED
                reason = "finite enumeration returned unknown without a proof object"
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                backend=SolverBackend.FINITE_ENUMERATION,
                checked_assignments=checked,
                reason=reason,
                budget_outcome=outcome,
                budget_reason=reason,
            )
        return SolverResult(
            status=SolverStatus.UNSAT,
            backend=SolverBackend.FINITE_ENUMERATION,
            unsat_core=self._enumerated_unsat_core(),
            checked_assignments=checked,
            budget_outcome=SolverBudgetOutcome.PROVED,
            budget_reason="Finite enumeration exhausted the modeled domain and proved unsatisfiability.",
        )

    def _enumerated_unsat_core(self) -> tuple[str, ...]:
        remaining = list(self.constraints)
        changed = True
        while changed and len(remaining) > 1:
            changed = False
            for constraint in tuple(remaining):
                candidate = [item for item in remaining if item is not constraint]
                if not _is_satisfiable(self.variables, candidate):
                    remaining = candidate
                    changed = True
                    break
        return tuple(constraint.name for constraint in remaining)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "variables": [variable.to_dict() for variable in self.variables],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FiniteContractProblem":
        variables = data.get("variables", ())
        constraints = data.get("constraints", ())
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
            raise ValueError("finite contract variables must be a sequence")
        if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
            raise ValueError("finite contract constraints must be a sequence")
        return cls(
            name=str(data.get("name", "finite-contract")),
            variables=tuple(_domain_from_dict(_require_mapping(item, "variable")) for item in variables),
            constraints=tuple(NamedConstraint.from_dict(_require_mapping(item, "constraint")) for item in constraints),
        )


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: SolverStatus
    backend: SolverBackend
    assignment: Mapping[str, object] | None = None
    unsat_core: tuple[str, ...] = ()
    checked_assignments: int = 0
    reason: str | None = None
    budget_outcome: SolverBudgetOutcome = SolverBudgetOutcome.ABSTAINED
    budget_reason: str | None = None
    cache_key: str | None = None
    cache_hit: bool = False

    @property
    def sat(self) -> bool:
        return self.status is SolverStatus.SAT

    @property
    def unsat(self) -> bool:
        return self.status is SolverStatus.UNSAT

    @property
    def conclusion(self) -> SolverConclusion:
        if self.status is SolverStatus.SAT:
            return SolverConclusion.COUNTEREXAMPLE
        if self.status is SolverStatus.UNSAT:
            return SolverConclusion.UNSAT_CORE_PROOF
        return SolverConclusion.ABSTENTION

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "status": self.status.value,
            "backend": self.backend.value,
            "conclusion": self.conclusion.value,
            "checked_assignments": self.checked_assignments,
            "solver_budget_outcome": self.budget_outcome.value,
        }
        if self.assignment is not None:
            data["assignment"] = dict(sorted(self.assignment.items()))
        if self.unsat_core:
            data["unsat_core"] = list(self.unsat_core)
        if self.reason is not None:
            data["reason"] = self.reason
        if self.budget_reason is not None:
            data["solver_budget_reason"] = self.budget_reason
        if self.cache_key is not None:
            data["cache_key"] = self.cache_key
            data["cache_hit"] = self.cache_hit
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolverResult":
        assignment = data.get("assignment")
        if assignment is not None and not isinstance(assignment, Mapping):
            raise ValueError("solver result assignment must be an object when present")
        unsat_core = data.get("unsat_core", ())
        if not isinstance(unsat_core, Sequence) or isinstance(unsat_core, (str, bytes)):
            raise ValueError("solver result unsat_core must be a sequence")
        reason = data.get("reason")
        budget_reason = data.get("solver_budget_reason")
        cache_key = data.get("cache_key")
        raw_outcome = data.get("solver_budget_outcome")
        if raw_outcome is None:
            budget_outcome = _legacy_solver_budget_outcome(
                SolverStatus(str(data["status"])),
                str(reason) if reason is not None else None,
            )
        else:
            budget_outcome = SolverBudgetOutcome(str(raw_outcome))
        return cls(
            status=SolverStatus(str(data["status"])),
            backend=SolverBackend(str(data["backend"])),
            assignment=dict(assignment) if assignment is not None else None,
            unsat_core=tuple(str(item) for item in unsat_core),
            checked_assignments=int(data.get("checked_assignments", 0)),
            reason=str(reason) if reason is not None else None,
            budget_outcome=budget_outcome,
            budget_reason=str(budget_reason) if budget_reason is not None else None,
            cache_key=str(cache_key) if cache_key is not None else None,
            cache_hit=bool(data.get("cache_hit", False)),
        )

    def with_cache_metadata(self, *, cache_key: str, cache_hit: bool) -> "SolverResult":
        return replace(self, cache_key=cache_key, cache_hit=cache_hit)


def _legacy_solver_budget_outcome(status: SolverStatus, reason: str | None) -> SolverBudgetOutcome:
    if status is not SolverStatus.UNKNOWN:
        return SolverBudgetOutcome.PROVED
    if reason is not None and "unsupported solver fragment" in reason:
        return SolverBudgetOutcome.ABSTAINED
    return SolverBudgetOutcome.ABSTAINED


@dataclass(slots=True)
class SolverQueryCache:
    """Deterministic finite/SMT query cache keyed by normalized solver inputs."""

    entries: dict[str, SolverResult] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, cache_key: str) -> SolverResult | None:
        result = self.entries.get(cache_key)
        if result is None:
            self.misses += 1
            return None
        self.hits += 1
        return result.with_cache_metadata(cache_key=cache_key, cache_hit=True)

    def put(self, cache_key: str, result: SolverResult) -> SolverResult:
        stored = result.with_cache_metadata(cache_key=cache_key, cache_hit=False)
        self.entries[cache_key] = stored
        return stored

    def solve(
        self,
        problem: FiniteContractProblem,
        *,
        prefer_z3: bool = True,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        supported_fragment_metadata: Mapping[str, object] | None = None,
    ) -> SolverResult:
        return problem.solve(
            prefer_z3=prefer_z3,
            max_assignments=max_assignments,
            timeout_seconds=timeout_seconds,
            query_cache=self,
            artifact_hashes=artifact_hashes,
            supported_fragment_metadata=supported_fragment_metadata,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "promptabi.solver-cache.v1",
            "hits": self.hits,
            "misses": self.misses,
            "entries": {
                key: result.with_cache_metadata(cache_key=key, cache_hit=False).to_dict()
                for key, result in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolverQueryCache":
        entries_data = data.get("entries", {})
        if not isinstance(entries_data, Mapping):
            raise ValueError("solver cache entries must be an object")
        entries: dict[str, SolverResult] = {}
        for key, value in entries_data.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"solver cache entry {key!r} must be an object")
            entries[str(key)] = SolverResult.from_dict(value)
        return cls(
            entries=entries,
            hits=int(data.get("hits", 0)),
            misses=int(data.get("misses", 0)),
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "SolverQueryCache":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


SOLVER_REPLAY_SCHEMA = "promptabi.solver-replay.v1"


@dataclass(frozen=True, slots=True)
class SolverReplayReport:
    """Result of replaying a reduced solver obligation from JSON."""

    replay_id: str
    query_key: str
    expected: SolverResult
    actual: SolverResult
    status_matches: bool
    stored_sat_witness_valid: bool | None
    query_environment_matches: bool

    @property
    def ok(self) -> bool:
        return self.status_matches and self.stored_sat_witness_valid is not False

    def to_dict(self) -> dict[str, object]:
        return {
            "replay_id": self.replay_id,
            "query_key": self.query_key,
            "ok": self.ok,
            "status_matches": self.status_matches,
            "stored_sat_witness_valid": self.stored_sat_witness_valid,
            "query_environment_matches": self.query_environment_matches,
            "expected": self.expected.to_dict(),
            "actual": self.actual.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SolverReplayFile:
    """Artifact-free replay for a reduced finite SMT obligation.

    Replay files intentionally store the symbolic obligation, options, solver
    provenance, expected logical result, and artifact hashes. They do not store
    artifact files, datasets, provider credentials, or prompts needed to derive
    the obligation. Literal tokens or schema/provider names that are part of the
    reduced formula may still appear because they are required to rerun it.
    """

    replay_id: str
    query_key: str
    problem: FiniteContractProblem
    options: Mapping[str, object]
    artifact_hashes: Mapping[str, str]
    supported_fragment_metadata: Mapping[str, object]
    normalized_query: Mapping[str, object]
    expected_result: SolverResult

    @classmethod
    def from_problem(
        cls,
        problem: FiniteContractProblem,
        *,
        replay_id: str | None = None,
        prefer_z3: bool = True,
        max_assignments: int | None = None,
        timeout_seconds: float | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        supported_fragment_metadata: Mapping[str, object] | None = None,
        expected_result: SolverResult | None = None,
    ) -> "SolverReplayFile":
        normalized_query = problem.normalized_solver_query(
            prefer_z3=prefer_z3,
            max_assignments=max_assignments,
            timeout_seconds=timeout_seconds,
            artifact_hashes=artifact_hashes,
            supported_fragment_metadata=supported_fragment_metadata,
        )
        query_key = problem.solver_query_key(
            prefer_z3=prefer_z3,
            max_assignments=max_assignments,
            timeout_seconds=timeout_seconds,
            artifact_hashes=artifact_hashes,
            supported_fragment_metadata=supported_fragment_metadata,
        )
        result = expected_result or problem.solve(
            prefer_z3=prefer_z3,
            max_assignments=max_assignments,
            timeout_seconds=timeout_seconds,
            artifact_hashes=artifact_hashes,
            supported_fragment_metadata=supported_fragment_metadata,
        )
        return cls(
            replay_id=replay_id or problem.name,
            query_key=query_key,
            problem=problem,
            options={
                "prefer_z3": prefer_z3,
                "max_assignments": max_assignments,
                "timeout_seconds": timeout_seconds,
            },
            artifact_hashes=dict(sorted((artifact_hashes or {}).items())),
            supported_fragment_metadata=dict(supported_fragment_metadata or {}),
            normalized_query=normalized_query,
            expected_result=result,
        )

    def replay(self) -> SolverReplayReport:
        prefer_z3 = bool(self.options.get("prefer_z3", True))
        max_assignments = self.options.get("max_assignments")
        timeout_seconds = self.options.get("timeout_seconds")
        actual = self.problem.solve(
            prefer_z3=prefer_z3,
            max_assignments=int(max_assignments) if max_assignments is not None else None,
            timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
            artifact_hashes=self.artifact_hashes,
            supported_fragment_metadata=self.supported_fragment_metadata,
        )
        stored_witness_valid = self._stored_sat_witness_valid()
        current_query = self.problem.normalized_solver_query(
            prefer_z3=prefer_z3,
            max_assignments=int(max_assignments) if max_assignments is not None else None,
            timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
            artifact_hashes=self.artifact_hashes,
            supported_fragment_metadata=self.supported_fragment_metadata,
        )
        return SolverReplayReport(
            replay_id=self.replay_id,
            query_key=self.query_key,
            expected=self.expected_result,
            actual=actual,
            status_matches=actual.status is self.expected_result.status,
            stored_sat_witness_valid=stored_witness_valid,
            query_environment_matches=_canonical_json(current_query) == _canonical_json(self.normalized_query),
        )

    def _stored_sat_witness_valid(self) -> bool | None:
        if self.expected_result.status is not SolverStatus.SAT:
            return None
        if self.expected_result.assignment is None:
            return False
        assignment = dict(self.expected_result.assignment)
        variable_names = {variable.name for variable in self.problem.variables}
        if set(assignment) != variable_names:
            return False
        return all(bool(constraint.expression.evaluate(assignment)) for constraint in self.problem.constraints)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SOLVER_REPLAY_SCHEMA,
            "replay_id": self.replay_id,
            "query_key": self.query_key,
            "problem": self.problem.to_dict(),
            "options": dict(self.options),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "supported_fragment_metadata": _stable_json_value(self.supported_fragment_metadata),
            "normalized_query": _stable_json_value(self.normalized_query),
            "expected_result": self.expected_result.to_dict(),
            "privacy": {
                "stores_artifact_files": False,
                "stores_datasets": False,
                "stores_provider_credentials": False,
                "stores_full_prompts": False,
                "stores_reduced_formula_literals": True,
                "note": (
                    "Replay uses only the reduced finite solver obligation. Literal tokens, "
                    "delimiters, provider names, or schema field names required by that formula "
                    "may appear; original artifact files and credential values are not loaded."
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolverReplayFile":
        schema = data.get("schema")
        if schema != SOLVER_REPLAY_SCHEMA:
            raise ValueError(f"unsupported solver replay schema: {schema!r}")
        problem = FiniteContractProblem.from_dict(_require_mapping(data.get("problem"), "problem"))
        options = _require_mapping(data.get("options", {}), "options")
        artifact_hashes = _require_mapping(data.get("artifact_hashes", {}), "artifact_hashes")
        supported_fragment_metadata = _require_mapping(
            data.get("supported_fragment_metadata", {}),
            "supported_fragment_metadata",
        )
        normalized_query = _require_mapping(data.get("normalized_query", {}), "normalized_query")
        expected_result = SolverResult.from_dict(_require_mapping(data.get("expected_result"), "expected_result"))
        return cls(
            replay_id=str(data.get("replay_id", problem.name)),
            query_key=str(data.get("query_key", "")),
            problem=problem,
            options=dict(options),
            artifact_hashes={str(key): str(value) for key, value in artifact_hashes.items()},
            supported_fragment_metadata=dict(supported_fragment_metadata),
            normalized_query=dict(normalized_query),
            expected_result=expected_result,
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "SolverReplayFile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def render_solver_replay_text(report: SolverReplayReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    witness = (
        "n/a"
        if report.stored_sat_witness_valid is None
        else ("valid" if report.stored_sat_witness_valid else "invalid")
    )
    environment = "matched" if report.query_environment_matches else "different solver environment"
    return "\n".join(
        (
            f"PromptABI solver replay: {report.replay_id}",
            f"status: {status}",
            f"query: {report.query_key}",
            f"expected: {report.expected.status.value} ({report.expected.backend.value})",
            f"actual: {report.actual.status.value} ({report.actual.backend.value})",
            f"stored SAT witness: {witness}",
            f"normalized query environment: {environment}",
            "",
        )
    )


def render_solver_replay_json(report: SolverReplayReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def _stable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_stable_json_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_stable_json_value(item) for item in value), key=repr))
    return repr(value)


def _canonical_json(value: object) -> str:
    return json.dumps(_stable_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _domain_from_dict(data: Mapping[str, object]) -> VariableDomain:
    domain_type = str(data.get("type", ""))
    name = str(data.get("name", ""))
    if domain_type == "bool":
        return BoolDomain(name)
    if domain_type == "enum":
        return EnumDomain(name, tuple(str(item) for item in _require_sequence(data.get("members", ()), "enum members")))
    if domain_type == "int-range":
        return IntRangeDomain(name, int(data["minimum"]), int(data["maximum"]))
    if domain_type == "bounded-string":
        return BoundedStringDomain(
            name,
            tuple(str(item) for item in _require_sequence(data.get("alphabet", ()), "bounded string alphabet")),
            min_length=int(data.get("min_length", 0)),
            max_length=int(data.get("max_length", 8)),
        )
    raise ValueError(f"unsupported variable domain type: {domain_type!r}")


def _expression_from_dict(data: Mapping[str, object]) -> Expression:
    if "var" in data:
        return Var(str(data["var"]))
    if "value" in data:
        value = data["value"]
        if not isinstance(value, (bool, int, str)):
            raise ValueError("expression value must be bool, int, or string")
        return Value(value)
    binary_builders: dict[str, type[Eq] | type[Ne] | type[Le] | type[Lt] | type[Ge] | type[Gt]] = {
        "eq": Eq,
        "ne": Ne,
        "le": Le,
        "lt": Lt,
        "ge": Ge,
        "gt": Gt,
    }
    for key, builder in binary_builders.items():
        if key in data:
            parts = _require_sequence(data[key], key)
            if len(parts) != 2:
                raise ValueError(f"{key} expression must have exactly two operands")
            return builder(
                _expression_from_dict(_require_mapping(parts[0], f"{key} left operand")),
                _expression_from_dict(_require_mapping(parts[1], f"{key} right operand")),
            )
    if "and" in data:
        return And(*(_expression_from_dict(_require_mapping(item, "and term")) for item in _require_sequence(data["and"], "and terms")))
    if "or" in data:
        return Or(*(_expression_from_dict(_require_mapping(item, "or term")) for item in _require_sequence(data["or"], "or terms")))
    if "not" in data:
        return Not(_expression_from_dict(_require_mapping(data["not"], "not term")))
    if "implies" in data:
        parts = _require_sequence(data["implies"], "implies operands")
        if len(parts) != 2:
            raise ValueError("implies expression must have exactly two operands")
        return Implies(
            _expression_from_dict(_require_mapping(parts[0], "implies condition")),
            _expression_from_dict(_require_mapping(parts[1], "implies consequence")),
        )
    if "in" in data:
        parts = _require_sequence(data["in"], "in operands")
        if len(parts) != 2:
            raise ValueError("in expression must have exactly two operands")
        values = _require_sequence(parts[1], "in values")
        parsed_values: list[bool | int | str] = []
        for value in values:
            if not isinstance(value, (bool, int, str)):
                raise ValueError("in values must be bool, int, or string")
            parsed_values.append(value)
        return InSet(
            _expression_from_dict(_require_mapping(parts[0], "in term")),
            parsed_values,
        )
    if "length" in data:
        return Length(_expression_from_dict(_require_mapping(data["length"], "length term")))
    if "sum" in data:
        return Sum(*(_expression_from_dict(_require_mapping(item, "sum term")) for item in _require_sequence(data["sum"], "sum terms")))
    if "mul" in data:
        parts = _require_sequence(data["mul"], "mul operands")
        if len(parts) != 2 or not isinstance(parts[0], int) or isinstance(parts[0], bool):
            raise ValueError("mul expression must be [integer-coefficient, term]")
        return Mul(int(parts[0]), _expression_from_dict(_require_mapping(parts[1], "mul term")))
    if "contains" in data:
        parts = _require_sequence(data["contains"], "contains operands")
        if len(parts) != 2:
            raise ValueError("contains expression must have exactly two operands")
        return Contains(
            _expression_from_dict(_require_mapping(parts[0], "contains haystack")),
            _expression_from_dict(_require_mapping(parts[1], "contains needle")),
        )
    raise ValueError(f"unsupported expression shape: {sorted(data)}")


def _solver_version_fingerprints(*, prefer_z3: bool) -> tuple[tuple[str, str], ...]:
    fingerprints = [("promptabi-finite-contract-solver", "v2-query-cache")]
    if prefer_z3:
        try:
            import z3  # type: ignore[import-not-found]
        except ImportError:
            fingerprints.append(("z3", "unavailable"))
        else:
            fingerprints.append(("z3", str(z3.get_version_string())))
    return tuple(fingerprints)


def _normalized_z3_formulas(problem: FiniteContractProblem) -> tuple[tuple[str, object], ...]:
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError:
        return (("z3", "unavailable"),)
    try:
        context = _Z3Context(z3=z3, variables={})
        domain_formulas: list[tuple[str, object]] = []
        for variable in problem.variables:
            z3_variable = variable.z3_variable(context)
            context.variables[variable.name] = z3_variable
            domain_formulas.append((f"domain:{variable.name}", z3.simplify(variable.z3_domain_constraint(z3_variable, context)).sexpr()))
        constraint_formulas: list[tuple[str, object]] = []
        for constraint in sorted(problem.constraints, key=lambda item: item.name):
            constraint_formulas.append((constraint.name, z3.simplify(constraint.expression.to_z3(context)).sexpr()))
        return tuple(domain_formulas + constraint_formulas)
    except (TypeError, ValueError, AttributeError, z3.Z3Exception) as exc:
        return (
            (
                "unsupported-z3-normalization",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "constraints": [
                        _stable_json_value(constraint.to_dict())
                        for constraint in sorted(problem.constraints, key=lambda item: item.name)
                    ],
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class _Z3Context:
    z3: Any
    variables: dict[str, Any]


def _assignments(variables: Sequence[VariableDomain]) -> Iterator[dict[str, object]]:
    for values in product(*(variable.values() for variable in variables)):
        yield {variable.name: value for variable, value in zip(variables, values, strict=True)}


def _is_satisfiable(variables: Sequence[VariableDomain], constraints: Sequence[NamedConstraint]) -> bool:
    constraint_variables = tuple((constraint, _expression_variables(constraint.expression)) for constraint in constraints)

    def search(index: int, assignment: dict[str, object], assigned: frozenset[str]) -> bool:
        for constraint, required_variables in constraint_variables:
            if required_variables <= assigned and not bool(constraint.expression.evaluate(assignment)):
                return False
        if index == len(variables):
            return True
        domain = variables[index]
        for value in domain.values():
            assignment[domain.name] = value
            if search(index + 1, assignment, assigned | {domain.name}):
                assignment.pop(domain.name, None)
                return True
            assignment.pop(domain.name, None)
        return False

    return search(0, {}, frozenset())


def _expression_variables(expression: Expression) -> frozenset[str]:
    if isinstance(expression, Var):
        return frozenset({expression.name})
    if isinstance(expression, Value):
        return frozenset()
    if isinstance(expression, (Eq, Ne, Le, Lt, Ge, Gt)):
        return _expression_variables(expression.left) | _expression_variables(expression.right)
    if isinstance(expression, Sum):
        variables: set[str] = set()
        for term in expression.terms:
            variables.update(_expression_variables(term))
        return frozenset(variables)
    if isinstance(expression, Mul):
        return _expression_variables(expression.term)
    if isinstance(expression, (And, Or)):
        variables: set[str] = set()
        for term in expression.terms:
            variables.update(_expression_variables(term))
        return frozenset(variables)
    if isinstance(expression, Not):
        return _expression_variables(expression.term)
    if isinstance(expression, Implies):
        return _expression_variables(expression.condition) | _expression_variables(expression.consequence)
    if isinstance(expression, InSet):
        return _expression_variables(expression.term)
    if isinstance(expression, Length):
        return _expression_variables(expression.term)
    if isinstance(expression, Contains):
        return _expression_variables(expression.haystack) | _expression_variables(expression.needle)
    return frozenset()


def _minimize_z3_unsat_core(
    z3: Any,
    domain_constraints: Sequence[Any],
    tracked_constraints: Sequence[tuple[str, str, Any]],
    core_trackers: Sequence[str],
) -> tuple[str, ...]:
    """Return a deletion-minimal unsat core over named constraints.

    Z3's tracked core is not guaranteed to be minimal. PromptABI diagnostics use
    the core as a user-facing proof object, so we deterministically shrink it
    while retaining all finite-domain restrictions as background assumptions.
    """

    by_tracker = {tracker: (name, expression) for tracker, name, expression in tracked_constraints}
    if core_trackers:
        remaining = [tracker for tracker, _, _ in tracked_constraints if tracker in set(core_trackers)]
    else:
        remaining = [tracker for tracker, _, _ in tracked_constraints]
    changed = True
    while changed and len(remaining) > 1:
        changed = False
        for tracker in tuple(remaining):
            candidate = [item for item in remaining if item != tracker]
            if _z3_constraints_unsat(
                z3,
                domain_constraints,
                tuple(by_tracker[item][1] for item in candidate),
            ):
                remaining = candidate
                changed = True
                break
    return tuple(by_tracker[tracker][0] for tracker in remaining)


def _z3_constraints_unsat(z3: Any, domain_constraints: Sequence[Any], constraints: Sequence[Any]) -> bool:
    solver = z3.Solver()
    solver.add(*domain_constraints)
    solver.add(*constraints)
    return solver.check() == z3.unsat
