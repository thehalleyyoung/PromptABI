"""First-class products for streaming parser state machines."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .formal import AutomatonError, DeterministicFiniteAutomaton, State, Symbol


STREAMING_PARSER_PRODUCT_VERSION = "promptabi.streaming-parser-product.v1"
_PAIR_SEPARATOR = "\u241f"


@dataclass(frozen=True, slots=True)
class StreamingParserTransition:
    """One transition in a streaming parser state machine."""

    source: State
    symbol: Symbol
    target: State

    def to_dict(self) -> dict[str, object]:
        return {"from": self.source, "symbol": self.symbol, "to": self.target}


@dataclass(frozen=True, slots=True)
class StreamingParserStateMachine:
    """A deterministic parser over streamed text symbols.

    The machine is intentionally finite: parser implementations choose a bounded
    abstraction and record that abstraction in ``approximation``.
    """

    states: frozenset[State]
    alphabet: tuple[Symbol, ...]
    start: State
    accepts: frozenset[State]
    error_states: frozenset[State]
    protected_states: frozenset[State]
    transitions: Mapping[tuple[State, Symbol], State]
    name: str = "streaming-parser"
    approximation: str = "exact"

    def __post_init__(self) -> None:
        if not self.states:
            raise AutomatonError("streaming parser products must define at least one state")
        if self.start not in self.states:
            raise AutomatonError("streaming parser start state must be declared")
        if not self.accepts <= self.states:
            raise AutomatonError("streaming parser accepting states must be declared states")
        if not self.error_states <= self.states:
            raise AutomatonError("streaming parser error states must be declared states")
        if not self.protected_states <= self.states:
            raise AutomatonError("streaming parser protected states must be declared states")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise AutomatonError("streaming parser alphabet must not contain duplicates")
        if any(symbol == "" for symbol in self.alphabet):
            raise AutomatonError("streaming parser symbols must be non-empty")
        alphabet = tuple(sorted(self.alphabet))
        normalized = dict(self.transitions)
        for (source, symbol), target in normalized.items():
            if source not in self.states:
                raise AutomatonError(f"streaming parser transition source is not declared: {source}")
            if target not in self.states:
                raise AutomatonError(f"streaming parser transition target is not declared: {target}")
            if symbol not in alphabet:
                raise AutomatonError(f"streaming parser transition symbol is not in alphabet: {symbol!r}")
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "transitions", normalized)

    def step(self, state: State, symbol: Symbol) -> State:
        if symbol not in self.alphabet:
            raise AutomatonError(f"streaming parser symbol is not in alphabet: {symbol!r}")
        return self.transitions.get((state, symbol), _ERROR_STATE)

    def replay(self, chunks: Sequence[str]) -> "StreamingParserReplay":
        """Replay chunks through the parser state machine."""

        state = self.start
        states = [state]
        events: list[Symbol] = []
        chunk_offsets: list[int] = []
        first_error_index: int | None = None
        for chunk_index, chunk in enumerate(chunks):
            for symbol in chunk:
                events.append(symbol)
                chunk_offsets.append(chunk_index)
                state = self.step(state, symbol)
                states.append(state)
                if state in self.error_states and first_error_index is None:
                    first_error_index = len(events) - 1
        return StreamingParserReplay(
            parser_name=self.name,
            chunks=tuple(chunks),
            events=tuple(events),
            chunk_offsets=tuple(chunk_offsets),
            states=tuple(states),
            accepted=state in self.accepts,
            error=state in self.error_states,
            first_error_index=first_error_index,
            final_state=state,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "approximation": self.approximation,
            "states": sorted(self.states),
            "alphabet": list(self.alphabet),
            "start": self.start,
            "accepts": sorted(self.accepts),
            "error_states": sorted(self.error_states),
            "protected_states": sorted(self.protected_states),
            "transitions": [
                StreamingParserTransition(source, symbol, target).to_dict()
                for (source, symbol), target in sorted(self.transitions.items())
            ],
        }


@dataclass(frozen=True, slots=True)
class StreamingParserReplay:
    """A replayed stream with parser state evidence."""

    parser_name: str
    chunks: tuple[str, ...]
    events: tuple[Symbol, ...]
    chunk_offsets: tuple[int, ...]
    states: tuple[State, ...]
    accepted: bool
    error: bool
    first_error_index: int | None
    final_state: State

    @property
    def complete(self) -> bool:
        return self.accepted and not self.error

    def to_dict(self) -> dict[str, object]:
        return {
            "parser_name": self.parser_name,
            "chunks": list(self.chunks),
            "events": list(self.events),
            "chunk_offsets": list(self.chunk_offsets),
            "states": list(self.states),
            "accepted": self.accepted,
            "error": self.error,
            "complete": self.complete,
            "first_error_index": self.first_error_index,
            "final_state": self.final_state,
        }


@dataclass(frozen=True, slots=True)
class StreamingParserMonitorViolation:
    """A monitor match observed while the parser was in a protected state."""

    monitor: str
    end_event_index: int
    chunk_index: int
    parser_state: State
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "monitor": self.monitor,
            "end_event_index": self.end_event_index,
            "chunk_index": self.chunk_index,
            "parser_state": self.parser_state,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class StreamingParserProductReport:
    """Product of a streaming parser and a substring monitor DFA."""

    version: str
    parser: StreamingParserStateMachine
    monitor: DeterministicFiniteAutomaton | None
    replay: StreamingParserReplay
    product_states: tuple[State, ...]
    violations: tuple[StreamingParserMonitorViolation, ...]

    @property
    def ok(self) -> bool:
        return self.replay.complete and not self.violations

    @property
    def guarantee_mode(self) -> str:
        if self.parser.approximation == "exact":
            return "complete"
        return "bounded"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ok": self.ok,
            "guarantee_mode": self.guarantee_mode,
            "parser": self.parser.to_dict(),
            "monitor": self.monitor.to_dict() if self.monitor is not None else None,
            "replay": self.replay.to_dict(),
            "product_states": list(self.product_states),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def build_json_boundary_streaming_parser(
    *,
    alphabet: Iterable[Symbol],
    max_depth: int = 8,
    name: str = "json-boundary-streaming-parser",
) -> StreamingParserStateMachine:
    """Build a bounded JSON boundary parser over a concrete stream alphabet.

    The abstraction tracks string/escape regions and balanced bracket depth. It
    is deliberately a boundary parser, not a full JSON value validator; callers
    can pair it with ``json.loads`` when complete JSON syntax is required.
    """

    if max_depth < 1:
        raise AutomatonError("json streaming parser max_depth must be positive")
    base_alphabet = set(alphabet).union({"{", "}", "[", "]", '"', "\\", " ", "\n", "\r", "\t"})
    symbols = tuple(sorted(symbol for symbol in base_alphabet if symbol))
    states = {_START_STATE, _DONE_STATE, _ERROR_STATE}
    transitions: dict[tuple[State, Symbol], State] = {}
    accepts = {_DONE_STATE}
    errors = {_ERROR_STATE}
    protected: set[State] = set()
    for depth in range(1, max_depth + 1):
        states.update({_outside_state(depth), _string_state(depth), _escape_state(depth)})
        protected.update({_string_state(depth), _escape_state(depth)})

    whitespace = {" ", "\n", "\r", "\t"}
    openings = {"{", "["}
    closings = {"}", "]"}
    for state in tuple(states):
        for symbol in symbols:
            transitions[(state, symbol)] = _ERROR_STATE
    for symbol in symbols:
        if symbol in whitespace:
            transitions[(_START_STATE, symbol)] = _START_STATE
            transitions[(_DONE_STATE, symbol)] = _DONE_STATE
        elif symbol in openings:
            transitions[(_START_STATE, symbol)] = _outside_state(1)
        else:
            transitions[(_START_STATE, symbol)] = _ERROR_STATE
            transitions[(_DONE_STATE, symbol)] = _ERROR_STATE
        transitions[(_ERROR_STATE, symbol)] = _ERROR_STATE

    for depth in range(1, max_depth + 1):
        outside = _outside_state(depth)
        in_string = _string_state(depth)
        escape = _escape_state(depth)
        for symbol in symbols:
            if symbol == '"':
                transitions[(outside, symbol)] = in_string
                transitions[(in_string, symbol)] = outside
                transitions[(escape, symbol)] = in_string
            elif symbol == "\\":
                transitions[(outside, symbol)] = outside
                transitions[(in_string, symbol)] = escape
                transitions[(escape, symbol)] = in_string
            elif symbol in openings:
                transitions[(outside, symbol)] = _outside_state(depth + 1) if depth < max_depth else _ERROR_STATE
                transitions[(in_string, symbol)] = in_string
                transitions[(escape, symbol)] = in_string
            elif symbol in closings:
                transitions[(outside, symbol)] = _outside_state(depth - 1) if depth > 1 else _DONE_STATE
                transitions[(in_string, symbol)] = in_string
                transitions[(escape, symbol)] = in_string
            else:
                transitions[(outside, symbol)] = outside
                transitions[(in_string, symbol)] = in_string
                transitions[(escape, symbol)] = in_string

    return StreamingParserStateMachine(
        states=frozenset(states),
        alphabet=symbols,
        start=_START_STATE,
        accepts=frozenset(accepts),
        error_states=frozenset(errors),
        protected_states=frozenset(protected),
        transitions=transitions,
        name=name,
        approximation=f"bounded-json-boundary-depth-{max_depth}",
    )


def build_substring_monitor(
    literal: str,
    *,
    alphabet: Iterable[Symbol],
    name: str | None = None,
) -> DeterministicFiniteAutomaton:
    """Build a DFA that accepts whenever the stream suffix equals ``literal``."""

    if not literal:
        raise AutomatonError("substring monitor literal must be non-empty")
    symbols = tuple(sorted(set(alphabet).union(literal)))
    states = frozenset(f"m{index}" for index in range(len(literal) + 1))
    transitions: dict[tuple[State, Symbol], State] = {}
    for index in range(len(literal) + 1):
        prefix = literal[:index]
        for symbol in symbols:
            candidate = prefix + symbol
            next_index = _longest_literal_prefix_suffix(literal, candidate)
            transitions[(f"m{index}", symbol)] = f"m{next_index}"
    return DeterministicFiniteAutomaton(
        states=states,
        alphabet=symbols,
        start="m0",
        accepts=frozenset({f"m{len(literal)}"}),
        transitions=transitions,
        name=name or f"contains({literal!r})",
    )


def analyze_streaming_parser_product(
    chunks: Sequence[str],
    *,
    monitor_literal: str | None = None,
    max_depth: int = 8,
    validate_json: bool = True,
) -> StreamingParserProductReport:
    """Analyze a streamed JSON boundary parser and optional monitor product."""

    alphabet = set("".join(chunks))
    if monitor_literal is not None:
        alphabet.update(monitor_literal)
    parser = build_json_boundary_streaming_parser(alphabet=alphabet, max_depth=max_depth)
    replay = parser.replay(chunks)
    monitor = build_substring_monitor(monitor_literal, alphabet=parser.alphabet) if monitor_literal else None
    product_states: list[State] = []
    violations: list[StreamingParserMonitorViolation] = []
    monitor_state = monitor.start if monitor is not None else None
    for index, symbol in enumerate(replay.events):
        parser_state = replay.states[index + 1]
        if monitor is not None and monitor_state is not None:
            next_monitor_state = monitor.step(monitor_state, symbol)
            if next_monitor_state is None:
                raise AutomatonError("substring monitor must be total")
            monitor_state = next_monitor_state
            product_states.append(_product_state(parser_state, monitor_state))
            if parser_state in parser.protected_states and monitor_state in monitor.accepts:
                violations.append(
                    StreamingParserMonitorViolation(
                        monitor=monitor_literal or "",
                        end_event_index=index,
                        chunk_index=replay.chunk_offsets[index],
                        parser_state=parser_state,
                        excerpt=_excerpt(replay.events, index, len(monitor_literal or "")),
                    )
                )
        else:
            product_states.append(parser_state)
    if validate_json and replay.complete:
        try:
            json.loads("".join(chunks))
        except json.JSONDecodeError:
            object.__setattr__(replay, "accepted", False)
    return StreamingParserProductReport(
        version=STREAMING_PARSER_PRODUCT_VERSION,
        parser=parser,
        monitor=monitor,
        replay=replay,
        product_states=tuple(product_states),
        violations=tuple(violations),
    )


def render_streaming_parser_product_json(report: StreamingParserProductReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_streaming_parser_product_text(report: StreamingParserProductReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        "PromptABI streaming parser product",
        f"status: {status}",
        f"guarantee: {report.guarantee_mode}",
        f"parser: {report.parser.name} ({report.parser.approximation})",
        f"chunks: {len(report.replay.chunks)}",
        f"events: {len(report.replay.events)}",
        f"final_state: {report.replay.final_state}",
        f"complete: {str(report.replay.complete).lower()}",
    ]
    if report.monitor is not None:
        lines.append(f"monitor: {report.monitor.name}")
    if report.replay.first_error_index is not None:
        lines.append(f"first_error_event: {report.replay.first_error_index}")
    for violation in report.violations:
        lines.append(
            "violation: "
            f"{violation.monitor!r} ended at event {violation.end_event_index} "
            f"in chunk {violation.chunk_index} while parser_state={violation.parser_state}; "
            f"excerpt={violation.excerpt!r}"
        )
    return "\n".join(lines) + "\n"


_START_STATE = "start"
_DONE_STATE = "done"
_ERROR_STATE = "error"


def _outside_state(depth: int) -> State:
    return f"outside:{depth}"


def _string_state(depth: int) -> State:
    return f"string:{depth}"


def _escape_state(depth: int) -> State:
    return f"escape:{depth}"


def _product_state(parser_state: State, monitor_state: State) -> State:
    return f"{parser_state}{_PAIR_SEPARATOR}{monitor_state}"


def _longest_literal_prefix_suffix(literal: str, candidate: str) -> int:
    max_length = min(len(literal), len(candidate))
    for length in range(max_length, -1, -1):
        if candidate.endswith(literal[:length]):
            return length
    return 0


def _excerpt(events: Sequence[Symbol], end_index: int, width: int) -> str:
    start = max(0, end_index - max(width, 8) + 1)
    stop = min(len(events), end_index + 1 + 8)
    return "".join(events[start:stop])
