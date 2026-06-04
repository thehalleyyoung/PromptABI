"""Finite automata and finite-domain SMT primitives for PromptABI.

The classes in this module are intentionally small, deterministic, and CPU-only.
They are the executable core used by early checkers before richer template,
grammar, and tokenizer products exist.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
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


def _product(
    left: DeterministicFiniteAutomaton,
    right: DeterministicFiniteAutomaton,
    accept: Callable[[bool, bool], bool],
    *,
    name: str,
) -> DeterministicFiniteAutomaton:
    alphabet = tuple(sorted(set(left.alphabet).union(right.alphabet)))
    left_complete = left.complete(alphabet=alphabet)
    right_complete = right.complete(alphabet=alphabet)
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


class SolverStatus(StrEnum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class SolverBackend(StrEnum):
    Z3 = "z3"
    FINITE_ENUMERATION = "finite-enumeration"


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

    def solve(self, *, prefer_z3: bool = True) -> "SolverResult":
        if prefer_z3:
            z3_result = self._solve_with_z3()
            if z3_result is not None:
                return z3_result
        return self._solve_by_enumeration()

    def _solve_with_z3(self) -> "SolverResult | None":
        try:
            import z3  # type: ignore[import-not-found]
        except ImportError:
            return None

        context = _Z3Context(z3=z3, variables={})
        solver = z3.Solver()
        for variable in self.variables:
            z3_variable = variable.z3_variable(context)
            context.variables[variable.name] = z3_variable
            solver.add(variable.z3_domain_constraint(z3_variable, context))
        tracker_names: dict[str, str] = {}
        for index, constraint in enumerate(self.constraints):
            tracker = f"constraint_{index}_{constraint.name}"
            tracker_names[tracker] = constraint.name
            solver.assert_and_track(constraint.expression.to_z3(context), tracker)
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
            )
        if status == z3.unsat:
            core = tuple(tracker_names.get(str(item), str(item)) for item in solver.unsat_core())
            return SolverResult(
                status=SolverStatus.UNSAT,
                backend=SolverBackend.Z3,
                unsat_core=core or tuple(tracker_names.values()),
                checked_assignments=0,
            )
        return SolverResult(status=SolverStatus.UNKNOWN, backend=SolverBackend.Z3, checked_assignments=0)

    def _solve_by_enumeration(self) -> "SolverResult":
        checked = 0
        for assignment in _assignments(self.variables):
            checked += 1
            if all(bool(constraint.expression.evaluate(assignment)) for constraint in self.constraints):
                return SolverResult(
                    status=SolverStatus.SAT,
                    backend=SolverBackend.FINITE_ENUMERATION,
                    assignment=dict(assignment),
                    checked_assignments=checked,
                )
        return SolverResult(
            status=SolverStatus.UNSAT,
            backend=SolverBackend.FINITE_ENUMERATION,
            unsat_core=self._enumerated_unsat_core(),
            checked_assignments=checked,
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


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: SolverStatus
    backend: SolverBackend
    assignment: Mapping[str, object] | None = None
    unsat_core: tuple[str, ...] = ()
    checked_assignments: int = 0

    @property
    def sat(self) -> bool:
        return self.status is SolverStatus.SAT

    @property
    def unsat(self) -> bool:
        return self.status is SolverStatus.UNSAT

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "status": self.status.value,
            "backend": self.backend.value,
            "checked_assignments": self.checked_assignments,
        }
        if self.assignment is not None:
            data["assignment"] = dict(sorted(self.assignment.items()))
        if self.unsat_core:
            data["unsat_core"] = list(self.unsat_core)
        return data


@dataclass(frozen=True, slots=True)
class _Z3Context:
    z3: Any
    variables: dict[str, Any]


def _assignments(variables: Sequence[VariableDomain]) -> Iterator[dict[str, object]]:
    for values in product(*(variable.values() for variable in variables)):
        yield {variable.name: value for variable, value in zip(variables, values, strict=True)}


def _is_satisfiable(variables: Sequence[VariableDomain], constraints: Sequence[NamedConstraint]) -> bool:
    return any(
        all(bool(constraint.expression.evaluate(assignment)) for constraint in constraints)
        for assignment in _assignments(variables)
    )
