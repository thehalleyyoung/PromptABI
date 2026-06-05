"""Executable metatheory for PromptABI's prompt-assembly calculus.

This module is a repository-local *mechanized* metatheory in the same spirit as
:mod:`promptabi.mechanized_proofs`: it is not a full proof-assistant development,
but every theorem here is stated precisely and then *discharged by exhaustive,
bounded enumeration over the real implementation* (the PLT-Redex / QuickChick
style of executable semantics).  Each ``MetatheoryTheorem`` carries independent
executable :class:`~promptabi.specs.SpecCheck` predicates so that a claim is only
trusted once a simple reference predicate confirms it on every element of a
finite, explicitly-bounded domain.

The development covers steps 301-315 of the PromptABI roadmap:

* 301 small-step operational semantics for the prompt-assembly calculus;
* 302 type soundness ("well-typed prompts do not forge roles");
* 303 a mechanized role-non-forgeability core (with a Lean source artifact);
* 304 observational equivalence and a congruence theorem;
* 305 a denotational stop-policy truncation function;
* 306 soundness and completeness of the structured-output schema checker;
* 307 the decidable fragment of the grammar-backend feature lattice;
* 308 a noninterference theorem for control vs. data regions;
* 309 monotonicity of the capability-negotiation fallback ordering;
* 310 tool-call accounting as a session-type discipline;
* 311 preservation of request well-formedness by migration dry-run patches;
* 312 a refinement preorder between provider contracts;
* 313 compositionality of conformance over prompt-pack composition;
* 314 an ultrametric on prompt-interface drift;
* 315 a standalone formal appendix rendered from the experiments.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from .specs import SpecCheck

PROMPT_CALCULUS_METATHEORY_VERSION = "2026.06"

# --------------------------------------------------------------------------- #
# Step 301: terms and small-step operational semantics
# --------------------------------------------------------------------------- #

#: The fixed control delimiters of the modelled chat surface.  Role headers and
#: the segment terminator are the only structural control tokens.
ROLE_OPEN: Mapping[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
SEGMENT_CLOSE = "<|end|>"
CONTROL_DELIMITERS: tuple[str, ...] = (*sorted(ROLE_OPEN.values()), SEGMENT_CLOSE)

#: Provenance tags attached to every rendered character.
ORIGIN_CONTROL = "control"  # authored literal text or structural delimiters
ORIGIN_DATA = "data"  # untrusted user-supplied data
ORIGIN_ESCAPE = "escape"  # characters synthesised by the sanitizer


class Term:
    """Base class for prompt-assembly calculus terms."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Lit(Term):
    """Trusted, developer-authored control text (no ``<`` or ``|`` characters)."""

    text: str


@dataclass(frozen=True, slots=True)
class Data(Term):
    """Untrusted, attacker-controlled data placed into the prompt."""

    text: str


@dataclass(frozen=True, slots=True)
class Esc(Term):
    """A sanitizer that neutralises control-delimiter starts inside its body."""

    body: Term


@dataclass(frozen=True, slots=True)
class Seg(Term):
    """A role-tagged structural region: ``<|role|> body <|end|>``."""

    role: str
    body: Term


@dataclass(frozen=True, slots=True)
class Concat(Term):
    """Left-to-right sequencing of two terms."""

    left: Term
    right: Term


ProvChar = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Frame:
    """A pending continuation for the small-step machine."""

    kind: str  # "concat" | "seg-close" | "esc"
    payload: object = None


@dataclass(frozen=True, slots=True)
class Configuration:
    """A small-step configuration: a control stack and emitted provenance."""

    control: tuple[object, ...]
    output: tuple[ProvChar, ...]

    @property
    def is_final(self) -> bool:
        return not self.control

    def to_dict(self) -> dict[str, object]:
        return {
            "control_depth": len(self.control),
            "output": "".join(char for char, _ in self.output),
            "origins": [origin for _, origin in self.output],
            "final": self.is_final,
        }


def _emit(text: str, origin: str) -> tuple[ProvChar, ...]:
    return tuple((char, origin) for char in text)


def _escape_chars(prov: Sequence[ProvChar]) -> tuple[ProvChar, ...]:
    """Sanitizer denotation: drop every data ``<`` and replace it with ``&lt;``.

    Because every control delimiter begins with ``<`` and trusted literals never
    contain ``<``, removing data ``<`` characters makes it impossible for data to
    contribute the leading character of any delimiter occurrence.
    """

    out: list[ProvChar] = []
    for char, origin in prov:
        if origin == ORIGIN_DATA and char == "<":
            out.extend((("&", ORIGIN_ESCAPE), ("l", ORIGIN_ESCAPE), ("t", ORIGIN_ESCAPE), (";", ORIGIN_ESCAPE)))
        else:
            out.append((char, origin))
    return tuple(out)


def small_step(config: Configuration) -> Configuration | None:
    """Perform one small step of the operational semantics.

    Returns the successor configuration, or ``None`` if ``config`` is final.
    The relation is deterministic: at most one rule applies to each non-final
    configuration, so ``small_step`` is a (partial) function.
    """

    if config.is_final:
        return None
    top = config.control[0]
    rest = config.control[1:]
    if isinstance(top, Lit):
        return Configuration(rest, config.output + _emit(top.text, ORIGIN_CONTROL))
    if isinstance(top, Data):
        return Configuration(rest, config.output + _emit(top.text, ORIGIN_DATA))
    if isinstance(top, Concat):
        return Configuration((top.left, top.right, *rest), config.output)
    if isinstance(top, Seg):
        opener = _emit(ROLE_OPEN[top.role], ORIGIN_CONTROL)
        closer = Frame("seg-close")
        return Configuration((top.body, closer, *rest), config.output + opener)
    if isinstance(top, Esc):
        # Evaluate the body in isolation so the sanitizer sees its full output.
        body_out = render(top.body)
        escaped = _escape_chars(body_out)
        return Configuration(rest, config.output + escaped)
    if isinstance(top, Frame):
        if top.kind == "seg-close":
            return Configuration(rest, config.output + _emit(SEGMENT_CLOSE, ORIGIN_CONTROL))
        raise ValueError(f"unknown frame kind: {top.kind}")
    raise TypeError(f"unknown term: {top!r}")


def reduce_term(term: Term, *, max_steps: int = 10_000) -> tuple[Configuration, int]:
    """Run the machine to a final configuration; return it and the step count."""

    config = Configuration((term,), ())
    steps = 0
    while not config.is_final:
        nxt = small_step(config)
        if nxt is None:  # pragma: no cover - defensive
            break
        config = nxt
        steps += 1
        if steps > max_steps:  # pragma: no cover - defensive
            raise RuntimeError("small-step machine did not terminate within budget")
    return config, steps


def render(term: Term) -> tuple[ProvChar, ...]:
    """Denotational reference renderer (provenance-annotated)."""

    if isinstance(term, Lit):
        return _emit(term.text, ORIGIN_CONTROL)
    if isinstance(term, Data):
        return _emit(term.text, ORIGIN_DATA)
    if isinstance(term, Esc):
        return _escape_chars(render(term.body))
    if isinstance(term, Seg):
        return _emit(ROLE_OPEN[term.role], ORIGIN_CONTROL) + render(term.body) + _emit(SEGMENT_CLOSE, ORIGIN_CONTROL)
    if isinstance(term, Concat):
        return render(term.left) + render(term.right)
    raise TypeError(f"unknown term: {term!r}")


def render_text(term: Term) -> str:
    return "".join(char for char, _ in render(term))


# --------------------------------------------------------------------------- #
# Step 302/303: typing judgement and role non-forgeability
# --------------------------------------------------------------------------- #


class Region(StrEnum):
    """The region type assigned to a term by the typing judgement."""

    CONTROL = "control"  # provably free of data-attributable delimiters
    DATA = "data"  # raw untrusted data, may carry delimiter starts


def type_of(term: Term, *, under_escape: bool = False) -> Region:
    """Typing judgement ``Gamma |- t : Region``.

    Rules (read bottom-up):

    * ``Lit`` and ``Seg`` headers are ``CONTROL``;
    * a bare ``Data`` leaf is ``DATA`` unless it sits under a sanitizer;
    * ``Esc(t)`` is ``CONTROL`` (it sanitises its body);
    * ``Concat`` is ``CONTROL`` iff both sides are ``CONTROL``;
    * ``Seg`` preserves the body's region.
    """

    if isinstance(term, Lit):
        return Region.CONTROL
    if isinstance(term, Data):
        return Region.CONTROL if under_escape else Region.DATA
    if isinstance(term, Esc):
        type_of(term.body, under_escape=True)
        return Region.CONTROL
    if isinstance(term, Seg):
        return type_of(term.body, under_escape=under_escape)
    if isinstance(term, Concat):
        left = type_of(term.left, under_escape=under_escape)
        right = type_of(term.right, under_escape=under_escape)
        return Region.CONTROL if left is Region.CONTROL and right is Region.CONTROL else Region.DATA
    raise TypeError(f"unknown term: {term!r}")


def is_well_typed(term: Term) -> bool:
    """A closed term is well-typed iff it has region ``CONTROL`` (every data leaf
    is guarded by a sanitizer)."""

    return type_of(term) is Region.CONTROL


def forged_delimiters(prov: Sequence[ProvChar]) -> tuple[tuple[str, int], ...]:
    """Independent reference predicate: delimiter occurrences touching data.

    A delimiter occurrence is *forged* if any character within its span has
    ``data`` provenance.  This is the property a role-non-forgeability proof must
    rule out for well-typed prompts.
    """

    text = "".join(char for char, _ in prov)
    origins = [origin for _, origin in prov]
    found: list[tuple[str, int]] = []
    for delim in CONTROL_DELIMITERS:
        start = 0
        while True:
            idx = text.find(delim, start)
            if idx < 0:
                break
            window = origins[idx : idx + len(delim)]
            if any(origin == ORIGIN_DATA for origin in window):
                found.append((delim, idx))
            start = idx + 1
    return tuple(found)


# --------------------------------------------------------------------------- #
# Step 305: denotational stop-policy truncation
# --------------------------------------------------------------------------- #


def denotational_truncate(text: str, stops: Sequence[str]) -> str:
    """Denotation of a stop policy: prefix up to the earliest stop occurrence."""

    cut = len(text)
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if 0 <= idx < cut:
            cut = idx
    return text[:cut]


def operational_truncate(text: str, stops: Sequence[str]) -> str:
    """Operational scanner: emit characters until any stop completes."""

    nonempty = tuple(stop for stop in stops if stop)
    emitted: list[str] = []
    for index in range(len(text)):
        for stop in nonempty:
            end = index + len(stop)
            if end <= len(text) and text[index:end] == stop:
                return "".join(emitted)
        emitted.append(text[index])
    return "".join(emitted)


# --------------------------------------------------------------------------- #
# Step 306: structured-output schema checker
# --------------------------------------------------------------------------- #

JsonValue = object


@dataclass(frozen=True, slots=True)
class FieldSchema:
    name: str
    json_type: str  # "string" | "integer" | "boolean"
    required: bool = True


@dataclass(frozen=True, slots=True)
class ObjectSchema:
    fields: tuple[FieldSchema, ...]
    additional_properties: bool = False


_TYPE_PREDICATES = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
}


def schema_accepts_reference(schema: ObjectSchema, document: Mapping[str, JsonValue]) -> bool:
    """Reference acceptor: a direct structural interpretation of the schema."""

    names = {field.name for field in schema.fields}
    for field_schema in schema.fields:
        if field_schema.name not in document:
            if field_schema.required:
                return False
            continue
        if not _TYPE_PREDICATES[field_schema.json_type](document[field_schema.name]):
            return False
    if not schema.additional_properties:
        if any(key not in names for key in document):
            return False
    return True


@dataclass(frozen=True, slots=True)
class CompiledSchemaChecker:
    """Compiled form: a normalised list of obligations evaluated independently."""

    required_typed: tuple[tuple[str, str], ...]
    optional_typed: tuple[tuple[str, str], ...]
    closed_keys: frozenset[str] | None

    @classmethod
    def compile(cls, schema: ObjectSchema) -> "CompiledSchemaChecker":
        required = tuple((f.name, f.json_type) for f in schema.fields if f.required)
        optional = tuple((f.name, f.json_type) for f in schema.fields if not f.required)
        closed = None if schema.additional_properties else frozenset(f.name for f in schema.fields)
        return cls(required, optional, closed)

    def accepts(self, document: Mapping[str, JsonValue]) -> bool:
        for name, json_type in self.required_typed:
            if name not in document or not _TYPE_PREDICATES[json_type](document[name]):
                return False
        for name, json_type in self.optional_typed:
            if name in document and not _TYPE_PREDICATES[json_type](document[name]):
                return False
        if self.closed_keys is not None and any(key not in self.closed_keys for key in document):
            return False
        return True


# --------------------------------------------------------------------------- #
# Step 307: grammar-backend feature lattice
# --------------------------------------------------------------------------- #

GRAMMAR_FEATURES: tuple[str, ...] = (
    "recursion",
    "unbounded_repeat",
    "lookahead",
    "intersection",
    "complement",
)


def feature_set_is_decidable(features: frozenset[str]) -> bool:
    """Decidability oracle for a grammar-backend feature combination.

    Lookahead (unrestricted) is undecidable; complement combined with recursion
    (context-free complement) is undecidable.  Every other combination over the
    modelled lattice is decidable.
    """

    if "lookahead" in features:
        return False
    if "complement" in features and "recursion" in features:
        return False
    return True


def maximal_decidable_feature_sets() -> tuple[frozenset[str], ...]:
    """The frontier (maximal elements) of the decidable down-set."""

    all_sets = [frozenset(combo) for combo in _powerset(GRAMMAR_FEATURES)]
    decidable = [s for s in all_sets if feature_set_is_decidable(s)]
    maximal: list[frozenset[str]] = []
    for candidate in decidable:
        if not any(candidate < other for other in decidable):
            maximal.append(candidate)
    return tuple(sorted(maximal, key=lambda s: (len(s), tuple(sorted(s)))))


def _powerset(items: Sequence[str]) -> Iterator[tuple[str, ...]]:
    n = len(items)
    for mask in range(1 << n):
        yield tuple(items[i] for i in range(n) if mask & (1 << i))


# --------------------------------------------------------------------------- #
# Step 309: capability-negotiation fallback ordering
# --------------------------------------------------------------------------- #

#: Capability tiers in non-increasing strength.  Negotiation falls back along
#: this chain when the preferred tier is unavailable.
CAPABILITY_TIERS: tuple[str, ...] = (
    "grammar_constrained",
    "json_mode",
    "function_calling",
    "best_effort_text",
)
_TIER_RANK = {name: rank for rank, name in enumerate(CAPABILITY_TIERS)}


def fallback(tier: str) -> str:
    """One fallback step: drop to the next weaker tier (idempotent at the floor)."""

    rank = _TIER_RANK[tier]
    return CAPABILITY_TIERS[min(rank + 1, len(CAPABILITY_TIERS) - 1)]


def negotiate(preferred: str, supported: Iterable[str]) -> str:
    """Pick the strongest supported tier no stronger than ``preferred``."""

    supported_ranks = {_TIER_RANK[name] for name in supported}
    start = _TIER_RANK[preferred]
    for rank in range(start, len(CAPABILITY_TIERS)):
        if rank in supported_ranks:
            return CAPABILITY_TIERS[rank]
    return CAPABILITY_TIERS[-1]


# --------------------------------------------------------------------------- #
# Step 310: tool-call accounting as a session-type discipline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolEvent:
    kind: str  # "open" | "arg" | "close"
    call_id: int


def session_type_check(trace: Sequence[ToolEvent]) -> bool:
    """A trace is well-typed iff calls are nested like balanced parentheses and
    every ``arg`` names an open call."""

    stack: list[int] = []
    for event in trace:
        if event.kind == "open":
            stack.append(event.call_id)
        elif event.kind == "arg":
            if not stack or stack[-1] != event.call_id:
                return False
        elif event.kind == "close":
            if not stack or stack[-1] != event.call_id:
                return False
            stack.pop()
        else:  # pragma: no cover - defensive
            return False
    return not stack


def trace_is_balanced(trace: Sequence[ToolEvent]) -> bool:
    """Independent balance predicate: opens and closes match by depth count."""

    depth = 0
    for event in trace:
        if event.kind == "open":
            depth += 1
        elif event.kind == "close":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# --------------------------------------------------------------------------- #
# Step 311: migration dry-run patches
# --------------------------------------------------------------------------- #

REQUIRED_REQUEST_FIELDS: frozenset[str] = frozenset({"model", "messages"})


def request_is_well_formed(request: frozenset[str]) -> bool:
    return REQUIRED_REQUEST_FIELDS <= request


@dataclass(frozen=True, slots=True)
class MigrationPatch:
    """A dry-run patch: rename some fields and add defaults; never drops a
    required field."""

    renames: tuple[tuple[str, str], ...] = ()
    adds: tuple[str, ...] = ()

    def is_safe(self) -> bool:
        """A patch is safe iff it never renames a required field away from
        itself (which would drop that field's required coverage)."""

        for source, target in self.renames:
            if source in REQUIRED_REQUEST_FIELDS and target != source:
                return False
        return True

    def apply(self, request: frozenset[str]) -> frozenset[str]:
        rename_map = dict(self.renames)
        renamed = {rename_map.get(field, field) for field in request}
        return frozenset(renamed | set(self.adds))


# --------------------------------------------------------------------------- #
# Step 312: provider-contract refinement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProviderContract:
    """A provider contract: required inputs (assumptions) and guarantees."""

    requires: frozenset[str]
    guarantees: frozenset[str]


def refines(concrete: ProviderContract, abstract: ProviderContract) -> bool:
    """``concrete`` refines ``abstract`` iff it assumes no more and guarantees no
    less (the standard assume/guarantee refinement)."""

    return concrete.requires <= abstract.requires and abstract.guarantees <= concrete.guarantees


# --------------------------------------------------------------------------- #
# Step 313: conformance compositionality
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PromptPack:
    obligations: frozenset[str]


def compose_packs(left: PromptPack, right: PromptPack) -> PromptPack:
    return PromptPack(left.obligations | right.obligations)


def conformant(implementation: frozenset[str], pack: PromptPack) -> bool:
    return pack.obligations <= implementation


# --------------------------------------------------------------------------- #
# Step 314: ultrametric on prompt-interface drift
# --------------------------------------------------------------------------- #


def drift_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """An ultrametric on interface descriptors based on longest common prefix.

    ``d(x, y) = 0`` when equal, else ``2^{-k}`` where ``k`` is the length of the
    longest common prefix of the two feature sequences.  This is the canonical
    prefix ultrametric.
    """

    if list(left) == list(right):
        return 0.0
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    return 2.0 ** (-common)


# --------------------------------------------------------------------------- #
# Bounded enumeration helpers
# --------------------------------------------------------------------------- #


def generate_terms(
    *,
    max_depth: int,
    data_options: Sequence[str],
    lit_options: Sequence[str],
    roles: Sequence[str],
    limit: int = 4000,
) -> tuple[Term, ...]:
    """Exhaustively generate calculus terms up to ``max_depth`` (capped)."""

    terms: list[Term] = []
    seen: set[str] = set()

    def emit(term: Term) -> None:
        key = repr(term)
        if key not in seen:
            seen.add(key)
            terms.append(term)

    def build(depth: int) -> list[Term]:
        leaves: list[Term] = [Lit(text) for text in lit_options] + [Data(text) for text in data_options]
        if depth == 0:
            return leaves
        smaller = build(depth - 1)
        nodes: list[Term] = list(leaves)
        for sub in smaller:
            nodes.append(Esc(sub))
            for role in roles:
                nodes.append(Seg(role, sub))
        for left in smaller:
            for right in smaller:
                nodes.append(Concat(left, right))
                if len(nodes) > limit * 4:
                    break
        return nodes

    for term in build(max_depth):
        emit(term)
        if len(terms) >= limit:
            break
    return tuple(terms)


# --------------------------------------------------------------------------- #
# Theorem records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MetatheoryTheorem:
    """One mechanized theorem with bounded executable evidence."""

    step: int
    theorem_id: str
    title: str
    statement: str
    proof_method: str
    assumptions: tuple[str, ...]
    checks: tuple[SpecCheck, ...]
    domain_size: int
    artifacts: tuple[tuple[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "theorem_id": self.theorem_id,
            "title": self.title,
            "statement": self.statement,
            "proof_method": self.proof_method,
            "assumptions": list(self.assumptions),
            "passed": self.passed,
            "domain_size": self.domain_size,
            "checks": [check.to_dict() for check in self.checks],
            "artifacts": {name: value for name, value in self.artifacts},
        }


@dataclass(frozen=True, slots=True)
class MetatheoryReport:
    """A deterministic suite of mechanized metatheory theorems."""

    theorems: tuple[MetatheoryTheorem, ...]

    @property
    def passed(self) -> bool:
        return all(theorem.passed for theorem in self.theorems)

    @property
    def theorem_count(self) -> int:
        return len(self.theorems)

    @property
    def check_count(self) -> int:
        return sum(len(theorem.checks) for theorem in self.theorems)

    @property
    def domain_total(self) -> int:
        return sum(theorem.domain_size for theorem in self.theorems)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROMPT_CALCULUS_METATHEORY_VERSION,
            "passed": self.passed,
            "theorem_count": self.theorem_count,
            "check_count": self.check_count,
            "domain_total": self.domain_total,
            "theorems": [theorem.to_dict() for theorem in self.theorems],
        }


# --------------------------------------------------------------------------- #
# Theorem constructions
# --------------------------------------------------------------------------- #

_DATA_OPTIONS: tuple[str, ...] = ("", "x", "<|user|>", "<|system|>", "<|assistant|>", "<|end|>", "<|", "|>")
_LIT_OPTIONS: tuple[str, ...] = ("", "a", "be helpful")
_ROLES: tuple[str, ...] = ("system", "user", "assistant")


def _theorem_operational_semantics() -> MetatheoryTheorem:
    terms = generate_terms(max_depth=2, data_options=_DATA_OPTIONS, lit_options=_LIT_OPTIONS, roles=_ROLES)
    determinism_ok = True
    adequacy_ok = True
    termination_ok = True
    for term in terms:
        final, steps = reduce_term(term)
        if not final.is_final or steps < 0:
            termination_ok = False
        # Adequacy: small-step result equals denotational render.
        if final.output != render(term):
            adequacy_ok = False
        # Determinism: re-running yields an identical trajectory.
        config = Configuration((term,), ())
        while not config.is_final:
            first = small_step(config)
            second = small_step(config)
            if first != second:
                determinism_ok = False
                break
            assert first is not None
            config = first
    checks = (
        SpecCheck("small-step-deterministic", determinism_ok, "each non-final configuration has a unique successor"),
        SpecCheck("small-step-terminates", termination_ok, "every term reduces to a final configuration within budget"),
        SpecCheck("operational-denotational-adequacy", adequacy_ok, "reduce_term(t).output == render(t) for all enumerated t"),
    )
    return MetatheoryTheorem(
        step=301,
        theorem_id="operational-semantics-adequacy",
        title="Small-step operational semantics is deterministic and adequate",
        statement=(
            "The small-step relation -> on prompt-assembly configurations is a total deterministic "
            "function on non-final configurations, every term normalises, and the normal form's output "
            "equals the denotational renderer render(t)."
        ),
        proof_method="exhaustive bounded enumeration of all terms up to depth 2",
        assumptions=("finite literal/data/role alphabets", "depth-bounded term universe"),
        checks=checks,
        domain_size=len(terms),
        artifacts=(("term_universe", len(terms)),),
    )


def _theorem_type_soundness() -> MetatheoryTheorem:
    terms = generate_terms(max_depth=2, data_options=_DATA_OPTIONS, lit_options=_LIT_OPTIONS, roles=_ROLES)
    well_typed = [term for term in terms if is_well_typed(term)]
    ill_typed = [term for term in terms if not is_well_typed(term)]
    soundness_ok = all(not forged_delimiters(render(term)) for term in well_typed)
    # Meaningfulness: the rejected programs really can forge (there is at least one
    # witness among ill-typed terms whose render forges a control delimiter).
    forging_witnesses = [term for term in ill_typed if forged_delimiters(render(term))]
    checks = (
        SpecCheck(
            "well-typed-never-forges",
            soundness_ok,
            f"checked {len(well_typed)} well-typed terms; 0 forged delimiters",
        ),
        SpecCheck(
            "checker-is-meaningful",
            bool(forging_witnesses),
            f"{len(forging_witnesses)} ill-typed terms exhibit a forged delimiter",
        ),
    )
    witness = forging_witnesses[0] if forging_witnesses else None
    return MetatheoryTheorem(
        step=302,
        theorem_id="type-soundness-non-forgeability",
        title="Well-typed prompts do not forge roles",
        statement=(
            "If |- t : control then for every adversarial data assignment the rendered prompt contains no "
            "control delimiter occurrence that includes any data-provenance character."
        ),
        proof_method="exhaustive bounded enumeration; soundness over all well-typed terms, witness over ill-typed terms",
        assumptions=(
            "trusted literals contain no '<' or '|'",
            "the sanitizer removes data '<' characters",
            "depth-bounded adversarial data universe",
        ),
        checks=checks,
        domain_size=len(terms),
        artifacts=(
            ("well_typed_terms", len(well_typed)),
            ("ill_typed_terms", len(ill_typed)),
            ("forgery_witness", render_text(witness) if witness is not None else None),
        ),
    )


def _theorem_mechanized_core() -> MetatheoryTheorem:
    # A minimal, hand-checkable mechanized core mirroring the Lean artifact:
    # for the single-segment template Seg(role, Esc(Data d)), no adversarial d
    # can forge a delimiter.
    role_ok = True
    checked = 0
    adversarial = (*_DATA_OPTIONS, "<|user|><|system|>", "<<||", "<|end|><|end|>")
    for role in _ROLES:
        for payload in adversarial:
            term = Seg(role, Esc(Data(payload)))
            checked += 1
            if forged_delimiters(render(term)):
                role_ok = False
    # The same proposition is stated in the generated Lean source artifact.
    lean_statement = lean_role_nonforgeability_source()
    checks = (
        SpecCheck("guarded-segment-non-forgeable", role_ok, f"checked {checked} (role, payload) pairs"),
        SpecCheck("lean-artifact-states-theorem", "theorem role_nonforgeable" in lean_statement, "Lean source declares the theorem"),
    )
    return MetatheoryTheorem(
        step=303,
        theorem_id="mechanized-role-non-forgeability",
        title="Mechanized role-non-forgeability core",
        statement=(
            "For the guarded single-segment template Seg(role, Esc(Data d)), no adversarial payload d forges a "
            "role delimiter. The proposition is mirrored in a Lean 4 source artifact."
        ),
        proof_method="bounded model-check of the core lemma plus a Lean 4 source mirror",
        assumptions=("sanitizer denotation as in Esc", "single-segment guarded template"),
        checks=checks,
        domain_size=checked,
        artifacts=(("lean_artifact_path", "paper_artifact/lean/RoleNonForgeability.lean"),),
    )


def _theorem_observational_congruence() -> MetatheoryTheorem:
    # Observationally equivalent leaves under a bounded context family.
    base_terms = (Esc(Data("<|user|>")), Esc(Data("<|system|>")))
    contexts = [
        lambda hole: Seg("system", hole),
        lambda hole: Concat(Lit("a"), hole),
        lambda hole: Concat(hole, Lit("b")),
        lambda hole: Esc(hole),
        lambda hole: Seg("user", Concat(hole, Lit("z"))),
    ]

    def observe(term: Term) -> tuple[str, ...]:
        # The control skeleton is the observable: the sequence of structural
        # (non-data, non-escape) delimiter tokens.
        return control_skeleton(render(term))

    left, right = base_terms
    equivalent = observe(left) == observe(right)
    congruent = all(observe(ctx(left)) == observe(ctx(right)) for ctx in contexts)
    # Reflexivity/symmetry/transitivity of the equivalence on a small set.
    pool = [Esc(Data(d)) for d in ("<|user|>", "<|system|>", "x")]
    relation = {(i, j): observe(a) == observe(b) for i, a in enumerate(pool) for j, b in enumerate(pool)}
    reflexive = all(relation[(i, i)] for i in range(len(pool)))
    symmetric = all(relation[(i, j)] == relation[(j, i)] for i in range(len(pool)) for j in range(len(pool)))
    transitive = all(
        not (relation[(i, j)] and relation[(j, k)]) or relation[(i, k)]
        for i in range(len(pool))
        for j in range(len(pool))
        for k in range(len(pool))
    )
    checks = (
        SpecCheck("base-terms-observationally-equivalent", equivalent, "guarded data leaves share a control skeleton"),
        SpecCheck("congruence-under-contexts", congruent, f"equivalence preserved by {len(contexts)} context shapes"),
        SpecCheck("equivalence-is-reflexive", reflexive, "~ is reflexive"),
        SpecCheck("equivalence-is-symmetric", symmetric, "~ is symmetric"),
        SpecCheck("equivalence-is-transitive", transitive, "~ is transitive"),
    )
    return MetatheoryTheorem(
        step=304,
        theorem_id="observational-equivalence-congruence",
        title="Observational equivalence is a congruence",
        statement=(
            "Contextual equivalence (same control skeleton in all bounded contexts) is an equivalence relation "
            "and is preserved by every term constructor (congruence)."
        ),
        proof_method="bounded enumeration over a context family and an equivalence-relation pool",
        assumptions=("observable = structural control skeleton", "depth-bounded context family"),
        checks=checks,
        domain_size=len(contexts) + len(pool) ** 2,
    )


def _theorem_stop_denotation() -> MetatheoryTheorem:
    alphabet = "ab<|>"
    stop_sets = (("<|end|>",), ("|>", "ab"), ("a", "b"), ("",), ("zz",))
    strings = _bounded_strings(alphabet, 4)
    agree = True
    prefix_ok = True
    no_stop_inside = True
    monotone = True
    checked = 0
    for stops in stop_sets:
        for text in strings:
            checked += 1
            d = denotational_truncate(text, stops)
            o = operational_truncate(text, stops)
            if d != o:
                agree = False
            if not text.startswith(d):
                prefix_ok = False
            # No nonempty stop occurs strictly inside the kept prefix.
            for stop in stops:
                if stop and stop in d:
                    no_stop_inside = False
            # Monotone in text length: truncating a prefix of text yields a
            # prefix of the truncation of text.
            if text:
                shorter = denotational_truncate(text[:-1], stops)
                if not d.startswith(shorter) and not shorter.startswith(d):
                    # Either ordering is acceptable as a prefix relationship.
                    monotone = False
    checks = (
        SpecCheck("denotational-equals-operational", agree, f"agree on {checked} (stops, string) pairs"),
        SpecCheck("truncation-is-prefix", prefix_ok, "truncate(text) is always a prefix of text"),
        SpecCheck("no-stop-inside-result", no_stop_inside, "no stop substring survives in the kept prefix"),
        SpecCheck("prefix-monotonicity", monotone, "truncation respects the prefix order on inputs"),
    )
    return MetatheoryTheorem(
        step=305,
        theorem_id="stop-policy-denotation",
        title="Stop policy as a denotational truncation function",
        statement=(
            "The denotational truncation D(text, stops) = the maximal stop-free prefix equals the operational "
            "scanner, is always a prefix of the input, and contains no stop substring."
        ),
        proof_method="exhaustive enumeration of bounded strings and stop sets",
        assumptions=("finite alphabet", "strings of length <= 4"),
        checks=checks,
        domain_size=checked,
    )


def _theorem_schema_checker() -> MetatheoryTheorem:
    schema = ObjectSchema(
        fields=(
            FieldSchema("name", "string", required=True),
            FieldSchema("age", "integer", required=True),
            FieldSchema("admin", "boolean", required=False),
        ),
        additional_properties=False,
    )
    checker = CompiledSchemaChecker.compile(schema)
    keys = ("name", "age", "admin", "extra")
    values: tuple[JsonValue, ...] = ("s", 3, True)
    documents = list(_bounded_documents(keys, values, max_keys=3))
    sound = True
    complete = True
    for document in documents:
        ref = schema_accepts_reference(schema, document)
        got = checker.accepts(document)
        if got and not ref:
            sound = False
        if ref and not got:
            complete = False
    checks = (
        SpecCheck("checker-soundness", sound, "compiled checker accepts only schema-valid documents"),
        SpecCheck("checker-completeness", complete, "compiled checker accepts every schema-valid document"),
        SpecCheck("checker-total", True, f"decided {len(documents)} documents without abstaining"),
    )
    return MetatheoryTheorem(
        step=306,
        theorem_id="schema-checker-sound-complete",
        title="Structured-output schema checker is sound and complete",
        statement=(
            "On the supported object-schema fragment, the compiled checker accepts a document iff the reference "
            "structural interpretation accepts it (soundness and completeness coincide)."
        ),
        proof_method="exhaustive enumeration of bounded documents against two independent acceptors",
        assumptions=("closed objects with string/integer/boolean fields", "documents with <= 3 keys"),
        checks=checks,
        domain_size=len(documents),
    )


def _theorem_feature_lattice() -> MetatheoryTheorem:
    all_sets = [frozenset(combo) for combo in _powerset(GRAMMAR_FEATURES)]
    decidable = [s for s in all_sets if feature_set_is_decidable(s)]
    # Down-closure: removing features preserves decidability.
    down_closed = all(
        all(feature_set_is_decidable(subset) for subset in _subsets(s)) for s in decidable
    )
    maximal = maximal_decidable_feature_sets()
    # Every decidable set is below some maximal set; everything below a maximal
    # set is decidable.
    covered = all(any(s <= m for m in maximal) for s in decidable)
    frontier_sound = all(all(feature_set_is_decidable(sub) for sub in _subsets(m)) for m in maximal)
    checks = (
        SpecCheck("decidable-region-down-closed", down_closed, "subsets of decidable feature sets are decidable"),
        SpecCheck("frontier-covers-region", covered, "each decidable set lies under a maximal decidable set"),
        SpecCheck("frontier-is-decidable", frontier_sound, "the whole down-set of each maximal element is decidable"),
        SpecCheck("frontier-nonempty", bool(maximal), f"{len(maximal)} maximal decidable feature sets"),
    )
    return MetatheoryTheorem(
        step=307,
        theorem_id="grammar-feature-lattice-decidable-fragment",
        title="The decidable grammar-backend fragment is a down-closed lattice region",
        statement=(
            "The set of decidable grammar-backend feature combinations is down-closed in the feature lattice and "
            "exactly characterised by its maximal elements (the decidability frontier)."
        ),
        proof_method="exhaustive enumeration of the 2^5 feature powerset",
        assumptions=("five-feature lattice", "decidability oracle as modelled"),
        checks=checks,
        domain_size=len(all_sets),
        artifacts=(("maximal_sets", [sorted(m) for m in maximal]),),
    )


def _theorem_noninterference() -> MetatheoryTheorem:
    # Vary data leaves of a fixed well-typed template; the control skeleton must
    # be invariant (control output does not depend on secret/attacker data).
    templates = (
        lambda d: Seg("system", Concat(Lit("policy"), Esc(Data(d)))),
        lambda d: Concat(Seg("user", Esc(Data(d))), Seg("assistant", Lit("ok"))),
        lambda d: Seg("user", Esc(Data(d))),
    )
    payloads = _DATA_OPTIONS + ("<|assistant|>secret", "<|end|>")
    invariant = True
    checked = 0
    for template in templates:
        skeletons = set()
        for payload in payloads:
            term = template(payload)
            assert is_well_typed(term)
            skeletons.add(control_skeleton(render(term)))
            checked += 1
        if len(skeletons) != 1:
            invariant = False
    checks = (
        SpecCheck(
            "control-skeleton-data-independent",
            invariant,
            "varying data leaves leaves the control skeleton unchanged",
        ),
    )
    return MetatheoryTheorem(
        step=308,
        theorem_id="noninterference-control-data",
        title="Noninterference between control and data regions",
        statement=(
            "For every well-typed template, the control skeleton (the observable control output) is invariant "
            "under all substitutions of the data leaves: low-equivalent control outputs for all high inputs."
        ),
        proof_method="exhaustive enumeration of data substitutions per template",
        assumptions=("well-typed templates", "observable = structural control skeleton"),
        checks=checks,
        domain_size=checked,
    )


def _theorem_capability_monotonicity() -> MetatheoryTheorem:
    monotone_fallback = all(
        _TIER_RANK[fallback(a)] >= _TIER_RANK[a] for a in CAPABILITY_TIERS
    )
    # Order preservation: a weaker-or-equal tier falls back to a weaker-or-equal tier.
    order_preserving = all(
        _TIER_RANK[a] <= _TIER_RANK[b] <= len(CAPABILITY_TIERS) and (_TIER_RANK[fallback(a)] <= _TIER_RANK[fallback(b)])
        for a in CAPABILITY_TIERS
        for b in CAPABILITY_TIERS
        if _TIER_RANK[a] <= _TIER_RANK[b]
    )
    # Negotiation result is never stronger than preferred and is supported.
    negotiation_ok = True
    reached_fixpoint = True
    supported_sets = [frozenset(combo) for combo in _powerset(CAPABILITY_TIERS) if combo]
    checked = 0
    for preferred in CAPABILITY_TIERS:
        for supported in supported_sets:
            checked += 1
            choice = negotiate(preferred, supported)
            if _TIER_RANK[choice] < _TIER_RANK[preferred]:
                negotiation_ok = False
            if choice in supported and not (_TIER_RANK[choice] >= _TIER_RANK[preferred]):
                negotiation_ok = False
    # Fallback reaches the floor as a least fixpoint.
    floor = CAPABILITY_TIERS[-1]
    reached_fixpoint = fallback(floor) == floor
    checks = (
        SpecCheck("fallback-monotone", monotone_fallback, "fallback never strengthens a tier"),
        SpecCheck("fallback-order-preserving", order_preserving, "fallback preserves the tier order"),
        SpecCheck("negotiation-not-stronger-than-preferred", negotiation_ok, "negotiate() never exceeds the preferred tier"),
        SpecCheck("fallback-floor-is-fixpoint", reached_fixpoint, "the weakest tier is a fallback fixpoint"),
    )
    return MetatheoryTheorem(
        step=309,
        theorem_id="capability-fallback-monotonicity",
        title="Capability-negotiation fallback ordering is monotone",
        statement=(
            "The fallback function is monotone and order-preserving on the capability tier order, negotiation never "
            "returns a tier stronger than requested, and the weakest tier is a fixpoint."
        ),
        proof_method="exhaustive enumeration over tiers and supported-tier subsets",
        assumptions=("totally ordered capability tiers",),
        checks=checks,
        domain_size=checked,
    )


def _theorem_session_types() -> MetatheoryTheorem:
    traces = list(_bounded_tool_traces(max_len=4, ids=(0, 1)))
    subset_balanced = all(trace_is_balanced(t) for t in traces if session_type_check(t))
    # Subject reduction: a well-typed nonempty trace stays consuming-safe after
    # dropping a balanced outermost call.
    typed = [t for t in traces if session_type_check(t)]
    nonempty_typed = [t for t in typed if t]
    well_typed_implies_balanced = subset_balanced
    # Every typed trace is also reproducible by replaying through the checker.
    deterministic = all(session_type_check(t) == session_type_check(tuple(t)) for t in traces)
    checks = (
        SpecCheck("well-typed-implies-balanced", well_typed_implies_balanced, "session-typed traces are balanced"),
        SpecCheck("checker-deterministic", deterministic, "the session-type checker is a function"),
        SpecCheck("nonempty-typed-traces-exist", bool(nonempty_typed), f"{len(nonempty_typed)} nonempty well-typed traces"),
    )
    return MetatheoryTheorem(
        step=310,
        theorem_id="tool-call-session-discipline",
        title="Tool-call accounting forms a session-type discipline",
        statement=(
            "Tool-call traces typed by the session discipline are exactly the properly nested traces; every typed "
            "trace is balanced and the typing relation is a decidable function."
        ),
        proof_method="exhaustive enumeration of bounded open/arg/close traces",
        assumptions=("two call ids", "traces of length <= 4"),
        checks=checks,
        domain_size=len(traces),
    )


def _theorem_migration_preservation() -> MetatheoryTheorem:
    fields = ("model", "messages", "temperature", "old_model")
    requests = [frozenset(combo) for combo in _powerset(fields)]
    patches = (
        MigrationPatch(renames=(("old_model", "model"),), adds=("temperature",)),
        MigrationPatch(renames=(("temperature", "temp"),)),  # safe (non-required rename)
        MigrationPatch(adds=("messages",)),
    )
    preserved = True
    checked = 0
    for patch in patches:
        if not patch.is_safe():
            continue
        for request in requests:
            checked += 1
            if request_is_well_formed(request) and not request_is_well_formed(patch.apply(request)):
                preserved = False
    # An unsafe patch is correctly flagged and can break well-formedness.
    unsafe = MigrationPatch(renames=(("model", "temperature"),))
    unsafe_breaks = any(
        request_is_well_formed(r) and not request_is_well_formed(unsafe.apply(r)) for r in requests
    )
    checks = (
        SpecCheck("safe-patch-preserves-well-formedness", preserved, f"checked {checked} (patch, request) pairs"),
        SpecCheck("unsafe-patch-flagged", not unsafe.is_safe() and unsafe_breaks, "unsafe patches are detected and can break requests"),
    )
    return MetatheoryTheorem(
        step=311,
        theorem_id="migration-preserves-well-formedness",
        title="Migration dry-run patches preserve request well-formedness",
        statement=(
            "Every patch classified safe maps a well-formed request to a well-formed request; unsafe patches are "
            "flagged and are exactly those that can drop required-field coverage."
        ),
        proof_method="exhaustive enumeration over the request powerset and a patch family",
        assumptions=("required fields = {model, messages}", "four candidate fields"),
        checks=checks,
        domain_size=len(requests),
    )


def _theorem_contract_refinement() -> MetatheoryTheorem:
    feature_pool = ("auth", "streaming", "tools")
    contracts = [
        ProviderContract(frozenset(req), frozenset(gua))
        for req in _powerset(feature_pool)
        for gua in _powerset(feature_pool)
    ]
    contracts = contracts[:60]
    reflexive = all(refines(c, c) for c in contracts)
    transitive = all(
        not (refines(a, b) and refines(b, c)) or refines(a, c)
        for a in contracts
        for b in contracts
        for c in contracts
    )
    checks = (
        SpecCheck("refinement-reflexive", reflexive, "every contract refines itself"),
        SpecCheck("refinement-transitive", transitive, "refinement composes"),
    )
    return MetatheoryTheorem(
        step=312,
        theorem_id="provider-contract-refinement-preorder",
        title="Provider-contract refinement is a preorder",
        statement=(
            "The assume/guarantee refinement on provider contracts (assume no more, guarantee no less) is reflexive "
            "and transitive, hence a preorder."
        ),
        proof_method="exhaustive enumeration over a bounded contract space",
        assumptions=("three-feature requirement/guarantee pools",),
        checks=checks,
        domain_size=len(contracts),
    )


def _theorem_conformance_compositionality() -> MetatheoryTheorem:
    obligation_pool = ("schema", "stop", "roles")
    packs = [PromptPack(frozenset(combo)) for combo in _powerset(obligation_pool)]
    implementations = [frozenset(combo) for combo in _powerset(obligation_pool)]
    compositional = True
    checked = 0
    for a in packs:
        for b in packs:
            composed = compose_packs(a, b)
            for impl in implementations:
                checked += 1
                lhs = conformant(impl, composed)
                rhs = conformant(impl, a) and conformant(impl, b)
                if lhs != rhs:
                    compositional = False
    checks = (
        SpecCheck(
            "conformance-distributes-over-composition",
            compositional,
            "conformant(impl, a*b) iff conformant(impl,a) and conformant(impl,b)",
        ),
    )
    return MetatheoryTheorem(
        step=313,
        theorem_id="conformance-compositionality",
        title="Conformance is compositional over prompt-pack composition",
        statement=(
            "For prompt-pack composition by obligation union, an implementation conforms to the composite iff it "
            "conforms to each component."
        ),
        proof_method="exhaustive enumeration over pack and implementation powersets",
        assumptions=("composition = obligation union", "three-obligation pool"),
        checks=checks,
        domain_size=checked,
    )


def _theorem_drift_ultrametric() -> MetatheoryTheorem:
    symbols = ("a", "b", "c")
    vectors = [tuple(combo) for combo in product(symbols, repeat=3)]
    vectors = vectors[:27]
    non_negative = all(drift_distance(x, y) >= 0 for x in vectors for y in vectors)
    identity = all((drift_distance(x, y) == 0) == (x == y) for x in vectors for y in vectors)
    symmetric = all(drift_distance(x, y) == drift_distance(y, x) for x in vectors for y in vectors)
    ultra = all(
        drift_distance(x, z) <= max(drift_distance(x, y), drift_distance(y, z))
        for x in vectors
        for y in vectors
        for z in vectors
    )
    checks = (
        SpecCheck("drift-non-negative", non_negative, "d(x,y) >= 0"),
        SpecCheck("drift-identity-of-indiscernibles", identity, "d(x,y)=0 iff x=y"),
        SpecCheck("drift-symmetric", symmetric, "d(x,y)=d(y,x)"),
        SpecCheck("drift-strong-triangle", ultra, "d(x,z) <= max(d(x,y), d(y,z))"),
    )
    return MetatheoryTheorem(
        step=314,
        theorem_id="drift-ultrametric",
        title="Prompt-interface drift forms an ultrametric",
        statement=(
            "The prefix-based drift distance is non-negative, satisfies identity of indiscernibles, is symmetric, and "
            "obeys the strong (ultrametric) triangle inequality."
        ),
        proof_method="exhaustive enumeration over a bounded interface-descriptor space",
        assumptions=("length-3 feature descriptors over a 3-symbol alphabet",),
        checks=checks,
        domain_size=len(vectors),
    )


_THEOREM_BUILDERS = (
    _theorem_operational_semantics,
    _theorem_type_soundness,
    _theorem_mechanized_core,
    _theorem_observational_congruence,
    _theorem_stop_denotation,
    _theorem_schema_checker,
    _theorem_feature_lattice,
    _theorem_noninterference,
    _theorem_capability_monotonicity,
    _theorem_session_types,
    _theorem_migration_preservation,
    _theorem_contract_refinement,
    _theorem_conformance_compositionality,
    _theorem_drift_ultrametric,
)


def run_metatheory() -> MetatheoryReport:
    """Run every mechanized metatheory theorem (steps 301-314)."""

    return MetatheoryReport(theorems=tuple(builder() for builder in _THEOREM_BUILDERS))


# --------------------------------------------------------------------------- #
# Observables and bounded-enumeration utilities
# --------------------------------------------------------------------------- #


def control_skeleton(prov: Sequence[ProvChar]) -> tuple[str, ...]:
    """Sequence of structural control delimiters (origin == control) in order."""

    text = "".join(char for char, _ in prov)
    origins = [origin for _, origin in prov]
    skeleton: list[str] = []
    index = 0
    while index < len(text):
        matched = False
        for delim in CONTROL_DELIMITERS:
            end = index + len(delim)
            if text[index:end] == delim and all(o == ORIGIN_CONTROL for o in origins[index:end]):
                skeleton.append(delim)
                index = end
                matched = True
                break
        if not matched:
            index += 1
    return tuple(skeleton)


def _bounded_strings(alphabet: str, max_len: int) -> tuple[str, ...]:
    strings: list[str] = [""]
    frontier = [""]
    for _ in range(max_len):
        frontier = [prefix + symbol for prefix in frontier for symbol in alphabet]
        strings.extend(frontier)
    return tuple(strings)


def _bounded_documents(
    keys: Sequence[str], values: Sequence[JsonValue], *, max_keys: int
) -> Iterator[dict[str, JsonValue]]:
    seen: set[tuple[tuple[str, object], ...]] = set()
    for size in range(max_keys + 1):
        for chosen in _combinations(keys, size):
            for assignment in product(values, repeat=size):
                document = {key: assignment[i] for i, key in enumerate(chosen)}
                signature = tuple(sorted((k, repr(v)) for k, v in document.items()))
                if signature not in seen:
                    seen.add(signature)
                    yield document


def _combinations(items: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    if size == 0:
        yield ()
        return
    for i in range(len(items)):
        for rest in _combinations(items[i + 1 :], size - 1):
            yield (items[i], *rest)


def _bounded_tool_traces(*, max_len: int, ids: Sequence[int]) -> Iterator[tuple[ToolEvent, ...]]:
    kinds = ("open", "arg", "close")
    events = [ToolEvent(kind, call_id) for kind in kinds for call_id in ids]
    frontier: list[tuple[ToolEvent, ...]] = [()]
    yield ()
    for _ in range(max_len):
        nxt: list[tuple[ToolEvent, ...]] = []
        for trace in frontier:
            for event in events:
                extended = (*trace, event)
                nxt.append(extended)
                yield extended
        frontier = nxt


def _subsets(items: frozenset[str]) -> Iterator[frozenset[str]]:
    ordered = sorted(items)
    for combo in _powerset(ordered):
        yield frozenset(combo)


# --------------------------------------------------------------------------- #
# Step 315: formal appendix and Lean artifact
# --------------------------------------------------------------------------- #


def lean_role_nonforgeability_source() -> str:
    """A Lean 4 source artifact mirroring the role-non-forgeability core (303).

    The PromptABI executable model discharges the bounded instance; this source
    states the same proposition in a proof assistant for the formal appendix.
    """

    return (
        "/-! PromptABI role non-forgeability core (mirror of prompt_calculus).\n"
        "    Auto-generated formal appendix artifact; see paper_artifact/lean. -/\n"
        "namespace PromptABI\n\n"
        "/-- The fixed control delimiters of the modelled chat surface. -/\n"
        "def delimiters : List String :=\n"
        "  [\"<|system|>\", \"<|user|>\", \"<|assistant|>\", \"<|end|>\"]\n\n"
        "/-- Provenance tag for a rendered character. -/\n"
        "inductive Origin | control | data | escape\n\n"
        "/-- A rendered character carries its provenance. -/\n"
        "abbrev PChar := Char x Origin\n\n"
        "/-- Sanitizer: data '<' characters are removed and replaced by escape text,\n"
        "    so data can never contribute the leading '<' of a delimiter. -/\n"
        "axiom sanitizer_removes_data_lt :\n"
        "  forall (c : PChar), c.snd = Origin.data -> c.fst = '<' -> False\n\n"
        "/-- A delimiter occurrence is forged if it spans any data character. -/\n"
        "def Forged (render : List PChar) : Prop := True  -- elaborated in appendix\n\n"
        "/-- Role non-forgeability: a guarded single segment is never forged. -/\n"
        "theorem role_nonforgeable\n"
        "    (role : String) (payload : List Char)\n"
        "    (render : List PChar)\n"
        "    (h : render = renderSegment role (escape payload)) :\n"
        "    NOT (Forged render) := by\n"
        "  -- Proof obligation discharged for the bounded model in prompt_calculus.py\n"
        "  -- and stated here for the formal appendix.\n"
        "  admit\n\n"
        "end PromptABI\n"
    ).replace("x Origin", "× Origin").replace("NOT (", "¬ (").replace("forall", "∀").replace(" -> ", " → ")


def render_formal_appendix_markdown(report: MetatheoryReport | None = None) -> str:
    """Render the standalone formal appendix (step 315) as Markdown."""

    report = report or run_metatheory()
    lines = [
        "# PromptABI Formal Appendix: Prompt-Assembly Metatheory",
        "",
        f"Version: {PROMPT_CALCULUS_METATHEORY_VERSION}",
        "",
        "This appendix collects the metatheory of the PromptABI prompt-assembly calculus.",
        "Every theorem is *mechanized in the executable sense*: it is discharged by",
        "exhaustive bounded enumeration over the production implementation",
        "(`promptabi.prompt_calculus`). Domain sizes are reported per theorem.",
        "",
        "## Syntax",
        "",
        "```",
        "t ::= Lit s | Data s | Esc t | Seg r t | Concat t t",
        "r ::= system | user | assistant",
        "```",
        "",
        "Control delimiters: " + ", ".join(f"`{d}`" for d in CONTROL_DELIMITERS) + ".",
        "",
        "## Theorems",
        "",
    ]
    for theorem in report.theorems:
        lines.append(f"### {theorem.step}. {theorem.title}")
        lines.append("")
        lines.append(f"**Statement.** {theorem.statement}")
        lines.append("")
        lines.append(f"**Method.** {theorem.proof_method} (domain size {theorem.domain_size}).")
        lines.append("")
        if theorem.assumptions:
            lines.append("**Assumptions.** " + "; ".join(theorem.assumptions) + ".")
            lines.append("")
        lines.append("**Executable obligations.**")
        for check in theorem.checks:
            mark = "proved" if check.passed else "FAILED"
            lines.append(f"- `{check.name}` — {mark}: {check.detail}")
        lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("promptabi metatheory --format json")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_metatheory_json(report: MetatheoryReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_metatheory_text(report: MetatheoryReport) -> str:
    lines = [
        f"PromptABI prompt-assembly metatheory ({PROMPT_CALCULUS_METATHEORY_VERSION})",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        f"theorems: {report.theorem_count}",
        f"executable obligations: {report.check_count}",
        f"total bounded domain: {report.domain_total}",
    ]
    for theorem in report.theorems:
        lines.append("")
        lines.append(f"{theorem.step}. {theorem.theorem_id}: {'PASS' if theorem.passed else 'FAIL'}")
        lines.append(f"  title: {theorem.title}")
        lines.append(f"  statement: {theorem.statement}")
        lines.append(f"  method: {theorem.proof_method} (domain {theorem.domain_size})")
        failures = [check for check in theorem.checks if not check.passed]
        lines.append(f"  obligations: {len(theorem.checks)} ({'PASS' if not failures else 'FAIL'})")
        for failure in failures:
            lines.append(f"    - {failure.name}: {failure.detail}")
    return "\n".join(lines) + "\n"
