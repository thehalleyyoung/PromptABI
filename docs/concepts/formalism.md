# Formalism

PromptABI models the LLM interface layer as a composition of finite languages,
finite-state transductions, and finite symbolic contracts. The point is not to
model a neural network. The point is to prove facts about the bytes, strings,
tokens, parser states, and request envelopes that exist before a model chooses a
next token.

## Objects in the model

| Object | Implementation | What it represents |
| --- | --- | --- |
| DFA | `promptabi.formal.DeterministicFiniteAutomaton` | A finite language such as a literal delimiter, a bounded grammar witness language, or a parser prefix language. |
| FST | `promptabi.formal.FiniteStateTransducer` | A relation such as render, tokenize, decode, escape, or provider serialization. |
| Finite contract | `promptabi.formal.FiniteContractProblem` | Boolean, enum, integer, membership, length, and bounded-string obligations lowered to Z3 or finite enumeration. |
| Witness | `AutomatonWitness`, `TransducerWitness`, `SolverResult` | A shortest path, input/output pair, satisfying assignment, or unsat core suitable for a diagnostic. |

The finite automata layer intentionally uses explicit states and alphabets. That
makes it small enough to audit and deterministic enough for CI snapshots. Large
products are searched lazily when possible; the implementation records explored
states, explored transitions, alphabet size, and representative symbols so a
diagnostic can distinguish an exact proof from a bounded search.

## Languages, relations, and products

PromptABI uses the following operations repeatedly:

1. **Intersection** asks whether two constraints can hold at once. A stop
   sequence intersected with a parser-prefix language yields a concrete firing
   witness if the stop is reachable in the parser state.
2. **Difference** asks whether a supposedly safe language excludes a forbidden
   language. User content minus role delimiters should still contain all allowed
   user strings and no control marker.
3. **Projection** turns a transducer relation into the language of its inputs or
   outputs. This is how a render/tokenize relation can be summarized for a
   downstream parser or provider adapter.
4. **Composition** chains relations. A template-render transducer followed by a
   tokenizer transducer gives a direct relation from application fields to token
   strings.
5. **Over-approximation** is explicit. If a relation cannot be kept exact, the
   approximation flag is part of the object and downstream diagnostics must avoid
   claiming more than the approximation supports.

These operations are executable in `src/promptabi/formal.py` and are covered by
`tests/test_formal.py`. The tests use concrete strings such as `</s>` and
`<user>` rather than mock-only placeholders, so documentation claims track real
code behavior.

## Proof modes

Every diagnostic carries check-mode metadata:

| Mode | Meaning in this formalism |
| --- | --- |
| `sound` | A reported violation exists under the stated abstraction. |
| `complete` | All violations inside the supported fragment are found. |
| `bounded` | The proof is exact for the declared finite bound, such as a maximum candidate count or template path bound. |
| `z3-backed-smt` | A finite contract was sent to Z3 when the expression fragment was supported. |
| `heuristic` | The result is useful evidence but not a language-equivalence proof. |
| `abstaining` | The checker deliberately refused an unsupported fragment instead of inventing a result. |

PromptABI prefers an explicit abstention over an unsound proof. Unsupported
template syntax, recursive schemas outside the bound, unknown token counts, and
non-finite provider behavior should therefore become warnings or abstentions,
not silent success.

## Minimal executable example

```python
from promptabi.formal import DeterministicFiniteAutomaton

stop = DeterministicFiniteAutomaton.literal("</s>", alphabet=set("</s>abc"))
prefixes = DeterministicFiniteAutomaton.prefix_closed_literal("</s>", alphabet=set("</s>abc"))
witness = stop.intersect(prefixes).shortest_witness()
assert witness is not None
assert witness.text == "</s>"
```

This is the shape behind larger PromptABI findings: build finite objects from
real artifacts, run an exact or bounded product, and attach the witness to a
stable diagnostic.
