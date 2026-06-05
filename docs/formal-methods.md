# Formal methods decision guide

PromptABI uses formal methods only where the LLM interface is finite,
observable, and replayable: rendered strings, tokenizer behavior, parser states,
tool envelopes, stop policies, prompt budgets, training masks, and bounded
symbolic fields. It does not model neural-network semantics. When a property
cannot be represented honestly inside those finite objects, the checker must
emit an abstention or a heuristic diagnostic rather than a proof-shaped result.

This page is the high-level routing guide for the implementation in
`src/promptabi/formal.py`, the proof catalog in
`src/promptabi/proof_sketches.py`, and the audit registry in
`src/promptabi/soundness_audits.py`.

## Choosing automata, Z3, composition, or abstention

| Use this method | When the property looks like | Primary implementation | Examples |
| --- | --- | --- | --- |
| **Finite automata** | Language reachability, emptiness, shortest witnesses, prefix/suffix collisions, or parser states over a finite alphabet. | `DeterministicFiniteAutomaton`, `AutomatonWitness`, lazy product search in `promptabi.formal`. | Stop reachability, grammar emptiness, role delimiter exclusion, parser-prefix witnesses. |
| **Finite-state transducers** | A bounded relation between inputs and outputs must be preserved or projected. | `FiniteStateTransducer`, `TransducerWitness`, composition/projection utilities in `promptabi.formal`. | Template rendering, escaping, tokenizer-like relations, provider serialization sketches. |
| **Z3-backed finite SMT** | The property is a finite Boolean/enum/integer/membership/length/bounded-string constraint. | `FiniteContractProblem`, `SolverResult`, `SolverReplayFile`, and the static-contract layer. | Prompt-segment survival, role-region non-forgeability, stop/control-token exclusion, tool-schema preconditions, training target alignment. |
| **Composed products** | Safety depends on two or more artifacts agreeing at the boundary. | Artifact-derived automata/transducer products plus `VerificationSession` diagnostics. | Template -> tokenizer -> grammar, provider -> tool parser, training manifest -> chat template -> loss mask. |
| **Abstention** | The artifact exceeds the supported finite fragment, a dependency is missing, or the approximation would be unsound. | Check modes `abstaining`/`heuristic`, `UNKNOWN` solver status, explicit diagnostic evidence. | Unbounded Jinja, recursive schemas beyond limits, unknown tokenizer normalization, opaque provider behavior. |

The design rule is simple: **prove exact finite facts, search bounded finite
facts, label heuristics as heuristics, and abstain before pretending a proof
exists**.

## Automata path

Automata are the default when the question is "does a string or token path
exist?" A checker builds a finite language for each side of the interface, then
uses intersection, difference, shortest-witness search, or lazy products to find
a concrete path.

Typical automata-backed checks:

1. Stop-overreachability builds valid structured-output regions and asks whether
   a configured stop can fire inside a still-valid prefix.
2. Grammar emptiness compiles supported JSON Schema/grammar fragments and asks
   whether tokenizer-decoded candidates can reach an accepting grammar state.
3. Role-boundary non-forgeability treats delimiters and special tokens as
   forbidden languages inside user/tool/retrieval-controlled regions.

The minimal shape is executable:

```python
from promptabi.formal import DeterministicFiniteAutomaton

stop = DeterministicFiniteAutomaton.literal("</s>", alphabet=set("</s>abc"))
prefixes = DeterministicFiniteAutomaton.prefix_closed_literal("</s>", alphabet=set("</s>abc"))
witness = stop.intersect(prefixes).shortest_witness()
assert witness is not None
assert witness.text == "</s>"
```

If the automaton cannot represent the artifact exactly, the checker must record
the bound or abstain. It should not silently widen an unsafe approximation into a
`sound` or `complete` diagnostic.

## Z3-backed finite SMT path

Z3 is used when the interface fact is naturally a finite symbolic contract:
integer budget arithmetic, enum compatibility, membership, string containment
over bounded alphabets, or Boolean implications. The public contract object is
`FiniteContractProblem`; production checks derive those problems from loaded
artifacts and report `StaticContractFinding` objects.

```python
from promptabi.formal import BoundedStringDomain, Contains, FiniteContractProblem
from promptabi.formal import NamedConstraint, SolverStatus, Value, Var

problem = FiniteContractProblem(
    name="delimiter-forgery",
    variables=(BoundedStringDomain("content", tuple("<a>bc"), min_length=0, max_length=3),),
    constraints=(NamedConstraint("contains-marker", Contains(Var("content"), Value("<a>"))),),
)
assert problem.solve(prefer_z3=False).status is SolverStatus.SAT
```

Solver results are not just pass/fail. `SAT` means PromptABI has a concrete
counterexample assignment; `UNSAT` means no assignment reaches the unsafe
condition inside the finite model; `UNKNOWN` means timeout, unsupported syntax,
budget exhaustion, or an intentionally refused fragment. `SolverReplayFile`
keeps reduced obligations replayable without private prompts:

```bash
promptabi solver replay fixtures/solver_replays/role-region-forgery.solver-replay.json
```

## Composition path

Most PromptABI value comes from composition: individually reasonable artifacts
can form an unsafe boundary when combined. A composed check should name the
interface edges it relies on, preserve source spans through the product, and
attach a witness that can be replayed through the same abstraction.

Representative composed products:

| Product | Failure caught |
| --- | --- |
| chat template -> tokenizer -> role regions | user content tokenizes as assistant/system/tool control syntax |
| tokenizer -> grammar -> application parser | constrained decoding accepts strings the parser rejects, or no valid string exists |
| provider envelope -> tool schema -> stop policy | streamed tool-call chunks, IDs, names, escaping, or stops disagree |
| prompt segments -> framework truncation -> context budget | a required system/RAG/training region is dropped before deployment |
| RAG chunks -> truncation policy -> tool schema | retrieved context omits, overflows, or truncates fields required by the tool call that consumes it |
| training manifest -> chat template -> packing/loss mask | supervised labels cover user/tool/retrieval text or invalid assistant regions |

The dependency graph command exposes these relationships for real configs:

```bash
promptabi graph --config examples/rag-chunking/promptabi.json --all-checks --format mermaid
```

## Abstention path

Abstention is a correctness feature. A checker should abstain when:

1. The artifact uses syntax outside the supported fragment, such as unbounded
   template control flow or a recursive schema without an accepted depth bound.
2. A required transition relation is unavailable, such as an opaque provider
   parser or tokenizer normalization the local abstraction cannot replay.
3. A finite product would exceed configured search, solver, or privacy budgets.
4. A claim would depend on model intent, semantic obedience, or live provider
   behavior rather than structural interface states.

Abstentions must include enough evidence to be actionable: the unsupported
fragment, the affected artifact span when available, the exact bound or missing
dependency, and the safest next step. The compatibility matrix and soundness
audit make those limits visible:

```bash
promptabi matrix --format text
promptabi soundness-audit --format markdown
```

## Evidence and review loop

Every formal claim should have three reviewable surfaces:

1. **Diagnostics** that identify the rule ID, guarantee mode, source span,
   witness, solver outcome, and suggested fix.
2. **Executable proof sketches** that replay central obligations through real
   checker reports:

   ```bash
   promptabi proofs --format json
   ```

3. **Soundness audits** that document assumptions, supported fragments, proof
   obligations, differential evidence, and blind spots:

   ```bash
   promptabi soundness-audit --rule static-contracts --format json
   ```

This loop is what keeps the project honest: README claims, paper claims, docs,
tests, fixtures, solver replays, and CLI behavior all point back to the same
finite artifacts instead of relying on informal prompt-security language.
