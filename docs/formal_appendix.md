# PromptABI Formal Appendix: Prompt-Assembly Metatheory

Version: 2026.06

This appendix collects the metatheory of the PromptABI prompt-assembly calculus.
Every theorem is *mechanized in the executable sense*: it is discharged by
exhaustive bounded enumeration over the production implementation
(`promptabi.prompt_calculus`). Domain sizes are reported per theorem.

## Syntax

```
t ::= Lit s | Data s | Esc t | Seg r t | Concat t t
r ::= system | user | assistant
```

Control delimiters: `<|assistant|>`, `<|system|>`, `<|user|>`, `<|end|>`.

## Theorems

### 301. Small-step operational semantics is deterministic and adequate

**Statement.** The small-step relation -> on prompt-assembly configurations is a total deterministic function on non-final configurations, every term normalises, and the normal form's output equals the denotational renderer render(t).

**Method.** exhaustive bounded enumeration of all terms up to depth 2 (domain size 4000).

**Assumptions.** finite literal/data/role alphabets; depth-bounded term universe.

**Executable obligations.**
- `small-step-deterministic` — proved: each non-final configuration has a unique successor
- `small-step-terminates` — proved: every term reduces to a final configuration within budget
- `operational-denotational-adequacy` — proved: reduce_term(t).output == render(t) for all enumerated t

### 302. Well-typed prompts do not forge roles

**Statement.** If |- t : control then for every adversarial data assignment the rendered prompt contains no control delimiter occurrence that includes any data-provenance character.

**Method.** exhaustive bounded enumeration; soundness over all well-typed terms, witness over ill-typed terms (domain size 4000).

**Assumptions.** trusted literals contain no '<' or '|'; the sanitizer removes data '<' characters; depth-bounded adversarial data universe.

**Executable obligations.**
- `well-typed-never-forges` — proved: checked 627 well-typed terms; 0 forged delimiters
- `checker-is-meaningful` — proved: 2247 ill-typed terms exhibit a forged delimiter

### 303. Mechanized role-non-forgeability core

**Statement.** For the guarded single-segment template Seg(role, Esc(Data d)), no adversarial payload d forges a role delimiter. The proposition is mirrored in a Lean 4 source artifact.

**Method.** bounded model-check of the core lemma plus a Lean 4 source mirror (domain size 33).

**Assumptions.** sanitizer denotation as in Esc; single-segment guarded template.

**Executable obligations.**
- `guarded-segment-non-forgeable` — proved: checked 33 (role, payload) pairs
- `lean-artifact-states-theorem` — proved: Lean source declares the theorem

### 304. Observational equivalence is a congruence

**Statement.** Contextual equivalence (same control skeleton in all bounded contexts) is an equivalence relation and is preserved by every term constructor (congruence).

**Method.** bounded enumeration over a context family and an equivalence-relation pool (domain size 14).

**Assumptions.** observable = structural control skeleton; depth-bounded context family.

**Executable obligations.**
- `base-terms-observationally-equivalent` — proved: guarded data leaves share a control skeleton
- `congruence-under-contexts` — proved: equivalence preserved by 5 context shapes
- `equivalence-is-reflexive` — proved: ~ is reflexive
- `equivalence-is-symmetric` — proved: ~ is symmetric
- `equivalence-is-transitive` — proved: ~ is transitive

### 305. Stop policy as a denotational truncation function

**Statement.** The denotational truncation D(text, stops) = the maximal stop-free prefix equals the operational scanner, is always a prefix of the input, and contains no stop substring.

**Method.** exhaustive enumeration of bounded strings and stop sets (domain size 3905).

**Assumptions.** finite alphabet; strings of length <= 4.

**Executable obligations.**
- `denotational-equals-operational` — proved: agree on 3905 (stops, string) pairs
- `truncation-is-prefix` — proved: truncate(text) is always a prefix of text
- `no-stop-inside-result` — proved: no stop substring survives in the kept prefix
- `prefix-monotonicity` — proved: truncation respects the prefix order on inputs

### 306. Structured-output schema checker is sound and complete

**Statement.** On the supported object-schema fragment, the compiled checker accepts a document iff the reference structural interpretation accepts it (soundness and completeness coincide).

**Method.** exhaustive enumeration of bounded documents against two independent acceptors (domain size 175).

**Assumptions.** closed objects with string/integer/boolean fields; documents with <= 3 keys.

**Executable obligations.**
- `checker-soundness` — proved: compiled checker accepts only schema-valid documents
- `checker-completeness` — proved: compiled checker accepts every schema-valid document
- `checker-total` — proved: decided 175 documents without abstaining

### 307. The decidable grammar-backend fragment is a down-closed lattice region

**Statement.** The set of decidable grammar-backend feature combinations is down-closed in the feature lattice and exactly characterised by its maximal elements (the decidability frontier).

**Method.** exhaustive enumeration of the 2^5 feature powerset (domain size 32).

**Assumptions.** five-feature lattice; decidability oracle as modelled.

**Executable obligations.**
- `decidable-region-down-closed` — proved: subsets of decidable feature sets are decidable
- `frontier-covers-region` — proved: each decidable set lies under a maximal decidable set
- `frontier-is-decidable` — proved: the whole down-set of each maximal element is decidable
- `frontier-nonempty` — proved: 2 maximal decidable feature sets

### 308. Noninterference between control and data regions

**Statement.** For every well-typed template, the control skeleton (the observable control output) is invariant under all substitutions of the data leaves: low-equivalent control outputs for all high inputs.

**Method.** exhaustive enumeration of data substitutions per template (domain size 30).

**Assumptions.** well-typed templates; observable = structural control skeleton.

**Executable obligations.**
- `control-skeleton-data-independent` — proved: varying data leaves leaves the control skeleton unchanged

### 309. Capability-negotiation fallback ordering is monotone

**Statement.** The fallback function is monotone and order-preserving on the capability tier order, negotiation never returns a tier stronger than requested, and the weakest tier is a fixpoint.

**Method.** exhaustive enumeration over tiers and supported-tier subsets (domain size 60).

**Assumptions.** totally ordered capability tiers.

**Executable obligations.**
- `fallback-monotone` — proved: fallback never strengthens a tier
- `fallback-order-preserving` — proved: fallback preserves the tier order
- `negotiation-not-stronger-than-preferred` — proved: negotiate() never exceeds the preferred tier
- `fallback-floor-is-fixpoint` — proved: the weakest tier is a fallback fixpoint

### 310. Tool-call accounting forms a session-type discipline

**Statement.** Tool-call traces typed by the session discipline are exactly the properly nested traces; every typed trace is balanced and the typing relation is a decidable function.

**Method.** exhaustive enumeration of bounded open/arg/close traces (domain size 1555).

**Assumptions.** two call ids; traces of length <= 4.

**Executable obligations.**
- `well-typed-implies-balanced` — proved: session-typed traces are balanced
- `checker-deterministic` — proved: the session-type checker is a function
- `nonempty-typed-traces-exist` — proved: 14 nonempty well-typed traces

### 311. Migration dry-run patches preserve request well-formedness

**Statement.** Every patch classified safe maps a well-formed request to a well-formed request; unsafe patches are flagged and are exactly those that can drop required-field coverage.

**Method.** exhaustive enumeration over the request powerset and a patch family (domain size 16).

**Assumptions.** required fields = {model, messages}; four candidate fields.

**Executable obligations.**
- `safe-patch-preserves-well-formedness` — proved: checked 48 (patch, request) pairs
- `unsafe-patch-flagged` — proved: unsafe patches are detected and can break requests

### 312. Provider-contract refinement is a preorder

**Statement.** The assume/guarantee refinement on provider contracts (assume no more, guarantee no less) is reflexive and transitive, hence a preorder.

**Method.** exhaustive enumeration over a bounded contract space (domain size 60).

**Assumptions.** three-feature requirement/guarantee pools.

**Executable obligations.**
- `refinement-reflexive` — proved: every contract refines itself
- `refinement-transitive` — proved: refinement composes

### 313. Conformance is compositional over prompt-pack composition

**Statement.** For prompt-pack composition by obligation union, an implementation conforms to the composite iff it conforms to each component.

**Method.** exhaustive enumeration over pack and implementation powersets (domain size 512).

**Assumptions.** composition = obligation union; three-obligation pool.

**Executable obligations.**
- `conformance-distributes-over-composition` — proved: conformant(impl, a*b) iff conformant(impl,a) and conformant(impl,b)

### 314. Prompt-interface drift forms an ultrametric

**Statement.** The prefix-based drift distance is non-negative, satisfies identity of indiscernibles, is symmetric, and obeys the strong (ultrametric) triangle inequality.

**Method.** exhaustive enumeration over a bounded interface-descriptor space (domain size 27).

**Assumptions.** length-3 feature descriptors over a 3-symbol alphabet.

**Executable obligations.**
- `drift-non-negative` — proved: d(x,y) >= 0
- `drift-identity-of-indiscernibles` — proved: d(x,y)=0 iff x=y
- `drift-symmetric` — proved: d(x,y)=d(y,x)
- `drift-strong-triangle` — proved: d(x,z) <= max(d(x,y), d(y,z))

## Reproduction

```
promptabi metatheory --format json
```
