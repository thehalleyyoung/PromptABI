# Z3/SMT-backed finite contracts

The static-contract layer derives finite symbolic obligations from already
loaded PromptABI artifacts. It is implemented in `promptabi.static_contracts`
and lowered through the finite solver in `promptabi.formal`. Z3 is used when the
expression fragment is supported; otherwise the same contract can fall back to
deterministic finite enumeration or explicitly abstain.

## Supported domains

| Domain | Examples |
| --- | --- |
| Booleans | `escaped`, `has_required_tool`, `preserve_system` |
| Enums | role names, stop strings, target roles, provider names |
| Integer ranges | context limits, token counts, length bounds |
| Bounded strings | small delimiter and content alphabets used for counterexamples |
| Membership/length | stop in special-token set, required parameter in provided argument set |

The supported expression layer includes equality, inequality, conjunction,
disjunction, implication, negation, membership, substring containment, length,
and integer comparison. Unsupported custom expressions are not approximated as
safe; the solver returns `UNKNOWN` with an abstention reason.

## Current obligations

`analyze_static_contracts` derives the following real obligations when matching
artifacts are present:

| Obligation | Source artifacts | Unsafe when |
| --- | --- | --- |
| Prompt-segment survival | prompt segments + truncation config | required prompt tokens exceed modeled input budget |
| Role-region non-forgeability | prompt segments + chat template + special tokens/stops/tokenizer metadata | controlled content contains a boundary marker for another structural role |
| Stop/control-token exclusion | stop policies + special-token maps/tokenizer metadata | a stop sequence is also a control or special token |
| Tool/provider compatibility | tool definitions + provider configs | required provider capability is absent or incompatible |
| Tool-schema preconditions | tool schemas/provider fixtures | required parameters cannot be satisfied by the declared calling contract |
| Training target alignment | training manifest + chat template | a supervised target role is not renderable by the template |
| Training supervised-span alignment | training manifest + chat template | an observed supervised span is outside its rendered role region, token bounds, preserved packing boundary, or loss mask |
| Training source-leakage exclusion | training manifest | declared user/tool/retrieval/preference source ranges overlap a supervised loss-masked target span after a dataset transform |
| Training preference-pair contract | training manifest | chosen and rejected branches diverge in prompt prefix, role layout, tokenizer pin, mask policy, truncation, response start, or preserved packing boundary before the compared response |

Each obligation produces a `StaticContractFinding` with severity, evidence,
affected artifacts, and the underlying `SolverResult` when one exists.

## Reading solver outcomes and budgets

| Solver status | Diagnostic interpretation |
| --- | --- |
| `SAT` | A concrete counterexample assignment exists, so the contract violation is reachable in the finite model. |
| `UNSAT` | No assignment satisfies the unsafe condition; the unsat core names the constraints responsible for safety. |
| `UNKNOWN` | The fragment, solver, timeout, or assignment limit prevented a proof. |

Every `SolverResult` also carries a solver-budget classification. `proved` means
the backend established a SAT or UNSAT result inside the declared finite/SMT
budget. `bounded` means deterministic enumeration reached `max_assignments`
before exhausting the domain. `timed-out` means the wall-clock solver budget
expired. `abstained` means the formula exceeded the supported encoding, for
example because Z3 rejected an unsupported expression fragment. `approximated`
is reserved for checks that deliberately solve an over- or under-approximation;
the current finite static-contract obligations prefer a proof, bounded unknown,
timeout, or abstention instead of silently approximating.

This is why a budget overflow diagnostic can include
`required_prompt_tokens = 80` and `input_budget_tokens = 56`, while a safe
sanitized role-region proof can include an unsat core such as
`controlled-region-contains-boundary-marker`. CLI JSON diagnostics include
`solver_status`, `solver_backend`, `solver_conclusion`,
`solver_budget_outcome`, `checked_assignments`, and, when applicable, a
`solver_budget_reason`; text/HTML witnesses include matching
`classify solver budget` and reason steps.

## Minimal executable example

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

The production checks use richer artifact-derived variables, but the proof
shape is the same: define an unsafe finite condition, solve it, and report either
a counterexample, an unsat proof, or an abstention.

## Human-authored contract language

Static contracts can be stored as JSON or as the line-oriented `.pabi` DSL. The
DSL is deliberately small: it is easy to diff, canonicalize, and lower into the
same `StaticContractArtifact` model used by verification.

```text
contract promptabi.contract/v1

rule app-boundaries type llm-app severity error applies_to chat-template,tool-definition:
  description "Production chat app boundary contract"
  allowed_roles assistant,system,tool,user
  required_regions assistant,system
  forbid_delimiters "<|im_start|>","<tool_call>","</tool_call>"
  schema tool_call requires arguments,name
  stop json-response stops "</json>","```" forbid_inside json-string
  invariant budget-survival: required_prompt_tokens <= input_budget_tokens
```

Use the formatter to validate reviewable contracts or emit the exact JSON
artifact shape consumed by loaders:

```bash
promptabi contract format examples/static-contract-language/app.pabi --check
promptabi contract format examples/static-contract-language/app.pabi --format json
promptabi verify --config examples/static-contract-language/promptabi.json --fail-on never
```

The examples directory includes contracts for an LLM app, a fine-tuning
manifest, a RAG pipeline, and an evaluation harness. Their rule names, source
lines, severity, schema obligations, stop obligations, forbidden delimiters, and
invariants are preserved when loading `.pabi` files, so diagnostics can still
point back to human-readable rules.
