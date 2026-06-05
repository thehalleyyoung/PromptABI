# Check families

The roadmap focuses on three initial families.

Every diagnostic carries an explicit verification mode so users can tell whether
PromptABI proved a property, searched a bounded fragment, used Z3-backed SMT,
reported heuristic evidence, or abstained outside the supported model.
The same diagnostics render to text, JSON, SARIF, GitHub annotations, and the
bundled GitHub Action summary, so CI and local debugging share one source of
truth. Diagnostic messages also expose stable localization keys and optional
placeholder metadata; see [Diagnostic localization](localization.md) for the
catalog format and `promptabi diagnostics catalog` workflow.

`promptabi bundle create` runs the same verifier and emits a signed audit bundle
containing config hashes, deterministic lockfile state, diagnostics, witnesses,
artifact excerpts, solver metadata, and a reproducibility hash; `promptabi bundle
verify` checks the HMAC signature without rerunning private artifacts.

| Mode | Meaning |
| --- | --- |
| `sound` | No violation is reported unless one exists under the stated abstraction. |
| `complete` | Every violation inside the supported fragment is found. |
| `bounded` | The result is exact only within declared finite limits. |
| `z3-backed-smt` | A finite symbolic contract is lowered to Z3 when available. |
| `heuristic` | The result is useful evidence, not a formal proof. |
| `abstaining` | The checker explicitly declines unsupported cases instead of guessing. |

## Role-boundary non-forgeability

Can attacker-controlled fields render as system, assistant, tool, or provider
control structure after chat-template expansion?

This check is intentionally structural: it proves that a rendered prompt can
contain forged control syntax, not that a model will obey that syntax. See
[Role-boundary non-forgeability](concepts/role-boundary-nonforgeability.md)
for the exact boundary, witness shape, sanitizer model, and runnable examples.

## Stop and grammar reachability

Can a stop sequence fire inside a valid structured output? Is a requested stop
sequence unreachable under the tokenizer and grammar?

See [Stop reachability](concepts/stop-reachability.md) and
[Grammar emptiness](concepts/grammar-emptiness.md) for the concrete finite
products, witness formats, and abstention boundaries.

`fixtures/real_world_bugs/` now includes public GitHub bug patterns reduced to
synthetic offline fixtures. The corresponding tests prove the current
role-boundary and stop-overreachability checkers catch Phi-style role delimiter
forgery, XML-like tool-call delimiter truncation, quote-sentinel truncation, and
tool-call parser-boundary stop failures.

## Must-survive budget verification

Do required prompt segments remain present after the actual framework truncation
policy is applied?

See [Must-survive budgets](concepts/must-survive-budgets.md) for the normalized
budget arithmetic, truncation policies, proof states, and RAG-specific fields.

`promptabi verify --config examples/token-budget/promptabi.json` now includes a
token-budget visualization in the `token-budget-model` diagnostic: each row
shows the prompt segment, token span, count source, kept/dropped status, and
must-survive guarantee. The same structured `token_budget_visualization` payload
is available in JSON output and SARIF properties.

For the shared machinery behind these families, read
[Formalism](concepts/formalism.md),
[Proof sketches](concepts/proof-sketches.md),
[Tokenizer/template composition](concepts/tokenizer-template-composition.md),
and [Z3/SMT-backed finite contracts](concepts/static-contracts.md).
