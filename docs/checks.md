# Check families

The roadmap focuses on three initial families.

Every diagnostic carries an explicit verification mode so users can tell whether
PromptABI proved a property, searched a bounded fragment, used Z3-backed SMT,
reported heuristic evidence, or abstained outside the supported model.

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

## Must-survive budget verification

Do required prompt segments remain present after the actual framework truncation
policy is applied?
