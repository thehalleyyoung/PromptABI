# Must-survive budgets

PromptABI treats context-window management as a proof obligation. If an
application declares that a system policy, safety instruction, citation, tool
definition, or training-alignment segment must survive packing, the configured
framework truncation policy must keep it under the modeled token budget.

The implementation lives in `promptabi.budgets` and is exercised by
`examples/token-budget/promptabi.json`.

## Budget arithmetic

`TokenBudgetReservation` separates the model window from reserved capacity:

```text
input_budget_tokens =
  max_context_tokens
  - reserve_output_tokens
  - reserved_tool_tokens
  - generation_prompt_tokens
  - special_token_overhead
```

Segments then contribute declared or tokenizer-derived token counts plus
overhead, metadata, and template overhead. Unknown counts are surfaced rather
than treated as zero.

## Truncation policies

`TruncationPolicy` normalizes framework behavior into a bounded prompt-packing
simulation:

| Policy concept | Examples |
| --- | --- |
| `left` | vLLM-style left truncation over declaration order. |
| `oldest-message` | LangChain-style memory that may preserve system text and drop older turns. |
| `priority` | Custom RAG policies that drop retrieval chunks before required prompt regions. |
| `preserve_system` / `preserve_tools` | Framework guarantees that protect specific roles. |
| `drop_roles` | Roles eligible for preferential truncation, such as retrieval context. |

The analyzer records kept segments, dropped segments, overflow tokens, unknown
segments, and a visualization row for each prompt region.

## Must-survive proof

`MustSurviveProof` has three important states:

| Status | Meaning |
| --- | --- |
| `proven` | Every required segment remains after the modeled truncation policy. |
| `violated` | At least one required segment is dropped, and a minimized counterexample is attached. |
| `unknown` | Missing token counts or unsupported policy behavior prevents a proof. |

A minimized counterexample is the smallest segment set needed to demonstrate the
drop under the real policy. This is more useful than reporting only aggregate
overflow because it names the exact system, user, retrieval, citation, or tool
segment that lost its must-survive guarantee.

## RAG-specific fields

Prompt segments can carry retrieval metadata: chunk IDs, document IDs, tokenizer
identity, source/chunk boundaries, expected and actual overlap, citations,
metadata token cost, template overhead, and retrieval payload limits. The same
budget proof therefore catches both generic context overflow and RAG-specific
failures such as citation loss, metadata inflation, and tokenizer mismatch at
chunk boundaries.
