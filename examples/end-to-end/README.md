# End-to-end PromptABI examples

These mini apps model production-shaped LLM boundaries with a deliberately
buggy contract and a fixed contract. Each pair is runnable with `promptabi
verify` and is covered by `tests/test_end_to_end_examples.py`.

| Example | Bug PromptABI catches | Fixed contract demonstrates |
| --- | --- | --- |
| `tool-calling` | Provider, parser, and tool schema disagree on tool names, argument encoding, IDs, parallel calls, and streaming chunks. | One canonical tool-call envelope from request through parser handoff. |
| `structured-output` | A JSON-only application parser accepts outputs broader than the constrained schema. | The runtime parser validates the same JSON Schema used for constrained decoding. |
| `rag-truncation` | Citation-required chunks can lose citations or exceed retrieval payload budgets. | Retrieval chunks carry citations, use the serving tokenizer, and fit the packed prompt. |
| `provider-migration` | An OpenAI-style stack is migrated to a provider with incompatible tool, stop, response, and structured-output contracts. | The target fixture preserves the source provider contract. |
| `training-alignment` | Supervised training targets include a role absent from the serving chat template. | Training targets align with the serving template's role universe. |

