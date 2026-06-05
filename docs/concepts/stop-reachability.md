# Stop reachability

Stop sequences are part of the interface contract, not merely generation
parameters. If a configured stop can fire inside a valid structured output, the
application may receive malformed JSON, prematurely accepted XML-like tool
calls, truncated markdown fences, or provider envelopes whose parser state no
longer matches the decoder state.

PromptABI has two complementary stop analyses:

| Analysis | Implementation | Question |
| --- | --- | --- |
| Tokenizer stop analysis | `promptabi.stop_analysis` | Is the stop reachable, ambiguous, normalized differently, multi-token, or colliding with special/added tokens? |
| Stop overreachability | `promptabi.stop_overreachability` | Can the stop fire before a structured output region is complete? |

## Bounded overreachability model

`analyze_stop_overreachability` combines a stop policy with built-in and
artifact-derived structured-output regions. The built-in regions cover common
JSON, markdown code-fence, XML-like tool-call, OpenAI tool-envelope, and
Anthropic tool-envelope shapes. Artifact-derived regions come from supported
schemas and tool definitions, especially string fields where the stop text is
valid content.

Each `StructuredOutputRegion` contains:

| Field | Meaning |
| --- | --- |
| `kind` and `name` | The parser family and region label. |
| `path` | A schema/tool/provider path such as `$.arguments.query`. |
| `witness_text` | A valid output containing either structural stop syntax or a content marker. |
| `parser_state` | The state at which truncation is evaluated. |
| `structural_stops` | Stop strings that are structural in this region. |

When a stop is reachable, PromptABI emits the valid output, the valid prefix
through the stop, the truncated prefix actually delivered to the application,
the exact line/column firing point, parser state at truncation, and the
resulting malformed or prematurely accepted structure.

## Structural versus content findings

| Category | Example |
| --- | --- |
| Structural | `</tool_call>` is configured as a stop and also closes an XML-like tool envelope, so generation halts before required trailing structure. |
| Content | `"` or a user-defined sentinel appears inside a JSON string field permitted by the schema, so the parser receives a truncated string. |

Structural findings prove that a protocol delimiter conflicts with a stop.
Content findings prove that the schema/tool language admits a value containing
the stop text in a position where truncation changes the parser result.

## What this does not claim

The checker does not claim a model will choose the dangerous stop sequence. It
claims that if the model emits a valid output containing that sequence at the
shown position, the configured stop policy will cut the output at a boundary
that violates the downstream parser contract.
