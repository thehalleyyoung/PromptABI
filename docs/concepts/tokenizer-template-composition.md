# Tokenizer/template composition

PromptABI treats template rendering and tokenization as separate protocol
relations. The application thinks it is sending messages; the model runtime
receives bytes, strings, and token IDs. A safe interface must preserve the
intended boundary through the whole path:

```text
messages -> chat template -> rendered prompt -> tokenizer -> token stream
```

## Template side

The Hugging Face chat-template parser extracts the template source, declared
roles, message fields, tool fields, generation-prompt behavior, and unsupported
constructs from `tokenizer_config.json`. The bounded symbolic executor then
enumerates representative paths through supported Jinja constructs:

| Construct | Treatment |
| --- | --- |
| Message loops | Expanded up to declared bounds and recorded as loop iterations. |
| Role conditionals | Preserved as path conditions such as `messages[0].role == "user"`. |
| Variables | Rendered as placeholders like `{messages[0].content}` until a checker supplies values. |
| Filters/wrappers | Recognized when they provide escaping, JSON encoding, or delimiter-safe alphabets. |
| Unsupported syntax | Recorded as an abstention rather than treated as safe. |

The result is a bounded symbolic rendered prompt, not a guess about model
behavior.

## Tokenizer side

Tokenizer adapters expose encode, decode, token texts, special-token behavior,
normalization, added tokens, and round-trip metadata. Differential tests compare
the abstraction against real libraries when optional dependencies are installed:
Hugging Face `tokenizers`, `tiktoken`, and SentencePiece. For core CI and docs
examples, `ByteLevelTokenizer` gives a deterministic CPU-only baseline.

Tokenization matters because a byte string that appears harmless in source can
be normalized, decoded, or marked special in the runtime. PromptABI therefore
records:

| Property | Why it matters |
| --- | --- |
| Normalized text | A grammar witness can be changed outside the accepted language. |
| Decoded text | A constrained decoder may validate decoded text rather than source text. |
| Token IDs and texts | Stops, added tokens, and special tokens can collide at token boundaries. |
| Special-token status | Delimiters can be control tokens rather than ordinary content. |

## Composition principle

A role boundary is safe only if attacker-controlled fields remain data after
both render and tokenize steps. A grammar is usable only if at least one accepted
grammar string survives tokenizer encode-normalize-decode behavior. A stop
policy is safe only if a configured stop cannot be reached in a parser state
where it truncates a still-valid structured output.

The same idea appears across the codebase:

| Concern | Product checked |
| --- | --- |
| Role forgery | symbolic render path x control-marker language x tokenizer evidence |
| Grammar emptiness | compiled bounded grammar DFA x tokenizer encode/decode product |
| Stop overreachability | stop language x structured-output parser-prefix model |
| Provider migration | source request/response envelope x target provider envelope |

## Practical reading of a witness

A good PromptABI witness should answer four questions:

1. Which application-controlled field was chosen?
2. What exact rendered excerpt or structured output was produced?
3. What did the tokenizer or parser see?
4. Which boundary, stop, grammar state, or provider field changed meaning?

That is why diagnostics include source spans, artifact provenance, witness steps,
and proof modes instead of a single "unsafe prompt" label.
