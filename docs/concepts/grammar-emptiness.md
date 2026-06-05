# Grammar emptiness

Constrained decoding only helps if the grammar language and tokenizer behavior
overlap. PromptABI checks a bounded tokenizer x grammar product to answer a
practical CI question: can this tokenizer produce at least one string accepted
by this grammar under the backend assumptions we model?

The implementation is `promptabi.grammar_emptiness.analyze_tokenizer_grammar_emptiness`.

## Pipeline

1. Load a schema or grammar artifact.
2. Compile the supported fragment into a bounded DFA or abstain with a reason.
3. Enumerate accepting grammar witnesses up to `max_candidates` and `max_depth`.
4. Encode each witness with the selected tokenizer adapter.
5. Decode the token IDs back to text.
6. Accept the product only if the decoded text is still accepted by the grammar
   automaton.

This intentionally models real constrained-decoder seams where libraries may
tokenize grammar literals, validate decoded text, or normalize input differently
from the author-written schema.

## Outcomes

| Status | Meaning |
| --- | --- |
| `satisfiable` | A concrete grammar witness survived encode-normalize-decode and is attached to the report. |
| `empty` | The bounded grammar had candidates, but none survived tokenizer assumptions, or the grammar automaton had no accepting path. |
| `abstained` | The grammar dialect, schema fragment, file, or bound could not be compiled into the supported finite product. |

The report records checked candidates, grammar states, accepting states,
assumptions, issue codes, rejected attempts, and the witness token IDs/texts.

## Example failure mode

A schema may require the literal `"OK"`, while a tokenizer or wrapper normalizes
input to lowercase. The grammar accepts `"OK"`, but encode/decode returns
`"ok"`, which is outside the grammar language. PromptABI reports an empty
bounded product with the rejected attempt instead of letting CI assume the
decoder can produce a valid value.

## Relationship to parser compatibility

Grammar emptiness checks whether the decoder can produce a grammar-accepted
string. Parser compatibility separately asks whether the application parser
accepts the same bounded language as the grammar. The first is a tokenizer x
grammar product; the second is a grammar x parser comparison. Both are needed
because a grammar can be non-empty yet still disagree with the parser that
consumes the generated output.
