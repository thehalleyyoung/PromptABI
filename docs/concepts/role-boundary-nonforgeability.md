# Role-boundary non-forgeability

Role-boundary non-forgeability is a structural property of the rendered prompt,
not a claim about model behavior. PromptABI checks whether attacker-controlled
fields can become indistinguishable from chat-template or provider control text
before inference begins.

For a supported bounded Hugging Face Jinja template, PromptABI:

1. Symbolically renders representative message/tool paths.
2. Marks structural regions such as system, user, assistant, tool, developer,
   and function turns.
3. Collects control markers: role headers, assistant prefixes, tool-call
   sentinels, special tokens, and template delimiters.
4. Tests user-controlled role/content fields against those markers unless the
   field is passed through a recognized sanitizer such as JSON encoding,
   escaping, or an alphabet-restricting wrapper.
5. Emits a minimized witness with malicious input, rendered excerpt, byte-level
   token evidence, and the exact forged boundary.

## What the check proves

If PromptABI reports a role-boundary finding, the configured template permits a
specific input field to render as structural control syntax. For example, raw
ChatML-style content can contain `<|im_start|>` or `<|im_end|>`, and a dynamic
role field can contain `assistant`.

That is enough to break the interface contract between the application and the
model runtime: text that the application thought was data can be represented as
control structure in the final prompt.

## What it does not prove

PromptABI does not prove that a model will follow the forged instruction, leak a
secret, call a tool, or ignore the real system prompt. Those are semantic and
behavioral claims about a trained model. The checker deliberately stops at the
discrete interface boundary:

```text
messages -> chat template -> rendered prompt -> tokenizer -> token stream
```

This narrower scope is why the check is CPU-only, deterministic, and suitable
for CI.

## Runnable example

The repository includes unsafe and sanitized ChatML-style templates:

```bash
promptabi verify --config examples/role-boundary/unsafe.promptabi.json
promptabi verify --config examples/role-boundary/safe.promptabi.json
```

The unsafe config fails with concrete witnesses for `assistant`,
`<|im_start|>`, and `<|im_end|>`. The sanitized config JSON-encodes dynamic
role and content fields, so delimiter bytes remain data inside a quoted JSON
string rather than becoming a new role boundary.

## Model details

The implementation lives in `promptabi.role_boundaries`. It first builds a
`RoleBoundaryModel` from bounded symbolic template paths. Each path records:

| Field | Meaning |
| --- | --- |
| `conditions` | Role or branch predicates required for the path. |
| `loop_iterations` | Bounded loop choices used for messages/tools. |
| `rendered_pattern` | The rendered prompt with placeholders for controlled fields. |
| `regions` | Structural role regions with offsets, message indexes, and control text. |
| `abstentions` | Unsupported template features or unmodeled constructs. |

A `RoleBoundaryRegion` is the unit checked for non-forgeability. It identifies
the structural role (`system`, `user`, `assistant`, `tool`, `developer`, or
`function`), the source of that role, the symbolic character offsets of the
region, the controlled expressions inside it, and any recognized sanitizers.

## Sanitizer boundary

PromptABI recognizes sanitizers only when they change the structural language in
a way the checker can reason about:

| Sanitizer class | Accepted intuition |
| --- | --- |
| JSON encoding | Control bytes appear inside a quoted JSON string with escaping. |
| Escaping filters | Delimiters are escaped before they reach the rendered prompt. |
| Delimiter-safe wrappers | The wrapper restricts the emitted alphabet so a marker cannot appear raw. |

If the checker cannot prove that a transformation prevents raw marker emission,
it does not silently bless the field. Depending on the surrounding evidence, it
either reports a bounded finding or records an abstention.

## Witness shape

A role-forgery finding includes the attacked input expression, the structural
role it came from, the malicious input, rendered excerpt, tokenized
representation, marker offsets, and a `forged_boundary` description. The witness
is deliberately operational: a maintainer can paste the malicious field into the
same template and observe the control marker in the final prompt.
