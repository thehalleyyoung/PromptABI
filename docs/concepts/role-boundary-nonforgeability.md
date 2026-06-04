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
