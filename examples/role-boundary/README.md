# Role-boundary non-forgeability example

This example checks a ChatML-style Hugging Face `tokenizer_config.json` with
the real `role-boundary-nonforgeability` CLI check.

```bash
promptabi verify --config examples/role-boundary/unsafe.promptabi.json
promptabi verify --config examples/role-boundary/safe.promptabi.json
```

The unsafe template directly inserts `message['role']` and
`message['content']`, so a user-controlled field can render strings such as
`assistant`, `<|im_start|>`, or `<|im_end|>` as structural control text. The
sanitized template JSON-encodes those fields before rendering, so the same
bytes remain data inside a quoted string rather than becoming a new role
boundary.

PromptABI is not claiming that a model will obey a forged prompt. It proves a
narrower structural property: whether attacker-controlled input can become
indistinguishable from template/provider control syntax before inference.
