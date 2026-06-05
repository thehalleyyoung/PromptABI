# Provider migration boundary

The buggy app assumes an OpenAI-style response envelope after switching the
target provider to Anthropic. PromptABI compares recorded source and target
fixtures and catches incompatible request fields, tool argument encoding, stop
behavior, response shape, and structured-output mode.

```bash
promptabi verify --config examples/end-to-end/provider-migration/buggy.promptabi.json
promptabi verify --config examples/end-to-end/provider-migration/fixed.promptabi.json
```

