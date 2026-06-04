# PromptABI

PromptABI is CI for the tokenizer, chat-template, tool-calling, structured-output,
and prompt-budget boundary of LLM applications. It verifies the discrete contract
around an LLM without loading model weights or running inference.

This repository skeleton already includes the public Python package, CLI
entrypoint, tests, examples, fixture corpus layout, benchmark layout, and docs
structure needed for the formal checkers in the roadmap.

## Quick shape

```bash
promptabi verify --config examples/minimal/promptabi.json
```

PromptABI treats LLM interface artifacts as composable protocol objects:
tokenizer metadata, chat templates, tool schemas, stop policies, grammar
fragments, provider request contracts, and framework truncation policies.

