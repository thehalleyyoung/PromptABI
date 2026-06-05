# PromptABI

PromptABI is CI for the tokenizer, chat-template, tool-calling, structured-output,
and prompt-budget boundary of LLM applications. It verifies the discrete contract
around an LLM without loading model weights or running inference.

The repository includes the public Python package, CLI entrypoint, GitHub Action,
integration guides, tests, examples, fixture corpus layout, benchmark layout, and
docs structure needed for formal PromptABI checks to run in real CI, including a
localization-ready diagnostic catalog for future translated interfaces.

Its security model is intentionally local and structural: PromptABI does not
claim semantic model safety, and it does not need live provider calls to prove
role-boundary, stop-policy, grammar, tool-call, provider-migration, or
must-survive prompt-budget interface failures.

## Quick shape

```bash
promptabi verify --config examples/minimal/promptabi.json
```

PromptABI treats LLM interface artifacts as composable protocol objects:
tokenizer metadata, chat templates, tool schemas, stop policies, grammar
fragments, provider request contracts, and framework truncation policies.

For isolated deployments, the air-gapped installation guide shows how to vendor
dependencies, Z3, offline corpora, provider fixture mirrors, prompt-pack mirrors,
and reproducibility checks without weakening the verifier's local guarantees.
