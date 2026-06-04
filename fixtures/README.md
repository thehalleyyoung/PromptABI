# Fixture corpus

Fixtures are small, CPU-only, and safe to run in CI. Each corpus entry should
record upstream provenance, hashes, licenses, minimized repro inputs, expected
diagnostics, and notes about unsupported fragments.

The seed corpus contains minimized tokenizer_config-style fixtures for Llama,
Mistral, Qwen, Gemma, Phi, DeepSeek, Zephyr, ChatML, OpenAI-compatible, and
common fine-tune templates. They are representative CPU-only fixtures rather
than heavyweight model downloads, so tests can validate provenance, template
shape, sentinel coverage, and artifact loading deterministically.

The structured schema corpus contains labeled structured-output and
tool-definition reductions from open-source-agent patterns, anonymized production
shapes, and synthetic stress cases. Each entry carries provenance, labels,
expected parser-compatibility outcomes, hashes, and a runnable PromptABI config.
