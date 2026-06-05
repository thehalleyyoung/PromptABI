# Plugin author guide

PromptABI plugins extend the verifier without weakening its deterministic,
CPU-only, local-security model. A plugin should register capabilities through
`PluginRegistry`, declare the exact artifact surface it touches, and preserve
stable diagnostics, source spans, witnesses, and renderers.

## Contract for plugins

Plugins may provide artifact loaders, checks, provider adapters, grammar
backends, solver encodings, or diagnostic renderers. Each capability should
state its supported fragment, failure modes, privacy behavior, and guarantee
mode. If a plugin needs network access, credentials, GPU inference, or mutable
external state, it is not suitable for the default CI path.

## Minimum contribution bar

Every plugin contribution should include:

1. A sanitized fixture or in-memory artifact proving the capability works.
2. A deterministic test in `tests/test_contributor_infrastructure.py` or a
   focused plugin test that imports the public plugin API.
3. Compatibility notes for public APIs, deprecations, and the PromptABI version
   range supported by the plugin.
4. Privacy documentation explaining why prompts, schemas, constraints,
   witnesses, and credentials are not transmitted.

Run `promptabi contribute validate` before opening a pull request so the
contributor docs, issue templates, labels, and CI gate stay synchronized.
