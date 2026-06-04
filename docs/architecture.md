# Architecture

PromptABI is organized around four stable layers.

1. **Artifacts** describe tokenizers, chat templates, special-token maps, stop
   policies, schemas, grammars, tool definitions, prompt segments, provider
   configs, and framework truncation configs with stable provenance.
2. **Sessions** collect artifacts into a reproducible verification run.
3. **Diagnostics** provide stable rule IDs, severities, source spans,
   provenance, witness steps, suggestions, and fingerprints.
4. **Renderers** produce CLI text, JSON, and SARIF now while leaving room for
   HTML and editor protocols.

The package uses a `src/` layout and ships `py.typed` so downstream tools can
depend on the public API without guessing at types.
