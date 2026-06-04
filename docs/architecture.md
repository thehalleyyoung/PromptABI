# Architecture

PromptABI is organized around four stable layers.

1. **Artifacts** describe tokenizer files, chat templates, schemas, tool
   definitions, stop policies, provider fixtures, and budget policies.
2. **Sessions** collect artifacts into a reproducible verification run.
3. **Diagnostics** provide stable rule IDs, severities, source spans,
   provenance, witnesses, and suggestions.
4. **Renderers** produce CLI text now and leave room for JSON, SARIF, HTML, and
   editor protocols.

The package uses a `src/` layout and ships `py.typed` so downstream tools can
depend on the public API without guessing at types.

