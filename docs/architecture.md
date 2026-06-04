# Architecture

PromptABI is organized around five stable layers.

1. **Artifacts** describe tokenizers, chat templates, special-token maps, stop
   policies, schemas, grammars, tool definitions, prompt segments, provider
   configs, and framework truncation configs with stable provenance.
2. **Sessions** load artifacts, register built-in or embedded checks, and collect
   diagnostics into a reproducible verification run.
3. **Diagnostics** provide stable rule IDs, severities, source spans,
   provenance, witness steps, suggestions, formal check modes, and fingerprints.
4. **Renderers** produce CLI text, JSON, and SARIF now while leaving room for
   HTML and editor protocols.
5. **Plugins** register artifact loaders, checks, provider adapters, grammar
   backends, template dialects, truncation policies, solver encodings, and
   diagnostic renderers through one typed registry.

The package uses a `src/` layout and ships `py.typed` so downstream tools can
depend on the public API without guessing at types.

The embedding surface mirrors the CLI: `create_session`, `load_artifacts`,
`collect_diagnostics`, `run_verification`, and `render_result` accept typed
configs or config paths. `PluginRegistry` lets extensions add loader hooks,
dependency-aware checks with check-mode metadata, plugin renderers, and capability
descriptors. The CLI can import `--plugin MODULE[:OBJECT]` for verification-time
checks and renderers; installed packages can expose the same registry contract
through the `promptabi.plugins` entry point group.

Check modes are part of the architecture, not renderer decoration. A diagnostic
can claim `sound`, `complete`, `bounded`, `z3-backed-smt`, `heuristic`, or
`abstaining`, making the CLI and machine-readable outputs explicit about whether
a result is a proof, bounded proof, solver-backed finite contract, useful signal,
or principled refusal to decide.
