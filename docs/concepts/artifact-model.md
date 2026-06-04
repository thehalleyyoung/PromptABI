# Artifact model

PromptABI separates artifact identity from artifact content. A diagnostic can
refer to a tokenizer config, schema, tool definition, or prompt segment without
requiring the renderer to know how that artifact was loaded.

The initial `ArtifactRef`, `SourceSpan`, `WitnessTrace`, and `Diagnostic` types
are deliberately small and deterministic. They are the compatibility surface for
future loaders, checkers, and renderers.

