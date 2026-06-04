# Artifact model

PromptABI separates artifact identity from artifact content. A diagnostic can
refer to a tokenizer, chat template, special-token map, stop policy, schema,
grammar, tool definition, prompt segment, provider config, or framework
truncation config without requiring the renderer to know how that artifact was
loaded.

The public artifact model stores kind, name, local path or URI, provenance,
version/hash metadata, and kind-specific fields such as stop strings, tool names,
prompt segment survival requirements, and truncation strategy. `ArtifactBundle`
keeps those inputs sorted and serializable so CLI text, JSON, SARIF, tests, and
embedding APIs share one deterministic contract.
