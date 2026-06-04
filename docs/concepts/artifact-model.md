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

The loader layer now materializes those references without network access. It
streams local files for sha256 validation, builds deterministic manifests for
tokenizer directories, records metadata-only Hugging Face repository refs with
revision strength, reads GGUF header stubs, validates provider config snapshots,
and summarizes zip/tar fixture bundles without extracting them. Missing files
remain hard errors; unpinned but readable artifacts are warnings so existing
workflows can adopt reproducible pins incrementally.
