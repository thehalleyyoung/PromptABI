# PromptABI public API

PromptABI exposes a generated stability manifest for downstream CI, editor, and
plugin authors. Regenerate this page with:

```bash
promptabi api-docs --output docs/public-api.md
```

The contract is intentionally conservative. The stable surface covers embedding
workflows, diagnostics, verification sessions, artifact loading, and plugin
registration. Other importable names remain provisional until they are promoted
into this compatibility set.

## Stability policy

- Stable symbols keep import paths, callability, dataclass fields, enum values,
  and documented constructor arguments compatible for the 1.x line.
- Stable callable signatures may add keyword-only parameters with defaults, but
  existing required parameters are not removed or renamed without a deprecation
  cycle.
- Provisional symbols remain importable for experimentation but may change before
  they are promoted into the stable plugin and embedding contract.
- Deprecated symbols emit `DeprecationWarning` through `deprecated_api` and carry
  replacement/removal metadata in the generated manifest before removal.

## Stable embedding and plugin surface

| Symbol | Purpose |
| --- | --- |
| `create_session`, `load_artifacts`, `collect_diagnostics`, `run_verification`, `render_result` | Embed PromptABI in Python tools without shelling out. |
| `VerificationConfig`, `VerificationSession`, `VerificationResult`, `CheckContext`, `CheckCallable` | Construct and run deterministic verification workflows. |
| `Diagnostic`, `DiagnosticSeverity`, `CheckMode`, `ArtifactRef`, `SourceSpan`, `WitnessStep`, `WitnessTrace` | Emit stable, renderable diagnostics with source spans and witnesses. |
| `ArtifactBundle`, `ArtifactKind`, `ArtifactLocation`, `ArtifactLoader`, `LoadedArtifact`, `SchemaArtifact`, `ProviderConfigArtifact` | Define and load the artifact types most plugin authors need. |
| `PluginRegistry`, `PluginCapability`, `PluginCapabilityKind`, `PluginError`, `PluginRegistrar`, `ArtifactLoadHook`, `DiagnosticRenderer` | Register typed loaders, checks, renderers, and capability metadata. |
| `load_plugin_modules`, `load_entry_point_plugins` | Load local module plugins or installed `promptabi.plugins` entry points. |
| `build_public_api_manifest`, `compare_public_api_manifests`, `public_api_reference` | Generate docs and enforce downstream compatibility baselines. |

## Compatibility workflow

Plugin authors can pin the JSON manifest in their own tests:

```bash
promptabi api-docs --format json > promptabi-public-api.json
```

On upgrade, load the baseline with `public_api_manifest_from_mapping()`, compare
it to `build_public_api_manifest()`, and fail if
`compare_public_api_manifests()` returns issues. PromptABI itself tests this path
with a downstream plugin that imports only stable symbols and registers a real
check against `PluginRegistry`.

## Deprecations

New deprecations should use:

```python
from promptabi import deprecated_api

@deprecated_api(since="1.1", replacement="promptabi.new_api", remove_in="2.0")
def old_api(...):
    ...
```

The wrapper preserves the callable metadata, emits `DeprecationWarning`, and
places machine-readable replacement/removal information in the generated API
manifest.
