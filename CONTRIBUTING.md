# Contributing to PromptABI

PromptABI is a verifier, not a prompt-style linter. Contributions should keep
checks deterministic, CPU-only, reproducible, and explicit about whether a result
is sound, bounded, heuristic, or an abstention.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_public_api.py tests/test_config.py tests/test_cli.py
promptabi contribute workflows --format text
promptabi contribute validate
```

## Expectations

- Add minimized fixtures for new artifact behavior.
- Keep CLI output deterministic and suitable for snapshots.
- Include typed public APIs for integration-facing features.
- Do not require GPU inference or network access for tests.
- Prefer precise diagnostics with provenance, source spans, witnesses, and
  actionable suggestions.

## Corpus contributions

Fixture packs should include provenance, license notes, revision pins, expected
diagnostics, and a statement that no secrets or private prompts are included.
Dedicated workflows also cover sanitized bug fixtures, minimized witnesses,
prompt-pack metadata, provider fixtures, and training-manifest adapters.

See the focused contributor guides for larger changes:

- [`docs/contributing/community-workflows.md`](docs/contributing/community-workflows.md)
  for structured submission lanes, required evidence, validation commands, and
  privacy review.
- [`docs/contributing/plugin-author-guide.md`](docs/contributing/plugin-author-guide.md)
  for `PluginRegistry` extensions, privacy expectations, and compatibility tests.
- [`docs/contributing/checker-design.md`](docs/contributing/checker-design.md)
  for `CheckMode`, witness, abstention, and differential-evidence standards.
- [`docs/contributing/corpus-contributions.md`](docs/contributing/corpus-contributions.md)
  for provenance, license, no secrets, and expected-diagnostic requirements.
