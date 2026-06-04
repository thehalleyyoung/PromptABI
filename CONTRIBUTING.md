# Contributing to PromptABI

PromptABI is a verifier, not a prompt-style linter. Contributions should keep
checks deterministic, CPU-only, reproducible, and explicit about whether a result
is sound, bounded, heuristic, or an abstention.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_public_api.py tests/test_config.py tests/test_cli.py
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

