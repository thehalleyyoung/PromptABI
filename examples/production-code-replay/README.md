# Production-code replay

This example replays real public bugs from `fixtures/real_world_bugs/production_code.json`.
The corpus stores exact pinned upstream source excerpts and hashes; PromptABI
extracts templates or parser delimiters from those bytes before running the
analyzers.

Run it from the repository root:

```bash
PYTHONPATH=src python examples/production-code-replay/replay.py
```
