# Quickstart

Install the package in editable mode while developing:

```bash
python -m pip install -e ".[dev]"
promptabi verify --config examples/minimal/promptabi.json
```

The first workflow intentionally stays small: it proves that the CLI, typed API,
configuration loading, deterministic rendering, and fixture paths are wired.
Later milestones plug formal checks into the same `VerificationSession` API.

