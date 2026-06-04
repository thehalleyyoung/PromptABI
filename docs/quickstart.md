# Quickstart

Install the package in editable mode while developing:

```bash
python -m pip install -e ".[dev]"
promptabi verify --config examples/minimal/promptabi.json
promptabi verify --config examples/minimal/promptabi.json --format sarif > promptabi.sarif
```

The first workflow intentionally stays small: it proves that the CLI, typed API,
artifact model, configuration loading, deterministic rendering, and fixture paths
are wired. Diagnostics already have stable fingerprints, witness traces,
suggestions, check-mode metadata (`sound`, `complete`, `bounded`,
`z3-backed-smt`, `heuristic`, or `abstaining`), and text/JSON/SARIF renderers.
Later milestones plug formal checks into the same `VerificationSession` API.
