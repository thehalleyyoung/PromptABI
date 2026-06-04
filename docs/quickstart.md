# Quickstart

Install the package in editable mode while developing:

```bash
python -m pip install -e ".[dev]"
promptabi verify --config examples/minimal/promptabi.json
promptabi verify --config examples/role-boundary/unsafe.promptabi.json
promptabi verify --config examples/minimal/promptabi.json --format sarif > promptabi.sarif
```

The first workflow intentionally stays small: it proves that the CLI, typed API,
artifact model, configuration loading, artifact loading, deterministic rendering,
and fixture paths are wired. Diagnostics already have stable fingerprints,
witness traces, suggestions, check-mode metadata (`sound`, `complete`, `bounded`,
`z3-backed-smt`, `heuristic`, or `abstaining`), and text/JSON/SARIF renderers.
Embedding tools can use `create_session`, `load_artifacts`,
`collect_diagnostics`, `run_verification`, and custom `VerificationSession`
checks without shelling out to the CLI.

For a concrete structural security check, run the role-boundary example. The
unsafe ChatML-style template fails because raw message fields can render as
role/control delimiters; the paired sanitized config shows the same template
shape with JSON-encoded dynamic fields.
