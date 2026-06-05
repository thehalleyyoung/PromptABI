# Quickstart

Install the package in editable mode while developing:

```bash
python -m pip install -e ".[dev]"
promptabi verify --config examples/minimal/promptabi.json
promptabi verify --config examples/role-boundary/unsafe.promptabi.json
promptabi verify --config examples/token-budget/promptabi.json --format json
promptabi verify --config examples/minimal/promptabi.json --format sarif > promptabi.sarif
promptabi github-action --config examples/minimal/promptabi.json --require-lockfile
promptabi verify --config examples/end-to-end/training-quickstart/fixed.promptabi.json
```

The first workflow intentionally stays small: it proves that the CLI, typed API,
artifact model, configuration loading, artifact loading, deterministic rendering,
and fixture paths are wired. Diagnostics already have stable fingerprints,
witness traces, suggestions, check-mode metadata (`sound`, `complete`, `bounded`,
`z3-backed-smt`, `heuristic`, or `abstaining`), and text/JSON/SARIF renderers.
Embedding tools can use `create_session`, `load_artifacts`,
`collect_diagnostics`, `run_verification`, and custom `VerificationSession`
checks without shelling out to the CLI.

For GitHub repositories, use `./.github/actions/promptabi` from a workflow. The
action restores `.promptabi/cache`, enforces a lockfile, skips pull requests that
do not touch configured PromptABI inputs, uploads SARIF to code scanning, and
writes a markdown job summary.

For stack-specific setup, see the integration guides for LangChain, LlamaIndex,
vLLM, llama.cpp, Hugging Face Transformers, OpenAI-compatible servers, LiteLLM,
MCP tools, custom agent frameworks, and training pipelines.

For disconnected CI or regulated environments, see the air-gapped installation
guide. It covers vendored wheels, pinned Z3 packages, offline corpora, provider
fixture mirrors, prompt-pack mirrors, and reproducibility gates.

For fine-tuning data, the training quickstart verifies a one-row chat SFT
dataset end to end from rendered roles through token spans, packing boundaries,
loss masks, tokenizer/template stage pins, and redacted witness hashes without
loading model weights.

For a concrete structural security check, run the role-boundary example. The
unsafe ChatML-style template fails because raw message fields can render as
role/control delimiters; the paired sanitized config shows the same template
shape with JSON-encoded dynamic fields.

Before sharing diagnostics outside your team, read the security model. PromptABI
is local and telemetry-free by default, but rendered reports and witnesses can
still contain sensitive artifact structure unless you minimize and sanitize them.
