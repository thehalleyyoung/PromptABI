# PromptABI

**CI for the tokenizer/template/tool-calling boundary of LLM apps.** PromptABI
verifies the discrete interface layer around an LLM system before deployment:
chat templates, tokenizers, special tokens, stop policies, structured-output
grammars, tool schemas, provider contracts, training manifests, and token
budgets. It is CPU-only because its claims are structural: it does not load
weights, call providers, or guess what a model will choose.

```bash
python -m pip install -e ".[dev,grammars,solver,tokenizers]"

# Prove a real ChatML-style template lets user content forge assistant control.
promptabi verify --config examples/role-boundary/unsafe.promptabi.json --fail-on never

# Expand one finding into source spans, formal property, witness, symptom, fix.
promptabi explain --config examples/role-boundary/unsafe.promptabi.json --index 1

# Gate a repo with cache, lockfile drift checks, annotations, and SARIF upload.
promptabi github-action --config examples/minimal/promptabi.json --require-lockfile

# In monorepos, recompute only checks affected by changed PromptABI inputs.
promptabi verify --config examples/minimal/promptabi.json --changed-from-git origin/main
```

```text
PromptABI verification: role-boundary-unsafe-chatml
checks: role-boundary-nonforgeability
status: FAIL
ERROR role-boundary-nonforgeability [bounded, sound]:
  {messages[0].content} can forge assistant-prefix '<|im_start|>'
  in a {messages[0].role} region
  span: examples/role-boundary/unsafe-tokenizer_config.json:2:20-2:197
  witness:
    1. build bounded role-region model
    2. substitute attacker-controlled field | output: <|im_start|>
    3. render forged boundary excerpt
    4. tokenize forged excerpt | output: byte-level ... '<|im_start|>'/special
    5. locate forged boundary | output: assistant-prefix at rendered chars 31:43
  suggestion: Render user-controlled fields through an escaping or encoding layer.
```

PromptABI is not another prompt-injection scanner and not an inference
benchmark. It checks whether the artifacts around an LLM system compose into
possible, impossible, ambiguous, or unsafe protocol states:

```text
messages -> chat template -> bytes/strings -> tokenizer -> token stream
        -> grammar/stop/tool parser -> application parser/provider contract
```

The same local verifier covers:

| Surface | What PromptABI proves or detects |
| --- | --- |
| **Role boundaries** | user/tool/RAG fields that can render as assistant, system, tool-call, BOS/EOS, or provider control structure |
| **Stops** | unreachable stops, tokenizer normalization ambiguity, special-token collisions, and stops that can fire inside JSON/tool/string fields |
| **Grammars + schemas** | JSON Schema/regex/EBNF/Outlines/xgrammar/llguidance fragments that are empty, ambiguous, or parser-incompatible under tokenizer assumptions |
| **Tools + providers** | OpenAI, Anthropic, MCP, LangChain, Pydantic, TypeScript-style, vLLM, llama.cpp, LiteLLM, Gemini, Bedrock, Groq, Together, and Ollama serialization drift |
| **Prompt packs** | reusable templates with compositional role/tool/stop/model-family/RAG/truncation guarantees, locks, upgrade gates, registries, signing, and offline mirrors |
| **Budgets + RAG** | must-survive prompt segments, dropped citations, metadata inflation, tokenizer mismatch, framework truncation, and context-window overflow |
| **Provenance** | artifact hashes, licenses, trusted sources, reproducible HF revisions, lockfile drift, offline fixture integrity, and upstream bug/fix/workaround links |
| **Enterprise posture** | org policy packs, no-network mirrors, severity gates, privacy rules, approved fixtures, and expiring witness-stable accepted-risk suppressions |
| **Training/eval contracts** | target-role alignment, benchmark-tokenizer drift, eval-harness prompt/few-shot/multi-turn/grading-parser/stop/private-field/cross-provider compatibility, proof-carrying streaming shards, data-loader adapters, synthetic-generator preflight, invalid roles/tools/JSON/stops, packing, loss masks, leakage, drift, RLHF/DPO defects, and real-bug benchmarks |

Under the hood, PromptABI combines a declarative static contract language,
deterministic automata, finite-state
transducers, executable specs for witnesses/products/SMT outcomes, differential
checks against real tokenizer/template libraries, and a Z3-backed finite
contract layer over booleans, enums, integer ranges, membership, lengths, and
bounded strings. Solver queries are cacheable by normalized formulas, artifact
hashes, supported-fragment metadata, and solver-version fingerprints. Every
diagnostic states its guarantee mode--`sound`, `complete`, `bounded`,
`z3-backed-smt`, `heuristic`, or `abstaining`--and formal counterexamples can be
shrunk to minimal strings, token paths, and solver assignments, so CI can
distinguish proof from best-effort evidence.

## Daily workflows

```bash
# Small deterministic smoke check.
promptabi verify --config examples/minimal/promptabi.json

# HTML report with interactive witness overlays, budgets, diffs, and corpus summaries.
promptabi verify --config examples/token-budget/promptabi.json --format html --fail-on never > promptabi-report.html

# Scaffold a real stack contract.
promptabi init --stack openai-tools --output-dir .promptabi-demo

# Follow stack guides for LangChain, LlamaIndex, vLLM, llama.cpp, HF, LiteLLM, MCP, and training.

# Replay paired buggy/fixed app contracts across tools, JSON, RAG, providers, and training.
promptabi verify --config examples/end-to-end/tool-calling/buggy.promptabi.json --fail-on never
promptabi verify --config examples/prompt-packs/promptabi.json
promptabi prompt-pack lock --config examples/prompt-packs/promptabi.json --write --lockfile /tmp/prompt-pack.lock.json
promptabi prompt-pack registry --config examples/prompt-packs/promptabi.json --output /tmp/prompt-pack.registry.json
promptabi prompt-pack provenance create --config examples/prompt-packs/promptabi.json --output /tmp/prompt-pack.provenance.json --key local-review-key
promptabi prompt-pack mirror build --config examples/prompt-packs/promptabi.json --mirror-dir /tmp/prompt-pack-mirror
promptabi prompt-pack upgrade --config examples/prompt-packs/promptabi.json --baseline-lockfile /tmp/prompt-pack.lock.json
python examples/agent-frameworks/dynamic_support_agent.py examples/agent-frameworks/safe.agent-prompt-pack.json --write-config /tmp/support-agent.promptabi.json
promptabi verify --config examples/end-to-end/training-quickstart/fixed.promptabi.json
promptabi verify-training --manifest examples/end-to-end/training-quickstart/fixed.training-manifest.json

# Shrink a failure into an upstreamable repro and sanitized issue.
promptabi minimize repro.json --keep-substring "<|im_start|>" --format json
promptabi bug-report --config examples/role-boundary/unsafe.promptabi.json --index 1 > upstream-issue.md

# Compare an upgrade before merge and audit supported guarantees.
promptabi diff promptabi.baseline.json promptabi.json
promptabi version-gate promptabi.baseline.json promptabi.json --allowed-impact patch-safe
PROMPTABI_BUNDLE_KEY=local-audit-key promptabi bundle create --config examples/minimal/promptabi.json --output promptabi.bundle.json
promptabi matrix --format text
promptabi soundness-audit --rule role-boundary-nonforgeability --format markdown
promptabi graph --config examples/rag-chunking/promptabi.json --all-checks --format mermaid
promptabi contract format examples/static-contract-language/app.pabi --check
promptabi contract migrate examples/static-contract-language/app.pabi --check
promptabi contract lint examples/static-contract-language/app.pabi
promptabi contract compose --contract organization-policy=examples/static-contract-language/app.pabi --contract app-config=examples/static-contract-language/rag.pabi
promptabi api-docs --format markdown
promptabi proofs --format text

# In notebooks, inspect tokenizer/template/stop/grammar/SMT/budget witnesses as rich reprs.
python -c "from promptabi import visualize_tokenization; from promptabi.tokenizers import ByteLevelTokenizer; print(visualize_tokenization('<|im_start|> hi', ByteLevelTokenizer(added_tokens=('<|im_start|>',))))"

# Share CI evidence without leaking prompt text: keep offsets/token IDs, hash witness payloads.
promptabi verify --config examples/role-boundary/unsafe.promptabi.json --witness-privacy hash-only --format json --fail-on never

# Show pinned, verified real-world-style configs with proof/risk badges.
promptabi gallery --format text

# Diagnose the local environment and emit editor-ready inline diagnostics.
promptabi doctor --config examples/minimal/promptabi.json
promptabi diagnostics cluster --config examples/minimal/promptabi.json --strategy rule
promptabi dashboard --config examples/role-boundary/unsafe.promptabi.json --history .promptabi/team-dashboard.jsonl --record
promptabi diagnostics catalog --config examples/minimal/promptabi.json --format json
promptabi diagnostics lsp --config examples/minimal/promptabi.json --format json
# Clusters rank fixes by safety, compatibility, blast radius, and prompt-behavior impact.
promptabi fix --config examples/minimal/promptabi.json --kind lockfile --write
promptabi fix --config examples/role-boundary/unsafe.promptabi.json --preview-risk high

# Run labeled benchmarks, release leaderboards, mutation fuzzing, and paper artifacts.
promptabi corpus verify --format text
promptabi corpus bug-gallery --format markdown > bug-gallery.md
promptabi corpus beta-report --format text
promptabi corpus evaluation --format text
promptabi corpus evaluation-reproducibility --config examples/evaluation-harness/safe.promptabi.json --format json
promptabi verify --config examples/evaluation-harness/safe.promptabi.json
promptabi corpus grammar-conformance --format text
promptabi corpus tokenizer-conformance --format text
promptabi corpus provider-conformance --format text
promptabi corpus framework-truncation-conformance --format text
promptabi corpus smt-benchmark --format text
promptabi corpus leaderboard --format text
promptabi corpus adversarial --format text
promptabi solver replay fixtures/solver_replays/role-region-forgery.solver-replay.json
promptabi launch-assets --output-dir launch_assets --force
promptabi fuzz mutations --format text
promptabi maintain refresh --output-dir maintainer_artifact --force
promptabi paper reproducibility --output-dir paper_artifact --force
promptabi release compatibility-audit --candidate-version tokenizer=seed-v1 --candidate-version template=seed-v1 --candidate-version provider=provider-fixtures-v1 --candidate-version grammar=grammar-differential-v1 --candidate-version framework=structured-schemas-v1
promptabi release drift-bisect --surface tokenizer --baseline tok-r0 --revision tok-r1=tok-r1 --revision tok-r2=tok-r2 --bad-field eos_token_id
promptabi release readiness --format text

# Local changed-artifact gate and local-only usage summaries; no telemetry.
promptabi pre-commit install --config examples/minimal/promptabi.json
promptabi verify --config examples/minimal/promptabi.json --local-summary .promptabi/usage.jsonl
promptabi usage metrics --config examples/minimal/promptabi.json --format json
promptabi usage privacy

# Incremental monorepo mode reuses cached unchanged-check diagnostics and recomputes dependencies.
promptabi verify --config examples/minimal/promptabi.json --changed-path examples/minimal/schema.json
```

## GitHub Actions

```yaml
permissions:
  contents: read
  security-events: write

jobs:
  promptabi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-python@v5
        with: { python-version: "3.14" }
      - uses: ./.github/actions/promptabi
        with:
          config: examples/minimal/promptabi.json
          lockfile: examples/minimal/promptabi.lock.json
          require-lockfile: "true"
          changed-only: "true"
          install-command: python -m pip install -e ".[dev,grammars,solver,tokenizers]"
```

The action restores `.promptabi/cache`, skips unrelated pull requests, enforces
lockfiles, emits workflow annotations, uploads SARIF to code scanning, and
writes a markdown job summary. Training-data PRs can use the same action with
`training-manifest:` to run `promptabi verify-training` before fine-tuning jobs
start.

## Why the boundary matters

PromptABI deliberately stays below model semantics. It can prove that a
user-controlled field can structurally forge a role delimiter; it cannot prove
the model will obey that forged role. It can prove a stop string can terminate
inside a valid JSON string before the object is complete; it cannot prove the
model will emit that string. This narrower claim is what makes the tool fast,
offline, reproducible, and suitable for CI.

The security model is local and non-telemetric: provider fixtures are validated
for credential-like values, solver inputs stay on the runner, bug reports are
sanitized markdown, and optional usage summaries record only aggregate command
metadata. See [`docs/security-model.md`](docs/security-model.md) for the exact
privacy and non-goal guarantees.
