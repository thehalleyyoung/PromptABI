# PromptABI

**CI for the tokenizer/template/tool-calling boundary of LLM apps.** PromptABI
verifies the discrete interface layer around LLM systems before deployment:
chat templates, tokenizers, special tokens, stop policies, tool schemas,
structured-output grammars, provider contracts, and token budgets.

```bash
promptabi verify --config examples/minimal/promptabi.json
# or, inside a repo with promptabi.json:
promptabi verify --artifact schema=schemas/answer.json --fail-on warning
```

```text
PromptABI verification: minimal-chat-template
checks: repository-skeleton
status: PASS
INFO repository-skeleton: PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.
  fingerprint: 9344d195f04b768d
  witness: The verification session constructed a typed config and produced deterministic output.
    1. load JSON config | input: minimal-chat-template | output: 3 artifacts
    2. normalize artifact paths
    3. render stable diagnostics
```

PromptABI is CPU-only because its claims are structural. It models the exact
artifacts around an LLM system--tokenizers, chat templates, special tokens, stop
policies, schemas, grammars, tools, prompt segments, providers, and truncation
configs--then checks whether their composed contract makes protocol states
possible, impossible, ambiguous, or unsafe.

```text
messages -> chat template -> byte/string prompt -> tokenizer -> token stream
        -> constrained decoder / stop logic / tool parser -> application parser
```

PromptABI now includes the first formal core: deterministic finite automata for
reachability, emptiness, set operations, and shortest witnesses, plus a finite
contract solver over booleans, enums, integer ranges, and bounded strings. The
solver uses Z3 when available and falls back to exhaustive finite-domain solving,
so checks can prove satisfiability or incompatibility and emit concrete
counterexample assignments without logits, GPUs, inference, or network calls.

The roadmap targets three high-value checks first: role-boundary
non-forgeability, stop/grammar/tokenizer reachability, and must-survive
token-budget verification. The repository already has the typed Python package,
core artifact model, stable diagnostic contract, text/JSON/SARIF renderers,
discoverable `promptabi verify` workflow, offline version-pinned artifact
loading, a real tokenizer abstraction spanning byte-level, Hugging Face
`tokenizers`, `tiktoken`, and SentencePiece backends, a differential harness that
checks those adapters against the actual libraries, plus docs, examples,
fixtures, benchmarks, and a contribution path for growing checks without
changing the public surface.

## Why this is clearly distinct from TensorGuard

TensorGuard verifies the numeric tensor-computation layer: shapes, devices, phases, dtypes, PyTorch operator contracts, checkpoint compatibility, export gates, and related model-execution invariants.

PromptABI verifies the language/protocol layer around LLMs. Its artifacts are tokenizer JSON files, Jinja chat templates, stop strings, JSON schemas, grammar compilers, tool-call schemas, context-window policies, and provider API conventions. The overlap is methodological, not topical: both are static/differential verifiers for real ML failure modes, but they operate on different objects and catch different bugs.

## The unifying formalism

The project should not be pitched as a bag of LLM linters. The paper-worthy framing is:

> LLM application interfaces are compositions of finite-state transducers over byte, string, and token alphabets, with partial parsers at the boundaries.

Most high-value checks become automata-theoretic properties:

- **Reachability:** can a declared stop sequence actually be produced under this tokenizer and grammar?
- **Over-reachability:** can a stop sequence fire inside a valid JSON string, tool argument, code block, or user-controlled field?
- **Emptiness:** is `tokenizer x grammar x schema` empty, meaning the constrained decoder can never produce a valid object?
- **Non-forgeability:** can attacker-controlled text render as role delimiters, system-message markers, assistant prefixes, tool-call sentinels, BOS/EOS tokens, or provider-specific control tokens?
- **Round-trip parseability:** does `render -> tokenize -> detokenize -> parse` preserve the intended message/tool structure?
- **Must-survive budget constraints:** do system instructions, tool definitions, safety preambles, retrieval citations, or output-format requirements remain present under the framework's real truncation policy?

The implementation combines finite-state automata, a Z3-backed finite-contract
solver, differential checks against real tokenizer libraries, and an explicit
static boundary: automata prove language/reachability facts, SMT proves bounded
symbolic compatibility, and differential tests validate that abstractions match
libraries people actually run.

## Why no GPU genuinely does not limit applicability

PromptABI's core claims are structural, not behavioral. It does not need to ask whether a model will choose a token, follow an instruction, or resist prompt injection. It asks whether the interface contract makes certain token/string/protocol states possible, impossible, ambiguous, or unsafe.

That makes the CPU-only story honest:

- Tokenization runs on CPU.
- Chat-template rendering runs on CPU.
- JSON Schema, grammar compilation, parser checks, and automata operations run on CPU.
- Stop-string and token-boundary analysis is independent of model weights.
- Provider request/response compatibility can be tested from recorded fixtures and SDK behavior.
- Context-window and truncation checks depend on token counts and framework policies, not logits.

The project must be explicit about this boundary. It can prove "an attacker-controlled field can structurally forge a role delimiter"; it cannot prove "the model will obey the forged role." It can prove "this stop string can terminate inside a valid JSON string"; it cannot prove "the model will emit that string in practice." That boundary is a strength because it keeps the tool broadly applicable without GPU access.

## High-value bugs it would catch

1. A chat template where user content containing a model's assistant delimiter can create an apparent assistant turn after rendering.
2. A system prompt that is preserved in unit tests but silently dropped by LangChain/vLLM/llama.cpp style truncation in production.
3. A JSON schema that is valid in Python but compiles to an empty or unsatisfiable constrained-decoding grammar under the selected tokenizer.
4. A stop string that cannot be reached because its byte/string form does not align with the tokenizer or grammar.
5. A stop string that can fire inside a legitimate tool argument, causing truncated JSON and flaky parsers.
6. A provider migration that changes tool-call serialization enough to make a downstream parser accept a different AST than intended.
7. A tokenizer/config update that introduces double-BOS, missing-EOS, changed special-token IDs, or changed role-control token spellings.
8. A RAG chunking policy whose token accounting differs from the serving tokenizer, making citation-bearing spans disappear at the context-window boundary.

## What would make it a 1000-star repo

The README demo should be painfully concrete:

```bash
promptabi verify \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --chat-template tokenizer_config.json \
  --tools tools.json \
  --schema answer.schema.json \
  --max-context 8192 \
  --framework vllm
```

Example output:

```text
FAIL role-boundary-nonforgeability
  user.content can render '<|start_header_id|>assistant<|end_header_id|>'
  as a real assistant boundary after template expansion.

FAIL stop-string-overreachability
  stop='</tool_call>' is reachable inside $.arguments.comment
  before the JSON object is complete.

FAIL must-survive-budget
  under vLLM left-truncation, tool schema `refund_user` is dropped
  while messages still reference it.
```

This is the kind of repo LLM app developers would star because it fits directly into CI and catches failures that are otherwise discovered through confusing production behavior.

## What would make it paper-grade

The paper needs one deep abstraction and a real-bug corpus.

The abstraction:

- model chat templates, tokenizer encoders/decoders, grammars, stop criteria, and parsers as composed transducers;
- define soundness/abstention contracts for each supported fragment;
- abstain on arbitrary Jinja or provider behavior that cannot be modeled soundly;
- differentially check the abstraction against the real Hugging Face tokenizer, `apply_chat_template`, grammar backends such as Outlines/xgrammar/llguidance, and provider SDK fixtures.

The evaluation:

- a corpus of popular tokenizers and templates: Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, OpenAI-compatible adapters, llama.cpp GGUF metadata, and common fine-tune templates;
- a corpus of real structured-output/tool-calling schemas from open-source agents;
- upstreamable bugs with minimized repros;
- version-drift tests showing that library/model updates break previously valid contracts.

## Minimal viable scope

Start with three checks that are both useful and formally interesting:

1. **Role-boundary non-forgeability:** prove whether user/tool/RAG-controlled fields can create control-token or role-delimiter structure after template rendering.
2. **Stop/grammar/tokenizer reachability:** determine whether stop strings are unreachable, ambiguous, or reachable inside valid structured outputs.
3. **Must-survive token-budget verification:** check whether required prompt segments survive real truncation policies for the selected framework and context window.

Everything else can be added later as breadth: schema linting, provider diffs, generation-config checks, tokenizer drift reports, and compatibility dashboards.

## Suggested positioning

**Tagline:** "CI for the tokenizer/template/tool-calling boundary of LLM apps."

**Longer pitch:** PromptABI verifies that the non-neural parts of an LLM system compose correctly before deployment. It catches token-boundary, role-forgery, structured-output, stop-sequence, tool-schema, and truncation bugs without loading model weights or running inference.

**Why now:** LLM engineering has moved from raw prompting to tool calling, structured decoding, model/provider swaps, RAG pipelines, OpenAI-compatible servers, and fragile chat templates. The interface layer is now complex enough to deserve its own verifier.

## Main risk

Avoid claiming semantic model safety. The tool should not promise to prevent prompt injection in the behavioral sense. It should promise structural non-forgeability of the prompt/interface representation. That narrower claim is still valuable, provable, CPU-only, and broadly applicable.
