# Integration guides

PromptABI is meant to sit beside the code that already builds prompts, calls
providers, parses structured outputs, retrieves context, or prepares supervised
fine-tuning data. The common pattern is:

1. Identify the discrete artifacts your stack already uses: tokenizer metadata,
   chat templates, tools, schemas, stop policies, provider fixtures, prompt
   segments, framework truncation settings, and training manifests.
2. Put stable local snapshots of those artifacts under review, or reference pinned
   immutable artifact URIs when local copies are not practical.
3. Create a `promptabi.json` that names the artifacts and the checks that protect
   the boundary.
4. Run the same command locally, in pre-commit, and in CI:

```bash
promptabi verify --config promptabi.json
```

When the target environment has no network access, stage the same artifacts with
the air-gapped installation guide: vendored wheels, pinned Z3 packages, local
corpora, provider fixture mirrors, prompt-pack mirrors, and reproducibility
manifests are all verified by repository commands rather than live provider
calls.

Use `promptabi init` when starting from a known stack. Every scaffold writes local
fixture stubs and a config that the repository tests verify through the real
PromptABI loader and checker scheduler:

```bash
promptabi init --stack openai-tools --output-dir .promptabi/openai-tools
promptabi init --stack huggingface-local --output-dir .promptabi/huggingface-local
promptabi init --stack vllm-openai --output-dir .promptabi/vllm-openai
promptabi init --stack llama-cpp --output-dir .promptabi/llama-cpp
promptabi init --stack langchain-rag --output-dir .promptabi/langchain-rag
promptabi init --stack llamaindex-agent --output-dir .promptabi/llamaindex-agent
promptabi init --stack json-schema --output-dir .promptabi/json-schema
```

## LangChain RAG

LangChain applications commonly fail at the boundary between retrieved documents,
message history, tool definitions, and the model context window. PromptABI models
that boundary with `prompt-segment` and `framework-truncation-config` artifacts.

Start from the scaffold when adding PromptABI to a LangChain repo:

```bash
promptabi init --stack langchain-rag --output-dir .promptabi/langchain-rag
promptabi verify --config .promptabi/langchain-rag/promptabi.json --format html > promptabi-report.html
```

For a concrete repository fixture, compare the checked example:

```bash
promptabi verify --config examples/rag-chunking/promptabi.json --fail-on never
promptabi verify --config examples/token-budget/promptabi.json --format json
```

Map LangChain concepts as follows:

| LangChain object | PromptABI artifact | Checks to enable |
| --- | --- | --- |
| System prompt, recent messages, tool catalog, retrieved chunks | `prompt-segment` | `token-budget-model`, `rag-chunking-compatibility` |
| Memory or retriever truncation settings | `framework-truncation-config` with `framework: "langchain"` | `token-budget-model` |
| OpenAI-style tools passed through LangChain | `tool-definition` | `tool-schema-ingestion`, `tool-call-serialization` |
| Output parser JSON Schema | `schema` | `parser-compatibility`, `tokenizer-grammar-emptiness` |

Mark policy, citations, tool catalogs, and safety context as required segments.
PromptABI then proves whether the configured truncation strategy can drop them
before the model sees the prompt. For RAG, include tokenizer identity,
document/chunk IDs, expected overlap, citation fields, metadata token cost, and
retrieval payload limits so the checker can distinguish a budget problem from a
chunking mismatch.

## LlamaIndex agents

LlamaIndex agents combine an index context, a tool catalog, memory, and provider
adapters. The scaffold represents those pieces as prompt segments plus tool
definitions:

```bash
promptabi init --stack llamaindex-agent --output-dir .promptabi/llamaindex-agent
promptabi verify --config .promptabi/llamaindex-agent/promptabi.json --fail-on never
```

Use a `prompt-segment` artifact for system policy, index context, tool catalog,
and user query regions. Use `required: true` only for regions that must survive
budget pressure; the budget checker will emit a minimized dropped-segment witness
when the LlamaIndex truncation policy cannot preserve them. Pair the segment file
with `tool-definition` artifacts for any OpenAI, Anthropic, Pydantic, LangChain,
TypeScript-style, or MCP tool schemas that the agent can call.

## vLLM OpenAI-compatible servers

vLLM often presents an OpenAI-shaped API while preserving local tokenizer,
sampling, stop, and streaming behavior. Capture that as a provider fixture and a
stop-policy artifact:

```bash
promptabi init --stack vllm-openai --output-dir .promptabi/vllm-openai
promptabi verify --config .promptabi/vllm-openai/promptabi.json --fail-on never
promptabi verify --config examples/stop-policies/promptabi.json --fail-on never
```

For production configs, snapshot the request fields, response fields, tool-call
encoding paths, streaming delta paths, context limits, and stop handling from the
server version you deploy. PromptABI can then replay the provider fixture offline,
check stop reachability, compare vLLM behavior with OpenAI-compatible assumptions,
and detect migrations that change tool-call IDs, argument encodings, parallel-call
support, or stop semantics.

## llama.cpp and Ollama local servers

llama.cpp and Ollama deployments often share OpenAI-compatible client code but
differ in stop handling, tool-call support, context budgets, and tokenizer
metadata. Start with the llama.cpp scaffold:

```bash
promptabi init --stack llama-cpp --output-dir .promptabi/llama-cpp
promptabi verify --config .promptabi/llama-cpp/promptabi.json --fail-on never
```

Record server options such as context length, stop strings, EOS handling, and
tool-call envelope shape. When a local server is used as a drop-in replacement for
OpenAI, add source and target provider fixtures so the provider-migration check
can report contract-breaking changes before the adapter swap reaches production:

```bash
promptabi verify --config fixtures/provider_migration/promptabi.json --fail-on never
```

## Hugging Face Transformers

Transformers applications should verify the tokenizer directory and
`tokenizer_config.json` chat template that are actually used by
`apply_chat_template`:

```bash
promptabi init --stack huggingface-local --output-dir .promptabi/huggingface-local
promptabi verify --config examples/role-boundary/unsafe.promptabi.json --fail-on never
promptabi verify --config examples/role-boundary/safe.promptabi.json
```

Use `tokenizer` and `chat-template` artifacts for local model directories. Pin
the tokenizer files with lockfiles when the directory is vendored:

```bash
promptabi verify --config promptabi.json --write-lockfile
promptabi verify --config promptabi.json --require-lockfile
```

For fine-tuned or instruction-tuned models, enable role-boundary checks on raw
message content and dynamic role fields. PromptABI distinguishes the structural
fact that a delimiter can be forged from the semantic question of whether the
model will obey it.

## OpenAI-compatible servers

OpenAI-compatible providers are a contract family, not one behavior. Capture the
specific server or route in a `provider-config` artifact and pair it with tools,
schemas, and stops:

```bash
promptabi init --stack openai-tools --output-dir .promptabi/openai-tools
promptabi verify --config examples/minimal/promptabi.json
promptabi github-action --config examples/minimal/promptabi.json --require-lockfile
```

For Azure OpenAI, Together, Groq, self-hosted vLLM, llama.cpp server, and other
compatible endpoints, compare against a baseline fixture before migration:

```bash
promptabi verify --config fixtures/provider_migration/promptabi.json --fail-on never
```

The provider migration checker focuses on request/response field loss, tool
argument encoding, tool-call IDs, parallel-call support, streaming chunks, stop
sequences, context limits, structured-output modes, error envelopes, and router
behavior.

## LiteLLM routers

LiteLLM adds an important extra boundary: one application contract may route to
multiple providers. Treat the router configuration as a provider artifact and
record the downstream provider family for each route. The fixture corpus includes
a LiteLLM provider-migration target at
`fixtures/provider_migration/litellm-target.json`:

```bash
promptabi verify --config fixtures/provider_migration/promptabi.json --fail-on never
```

In a production repository, store one fixture per route class rather than one
fixture per secret-bearing deployment. Fixtures should describe shapes and limits,
not credentials. PromptABI's provider fixture validation rejects credential-like
fields so sanitized route fixtures can be checked into private CI safely.

## MCP tools

MCP tool definitions map naturally to PromptABI `tool-definition` artifacts. Put
the MCP tool list or server-exported schema snapshot in a local JSON file, then
reference it with `provider: "mcp"` or the appropriate metadata used by your
adapter. Enable:

| MCP surface | PromptABI check |
| --- | --- |
| `inputSchema` closedness and required fields | `tool-schema-ingestion`, `static-contracts` |
| Tool-call name and argument envelope | `tool-call-serialization` |
| Provider-specific tool response format | `provider-fixture-replay`, `provider-migration` |
| Tool catalog budget survival | `token-budget-model` |

If an MCP server changes a tool name, required argument, enum, or response
envelope, PromptABI should fail in CI before an agent starts emitting calls the
runtime parser no longer accepts.

## Custom agent frameworks

For internal frameworks, do not start by writing a plugin. First model the
framework's observable boundary with existing artifacts:

| Framework boundary | PromptABI artifact |
| --- | --- |
| Prompt assembly output or message list | `prompt-segment` or `chat-template` |
| Truncation and memory policy | `framework-truncation-config` |
| Tool definitions and parser expectations | `tool-definition`, `schema`, `grammar` |
| Provider request/response shape | `provider-config` |
| Stops and sentinels | `stop-policy`, `special-token-map` |

Run the generic verifier first. Add a plugin only when the framework has a new
artifact format or a new contract that cannot be expressed by those objects. A
plugin should register typed loaders or checks while preserving deterministic
diagnostics, source spans, and local-only execution.

## Training and fine-tuning data pipelines

Training pipelines need the same interface checks before expensive jobs start.
The relevant artifact is a `training-manifest` describing dataset role labels,
target roles, loss-mask regions, packing windows, tokenizer/template versions,
and any supervised or preference-data transforms. The static-contract layer
already models finite target-role alignment against the serving chat template,
and the compatibility matrix marks the deeper loss-mask, packing, and preference
checks as planned abstentions until their supported fragments are implemented.

Use PromptABI as a preflight gate:

```bash
promptabi matrix --format text
promptabi verify-training --manifest training.training-manifest.json --fail-on warning
```

The reusable GitHub Action can run the same dedicated workflow for data-only
pull requests. The example in `.github/workflows/promptabi-training-data.yml`
watches manifests and JSONL shards, skips unrelated diffs, and fails before an
expensive fine-tuning job starts.

At minimum, pin the tokenizer and chat template used for dataset preparation,
training, evaluation, and serving. A useful manifest should make it impossible to
silently train on a target role that the serving template cannot render, leak user
or retrieval text into supervised loss regions, or pack examples across
boundaries without a reproducible witness.

## CI and local workflow

After the first local pass, make the check persistent:

```bash
promptabi pre-commit install --config promptabi.json
promptabi verify --config promptabi.json --write-lockfile
promptabi github-action --config promptabi.json --require-lockfile
```

Use `--fail-on warning` while hardening a new integration and `--fail-on error`
once accepted warnings are represented as expiring suppressions or policy files.
Use `promptabi explain` for a local tutorial trace and `promptabi bug-report` to
produce an upstream-ready, sanitized issue from the exact diagnostic object CI
reported:

```bash
promptabi explain --config examples/role-boundary/unsafe.promptabi.json --index 1
promptabi bug-report --config examples/role-boundary/unsafe.promptabi.json --index 1 > upstream-issue.md
```
