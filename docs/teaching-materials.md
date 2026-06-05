# Teaching materials

PromptABI teaching material is built from executable examples and proof notebooks, not slides detached from code.

## Modules

1. LLM interface artifacts as compiler boundaries
2. Tokenizer/template/tool parser products
3. SMT-backed finite contracts and honest abstention
4. Training and evaluation contract preflight
5. Operationalizing proof evidence in CI, IDEs, registries, and deployments

## Labs

- **Tokenizer/template non-forgeability:** `examples/role-boundary/unsafe.promptabi.json`
- **Tool-call stop and serialization contracts:** `examples/end-to-end/tool-calling/buggy.promptabi.json`
- **Must-survive RAG and truncation budgets:** `examples/end-to-end/rag-truncation/buggy.promptabi.json`
- **SMT-backed static contracts:** `examples/static-contract-language/app.pabi`
- **Provider-standard conformance and migration:** `examples/end-to-end/provider-migration/buggy.promptabi.json`

## Proof notebooks

- `examples/proof-sketch-notebooks/01-role-boundary-nonforgeability.ipynb`
- `examples/proof-sketch-notebooks/02-stop-reachability.ipynb`
- `examples/proof-sketch-notebooks/03-grammar-emptiness.ipynb`
- `examples/proof-sketch-notebooks/04-budget-survival.ipynb`
- `examples/proof-sketch-notebooks/05-training-mask-alignment.ipynb`
