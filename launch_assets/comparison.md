# PromptABI comparison

Evidence: 11 labeled cases, 7 replayed real-bug reductions, precision=1.0, recall=1.0.

| System class | What it checks | What PromptABI adds |
| --- | --- | --- |
| Prompt linters | textual heuristics over prompts | finite artifacts, source spans, witnesses, and explicit proof/abstention modes |
| Schema validators | final JSON/object shape | tokenizer x grammar emptiness, ambiguity, parser compatibility, and stop overreachability before inference |
| Constrained decoders | runtime token masking | offline checks that the declared tokenizer, grammar backend, parser, stop policy, and provider envelope agree |
| Tokenizer diff tools | revision changes | end-to-end contract drift across templates, stops, tools, providers, budgets, and lockfiles |
| Generic static analyzers | application code smells | LLM-interface contracts over chat templates, tool calls, RAG chunks, provider fixtures, and training manifests |
