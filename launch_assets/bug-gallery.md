# PromptABI bug gallery

All 7 replayed reductions passed against local PromptABI analyzers.

| Case | Category | Bug class | Evidence |
| --- | --- | --- | --- |
| Tokenizer special/added-token expectation drift | tokenizer | normalization plus added special tokens invalidates a pinned tokenizer expectation | 4 tokenizer differential mismatch(es) |
| Structured-output parser/schema mismatch | structured-output-library | application parser accepts outputs outside the constrained structured-output contract | open-source-agent-ticket replayed with parser status mismatch |
| OpenAI-compatible provider migration contract regressions | provider-migration | provider migration can lose tool IDs, streaming chunks, stop semantics, context limits, and structured-output fields | 58 diagnostic(s) from fixtures/provider_migration/promptabi.json |
| Phi-style chat template role delimiter forgery | popular-template | attacker-controlled role/content fields can render model control delimiters | 18 role-boundary witness(es) for phi_system_turn_forgery |
| Qwen XML-like tool parameter stop overreach | tool-schema | tool argument strings can contain configured XML close-marker stops | 2/2 stop-overreach witness(es) matched for qwen3_xml_tool_parameter_stop |
| RAG chunk metadata and citation truncation | rag-truncation | RAG chunk packing can drift tokenizers, drop citations, miscount overlap, and exceed payload limits | 11 diagnostic(s) from examples/rag-chunking/promptabi.json |
| Supervised fine-tuning target role misalignment | training-pipeline | training manifest target roles include a role absent from the serving chat template | 1 static-contract violation(s), including training-target-role-alignment |
