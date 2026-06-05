# Real-world bug corpus

This corpus records public GitHub bug patterns with deterministic offline replay.
The benchmarked template/parser cases in `production_code.json` include exact
pinned upstream source excerpts, SHA-256 hashes, repository refs, file paths, and
licenses. PromptABI traverses those bytes to extract the template or parser
delimiters before running the local analyzers; replay fails if the expected
boundary is not source-derived.

`corpus.json` still carries minimized local payload models (tool schemas, JSON
schemas, and expected witnesses) so the analyzers can exercise the vulnerable
interface boundary without depending on private data or a network call.

The entries cover bugs from `ggml-org/llama.cpp`, `vllm-project/vllm`, and
`huggingface/transformers`:

| Fixture | Public reference | PromptABI check |
| --- | --- | --- |
| `phi_system_turn_forgery` | <https://github.com/ggml-org/llama.cpp/pull/18462> | `role-boundary-nonforgeability` |
| `hf_apply_chat_template_special_token_injection` | <https://github.com/huggingface/transformers/issues/29279> | `role-boundary-nonforgeability` |
| `qwen3_xml_tool_parameter_stop` | <https://github.com/vllm-project/vllm/pull/40861> | `stop-overreachability` |
| `vllm_streaming_stop_interrupts_tool_parser` | <https://github.com/vllm-project/vllm/issues/42210> | `stop-overreachability` |
| `llama_cpp_qwen_array_object_tool_leak` | <https://github.com/ggml-org/llama.cpp/issues/21771> | `stop-overreachability` |
| `vllm_qwen_multi_function_block_boundary` | <https://github.com/vllm-project/vllm/issues/43713> | `stop-overreachability` |
| `gemma4_quote_sentinel_truncation` | <https://github.com/vllm-project/vllm/issues/39069> | `stop-overreachability` |
| `llama_cpp_tool_call_parser_boundary` | <https://github.com/ggml-org/llama.cpp/pull/20660> | `stop-overreachability` |

The tests in `tests/test_real_world_bugs.py` load both `production_code.json`
and `corpus.json`, asserting that the production-code cases replay through exact
source excerpts and that the current detectors emit concrete witnesses for every
labeled bug pattern.
`fixtures/real_bug_benchmarks/benchmark.json` reuses these reductions alongside
tokenizer, provider-migration, structured-output, RAG, and training-pipeline
cases to form the broader replayable real-bug benchmark suite.
