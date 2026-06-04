# Real-world bug corpus

This corpus records public GitHub bug patterns as minimized, synthetic fixtures.
It does not copy upstream source code; each entry keeps only the public reference,
the interface failure class, and a small repro artifact that PromptABI can verify
offline.

The first entries cover bugs from `ggml-org/llama.cpp` and
`vllm-project/vllm`:

| Fixture | Public reference | PromptABI check |
| --- | --- | --- |
| `phi_system_turn_forgery` | <https://github.com/ggml-org/llama.cpp/pull/18462> | `role-boundary-nonforgeability` |
| `qwen3_xml_tool_parameter_stop` | <https://github.com/vllm-project/vllm/pull/40861> | `stop-overreachability` |
| `gemma4_quote_sentinel_truncation` | <https://github.com/vllm-project/vllm/issues/39069> | `stop-overreachability` |
| `llama_cpp_tool_call_parser_boundary` | <https://github.com/ggml-org/llama.cpp/pull/20660> | `stop-overreachability` |

The tests in `tests/test_real_world_bugs.py` load `corpus.json` and assert that
the current detectors emit concrete witnesses for every labeled bug pattern.
