# Changelog

PromptABI follows semantic versioning. Release notes are generated from GitHub
releases; this file keeps the high-level human contract for notable changes.

## Unreleased

- New capability: `promptabi scan-parser-source <file.py>` (and
  `promptabi.scan_parser_source`) statically scans real serving-stack parser
  source for tool-call **boundary-confusion** candidates — code that locates a
  tool-call boundary by naively splitting/searching for a closing sentinel
  (`</tool_call>`, `</function_calls>`, `[/TOOL_CALLS]`, ...) or capturing
  arguments with a greedy regex over a buffer that also holds attacker-influenced
  JSON `arguments`. Findings are honest heuristic *candidates*; when PromptABI's
  streaming JSON boundary parser can place the sentinel inside a protected
  JSON-string state, the candidate carries a concrete `bounded` witness.
  Reasoning chain-of-thought delimiters are excluded to stay high-precision.
  Running it across 46 current vLLM tool parsers (412 functions) surfaced one
  focused candidate (`olmo3` greedy `<function_calls>(.*?)</function_calls>`
  capture).
- Certified verification layer (`promptabi.certified`, `promptabi certify`): a
  small trusted proof kernel independently re-checks machine-checkable proof
  certificates for seven finite theorems (role-boundary non-forgeability,
  `ByteLevelTokenizer` round-trip injectivity, stop-policy totality, token-budget
  arithmetic, abstract-interpretation soundness, the JSON-Schema decision
  procedure, and multi-agent handoff non-confusion) without trusting the
  production analyzer or Z3. Adds proof-carrying diagnostics, a proof-regression
  CI gate, `--certified` family gating, a trusted-computing-base audit, a
  formal-semantics technical report, and extracted OCaml/Rust kernels.
- Robustness fix: the chat-template symbolic executor no longer crashes with
  `symbolic segment value must be non-empty` on real-world templates that
  initialize an accumulator with an empty-string literal (e.g.
  `{% set content = '' %}`, as used by Qwen3, QwQ, and many tool-calling
  templates). Empty-string literals are now treated as legitimate
  render-nothing segments, so `analyze_role_boundary_nonforgeability` and the
  rest of the template pipeline analyze (or abstain on) these templates instead
  of aborting. Surfaced by running PromptABI against real Hugging Face
  `tokenizer_config.json` artifacts.
- Bug fix: `promptabi.__all__` exported the name `PackExample` twice (two
  distinct dataclasses from `prompt_pack_differential_fixtures` and
  `prompt_pack_example_certification`), which shadowed one class and broke
  `build_public_api_manifest()` (the public-API stability gate). The
  differential-fixtures class is now exported as `DifferentialPackExample`, so
  both classes are reachable and the manifest builds.

## 1.0.0

- Stable 1.0 release with a release-readiness gate that verifies package
  metadata, the stable CLI, GitHub Action, docs, seed corpus, formal/Z3-backed
  checks, real-bug benchmark, paper preprint, and reproducible evaluation bundle
  against live repository code.

## 0.1.0

- Initial pre-alpha package with static verification for tokenizer, template,
  tool-calling, grammar, provider, and prompt-interface contracts.
