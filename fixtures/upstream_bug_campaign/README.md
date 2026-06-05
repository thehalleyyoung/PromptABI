# Upstream interface-safety bug campaign dossier

This directory backs `promptabi.upstream_bug_campaign`, the auditable workflow for
discovering, triaging, and responsibly reporting a *new* interface-safety bug in a
real upstream project.

## Contents

- `campaign.json` — the dossier: scope definitions, selected upstream targets,
  pinned scanned sources, prompt-facing inventories, and triaged candidate
  findings (each with duplicate search, reproduction plan, disclosure routing,
  and—where reportable—a drafted report).
- `sources/` — exact upstream source captured from upstream HEAD at a pinned
  commit SHA. Each file's SHA-256 is recorded in `campaign.json` and re-verified
  by the engine on every run.

## Provenance

| source | repository | commit | path |
| --- | --- | --- | --- |
| `vllm_deepseek_r1_reasoning` | vllm-project/vllm | `fefce498…` | `vllm/reasoning/deepseek_r1_reasoning_parser.py` |
| `vllm_hermes_tool_template` | vllm-project/vllm | `41e95c52…` | `examples/tool_chat_template_hermes.jinja` |

All captured files are Apache-2.0 licensed and reproduced for analysis under their
original license headers.

## Honest triage outcomes

The campaign is conservative: a source-pattern flag is only a *candidate*. Each
candidate is routed through deterministic PromptABI analysis of the exact pinned
source before any claim is made.

- **rejected** — `vllm_deepseek_reasoning_tokenid`: the DeepSeek-R1 streaming
  parser decides the reasoning/content transition from token-id membership, but
  `<think>`/`</think>` are atomic added tokens for this model, so the boundary
  cannot be forged. Not triggerable.
- **abstained** — `vllm_hermes_tool_template_abstain`: the Hermes tool template
  uses Jinja macros outside the supported decidable fragment, so no soundness
  claim is made.
- **duplicate** — `vllm_qwen_multi_function_block_dup`: the streaming tool-call
  index-reuse symptom is already tracked upstream; the campaign contributes a
  deterministic regression test to the existing issue instead of filing a
  duplicate.

No new confirmed bug is claimed from this scan — an honest, auditable result. The
`confirmed_detection_references` in the run output replay public, already-confirmed
exemplars (e.g. llama.cpp PR #18462) to demonstrate the analyzer reproduces real
interface-safety bugs.
