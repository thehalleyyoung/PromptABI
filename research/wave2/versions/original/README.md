# PromptABI

PromptABI is a static checker for the interface layer of LLM applications, which
covers chat templates, tokenizers, special tokens, stop sequences, and grammars. Its
main check answers one question about a model's chat template and tokenizer:
**can text that a user types become a role boundary?** If a template
interpolates message content verbatim and the tokenizer maps a marker such as
`<|im_start|>assistant` in that content to the template's own token id, a user
can reproduce a role delimiter. PromptABI reports candidate fields and
markers, the template line, and the marker's token
ids, without loading weights or rendering prompts.

## Install

```bash
python -m pip install -e ".[templates]"          # taint analysis + tokenizer composition
python -m pip install -e ".[dev,solver,tokenizers,templates]"   # full test suite
```

Python 3.11 or newer. The core package has no required dependencies.

## Usage

Scan a Hugging Face model directory (the files `tokenizer_config.json`,
`chat_template.jinja` or `chat_template.json`, and `tokenizer.json`):

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct tokenizer_config.json tokenizer.json --local-dir qwen
promptabi template-scan qwen
```

```text
PromptABI template-scan: qwen
  messages[*].content (roles user, line 23) can render '</tool_call>' [control-token [151658]]
  messages[*].content (roles user, line 23) can render '</tools>' [text [522, 15918, 29]]
  messages[*].content (roles user, line 23) can render '<tool_call>' [control-token [151657]]
  messages[*].content (roles user, line 23) can render '<|im_start|>' [control-token [151644]]
  ...
```

`control-token` means the marker encodes to one added token, uses
`single_word=False`, and has no longer raw-string overlapping added token.
The rendering oracle checks token agreement in the tested contexts. `text` means the marker is ordinary text whose ids depend on the
neighboring characters. The command exits 1 when user content can render a
control-token marker (`--fail-on any|never` changes this) and prints JSON with
`--format json`.

The bounded symbolic version of the check, with explicit render-path
witnesses, runs as the `role-boundary-nonforgeability` rule of the config-driven
verifier, which also emits SARIF:

```bash
promptabi verify --config examples/role-boundary/unsafe.promptabi.json --fail-on never
promptabi explain --config examples/role-boundary/unsafe.promptabi.json --index 1
```

`docs/checks.md` lists the other rules (stop-sequence reachability, grammar
emptiness under a tokenizer, tool-schema and token-budget checks).

## Results on the Hugging Face Hub

We ran the check on the 100 most-downloaded `text-generation` repositories that
ship a chat template and `tokenizer.json` (snapshot 2026-10-07; 51 distinct
templates) and compared every verdict with an independent oracle that renders
the template with `transformers` and tokenizes with the model's tokenizer.
Numbers below are from `results/hub_scan/summary.json`, using study analyzer
revision `4131479`. The synthetic controls below evaluate the revised checker.

| Measure | Result |
| --- | --- |
| Models where one user message reproduces a genuine assistant turn token for token (oracle) | 100 / 100 (51 / 51 templates) |
| Models flagged by the taint analysis | 100 / 100 |
| Models with a symbolic-execution witness (fragment covers the template) | 31 / 100 |
| Control-token markers matched to genuine rendered occurrences | 281 / 281 |
| Additional control-token markers reproduced in user content without a genuine reference occurrence | 70 |
| Plain-text markers confirmed by rendering (excluding 65 with no reference occurrence) | 371 / 398 |
| Taint analysis time per template, median | 17 ms |

The same study evaluates a serving-side fix that tokenizes message content
separately from the template text. With `split_special_tokens=True` alone the
forged separator's control tokens are removed in 21 of 51 templates; in 26
templates some separator tokens are registered as non-special added tokens
(for example `<think>`, and DeepSeek-V3/R1's role tokens), which that option
does not split. Escaping every added token in content removes the forged
control tokens in all 47 templates whose separator contains one, and leaves
benign token sequences unchanged in 47 of 51 templates. The 4 templates with
plain-text role headers (`[/INST]`, `Assistant:`) need a template-side change.

The paper in `paper/tool_paper.tex` describes the analyses and the study.

A fixed synthetic sanitizer stress test found that single-pass replacement can
reconstruct role markers. The taint analysis therefore preserves candidate
findings through `replace`; [raw diagnostics](results/audit/stress-after.json)
record five reconstruction cases and added-token context checks.

## Reproduce

See [REPRODUCE.md](REPRODUCE.md) for details and the
[air-gapped installation guide](docs/air-gapped.md) for offline use. The main steps are

```bash
pip install -e ".[dev,solver,tokenizers,study]"
python study/hub_scan/fetch_models.py --pinned results/hub_scan/models.json   # pinned shas
python study/hub_scan/scan.py && python study/hub_scan/scan_taint.py
python study/hub_scan/oracle.py && python study/hub_scan/mitigation.py
python study/hub_scan/summarize.py && python paper/gen_numbers.py
```

The study needs no GPU and no paid API. The synthetic regression corpus in
`benchmarks/scaled_eval_repro.py --check-golden` (10,080 generated
configurations, 63 distinct analyzer runs) is a regression gate; it takes about
7.5 s of CPU time (38 to 44 s wall-clock on our shared, loaded host;
`results/timing/repro_timing.json`).

## Scope

The check is structural. It establishes that the model receives a forged turn,
not how the model responds to one. The taint analysis is a may-analysis over the
Jinja AST and is not path-sensitive; the dynamic oracle in
`study/hub_scan/oracle.py` is the reference for whether a finding materializes.

## Cite

```bibtex
@misc{promptabi2026,
  title  = {Static Detection of Role-Boundary Forgery in {LLM} Chat Templates},
  author = {{PromptABI contributors}},
  year   = {2026},
  note   = {Preprint, \url{https://github.com/thehalleyyoung/PromptABI}}
}
```

Apache-2.0.
