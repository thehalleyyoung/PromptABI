# Reproducing the results

All commands run from the repository root with Python 3.12. No GPU, paid API,
or cloud compute is used.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,solver,tokenizers,study]"
```

## Hub study (paper Section 6)

The study pins every model to the commit sha recorded in
`results/hub_scan/models.json`; re-running `fetch_models.py` today selects a
newer download ranking, so to reproduce the paper's numbers re-download the
pinned files instead:
`python study/hub_scan/fetch_models.py --pinned results/hub_scan/models.json`
(checks every file against its recorded sha256).

```bash
# 1. Select the 100 most-downloaded text-generation repos that ship a chat
#    template and tokenizer.json; download template/tokenizer files.
#    Gated repos need HF_TOKEN with access; without it they are skipped.
HF_TOKEN=... python study/hub_scan/fetch_models.py --n 100 --scan 500

# 2. Static analyses (no rendering, no transformers import).
python study/hub_scan/scan.py         # bounded symbolic executor -> analyzer.jsonl
python study/hub_scan/scan_taint.py   # taint analysis + tokenizer composition -> taint.jsonl

# 3. Independent dynamic oracle (transformers apply_chat_template + real tokenizer).
python study/hub_scan/oracle.py       # -> oracle.jsonl

# 4. Content-segmented tokenization fix.
python study/hub_scan/mitigation.py   # -> mitigation.jsonl

# 5. Aggregate; every number in the paper's evaluation is read from summary.json.
python study/hub_scan/summarize.py    # -> summary.json, per_model.csv
python paper/gen_numbers.py           # -> paper/numbers.tex
```

Raw outputs are committed under `results/hub_scan/`. Downloaded model files
(`results/hub_scan/cache/`) are not committed; their sha256 digests are in
`models.json`. Snapshot time, library versions (transformers 5.19.0,
tokenizers 0.23.2, jinja2 3.1.6, huggingface_hub 1.33.0) and the host are
recorded in the JSON files.

## Single model

```bash
promptabi template-scan path/to/model_dir            # text report, exit 1 on control-token findings
promptabi template-scan path/to/model_dir --format json
```

## Regression suite timing

`benchmarks/scaled_eval_repro.py --check-golden` replays a synthetic,
construction-labelled regression corpus of 10,080 configurations; analyzer
results are memoized per (template family, sanitizer strategy), so the analyzer
runs 63 times.
It is a regression gate, not an empirical evaluation. Its runtime is measured
by

```bash
python benchmarks/time_repro.py --runs 3   # -> results/timing/repro_timing.json
```

## Paper

```bash
cd paper && latexmk -pdf tool_paper.tex
```

Bibliography entries were checked against arXiv, Crossref, and the GitHub API;
the log is `results/bib/verification_log.txt`.

## Independent audit diagnostics

The fixed synthetic stress and pinned-cache verification run without paid services:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python3 study/audit_stress.py
python3 -m pytest tests/test_template_taint.py
python3 study/audit_prepare.py /path/to/original/PromptABI
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python3 study/hub_scan/scan_taint.py --models results/audit/models-all.json --out /tmp/taint.jsonl
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python3 study/hub_scan/oracle.py --models results/audit/models-subset.json --out /tmp/oracle.jsonl
```

Audit receipts use transformers 5.12.1 and tokenizers 0.22.2; these differ from
the original study versions. See `results/audit/` for original and repaired
synthetic outputs, cache hash comparisons, and representative replication.
