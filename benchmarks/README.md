# Benchmarks

Benchmarks are CPU-only, deterministic, and run against real PromptABI code
paths plus the checked-in examples/fixtures. The suite covers tokenizer
analysis, bounded chat-template symbolic execution, tokenizer x grammar
emptiness, stop tokenizer/overreach checks, SMT-backed static contracts,
token-budget modeling, corpus-wide verification, and cold/warm cache behavior.

```bash
PYTHONPATH=src python benchmarks/benchmark_smoke.py
PYTHONPATH=src python benchmarks/benchmark_smoke.py grammar-emptiness stop-checks --iterations 5
```
