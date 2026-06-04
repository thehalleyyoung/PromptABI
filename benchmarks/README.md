# Benchmarks

Benchmarks are CPU-only and deterministic. The initial smoke benchmark measures
the overhead of loading the minimal config and running the skeleton verifier.
Future benchmarks will cover tokenizer abstraction, chat-template execution,
automata products, stop reachability, grammar emptiness, and budget checks.

```bash
python benchmarks/benchmark_smoke.py
```

