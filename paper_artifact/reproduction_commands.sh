#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -e .
python -m pip install z3-solver==4.15.4
python -m promptabi.benchmarks all --iterations 1 --repo-root . > ${PROMPTABI_BENCHMARK_JSON:-benchmark-results.json}
promptabi corpus manifest --output seed-corpus-manifest.json
promptabi corpus structured-schema-manifest --output structured-schema-manifest.json
promptabi corpus provider-fixture-manifest --output provider-fixture-manifest.json
promptabi corpus real-bug-benchmark --output real-bug-benchmark-manifest.json
promptabi corpus evaluation --format json --output evaluation-report.json
promptabi paper reproducibility --output-dir paper_artifact --benchmark-iterations 1 --force
