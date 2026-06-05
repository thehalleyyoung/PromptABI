# Demo GIF source script

Record these deterministic, CPU-only commands into a terminal GIF for launch pages:

```bash
python -m pip install -e ".[dev,grammars,solver,tokenizers]"
promptabi verify --config examples/role-boundary/unsafe.promptabi.json --fail-on never
promptabi explain --config examples/role-boundary/unsafe.promptabi.json --index 1
promptabi corpus real-bug-benchmark --output /tmp/promptabi-real-bugs.json
promptabi corpus evaluation --format text
promptabi launch-assets --output-dir launch_assets --force
```

The accompanying placeholder `demo.gif` is valid GIF89a; regenerate it from this script for release. The evidence behind the script currently reports 7 real-bug cases and 11 labeled evaluation cases.
