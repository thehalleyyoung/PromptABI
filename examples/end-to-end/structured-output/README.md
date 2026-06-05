# JSON structured-output boundary

The buggy app only calls `json.loads`, so any JSON object is accepted even
though constrained decoding was configured with a stricter schema. The fixed
contract records that the application parser validates the same JSON Schema.

```bash
promptabi verify --config examples/end-to-end/structured-output/buggy.promptabi.json
promptabi verify --config examples/end-to-end/structured-output/fixed.promptabi.json
```

