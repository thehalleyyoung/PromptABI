# Tool-calling boundary

The buggy app assumes OpenAI-compatible tool calls arrive as parsed JSON objects,
single calls, and known tool names. The recorded provider fixture shows a real
boundary drift: JSON-string arguments, parallel calls, streaming fragments, a
missing tool-call ID, and an unexpected `cancel_order` call.

```bash
promptabi verify --config examples/end-to-end/tool-calling/buggy.promptabi.json
promptabi verify --config examples/end-to-end/tool-calling/fixed.promptabi.json
```

