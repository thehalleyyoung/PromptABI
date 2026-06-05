# Dynamic agent framework prompt-pack examples

These examples show how a LangChain/LlamaIndex-style agent can assemble prompts
at runtime from a reusable prompt pack while still emitting a deterministic
PromptABI contract before deployment.

```bash
python examples/agent-frameworks/dynamic_support_agent.py \
  examples/agent-frameworks/safe.agent-prompt-pack.json \
  --write-config /tmp/support-agent.promptabi.json

promptabi verify --config /tmp/support-agent.promptabi.json
```

The safe spec wires the prompt pack's required roles, tool, stop sequence, and
provider family. The buggy spec intentionally drifts those runtime choices so
`promptabi verify` reports the exact missing tool, missing stop, and unsupported
provider family.
