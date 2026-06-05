# Training-data alignment boundary

The buggy fine-tuning manifest says supervised target spans may be labeled
`critic`, but the serving chat template only renders `system`, `user`, and
`assistant` regions. The fixed manifest restricts targets to the assistant
region the serving template can actually represent.

```bash
promptabi verify --config examples/end-to-end/training-alignment/buggy.promptabi.json
promptabi verify --config examples/end-to-end/training-alignment/fixed.promptabi.json
```

