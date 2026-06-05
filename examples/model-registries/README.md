# Model registry publication

PromptABI can publish a signed, reproducible verification evidence payload beside
a model version without uploading prompts, datasets, provider credentials, or
model weights. The example target manifest covers Hugging Face Hub model cards,
MLflow-style registered models, internal registries, and generic artifact
repositories:

```bash
promptabi model-registry \
  --config examples/model-registries/promptabi.json \
  --targets examples/model-registries/targets.json \
  --bundle-key local-registry-key \
  --format json
```

The emitted JSON contains the model-registry integration surface, signed bundle
hash, reproducibility hash, artifact metadata, and per-registry publication
commands. A registry gate should require `ok: true` before promoting a model.
