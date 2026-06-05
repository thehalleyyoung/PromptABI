# Community contribution workflows

PromptABI accepts executable evidence, not screenshots. The canonical workflow
manifest is available with:

```bash
promptabi contribute workflows --format text
promptabi contribute validate --format text
```

## Accepted lanes

| Lane | Submit | Prove |
| --- | --- | --- |
| Sanitized bug fixtures | Minimal configs, artifacts, expected diagnostics, upstream links, and license/provenance notes | The finding is structural, offline-replayable, minimized, and free of secrets |
| Minimized witnesses | `promptabi minimize` cases, preserved oracles, shrunk traces, source spans, and fingerprints | The reduced witness still triggers the same rule without private prompt payloads |
| Prompt-pack metadata | Prompt-pack JSON, locks, registry entries, provenance, mirrors, and upgrade reports | The pack declares role/tool/stop/RAG/truncation compatibility and privacy behavior |
| Provider fixtures | Offline request/response shapes for tools, streams, JSON mode, stops, errors, and context limits | The fixture records exact provider/API semantics without live credentials or user data |
| Training-manifest adapters | Loader fixtures, manifests, redaction policies, shard sidecars, loss masks, and tokenizer pins | `promptabi verify-training` can replay the adapter without raw private training rows |

## Required review evidence

Every submission should include:

1. Provenance, license, exact revision or hash, and whether the artifact is public,
   synthetic, derived, or privately sanitized.
2. The smallest replayable fixture or witness that preserves the structural
   behavior under review.
3. Expected rule IDs, diagnostic fingerprints, conformance status, or upgrade
   impact.
4. Commands that run the relevant PromptABI check plus
   `promptabi contribute validate --format text`.
5. A privacy statement covering secrets, credentials, customer data, private
   prompts, provider account identifiers, and hash-only or bounded witnesses.

Use the dedicated GitHub templates for prompt-pack metadata, provider fixtures,
training-manifest adapters, minimized witnesses, and sanitized corpus fixtures so
maintainers can route reviews without asking for the same metadata twice.
