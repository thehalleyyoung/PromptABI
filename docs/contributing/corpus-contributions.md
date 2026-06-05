# Corpus contribution rules

PromptABI corpora are executable evidence, not screenshots. Corpus additions
must be reproducible, minimized, licensed, and safe to replay offline.

## Required metadata

Every fixture pack should include provenance, license, exact revision or hash,
expected diagnostics, and a short explanation of why the case matters. If the
source is derived from a public project, record the upstream path and revision;
if it is synthetic, state which real failure mode it models.

## Privacy and safety

Corpus contributions must contain no secrets, private prompts, customer data,
credentials, restricted metadata, or live provider tokens. Witnesses should be
minimized and sanitized before submission. Provider fixtures should use recorded
offline shapes and non-sensitive payloads only.

## Validation

Run the relevant corpus command plus `promptabi contribute validate` before
opening a pull request. New real-bug or evaluation cases should include expected
rule IDs and a reason that the finding is not just a model-semantics claim.
