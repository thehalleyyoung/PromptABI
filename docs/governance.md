# Governance

PromptABI's governance exists to keep structural verification claims honest as
the project grows. Maintainers accept new checks, corpus entries, and release
artifacts only when the evidence is deterministic, CPU-only, privacy-preserving,
and explicit about the boundary between proof, bounded evidence, heuristics, and
abstention.

## checker-acceptance: Checker acceptance criteria

A checker proposal must name the exact structural property it verifies, the
artifacts it consumes, source spans used in diagnostics, its `CheckMode`, the
supported fragment, and the cases that must abstain. Acceptance requires typed
code, deterministic text/JSON diagnostics, focused tests, and fixtures for safe,
unsafe, ambiguous, unsupported, and malformed inputs where those states apply.
It is release-blocking if a checker emits a safe result outside its supported
fragment or silently passes malformed artifacts.

## proof-standards: Proof and evidence standards

Sound or complete claims require replayable evidence. Depending on the check,
that means executable witness traces, automata/product replay, theorem-to-test
traceability, property tests, Z3 solver replays, or differential fixtures against
real tokenizer, provider, grammar, schema, or framework behavior. It is
release-blocking when a counterexample witness cannot be replayed or a central
proof claim lacks an executable regression.

## corpus-licensing: Corpus licensing and provenance

Corpus fixtures are executable evidence. Every entry must record provenance,
license, exact revision or generator, expected diagnostics, and a reason the case
models a structural interface failure rather than semantic model behavior.
Fixtures must be minimized, replayable offline, and contain no secrets, private
prompts, customer data, credentials, or restricted metadata. Missing license
provenance, secret-bearing fixtures, or non-replayable corpus entries are
release-blocking.

## security-disclosure: Security disclosure workflow

Sensitive structural vulnerabilities are coordinated privately first. Public
issues and upstream reports should use `promptabi explain`, `promptabi minimize`,
and `promptabi bug-report` to share only sanitized artifacts, hashes, bounded
excerpts, versions, and non-sensitive witness traces. PromptABI vulnerabilities
should use the private security advisory path when available. A release is
blocked by leaked secrets, raw private witness text, missing owners for sensitive
reports, or regressions in sanitized disclosure.

## release-regressions: Release-blocking regressions

The canonical blocker IDs are:

| ID | Meaning |
| --- | --- |
| `unsound-safe-result` | A check reports safe when the modeled contract is unsafe or outside the supported fragment. |
| `missing-abstention` | Unsupported artifacts pass silently instead of producing an abstention or explicit diagnostic. |
| `witness-replay-failure` | A reported witness, solver replay, or minimized counterexample no longer reproduces. |
| `secret-bearing-fixture` | A fixture or release artifact contains credentials, private prompts, or sensitive raw witness text. |
| `license-incompatible-corpus` | A corpus entry lacks usable provenance or has incompatible licensing. |
| `regression-on-labeled-real-bug` | A labeled real-world bug or benchmark fixture stops being detected without an intentional baseline update. |

Maintainers can inspect the executable governance manifest with:

```bash
promptabi governance --format text
```
