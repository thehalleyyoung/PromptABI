# Security model

PromptABI is a local, CPU-only verifier for the discrete interface layer around
LLM systems. Its security value is structural: it proves or refutes whether the
artifacts that surround a model can represent unsafe protocol states before a
deployment sends prompts to a provider or runs inference.

## Assets and trust boundary

PromptABI treats these inputs as sensitive local artifacts:

- chat templates, tokenizer configs, special-token maps, stop policies, JSON
  Schemas, grammars, tool definitions, provider fixtures, prompt-budget policies,
  training manifests, lockfiles, suppressions, and verification configs;
- rendered prompt excerpts, witness strings, parser states, solver assignments,
  source spans, artifact paths, and diagnostic fingerprints that may reveal
  application structure.

The trust boundary is the local machine or CI runner executing `promptabi`.
Verification, explanation, minimization, report generation, corpus replay, local
usage summaries, and paper reproducibility commands operate on local files and
recorded fixtures. PromptABI does not require model weights, GPUs, live provider
credentials, or network calls for its core claims.

## What PromptABI is designed to find

PromptABI checks structural prompt-interface risks, including:

- attacker-controlled text rendering as role delimiters, assistant prefixes,
  tool-call sentinels, BOS/EOS markers, or provider/model control tokens;
- stop strings that are unreachable, ambiguous, tokenizer-sensitive, or able to
  fire inside otherwise valid JSON, XML-like tool calls, markdown fences, or
  provider envelopes;
- grammar, schema, tokenizer, tool-call, provider-migration, parser, and
  truncation contracts that disagree before any model is queried;
- must-survive prompt segments, tool definitions, safety preambles, citations, or
  output-format instructions that can be dropped by real framework truncation
  policies;
- unsafe provider fixture packs, corpus entries, or bug-report witnesses that
  contain credential-like values or excessive raw witness text.

Diagnostics explicitly mark whether a result is sound, complete, bounded,
Z3-backed SMT, heuristic, or abstaining. That mode is part of the security claim:
an abstention is a refusal to overclaim outside the modeled fragment.

## Non-goals

PromptABI does not prove semantic model behavior. It cannot guarantee that a
model will follow an instruction, resist prompt injection in the behavioral
sense, choose or avoid a token, be unbiased, or produce harmless content. It also
does not replace secrets scanning, dependency vulnerability scanning, sandboxing,
provider-side authorization, output moderation, or runtime policy enforcement.

The central guarantee is narrower and stronger: under the supported artifact
fragment, PromptABI can show that a discrete protocol state is possible,
impossible, ambiguous, unsafe, or outside the fragment before inference.

## Secret handling and artifact privacy

PromptABI is designed to fail closed around secrets:

- provider fixture packs are validated offline and rejected when metadata or
  recorded request/response fixtures contain credential-like keys or values such
  as API keys, authorization headers, bearer tokens, cloud access keys, passwords,
  or provider key patterns;
- `promptabi bug-report` emits sanitized markdown with length-limited witness
  traces and a privacy note; it does not upload issues or transmit artifacts;
- `promptabi minimize` shrinks local repros under local oracles and writes only
  where explicitly requested;
- SARIF, JSON, HTML, and text outputs are local renderings of diagnostics and
  should be treated as sensitive build artifacts when they contain source spans,
  witnesses, or repository paths;
- lockfiles pin hashes, revisions, supported fragments, library versions,
  provider fixture versions, and diagnostic baselines rather than embedding raw
  prompts.

Users should not commit real secrets, private prompts, proprietary schemas, or
unredacted provider traces to public fixtures. If a witness must be shared
upstream, generate a minimized and sanitized repro first.

## Solver-input privacy

Z3-backed and finite-enumeration checks run locally. Solver formulas may encode
bounded strings, lengths, enum memberships, role regions, stop reachability,
grammar states, tool-schema constraints, truncation decisions, and source
locations derived from local artifacts. PromptABI does not send these formulas to a remote solver.
Teams that require stronger isolation can run PromptABI inside
their normal CI sandbox or offline runner and treat solver logs, diagnostics, and
HTML reports as sensitive outputs.

## Provider fixture safety

Provider fixtures are snapshots of request/response shapes, tool-call encodings,
streaming deltas, stop behavior, error envelopes, limits, and edge cases. They
are intended to be secret-free, anonymized, deterministic, and replayable without
network access. The fixture-pack loader validates required metadata, rejects
download-required packs, rejects secrets, hashes fixture files, and records
whether entries are anonymized and license-compatible.

## Local usage summaries are not telemetry

`--local-summary` and `promptabi usage summary` are opt-in local analytics. The
records contain command names, exit codes, durations, and aggregate diagnostic
counts only. The implementation explicitly excludes prompts, schemas, configs,
constraints, witnesses, artifact contents, file paths, and network sends.
`promptabi usage privacy` prints the same guarantee from the CLI.

## Suppressions and accepted risk

Suppressions are policy records, not erasers. PromptABI policy files require
owners, justifications, accepted-risk statements, expiration dates, stable
fingerprints, and `witness_digest` proofs that the accepted counterexample is
unchanged, while severity thresholds keep CI strict. Expired, unjustified, or
drifted suppressions are reported as security debt.

## Responsible disclosure workflow

If PromptABI finds a structural vulnerability in a third-party tokenizer,
template, framework, provider adapter, schema library, or prompt pack:

1. Reproduce it locally with `promptabi verify` and expand the diagnostic with
   `promptabi explain`.
2. Minimize the artifact with `promptabi minimize` when a smaller repro is
   possible without losing the failing rule.
3. Generate a sanitized upstream report with `promptabi bug-report`; review the
   markdown for private names, paths, prompts, or business logic before sharing.
4. Share only the minimized fixture, artifact versions, hashes, expected/actual
   behavior, and non-sensitive witness trace needed for the maintainer to
   reproduce the structural bug.
5. Coordinate privately first when the issue could expose users to credential
   leakage, cross-tenant data exposure, unsafe tool execution, or widely deployed
   prompt-interface bypasses.

PromptABI itself should receive vulnerability reports through the repository's
private security advisory channel when available, or by a private maintainer
contact listed by the project. Public issues are appropriate for non-sensitive
false positives, supported-fragment disagreements, documentation problems, or
sanitized fixture improvements.

Maintainers treat secret-bearing fixtures, raw private witness text, missing
owners for sensitive reports, and regressions in the sanitized disclosure flow as
release-blocking security issues.
