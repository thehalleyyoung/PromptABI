# Maintainer health

PromptABI treats project maintenance as part of the verification surface: release
claims are only useful if maintainers rotate responsibilities, triage regressions
quickly, review corpus changes consistently, and track long-term health without
collecting private prompts.

## Rotation roles

### rotation-release-captain: Release captain

The release captain owns each release train from branch cut through publication.
They run `promptabi governance --format text`, `promptabi maintain health --format
text`, release readiness, version gates, and final release-blocker review before a
tag is cut. The release readiness report and governance report are saved or
linked from the release notes, along with the maintainer health report.

### rotation-corpus-steward: Corpus steward

The corpus steward reviews fixture provenance, license evidence, expected
diagnostics, sanitization notes, and annual corpus refresh work. They make sure
new model, provider, grammar, framework, training, and eval semantics are added
without deleting longitudinal benchmark coverage. Their review evidence includes
fixture manifests and the corpus review checklist for each affected corpus or
provider pack.

### rotation-triage-lead: Triage lead

The triage lead keeps incoming reports labeled, confirms reproducibility, routes
contributor-ready work, and escalates `status: release-blocking` issues when a
bug matches governance blockers or invalidates published claims. Weekly triage notes
should include issue template labels that were applied, changed, or found missing
during review.

## Triage labels

| ID | Label | Use |
| --- | --- | --- |
| `label-status-needs-triage` | `status: needs-triage` | New reports that need affected surface, severity, owner, and repro status. |
| `label-status-release-blocking` | `status: release-blocking` | Unsound safe results, missing abstentions, witness replay failures, secret-bearing fixtures, license blockers, or `regression-on-labeled-real-bug`. |
| `label-priority-high` | `priority: high` | Correctness, privacy, compatibility, or release risks that preempt roadmap work. |
| `label-area-corpus` | `area: corpus` | Fixture, benchmark, provenance, annual corpus refresh, and expected diagnostics work. |

The label set also keeps `area: checker` and `type: bug` available so issue
templates and maintainer rotations can distinguish checker defects from corpus
updates.

## Release checklist

### release-governance-gate: Governance gate

Every release candidate runs both governance validators:

```bash
promptabi governance --format text
promptabi maintain health --format text
```

### release-corpus-gate: Corpus gate

Affected corpus and conformance suites must run before the release. When fixtures
or expected diagnostics change, maintainers refresh auditable release artifacts:

```bash
promptabi corpus verify --format text
promptabi maintain refresh --output-dir maintainer_artifact --force
```

### release-privacy-gate: Privacy gate

The release captain confirms that artifacts, witnesses, bug reports, and fixture
packs contain no secret-bearing-fixture data, private prompts, or raw customer
content. Witness privacy modes and sanitized bug reports are required for public
evidence.

## Corpus review

### corpus-provenance-review: Provenance review

Every corpus addition records license, provenance, source revision or generator,
expected diagnostics, no secrets status, and a short reason the fixture exercises
a structural interface contract rather than model semantics.

### corpus-regression-review: Regression review

Corpus changes must preserve labeled bug coverage unless the maintainer snapshot
intentionally updates the baseline. `regression-on-labeled-real-bug` and
`witness-replay-failure` are release blockers.

### corpus-refresh-cadence: Refresh cadence

The project performs at least one annual corpus refresh to add newly common model
families, provider semantics, framework truncation behavior, and structured-output
libraries while preserving old fixtures for longitudinal benchmark comparisons.

## Long-term project health metrics

`promptabi maintain health --format json` reports deterministic local metrics:
documentation coverage, test-file count, triage label count, fixture JSON count,
and verification config count. These metrics are intentionally lightweight; full
precision/recall, runtime, witness-quality, and corpus replay evidence remain in
the dedicated corpus, leaderboard, release-readiness, and maintainer refresh
commands.
