# Proof sketches

PromptABI separates a diagnostic from the proof obligations behind it. The
`promptabi.proof_sketches` module turns real checker result objects into compact
certificates with a theorem statement, supported-fragment assumptions, lemmas,
evidence, and executable checks.

The command-line catalog is:

```bash
promptabi proofs
promptabi proofs --format json
```

## Supported proof families

| Family | Certificate input | Main obligation |
| --- | --- | --- |
| Role-boundary non-forgeability | `RoleBoundaryNonforgeabilityReport` | Controlled fields must be disjoint from structural role/control markers in the bounded role model. |
| Stop overreachability | `StopOverreachabilityReport` | A stop witness must split a valid output into delivered prefix, matched stop, and required suffix. |
| Grammar emptiness | `GrammarTokenizerEmptinessReport` plus the compiled DFA for executable proofs | A satisfiable witness must replay through the grammar automaton and remain accepted after tokenizer decode. |
| Must-survive budgets | `MustSurviveProof` | Required prompt segments are either all kept, or the dropped/survived sets form a concrete counterexample partition. |
| Z3-backed finite contracts | `StaticContractFinding` | SAT assignments satisfy every finite-domain constraint; UNSAT cores are known, unsatisfiable, and deletion-minimal. |

## Proof versus sketch

Certificates are explicit about outcome:

| Outcome | Meaning |
| --- | --- |
| `proven` | Executable checks discharge the bounded proof obligation. |
| `counterexample` | Executable checks validate a concrete violation witness. |
| `abstained` | The supported fragment, source data, or solver proof object is missing. |
| `sketch` | The theorem and assumptions are documented, but the available report lacks an independent transition relation or proof object. |

For example, grammar reports intentionally do not store the compiled DFA. A
grammar certificate built only from a report is therefore a sketch. Passing the
compiled `DeterministicFiniteAutomaton` lets the certificate replay the witness
with `check_dfa_witness` and recheck that decoded text is accepted.

## Minimal API example

```python
from promptabi import proof_sketches, render_proof_sketch_report_text

report = proof_sketches()
print(render_proof_sketch_report_text(report))
```

For concrete checker results, call `prove_role_boundary_nonforgeability`,
`prove_stop_reachability`, `prove_grammar_emptiness`,
`prove_must_survive_budget`, or `prove_static_contract` directly.
