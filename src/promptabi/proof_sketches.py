"""Lightweight proof sketches for PromptABI's supported finite fragments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from .budgets import MustSurviveProof
from .formal import AutomatonWitness, DeterministicFiniteAutomaton, SolverStatus
from .grammar_emptiness import GrammarTokenizerEmptinessReport, GrammarTokenizerEmptinessStatus
from .role_boundaries import RoleBoundaryNonforgeabilityReport
from .specs import SpecCheck, check_contract_result, check_dfa_witness
from .static_contracts import StaticContractFinding
from .stop_overreachability import StopOverreachabilityReport


class ProofOutcome(StrEnum):
    """Proof-sketch outcome for one bounded property."""

    PROVEN = "proven"
    COUNTEREXAMPLE = "counterexample"
    ABSTAINED = "abstained"
    SKETCH = "sketch"


@dataclass(frozen=True, slots=True)
class ProofLemma:
    """One named lemma or reduction used by a proof sketch."""

    name: str
    statement: str
    justification: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "statement": self.statement,
            "justification": self.justification,
        }


@dataclass(frozen=True, slots=True)
class ProofSketch:
    """A concise, executable proof certificate or honestly labeled sketch."""

    property_id: str
    title: str
    theorem: str
    supported_fragment: str
    assumptions: tuple[str, ...]
    outcome: ProofOutcome
    lemmas: tuple[ProofLemma, ...]
    checks: tuple[SpecCheck, ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def proven(self) -> bool:
        return self.outcome is ProofOutcome.PROVEN and self.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "property_id": self.property_id,
            "title": self.title,
            "theorem": self.theorem,
            "supported_fragment": self.supported_fragment,
            "assumptions": list(self.assumptions),
            "outcome": self.outcome.value,
            "passed": self.passed,
            "lemmas": [lemma.to_dict() for lemma in self.lemmas],
            "checks": [check.to_dict() for check in self.checks],
            "evidence": [{"name": name, "value": value} for name, value in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ProofSketchReport:
    """Collection of proof sketches."""

    sketches: tuple[ProofSketch, ...]

    @property
    def passed(self) -> bool:
        return all(sketch.passed for sketch in self.sketches)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "sketches": [sketch.to_dict() for sketch in self.sketches],
        }


def prove_role_boundary_nonforgeability(report: RoleBoundaryNonforgeabilityReport) -> ProofSketch:
    """Build a proof sketch for a bounded role-boundary report."""

    checks: list[SpecCheck] = [
        SpecCheck("model-supported", report.model.supported, f"abstentions={list(report.model.abstentions)}"),
    ]
    if not report.model.supported:
        return _sketch(
            "role-boundary-nonforgeability",
            ProofOutcome.ABSTAINED,
            checks=tuple(checks),
            evidence=(("abstentions", "; ".join(report.model.abstentions)),),
        )
    if not report.findings:
        return _sketch(
            "role-boundary-nonforgeability",
            ProofOutcome.PROVEN,
            checks=(*checks, SpecCheck("no-forgery-witnesses", True, "bounded model emitted no findings")),
            evidence=(("path_count", str(len(report.model.paths))),),
        )

    for index, finding in enumerate(report.findings):
        checks.extend(
            (
                SpecCheck(
                    f"finding-{index}-malicious-input-is-marker",
                    finding.malicious_input == finding.marker,
                    f"{finding.malicious_input!r} vs {finding.marker!r}",
                ),
                SpecCheck(
                    f"finding-{index}-marker-visible-in-rendered-excerpt",
                    finding.marker in finding.rendered_excerpt,
                    finding.rendered_excerpt,
                ),
                SpecCheck(
                    f"finding-{index}-offset-length-matches-marker",
                    finding.marker_end_offset - finding.marker_start_offset == len(finding.marker),
                    f"offsets={finding.marker_start_offset}:{finding.marker_end_offset} marker_len={len(finding.marker)}",
                ),
                SpecCheck(
                    f"finding-{index}-tokenizer-evidence-mentions-marker",
                    finding.marker in finding.tokenized_representation,
                    finding.tokenized_representation,
                ),
            )
        )
    return _sketch(
        "role-boundary-nonforgeability",
        ProofOutcome.COUNTEREXAMPLE,
        checks=tuple(checks),
        evidence=(
            ("finding_count", str(len(report.findings))),
            ("first_marker", report.findings[0].marker),
            ("first_input_expression", report.findings[0].input_expression),
        ),
    )


def prove_stop_reachability(report: StopOverreachabilityReport) -> ProofSketch:
    """Build a proof sketch for bounded stop-overreachability."""

    checks: list[SpecCheck] = []
    for index, finding in enumerate(report.findings):
        start = finding.firing_offset
        end = start + len(finding.stop_sequence)
        checks.extend(
            (
                SpecCheck(
                    f"finding-{index}-stop-at-firing-offset",
                    finding.valid_output[start:end] == finding.stop_sequence,
                    f"offset={start} stop={finding.stop_sequence!r}",
                ),
                SpecCheck(
                    f"finding-{index}-truncated-prefix-splits-output",
                    finding.truncated_prefix == finding.valid_output[:start],
                    finding.truncated_prefix,
                ),
                SpecCheck(
                    f"finding-{index}-valid-prefix-includes-stop",
                    finding.valid_output_prefix == finding.valid_output[:end],
                    finding.valid_output_prefix,
                ),
                SpecCheck(
                    f"finding-{index}-truncation-removes-required-suffix",
                    finding.truncated_prefix != finding.valid_output,
                    f"removed={len(finding.valid_output) - len(finding.truncated_prefix)}",
                ),
            )
        )
    outcome = ProofOutcome.COUNTEREXAMPLE if report.findings else ProofOutcome.PROVEN
    if not checks:
        checks.append(SpecCheck("no-stop-overreach-witnesses", True, report.bound))
    return _sketch(
        "stop-overreachability",
        outcome,
        checks=tuple(checks),
        evidence=(
            ("stop_policy", report.stop_policy_name),
            ("bound", report.bound),
            ("finding_count", str(len(report.findings))),
            ("abstention_count", str(len(report.abstentions))),
        ),
    )


def prove_grammar_emptiness(
    report: GrammarTokenizerEmptinessReport,
    *,
    automaton: DeterministicFiniteAutomaton | None = None,
) -> ProofSketch:
    """Build a grammar/tokenizer proof sketch.

    A satisfiable report is independently proven only when the caller supplies the
    compiled grammar automaton. Without that transition relation, the stored
    grammar states remain useful evidence but not an executable proof.
    """

    checks: list[SpecCheck] = [
        SpecCheck("status-is-known", report.status in set(GrammarTokenizerEmptinessStatus), report.status.value),
    ]
    outcome = ProofOutcome.SKETCH
    if report.status is GrammarTokenizerEmptinessStatus.ABSTAINED:
        outcome = ProofOutcome.ABSTAINED
        checks.append(SpecCheck("abstention-has-reason", bool(report.reason), report.reason or "<none>"))
    elif report.status is GrammarTokenizerEmptinessStatus.SATISFIABLE:
        checks.append(SpecCheck("satisfiable-has-witness", report.witness is not None, repr(report.witness)))
        if automaton is not None and report.witness is not None:
            witness = AutomatonWitness(
                symbols=tuple(report.witness.grammar_text),
                states=report.witness.grammar_states,
            )
            checks.extend(check_dfa_witness(automaton, witness).checks)
            checks.append(
                SpecCheck(
                    "decoded-text-accepted-by-grammar",
                    automaton.accepts_text(report.witness.decoded_text),
                    report.witness.decoded_text,
                )
            )
            outcome = ProofOutcome.PROVEN
    elif report.status is GrammarTokenizerEmptinessStatus.EMPTY:
        checks.append(SpecCheck("empty-has-no-witness", report.witness is None, repr(report.witness)))
        if automaton is not None and report.reason == "bounded grammar automaton has no accepting path":
            checks.append(
                SpecCheck(
                    "grammar-automaton-has-no-shortest-witness",
                    automaton.shortest_witness() is None,
                    f"states={len(automaton.states)} accepts={len(automaton.accepts)}",
                )
            )
            outcome = ProofOutcome.PROVEN
    return _sketch(
        "grammar-tokenizer-emptiness",
        outcome,
        checks=tuple(checks),
        evidence=(
            ("tokenizer", report.tokenizer_name),
            ("grammar", report.grammar_name),
            ("checked_candidates", str(report.checked_candidates)),
            ("assumptions", ", ".join(report.assumptions)),
        ),
    )


def prove_must_survive_budget(proof: MustSurviveProof) -> ProofSketch:
    """Build a proof sketch for a must-survive truncation proof."""

    required = set(proof.required_segments)
    survived = set(proof.survived_segments)
    dropped = set(proof.dropped_segments)
    checks: list[SpecCheck] = [
        SpecCheck("known-proof-status", proof.status in {"proven", "violated", "unknown", "abstained"}, proof.status),
        SpecCheck("survived-and-dropped-disjoint", survived.isdisjoint(dropped), repr((survived, dropped))),
    ]
    if proof.status == "proven":
        checks.extend(
            (
                SpecCheck("proven-drops-no-required-segments", not dropped, repr(dropped)),
                SpecCheck("proven-survives-every-required-segment", survived == required, repr((survived, required))),
            )
        )
        outcome = ProofOutcome.PROVEN
    elif proof.status == "violated":
        checks.extend(
            (
                SpecCheck("violation-drops-required-segment", bool(dropped), repr(dropped)),
                SpecCheck("required-partitioned-by-survival", survived | dropped == required, repr((survived, dropped, required))),
                SpecCheck("counterexample-names-dropped-required", bool(proof.minimal_counterexample), repr(proof.minimal_counterexample)),
            )
        )
        outcome = ProofOutcome.COUNTEREXAMPLE
    else:
        checks.append(SpecCheck("unknown-has-reason", bool(proof.reason), proof.reason or "<none>"))
        outcome = ProofOutcome.ABSTAINED
    return _sketch(
        "must-survive-budget",
        outcome,
        checks=tuple(checks),
        evidence=tuple((name, str(value)) for name, value in proof.to_metadata()),
    )


def prove_static_contract(finding: StaticContractFinding) -> ProofSketch:
    """Build an executable proof sketch for one finite static contract finding."""

    if finding.problem is None or finding.result is None:
        return _sketch(
            "z3-backed-finite-contract",
            ProofOutcome.ABSTAINED,
            checks=(SpecCheck("solver-proof-object-present", False, "missing problem or result"),),
            evidence=finding.evidence,
        )
    report = check_contract_result(finding.problem, finding.result)
    outcome = ProofOutcome.COUNTEREXAMPLE if finding.result.sat else ProofOutcome.PROVEN
    if finding.result.status is SolverStatus.UNKNOWN:
        outcome = ProofOutcome.ABSTAINED
    return _sketch(
        "z3-backed-finite-contract",
        outcome,
        checks=report.checks,
        evidence=(
            ("contract", finding.name),
            ("solver_status", finding.status.value),
            ("severity", finding.severity),
            *finding.evidence,
        ),
    )


def build_supported_proof_catalog() -> ProofSketchReport:
    """Return theorem sketches for the built-in proof families."""

    return ProofSketchReport(
        sketches=tuple(
            _sketch(property_id, ProofOutcome.SKETCH, checks=(SpecCheck("catalog-entry", True, "theorem registered"),))
            for property_id in (
                "role-boundary-nonforgeability",
                "stop-overreachability",
                "grammar-tokenizer-emptiness",
                "must-survive-budget",
                "z3-backed-finite-contract",
            )
        )
    )


def render_proof_sketch_report_json(report: ProofSketchReport) -> str:
    """Render proof sketches as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_proof_sketch_report_text(report: ProofSketchReport) -> str:
    """Render proof sketches as concise CLI text."""

    lines = ["PromptABI proof sketches", f"status: {'PASS' if report.passed else 'FAIL'}"]
    for sketch in report.sketches:
        lines.append("")
        lines.append(f"{sketch.property_id}: {sketch.outcome.value}")
        lines.append(f"  theorem: {sketch.theorem}")
        lines.append(f"  fragment: {sketch.supported_fragment}")
        if sketch.assumptions:
            lines.append("  assumptions:")
            lines.extend(f"    - {assumption}" for assumption in sketch.assumptions)
        if sketch.lemmas:
            lines.append("  lemmas:")
            lines.extend(f"    - {lemma.name}: {lemma.statement}" for lemma in sketch.lemmas)
        failed = [check for check in sketch.checks if not check.passed]
        lines.append(f"  executable checks: {len(sketch.checks)} ({'PASS' if not failed else 'FAIL'})")
        for check in failed:
            lines.append(f"    - {check.name}: {check.detail}")
    return "\n".join(lines) + "\n"


_THEOREMS: dict[str, tuple[str, str, tuple[str, ...], tuple[ProofLemma, ...]]] = {
    "role-boundary-nonforgeability": (
        "Role-boundary non-forgeability",
        "No attacker-controlled field can render as a structural role/control marker in the bounded role-region model.",
        (
            "bounded Hugging Face Jinja fragment",
            "recognized sanitizers remove raw marker emission",
            "token evidence is evaluated under the selected tokenizer abstraction",
        ),
        (
            ProofLemma(
                "region partition",
                "Each modeled render path is partitioned into structural role regions and unassigned text.",
                "RoleBoundaryModel records region offsets, roles, and controlled expressions per symbolic path.",
            ),
            ProofLemma(
                "marker disjointness",
                "Non-forgeability reduces to emptiness of controlled-region text intersected with marker language.",
                "A finding stores the concrete marker and controlled expression witnessing a non-empty intersection.",
            ),
        ),
    ),
    "stop-overreachability": (
        "Stop reachability",
        "A configured stop is unsafe when it occurs at a prefix of a valid structured output that leaves the parser in an incomplete or changed state.",
        (
            "bounded JSON, markdown, XML-like tool, provider-envelope, schema-string, and tool-argument regions",
            "provider stop semantics cut before the matched stop sequence",
        ),
        (
            ProofLemma(
                "prefix split",
                "The firing offset splits a valid output into delivered prefix, matched stop, and required suffix.",
                "The certificate checks the exact substring and reconstructed prefixes.",
            ),
            ProofLemma(
                "parser-state witness",
                "The truncated prefix is interpreted in the region parser state stored with the witness.",
                "Resulting structures are computed by the stop-overreachability parser simulators.",
            ),
        ),
    ),
    "grammar-tokenizer-emptiness": (
        "Tokenizer x grammar emptiness",
        "The bounded product is satisfiable iff an accepted grammar witness survives encode-normalize-decode and is accepted after decoding.",
        (
            "compiled bounded DFA for the supported grammar/schema fragment",
            "finite candidate and depth bounds are explicit",
            "tokenizer adapter exposes deterministic encode/decode metadata",
        ),
        (
            ProofLemma(
                "grammar witness replay",
                "A satisfiable proof replays the grammar-state path through the compiled DFA.",
                "The executable certificate requires the automaton rather than trusting stored state names alone.",
            ),
            ProofLemma(
                "tokenizer preservation",
                "The decoded token path must remain in the grammar language.",
                "The certificate rechecks DFA acceptance on the decoded text when the DFA is supplied.",
            ),
        ),
    ),
    "must-survive-budget": (
        "Must-survive budget",
        "Every required prompt segment survives iff the modeled truncation decision keeps exactly the required set under the declared input budget.",
        (
            "finite prompt-segment list with known token counts or explicit unknown outcome",
            "normalized framework truncation policy",
            "reserved output/tool/generation/special-token capacity is subtracted before packing",
        ),
        (
            ProofLemma(
                "required partition",
                "Required segments partition into survived and dropped sets.",
                "The certificate checks disjointness and exact coverage for violated proofs.",
            ),
            ProofLemma(
                "survival guarantee",
                "A proven result drops no required segment and marks every required segment as survived.",
                "Unknown counts or unsupported policies abstain rather than defaulting to zero.",
            ),
        ),
    ),
    "z3-backed-finite-contract": (
        "Z3-backed finite contract",
        "Finite static obligations are proved by either a satisfying counterexample assignment or an unsatisfiable minimal core over the supported expression fragment.",
        (
            "finite Boolean, enum, bounded-string, and integer-range domains",
            "constraints use the supported equality, membership, substring, length, Boolean, and integer operators",
            "unsupported solver fragments return UNKNOWN rather than safe",
        ),
        (
            ProofLemma(
                "assignment soundness",
                "SAT assignments must lie in every declared finite domain and satisfy every named constraint.",
                "check_contract_result independently re-evaluates assignments.",
            ),
            ProofLemma(
                "core soundness",
                "UNSAT cores name known constraints and are deletion-minimal in finite enumeration.",
                "The executable spec rechecks core unsatisfiability without trusting Z3.",
            ),
        ),
    ),
}


def _sketch(
    property_id: str,
    outcome: ProofOutcome,
    *,
    checks: tuple[SpecCheck, ...],
    evidence: tuple[tuple[str, str], ...] = (),
) -> ProofSketch:
    title, theorem, assumptions, lemmas = _THEOREMS[property_id]
    return ProofSketch(
        property_id=property_id,
        title=title,
        theorem=theorem,
        supported_fragment="; ".join(assumptions),
        assumptions=assumptions,
        outcome=outcome,
        lemmas=lemmas,
        checks=checks,
        evidence=evidence,
    )
