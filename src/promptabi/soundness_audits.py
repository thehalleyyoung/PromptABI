"""Soundness audit registry for PromptABI check families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from .compatibility_matrix import CHECK_RULE_IDS, CompatibilityMatrixEntry, build_compatibility_matrix
from .diagnostics import CheckMode
from .session import CHECK_MODE_CATALOG


SOUNDNESS_AUDIT_VERSION = "2026.06"


class SoundnessAuditStatus(StrEnum):
    """Honest status for one check family's audited soundness boundary."""

    SOUND_WITHIN_FRAGMENT = "sound-within-fragment"
    CONDITIONALLY_SOUND = "conditionally-sound"
    HEURISTIC = "heuristic-reviewed"
    ABSTAINING = "abstaining-reviewed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """One obligation that must remain true for a check's guarantee claim."""

    name: str
    claim: str
    discharge: str
    executable_reference: str
    satisfied: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "claim": self.claim,
            "discharge": self.discharge,
            "executable_reference": self.executable_reference,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True, slots=True)
class DifferentialEvidence:
    """Fixture, conformance, or executable-spec evidence backing an audit row."""

    name: str
    reference: str
    observation: str
    coverage: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "reference": self.reference,
            "observation": self.observation,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class BlindSpot:
    """Known edge that the audit must not silently promote to full soundness."""

    kind: str
    impact: str
    mitigation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "impact": self.impact,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True, slots=True)
class SoundnessAudit:
    """A review of one built-in check family's abstraction and guarantee boundary."""

    check: str
    rule_ids: tuple[str, ...]
    modes: tuple[CheckMode, ...]
    abstraction: str
    assumptions: tuple[str, ...]
    supported_fragments: tuple[str, ...]
    proof_obligations: tuple[ProofObligation, ...]
    differential_evidence: tuple[DifferentialEvidence, ...]
    blind_spots: tuple[BlindSpot, ...]

    @property
    def status(self) -> SoundnessAuditStatus:
        if (
            not self.assumptions
            or not self.supported_fragments
            or not self.proof_obligations
            or not self.differential_evidence
            or any(not obligation.satisfied for obligation in self.proof_obligations)
        ):
            return SoundnessAuditStatus.INCOMPLETE
        if CheckMode.SOUND in self.modes and not self.blind_spots:
            return SoundnessAuditStatus.SOUND_WITHIN_FRAGMENT
        if CheckMode.SOUND in self.modes or CheckMode.BOUNDED in self.modes or CheckMode.Z3_BACKED_SMT in self.modes:
            return SoundnessAuditStatus.CONDITIONALLY_SOUND
        if CheckMode.HEURISTIC in self.modes:
            return SoundnessAuditStatus.HEURISTIC
        return SoundnessAuditStatus.ABSTAINING

    @property
    def canonical(self) -> bool:
        return all(rule_id in CHECK_MODE_CATALOG for rule_id in self.rule_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "status": self.status.value,
            "canonical": self.canonical,
            "rule_ids": list(self.rule_ids),
            "modes": [mode.value for mode in self.modes],
            "abstraction": self.abstraction,
            "assumptions": list(self.assumptions),
            "supported_fragments": list(self.supported_fragments),
            "proof_obligations": [obligation.to_dict() for obligation in self.proof_obligations],
            "differential_evidence": [evidence.to_dict() for evidence in self.differential_evidence],
            "blind_spots": [blind_spot.to_dict() for blind_spot in self.blind_spots],
        }


@dataclass(frozen=True, slots=True)
class SoundnessAuditReport:
    """Deterministic report over audited PromptABI check families."""

    audits: tuple[SoundnessAudit, ...]

    @property
    def passed(self) -> bool:
        return all(audit.canonical and audit.status is not SoundnessAuditStatus.INCOMPLETE for audit in self.audits)

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in SoundnessAuditStatus}
        for audit in self.audits:
            counts[audit.status.value] += 1
        return {key: value for key, value in counts.items() if value}

    def to_dict(self) -> dict[str, object]:
        return {
            "version": SOUNDNESS_AUDIT_VERSION,
            "passed": self.passed,
            "status_counts": self.status_counts,
            "audits": [audit.to_dict() for audit in self.audits],
        }


def build_soundness_audit_report(rule: str | None = None) -> SoundnessAuditReport:
    """Build soundness audits from canonical built-in check metadata."""

    entries = tuple(entry for entry in build_compatibility_matrix(include_plugins=False).entries if entry.source == "built-in")
    audits = tuple(_audit_entry(entry) for entry in entries)
    if rule is not None:
        audits = tuple(audit for audit in audits if audit.check == rule or rule in audit.rule_ids)
        if not audits:
            known = sorted({entry.check for entry in entries} | {rule_id for entry in entries for rule_id in entry.rule_ids})
            raise ValueError(f"unknown soundness audit rule '{rule}' (known: {', '.join(known)})")
    return SoundnessAuditReport(audits=audits)


def render_soundness_audit_text(report: SoundnessAuditReport) -> str:
    """Render soundness audits for terminals and CI logs."""

    lines = [
        f"PromptABI soundness audit ({SOUNDNESS_AUDIT_VERSION})",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        "statuses: " + ", ".join(f"{name}={count}" for name, count in report.status_counts.items()),
        "",
    ]
    for audit in report.audits:
        modes = ",".join(mode.value for mode in audit.modes) or "unspecified"
        lines.append(f"{audit.check} [{audit.status.value}; {modes}]")
        lines.append(f"  rules: {', '.join(audit.rule_ids)}")
        lines.append(f"  abstraction: {audit.abstraction}")
        lines.append("  assumptions:")
        lines.extend(f"    - {assumption}" for assumption in audit.assumptions)
        lines.append("  supported fragments:")
        lines.extend(f"    - {fragment}" for fragment in audit.supported_fragments)
        lines.append("  proof obligations:")
        for obligation in audit.proof_obligations:
            state = "ok" if obligation.satisfied else "open"
            lines.append(f"    - {obligation.name} [{state}]: {obligation.claim}")
            lines.append(f"      discharge: {obligation.discharge}")
            lines.append(f"      reference: {obligation.executable_reference}")
        lines.append("  differential evidence:")
        for evidence in audit.differential_evidence:
            lines.append(f"    - {evidence.name}: {evidence.observation}")
            lines.append(f"      reference: {evidence.reference}; coverage: {evidence.coverage}")
        if audit.blind_spots:
            lines.append("  known blind spots:")
            for blind_spot in audit.blind_spots:
                lines.append(f"    - {blind_spot.kind}: {blind_spot.impact}")
                lines.append(f"      mitigation: {blind_spot.mitigation}")
        else:
            lines.append("  known blind spots: none recorded inside the declared fragment")
        lines.append("")
    return "\n".join(lines)


def render_soundness_audit_markdown(report: SoundnessAuditReport) -> str:
    """Render soundness audits as concise markdown for papers and review artifacts."""

    lines = [
        f"# PromptABI soundness audit ({SOUNDNESS_AUDIT_VERSION})",
        "",
        f"**Status:** {'PASS' if report.passed else 'FAIL'}",
        "",
        "| Check | Status | Modes | Canonical rules | Key blind spot |",
        "| --- | --- | --- | --- | --- |",
    ]
    for audit in report.audits:
        modes = ", ".join(mode.value for mode in audit.modes) or "unspecified"
        blind_spot = audit.blind_spots[0].kind if audit.blind_spots else "none within fragment"
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(audit.check),
                    _md(audit.status.value),
                    _md(modes),
                    _md(", ".join(audit.rule_ids)),
                    _md(blind_spot),
                )
            )
            + " |"
        )
    lines.append("")
    for audit in report.audits:
        lines.extend(
            (
                f"## {audit.check}",
                "",
                audit.abstraction,
                "",
                "**Proof obligations.** "
                + "; ".join(f"{item.name}: {item.discharge}" for item in audit.proof_obligations),
                "",
                "**Evidence.** "
                + "; ".join(f"{item.name} ({item.reference})" for item in audit.differential_evidence),
                "",
                "**Blind spots.** "
                + (
                    "; ".join(f"{item.kind}: {item.mitigation}" for item in audit.blind_spots)
                    if audit.blind_spots
                    else "None recorded inside the declared fragment."
                ),
                "",
            )
        )
    return "\n".join(lines)


def render_soundness_audit_json(report: SoundnessAuditReport) -> str:
    """Render soundness audits as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def _audit_entry(entry: CompatibilityMatrixEntry) -> SoundnessAudit:
    override = _DETAILED_AUDITS.get(entry.check, {})
    modes = entry.modes
    surfaces = ", ".join(surface.key for surface in entry.surfaces[:8])
    if len(entry.surfaces) > 8:
        surfaces += f", +{len(entry.surfaces) - 8} more"
    abstraction = str(
        override.get(
            "abstraction",
            f"Audits {entry.check} diagnostics over {surfaces or 'configured artifacts'} using the scheduler "
            "metadata and canonical diagnostic-mode catalog rather than a model-semantic claim.",
        )
    )
    assumptions = tuple(
        override.get(
            "assumptions",
            (
                "Configured artifacts are loaded through PromptABI artifact loaders with stable source/provenance metadata.",
                "The claim is structural and finite: it does not assert how a neural model will respond.",
                "Unsupported or underspecified inputs must emit an abstaining, bounded, heuristic, or drift diagnostic instead of a proof-shaped pass.",
            ),
        )
    )
    supported_fragments = tuple(
        override.get(
            "supported_fragments",
            (
                f"canonical rules: {', '.join(entry.rule_ids)}",
                f"guarantee modes: {', '.join(mode.value for mode in modes) or 'unspecified'}",
                f"artifact surfaces: {surfaces or 'none declared'}",
            ),
        )
    )
    obligations = tuple(
        override.get(
            "proof_obligations",
            _default_obligations(entry),
        )
    )
    evidence = tuple(
        override.get(
            "differential_evidence",
            _default_evidence(entry),
        )
    )
    blind_spots = tuple(
        override.get(
            "blind_spots",
            _default_blind_spots(entry),
        )
    )
    return SoundnessAudit(
        check=entry.check,
        rule_ids=entry.rule_ids,
        modes=modes,
        abstraction=abstraction,
        assumptions=assumptions,
        supported_fragments=supported_fragments,
        proof_obligations=obligations,
        differential_evidence=evidence,
        blind_spots=blind_spots,
    )


def _default_obligations(entry: CompatibilityMatrixEntry) -> tuple[ProofObligation, ...]:
    return (
        ProofObligation(
            "canonical-rule-modes",
            "Every emitted diagnostic rule has an explicit guarantee mode and no audit row invents a non-emitted rule id.",
            "Cross-checked against promptabi.session.CHECK_MODE_CATALOG and compatibility_matrix.CHECK_RULE_IDS.",
            "tests/test_soundness_audits.py::test_soundness_audit_covers_canonical_built_in_rules",
        ),
        ProofObligation(
            "artifact-scope-preservation",
            "The check only claims coverage for artifact kinds and surfaces registered with the scheduler/compatibility matrix.",
            "Audit rows are generated from build_compatibility_matrix(include_plugins=False).",
            "src/promptabi/compatibility_matrix.py::build_compatibility_matrix",
        ),
        ProofObligation(
            "honest-status-classification",
            "Heuristic, bounded, abstaining, and blind-spot-bearing checks cannot render as unqualified soundness.",
            "SoundnessAudit.status demotes rows with blind spots or heuristic/abstaining-only modes.",
            "tests/test_soundness_audits.py::test_soundness_status_never_promotes_blind_spots_to_full_soundness",
        ),
    )


def _default_evidence(entry: CompatibilityMatrixEntry) -> tuple[DifferentialEvidence, ...]:
    module_hint = entry.check.replace("-", "_")
    return (
        DifferentialEvidence(
            "scheduler-metadata",
            "src/promptabi/session.py",
            "Check dependencies, required resources, and per-rule modes are the source of truth for scope.",
            f"{len(entry.rule_ids)} canonical rule ids",
        ),
        DifferentialEvidence(
            "compatibility-matrix",
            "tests/test_compatibility_matrix.py",
            "The public matrix already proves built-in checks and guarantee modes are enumerated deterministically.",
            f"{len(entry.surfaces)} documented compatibility surfaces",
        ),
        DifferentialEvidence(
            "focused-check-tests",
            f"tests/test_{module_hint}.py",
            "The audit records the focused test module that exercises this checker or its public report surface.",
            "repository-local pytest target",
        ),
    )


def _default_blind_spots(entry: CompatibilityMatrixEntry) -> tuple[BlindSpot, ...]:
    blind_spots = [
        BlindSpot(
            "finite-artifact-scope",
            "Inputs not represented in the configured PromptABI artifacts are outside the proof boundary.",
            "Require lockfiles, provenance, and dependency graph review for every production artifact edge.",
        )
    ]
    if CheckMode.HEURISTIC in entry.modes:
        blind_spots.append(
            BlindSpot(
                "heuristic-subclaim",
                "At least one diagnostic mode intentionally provides best-effort evidence rather than a proof.",
                "Render the heuristic mode explicitly and gate strict CI on sound/bounded/Z3-backed rules when needed.",
            )
        )
    if CheckMode.ABSTAINING in entry.modes:
        blind_spots.append(
            BlindSpot(
                "unsupported-fragment-abstention",
                "Unsupported constructs may produce abstentions instead of positive or negative proof.",
                "Treat abstentions as release-blocking when policy requires full coverage.",
            )
        )
    if CheckMode.BOUNDED in entry.modes:
        blind_spots.append(
            BlindSpot(
                "bounded-search",
                "The guarantee is limited by declared bounds, fixture sizes, or finite summaries.",
                "Increase bounds or add differential fixtures for deployment-specific envelopes.",
            )
        )
    return tuple(blind_spots)


def _obligation(name: str, claim: str, discharge: str, reference: str) -> ProofObligation:
    return ProofObligation(name, claim, discharge, reference)


def _evidence(name: str, reference: str, observation: str, coverage: str) -> DifferentialEvidence:
    return DifferentialEvidence(name, reference, observation, coverage)


def _blind_spot(kind: str, impact: str, mitigation: str) -> BlindSpot:
    return BlindSpot(kind, impact, mitigation)


_DETAILED_AUDITS: dict[str, dict[str, object]] = {
    "role-boundary-nonforgeability": {
        "abstraction": (
            "Models chat-template rendering as bounded symbolic role regions and asks whether attacker-controlled "
            "fields can render provider/model control delimiters or assistant prefixes."
        ),
        "assumptions": (
            "The Hugging Face/Jinja template fragment has been parsed without unsupported constructs or has explicit abstentions.",
            "Attacker-controlled regions are exactly the symbolic message/tool fields classified by the role-boundary model.",
            "Special/control delimiters are sourced from tokenizer/template metadata, not guessed from natural-language semantics.",
        ),
        "supported_fragments": (
            "Hugging Face tokenizer_config chat templates in the bounded symbolic executor fragment.",
            "system/user/assistant/tool/developer/function/provider role regions and configured control markers.",
            "Escaping, JSON encoding, and safe-wrapper sanitizer declarations recognized by role-boundary checks.",
        ),
        "proof_obligations": (
            _obligation("region-partition", "Rendered spans preserve role ownership.", "RoleBoundaryModel paths record owner, field, and rendered offsets.", "src/promptabi/role_boundaries.py::build_role_boundary_model"),
            _obligation("marker-reachability", "A reported forgery witness embeds a real control marker at the reported offset.", "Proof sketches replay malicious_input, rendered excerpt, token evidence, and marker offsets.", "tests/test_proof_sketches.py::test_role_boundary_proof_validates_real_forgery_witness"),
            _obligation("abstain-outside-fragment", "Unsupported template constructs cannot be silently treated as safe.", "Unsupported parser/symbolic-executor paths emit role-boundary-abstained modes.", "tests/test_role_boundaries.py"),
        ),
        "differential_evidence": (
            _evidence("chat-template-differential", "tests/test_chat_templates.py", "Symbolic rendering is compared with real apply_chat_template-style fixtures.", "HF tokenizer_config templates"),
            _evidence("real-bug-benchmark", "fixtures/real_bug_benchmarks/benchmark.json", "Known delimiter-collision bugs remain labeled corpus cases.", "ChatML/Llama/Mistral-style role markers"),
            _evidence("proof-sketch-replay", "src/promptabi/proof_sketches.py::prove_role_boundary_nonforgeability", "Witness fields are replayed through executable checks.", "counterexample offsets and tokenizer evidence"),
        ),
        "blind_spots": (
            _blind_spot("semantic-obedience", "The check proves structural forgeability, not that a model will obey the forged role.", "Keep diagnostics phrased as interface-state reachability."),
            _blind_spot("bounded-template-execution", "Loops/branches beyond configured symbolic bounds may require abstention.", "Raise bounds or add differential fixtures for deployment templates."),
        ),
    },
    "stop-overreachability": {
        "abstraction": "Treats valid structured-output regions as finite languages and proves whether configured stop strings can fire inside them before the parser receives a complete object.",
        "assumptions": (
            "Structured regions are JSON/XML/markdown/custom fragments represented by the supported parser model.",
            "Stop firing follows provider/framework stop semantics recorded in StopPolicyArtifact fixtures.",
            "The claim ends at structural truncation; it does not predict token probabilities.",
        ),
        "supported_fragments": (
            "JSON strings and objects, XML-ish tool envelopes, markdown fences, escaped characters, and provider stop fixtures.",
            "OpenAI/HF/vLLM/llama.cpp/LiteLLM stop-policy shapes parsed by stop_policies.py.",
            "Bounded parser states with explicit abstentions for unsupported recursive/custom languages.",
        ),
        "proof_obligations": (
            _obligation("stop-offset-validity", "The stop sequence occurs exactly where the diagnostic says it fires.", "Proof sketch compares valid_output[firing:firing+len(stop)] with stop_sequence.", "tests/test_proof_sketches.py::test_stop_proof_validates_prefix_split_for_real_overreach_witness"),
            _obligation("prefix-malformation", "The truncated prefix omits a required suffix of an otherwise valid output.", "Executable sketch checks prefix split and removed suffix length.", "src/promptabi/proof_sketches.py::prove_stop_reachability"),
            _obligation("provider-stop-fixtures", "Provider-specific stop behavior is fixture-backed when claimed.", "Stop differential tests replay provider trace expectations.", "tests/test_stop_differential.py"),
        ),
        "differential_evidence": (
            _evidence("stop-simulator", "src/promptabi/stop_differential.py", "CPU-only traces replay expected stop points.", "HF/vLLM/llama.cpp-style stop policies"),
            _evidence("structured-output-fixtures", "tests/test_stop_overreachability.py", "Real checker output includes parser state and malformed-prefix witnesses.", "JSON/XML/markdown regions"),
            _evidence("proof-sketch-replay", "tests/test_proof_sketches.py", "Stop counterexamples are independently replayed.", "valid_output/truncated_prefix invariants"),
        ),
        "blind_spots": (
            _blind_spot("provider-undocumented-semantics", "Undocumented provider changes can invalidate recorded stop behavior.", "Refresh provider fixture packs before relying on migration gates."),
            _blind_spot("unsupported-parser-language", "Arbitrary application parsers are outside the supported finite parser model.", "Use parser-compatibility fixtures or custom plugin checks."),
        ),
    },
    "grammar-tokenizer-emptiness": {
        "abstraction": "Compiles supported grammar/schema fragments to finite automata and checks whether the selected tokenizer can emit at least one accepted byte string/token path.",
        "assumptions": (
            "JSON Schema, regex, EBNF, Outlines, xgrammar, llguidance, and PromptABI grammar inputs are normalized to the supported finite subset.",
            "Tokenizer adapters expose deterministic encode/decode and special-token behavior for the checked candidates.",
            "Recursive or backend-specific grammar features outside the finite fragment produce abstentions.",
        ),
        "supported_fragments": (
            "JSON Schema objects/arrays/enums/consts and bounded string/numeric constraints.",
            "Byte-level, Hugging Face tokenizers, tiktoken, and SentencePiece adapters with recorded normalization behavior.",
            "Finite automata witnesses with source-mapped parser states.",
        ),
        "proof_obligations": (
            _obligation("automaton-witness", "A satisfiable result carries text accepted by the compiled grammar automaton.", "Proof sketch checks automaton.accepts_text(decoded_text).", "tests/test_proof_sketches.py::test_grammar_proof_uses_compiled_dfa_for_real_tokenizer_product"),
            _obligation("empty-no-witness", "An empty result cannot retain a satisfiable witness.", "Proof sketch asserts witness is absent and shortest_witness is absent when automaton is supplied.", "src/promptabi/proof_sketches.py::prove_grammar_emptiness"),
            _obligation("backend-normalization", "Tokenizer normalization assumptions are differential-tested against real libraries where available.", "Tokenizer conformance and differential suites replay encode/decode expectations.", "tests/test_tokenizer_conformance.py"),
        ),
        "differential_evidence": (
            _evidence("grammar-conformance", "fixtures/grammar_conformance/suite.json", "Backend fragments are replayed as conformance cases.", "Outlines/xgrammar/llguidance/lm-format-enforcer/Guidance/Instructor/provider-native"),
            _evidence("tokenizer-conformance", "fixtures/tokenizer_conformance/suite.json", "Tokenizer family edge cases are replayed.", "BPE/Unigram/byte fallback/added tokens/normalization"),
            _evidence("executable-specs", "tests/test_executable_specs.py", "DFA witness invariants are tested as executable specs.", "automata reachability/product laws"),
        ),
        "blind_spots": (
            _blind_spot("backend-extra-semantics", "A constrained-decoding backend may implement features not captured by the normalized finite subset.", "Record a grammar-differential case and mark unsupported features as abstaining."),
            _blind_spot("candidate-bound", "Large alphabets or recursive schemas may require bounded enumeration.", "Use alphabet compression and raise explicit bounds for deployment schemas."),
        ),
    },
    "static-contracts": {
        "abstraction": "Lowers declared finite prompt-interface contracts to executable automata products and finite SMT obligations over booleans, enums, integer ranges, membership, lengths, and bounded strings.",
        "assumptions": (
            "Contracts use the parsed .pabi/JSON supported fragment and retain source spans for every lowered rule.",
            "Z3-backed results are scoped to finite domains and solver metadata recorded by the contract runner.",
            "Unsupported formulas, missing artifacts, or timeout-prone fragments become abstained/unknown diagnostics.",
        ),
        "supported_fragments": (
            "Role-region nonforgeability, control-token exclusion, tool-schema preconditions, stop reachability, prompt survival, and training target alignment.",
            "Finite SMT domains: boolean, enum, integer ranges, membership, length, and bounded strings.",
            "Contract composition and migration artifacts with precedence-preserving diagnostics.",
        ),
        "proof_obligations": (
            _obligation("solver-assignment-validity", "SAT findings include assignments that satisfy the lowered finite contract.", "Proof sketches recheck finite assignments and witness facts.", "tests/test_proof_sketches.py::test_static_contract_proof_rechecks_solver_assignment_and_unsat_core"),
            _obligation("unsat-core-minimality", "UNSAT proof sketches report deletion-minimal unsat cores when available.", "Executable proof sketch tests core deletion checks.", "src/promptabi/proof_sketches.py::prove_static_contract"),
            _obligation("source-span-preservation", "Lowered diagnostics retain human rule names and source spans.", "Contract parser/linter/migration tests assert source-mapped diagnostics.", "tests/test_static_contracts.py"),
        ),
        "differential_evidence": (
            _evidence("solver-replay", "fixtures/solver_replays/role-region-forgery.solver-replay.json", "Reduced SMT obligations replay without private source artifacts.", "finite solver constraints and metadata"),
            _evidence("contract-language-tests", "tests/test_static_contracts.py", "Parser, formatter, linter, composition, and lowering paths are exercised.", ".pabi and JSON contract fragments"),
            _evidence("smt-benchmark", "fixtures/smt_benchmarks/benchmark.json", "SAT/UNSAT/timeout/unsupported examples are benchmarked.", "finite prompt-interface failures"),
        ),
        "blind_spots": (
            _blind_spot("finite-domain-only", "Unbounded string or arithmetic claims are outside the theorem.", "Require explicit bounded domains or abstain."),
            _blind_spot("solver-timeout", "Solver budget exhaustion can produce unknown/abstained outcomes.", "Treat unknown solver results as policy failures for strict gates."),
        ),
    },
}


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
