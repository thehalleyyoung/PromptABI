"""Trace proof claims to executable tests, corpus cases, and release gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .proof_sketches import build_supported_proof_catalog


THEOREM_TRACEABILITY_VERSION = "2026.06"


class TraceEvidenceKind(StrEnum):
    """Evidence categories required for every core proof claim."""

    EXECUTABLE_SPEC = "executable-spec"
    PROPERTY_TEST = "property-test"
    CORPUS_CASE = "corpus-case"
    RELEASE_GATE = "release-gate"


@dataclass(frozen=True, slots=True)
class TheoremTraceEvidence:
    """One repository-local evidence link for a theorem claim."""

    kind: TraceEvidenceKind
    path: str
    description: str
    symbol: str | None = None

    def validate(self, repo_root: Path) -> tuple[str, ...]:
        target = repo_root / self.path
        failures: list[str] = []
        if not target.exists():
            failures.append(f"missing path: {self.path}")
            return tuple(failures)
        if self.symbol is not None:
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = target.read_text(encoding="utf-8", errors="ignore")
            if self.symbol not in content:
                failures.append(f"missing symbol {self.symbol!r} in {self.path}")
        return tuple(failures)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "path": self.path,
            "description": self.description,
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        return payload


@dataclass(frozen=True, slots=True)
class TheoremTrace:
    """Traceability row for one proof-catalog theorem."""

    property_id: str
    theorem: str
    supported_fragment: str
    evidence: tuple[TheoremTraceEvidence, ...]
    failures: tuple[str, ...] = ()

    @property
    def missing_kinds(self) -> tuple[TraceEvidenceKind, ...]:
        present = {item.kind for item in self.evidence}
        return tuple(kind for kind in TraceEvidenceKind if kind not in present)

    @property
    def passed(self) -> bool:
        return not self.failures and not self.missing_kinds

    def to_dict(self) -> dict[str, object]:
        return {
            "property_id": self.property_id,
            "theorem": self.theorem,
            "supported_fragment": self.supported_fragment,
            "passed": self.passed,
            "missing_kinds": [kind.value for kind in self.missing_kinds],
            "failures": list(self.failures),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class TheoremTraceabilityReport:
    """Complete theorem-to-test traceability report."""

    repository_root: Path
    traces: tuple[TheoremTrace, ...]

    @property
    def passed(self) -> bool:
        return all(trace.passed for trace in self.traces)

    @property
    def theorem_count(self) -> int:
        return len(self.traces)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": THEOREM_TRACEABILITY_VERSION,
            "repository_root": str(self.repository_root),
            "passed": self.passed,
            "theorem_count": self.theorem_count,
            "traces": [trace.to_dict() for trace in self.traces],
        }


def build_theorem_traceability_report(repo_root: str | Path | None = None) -> TheoremTraceabilityReport:
    """Build and validate theorem-to-test traceability for core proof claims."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    catalog = build_supported_proof_catalog()
    evidence_by_property = _trace_evidence()
    traces: list[TheoremTrace] = []
    for sketch in catalog.sketches:
        evidence = evidence_by_property.get(sketch.property_id, ())
        failures: list[str] = []
        for item in evidence:
            failures.extend(item.validate(root))
        unknown_property = sketch.property_id not in evidence_by_property
        if unknown_property:
            failures.append(f"no traceability row registered for {sketch.property_id}")
        trace = TheoremTrace(
            property_id=sketch.property_id,
            theorem=sketch.theorem,
            supported_fragment=sketch.supported_fragment,
            evidence=evidence,
            failures=tuple(failures),
        )
        traces.append(trace)
    return TheoremTraceabilityReport(repository_root=root, traces=tuple(traces))


def render_theorem_traceability_json(report: TheoremTraceabilityReport) -> str:
    """Render theorem traceability as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_theorem_traceability_text(report: TheoremTraceabilityReport) -> str:
    """Render theorem traceability for CI logs and reviewers."""

    lines = [
        f"PromptABI theorem traceability ({THEOREM_TRACEABILITY_VERSION})",
        f"status: {'PASS' if report.passed else 'FAIL'}",
        f"theorems: {report.theorem_count}",
    ]
    for trace in report.traces:
        lines.append("")
        lines.append(f"{trace.property_id}: {'PASS' if trace.passed else 'FAIL'}")
        lines.append(f"  theorem: {trace.theorem}")
        lines.append(f"  fragment: {trace.supported_fragment}")
        if trace.missing_kinds:
            lines.append("  missing evidence kinds: " + ", ".join(kind.value for kind in trace.missing_kinds))
        if trace.failures:
            lines.append("  failures:")
            lines.extend(f"    - {failure}" for failure in trace.failures)
        lines.append("  evidence:")
        for item in trace.evidence:
            symbol = f"::{item.symbol}" if item.symbol is not None else ""
            lines.append(f"    - {item.kind.value}: {item.path}{symbol}")
            lines.append(f"      {item.description}")
    return "\n".join(lines) + "\n"


def _evidence(
    kind: TraceEvidenceKind,
    path: str,
    description: str,
    *,
    symbol: str | None = None,
) -> TheoremTraceEvidence:
    return TheoremTraceEvidence(kind=kind, path=path, description=description, symbol=symbol)


def _trace_evidence() -> dict[str, tuple[TheoremTraceEvidence, ...]]:
    executable = TraceEvidenceKind.EXECUTABLE_SPEC
    property_test = TraceEvidenceKind.PROPERTY_TEST
    corpus = TraceEvidenceKind.CORPUS_CASE
    release = TraceEvidenceKind.RELEASE_GATE
    return {
        "role-boundary-nonforgeability": (
            _evidence(executable, "tests/test_proof_sketches.py", "replays a real ChatML forgery witness through the proof sketch", symbol="test_role_boundary_proof_validates_real_forgery_witness"),
            _evidence(executable, "tests/test_executable_specs.py", "checks DFA witnesses and product-language laws used by boundary proofs", symbol="test_executable_spec_checks_dfa_reachability_witness"),
            _evidence(property_test, "tests/test_checker_properties.py", "generates safe and unsafe bounded role-boundary templates", symbol="analyze_role_boundary_nonforgeability"),
            _evidence(corpus, "fixtures/real_bug_benchmarks/benchmark.json", "keeps delimiter-collision cases from real-world bug patterns"),
            _evidence(release, "src/promptabi/release.py", "release readiness requires formal proof catalog coverage", symbol="_formal_checks_check"),
        ),
        "stop-overreachability": (
            _evidence(executable, "tests/test_proof_sketches.py", "replays stop firing offsets, truncated prefixes, and valid-output suffixes", symbol="test_stop_proof_validates_prefix_split_for_real_overreach_witness"),
            _evidence(property_test, "tests/test_checker_properties.py", "generates unsafe and abstaining stop-overreachability cases", symbol="analyze_stop_overreachability"),
            _evidence(corpus, "fixtures/structured_schemas/tool-serialization-contract/stops.json", "anchors stop-policy fixtures that can interact with tool-call structures"),
            _evidence(corpus, "fixtures/provider_fixture_packs/openai-chat-completions/pack.json", "records provider stop trace shapes for differential replay"),
            _evidence(release, "tests/test_release_readiness.py", "ensures release readiness remains a passing repository gate", symbol="test_release_readiness_gate_passes_against_live_repository"),
        ),
        "grammar-tokenizer-emptiness": (
            _evidence(executable, "tests/test_proof_sketches.py", "supplies the compiled DFA and rechecks decoded grammar witnesses", symbol="test_grammar_proof_uses_compiled_dfa_for_real_tokenizer_product"),
            _evidence(executable, "tests/test_executable_specs.py", "checks bounded product-language laws used by tokenizer x grammar products", symbol="test_executable_spec_checks_bounded_product_language_laws"),
            _evidence(property_test, "tests/test_checker_properties.py", "generates SAT, empty, ambiguous, and abstaining grammar/tokenizer cases", symbol="analyze_tokenizer_grammar_emptiness"),
            _evidence(corpus, "fixtures/grammar_conformance/suite.json", "tracks backend conformance cases for normalized grammar fragments"),
            _evidence(corpus, "fixtures/tokenizer_conformance/suite.json", "tracks tokenizer family cases used by grammar-product assumptions"),
            _evidence(release, "src/promptabi/release.py", "release readiness requires grammar/tokenizer proof catalog coverage", symbol="grammar-tokenizer-emptiness"),
        ),
        "must-survive-budget": (
            _evidence(executable, "tests/test_proof_sketches.py", "rechecks survived/dropped required-segment partitions", symbol="test_must_survive_budget_proof_checks_real_truncation_partition"),
            _evidence(property_test, "tests/test_checker_properties.py", "generates safe and unsafe token-budget cases", symbol="analyze_token_budget"),
            _evidence(corpus, "fixtures/framework_truncation_conformance/suite.json", "pins framework truncation cases for budget survival assumptions"),
            _evidence(corpus, "examples/end-to-end/rag-truncation/buggy.promptabi.json", "keeps an end-to-end RAG truncation failure fixture"),
            _evidence(release, "src/promptabi/release.py", "release readiness requires token-budget model registration", symbol="token-budget-model"),
        ),
        "z3-backed-finite-contract": (
            _evidence(executable, "tests/test_proof_sketches.py", "rechecks finite SAT assignments and deletion-minimal UNSAT cores", symbol="test_static_contract_proof_rechecks_solver_assignment_and_unsat_core"),
            _evidence(executable, "tests/test_executable_specs.py", "independently enumerates finite solver domains and constraints", symbol="test_executable_spec_checks_sat_contract_assignment"),
            _evidence(executable, "tests/test_mechanized_proofs.py", "runs mechanized finite-contract proof experiments over SAT assignments and UNSAT cores", symbol="test_mechanized_proof_experiments_pass_and_cover_core_fragments"),
            _evidence(property_test, "tests/test_checker_properties.py", "generates safe and unsafe static-contract obligations", symbol="analyze_static_contracts"),
            _evidence(corpus, "fixtures/smt_benchmarks/benchmark.json", "pins SAT/UNSAT/timeout/unsupported SMT benchmark cases"),
            _evidence(corpus, "fixtures/solver_replays/role-region-forgery.solver-replay.json", "stores reduced solver replay evidence without private artifacts"),
            _evidence(release, "src/promptabi/release.py", "release readiness requires Z3-backed and bounded static-contract modes", symbol="CheckMode.Z3_BACKED_SMT"),
        ),
    }
