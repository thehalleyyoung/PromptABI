"""A publishable benchmark suite for PromptABI's compositional proofs.

Steps 200-214 added a family of *compositional* proofs that combine multiple
artifacts -- contract composition, attestation x deployment-gate admission,
local policy-pack semantics preservation, chat-template abstract interpretation,
nested tool-call framing, multi-agent handoffs, and streaming parser products.

This module turns those proofs into a deterministic, publishable benchmark.  Each
case declares its expected verdict (``proven`` or ``refuted``), the suite runs the
real proof entrypoint, checks the verdict matches, records wall-clock timing and
the number of replayable witnesses produced, and emits a content-addressed
manifest so results are comparable across runs and machines.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .attestation_gate_composition import compose_attestation_gate_from_config
from .multi_agent_handoffs import analyze_multi_agent_handoffs
from .nested_tool_calls import NestedToolCall, ToolCallEncoding, analyze_nested_tool_call
from .policies import VerificationPolicy
from .policy_pack_semantics import prove_policy_pack_preserves_semantics
from .streaming_parser_products import analyze_streaming_parser_product
from .template_abstract_interpretation import interpret_chat_template_file


COMPOSITIONAL_PROOF_BENCHMARK_VERSION = "promptabi.compositional-proof-benchmarks.v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]


class CompositionalProofVerdict(StrEnum):
    """Expected/observed verdict for a compositional proof case."""

    PROVEN = "proven"
    REFUTED = "refuted"


class CompositionalProofBenchmarkError(ValueError):
    """Raised when the benchmark suite cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class _Probe:
    """Observed outcome of running one proof entrypoint."""

    verdict: CompositionalProofVerdict
    witness_count: int
    detail: str


@dataclass(frozen=True, slots=True)
class CompositionalProofBenchmarkCase:
    """One compositional proof case with a declared expected verdict."""

    case_id: str
    family: str
    description: str
    expected: CompositionalProofVerdict
    run: Callable[[], _Probe]


@dataclass(frozen=True, slots=True)
class CompositionalProofBenchmarkResult:
    """Result of running one compositional proof case."""

    case_id: str
    family: str
    description: str
    expected: CompositionalProofVerdict
    observed: CompositionalProofVerdict
    passed: bool
    witness_count: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "detail": self.detail,
            "expected": self.expected.value,
            "family": self.family,
            "observed": self.observed.value,
            "passed": self.passed,
            "witness_count": self.witness_count,
        }


@dataclass(frozen=True, slots=True)
class CompositionalProofBenchmarkReport:
    """Deterministic, content-addressed compositional proof benchmark report."""

    results: tuple[CompositionalProofBenchmarkResult, ...]
    duration_ms: float
    manifest_sha256: str

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def ok(self) -> bool:
        return self.passed == self.total and self.total > 0

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(result.family for result in self.results))

    def manifest(self) -> dict[str, object]:
        return {
            "families": list(self.families),
            "ok": self.ok,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
            "total": self.total,
            "version": COMPOSITIONAL_PROOF_BENCHMARK_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        data = self.manifest()
        data["manifest_sha256"] = self.manifest_sha256
        return data


def compositional_proof_benchmark_cases() -> tuple[CompositionalProofBenchmarkCase, ...]:
    """The curated, deterministic compositional proof benchmark cases."""

    return (
        CompositionalProofBenchmarkCase(
            case_id="nested-tool-call/json-roundtrips",
            family="nested-tool-calls",
            description="A self-delimiting json encoding of a 2-level tool call round-trips and stays framed.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_nested_json,
        ),
        CompositionalProofBenchmarkCase(
            case_id="nested-tool-call/xml-desyncs",
            family="nested-tool-calls",
            description="An unescaped xml-tag nesting desynchronizes a non-nesting marker parser.",
            expected=CompositionalProofVerdict.REFUTED,
            run=_probe_nested_xml,
        ),
        CompositionalProofBenchmarkCase(
            case_id="attestation-gate/admit",
            family="attestation-gate",
            description="A live attestation whose bundle matches the gate is admitted end-to-end.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_attestation_admit,
        ),
        CompositionalProofBenchmarkCase(
            case_id="attestation-gate/deny-key-mismatch",
            family="attestation-gate",
            description="A bundle signed by an untrusted key is denied admission with a witness.",
            expected=CompositionalProofVerdict.REFUTED,
            run=_probe_attestation_deny,
        ),
        CompositionalProofBenchmarkCase(
            case_id="policy-pack/severity-remap-preserves",
            family="policy-pack-semantics",
            description="A severity-only policy override preserves every checker finding.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_policy_preserves,
        ),
        CompositionalProofBenchmarkCase(
            case_id="policy-pack/drop-detected",
            family="policy-pack-semantics",
            description="An untrusted transformation that erases a finding is refuted.",
            expected=CompositionalProofVerdict.REFUTED,
            run=_probe_policy_drops,
        ),
        CompositionalProofBenchmarkCase(
            case_id="template-ai/qwen-balanced",
            family="template-abstract-interpretation",
            description="Qwen ChatML frames are proven balanced for any message count.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_template_qwen,
        ),
        CompositionalProofBenchmarkCase(
            case_id="template-ai/imbalanced-refuted",
            family="template-abstract-interpretation",
            description="A loop missing its close marker is refuted across all message counts.",
            expected=CompositionalProofVerdict.REFUTED,
            run=_probe_template_imbalanced,
        ),
        CompositionalProofBenchmarkCase(
            case_id="multi-agent-handoff/safe",
            family="multi-agent-handoffs",
            description="A typed, role-checked handoff between two agents carries no contract violation.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_handoff_safe,
        ),
        CompositionalProofBenchmarkCase(
            case_id="streaming-parser/balanced-json",
            family="streaming-parser-products",
            description="A streamed balanced JSON object composes cleanly with its parser monitor.",
            expected=CompositionalProofVerdict.PROVEN,
            run=_probe_streaming_safe,
        ),
    )


def run_compositional_proof_benchmarks() -> CompositionalProofBenchmarkReport:
    """Run every compositional proof case and produce a content-addressed report."""

    results: list[CompositionalProofBenchmarkResult] = []
    started = time.perf_counter()
    for case in compositional_proof_benchmark_cases():
        probe = case.run()
        results.append(
            CompositionalProofBenchmarkResult(
                case_id=case.case_id,
                family=case.family,
                description=case.description,
                expected=case.expected,
                observed=probe.verdict,
                passed=probe.verdict == case.expected,
                witness_count=probe.witness_count,
                detail=probe.detail,
            )
        )
    duration_ms = (time.perf_counter() - started) * 1000.0
    results.sort(key=lambda item: item.case_id)
    manifest = {
        "results": [result.to_dict() for result in results],
        "version": COMPOSITIONAL_PROOF_BENCHMARK_VERSION,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return CompositionalProofBenchmarkReport(
        results=tuple(results),
        duration_ms=duration_ms,
        manifest_sha256=manifest_sha256,
    )


def publish_compositional_proof_benchmarks(
    output_dir: str | Path,
    *,
    report: CompositionalProofBenchmarkReport | None = None,
) -> Path:
    """Publish the benchmark manifest to ``output_dir`` and return the manifest path."""

    report = report or run_compositional_proof_benchmarks()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "compositional-proof-benchmarks.json"
    manifest_path.write_text(render_compositional_proof_benchmark_json(report), encoding="utf-8")
    return manifest_path


def render_compositional_proof_benchmark_json(report: CompositionalProofBenchmarkReport) -> str:
    """Render the benchmark report as a stable, publishable JSON manifest."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_compositional_proof_benchmark_text(report: CompositionalProofBenchmarkReport) -> str:
    """Render the benchmark report as a leaderboard-style summary."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        "PromptABI compositional proof benchmarks",
        f"status: {status}",
        f"passed: {report.passed}/{report.total}",
        f"families: {len(report.families)}",
        f"manifest_sha256: {report.manifest_sha256}",
        f"duration_ms: {report.duration_ms:.2f}",
    ]
    for result in report.results:
        mark = "ok" if result.passed else "MISMATCH"
        lines.append(
            f"  [{mark}] {result.case_id} ({result.family}) "
            f"expected={result.expected.value} observed={result.observed.value} "
            f"witnesses={result.witness_count}"
        )
    return "\n".join(lines) + "\n"


# --- probes -----------------------------------------------------------------


def _verdict(ok: bool) -> CompositionalProofVerdict:
    return CompositionalProofVerdict.PROVEN if ok else CompositionalProofVerdict.REFUTED


def _probe_nested_json() -> _Probe:
    inner = NestedToolCall(name="search_kb", arguments={"q": "refund", "limit": 5})
    outer = NestedToolCall(name="run_subagent", arguments={"goal": "summarize", "inner": inner})
    report = analyze_nested_tool_call(outer, ToolCallEncoding(style="json"))
    return _Probe(_verdict(report.ok), len(report.violations), f"depth={report.depth}")


def _probe_nested_xml() -> _Probe:
    inner = NestedToolCall(name="search_kb", arguments={"q": "refund"})
    outer = NestedToolCall(name="run_subagent", arguments={"goal": "summarize", "inner": inner})
    report = analyze_nested_tool_call(outer, ToolCallEncoding(style="xml-tags"))
    return _Probe(_verdict(report.ok), len(report.violations), f"violations={len(report.violations)}")


def _attestation_config() -> Path:
    return _REPO_ROOT / "examples" / "attestation-gate" / "promptabi.json"


def _probe_attestation_admit() -> _Probe:
    report = compose_attestation_gate_from_config(
        _attestation_config(), attestation_key="release-key", gate_key="release-key"
    )
    return _Probe(_verdict(report.admitted), len(report.findings), report.decision.value)


def _probe_attestation_deny() -> _Probe:
    report = compose_attestation_gate_from_config(
        _attestation_config(), attestation_key="running-key", gate_key="trusted-key"
    )
    return _Probe(_verdict(report.admitted), len(report.findings), report.decision.value)


def _probe_policy_preserves() -> _Probe:
    from .diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity, WitnessTrace

    raw = (
        Diagnostic(
            rule_id="template-role-leak",
            severity=DiagnosticSeverity.ERROR,
            message="role leak",
            artifact=ArtifactRef(kind="chat-template", name="t", path="memory://t"),
            witness=WitnessTrace(summary="why"),
        ),
    )
    policy = VerificationPolicy(severity_overrides=(("template-role-leak", DiagnosticSeverity.WARNING),))
    report = prove_policy_pack_preserves_semantics(raw, policy)
    return _Probe(_verdict(report.preserves_semantics), len(report.violations), f"preserved={report.preserved_findings}")


def _probe_policy_drops() -> _Probe:
    from .diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity, WitnessTrace

    art = ArtifactRef(kind="tokenizer", name="t", path="memory://t")
    raw = (
        Diagnostic(rule_id="a", severity=DiagnosticSeverity.ERROR, message="a", artifact=art, witness=WitnessTrace(summary="a")),
        Diagnostic(rule_id="b", severity=DiagnosticSeverity.ERROR, message="b", artifact=art, witness=WitnessTrace(summary="b")),
    )
    claimed = (raw[1],)
    report = prove_policy_pack_preserves_semantics(raw, VerificationPolicy(), claimed_applied=claimed)
    return _Probe(_verdict(report.preserves_semantics), len(report.violations), f"violations={len(report.violations)}")


def _probe_template_qwen() -> _Probe:
    report = interpret_chat_template_file(
        _REPO_ROOT / "fixtures" / "seed_corpus" / "qwen" / "tokenizer_config.json", name="qwen"
    )
    return _Probe(_verdict(report.ok), len(report.violations), f"markers={len(report.marker_counts)}")


def _probe_template_imbalanced() -> _Probe:
    report = interpret_chat_template_file(
        _REPO_ROOT / "examples" / "template-abstract-interpretation" / "imbalanced-chatml.json",
        name="imbalanced",
    )
    return _Probe(_verdict(report.ok), len(report.violations), f"violations={len(report.violations)}")


def _probe_handoff_safe() -> _Probe:
    report = analyze_multi_agent_handoffs(
        {
            "name": "bench-safe-handoff",
            "agents": [
                {"name": "triage", "accepts_roles": ["user"], "emits_roles": ["tool"]},
                {
                    "name": "refund",
                    "accepts_roles": ["tool"],
                    "emits_roles": ["assistant"],
                    "required_fields": ["case_id"],
                    "input_schema": {"case_id": "string"},
                },
            ],
            "handoffs": [
                {
                    "name": "triage-to-refund",
                    "from": "triage",
                    "to": "refund",
                    "payload": {"role": "tool", "content": "encoded", "fields": {"case_id": "SUP-1"}},
                }
            ],
        }
    )
    return _Probe(_verdict(report.ok), len(report.violations), f"handoffs={len(report.handoffs)}")


def _probe_streaming_safe() -> _Probe:
    report = analyze_streaming_parser_product(['{"case": ', '"SUP-1"}'])
    return _Probe(_verdict(report.ok), len(report.violations), "balanced-json")
