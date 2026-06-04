"""Deterministic CPU-only performance benchmarks for PromptABI."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    SchemaArtifact,
    SpecialToken,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    TruncationStrategy,
)
from .budgets import analyze_token_budget
from .chat_templates import ChatTemplateSymbolicBounds, parse_hf_tokenizer_config_chat_template, symbolically_execute_chat_template
from .config import VerificationConfig
from .grammar_emptiness import analyze_tokenizer_grammar_emptiness
from .loaders import ArtifactLoader, LoadedArtifact
from .session import CheckContext, VerificationSession
from .static_contracts import analyze_static_contracts
from .stop_analysis import analyze_stop_policy_tokenizer
from .stop_overreachability import analyze_stop_overreachability
from .tokenizers import ByteLevelTokenizer


BenchmarkCallable = Callable[[Path], dict[str, object]]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One deterministic benchmark measurement."""

    name: str
    iterations: int
    seconds: float
    runs_per_second: float
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.name,
            "iterations": self.iterations,
            "seconds": round(self.seconds, 6),
            "runs_per_second": round(self.runs_per_second, 2),
            "metrics": dict(sorted(self.metrics.items())),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A named benchmark case and its default iteration count."""

    name: str
    func: BenchmarkCallable
    default_iterations: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase("tokenizer-analysis", _benchmark_tokenizer_analysis, 100),
        BenchmarkCase("template-symbolic-execution", _benchmark_template_symbolic_execution, 20),
        BenchmarkCase("grammar-emptiness", _benchmark_grammar_emptiness, 50),
        BenchmarkCase("stop-checks", _benchmark_stop_checks, 100),
        BenchmarkCase("z3-static-contracts", _benchmark_static_contracts, 40),
        BenchmarkCase("budget-checks", _benchmark_budget_checks, 100),
        BenchmarkCase("corpus-wide-verification", _benchmark_corpus_wide_verification, 10),
        BenchmarkCase("cache-cold-warm", _benchmark_cache_cold_warm, 30),
    )


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    iterations: int | None = None,
    root: Path | None = None,
) -> BenchmarkResult:
    """Run one benchmark case against real PromptABI code paths."""

    count = iterations if iterations is not None else case.default_iterations
    if count <= 0:
        raise ValueError("iterations must be positive")
    benchmark_root = root or repo_root()
    last_metrics: dict[str, object] = {}
    start = time.perf_counter()
    for _ in range(count):
        last_metrics = case.func(benchmark_root)
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name=case.name,
        iterations=count,
        seconds=elapsed,
        runs_per_second=count / elapsed if elapsed else float("inf"),
        metrics=last_metrics,
    )


def run_benchmarks(
    selected: Sequence[str] = ("all",),
    *,
    iterations: int | None = None,
    root: Path | None = None,
) -> tuple[BenchmarkResult, ...]:
    """Run selected benchmark cases by name."""

    cases = benchmark_cases()
    by_name = {case.name: case for case in cases}
    names = tuple(selected or ("all",))
    if "all" in names:
        scheduled = cases
    else:
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(f"unknown benchmark case(s): {', '.join(unknown)}")
        scheduled = tuple(by_name[name] for name in names)
    return tuple(run_benchmark_case(case, iterations=iterations, root=root) for case in scheduled)


def render_benchmark_json(results: Iterable[BenchmarkResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic PromptABI performance benchmarks.")
    parser.add_argument(
        "cases",
        nargs="*",
        default=("all",),
        help="Benchmark case names, or 'all'.",
    )
    parser.add_argument("--iterations", type=int, help="Override the default iteration count for every selected case.")
    parser.add_argument("--repo-root", type=Path, default=repo_root(), help="Repository root containing examples/ and fixtures/.")
    args = parser.parse_args(argv)
    results = run_benchmarks(args.cases, iterations=args.iterations, root=args.repo_root)
    print(render_benchmark_json(results))
    return 0


def _benchmark_tokenizer_analysis(root: Path) -> dict[str, object]:
    del root
    tokenizer = ByteLevelTokenizer(
        added_tokens=("<|assistant|>", "<|tool_call|>", "</tool_call>"),
        special_tokens={"<|assistant|>": 32001},
        normalization=("nfkc",),
    )
    samples = _tokenizer_samples()
    token_total = 0
    exact = 0
    for sample in samples:
        result = tokenizer.round_trip(sample)
        token_total += len(result.token_ids)
        exact += int(result.exact_match)
    return {"samples": len(samples), "tokens": token_total, "exact_round_trips": exact}


def _benchmark_template_symbolic_execution(root: Path) -> dict[str, object]:
    paths = tuple(sorted((root / "fixtures" / "seed_corpus").glob("*/tokenizer_config.json")))
    bounds = ChatTemplateSymbolicBounds(max_messages=2, max_tools=1, max_loop_iterations=2, max_paths=64)
    symbolic_paths = 0
    abstentions = 0
    for path in paths:
        parsed = parse_hf_tokenizer_config_chat_template(path)
        execution = symbolically_execute_chat_template(parsed, bounds=bounds)
        symbolic_paths += len(execution.paths)
        abstentions += len(execution.abstentions)
    return {"templates": len(paths), "symbolic_paths": symbolic_paths, "abstentions": abstentions}


def _benchmark_grammar_emptiness(root: Path) -> dict[str, object]:
    tokenizer_artifact = _byte_tokenizer_artifact()
    schema_artifact = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="ticket-routing-schema",
        location=ArtifactLocation(path=str(root / "fixtures" / "structured_schemas" / "open-source-agent-ticket" / "schema.json")),
    )
    report = analyze_tokenizer_grammar_emptiness(
        tokenizer_artifact,
        schema_artifact,
        ByteLevelTokenizer(),
        max_candidates=16,
    )
    return {
        "status": report.status.value,
        "checked_candidates": report.checked_candidates,
        "grammar_states": report.grammar_state_count,
        "grammar_accepts": report.grammar_accept_count,
    }


def _benchmark_stop_checks(root: Path) -> dict[str, object]:
    del root
    stop_policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="tool-stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("}", "</tool_call>", "```", "<|assistant|>"),
        stop_token_ids=(0, 255, 999999),
    )
    tokenizer = ByteLevelTokenizer(added_tokens=("</tool_call>", "<|assistant|>"), special_tokens={"<|assistant|>": 32001})
    tokenizer_report = analyze_stop_policy_tokenizer(stop_policy, tokenizer)
    overreach_report = analyze_stop_overreachability(stop_policy)
    return {
        "stop_sequences": len(tokenizer_report.sequences),
        "stop_token_ids": len(tokenizer_report.token_ids),
        "unreachable_token_ids": len(tokenizer_report.unreachable_token_ids),
        "collisions": len(tokenizer_report.collisions),
        "overreach_findings": len(overreach_report.findings),
    }


def _benchmark_static_contracts(root: Path) -> dict[str, object]:
    del root
    artifacts = _budget_loaded_artifacts()
    special_tokens = SpecialTokenMapArtifact(
        kind=ArtifactKind.SPECIAL_TOKEN_MAP,
        name="specials",
        location=ArtifactLocation(uri="memory://specials"),
        tokens=(SpecialToken("assistant", "<|assistant|>", 32001),),
    )
    stop_policy = StopPolicyArtifact(
        kind=ArtifactKind.STOP_POLICY,
        name="stops",
        location=ArtifactLocation(uri="memory://stops"),
        stop_sequences=("<|assistant|>",),
    )
    loaded = (*artifacts, ArtifactLoader().load(special_tokens), ArtifactLoader().load(stop_policy), ArtifactLoader().load(_byte_tokenizer_artifact()))
    report = analyze_static_contracts(VerificationConfig(name="static-benchmark", max_context_tokens=96), loaded)
    return {"findings": len(report.findings), "violations": len(report.violations)}


def _benchmark_budget_checks(root: Path) -> dict[str, object]:
    del root
    report = analyze_token_budget(
        VerificationConfig(name="budget-benchmark", artifact_bundle=ArtifactBundle(())),
        _budget_loaded_artifacts(),
        tokenizers=((_byte_tokenizer_artifact(), ByteLevelTokenizer()),),
    )
    return {
        "segments": len(report.segments),
        "findings": len(report.findings),
        "required_prompt_tokens": report.required_prompt_tokens,
        "input_budget_tokens": report.reservation.input_budget_tokens if report.reservation else None,
    }


def _benchmark_corpus_wide_verification(root: Path) -> dict[str, object]:
    configs = (
        root / "examples" / "minimal" / "promptabi.json",
        root / "examples" / "token-budget" / "promptabi.json",
        root / "examples" / "rag-chunking" / "promptabi.json",
        root / "examples" / "stop-policies" / "promptabi.json",
        root / "fixtures" / "structured_schemas" / "open-source-agent-ticket" / "promptabi.json",
        root / "fixtures" / "structured_schemas" / "tool-serialization-contract" / "promptabi.json",
    )
    diagnostics = 0
    failures = 0
    for config in configs:
        result = VerificationSession.from_config_file(config).run()
        diagnostics += len(result.diagnostics)
        failures += int(not result.ok)
    return {"configs": len(configs), "diagnostics": diagnostics, "failing_configs": failures}


def _benchmark_cache_cold_warm(root: Path) -> dict[str, object]:
    del root
    loaded = _budget_loaded_artifacts()
    config = VerificationConfig(
        name="cache-benchmark",
        checks=("token-budget-model", "rag-chunking-compatibility"),
        artifact_bundle=ArtifactBundle(tuple(artifact.artifact for artifact in loaded)),
    )
    cold_start = time.perf_counter()
    cold_report = analyze_token_budget(
        config,
        loaded,
        tokenizers=((_byte_tokenizer_artifact(), ByteLevelTokenizer()),),
    )
    cold_seconds = time.perf_counter() - cold_start

    warm_session = VerificationSession(config)
    context = CheckContext(config=config, loaded_artifacts=loaded)
    warm_start = time.perf_counter()
    first = warm_session._analyze_token_budget_context(context)
    second = warm_session._analyze_token_budget_context(context)
    warm_seconds = time.perf_counter() - warm_start
    return {
        "loaded_artifacts": len(loaded),
        "cold_findings": len(cold_report.findings),
        "warm_findings": len(first.findings) + len(second.findings),
        "cache_reused": first is second,
        "cold_seconds": round(cold_seconds, 6),
        "warm_pair_seconds": round(warm_seconds, 6),
    }


def _budget_loaded_artifacts() -> tuple[LoadedArtifact, ...]:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="benchmark-segments",
        location=ArtifactLocation(uri="memory://benchmark-segments"),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=54, overhead_tokens=2),
            PromptSegment("retrieval-context", role="retrieval", required=False, token_count=38, metadata_tokens=4),
            PromptSegment("developer-instructions", role="developer", required=True, content="Return JSON only."),
            PromptSegment("user-request", role="user", required=True, token_count=18),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="benchmark-runtime-budget",
        location=ArtifactLocation(uri="memory://benchmark-runtime-budget"),
        framework="vllm",
        strategy=TruncationStrategy.LEFT,
        max_context_tokens=96,
        reserve_output_tokens=20,
        reserved_tool_tokens=8,
        generation_prompt_tokens=3,
        special_token_overhead=5,
    )
    loader = ArtifactLoader()
    return (loader.load(segments), loader.load(budget))


def _byte_tokenizer_artifact() -> TokenizerArtifact:
    return TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="byte-level",
        location=ArtifactLocation(uri="memory://byte-level"),
        family="byte-level",
    )


def _tokenizer_samples() -> tuple[str, ...]:
    base = (
        "plain user message",
        "tool call: </tool_call>",
        "assistant marker <|assistant|>",
        "unicode cafe\u0301 and snowman \u2603",
        "{\"action\":\"route\",\"ticket_id\":\"TCK-123\",\"priority\":\"high\"}",
    )
    return tuple(f"{sample} #{index}" for index in range(10) for sample in base)


if __name__ == "__main__":
    raise SystemExit(main())
