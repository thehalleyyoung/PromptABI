"""Differential stop-policy simulation against recorded CPU-only traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ProviderConfigArtifact, StopPolicyArtifact


class StopDifferentialError(ValueError):
    """Raised when a recorded stop trace is outside the supported fixture format."""


@dataclass(frozen=True, slots=True)
class StopTraceExpectation:
    """Expected runtime stop behavior captured from a provider/framework fixture."""

    stopped: bool
    output: str
    matched_stop: str | None = None
    finish_reason: str | None = None
    include_stop_in_output: bool | None = None


@dataclass(frozen=True, slots=True)
class StopTraceCase:
    """One deterministic generation trace for stop-policy differential testing."""

    name: str
    family: str
    chunks: tuple[str, ...]
    token_ids: tuple[int, ...]
    expectation: StopTraceExpectation


@dataclass(frozen=True, slots=True)
class StopSimulationResult:
    """PromptABI's local simulation of one stop trace."""

    stopped: bool
    output: str
    matched_stop: str | None = None
    finish_reason: str | None = None
    stop_offset: int | None = None
    token_index: int | None = None
    include_stop_in_output: bool = False


@dataclass(frozen=True, slots=True)
class StopDifferentialMismatch:
    """A recorded trace disagrees with PromptABI's stop simulator."""

    case: StopTraceCase
    expected: StopTraceExpectation
    actual: StopSimulationResult
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StopDifferentialAbstention:
    """A provider fixture could not be used for stop differential testing."""

    artifact_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class StopDifferentialReport:
    """Differential result for one stop policy against recorded fixtures."""

    stop_policy_name: str
    cases: tuple[StopTraceCase, ...]
    matches: tuple[StopSimulationResult, ...]
    mismatches: tuple[StopDifferentialMismatch, ...]
    abstentions: tuple[StopDifferentialAbstention, ...] = ()


def analyze_stop_differential(
    stop_policy: StopPolicyArtifact,
    provider_configs: Sequence[ProviderConfigArtifact],
) -> StopDifferentialReport:
    """Replay recorded stop traces through small provider/framework simulators."""

    cases: list[StopTraceCase] = []
    matches: list[StopSimulationResult] = []
    mismatches: list[StopDifferentialMismatch] = []
    abstentions: list[StopDifferentialAbstention] = []
    for artifact in provider_configs:
        if artifact.location.path is None:
            abstentions.append(
                StopDifferentialAbstention(
                    artifact_name=artifact.name,
                    reason="provider fixture is not a local recorded JSON snapshot",
                )
            )
            continue
        try:
            artifact_cases = load_stop_trace_cases(Path(artifact.location.path), default_family=artifact.provider)
        except StopDifferentialError as exc:
            abstentions.append(StopDifferentialAbstention(artifact_name=artifact.name, reason=str(exc)))
            continue
        for case in artifact_cases:
            actual = simulate_stop_trace(stop_policy, case)
            mismatch_fields = _mismatch_fields(case.expectation, actual)
            cases.append(case)
            if mismatch_fields:
                mismatches.append(
                    StopDifferentialMismatch(
                        case=case,
                        expected=case.expectation,
                        actual=actual,
                        fields=mismatch_fields,
                    )
                )
            else:
                matches.append(actual)
    return StopDifferentialReport(
        stop_policy_name=stop_policy.name,
        cases=tuple(cases),
        matches=tuple(matches),
        mismatches=tuple(sorted(mismatches, key=lambda item: item.case.name)),
        abstentions=tuple(sorted(abstentions, key=lambda item: (item.artifact_name, item.reason))),
    )


def load_stop_trace_cases(path: Path, *, default_family: str) -> tuple[StopTraceCase, ...]:
    """Load stop traces from a recorded provider fixture JSON file."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StopDifferentialError(f"stop trace fixture is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise StopDifferentialError("stop trace fixture root must be a JSON object")
    traces = raw.get("stop_traces", raw.get("stop_trace"))
    if traces is None:
        raise StopDifferentialError("provider fixture has no stop_trace or stop_traces entries")
    if isinstance(traces, Mapping):
        trace_items = (traces,)
    elif isinstance(traces, list) and all(isinstance(item, Mapping) for item in traces):
        trace_items = tuple(traces)
    else:
        raise StopDifferentialError("stop_trace(s) must be an object or list of objects")
    cases = tuple(_case_from_mapping(item, index, default_family=default_family) for index, item in enumerate(trace_items))
    if not cases:
        raise StopDifferentialError("stop trace fixture contains no cases")
    return cases


def simulate_stop_trace(stop_policy: StopPolicyArtifact, case: StopTraceCase) -> StopSimulationResult:
    """Run the bounded local stop matcher for one recorded generation trace."""

    family = _canonical_family(case.family or stop_policy.source_family or "")
    include_stop = family == "huggingface"
    cumulative = ""
    for chunk in case.chunks:
        cumulative += chunk
        match = _first_string_stop(cumulative, stop_policy.stop_sequences)
        if match is None:
            continue
        stop, offset = match
        end = offset + len(stop)
        return StopSimulationResult(
            stopped=True,
            output=cumulative[:end] if include_stop else cumulative[:offset],
            matched_stop=stop,
            finish_reason="stop",
            stop_offset=offset,
            include_stop_in_output=include_stop,
        )

    token_match = _first_token_stop(case.token_ids, stop_policy.stop_token_ids)
    if token_match is not None:
        token_id, index = token_match
        return StopSimulationResult(
            stopped=True,
            output=cumulative,
            matched_stop=f"token:{token_id}",
            finish_reason="stop",
            token_index=index,
            include_stop_in_output=False,
        )

    return StopSimulationResult(stopped=False, output=cumulative, finish_reason="length")


def _case_from_mapping(data: Mapping[str, Any], index: int, *, default_family: str) -> StopTraceCase:
    name = data.get("name", f"trace-{index}")
    family = data.get("family", default_family)
    chunks = data.get("chunks", data.get("text_chunks"))
    token_ids = data.get("token_ids", [])
    expected = data.get("expected")
    if not isinstance(name, str) or not name:
        raise StopDifferentialError("stop trace field 'name' must be a non-empty string")
    if not isinstance(family, str) or not family:
        raise StopDifferentialError(f"stop trace '{name}' field 'family' must be a non-empty string")
    if not isinstance(chunks, list) or not all(isinstance(item, str) for item in chunks):
        raise StopDifferentialError(f"stop trace '{name}' field 'chunks' must be a list of strings")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in token_ids):
        raise StopDifferentialError(f"stop trace '{name}' field 'token_ids' must be a list of integers")
    if not isinstance(expected, Mapping):
        raise StopDifferentialError(f"stop trace '{name}' field 'expected' must be an object")
    return StopTraceCase(
        name=name,
        family=_canonical_family(family),
        chunks=tuple(chunks),
        token_ids=tuple(token_ids),
        expectation=_expectation_from_mapping(name, expected),
    )


def _expectation_from_mapping(name: str, data: Mapping[str, Any]) -> StopTraceExpectation:
    stopped = data.get("stopped")
    output = data.get("output")
    matched_stop = data.get("matched_stop")
    finish_reason = data.get("finish_reason")
    include_stop = data.get("include_stop_in_output")
    if not isinstance(stopped, bool):
        raise StopDifferentialError(f"stop trace '{name}' expected.stopped must be a boolean")
    if not isinstance(output, str):
        raise StopDifferentialError(f"stop trace '{name}' expected.output must be a string")
    if matched_stop is not None and not isinstance(matched_stop, str):
        raise StopDifferentialError(f"stop trace '{name}' expected.matched_stop must be a string or null")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise StopDifferentialError(f"stop trace '{name}' expected.finish_reason must be a string or null")
    if include_stop is not None and not isinstance(include_stop, bool):
        raise StopDifferentialError(f"stop trace '{name}' expected.include_stop_in_output must be a boolean or null")
    return StopTraceExpectation(
        stopped=stopped,
        output=output,
        matched_stop=matched_stop,
        finish_reason=finish_reason,
        include_stop_in_output=include_stop,
    )


def _first_string_stop(text: str, stop_sequences: tuple[str, ...]) -> tuple[str, int] | None:
    candidates = tuple(
        (offset, sequence)
        for sequence in stop_sequences
        if (offset := text.find(sequence)) >= 0
    )
    if not candidates:
        return None
    offset, sequence = sorted(candidates, key=lambda item: (item[0], -len(item[1]), item[1]))[0]
    return sequence, offset


def _first_token_stop(token_ids: tuple[int, ...], stop_token_ids: tuple[int, ...]) -> tuple[int, int] | None:
    if not stop_token_ids:
        return None
    stops = set(stop_token_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in stops:
            return token_id, index
    return None


def _mismatch_fields(expected: StopTraceExpectation, actual: StopSimulationResult) -> tuple[str, ...]:
    fields: list[str] = []
    if expected.stopped != actual.stopped:
        fields.append("stopped")
    if expected.output != actual.output:
        fields.append("output")
    if expected.matched_stop is not None and expected.matched_stop != actual.matched_stop:
        fields.append("matched_stop")
    if expected.finish_reason is not None and expected.finish_reason != actual.finish_reason:
        fields.append("finish_reason")
    if (
        expected.include_stop_in_output is not None
        and expected.include_stop_in_output != actual.include_stop_in_output
    ):
        fields.append("include_stop_in_output")
    return tuple(fields)


def _canonical_family(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if "huggingface" in normalized or "transformers" in normalized:
        return "huggingface"
    if "llama" in normalized or "ollama" in normalized:
        return "llama.cpp"
    if "vllm" in normalized:
        return "vllm"
    if "litellm" in normalized:
        return "litellm"
    if "openai" in normalized or "responses" in normalized or "chat-completions" in normalized:
        return "openai-compatible"
    return normalized or "framework-wrapper"
