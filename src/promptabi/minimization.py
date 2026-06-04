"""Failure-preserving minimizers for compact PromptABI repros."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .diagnostics import WitnessStep, WitnessTrace


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
FailurePredicate = Callable[[JsonValue], bool]


class MinimizationKind(StrEnum):
    """Prompt-interface failure surfaces with domain-specific shrinking order."""

    TEMPLATE = "template"
    SCHEMA = "schema"
    STOP_STRINGS = "stop-strings"
    MESSAGE_SET = "message-set"
    SOLVER_CONSTRAINTS = "solver-constraints"
    PROVIDER_FIXTURE = "provider-fixture"


class MinimizationOracle(StrEnum):
    """Safe built-in CLI oracles that do not execute arbitrary user code."""

    CONTAINS = "contains"
    DIAGNOSTIC = "diagnostic"


class MinimizationError(ValueError):
    """Raised when a repro cannot be minimized soundly."""


@dataclass(frozen=True, slots=True)
class MinimizationStep:
    """One accepted shrink that preserved the failure predicate."""

    action: str
    before_size: int
    after_size: int
    candidate: JsonValue

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "before_size": self.before_size,
            "after_size": self.after_size,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class MinimizationStats:
    """Cost counters for a deterministic minimization run."""

    predicate_calls: int
    cache_hits: int
    accepted_steps: int
    original_size: int
    minimized_size: int
    hit_step_limit: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_calls": self.predicate_calls,
            "cache_hits": self.cache_hits,
            "accepted_steps": self.accepted_steps,
            "original_size": self.original_size,
            "minimized_size": self.minimized_size,
            "hit_step_limit": self.hit_step_limit,
        }


@dataclass(frozen=True, slots=True)
class MinimizationResult:
    """A minimized upstream-ready repro plus the shrink trace that produced it."""

    kind: MinimizationKind
    original: JsonValue
    minimized: JsonValue
    steps: tuple[MinimizationStep, ...]
    stats: MinimizationStats

    @property
    def changed(self) -> bool:
        return self.original != self.minimized

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "changed": self.changed,
            "original_size": self.stats.original_size,
            "minimized_size": self.stats.minimized_size,
            "minimized": self.minimized,
            "steps": [step.to_dict() for step in self.steps],
            "stats": self.stats.to_dict(),
        }

    def witness(self) -> WitnessTrace:
        witness_steps: list[WitnessStep] = [
            WitnessStep(
                action="validate original repro",
                input=self.kind.value,
                output=f"{self.stats.original_size} JSON bytes",
            )
        ]
        for step in self.steps:
            witness_steps.append(
                WitnessStep(
                    action=step.action,
                    input=f"{step.before_size} JSON bytes",
                    output=f"{step.after_size} JSON bytes",
                )
            )
        witness_steps.append(
            WitnessStep(
                action="validate minimized repro",
                input=self.kind.value,
                output=f"{self.stats.minimized_size} JSON bytes",
            )
        )
        return WitnessTrace(
            summary=f"{self.kind.value} repro minimized from {self.stats.original_size} to {self.stats.minimized_size} JSON bytes",
            steps=tuple(witness_steps),
        )


class _PredicateRunner:
    def __init__(self, predicate: FailurePredicate, *, max_steps: int | None) -> None:
        self.predicate = predicate
        self.max_steps = max_steps
        self.calls = 0
        self.cache_hits = 0
        self.accepted_steps = 0
        self.hit_step_limit = False
        self._cache: dict[str, bool] = {}

    def evaluate(self, value: JsonValue) -> bool:
        key = _stable_json(value)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.calls += 1
        result = bool(self.predicate(value))
        self._cache[key] = result
        return result

    def can_accept(self) -> bool:
        if self.max_steps is None:
            return True
        if self.accepted_steps < self.max_steps:
            return True
        self.hit_step_limit = True
        return False

    def accept(self) -> None:
        self.accepted_steps += 1


def minimize_repro(
    value: JsonValue,
    predicate: FailurePredicate,
    *,
    kind: str | MinimizationKind,
    max_steps: int | None = None,
) -> MinimizationResult:
    """Shrink ``value`` while ``predicate(candidate)`` still reproduces the failure."""

    try:
        minimization_kind = MinimizationKind(kind)
    except ValueError as exc:
        raise MinimizationError(f"unsupported minimization kind: {kind}") from exc
    runner = _PredicateRunner(predicate, max_steps=max_steps)
    original = _canonicalize_for_kind(minimization_kind, value)
    if not runner.evaluate(original):
        raise MinimizationError("the original repro does not satisfy the failure predicate")

    minimized, steps = _shrink_value(original, runner, path=(minimization_kind.value,))
    if not runner.evaluate(minimized):
        raise MinimizationError("internal minimizer error: minimized repro no longer satisfies the predicate")

    return MinimizationResult(
        kind=minimization_kind,
        original=original,
        minimized=minimized,
        steps=tuple(steps),
        stats=MinimizationStats(
            predicate_calls=runner.calls,
            cache_hits=runner.cache_hits,
            accepted_steps=runner.accepted_steps,
            original_size=_json_size(original),
            minimized_size=_json_size(minimized),
            hit_step_limit=runner.hit_step_limit,
        ),
    )


def render_minimization_json(result: MinimizationResult) -> str:
    """Render a minimization result as stable JSON."""

    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


def render_minimization_text(result: MinimizationResult) -> str:
    """Render a concise human-readable minimization report."""

    lines = [
        f"PromptABI minimization: {result.kind.value}",
        f"status: {'SHRUNK' if result.changed else 'UNCHANGED'}",
        f"size: {result.stats.original_size} -> {result.stats.minimized_size} JSON bytes",
        f"predicate calls: {result.stats.predicate_calls} ({result.stats.cache_hits} cache hits)",
        f"accepted steps: {result.stats.accepted_steps}",
    ]
    if result.stats.hit_step_limit:
        lines.append("limit: hit max accepted shrink steps")
    if result.steps:
        lines.append("shrinks:")
        for index, step in enumerate(result.steps, start=1):
            lines.append(f"  {index}. {step.action}: {step.before_size} -> {step.after_size}")
    lines.append("minimized:")
    lines.append(json.dumps(result.minimized, indent=2, sort_keys=True))
    return "\n".join(lines) + "\n"


def load_minimization_case(path: str | Path) -> tuple[MinimizationKind, JsonValue]:
    """Load a JSON minimization case with ``kind`` and ``input`` fields."""

    case_path = Path(path)
    try:
        raw = json.loads(case_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MinimizationError(f"minimization case not found: {case_path}") from exc
    except json.JSONDecodeError as exc:
        raise MinimizationError(
            f"minimization case is not valid JSON at {case_path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise MinimizationError("minimization case root must be a JSON object")
    raw_kind = raw.get("kind")
    if not isinstance(raw_kind, str):
        raise MinimizationError("minimization case field 'kind' must be a string")
    if "input" not in raw:
        raise MinimizationError("minimization case must include an 'input' field")
    try:
        kind = MinimizationKind(raw_kind)
    except ValueError as exc:
        raise MinimizationError(f"unsupported minimization kind: {raw_kind}") from exc
    return kind, _as_json_value(raw["input"])


def contains_oracle(needle: str) -> FailurePredicate:
    """Return an oracle that preserves candidates whose JSON/text contains ``needle``."""

    if not needle:
        raise MinimizationError("contains oracle requires a non-empty substring")

    def _predicate(value: JsonValue) -> bool:
        if isinstance(value, str):
            return needle in value
        return needle in _stable_json(value)

    return _predicate


def diagnostic_oracle(
    *,
    config_path: str | Path,
    artifact_name: str,
    rule_id: str,
    case_path: str | Path,
) -> FailurePredicate:
    """Return an oracle that writes candidates to an artifact and checks a rule fires."""

    from .config import load_config
    from .session import VerificationSession

    config = load_config(config_path)
    target_path = Path(case_path)

    def _predicate(value: JsonValue) -> bool:
        target_path.write_text(_artifact_text(value), encoding="utf-8")
        config_with_override = config.with_artifact_overrides({artifact_name: str(target_path)}, base_dir=Path.cwd())
        result = VerificationSession(config_with_override).run()
        return any(diagnostic.rule_id == rule_id for diagnostic in result.diagnostics)

    return _predicate


def _shrink_value(
    value: JsonValue,
    runner: _PredicateRunner,
    *,
    path: tuple[str, ...],
) -> tuple[JsonValue, list[MinimizationStep]]:
    steps: list[MinimizationStep] = []
    current = value
    while runner.can_accept():
        for action, next_value in _candidate_values(current, path=path):
            if runner.evaluate(next_value):
                steps.append(_accepted_step(action, current, next_value))
                runner.accept()
                current = next_value
                break
        else:
            break
    return current, steps


def _candidate_values(value: JsonValue, *, path: tuple[str, ...]) -> tuple[tuple[str, JsonValue], ...]:
    label = ".".join(path)
    if isinstance(value, str):
        return _string_candidate(value, label)
    if isinstance(value, list):
        return _list_candidate(value, label)
    if isinstance(value, dict):
        return _dict_candidate(value, label)
    return ()


def _string_candidate(value: str, label: str) -> tuple[tuple[str, JsonValue], ...]:
    candidates: list[tuple[str, JsonValue]] = []
    if "\n" in value:
        lines = value.splitlines(keepends=True)
        if len(lines) > 1:
            candidates.extend(
                (f"remove line {index + 1} from {label}", "".join((*lines[:index], *lines[index + 1 :])))
                for index in range(len(lines))
            )
    if len(value) > 1:
        for chunk_size in _chunk_sizes(len(value)):
            for start in range(0, len(value), chunk_size):
                candidate = value[:start] + value[start + chunk_size :]
                if candidate:
                    candidates.append((f"remove {min(chunk_size, len(value) - start)} chars from {label}", candidate))
    return _dedupe_candidates(candidates, original=value)


def _list_candidate(value: list[JsonValue], label: str) -> tuple[tuple[str, JsonValue], ...]:
    if not value:
        return ()
    candidates: list[tuple[str, JsonValue]] = []
    for chunk_size in _chunk_sizes(len(value)):
        for start in range(0, len(value), chunk_size):
            candidate = [*value[:start], *value[start + chunk_size :]]
            if candidate:
                candidates.append((f"remove {min(chunk_size, len(value) - start)} items from {label}", candidate))
    for index, item in enumerate(value):
        for action, nested in _candidate_values(item, path=(*label.split("."), str(index))):
            candidate = list(value)
            candidate[index] = nested
            candidates.append((action, candidate))
    return _dedupe_candidates(candidates, original=value)


def _dict_candidate(value: dict[str, JsonValue], label: str) -> tuple[tuple[str, JsonValue], ...]:
    if len(value) <= 1:
        return ()
    keys = sorted(value)
    candidates: list[tuple[str, JsonValue]] = []
    for chunk_size in _chunk_sizes(len(keys)):
        for start in range(0, len(keys), chunk_size):
            removed = set(keys[start : start + chunk_size])
            candidate = {key: value[key] for key in keys if key not in removed}
            if candidate:
                candidates.append((f"remove {min(chunk_size, len(keys) - start)} keys from {label}", candidate))
    for key in keys:
        for action, nested in _candidate_values(value[key], path=(*label.split("."), key)):
            candidate = dict(value)
            candidate[key] = nested
            candidates.append((action, candidate))
    return _dedupe_candidates(candidates, original=value)


def _chunk_sizes(length: int) -> tuple[int, ...]:
    sizes: list[int] = []
    size = max(1, length // 2)
    while size >= 1:
        sizes.append(size)
        if size == 1:
            break
        size //= 2
    return tuple(dict.fromkeys(sizes))


def _dedupe_candidates(
    candidates: Sequence[tuple[str, JsonValue]],
    *,
    original: JsonValue,
) -> tuple[tuple[str, JsonValue], ...]:
    seen = {_stable_json(original)}
    deduped: list[tuple[str, JsonValue]] = []
    for action, candidate in candidates:
        key = _stable_json(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, candidate))
    return tuple(sorted(deduped, key=lambda item: (_json_size(item[1]), item[0], _stable_json(item[1]))))


def _accepted_step(action: str, before: JsonValue, after: JsonValue) -> MinimizationStep:
    return MinimizationStep(
        action=action,
        before_size=_json_size(before),
        after_size=_json_size(after),
        candidate=after,
    )


def _canonicalize_for_kind(kind: MinimizationKind, value: JsonValue) -> JsonValue:
    if kind is MinimizationKind.STOP_STRINGS:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return sorted(dict.fromkeys(value))
        raise MinimizationError("stop-strings minimization input must be a string or list of strings")
    if kind in (MinimizationKind.MESSAGE_SET, MinimizationKind.SOLVER_CONSTRAINTS):
        if not isinstance(value, list):
            raise MinimizationError(f"{kind.value} minimization input must be a JSON array")
        return _canonical_json(value)
    if kind in (MinimizationKind.SCHEMA, MinimizationKind.PROVIDER_FIXTURE):
        if not isinstance(value, dict):
            raise MinimizationError(f"{kind.value} minimization input must be a JSON object")
        return _canonical_json(value)
    if kind is MinimizationKind.TEMPLATE:
        if not isinstance(value, str):
            raise MinimizationError("template minimization input must be a string")
        return value
    return _canonical_json(value)


def _canonical_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def _stable_json(value: JsonValue) -> str:
    return json.dumps(_canonical_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_size(value: JsonValue) -> int:
    return len(_stable_json(value).encode("utf-8"))


def _as_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_as_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MinimizationError("minimization inputs must use string JSON object keys")
            normalized[key] = _as_json_value(item)
        return normalized
    raise MinimizationError(f"unsupported JSON value in minimization input: {type(value).__name__}")


def _artifact_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
