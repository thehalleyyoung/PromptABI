"""Telemetry-free local usage summaries for PromptABI commands."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._version import __version__

SUMMARY_SCHEMA_VERSION = 1
SAFE_METADATA_TYPES = (str, int, float, bool, type(None))


class UsageAnalyticsError(ValueError):
    """Raised when local usage summaries cannot be read or written."""


@dataclass(frozen=True, slots=True)
class UsageSummaryReport:
    """Aggregate, local-only command summary statistics."""

    path: Path
    command_count: int
    commands: dict[str, int]
    exit_codes: dict[str, int]
    latest_timestamp: str | None
    total_duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "command_count": self.command_count,
            "commands": self.commands,
            "exit_codes": self.exit_codes,
            "latest_timestamp": self.latest_timestamp,
            "total_duration_ms": self.total_duration_ms,
            "privacy": privacy_guarantees(),
        }


def resolve_usage_summary_path(value: str | Path | None = None) -> Path:
    """Resolve the local JSONL summary path without touching the network."""

    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("PROMPTABI_USAGE_SUMMARY_PATH")
    if env_value:
        return Path(env_value).expanduser().resolve()
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return (Path(xdg_state_home).expanduser() / "promptabi" / "usage-summaries.jsonl").resolve()
    return (Path.home() / ".local" / "state" / "promptabi" / "usage-summaries.jsonl").resolve()


def append_local_command_summary(
    *,
    path: str | Path | None = None,
    command: str,
    exit_code: int,
    duration_ms: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Append one sanitized command summary to a local JSONL file."""

    resolved = resolve_usage_summary_path(path)
    safe_metadata = _sanitize_metadata(metadata or {})
    record = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "promptabi_version": __version__,
        "command": command,
        "exit_code": int(exit_code),
        "duration_ms": max(0, int(duration_ms)),
        "metadata": safe_metadata,
    }
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError as exc:
        raise UsageAnalyticsError(f"cannot write local usage summary at {resolved}: {exc}") from exc
    return resolved


def load_local_command_summaries(path: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    """Load local command summaries from JSONL."""

    resolved = resolve_usage_summary_path(path)
    if not resolved.exists():
        return ()
    summaries: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UsageAnalyticsError(
                    f"invalid JSON in local usage summary {resolved}:{line_number}: {exc.msg}"
                ) from exc
            summaries.append(_validate_summary(payload, resolved, line_number))
    except OSError as exc:
        raise UsageAnalyticsError(f"cannot read local usage summary at {resolved}: {exc}") from exc
    return tuple(summaries)


def summarize_local_command_usage(path: str | Path | None = None) -> UsageSummaryReport:
    """Aggregate local command summaries without exposing raw records."""

    resolved = resolve_usage_summary_path(path)
    summaries = load_local_command_summaries(resolved)
    commands: Counter[str] = Counter()
    exit_codes: Counter[str] = Counter()
    total_duration_ms = 0
    latest_timestamp: str | None = None
    for summary in summaries:
        commands[str(summary["command"])] += 1
        exit_codes[str(summary["exit_code"])] += 1
        total_duration_ms += int(summary["duration_ms"])
        timestamp = str(summary["timestamp"])
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
    return UsageSummaryReport(
        path=resolved,
        command_count=len(summaries),
        commands=dict(sorted(commands.items())),
        exit_codes=dict(sorted(exit_codes.items())),
        latest_timestamp=latest_timestamp,
        total_duration_ms=total_duration_ms,
    )


def render_usage_summary_text(report: UsageSummaryReport) -> str:
    lines = [
        "PromptABI local usage summary",
        f"path: {report.path}",
        f"commands: {report.command_count}",
        f"total duration: {report.total_duration_ms} ms",
    ]
    if report.latest_timestamp is not None:
        lines.append(f"latest: {report.latest_timestamp}")
    if report.commands:
        lines.append("by command:")
        lines.extend(f"  {command}: {count}" for command, count in report.commands.items())
    if report.exit_codes:
        lines.append("by exit code:")
        lines.extend(f"  {exit_code}: {count}" for exit_code, count in report.exit_codes.items())
    lines.append("privacy: local JSONL only; no prompts, schemas, configs, constraints, witnesses, paths, or network sends")
    return "\n".join(lines) + "\n"


def render_usage_summary_json(report: UsageSummaryReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_usage_privacy_text() -> str:
    guarantees = privacy_guarantees()
    lines = ["PromptABI usage privacy guarantees:"]
    lines.extend(f"- {guarantee}" for guarantee in guarantees)
    return "\n".join(lines) + "\n"


def privacy_guarantees() -> tuple[str, ...]:
    return (
        "No telemetry is sent by PromptABI usage summaries.",
        "Summaries are opt-in and written only to a local JSONL file.",
        "Records contain command names, exit codes, durations, and aggregate counts only.",
        "Prompts, schemas, configs, constraints, witnesses, artifact contents, and file paths are never recorded.",
        "The summary path is controlled by --local-summary, PROMPTABI_USAGE_SUMMARY_PATH, or local state defaults.",
    )


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise UsageAnalyticsError(f"unsafe usage-summary metadata key: {key!r}")
        if not isinstance(value, SAFE_METADATA_TYPES):
            raise UsageAnalyticsError(f"unsafe usage-summary metadata value for {key!r}: {type(value).__name__}")
        safe[key] = value
    return safe


def _validate_summary(payload: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise UsageAnalyticsError(f"local usage summary {path}:{line_number} is not an object")
    required = {"schema_version", "timestamp", "promptabi_version", "command", "exit_code", "duration_ms", "metadata"}
    missing = sorted(required.difference(payload))
    if missing:
        raise UsageAnalyticsError(f"local usage summary {path}:{line_number} missing keys: {', '.join(missing)}")
    if payload["schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise UsageAnalyticsError(
            f"local usage summary {path}:{line_number} has unsupported schema_version {payload['schema_version']!r}"
        )
    if not isinstance(payload["metadata"], dict):
        raise UsageAnalyticsError(f"local usage summary {path}:{line_number} metadata is not an object")
    payload["metadata"] = _sanitize_metadata(payload["metadata"])
    return payload
