"""Typed diagnostics shared by the CLI and embedding API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiagnosticSeverity(StrEnum):
    """Severity levels used across human, JSON, and SARIF renderers."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return {
            DiagnosticSeverity.ERROR: 0,
            DiagnosticSeverity.WARNING: 1,
            DiagnosticSeverity.INFO: 2,
        }[self]

    @property
    def sarif_level(self) -> str:
        return {
            DiagnosticSeverity.ERROR: "error",
            DiagnosticSeverity.WARNING: "warning",
            DiagnosticSeverity.INFO: "note",
        }[self]


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A stable reference to an input artifact and its provenance."""

    kind: str
    name: str
    path: str | None = None
    uri: str | None = None
    version: str | None = None
    revision: str | None = None
    sha256: str | None = None
    license: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("artifact kind must be non-empty")
        if not self.name:
            raise ValueError("artifact name must be non-empty")
        for field_name in ("path", "uri", "version", "revision", "sha256", "license", "source"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"artifact reference field '{field_name}' must be non-empty")

    @property
    def location_uri(self) -> str | None:
        return self.path if self.path is not None else self.uri

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "name": self.name}
        for key in ("path", "uri", "version", "revision", "sha256", "license", "source"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A one-based source span inside a local artifact."""

    path: str
    start_line: int = 1
    start_column: int = 1
    end_line: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.start_column < 1:
            raise ValueError("source spans are one-based")
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if (
            self.end_line == self.start_line
            and self.end_column is not None
            and self.end_column < self.start_column
        ):
            raise ValueError("end_column must be greater than or equal to start_column")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "start_line": self.start_line,
            "start_column": self.start_column,
        }
        if self.end_line is not None:
            data["end_line"] = self.end_line
        if self.end_column is not None:
            data["end_column"] = self.end_column
        return data


@dataclass(frozen=True, slots=True)
class WitnessStep:
    """One reproducible action or observation inside a diagnostic witness."""

    action: str
    input: str | None = None
    output: str | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("witness step action must be non-empty")
        if self.input is not None and not self.input:
            raise ValueError("witness step input must be non-empty")
        if self.output is not None and not self.output:
            raise ValueError("witness step output must be non-empty")

    def to_dict(self) -> dict[str, str]:
        data = {"action": self.action}
        if self.input is not None:
            data["input"] = self.input
        if self.output is not None:
            data["output"] = self.output
        return data


@dataclass(frozen=True, slots=True)
class WitnessTrace:
    """A reproducible trace showing why a diagnostic was emitted."""

    summary: str
    steps: tuple[str | WitnessStep, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary:
            raise ValueError("witness summary must be non-empty")
        normalized_steps = tuple(
            step if isinstance(step, WitnessStep) else WitnessStep(action=step) for step in self.steps
        )
        object.__setattr__(self, "steps", normalized_steps)
        object.__setattr__(self, "artifacts", tuple(sorted(self.artifacts, key=lambda item: (item.kind, item.name))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A deterministic diagnostic that can be rendered in multiple formats."""

    rule_id: str
    severity: DiagnosticSeverity
    message: str
    artifact: ArtifactRef | None = None
    span: SourceSpan | None = None
    witness: WitnessTrace | None = None
    suggestions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("diagnostic rule_id must be non-empty")
        if not self.message:
            raise ValueError("diagnostic message must be non-empty")
        if any(not suggestion for suggestion in self.suggestions):
            raise ValueError("diagnostic suggestions must be non-empty")
        object.__setattr__(self, "suggestions", tuple(self.suggestions))

    @property
    def fingerprint(self) -> str:
        stable_payload = {
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "message": self.message,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "span": self.span.to_dict() if self.span is not None else None,
        }
        encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def sort_key(self) -> tuple[int, str, str, str, str]:
        span_path = self.span.path if self.span is not None else ""
        artifact_name = self.artifact.name if self.artifact is not None else ""
        return (self.severity.rank, self.rule_id, artifact_name, span_path, self.message)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "fingerprint": self.fingerprint,
            "suggestions": list(self.suggestions),
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact.to_dict()
        if self.span is not None:
            data["span"] = self.span.to_dict()
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        return data


def diagnostic_sort_key(diagnostic: Diagnostic) -> tuple[int, str, str, str, str]:
    """Return the canonical ordering key for deterministic diagnostic output."""

    return diagnostic.sort_key
