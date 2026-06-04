"""Typed diagnostics shared by the CLI and embedding API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiagnosticSeverity(StrEnum):
    """Severity levels used across human, JSON, and future SARIF renderers."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A stable reference to an input artifact."""

    kind: str
    name: str
    path: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.path is not None:
            data["path"] = self.path
        if self.version is not None:
            data["version"] = self.version
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
class WitnessTrace:
    """A reproducible trace showing why a diagnostic was emitted."""

    summary: str
    steps: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "steps": list(self.steps),
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

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "suggestions": list(self.suggestions),
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact.to_dict()
        if self.span is not None:
            data["span"] = self.span.to_dict()
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        return data

