"""Typed diagnostics shared by the CLI and embedding API."""

from __future__ import annotations

import hashlib
import json
import re
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


class CheckMode(StrEnum):
    """Verification-contract modes a PromptABI check may claim."""

    SOUND = "sound"
    COMPLETE = "complete"
    BOUNDED = "bounded"
    Z3_BACKED_SMT = "z3-backed-smt"
    HEURISTIC = "heuristic"
    ABSTAINING = "abstaining"

    @property
    def label(self) -> str:
        return {
            CheckMode.SOUND: "sound",
            CheckMode.COMPLETE: "complete",
            CheckMode.BOUNDED: "bounded",
            CheckMode.Z3_BACKED_SMT: "Z3-backed SMT",
            CheckMode.HEURISTIC: "heuristic",
            CheckMode.ABSTAINING: "abstaining",
        }[self]

    @property
    def description(self) -> str:
        return CHECK_MODE_DESCRIPTIONS[self]


CHECK_MODE_DESCRIPTIONS: dict[CheckMode, str] = {
    CheckMode.SOUND: "The check does not report a violation unless one exists under the stated abstraction.",
    CheckMode.COMPLETE: "The check finds every violation inside the stated supported fragment.",
    CheckMode.BOUNDED: "The check is exact only within declared finite limits such as depth, length, or domains.",
    CheckMode.Z3_BACKED_SMT: "The check lowers a finite symbolic contract to Z3 when the optional solver is available.",
    CheckMode.HEURISTIC: "The check is useful evidence but is not a proof over a fully modeled fragment.",
    CheckMode.ABSTAINING: "The check explicitly declines to decide cases outside its supported fragment.",
}

LOCALIZATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
LOCALIZATION_ARG_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(data["kind"]),
            name=str(data["name"]),
            path=_optional_str_value(data.get("path")),
            uri=_optional_str_value(data.get("uri")),
            version=_optional_str_value(data.get("version")),
            revision=_optional_str_value(data.get("revision")),
            sha256=_optional_str_value(data.get("sha256")),
            license=_optional_str_value(data.get("license")),
            source=_optional_str_value(data.get("source")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceSpan":
        return cls(
            path=str(data["path"]),
            start_line=int(data.get("start_line", 1)),
            start_column=int(data.get("start_column", 1)),
            end_line=_optional_int_value(data.get("end_line")),
            end_column=_optional_int_value(data.get("end_column")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WitnessStep":
        return cls(
            action=str(data["action"]),
            input=_optional_str_value(data.get("input")),
            output=_optional_str_value(data.get("output")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WitnessTrace":
        return cls(
            summary=str(data["summary"]),
            steps=tuple(WitnessStep.from_dict(step) for step in data.get("steps", ())),
            artifacts=tuple(ArtifactRef.from_dict(artifact) for artifact in data.get("artifacts", ())),
        )


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
    check_modes: tuple[CheckMode | str, ...] = field(default_factory=tuple)
    properties: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    message_id: str | None = None
    message_args: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("diagnostic rule_id must be non-empty")
        if not self.message:
            raise ValueError("diagnostic message must be non-empty")
        if any(not suggestion for suggestion in self.suggestions):
            raise ValueError("diagnostic suggestions must be non-empty")
        object.__setattr__(self, "suggestions", tuple(self.suggestions))
        modes = tuple(_coerce_check_mode(mode) for mode in self.check_modes)
        if len(set(modes)) != len(modes):
            raise ValueError("diagnostic check_modes must not contain duplicates")
        object.__setattr__(self, "check_modes", tuple(sorted(modes, key=lambda mode: mode.value)))
        if any(not key for key, _value in self.properties):
            raise ValueError("diagnostic property keys must be non-empty")
        object.__setattr__(self, "properties", tuple(sorted(self.properties, key=lambda item: item[0])))
        if self.message_id is not None and not LOCALIZATION_ID_PATTERN.fullmatch(self.message_id):
            raise ValueError("diagnostic message_id must be a stable dotted lowercase identifier")
        if any(not LOCALIZATION_ARG_PATTERN.fullmatch(key) for key, _value in self.message_args):
            raise ValueError("diagnostic message_args keys must be valid placeholder identifiers")
        message_args = tuple(sorted(self.message_args, key=lambda item: item[0]))
        if len({key for key, _value in message_args}) != len(message_args):
            raise ValueError("diagnostic message_args must not contain duplicate keys")
        object.__setattr__(self, "message_args", message_args)

    @property
    def localization_key(self) -> str:
        """Return the stable catalog key used for future translations."""

        if self.message_id is not None:
            return self.message_id
        normalized_rule = re.sub(r"[^a-z0-9]+", ".", self.rule_id.lower()).strip(".")
        return f"promptabi.diagnostic.{normalized_rule}"

    @property
    def fingerprint(self) -> str:
        stable_payload = {
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "message": self.message,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "span": self.span.to_dict() if self.span is not None else None,
            "check_modes": [mode.value for mode in self.check_modes],
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
            "check_modes": [mode.value for mode in self.check_modes],
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact.to_dict()
        if self.span is not None:
            data["span"] = self.span.to_dict()
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        if self.properties:
            data["properties"] = dict(self.properties)
        if self.message_id is not None or self.message_args:
            data["localization"] = {
                "message_id": self.localization_key,
                "default_locale": "en",
                "message_args": dict(self.message_args),
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Diagnostic":
        localization = data.get("localization")
        message_id = None
        message_args: tuple[tuple[str, Any], ...] = ()
        if isinstance(localization, dict):
            raw_message_id = localization.get("message_id")
            message_id = str(raw_message_id) if raw_message_id is not None else None
            raw_message_args = localization.get("message_args", {})
            if isinstance(raw_message_args, dict):
                message_args = tuple((str(key), value) for key, value in raw_message_args.items())
        return cls(
            rule_id=str(data["rule_id"]),
            severity=DiagnosticSeverity(str(data["severity"])),
            message=str(data["message"]),
            artifact=ArtifactRef.from_dict(data["artifact"]) if isinstance(data.get("artifact"), dict) else None,
            span=SourceSpan.from_dict(data["span"]) if isinstance(data.get("span"), dict) else None,
            witness=WitnessTrace.from_dict(data["witness"]) if isinstance(data.get("witness"), dict) else None,
            suggestions=tuple(str(item) for item in data.get("suggestions", ())),
            check_modes=tuple(str(item) for item in data.get("check_modes", ())),
            properties=tuple((str(key), value) for key, value in data.get("properties", {}).items())
            if isinstance(data.get("properties"), dict)
            else (),
            message_id=message_id,
            message_args=message_args,
        )


def diagnostic_sort_key(diagnostic: Diagnostic) -> tuple[int, str, str, str, str]:
    """Return the canonical ordering key for deterministic diagnostic output."""

    return diagnostic.sort_key


def _coerce_check_mode(mode: CheckMode | str) -> CheckMode:
    if isinstance(mode, CheckMode):
        return mode
    try:
        return CheckMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in CheckMode)
        raise ValueError(f"unknown check mode: {mode!r}; expected one of {choices}") from exc


def _optional_str_value(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int_value(value: object) -> int | None:
    return int(value) if value is not None else None
