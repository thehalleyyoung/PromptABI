"""Verification session orchestration for PromptABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import VerificationConfig, load_config
from .diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessTrace


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of running a verification session."""

    config: VerificationConfig
    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return not any(diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in self.diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "ok": self.ok,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


class VerificationSession:
    """A small public API surface that later checkers can plug into."""

    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    @classmethod
    def from_config_file(cls, path: str | Path) -> "VerificationSession":
        return cls(load_config(path))

    def run(self) -> VerificationResult:
        diagnostics = [self._repository_skeleton_diagnostic()]
        diagnostics.extend(self._artifact_existence_diagnostics())
        diagnostics.sort(key=lambda item: (item.severity.value, item.rule_id, item.message))
        return VerificationResult(config=self.config, diagnostics=tuple(diagnostics))

    def _repository_skeleton_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            rule_id="repository-skeleton",
            severity=DiagnosticSeverity.INFO,
            message="PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
            witness=WitnessTrace(
                summary="The verification session constructed a typed config and produced deterministic output.",
                steps=(
                    "load JSON config",
                    "normalize artifact paths",
                    "render stable diagnostics",
                ),
            ),
        )

    def _artifact_existence_diagnostics(self) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for artifact_model in self.config.artifact_bundle:
            path = artifact_model.location.path
            if path is None:
                continue
            artifact = artifact_model.to_ref()
            if not Path(path).exists():
                diagnostics.append(
                    Diagnostic(
                        rule_id="artifact-missing",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"artifact '{artifact_model.name}' does not exist",
                        artifact=artifact,
                        span=SourceSpan(path=path),
                        suggestions=("Check the path relative to the PromptABI config file.",),
                    )
                )
        return tuple(diagnostics)
