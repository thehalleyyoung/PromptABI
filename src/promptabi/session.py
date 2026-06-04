"""Verification session orchestration for PromptABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import VerificationConfig, load_config
from .diagnostics import Diagnostic, DiagnosticSeverity, SourceSpan, WitnessStep, WitnessTrace, diagnostic_sort_key


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
        diagnostics.sort(key=diagnostic_sort_key)
        return VerificationResult(config=self.config, diagnostics=tuple(diagnostics))

    def _repository_skeleton_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            rule_id="repository-skeleton",
            severity=DiagnosticSeverity.INFO,
            message="PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
            witness=WitnessTrace(
                summary="The verification session constructed a typed config and produced deterministic output.",
                steps=(
                    WitnessStep(
                        action="load JSON config",
                        input=self.config.name,
                        output=f"{len(self.config.artifact_bundle.artifacts)} artifacts",
                    ),
                    WitnessStep(action="normalize artifact paths"),
                    WitnessStep(action="render stable diagnostics"),
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
                        witness=WitnessTrace(
                            summary="The configured local artifact path was resolved but was absent on disk.",
                            steps=(
                                WitnessStep(action="resolve artifact path", output=path),
                                WitnessStep(action="check local filesystem", output="missing"),
                            ),
                            artifacts=(artifact,),
                        ),
                    )
                )
        return tuple(diagnostics)
