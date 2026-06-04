"""Verification session orchestration for PromptABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import VerificationConfig, load_config
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessStep, WitnessTrace, diagnostic_sort_key
from .loaders import ArtifactLoadError, ArtifactLoadWarning, ArtifactLoader


CHECK_MODE_CATALOG: dict[str, tuple[CheckMode, ...]] = {
    "repository-skeleton": (CheckMode.HEURISTIC,),
    "artifact-missing": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-load-failed": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-unpinned": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-weak-pin": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-pin-invalid": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-hash-mismatch": (CheckMode.SOUND, CheckMode.COMPLETE),
}


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
        self.loader = ArtifactLoader()

    @classmethod
    def from_config_file(cls, path: str | Path) -> "VerificationSession":
        return cls(load_config(path))

    def run(self) -> VerificationResult:
        diagnostics = [self._repository_skeleton_diagnostic()]
        missing_paths = self._missing_local_paths()
        diagnostics.extend(self._artifact_existence_diagnostics(missing_paths))
        diagnostics.extend(self._artifact_loader_diagnostics(missing_paths))
        diagnostics.sort(key=diagnostic_sort_key)
        return VerificationResult(config=self.config, diagnostics=tuple(diagnostics))

    def _repository_skeleton_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            rule_id="repository-skeleton",
            severity=DiagnosticSeverity.INFO,
            message="PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
            check_modes=CHECK_MODE_CATALOG["repository-skeleton"],
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

    def _missing_local_paths(self) -> set[str]:
        return {
            artifact.location.path
            for artifact in self.config.artifact_bundle
            if artifact.location.path is not None and not Path(artifact.location.path).exists()
        }

    def _artifact_existence_diagnostics(self, missing_paths: set[str]) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for artifact_model in self.config.artifact_bundle:
            path = artifact_model.location.path
            if path is None or path not in missing_paths:
                continue
            artifact = artifact_model.to_ref()
            diagnostics.append(
                Diagnostic(
                    rule_id="artifact-missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"artifact '{artifact_model.name}' does not exist",
                    artifact=artifact,
                    span=_artifact_span(artifact_model),
                    check_modes=CHECK_MODE_CATALOG["artifact-missing"],
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

    def _artifact_loader_diagnostics(self, missing_paths: set[str]) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for artifact_model in self.config.artifact_bundle:
            if artifact_model.location.path in missing_paths:
                continue
            try:
                loaded = self.loader.load(artifact_model)
            except ArtifactLoadError as exc:
                diagnostics.append(self._load_error_diagnostic(artifact_model, exc))
                continue
            for warning in loaded.warnings:
                diagnostics.append(self._load_warning_diagnostic(artifact_model, warning))
        return tuple(diagnostics)

    def _load_error_diagnostic(self, artifact_model, exc: ArtifactLoadError) -> Diagnostic:
        artifact = artifact_model.to_ref()
        return Diagnostic(
            rule_id=exc.rule_id,
            severity=DiagnosticSeverity.ERROR,
            message=exc.message,
            artifact=artifact,
            span=exc.span or _artifact_span(artifact_model),
            check_modes=_catalog_modes(exc.rule_id),
            suggestions=(exc.suggestion,),
            witness=WitnessTrace(
                summary="PromptABI could not load the configured artifact deterministically.",
                steps=_witness_steps(exc.steps),
                artifacts=(artifact,),
            ),
        )

    def _load_warning_diagnostic(self, artifact_model, warning: ArtifactLoadWarning) -> Diagnostic:
        artifact = artifact_model.to_ref()
        return Diagnostic(
            rule_id=warning.rule_id,
            severity=DiagnosticSeverity.WARNING,
            message=warning.message,
            artifact=artifact,
            span=_artifact_span(artifact_model),
            check_modes=_catalog_modes(warning.rule_id),
            suggestions=(warning.suggestion,),
            witness=WitnessTrace(
                summary="The artifact loaded, but its provenance is not fully reproducible.",
                steps=_witness_steps(warning.steps),
                artifacts=(artifact,),
            ),
        )


def _artifact_span(artifact_model) -> SourceSpan | None:
    if artifact_model.source_span is not None:
        return artifact_model.source_span
    path = artifact_model.location.path
    return SourceSpan(path=path) if path is not None else None


def _witness_steps(raw_steps: tuple[tuple[str, str | None, str | None], ...]) -> tuple[WitnessStep, ...]:
    return tuple(
        WitnessStep(action=action, input=input_value, output=output_value)
        for action, input_value, output_value in raw_steps
    )


def _catalog_modes(rule_id: str) -> tuple[CheckMode, ...]:
    return CHECK_MODE_CATALOG.get(rule_id, (CheckMode.HEURISTIC,))
