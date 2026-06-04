"""Verification session orchestration for PromptABI."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactKind
from .chat_templates import ChatTemplateParseError, parse_hf_tokenizer_config_chat_template
from .config import VerificationConfig, load_config
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessStep, WitnessTrace, diagnostic_sort_key
from .loaders import ArtifactLoadError, ArtifactLoadWarning, ArtifactLoader, LoadedArtifact
from .role_boundaries import RoleBoundaryForgeryFinding, analyze_role_boundary_nonforgeability


CHECK_MODE_CATALOG: dict[str, tuple[CheckMode, ...]] = {
    "repository-skeleton": (CheckMode.HEURISTIC,),
    "artifact-missing": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-load-failed": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-unpinned": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-weak-pin": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-pin-invalid": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-hash-mismatch": (CheckMode.SOUND, CheckMode.COMPLETE),
    "role-boundary-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "role-boundary-nonforgeability": (CheckMode.SOUND, CheckMode.BOUNDED),
    "check-unknown": (CheckMode.SOUND, CheckMode.COMPLETE),
    "check-failed": (CheckMode.HEURISTIC,),
}


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Inputs available to public and built-in verification checks."""

    config: VerificationConfig
    loaded_artifacts: tuple[LoadedArtifact, ...]

    def artifact(self, name: str) -> LoadedArtifact:
        for loaded in self.loaded_artifacts:
            if loaded.artifact.name == name:
                return loaded
        raise KeyError(name)


CheckCallable = Callable[[CheckContext], Iterable[Diagnostic]]


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
    """A public verification session that embedding tools can extend."""

    def __init__(
        self,
        config: VerificationConfig,
        *,
        checks: Mapping[str, CheckCallable] | None = None,
        loader: ArtifactLoader | None = None,
    ) -> None:
        self.config = config
        self.loader = loader or ArtifactLoader()
        self.checks: dict[str, CheckCallable] = {
            "repository-skeleton": self._repository_skeleton_check,
            "role-boundary-nonforgeability": self._role_boundary_nonforgeability_check,
        }
        if checks:
            self.checks.update(checks)

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        checks: Mapping[str, CheckCallable] | None = None,
        loader: ArtifactLoader | None = None,
    ) -> "VerificationSession":
        return cls(load_config(path), checks=checks, loader=loader)

    def load_artifacts(self) -> tuple[LoadedArtifact, ...]:
        """Load all configured artifacts or raise the first deterministic loader error."""

        loaded_artifacts, diagnostics = self._load_artifacts_with_diagnostics()
        fatal = next((diagnostic for diagnostic in diagnostics if diagnostic.severity is DiagnosticSeverity.ERROR), None)
        if fatal is not None:
            raise ArtifactLoadError(
                rule_id=fatal.rule_id,
                message=fatal.message,
                suggestion=fatal.suggestions[0] if fatal.suggestions else "Inspect the diagnostic for details.",
            )
        return loaded_artifacts

    def collect_diagnostics(self, *, checks: Sequence[str | CheckCallable] | None = None) -> tuple[Diagnostic, ...]:
        """Run preflight loading plus selected checks and return sorted diagnostics."""

        loaded_artifacts, diagnostics = self._load_artifacts_with_diagnostics()
        context = CheckContext(config=self.config, loaded_artifacts=loaded_artifacts)
        diagnostics.extend(self._check_diagnostics(context, checks or self.config.checks))
        diagnostics.sort(key=diagnostic_sort_key)
        return tuple(diagnostics)

    def run(self, *, checks: Sequence[str | CheckCallable] | None = None) -> VerificationResult:
        diagnostics = self.collect_diagnostics(checks=checks)
        return VerificationResult(config=self.config, diagnostics=tuple(diagnostics))

    def _repository_skeleton_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        return (
            Diagnostic(
                rule_id="repository-skeleton",
                severity=DiagnosticSeverity.INFO,
                message="PromptABI package, CLI, docs, examples, fixtures, and benchmarks are wired.",
                check_modes=CHECK_MODE_CATALOG["repository-skeleton"],
                witness=WitnessTrace(
                    summary="The verification session constructed a typed config and produced deterministic output.",
                    steps=(
                        WitnessStep(
                            action="load JSON config",
                            input=context.config.name,
                            output=f"{len(context.config.artifact_bundle.artifacts)} artifacts",
                        ),
                        WitnessStep(action="normalize artifact paths"),
                        WitnessStep(
                            action="load artifacts",
                            output=f"{len(context.loaded_artifacts)} loaded",
                        ),
                        WitnessStep(action="render stable diagnostics"),
                    ),
                ),
            ),
        )

    def _role_boundary_nonforgeability_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for loaded in context.loaded_artifacts:
            artifact = loaded.artifact
            if artifact.kind is not ArtifactKind.CHAT_TEMPLATE or artifact.location.path is None:
                continue
            path = Path(artifact.location.path)
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            try:
                parsed = parse_hf_tokenizer_config_chat_template(path)
            except ChatTemplateParseError as exc:
                diagnostics.append(
                    Diagnostic(
                        rule_id="role-boundary-abstained",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"chat-template artifact '{artifact.name}' is outside the supported role-boundary fragment",
                        artifact=artifact.to_ref(),
                        span=artifact.source_span,
                        check_modes=CHECK_MODE_CATALOG["role-boundary-abstained"],
                        suggestions=("Simplify the chat template or add a supported minimized fixture.",),
                        witness=WitnessTrace(
                            summary="PromptABI could not parse the chat template for bounded role-boundary analysis.",
                            steps=(WitnessStep(action="parse chat template", input=str(path), output=str(exc)),),
                            artifacts=(artifact.to_ref(),),
                        ),
                    )
                )
                continue
            report = analyze_role_boundary_nonforgeability(parsed)
            if not report.model.supported:
                diagnostics.append(
                    Diagnostic(
                        rule_id="role-boundary-abstained",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"chat-template artifact '{artifact.name}' uses constructs outside bounded role analysis",
                        artifact=artifact.to_ref(),
                        span=parsed.source_span or artifact.source_span,
                        check_modes=CHECK_MODE_CATALOG["role-boundary-abstained"],
                        suggestions=("Review the symbolic abstentions before trusting non-forgeability results.",),
                        witness=WitnessTrace(
                            summary="The bounded symbolic executor abstained on part of the template.",
                            steps=tuple(
                                WitnessStep(action="abstain on template construct", output=abstention)
                                for abstention in report.model.abstentions
                            ),
                            artifacts=(artifact.to_ref(),),
                        ),
                    )
                )
            diagnostics.extend(
                _role_boundary_forgery_diagnostic(artifact.to_ref(), parsed.source_span, finding)
                for finding in report.findings
            )
        return tuple(diagnostics)

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

    def _load_artifacts_with_diagnostics(self) -> tuple[tuple[LoadedArtifact, ...], list[Diagnostic]]:
        missing_paths = self._missing_local_paths()
        diagnostics = list(self._artifact_existence_diagnostics(missing_paths))
        loaded_artifacts: list[LoadedArtifact] = []
        for artifact_model in self.config.artifact_bundle:
            if artifact_model.location.path in missing_paths:
                continue
            try:
                loaded = self.loader.load(artifact_model)
            except ArtifactLoadError as exc:
                diagnostics.append(self._load_error_diagnostic(artifact_model, exc))
                continue
            loaded_artifacts.append(loaded)
            for warning in loaded.warnings:
                diagnostics.append(self._load_warning_diagnostic(artifact_model, warning))
        return tuple(loaded_artifacts), diagnostics

    def _check_diagnostics(
        self,
        context: CheckContext,
        checks: Sequence[str | CheckCallable],
    ) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for check in checks:
            if isinstance(check, str):
                check_name = check
                check_callable = self.checks.get(check)
                if check_callable is None:
                    diagnostics.append(_unknown_check_diagnostic(check_name))
                    continue
            else:
                check_name = getattr(check, "__name__", "embedded-check")
                check_callable = check
            try:
                diagnostics.extend(tuple(check_callable(context)))
            except Exception as exc:
                diagnostics.append(_failed_check_diagnostic(check_name, exc))
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


def _role_boundary_forgery_diagnostic(artifact, span, finding: RoleBoundaryForgeryFinding) -> Diagnostic:
    return Diagnostic(
        rule_id="role-boundary-nonforgeability",
        severity=DiagnosticSeverity.ERROR,
        message=(
            f"{finding.input_expression} can forge {finding.marker_kind} {finding.marker!r} "
            f"in a {finding.input_role} region"
        ),
        artifact=artifact,
        span=span,
        check_modes=CHECK_MODE_CATALOG["role-boundary-nonforgeability"],
        suggestions=(
            "Render user-controlled fields through an escaping or encoding layer before adjacent role delimiters.",
            "Avoid raw dynamic role headers; map roles through an explicit allowlist.",
        ),
        witness=WitnessTrace(
            summary=finding.boundary_description,
            steps=(
                WitnessStep(
                    action="build bounded role-region model",
                    output=f"path {finding.path_index}, region {finding.region_index}",
                ),
                WitnessStep(
                    action="substitute attacker-controlled field",
                    input=finding.input_expression,
                    output=finding.malicious_input,
                ),
                WitnessStep(action="render forged boundary excerpt", output=finding.rendered_excerpt),
                WitnessStep(action="tokenize forged excerpt", output=finding.tokenized_representation),
                WitnessStep(action="locate forged boundary", output=finding.forged_boundary),
            ),
            artifacts=(artifact,),
        ),
    )


def _catalog_modes(rule_id: str) -> tuple[CheckMode, ...]:
    return CHECK_MODE_CATALOG.get(rule_id, (CheckMode.HEURISTIC,))


def _unknown_check_diagnostic(check_name: str) -> Diagnostic:
    return Diagnostic(
        rule_id="check-unknown",
        severity=DiagnosticSeverity.ERROR,
        message=f"configured check '{check_name}' is not registered",
        check_modes=CHECK_MODE_CATALOG["check-unknown"],
        suggestions=("Register the check with VerificationSession(checks=...) or remove it from the config.",),
        witness=WitnessTrace(
            summary="The config requested a check that the session cannot execute.",
            steps=(WitnessStep(action="resolve check", input=check_name, output="not registered"),),
        ),
    )


def _failed_check_diagnostic(check_name: str, exc: Exception) -> Diagnostic:
    return Diagnostic(
        rule_id="check-failed",
        severity=DiagnosticSeverity.ERROR,
        message=f"check '{check_name}' raised {type(exc).__name__}: {exc}",
        check_modes=CHECK_MODE_CATALOG["check-failed"],
        suggestions=("Fix the embedded check or let the exception propagate before creating diagnostics.",),
        witness=WitnessTrace(
            summary="PromptABI converted an embedded check failure into a deterministic diagnostic.",
            steps=(WitnessStep(action="run check", input=check_name, output=type(exc).__name__),),
        ),
    )
