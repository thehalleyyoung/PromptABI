"""Ergonomic embedding API for PromptABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import VerificationConfig, load_config
from .diagnostics import Diagnostic
from .explain import DiagnosticExplanation, explain_diagnostic, render_explanation_json, render_explanation_text
from .loaders import ArtifactLoader, LoadedArtifact
from .render import render_json, render_sarif, render_text
from .session import CheckCallable, VerificationResult, VerificationSession


def create_session(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    checks: Mapping[str, CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
) -> VerificationSession:
    """Create a verification session from a config object or JSON config path."""

    resolved_config = _resolve_config(config)
    if artifact_overrides:
        base_dir = Path(override_base_dir) if override_base_dir is not None else Path.cwd()
        resolved_config = resolved_config.with_artifact_overrides(dict(artifact_overrides), base_dir=base_dir)
    return VerificationSession(resolved_config, checks=checks, loader=loader)


def load_artifacts(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    loader: ArtifactLoader | None = None,
) -> tuple[LoadedArtifact, ...]:
    """Load all artifacts for an embedding workflow using the default session semantics."""

    return create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        loader=loader,
    ).load_artifacts()


def collect_diagnostics(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    checks: Mapping[str, CheckCallable] | None = None,
    selected_checks: Sequence[str | CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
) -> tuple[Diagnostic, ...]:
    """Run verification and return diagnostics without constructing a result wrapper."""

    session = create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        checks=checks,
        loader=loader,
    )
    return session.collect_diagnostics(checks=selected_checks)


def run_verification(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    checks: Mapping[str, CheckCallable] | None = None,
    selected_checks: Sequence[str | CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
) -> VerificationResult:
    """Run PromptABI verification from Python and return a typed result."""

    session = create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        checks=checks,
        loader=loader,
    )
    return session.run(checks=selected_checks)


def render_result(
    result: VerificationResult,
    *,
    output_format: str = "text",
    verbosity: int = 0,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Render a typed verification result as text, JSON, or SARIF."""

    if output_format == "text":
        return render_text(
            result,
            verbosity=verbosity,
            config_path=Path(config_path) if config_path is not None else None,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
        )
    if output_format == "json":
        return render_json(result)
    if output_format == "sarif":
        return render_sarif(result)
    raise ValueError("output_format must be one of: text, json, sarif")


def explain_result(
    result: VerificationResult,
    *,
    fingerprint: str | None = None,
    rule_id: str | None = None,
    index: int | None = None,
    base_dir: str | Path | None = None,
) -> DiagnosticExplanation:
    """Select and explain one diagnostic from an existing verification result."""

    return explain_diagnostic(
        result,
        fingerprint=fingerprint,
        rule_id=rule_id,
        index=index,
        base_dir=base_dir,
    )


def render_explanation(
    explanation: DiagnosticExplanation,
    *,
    output_format: str = "text",
) -> str:
    """Render a diagnostic explanation as text or JSON."""

    if output_format == "text":
        return render_explanation_text(explanation)
    if output_format == "json":
        return render_explanation_json(explanation)
    raise ValueError("output_format must be one of: text, json")


def _resolve_config(config: str | Path | VerificationConfig) -> VerificationConfig:
    if isinstance(config, VerificationConfig):
        return config
    return load_config(config)
