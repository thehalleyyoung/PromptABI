"""Ergonomic embedding API for PromptABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import VerificationConfig, load_config
from .compatibility_matrix import (
    CompatibilityMatrix,
    build_compatibility_matrix,
    render_compatibility_matrix_json,
    render_compatibility_matrix_text,
)
from .corpus_verification import (
    CorpusVerificationReport,
    CorpusVerificationThresholds,
    render_corpus_verification_json,
    render_corpus_verification_text,
    run_corpus_verification,
)
from .diagnostics import Diagnostic
from .editor_protocol import (
    EditorDiagnosticReport,
    build_editor_diagnostic_report,
    render_editor_diagnostic_json,
    render_editor_diagnostic_text,
)
from .evaluation import (
    EvaluationReport,
    render_evaluation_json,
    render_evaluation_text,
    run_evaluation,
)
from .bug_reports import BugReport, generate_bug_report, render_bug_report
from .explain import DiagnosticExplanation, explain_diagnostic, render_explanation_json, render_explanation_text
from .loaders import ArtifactLoader, LoadedArtifact
from .first_party_plugins import create_first_party_plugin_registry
from .localization import (
    DiagnosticCatalogEntry,
    build_diagnostic_catalog,
    render_diagnostic_catalog_json,
    render_diagnostic_catalog_text,
)
from .maintainer import MaintainerRefresh, refresh_maintainer_artifacts
from .minimization import (
    FailurePredicate,
    MinimizationKind,
    MinimizationResult,
    minimize_repro,
    render_minimization_json,
    render_minimization_text,
)
from .mutation_fuzzing import (
    FuzzSurface,
    MutationFuzzReport,
    render_mutation_fuzz_json,
    render_mutation_fuzz_text,
    run_mutation_fuzzing,
)
from .plugins import PluginRegistry
from .policies import Suppression, VerificationPolicy, apply_policy_diagnostics, load_policy_file
from .reproducibility import (
    ReproducibilityInputs,
    ReproducibilityPackage,
    build_reproducibility_package,
    write_reproducibility_package,
)
from .render import SarifRenderOptions, render_github_annotations, render_html, render_json, render_sarif, render_text
from .session import CheckCallable, VerificationResult, VerificationSession


def create_session(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    checks: Mapping[str, CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> VerificationSession:
    """Create a verification session from a config object or JSON config path."""

    resolved_config = _resolve_config(config)
    if artifact_overrides:
        base_dir = Path(override_base_dir) if override_base_dir is not None else Path.cwd()
        resolved_config = resolved_config.with_artifact_overrides(dict(artifact_overrides), base_dir=base_dir)
    return VerificationSession(
        resolved_config,
        checks=checks,
        loader=loader,
        plugin_registry=plugin_registry or create_first_party_plugin_registry(),
    )


def load_artifacts(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    loader: ArtifactLoader | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> tuple[LoadedArtifact, ...]:
    """Load all artifacts for an embedding workflow using the default session semantics."""

    return create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        loader=loader,
        plugin_registry=plugin_registry,
    ).load_artifacts()


def collect_diagnostics(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    checks: Mapping[str, CheckCallable] | None = None,
    selected_checks: Sequence[str | CheckCallable] | None = None,
    loader: ArtifactLoader | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> tuple[Diagnostic, ...]:
    """Run verification and return diagnostics without constructing a result wrapper."""

    session = create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        checks=checks,
        loader=loader,
        plugin_registry=plugin_registry,
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
    plugin_registry: PluginRegistry | None = None,
) -> VerificationResult:
    """Run PromptABI verification from Python and return a typed result."""

    session = create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        checks=checks,
        loader=loader,
        plugin_registry=plugin_registry,
    )
    return session.run(checks=selected_checks)


def render_result(
    result: VerificationResult,
    *,
    output_format: str = "text",
    verbosity: int = 0,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
    sarif_options: SarifRenderOptions | None = None,
    github_checkout_uri_base: str | Path | None = None,
) -> str:
    """Render a typed verification result as text, HTML, JSON, SARIF, or GitHub annotations."""

    if output_format == "text":
        return render_text(
            result,
            verbosity=verbosity,
            config_path=Path(config_path) if config_path is not None else None,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
        )
    if output_format == "json":
        return render_json(result)
    if output_format == "html":
        return render_html(result)
    if output_format == "sarif":
        return render_sarif(result, options=sarif_options)
    if output_format == "github-annotations":
        checkout_base = Path(github_checkout_uri_base) if github_checkout_uri_base is not None else None
        return render_github_annotations(result, checkout_uri_base=checkout_base)
    if plugin_registry is not None and output_format in plugin_registry.renderers:
        return plugin_registry.render(output_format, result)
    raise ValueError("output_format must be one of: text, html, json, sarif, github-annotations")


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


def create_bug_report(
    result: VerificationResult,
    *,
    config_path: str | Path | None = None,
    fingerprint: str | None = None,
    rule_id: str | None = None,
    index: int | None = None,
    expected_behavior: str | None = None,
    actual_behavior: str | None = None,
    command: str | None = None,
    base_dir: str | Path | None = None,
) -> BugReport:
    """Create a sanitized upstream markdown issue report from a verification result."""

    return generate_bug_report(
        result,
        config_path=config_path,
        fingerprint=fingerprint,
        rule_id=rule_id,
        index=index,
        expected_behavior=expected_behavior,
        actual_behavior=actual_behavior,
        command=command,
        base_dir=base_dir,
    )


def minimize_failure_repro(
    value,
    predicate: FailurePredicate,
    *,
    kind: str | MinimizationKind,
    max_steps: int | None = None,
) -> MinimizationResult:
    """Shrink a failing PromptABI repro while the failure predicate still holds."""

    return minimize_repro(value, predicate, kind=kind, max_steps=max_steps)


def render_minimization(result: MinimizationResult, *, output_format: str = "text") -> str:
    """Render a minimization result as text or JSON."""

    if output_format == "text":
        return render_minimization_text(result)
    if output_format == "json":
        return render_minimization_json(result)
    raise ValueError("output_format must be one of: text, json")


def diagnostic_message_catalog(
    diagnostics: Sequence[Diagnostic],
    *,
    output_format: str | None = None,
) -> tuple[DiagnosticCatalogEntry, ...] | str:
    """Build or render a localization-ready catalog from diagnostic objects."""

    catalog = build_diagnostic_catalog(diagnostics)
    if output_format is None:
        return catalog
    if output_format == "json":
        return render_diagnostic_catalog_json(catalog)
    if output_format == "text":
        return render_diagnostic_catalog_text(catalog)
    raise ValueError("output_format must be one of: text, json")


def editor_diagnostics(
    *,
    config_path: str | Path | None = None,
    artifact_overrides: Mapping[str, str] | None = None,
    workspace_root: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
    output_format: str | None = None,
) -> EditorDiagnosticReport | str:
    """Run PromptABI and return or render LSP-style editor diagnostics."""

    report = build_editor_diagnostic_report(
        config_path=config_path,
        artifact_overrides=dict(artifact_overrides) if artifact_overrides is not None else None,
        workspace_root=workspace_root,
        plugin_registry=plugin_registry,
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_editor_diagnostic_json(report)
    if output_format == "text":
        return render_editor_diagnostic_text(report)
    raise ValueError("output_format must be one of: text, json")


def compatibility_matrix(
    *,
    plugin_registry: PluginRegistry | None = None,
    include_plugins: bool = True,
) -> CompatibilityMatrix:
    """Return the check compatibility matrix used by the CLI and docs."""

    return build_compatibility_matrix(plugin_registry=plugin_registry, include_plugins=include_plugins)


def render_compatibility_matrix(
    matrix: CompatibilityMatrix | None = None,
    *,
    output_format: str = "text",
    plugin_registry: PluginRegistry | None = None,
    include_plugins: bool = True,
) -> str:
    """Render a compatibility matrix as text or JSON."""

    resolved = matrix or build_compatibility_matrix(
        plugin_registry=plugin_registry,
        include_plugins=include_plugins,
    )
    if output_format == "text":
        return render_compatibility_matrix_text(resolved)
    if output_format == "json":
        return render_compatibility_matrix_json(resolved)
    raise ValueError("output_format must be one of: text, json")


def evaluate_corpus(
    path: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> EvaluationReport | str:
    """Run the labeled PromptABI evaluation corpus, optionally rendering it."""

    report = run_evaluation(path)
    if output_format is None:
        return report
    if output_format == "json":
        return render_evaluation_json(report)
    if output_format == "text":
        return render_evaluation_text(report)
    raise ValueError("output_format must be one of: text, json")


def verify_corpora(
    *,
    thresholds: CorpusVerificationThresholds | None = None,
    output_format: str | None = None,
) -> CorpusVerificationReport | str:
    """Run the maintainer release gate across all maintained corpora."""

    report = run_corpus_verification(thresholds=thresholds)
    if output_format is None:
        return report
    if output_format == "json":
        return render_corpus_verification_json(report)
    if output_format == "text":
        return render_corpus_verification_text(report)
    raise ValueError("output_format must be one of: text, json")


def create_reproducibility_package(
    *,
    inputs: ReproducibilityInputs | None = None,
    benchmark_iterations: int = 1,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> ReproducibilityPackage:
    """Build or write the paper reproducibility package with frozen fixtures and expected tables."""

    if output_dir is None:
        return build_reproducibility_package(inputs=inputs, benchmark_iterations=benchmark_iterations)
    return write_reproducibility_package(
        output_dir,
        inputs=inputs,
        benchmark_iterations=benchmark_iterations,
        force=force,
    )


def refresh_maintainer_tooling(
    output_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> MaintainerRefresh:
    """Regenerate maintainer manifests, expected diagnostics, diffs, and release notes."""

    return refresh_maintainer_artifacts(
        output_dir,
        baseline_dir=baseline_dir,
        repo_root=repo_root,
        force=force,
    )


def fuzz_mutations(
    surfaces: Sequence[str | FuzzSurface] = ("all",),
    *,
    output_format: str | None = None,
) -> MutationFuzzReport | str:
    """Run deterministic mutation fuzzing over PromptABI artifact contracts."""

    report = run_mutation_fuzzing(surfaces)
    if output_format is None:
        return report
    if output_format == "json":
        return render_mutation_fuzz_json(report)
    if output_format == "text":
        return render_mutation_fuzz_text(report)
    raise ValueError("output_format must be one of: text, json")


def _resolve_config(config: str | Path | VerificationConfig) -> VerificationConfig:
    if isinstance(config, VerificationConfig):
        return config
    return load_config(config)
