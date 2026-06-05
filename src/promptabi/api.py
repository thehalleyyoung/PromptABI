"""Ergonomic embedding API for PromptABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .adversarial_corpus import (
    AdversarialCorpusReport,
    generate_adversarial_corpus,
    render_adversarial_corpus_json,
    render_adversarial_corpus_text,
    replay_adversarial_corpus,
)
from .config import VerificationConfig, load_config
from .compatibility_matrix import (
    CompatibilityMatrix,
    build_compatibility_matrix,
    render_compatibility_matrix_json,
    render_compatibility_matrix_text,
)
from .compatibility_audit import (
    CompatibilityAuditReport,
    render_compatibility_audit_json,
    render_compatibility_audit_text,
    run_compatibility_audit,
)
from .contribution_workflows import (
    ContributionWorkflow,
    build_contribution_workflows,
    render_contribution_workflows_json,
    render_contribution_workflows_text,
)
from .contributor_validation import (
    ContributorValidationReport,
    render_contributor_validation_json,
    render_contributor_validation_text,
    validate_contributor_infrastructure,
)
from .governance import (
    GovernanceReport,
    render_governance_json,
    render_governance_text,
    validate_governance,
)
from .corpus_verification import (
    CorpusVerificationReport,
    CorpusVerificationThresholds,
    render_corpus_verification_json,
    render_corpus_verification_text,
    run_corpus_verification,
)
from .beta import (
    BetaProgramReport,
    render_beta_program_json,
    render_beta_program_text,
    run_beta_program,
)
from .benchmark_leaderboards import (
    BenchmarkLeaderboardReport,
    build_benchmark_leaderboard,
    render_benchmark_leaderboard_json,
    render_benchmark_leaderboard_text,
)
from .diagnostics import Diagnostic
from .diagnostic_clustering import (
    DiagnosticClusterReport,
    build_diagnostic_clusters,
    render_diagnostic_clusters_json,
    render_diagnostic_clusters_text,
)
from .dependency_graph import (
    DependencyGraphReport,
    build_dependency_graph,
    render_dependency_graph_json,
    render_dependency_graph_mermaid,
    render_dependency_graph_text,
)
from .deployment_gates import (
    DeploymentGateReport,
    build_deployment_gate_report,
    render_deployment_gate_json,
    render_deployment_gate_text,
    write_deployment_gate_examples,
)
from .runtime_attestation import (
    RuntimeAttestationReport,
    build_runtime_attestation_report,
    render_runtime_attestation_json,
    render_runtime_attestation_text,
)
from .runtime_alarms import (
    RuntimeAlarmReport,
    build_runtime_alarm_report,
    render_runtime_alarm_json,
    render_runtime_alarm_text,
)
from .roadmap import (
    RoadmapReport,
    build_annual_corpus_refresh_report,
    build_award_submission_report,
    build_historical_trend_report,
    build_research_agenda_report,
    build_teaching_materials_report,
    render_roadmap_json,
    render_roadmap_markdown,
    render_roadmap_text,
)
from .enterprise import (
    EnterpriseSettings,
    enterprise_readiness_diagnostics,
    render_enterprise_readiness_json,
    render_enterprise_readiness_text,
)
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
from .evaluation_reproducibility import (
    EvaluationReproducibilityReport,
    build_evaluation_reproducibility_report,
    render_evaluation_reproducibility_json,
    render_evaluation_reproducibility_text,
)
from .framework_truncation_conformance import (
    FrameworkTruncationConformanceReport,
    build_framework_truncation_conformance_report,
    render_framework_truncation_conformance_json,
    render_framework_truncation_conformance_text,
)
from .grammar_conformance import (
    GrammarConformanceReport,
    build_grammar_conformance_report,
    render_grammar_conformance_json,
    render_grammar_conformance_text,
)
from .provider_conformance import (
    ProviderConformanceReport,
    build_provider_conformance_report,
    render_provider_conformance_json,
    render_provider_conformance_text,
)
from .tokenizer_conformance import (
    TokenizerConformanceReport,
    build_tokenizer_conformance_report,
    render_tokenizer_conformance_json,
    render_tokenizer_conformance_text,
)
from .bug_reports import BugReport, generate_bug_report, render_bug_report
from .bundles import (
    VerificationBundle,
    VerificationBundleVerification,
    create_signed_verification_bundle,
    render_bundle_verification_text,
    verify_signed_verification_bundle,
    write_signed_verification_bundle,
)
from .explain import DiagnosticExplanation, explain_diagnostic, render_explanation_json, render_explanation_text
from .loaders import ArtifactLoader, LoadedArtifact
from .first_party_plugins import create_first_party_plugin_registry
from .api_stability import (
    PublicApiManifest,
    build_public_api_manifest,
    render_public_api_manifest_json,
    render_public_api_manifest_markdown,
)
from .artifact_bisection import (
    ArtifactBisectionReport,
    ArtifactRevision,
    bisect_artifact_drift,
    render_artifact_bisection_json,
    render_artifact_bisection_text,
)
from .autofix import (
    AutoFixReport,
    GuardedAutoFixPreviewReport,
    render_guarded_autofix_preview_json,
    render_guarded_autofix_preview_text,
    run_guarded_autofix_preview,
    render_autofix_json,
    render_autofix_text,
    run_low_risk_autofix,
)
from .localization import (
    DiagnosticCatalogEntry,
    build_diagnostic_catalog,
    render_diagnostic_catalog_json,
    render_diagnostic_catalog_text,
)
from .local_metrics import (
    LocalMetricsReport,
    build_local_metrics_report,
    render_local_metrics_json,
    render_local_metrics_text,
)
from .maintainer import (
    MaintainerHealthReport,
    MaintainerRefresh,
    refresh_maintainer_artifacts,
    render_maintainer_health_json,
    render_maintainer_health_text,
    validate_maintainer_health,
)
from .minimization import (
    FailurePredicate,
    MinimizationKind,
    MinimizationResult,
    minimize_repro,
    render_minimization_json,
    render_minimization_text,
)
from .notebook import (
    NotebookSection,
    NotebookVisualization,
    render_notebook_visualization_html,
    render_notebook_visualization_text,
    visualize_grammar_product,
    visualize_smt_constraints,
    visualize_stop_reachability,
    visualize_template_rendering,
    visualize_tokenization,
    visualize_truncation,
)
from .mutation_fuzzing import (
    FuzzSurface,
    MutationFuzzReport,
    render_mutation_fuzz_json,
    render_mutation_fuzz_text,
    run_mutation_fuzzing,
)
from .plugins import PluginRegistry
from .plugin_certification import (
    PluginCertificationReport,
    certify_plugin_registry,
    render_plugin_certification_json,
    render_plugin_certification_text,
)
from .plugin_marketplace import (
    PluginMarketplaceIndex,
    build_plugin_marketplace_index,
    render_plugin_marketplace_json,
    render_plugin_marketplace_text,
)
from .policies import (
    OrgPolicyPack,
    Suppression,
    VerificationPolicy,
    apply_org_policy_diagnostics,
    apply_policy_diagnostics,
    load_policy_file,
    policy_forbids_local_summary,
)
from .proof_sketches import (
    ProofSketchNotebookReport,
    ProofSketchReport,
    build_supported_proof_catalog,
    render_proof_sketch_report_json,
    render_proof_sketch_report_text,
    write_proof_sketch_notebooks,
)
from .mechanized_proofs import (
    MechanizedProofExperimentReport,
    render_mechanized_proof_experiments_json,
    render_mechanized_proof_experiments_text,
    run_mechanized_proof_experiments,
)
from .prompt_calculus import (
    MetatheoryReport,
    render_metatheory_json,
    render_metatheory_text,
    run_metatheory,
)
from .scaled_evaluation import (
    ScaledEvaluationReport,
    render_scaled_evaluation_json,
    render_scaled_evaluation_text,
    run_scaled_evaluation,
)
from .soundness_audits import (
    SoundnessAuditReport,
    build_soundness_audit_report,
    render_soundness_audit_json,
    render_soundness_audit_markdown,
    render_soundness_audit_text,
)
from .theorem_traceability import (
    TheoremTraceabilityReport,
    build_theorem_traceability_report,
    render_theorem_traceability_json,
    render_theorem_traceability_text,
)
from .release import (
    LTSMaintenanceItem,
    LTSReleasePlan,
    ReleaseReadinessReport,
    build_lts_release_plan,
    build_release_readiness_report,
    render_lts_release_plan_json,
    render_lts_release_plan_text,
    render_release_readiness_json,
    render_release_readiness_text,
)
from .team_dashboard import (
    DashboardSnapshot,
    TeamDashboardReport,
    build_team_dashboard,
    load_dashboard_history,
    render_team_dashboard_json,
    render_team_dashboard_text,
)
from .version_gates import (
    VersionGatePolicy,
    VersionGateReport,
    render_version_gate_json,
    render_version_gate_text,
    run_version_gate,
)
from .witness_privacy import WitnessPrivacyMode, apply_witness_privacy
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


def local_metrics(
    configs: str | Path | VerificationConfig | Sequence[str | Path | VerificationConfig],
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    output_format: str = "json",
    plugin_registry: PluginRegistry | None = None,
) -> str:
    """Render privacy-preserving local metrics for one or more verification configs."""

    config_items: Sequence[str | Path | VerificationConfig]
    if isinstance(configs, (str, Path, VerificationConfig)):
        config_items = (configs,)
    else:
        config_items = configs
    results = tuple(
        run_verification(
            config,
            artifact_overrides=artifact_overrides,
            override_base_dir=override_base_dir,
            plugin_registry=plugin_registry,
        )
        for config in config_items
    )
    report = build_local_metrics_report(results)
    if output_format == "json":
        return render_local_metrics_json(report)
    if output_format == "text":
        return render_local_metrics_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def runtime_alarms(
    attestation: str | Path | Mapping[str, object] | RuntimeAttestationReport,
    *,
    lockfile: str | Path | None = None,
    policy_pack: str | Path | Mapping[str, object] | None = None,
    corpus_baseline: str | Path | Mapping[str, object] | None = None,
    known_bad: str | Path | Mapping[str, object] | None = None,
    output_format: str | None = None,
) -> RuntimeAlarmReport | str:
    """Compare runtime attestation against latest deployment evidence and render alarms."""

    report = build_runtime_alarm_report(
        attestation,
        lockfile=lockfile,
        policy_pack=policy_pack,
        corpus_baseline=corpus_baseline,
        known_bad=known_bad,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_runtime_alarm_text(report)
    if output_format == "json":
        return render_runtime_alarm_json(report)
    raise ValueError("output_format must be 'json' or 'text'")


def historical_trends(
    history: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    output_format: str | None = None,
) -> RoadmapReport | str:
    """Render or return historical structural-risk trends."""

    report = build_historical_trend_report(history, repo_root=repo_root)
    return _render_roadmap_api(report, output_format)


def annual_corpus_refresh(
    *,
    repo_root: str | Path | None = None,
    output_format: str | None = None,
) -> RoadmapReport | str:
    """Render or return the annual corpus refresh procedure."""

    report = build_annual_corpus_refresh_report(repo_root=repo_root)
    return _render_roadmap_api(report, output_format)


def award_submission(
    *,
    repo_root: str | Path | None = None,
    output_format: str | None = None,
) -> RoadmapReport | str:
    """Render or return evidence-bound award submission material."""

    report = build_award_submission_report(repo_root=repo_root)
    return _render_roadmap_api(report, output_format)


def teaching_materials(
    *,
    repo_root: str | Path | None = None,
    output_format: str | None = None,
) -> RoadmapReport | str:
    """Render or return PromptABI teaching materials."""

    report = build_teaching_materials_report(repo_root=repo_root)
    return _render_roadmap_api(report, output_format)


def research_agenda(
    *,
    repo_root: str | Path | None = None,
    output_format: str | None = None,
) -> RoadmapReport | str:
    """Render or return the next 100 PromptABI research-agenda steps."""

    report = build_research_agenda_report(repo_root=repo_root)
    return _render_roadmap_api(report, output_format)


def _render_roadmap_api(report: RoadmapReport, output_format: str | None) -> RoadmapReport | str:
    if output_format is None:
        return report
    if output_format == "json":
        return render_roadmap_json(report)
    if output_format == "text":
        return render_roadmap_text(report)
    if output_format == "markdown":
        return render_roadmap_markdown(report)
    raise ValueError("output_format must be 'json', 'text', or 'markdown'")


def generate_adversarial_cases(*, output_format: str | None = None) -> AdversarialCorpusReport | str:
    """Generate and replay adversarial prompt-interface corpus cases."""

    cases = generate_adversarial_corpus()
    report = AdversarialCorpusReport(cases=cases, replays=replay_adversarial_corpus(cases))
    if output_format is None:
        return report
    if output_format == "json":
        return render_adversarial_corpus_json(report)
    if output_format == "text":
        return render_adversarial_corpus_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def proof_sketch_notebooks(
    output_dir: str | Path,
    *,
    force: bool = False,
) -> ProofSketchNotebookReport:
    """Write executable educational proof-sketch notebooks for core proof families."""

    return write_proof_sketch_notebooks(output_dir, force=force)


def theorem_traceability(*, output_format: str | None = None) -> TheoremTraceabilityReport | str:
    """Render or return theorem-to-test traceability for core proof claims."""

    report = build_theorem_traceability_report()
    if output_format is None:
        return report
    if output_format == "json":
        return render_theorem_traceability_json(report)
    if output_format == "text":
        return render_theorem_traceability_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def mechanized_proof_experiments(*, output_format: str | None = None) -> MechanizedProofExperimentReport | str:
    """Run mechanized proof experiments for small automata and finite-contract fragments."""

    report = run_mechanized_proof_experiments()
    if output_format is None:
        return report
    if output_format == "json":
        return render_mechanized_proof_experiments_json(report)
    if output_format == "text":
        return render_mechanized_proof_experiments_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def prompt_calculus_metatheory(*, output_format: str | None = None) -> MetatheoryReport | str:
    """Run the executable prompt-assembly metatheory (operational semantics, type
    soundness/non-forgeability, observational congruence, denotational stop
    policy, schema soundness/completeness, decidable grammar fragment,
    noninterference, capability monotonicity, session-typed tool calls, migration
    preservation, contract refinement, conformance compositionality, drift
    ultrametric)."""

    report = run_metatheory()
    if output_format is None:
        return report
    if output_format == "json":
        return render_metatheory_json(report)
    if output_format == "text":
        return render_metatheory_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def scaled_empirical_evaluation(
    *, output_format: str | None = None, corpus_limit: int | None = None
) -> ScaledEvaluationReport | str:
    """Run the scaled empirical evaluation (roadmap steps 316-330): a >=10k-case
    labeled prompt corpus replayed through the production analyzers with
    precision/recall/F1, ablation, longitudinal drift, inter-rater agreement,
    throughput, fuzzing, a conformance leaderboard, reproduced CVE vectors,
    false-positive cost, cross-tokenizer alignment, and training-data contracts."""

    report = run_scaled_evaluation(corpus_limit=corpus_limit)
    if output_format is None:
        return report
    if output_format == "json":
        return render_scaled_evaluation_json(report)
    if output_format == "text":
        return render_scaled_evaluation_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def grammar_conformance_suite(
    path: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> GrammarConformanceReport | str:
    """Replay the maintained grammar-backend conformance suite."""

    report = build_grammar_conformance_report(path)
    if output_format is None:
        return report
    if output_format == "json":
        return render_grammar_conformance_json(report)
    if output_format == "text":
        return render_grammar_conformance_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def tokenizer_conformance_suite(
    path: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> TokenizerConformanceReport | str:
    """Replay the maintained tokenizer-family conformance suite."""

    report = build_tokenizer_conformance_report(path)
    if output_format is None:
        return report
    if output_format == "json":
        return render_tokenizer_conformance_json(report)
    if output_format == "text":
        return render_tokenizer_conformance_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def provider_conformance_suite(
    root: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> ProviderConformanceReport | str:
    """Replay maintained provider-fixture conformance suites."""

    report = build_provider_conformance_report(root)
    if output_format is None:
        return report
    if output_format == "json":
        return render_provider_conformance_json(report)
    if output_format == "text":
        return render_provider_conformance_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def framework_truncation_conformance_suite(
    path: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> FrameworkTruncationConformanceReport | str:
    """Replay maintained framework-truncation conformance suites."""

    report = build_framework_truncation_conformance_report(path)
    if output_format is None:
        return report
    if output_format == "json":
        return render_framework_truncation_conformance_json(report)
    if output_format == "text":
        return render_framework_truncation_conformance_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def enterprise_readiness(config: str | Path | VerificationConfig) -> tuple[Diagnostic, ...]:
    """Run the declarative enterprise readiness check for a config."""

    resolved_config = _resolve_config(config)
    artifact_locations = tuple(
        location
        for artifact in resolved_config.artifact_bundle
        if (location := artifact.location.ref_path) is not None
    )
    return enterprise_readiness_diagnostics(
        resolved_config.enterprise,
        artifact_locations=tuple(sorted(set(artifact_locations))),
    )


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


def plugin_certification(
    plugin_registry: PluginRegistry | None = None,
    *,
    output_format: str | None = None,
) -> PluginCertificationReport | str:
    """Certify plugin extension points against diagnostic and privacy contracts."""

    report = certify_plugin_registry(plugin_registry or create_first_party_plugin_registry())
    if output_format is None:
        return report
    if output_format == "json":
        return render_plugin_certification_json(report)
    if output_format == "text":
        return render_plugin_certification_text(report)
    raise ValueError("output_format must be 'json' or 'text'")


def plugin_marketplace_index(
    plugin_registry: PluginRegistry | None = None,
    *,
    output_format: str | None = None,
) -> PluginMarketplaceIndex | str:
    """Build or render a marketplace-style index for registered PromptABI plugins."""

    registry = plugin_registry or create_first_party_plugin_registry()
    report = certify_plugin_registry(registry)
    index = build_plugin_marketplace_index(registry, certification_report=report)
    if output_format is None:
        return index
    if output_format == "json":
        return render_plugin_marketplace_json(index)
    if output_format == "text":
        return render_plugin_marketplace_text(index)
    raise ValueError("output_format must be 'json' or 'text'")


def low_risk_autofix(
    config_path: str | Path,
    *,
    kinds: Sequence[str] | None = None,
    write: bool = False,
    lockfile_path: str | Path | None = None,
    artifact_overrides: Mapping[str, str] | None = None,
    output_format: str | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> AutoFixReport | str:
    """Preview or apply low-risk fixes that do not alter prompt rendering behavior."""

    report = run_low_risk_autofix(
        config_path,
        kinds=kinds,
        write=write,
        lockfile_path=lockfile_path,
        artifact_overrides=artifact_overrides,
        plugin_registry=plugin_registry,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_autofix_text(report)
    if output_format == "json":
        return render_autofix_json(report)
    raise ValueError("output_format must be one of: text, json")


def guarded_autofix_preview(
    config_path: str | Path,
    *,
    risk: str = "high",
    artifact_overrides: Mapping[str, str] | None = None,
    output_format: str | None = None,
    plugin_registry: PluginRegistry | None = None,
) -> GuardedAutoFixPreviewReport | str:
    """Preview higher-risk prompt-interface fixes with before/after witness guardrails."""

    report = run_guarded_autofix_preview(
        config_path,
        risk=risk,
        artifact_overrides=artifact_overrides,
        plugin_registry=plugin_registry,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_guarded_autofix_preview_text(report)
    if output_format == "json":
        return render_guarded_autofix_preview_json(report)
    raise ValueError("output_format must be one of: text, json")


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
    witness_privacy: WitnessPrivacyMode | str = WitnessPrivacyMode.RAW,
) -> str:
    """Render a typed verification result as text, HTML, JSON, SARIF, or GitHub annotations."""

    result = apply_witness_privacy(result, witness_privacy)
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


def create_verification_bundle(
    config_path: str | Path,
    *,
    key: str | bytes | None = None,
    key_id: str = "local",
    artifact_overrides: Mapping[str, str] | None = None,
    output: str | Path | None = None,
    excerpt_bytes: int = 4096,
    force: bool = False,
) -> VerificationBundle:
    """Run verification and return or write a signed audit bundle."""

    bundle = create_signed_verification_bundle(
        config_path,
        key=key,
        key_id=key_id,
        artifact_overrides=dict(artifact_overrides) if artifact_overrides is not None else None,
        excerpt_bytes=excerpt_bytes,
    )
    if output is not None:
        write_signed_verification_bundle(output, bundle, force=force)
    return bundle


def verify_verification_bundle(
    bundle: VerificationBundle | dict[str, object] | str | Path,
    *,
    key: str | bytes | None = None,
    output_format: str | None = None,
) -> VerificationBundleVerification | str:
    """Verify a signed audit bundle and optionally render the result."""

    result = verify_signed_verification_bundle(bundle, key=key)
    if output_format is None:
        return result
    if output_format == "text":
        return render_bundle_verification_text(result)
    if output_format == "json":
        import json

        return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    raise ValueError("output_format must be one of: text, json")


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


def diagnostic_clusters(
    diagnostics_or_result: Sequence[Diagnostic] | VerificationResult,
    *,
    strategies: Sequence[str] | None = None,
    min_cluster_size: int = 2,
    output_format: str | None = None,
) -> DiagnosticClusterReport | str:
    """Group related findings by root cause, artifact edge, rule, provider behavior, or witness."""

    diagnostics = (
        diagnostics_or_result.diagnostics
        if isinstance(diagnostics_or_result, VerificationResult)
        else diagnostics_or_result
    )
    report = build_diagnostic_clusters(
        diagnostics,
        strategies=tuple(strategies) if strategies is not None else (),
        min_cluster_size=min_cluster_size,
    ) if strategies is not None else build_diagnostic_clusters(
        diagnostics,
        min_cluster_size=min_cluster_size,
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_diagnostic_clusters_json(report)
    if output_format == "text":
        return render_diagnostic_clusters_text(report)
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


def dependency_graph(
    config: str | Path | VerificationConfig,
    *,
    artifact_overrides: Mapping[str, str] | None = None,
    override_base_dir: str | Path | None = None,
    plugin_registry: PluginRegistry | None = None,
    include_all_checks: bool = False,
    output_format: str | None = None,
) -> DependencyGraphReport | str:
    """Build or render the artifact/check dependency graph for a config."""

    session = create_session(
        config,
        artifact_overrides=artifact_overrides,
        override_base_dir=override_base_dir,
        plugin_registry=plugin_registry,
    )
    report = build_dependency_graph(
        session.config,
        plugin_registry=session.plugin_registry,
        include_all_checks=include_all_checks,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_dependency_graph_text(report)
    if output_format == "json":
        return render_dependency_graph_json(report)
    if output_format == "mermaid":
        return render_dependency_graph_mermaid(report)
    raise ValueError("output_format must be one of: text, json, mermaid")


def deployment_gates(
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "deployment-gate",
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
    output_format: str | None = None,
) -> DeploymentGateReport | str:
    """Build or render deployment gates backed by current signed bundle evidence."""

    report = build_deployment_gate_report(
        config,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        fail_on=fail_on,
        workspace_root=workspace_root,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_deployment_gate_text(report)
    if output_format == "json":
        return render_deployment_gate_json(report)
    raise ValueError("output_format must be one of: text, json")


def runtime_attestation(
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "runtime-attestation",
    service: str = "promptabi-service",
    environment: str = "production",
    revision: str | None = None,
    instance_id: str | None = None,
    runtime_contract_refs: Mapping[str, str] | None = None,
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
    output_format: str | None = None,
) -> RuntimeAttestationReport | str:
    """Build or render runtime hooks that report verified PromptABI contracts."""

    report = build_runtime_attestation_report(
        config,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        service=service,
        environment=environment,
        revision=revision,
        instance_id=instance_id,
        runtime_contract_refs=runtime_contract_refs,
        fail_on=fail_on,
        workspace_root=workspace_root,
    )
    if output_format is None:
        return report
    if output_format == "text":
        return render_runtime_attestation_text(report)
    if output_format == "json":
        return render_runtime_attestation_json(report)
    raise ValueError("output_format must be one of: text, json")


def compatibility_audit(
    candidate_versions: Mapping[str, str],
    *,
    output_format: str | None = None,
) -> CompatibilityAuditReport | str:
    """Run or render the post-1.0 fixture-backed compatibility audit."""

    report = run_compatibility_audit(candidate_versions)
    if output_format is None:
        return report
    if output_format == "json":
        return render_compatibility_audit_json(report)
    if output_format == "text":
        return render_compatibility_audit_text(report)
    raise ValueError("output_format must be one of: text, json")


def artifact_drift_bisection(
    surface: str,
    baseline_path: str | Path,
    revisions: Sequence[ArtifactRevision],
    *,
    baseline_label: str = "baseline",
    bad_fields: Sequence[str] = (),
    output_format: str | None = None,
) -> ArtifactBisectionReport | str:
    """Run or render a local artifact-drift regression bisection."""

    report = bisect_artifact_drift(
        surface,
        baseline_path,
        tuple(revisions),
        baseline_label=baseline_label,
        bad_fields=tuple(bad_fields),
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_artifact_bisection_json(report)
    if output_format == "text":
        return render_artifact_bisection_text(report)
    raise ValueError("output_format must be one of: text, json")


def semantic_version_gate(
    baseline_path: str | Path,
    current_path: str | Path,
    *,
    allowed_impact: str = "patch-safe",
    policy: VersionGatePolicy | None = None,
    policy_path: str | Path | None = None,
    output_format: str | None = None,
) -> VersionGateReport | str:
    """Run or render a semantic-version gate over two verified contract configs."""

    report = run_version_gate(
        baseline_path,
        current_path,
        allowed_impact=allowed_impact,
        policy=policy,
        policy_path=policy_path,
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_version_gate_json(report)
    if output_format == "text":
        return render_version_gate_text(report)
    raise ValueError("output_format must be one of: text, json")


def proof_sketches(*, output_format: str | None = None) -> ProofSketchReport | str:
    """Return or render theorem sketches for supported PromptABI proof families."""

    report = build_supported_proof_catalog()
    if output_format is None:
        return report
    if output_format == "json":
        return render_proof_sketch_report_json(report)
    if output_format == "text":
        return render_proof_sketch_report_text(report)
    raise ValueError("output_format must be one of: text, json")


def soundness_audits(
    *,
    rule: str | None = None,
    output_format: str | None = None,
) -> SoundnessAuditReport | str:
    """Return or render check soundness audits for built-in PromptABI rule families."""

    report = build_soundness_audit_report(rule=rule)
    if output_format is None:
        return report
    if output_format == "json":
        return render_soundness_audit_json(report)
    if output_format == "markdown":
        return render_soundness_audit_markdown(report)
    if output_format == "text":
        return render_soundness_audit_text(report)
    raise ValueError("output_format must be one of: text, json, markdown")


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


def benchmark_leaderboard(
    *,
    release: str | None = None,
    benchmark_iterations: int = 1,
    output_format: str | None = None,
) -> BenchmarkLeaderboardReport | str:
    """Run or render the release benchmark leaderboard against local fixtures."""

    kwargs: dict[str, object] = {"benchmark_iterations": benchmark_iterations}
    if release is not None:
        kwargs["release"] = release
    report = build_benchmark_leaderboard(**kwargs)
    if output_format is None:
        return report
    if output_format == "json":
        return render_benchmark_leaderboard_json(report)
    if output_format == "text":
        return render_benchmark_leaderboard_text(report)
    raise ValueError("output_format must be one of: text, json")


def evaluation_reproducibility(
    configs: Sequence[str | Path] | None = None,
    *,
    output_format: str | None = None,
) -> EvaluationReproducibilityReport | str:
    """Pin benchmark-interface surfaces for evaluation harness reproducibility."""

    report = build_evaluation_reproducibility_report(configs)
    if output_format is None:
        return report
    if output_format == "json":
        return render_evaluation_reproducibility_json(report)
    if output_format == "text":
        return render_evaluation_reproducibility_text(report)
    raise ValueError("output_format must be one of: text, json")


def verify_corpora(
    *,
    thresholds: CorpusVerificationThresholds | None = None,
    seed_root: str | Path | None = None,
    structured_schema_root: str | Path | None = None,
    provider_fixture_root: str | Path | None = None,
    framework_truncation_conformance_suite_path: str | Path | None = None,
    grammar_conformance_suite_path: str | Path | None = None,
    tokenizer_conformance_suite_path: str | Path | None = None,
    real_bug_benchmark_path: str | Path | None = None,
    evaluation_corpus_path: str | Path | None = None,
    evaluation_fixture_pack_path: str | Path | None = None,
    smt_benchmark_path: str | Path | None = None,
    output_format: str | None = None,
) -> CorpusVerificationReport | str:
    """Run the maintainer release gate across all maintained corpora."""

    report = run_corpus_verification(
        seed_root=seed_root,
        structured_schema_root=structured_schema_root,
        provider_fixture_root=provider_fixture_root,
        framework_truncation_conformance_suite_path=framework_truncation_conformance_suite_path,
        grammar_conformance_suite_path=grammar_conformance_suite_path,
        tokenizer_conformance_suite_path=tokenizer_conformance_suite_path,
        real_bug_benchmark_path=real_bug_benchmark_path,
        evaluation_corpus_path=evaluation_corpus_path,
        evaluation_fixture_pack_path=evaluation_fixture_pack_path,
        smt_benchmark_path=smt_benchmark_path,
        thresholds=thresholds,
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_corpus_verification_json(report)
    if output_format == "text":
        return render_corpus_verification_text(report)
    raise ValueError("output_format must be one of: text, json")


def beta_program(
    path: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> BetaProgramReport | str:
    """Run the offline beta-program case-study replay, optionally rendering it."""

    report = run_beta_program(path)
    if output_format is None:
        return report
    if output_format == "json":
        return render_beta_program_json(report)
    if output_format == "text":
        return render_beta_program_text(report)
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


def release_readiness(
    repo_root: str | Path | None = None,
    *,
    expected_version: str = "1.0.0",
    output_format: str | None = None,
) -> ReleaseReadinessReport | str:
    """Run or render the 1.0 release-readiness gate."""

    report = build_release_readiness_report(repo_root, expected_version=expected_version)
    if output_format is None:
        return report
    if output_format == "json":
        return render_release_readiness_json(report)
    if output_format == "text":
        return render_release_readiness_text(report)
    raise ValueError("output_format must be one of: text, json")


def lts_release_plan(
    items: Sequence[LTSMaintenanceItem],
    *,
    series: str,
    base_version: str,
    target_version: str,
    repo_root: str | Path | None = None,
    candidate_versions: Mapping[str, str] | None = None,
    output_format: str | None = None,
) -> LTSReleasePlan | str:
    """Build or render a fixture-backed long-term support release plan."""

    report = build_lts_release_plan(
        tuple(items),
        series=series,
        base_version=base_version,
        target_version=target_version,
        repo_root=repo_root,
        candidate_versions=dict(candidate_versions) if candidate_versions is not None else None,
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_lts_release_plan_json(report)
    if output_format == "text":
        return render_lts_release_plan_text(report)
    raise ValueError("output_format must be one of: text, json")


def team_dashboard(
    configs: Sequence[str | Path | VerificationConfig] | str | Path | VerificationConfig,
    *,
    history_path: str | Path | None = None,
    output_format: str | None = None,
) -> TeamDashboardReport | str:
    """Build or render a team risk dashboard from one or more real configs."""

    if isinstance(configs, (str, Path, VerificationConfig)):
        config_sequence: Sequence[str | Path | VerificationConfig] = (configs,)
    else:
        config_sequence = configs
    report = build_team_dashboard(
        tuple(run_verification(config) for config in config_sequence),
        history=load_dashboard_history(history_path),
    )
    if output_format is None:
        return report
    if output_format == "json":
        return render_team_dashboard_json(report)
    if output_format == "text":
        return render_team_dashboard_text(report)
    raise ValueError("output_format must be one of: text, json")


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


def maintainer_health(
    repo_root: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> MaintainerHealthReport | str:
    """Validate and render PromptABI maintainer rotation, triage, release, and corpus-review health."""

    report = validate_maintainer_health(repo_root)
    if output_format is None:
        return report
    if output_format == "json":
        return render_maintainer_health_json(report)
    if output_format == "text":
        return render_maintainer_health_text(report)
    raise ValueError("output_format must be one of: text, json")


def contributor_infrastructure(
    repo_root: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> ContributorValidationReport | str:
    """Validate contributor templates, labels, docs, and CI gates."""

    report = validate_contributor_infrastructure(repo_root)
    if output_format is None:
        return report
    if output_format == "json":
        return render_contributor_validation_json(report)
    if output_format == "text":
        return render_contributor_validation_text(report)
    raise ValueError("output_format must be one of: text, json")


def contribution_workflows(*, output_format: str | None = None) -> tuple[ContributionWorkflow, ...] | str:
    """Return community workflows for fixtures, witnesses, prompt packs, providers, and training adapters."""

    workflows = build_contribution_workflows()
    if output_format is None:
        return workflows
    if output_format == "json":
        return render_contribution_workflows_json(workflows)
    if output_format == "text":
        return render_contribution_workflows_text(workflows)
    raise ValueError("output_format must be one of: text, json")


def governance_policy(
    repo_root: str | Path | None = None,
    *,
    output_format: str | None = None,
) -> GovernanceReport | str:
    """Validate and render PromptABI checker, corpus, disclosure, and release governance."""

    report = validate_governance(repo_root)
    if output_format is None:
        return report
    if output_format == "json":
        return render_governance_json(report)
    if output_format == "text":
        return render_governance_text(report)
    raise ValueError("output_format must be one of: text, json")


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


def public_api_reference(*, output_format: str | None = None) -> PublicApiManifest | str:
    """Return or render the generated public API stability manifest."""

    manifest = build_public_api_manifest()
    if output_format is None:
        return manifest
    if output_format == "json":
        return render_public_api_manifest_json(manifest)
    if output_format in {"markdown", "text"}:
        return render_public_api_manifest_markdown(manifest)
    raise ValueError("output_format must be one of: json, markdown")


def _resolve_config(config: str | Path | VerificationConfig) -> VerificationConfig:
    if isinstance(config, VerificationConfig):
        return config
    return load_config(config)
