"""Command line entrypoint for PromptABI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .api_stability import (
    build_public_api_manifest,
    render_public_api_manifest_json,
    render_public_api_manifest_markdown,
)
from .bug_reports import BugReportError, generate_bug_report, render_bug_report
from .bundles import (
    VerificationBundleError,
    create_signed_verification_bundle,
    load_signed_verification_bundle,
    render_bundle_verification_text,
    verify_signed_verification_bundle,
    write_signed_verification_bundle,
)
from .beta import (
    BetaProgramError,
    render_beta_program_json,
    render_beta_program_text,
    run_beta_program,
)
from .config import ConfigError, discover_config, load_config
from .compatibility_matrix import (
    build_compatibility_matrix,
    render_compatibility_matrix_json,
    render_compatibility_matrix_text,
)
from .compatibility_audit import (
    CompatibilityAuditError,
    render_compatibility_audit_json,
    render_compatibility_audit_text,
    run_compatibility_audit,
)
from .contributor_validation import (
    ContributorValidationError,
    render_contributor_validation_json,
    render_contributor_validation_text,
    validate_contributor_infrastructure,
)
from .corpus_verification import (
    CorpusVerificationError,
    CorpusVerificationThresholds,
    render_corpus_verification_json,
    render_corpus_verification_text,
    run_corpus_verification,
)
from .diff import diff_config_files
from .doctor import render_doctor_json, render_doctor_text, run_doctor
from .dependency_graph import (
    build_dependency_graph,
    render_dependency_graph_json,
    render_dependency_graph_mermaid,
    render_dependency_graph_text,
)
from .editor_protocol import (
    EditorProtocolError,
    build_editor_diagnostic_report,
    render_editor_diagnostic_json,
    render_editor_diagnostic_text,
)
from .explain import ExplainError, explain_diagnostic, render_explanation_json, render_explanation_text
from .evaluation import EvaluationError, render_evaluation_json, render_evaluation_text, run_evaluation
from .evaluation_reproducibility import (
    EvaluationReproducibilityError,
    build_evaluation_reproducibility_report,
    render_evaluation_reproducibility_json,
    render_evaluation_reproducibility_text,
)
from .evaluation_fixture_packs import (
    EvaluationFixturePackError,
    build_evaluation_fixture_pack_manifest,
    write_evaluation_fixture_pack_manifest,
)
from .first_party_plugins import create_first_party_plugin_registry, render_plugin_capabilities
from .formal import SolverReplayFile, render_solver_replay_json, render_solver_replay_text
from .github_action import GitHubActionError, run_github_action
from .gallery import GalleryError, build_gallery, render_gallery_json, render_gallery_text
from .incremental import (
    IncrementalVerificationError,
    explicit_changed_paths,
    git_changed_paths,
    run_incremental_verification,
)
from .init import InitError, available_stacks, scaffold_promptabi_project
from .launch_assets import (
    LaunchAssetError,
    render_launch_asset_summary,
    write_launch_assets,
)
from .local_workflows import (
    LocalWorkflowError,
    install_pre_commit_hook,
    render_local_workflow_text,
    run_local_workflow,
)
from .localization import (
    LocalizationError,
    build_diagnostic_catalog,
    render_diagnostic_catalog_json,
    render_diagnostic_catalog_text,
)
from .lockfiles import (
    LockfileError,
    build_lockfile,
    compare_lockfile,
    load_lockfile,
    lockfile_error_diagnostic,
    write_lockfile,
)
from .maintainer import MaintainerToolingError, refresh_maintainer_artifacts
from .minimization import (
    MinimizationError,
    MinimizationOracle,
    contains_oracle,
    diagnostic_oracle,
    load_minimization_case,
    minimize_repro,
    render_minimization_json,
    render_minimization_text,
)
from .mutation_fuzzing import (
    ALL_FUZZ_SURFACES,
    MutationFuzzingError,
    render_mutation_fuzz_json,
    render_mutation_fuzz_text,
    run_mutation_fuzzing,
)
from .plugins import PluginError, PluginRegistry, load_plugin_modules
from .policies import policy_forbids_local_summary
from .prompt_packs import (
    PromptPackLockError,
    build_prompt_pack_registry,
    build_prompt_pack_lockfile,
    compare_prompt_pack_lockfile,
    compare_prompt_pack_upgrade,
    load_prompt_pack_lockfile,
    prompt_pack_lock_error_diagnostic,
    prompt_pack_registry_to_json,
    render_prompt_pack_registry_text,
    write_prompt_pack_registry,
    write_prompt_pack_lockfile,
)
from .proof_sketches import (
    build_supported_proof_catalog,
    render_proof_sketch_report_json,
    render_proof_sketch_report_text,
)
from .render import SarifRenderOptions, render_github_annotations, render_html, render_json, render_sarif, render_text
from .seed_corpus import SeedCorpusError, build_seed_corpus_manifest, write_seed_corpus_manifest
from .session import VerificationResult, VerificationSession
from .smt_benchmarks import (
    SmtBenchmarkError,
    build_smt_benchmark_manifest,
    render_smt_benchmark_text,
    write_smt_benchmark_manifest,
)
from .structured_schema_corpus import (
    StructuredSchemaCorpusError,
    build_structured_schema_corpus_manifest,
    write_structured_schema_corpus_manifest,
)
from .training_workflow import (
    TRAINING_WORKFLOW_CHECKS,
    build_training_config,
    run_training_verification,
)
from .usage_analytics import (
    UsageAnalyticsError,
    append_local_command_summary,
    render_usage_privacy_text,
    render_usage_summary_json,
    render_usage_summary_text,
    summarize_local_command_usage,
)
from .version_gates import (
    SemverImpact,
    render_version_gate_json,
    render_version_gate_text,
    run_version_gate,
)
from .provider_fixture_packs import (
    ProviderFixturePackError,
    build_provider_fixture_pack_manifest,
    write_provider_fixture_pack_manifest,
)
from .real_bug_benchmarks import (
    RealBugBenchmarkError,
    build_real_bug_benchmark_manifest,
    write_real_bug_benchmark_manifest,
)
from .release import (
    ReleaseReadinessError,
    build_release_readiness_report,
    render_release_readiness_json,
    render_release_readiness_text,
)
from .reproducibility import (
    ReproducibilityPackageError,
    write_reproducibility_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptabi",
        description="Verify tokenizer/template/tool-calling interface contracts for LLM apps.",
    )
    parser.add_argument("--version", action="version", version=f"promptabi {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="run PromptABI checks for a config")
    verify.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    verify.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    verify.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    verify.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    verify.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    verify.add_argument("-v", "--verbose", action="count", default=0, help="include additional workflow metadata")
    verify.add_argument(
        "--format",
        default="text",
        help="output format: text, html, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    _add_github_output_arguments(verify)
    verify.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for this run; may be repeated",
    )
    verify.add_argument(
        "--lockfile",
        help="path to a PromptABI lockfile (default: promptabi.lock.json beside the config)",
    )
    verify.add_argument(
        "--write-lockfile",
        action="store_true",
        help="write a lockfile pinning artifact hashes, revisions, tool versions, and diagnostic baselines",
    )
    verify.add_argument(
        "--require-lockfile",
        action="store_true",
        help="fail verification if the current artifacts or diagnostic baseline drift from the lockfile",
    )
    verify.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="config-relative or absolute changed path for incremental monorepo verification; may be repeated",
    )
    verify.add_argument(
        "--changed-from-git",
        metavar="REF",
        help="run incrementally from paths changed since this git ref, including untracked files",
    )
    _add_local_summary_argument(verify)

    verify_training = subparsers.add_parser(
        "verify-training",
        help="verify training manifests, dataset contracts, tokenizers, templates, packing, and loss masks",
    )
    verify_training.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json when --manifest is omitted",
    )
    verify_training.add_argument(
        "--manifest",
        help="training manifest JSON path; builds a focused temporary PromptABI config",
    )
    verify_training.add_argument(
        "--name",
        help="run name used with --manifest (default: training-<manifest-stem>)",
    )
    verify_training.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="tokenizer artifact to align with manifest tokenizer pins; may be repeated",
    )
    verify_training.add_argument(
        "--chat-template",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="chat-template artifact to align with manifest template pins; may be repeated",
    )
    verify_training.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact when using --config; may be repeated",
    )
    verify_training.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    verify_training.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    verify_training.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    verify_training.add_argument("-v", "--verbose", action="count", default=0, help="include additional workflow metadata")
    verify_training.add_argument(
        "--format",
        default="text",
        help="output format: text, html, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    _add_github_output_arguments(verify_training)
    verify_training.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for this run; may be repeated",
    )
    _add_local_summary_argument(verify_training)

    explain = subparsers.add_parser("explain", help="expand one diagnostic into a tutorial-style explanation")
    explain.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    explain.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    explain.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    explain_selector = explain.add_mutually_exclusive_group()
    explain_selector.add_argument("--fingerprint", help="explain the diagnostic with this stable fingerprint")
    explain_selector.add_argument("--rule-id", help="explain the only diagnostic with this rule id")
    explain_selector.add_argument("--index", type=int, help="explain the one-based diagnostic index after de-duplication")
    explain.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    bug_report = subparsers.add_parser(
        "bug-report",
        help="generate a sanitized upstream markdown issue from one diagnostic",
    )
    bug_report.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    bug_report.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    bug_selector = bug_report.add_mutually_exclusive_group()
    bug_selector.add_argument("--fingerprint", help="report the diagnostic with this stable fingerprint")
    bug_selector.add_argument("--rule-id", help="report the only diagnostic with this rule id")
    bug_selector.add_argument("--index", type=int, help="report the one-based diagnostic index after de-duplication")
    bug_report.add_argument(
        "--expected",
        help="expected behavior to include in the issue (default: structural contract should be impossible)",
    )
    bug_report.add_argument(
        "--actual",
        help="actual behavior to include in the issue (default: PromptABI likely symptom)",
    )
    bug_report.add_argument(
        "--output",
        help="write markdown to this path instead of stdout",
    )

    bundle = subparsers.add_parser("bundle", help="create and verify signed PromptABI audit bundles")
    bundle_subparsers = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_create = bundle_subparsers.add_parser(
        "create",
        help="run verification and write a signed bundle with diagnostics, witnesses, lockfile state, and hashes",
    )
    bundle_create.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    bundle_create.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    bundle_create.add_argument("--output", required=True, help="write signed bundle JSON to this path")
    bundle_create.add_argument("--key", help="bundle signing key (default: PROMPTABI_BUNDLE_KEY)")
    bundle_create.add_argument("--key-id", default="local", help="identifier recorded with the signature")
    bundle_create.add_argument(
        "--excerpt-bytes",
        type=int,
        default=4096,
        help="maximum artifact bytes to embed as base64 excerpts for audit replay (default: 4096)",
    )
    bundle_create.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    bundle_verify = bundle_subparsers.add_parser("verify", help="verify a signed bundle without rerunning checks")
    bundle_verify.add_argument("bundle", help="signed bundle JSON path")
    bundle_verify.add_argument("--key", help="bundle signing key (default: PROMPTABI_BUNDLE_KEY)")
    bundle_verify.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    diff = subparsers.add_parser("diff", help="compare two PromptABI configs for contract-breaking changes")
    diff.add_argument("baseline", help="baseline PromptABI JSON config")
    diff.add_argument("current", help="current PromptABI JSON config")
    diff.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    diff.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    diff.add_argument("-v", "--verbose", action="count", default=0, help="include compared config paths")
    diff.add_argument(
        "--format",
        default="text",
        help="output format: text, html, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    _add_github_output_arguments(diff)
    diff.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for renderer extensions; may be repeated",
    )
    _add_local_summary_argument(diff)

    version_gate = subparsers.add_parser(
        "version-gate",
        help="gate a config diff against declared patch/minor/major contract impact",
    )
    version_gate.add_argument("baseline", help="baseline PromptABI JSON config")
    version_gate.add_argument("current", help="current PromptABI JSON config")
    version_gate.add_argument(
        "--allowed-impact",
        choices=tuple(item.value for item in SemverImpact),
        default=SemverImpact.PATCH_SAFE.value,
        help="maximum semantic-version impact allowed by this deployment gate (default: patch-safe)",
    )
    version_gate.add_argument("--policy", help="JSON policy file with semver impact overrides")
    version_gate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    version_gate.add_argument("--output", help="write version-gate report to this path instead of stdout")

    init = subparsers.add_parser("init", help="scaffold a PromptABI config for a common LLM stack")
    init.add_argument(
        "--stack",
        choices=available_stacks(),
        default="openai-tools",
        help="application stack to scaffold (default: openai-tools)",
    )
    init.add_argument(
        "--output-dir",
        default=".",
        help="directory to write promptabi.json and local fixture stubs into (default: cwd)",
    )
    init.add_argument(
        "--name",
        help="verification config name (default: <stack>-promptabi)",
    )
    init.add_argument(
        "--config",
        default="promptabi.json",
        help="config file name to write inside output-dir (default: promptabi.json)",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing scaffold files",
    )

    prompt_pack = subparsers.add_parser("prompt-pack", help="manage reusable prompt-pack contracts")
    prompt_pack_subparsers = prompt_pack.add_subparsers(dest="prompt_pack_command", required=True)
    prompt_pack_lock = prompt_pack_subparsers.add_parser(
        "lock",
        help="write or enforce a package-manager-style lockfile for prompt-pack contracts",
    )
    prompt_pack_lock.add_argument(
        "--config",
        help="path to a PromptABI JSON config containing prompt-pack artifacts; defaults to discovering promptabi.json",
    )
    prompt_pack_lock.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    prompt_pack_lock.add_argument(
        "--lockfile",
        help="path to the prompt-pack lockfile (default: prompt-pack.lock.json beside the config)",
    )
    prompt_pack_lock.add_argument(
        "--write",
        action="store_true",
        help="write the prompt-pack lockfile after verifying the configured prompt packs",
    )
    prompt_pack_lock.add_argument(
        "--check",
        action="store_true",
        help="compare current prompt-pack packages and diagnostics against the lockfile",
    )
    prompt_pack_lock.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    prompt_pack_lock.add_argument(
        "--format",
        default="text",
        help="output format: text, html, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    prompt_pack_lock.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    prompt_pack_lock.add_argument("-v", "--verbose", action="count", default=0, help="include config and lockfile paths")
    prompt_pack_lock.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for renderer extensions; may be repeated",
    )
    _add_github_output_arguments(prompt_pack_lock)
    _add_local_summary_argument(prompt_pack_lock)
    prompt_pack_upgrade = prompt_pack_subparsers.add_parser(
        "upgrade",
        help="prove a prompt-pack upgrade preserves baseline package guarantees",
    )
    prompt_pack_upgrade.add_argument(
        "--config",
        help="path to the upgrade candidate PromptABI config; defaults to discovering promptabi.json",
    )
    prompt_pack_upgrade.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for the upgrade candidate; may be repeated",
    )
    prompt_pack_upgrade.add_argument(
        "--baseline-lockfile",
        required=True,
        help="prompt-pack lockfile from the baseline version to preserve",
    )
    prompt_pack_upgrade.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    prompt_pack_upgrade.add_argument(
        "--format",
        default="text",
        help="output format: text, html, json, sarif, github-annotations, or a plugin renderer (default: text)",
    )
    prompt_pack_upgrade.add_argument("-q", "--quiet", action="count", default=0, help="suppress informational text output")
    prompt_pack_upgrade.add_argument("-v", "--verbose", action="count", default=0, help="include config and baseline paths")
    prompt_pack_upgrade.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for renderer extensions; may be repeated",
    )
    _add_github_output_arguments(prompt_pack_upgrade)
    _add_local_summary_argument(prompt_pack_upgrade)
    prompt_pack_registry = prompt_pack_subparsers.add_parser(
        "registry",
        help="emit a public verified prompt-pack registry manifest without prompt contents",
    )
    prompt_pack_registry.add_argument(
        "--config",
        help="path to a PromptABI JSON config containing prompt-pack artifacts; defaults to discovering promptabi.json",
    )
    prompt_pack_registry.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    prompt_pack_registry.add_argument(
        "--output",
        help="write the registry manifest JSON to this path instead of stdout",
    )
    prompt_pack_registry.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="output format for stdout (default: json); --output always writes JSON",
    )
    prompt_pack_registry.add_argument("-q", "--quiet", action="count", default=0, help="suppress stdout when --output is set")
    prompt_pack_registry.add_argument("-v", "--verbose", action="count", default=0, help="include config and output paths in text output")
    prompt_pack_registry.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import a PromptABI plugin module for artifact loading compatibility; renderers are not used",
    )
    _add_local_summary_argument(prompt_pack_registry)

    corpus = subparsers.add_parser("corpus", help="seed corpus maintenance commands")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)
    manifest = corpus_subparsers.add_parser("manifest", help="validate corpus and emit its manifest")
    manifest.add_argument(
        "--root",
        help="seed corpus root (default: repository fixtures/seed_corpus)",
    )
    manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    schema_manifest = corpus_subparsers.add_parser(
        "structured-schema-manifest",
        help="validate structured-output/tool schema corpus and emit its manifest",
    )
    schema_manifest.add_argument(
        "--root",
        help="structured schema corpus root (default: repository fixtures/structured_schemas)",
    )
    schema_manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    provider_fixture_manifest = corpus_subparsers.add_parser(
        "provider-fixture-manifest",
        help="validate recorded provider fixture packs and emit their manifest",
    )
    provider_fixture_manifest.add_argument(
        "--root",
        help="provider fixture pack root (default: repository fixtures/provider_fixture_packs)",
    )
    provider_fixture_manifest.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    real_bug_benchmark = corpus_subparsers.add_parser(
        "real-bug-benchmark",
        help="validate and replay the real-bug benchmark suite, then emit its manifest",
    )
    real_bug_benchmark.add_argument(
        "--path",
        help="real-bug benchmark JSON path (default: repository fixtures/real_bug_benchmarks/benchmark.json)",
    )
    real_bug_benchmark.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    evaluation_fixture_pack = corpus_subparsers.add_parser(
        "evaluation-fixture-pack",
        help="validate and replay eval bug fixture packs, then emit their manifest",
    )
    evaluation_fixture_pack.add_argument(
        "--path",
        help="evaluation fixture pack JSON path (default: repository fixtures/evaluation_fixture_packs/pack.json)",
    )
    evaluation_fixture_pack.add_argument(
        "--output",
        help="write manifest JSON to this path instead of stdout",
    )
    smt_benchmark = corpus_subparsers.add_parser(
        "smt-benchmark",
        help="validate and replay minimized SMT benchmark obligations, then emit their manifest",
    )
    smt_benchmark.add_argument(
        "--path",
        help="SMT benchmark JSON path (default: repository fixtures/smt_benchmarks/benchmark.json)",
    )
    smt_benchmark.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    smt_benchmark.add_argument(
        "--output",
        help="write manifest to this path instead of stdout",
    )
    evaluation = corpus_subparsers.add_parser(
        "evaluation",
        help="run labeled-corpus evaluation metrics over real PromptABI analyzers",
    )
    evaluation.add_argument(
        "--corpus",
        help="evaluation corpus JSON path (default: repository fixtures/evaluation/labeled_corpus.json)",
    )
    evaluation.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    evaluation.add_argument(
        "--output",
        help="write evaluation report to this path instead of stdout",
    )
    evaluation_reproducibility = corpus_subparsers.add_parser(
        "evaluation-reproducibility",
        help="pin evaluation-harness prompt, tokenizer, provider, stop, and parser contracts",
    )
    evaluation_reproducibility.add_argument(
        "--config",
        action="append",
        default=[],
        help="evaluation-harness PromptABI config to pin; may be repeated (default: examples/evaluation-harness/safe.promptabi.json)",
    )
    evaluation_reproducibility.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    evaluation_reproducibility.add_argument(
        "--output",
        help="write evaluation reproducibility report to this path instead of stdout",
    )
    corpus_verify = corpus_subparsers.add_parser(
        "verify",
        help="run release-blocking verification across all maintained corpora",
    )
    corpus_verify.add_argument("--seed-root", help="seed corpus root (default: repository fixtures/seed_corpus)")
    corpus_verify.add_argument(
        "--structured-schema-root",
        help="structured schema corpus root (default: repository fixtures/structured_schemas)",
    )
    corpus_verify.add_argument(
        "--provider-fixture-root",
        help="provider fixture pack root (default: repository fixtures/provider_fixture_packs)",
    )
    corpus_verify.add_argument(
        "--real-bug-benchmark",
        help="real-bug benchmark JSON path (default: repository fixtures/real_bug_benchmarks/benchmark.json)",
    )
    corpus_verify.add_argument(
        "--evaluation-corpus",
        help="labeled evaluation corpus JSON path (default: repository fixtures/evaluation/labeled_corpus.json)",
    )
    corpus_verify.add_argument(
        "--evaluation-fixture-pack",
        help="evaluation fixture pack JSON path (default: repository fixtures/evaluation_fixture_packs/pack.json)",
    )
    corpus_verify.add_argument(
        "--smt-benchmark",
        help="SMT benchmark JSON path (default: repository fixtures/smt_benchmarks/benchmark.json)",
    )
    corpus_verify.add_argument(
        "--min-witness-quality",
        type=float,
        default=0.75,
        help="minimum aggregate witness quality required for release (default: 0.75)",
    )
    corpus_verify.add_argument(
        "--min-differential-agreement",
        type=float,
        default=0.30,
        help="minimum differential agreement rate required for release (default: 0.30)",
    )
    corpus_verify.add_argument(
        "--max-runtime-seconds",
        type=float,
        help="optional wall-clock runtime ceiling for the whole release gate",
    )
    corpus_verify.add_argument(
        "--max-peak-memory-mib",
        type=float,
        help="optional tracemalloc Python-heap ceiling in MiB for the whole release gate",
    )
    corpus_verify.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    corpus_verify.add_argument("--output", help="write corpus verification report to this path instead of stdout")
    beta_report = corpus_subparsers.add_parser(
        "beta-report",
        help="replay beta case studies, upstream issue evidence, and abstention/false-positive tuning",
    )
    beta_report.add_argument(
        "--path",
        help="beta program JSON path (default: repository fixtures/beta/beta_program.json)",
    )
    beta_report.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    beta_report.add_argument("--output", help="write beta report to this path instead of stdout")

    plugins = subparsers.add_parser("plugins", help="inspect PromptABI plugin capabilities")
    plugins.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin module before listing capabilities; may be repeated",
    )
    plugins.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    api_docs = subparsers.add_parser("api-docs", help="generate public API stability docs")
    api_docs.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format (default: markdown)",
    )
    api_docs.add_argument("--output", help="write generated API docs to this path instead of stdout")

    matrix = subparsers.add_parser("matrix", help="show check compatibility and guarantee modes")
    matrix.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin before building the matrix; may be repeated",
    )
    matrix.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    graph = subparsers.add_parser(
        "graph",
        help="visualize artifact, checker, prerequisite, and analysis-resource dependencies",
    )
    graph.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    graph.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location before graphing; may be repeated",
    )
    graph.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin before graphing; may be repeated",
    )
    graph.add_argument(
        "--all-checks",
        action="store_true",
        help="include every registered checker instead of only checks selected by the config",
    )
    graph.add_argument(
        "--format",
        choices=("text", "json", "mermaid"),
        default="text",
        help="output format (default: text)",
    )

    proofs = subparsers.add_parser("proofs", help="show formal proof sketches for supported check families")
    proofs.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    solver = subparsers.add_parser("solver", help="replay reduced finite-SMT obligations without source artifacts")
    solver_subparsers = solver.add_subparsers(dest="solver_command", required=True)
    solver_replay = solver_subparsers.add_parser(
        "replay",
        help="rerun a deterministic solver replay JSON file",
    )
    solver_replay.add_argument("file", help="solver replay JSON file")
    solver_replay.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="inspect environment, optional backends, cache, plugins, config, and artifact setup",
    )
    doctor.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    doctor.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: PROMPTABI_CACHE_DIR or user cache)",
    )
    doctor.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin before inspecting supported backends; may be repeated",
    )
    doctor.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    gallery = subparsers.add_parser("gallery", help="run the curated verified configuration gallery")
    gallery.add_argument(
        "--root",
        help="gallery root containing manifest.json (default: repository examples/gallery)",
    )
    gallery.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    gallery.add_argument(
        "--output",
        help="write gallery report to this path instead of stdout",
    )

    launch_assets = subparsers.add_parser(
        "launch-assets",
        help="generate launch comparison, diagram, demo, benchmark, bug-gallery, and positioning assets",
    )
    launch_assets.add_argument(
        "--output-dir",
        default="launch_assets",
        help="directory to write launch assets (default: launch_assets)",
    )
    launch_assets.add_argument(
        "--repo-root",
        help="repository root containing examples/ and fixtures/ (default: installed PromptABI repository root)",
    )
    launch_assets.add_argument(
        "--benchmark-iterations",
        type=int,
        default=1,
        help="iterations per benchmark case used for the benchmark chart (default: 1)",
    )
    launch_assets.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing generated launch asset files",
    )

    fuzz = subparsers.add_parser("fuzz", help="mutation-based fuzzing workflows")
    fuzz_subparsers = fuzz.add_subparsers(dest="fuzz_command", required=True)
    mutation_fuzz = fuzz_subparsers.add_parser(
        "mutations",
        help="run deterministic mutation fuzzing over PromptABI artifact contracts",
    )
    mutation_fuzz.add_argument(
        "--surface",
        action="append",
        default=[],
        choices=("all", *(surface.value for surface in ALL_FUZZ_SURFACES)),
        help="artifact surface to fuzz; may be repeated (default: all)",
    )
    mutation_fuzz.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    mutation_fuzz.add_argument("--output", help="write mutation-fuzzing report to this path instead of stdout")

    paper = subparsers.add_parser("paper", help="paper artifact and reproducibility commands")
    paper_subparsers = paper.add_subparsers(dest="paper_command", required=True)
    reproducibility = paper_subparsers.add_parser(
        "reproducibility",
        help="write the paper reproducibility package with fixture hashes and expected tables",
    )
    reproducibility.add_argument(
        "--output-dir",
        default="paper_artifact",
        help="directory to write the reproducibility package (default: paper_artifact)",
    )
    reproducibility.add_argument(
        "--benchmark-iterations",
        type=int,
        default=1,
        help="iterations per benchmark case for expected table regeneration (default: 1)",
    )
    reproducibility.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing reproducibility package files in the output directory",
    )

    release = subparsers.add_parser("release", help="release readiness and shipping gates")
    release_subparsers = release.add_subparsers(dest="release_command", required=True)
    release_readiness = release_subparsers.add_parser(
        "readiness",
        help="run the 1.0 release-readiness gate against live repository assets",
    )
    release_readiness.add_argument(
        "--repo-root",
        help="repository root to inspect (default: installed PromptABI repository root)",
    )
    release_readiness.add_argument(
        "--expected-version",
        default="1.0.0",
        help="release version expected in pyproject and package metadata (default: 1.0.0)",
    )
    release_readiness.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    release_readiness.add_argument("--output", help="write release-readiness report to this path instead of stdout")
    compatibility_audit = release_subparsers.add_parser(
        "compatibility-audit",
        help="run the post-1.0 fixture-backed compatibility audit before a minor release",
    )
    compatibility_audit.add_argument(
        "--candidate-version",
        action="append",
        required=True,
        metavar="SURFACE=VERSION",
        help=(
            "candidate version to certify for tokenizer, template, provider, grammar, or framework; "
            "may be repeated, or use all=VERSION"
        ),
    )
    compatibility_audit.add_argument("--seed-root", help="seed corpus root (default: repository fixtures/seed_corpus)")
    compatibility_audit.add_argument(
        "--provider-fixture-root",
        help="provider fixture pack root (default: repository fixtures/provider_fixture_packs)",
    )
    compatibility_audit.add_argument(
        "--structured-schema-root",
        help="structured schema corpus root (default: repository fixtures/structured_schemas)",
    )
    compatibility_audit.add_argument(
        "--grammar-differential-corpus",
        help="grammar differential corpus JSON path (default: repository fixtures/grammar_differential/corpus.json)",
    )
    compatibility_audit.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    compatibility_audit.add_argument("--output", help="write compatibility audit report to this path instead of stdout")

    usage = subparsers.add_parser("usage", help="inspect telemetry-free local command summaries")
    usage_subparsers = usage.add_subparsers(dest="usage_command", required=True)
    usage_summary = usage_subparsers.add_parser(
        "summary",
        help="summarize local PromptABI command summary JSONL records",
    )
    usage_summary.add_argument(
        "--path",
        help="local usage summary JSONL path (default: PROMPTABI_USAGE_SUMMARY_PATH or local state)",
    )
    usage_summary.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    usage_subparsers.add_parser("privacy", help="print local-summary privacy guarantees")

    diagnostics = subparsers.add_parser("diagnostics", help="diagnostic metadata and localization workflows")
    diagnostics_subparsers = diagnostics.add_subparsers(dest="diagnostics_command", required=True)
    diagnostics_catalog = diagnostics_subparsers.add_parser(
        "catalog",
        help="emit a localization-ready message catalog from real verification diagnostics",
    )
    diagnostics_catalog.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    diagnostics_catalog.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    diagnostics_catalog.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    diagnostics_catalog.add_argument("--output", help="write the catalog to this path instead of stdout")
    diagnostics_lsp = diagnostics_subparsers.add_parser(
        "lsp",
        help="emit language-server-style publishDiagnostics payloads for editors",
    )
    diagnostics_lsp.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json upward from cwd",
    )
    diagnostics_lsp.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH_OR_URI",
        help="override or add an artifact location for this run; may be repeated",
    )
    diagnostics_lsp.add_argument(
        "--workspace-root",
        help="workspace root used in diagnostic data (default: config directory)",
    )
    diagnostics_lsp.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:OBJECT]",
        help="import an additional PromptABI plugin before running editor diagnostics; may be repeated",
    )
    diagnostics_lsp.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format (default: json)",
    )
    diagnostics_lsp.add_argument("--output", help="write the editor diagnostics to this path instead of stdout")

    maintain = subparsers.add_parser("maintain", help="maintainer corpus, fixture, and release-note workflows")
    maintain_subparsers = maintain.add_subparsers(dest="maintain_command", required=True)
    maintain_refresh = maintain_subparsers.add_parser(
        "refresh",
        help=(
            "regenerate model-artifact manifests, provider fixtures, expected diagnostics, "
            "corpus diffs, and release notes"
        ),
    )
    maintain_refresh.add_argument(
        "--output-dir",
        required=True,
        help="directory to write maintainer refresh artifacts",
    )
    maintain_refresh.add_argument(
        "--baseline",
        help="previous maintainer refresh directory containing maintainer-snapshot.json",
    )
    maintain_refresh.add_argument(
        "--repo-root",
        help="repository root to refresh (default: installed PromptABI repository root)",
    )
    maintain_refresh.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing maintainer refresh files in the output directory",
    )

    contribute = subparsers.add_parser("contribute", help="contributor infrastructure and onboarding workflows")
    contribute_subparsers = contribute.add_subparsers(dest="contribute_command", required=True)
    contribute_validate = contribute_subparsers.add_parser(
        "validate",
        help="validate issue templates, labels, contributor guides, and CI contributor gates",
    )
    contribute_validate.add_argument(
        "--repo-root",
        help="repository root to validate (default: installed PromptABI repository root)",
    )
    contribute_validate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    minimize = subparsers.add_parser(
        "minimize",
        help="shrink failing PromptABI artifacts into compact upstream repros",
    )
    minimize.add_argument("case", help="JSON file with {'kind': ..., 'input': ...}")
    minimize.add_argument(
        "--oracle",
        choices=tuple(oracle.value for oracle in MinimizationOracle),
        default=MinimizationOracle.CONTAINS.value,
        help="failure-preservation oracle (default: contains)",
    )
    minimize.add_argument(
        "--keep-substring",
        help="substring that must remain in the minimized JSON/text for the contains oracle",
    )
    minimize.add_argument("--config", help="PromptABI config for the diagnostic oracle")
    minimize.add_argument("--artifact-name", help="artifact name to replace for the diagnostic oracle")
    minimize.add_argument("--rule-id", help="diagnostic rule id that must continue firing")
    minimize.add_argument(
        "--artifact-output",
        help="scratch artifact path used by the diagnostic oracle (default: <case>.candidate)",
    )
    minimize.add_argument("--max-steps", type=int, help="maximum accepted shrink steps before stopping")
    minimize.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    github_action = subparsers.add_parser(
        "github-action",
        help="run PromptABI with GitHub Actions caching, SARIF, lockfile, summary, and changed-artifact behavior",
    )
    github_action.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json in the workspace",
    )
    github_action.add_argument(
        "--training-manifest",
        help="training manifest JSON path; runs the dedicated verify-training workflow for data PRs",
    )
    github_action.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="tokenizer artifact to align with a training manifest; may be repeated with --training-manifest",
    )
    github_action.add_argument(
        "--chat-template",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="chat-template artifact to align with a training manifest; may be repeated with --training-manifest",
    )
    github_action.add_argument(
        "--lockfile",
        help="path to a PromptABI lockfile (default: promptabi.lock.json beside the config)",
    )
    github_action.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches (default: .promptabi/cache in the workspace)",
    )
    github_action.add_argument(
        "--sarif-output",
        default="promptabi.sarif",
        help="path for the SARIF log consumed by GitHub code scanning (default: promptabi.sarif)",
    )
    github_action.add_argument(
        "--summary-output",
        help="path for the markdown job summary (default: GITHUB_STEP_SUMMARY when set)",
    )
    github_action.add_argument(
        "--repo-root",
        help="repository checkout root (default: GITHUB_WORKSPACE or cwd)",
    )
    github_action.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    github_action.add_argument(
        "--require-lockfile",
        action="store_true",
        help="fail if artifacts or diagnostics drift from the PromptABI lockfile",
    )
    github_action.add_argument(
        "--changed-only",
        action="store_true",
        help="skip verification when git diff shows no configured PromptABI input changed",
    )
    github_action.add_argument("--base-ref", help="base git ref or SHA for changed-artifact detection")
    github_action.add_argument("--head-ref", help="head git ref or SHA for changed-artifact detection")
    github_action.add_argument(
        "--annotations",
        action="store_true",
        help="also emit GitHub workflow command annotations to stdout",
    )

    pre_commit = subparsers.add_parser(
        "pre-commit",
        help="install or run local PromptABI pre-commit verification workflows",
    )
    pre_commit_subparsers = pre_commit.add_subparsers(dest="pre_commit_command", required=True)
    pre_commit_install = pre_commit_subparsers.add_parser(
        "install",
        help="install a PromptABI-managed git pre-commit hook",
    )
    _add_pre_commit_common_arguments(pre_commit_install)
    pre_commit_install.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing non-PromptABI pre-commit hook",
    )
    pre_commit_install.add_argument(
        "--all",
        action="store_true",
        help="install a hook that runs verification for every commit instead of changed PromptABI inputs only",
    )

    pre_commit_run = pre_commit_subparsers.add_parser(
        "run",
        help="run local PromptABI verification, optionally gated to changed inputs",
    )
    _add_pre_commit_common_arguments(pre_commit_run)
    pre_commit_run.add_argument(
        "--changed-only",
        action="store_true",
        help="skip when no staged PromptABI config, schema, template, tokenizer, tool, budget, or training input changed",
    )
    pre_commit_run.add_argument(
        "--mode",
        choices=("staged", "unstaged", "working-tree"),
        default="staged",
        help="git changed-path view used with --changed-only (default: staged)",
    )
    pre_commit_run.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="explicit repo-relative changed path for tests or custom wrappers; may be repeated",
    )
    pre_commit_run.add_argument(
        "--allow-unstaged",
        action="store_true",
        help="allow verification when selected staged PromptABI inputs also have unstaged edits",
    )
    _add_local_summary_argument(pre_commit_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            config_path = Path(args.config).resolve() if args.config else discover_config()
            cache_dir = _resolve_cache_dir(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            plugin_registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            incremental_requested = bool(args.changed_path or args.changed_from_git)
            if incremental_requested and (args.write_lockfile or args.require_lockfile):
                parser.error("--changed-path/--changed-from-git cannot be combined with lockfile read/write enforcement")
            if args.local_summary is not None and policy_forbids_local_summary(config.policy):
                print("promptabi: organization policy pack forbids --local-summary writes", file=sys.stderr)
                return 2
            session = VerificationSession(config, plugin_registry=plugin_registry)
            if incremental_requested:
                changed_paths = (
                    *explicit_changed_paths(args.changed_path, base_dir=config_path.parent),
                    *(
                        git_changed_paths(args.changed_from_git, cwd=config_path.parent)
                        if args.changed_from_git
                        else ()
                    ),
                )
                result = run_incremental_verification(
                    session,
                    changed_paths=changed_paths,
                    cache_dir=cache_dir,
                    config_path=config_path,
                )
            else:
                result = session.run()
            lockfile_path = _resolve_lockfile_path(args.lockfile, config_path)
            if args.write_lockfile:
                loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
                load_failed = any(diagnostic.severity.value == "error" for diagnostic in load_diagnostics)
                if load_failed:
                    print("promptabi: cannot write lockfile while artifact loading has errors", file=sys.stderr)
                    return 2
                write_lockfile(
                    lockfile_path,
                    build_lockfile(config, loaded_artifacts, result.diagnostics, base_dir=lockfile_path.parent),
                )
            if args.require_lockfile:
                loaded_artifacts, _load_diagnostics = session.load_artifacts_with_diagnostics()
                try:
                    lockfile = load_lockfile(lockfile_path)
                    lock_diagnostics = compare_lockfile(
                        lockfile,
                        config,
                        loaded_artifacts,
                        result.diagnostics,
                        lockfile_path=lockfile_path,
                    )
                except LockfileError as exc:
                    lock_diagnostics = (lockfile_error_diagnostic(exc, lockfile_path=lockfile_path),)
                if lock_diagnostics:
                    result = type(result)(
                        config=result.config,
                        diagnostics=tuple(sorted((*result.diagnostics, *lock_diagnostics), key=lambda item: item.sort_key)),
                    )
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except IncrementalVerificationError as exc:
            print(f"promptabi: cannot resolve incremental changes: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare cache directory: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": config_path,
                    "cache_dir": cache_dir,
                },
                sarif_options=_sarif_options(args, argv),
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        exit_code = _exit_code(result, fail_on=args.fail_on)
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="verify",
            exit_code=exit_code,
            started_at=started_at,
            metadata=_verification_summary_metadata(result, output_format=args.format, fail_on=args.fail_on),
        ):
            return 2
        print(output, end="")
        return exit_code

    if args.command == "prompt-pack" and args.prompt_pack_command == "registry":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            config_path = Path(args.config).resolve() if args.config else discover_config()
            plugin_registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            if args.local_summary is not None and policy_forbids_local_summary(config.policy):
                print("promptabi: organization policy pack forbids --local-summary writes", file=sys.stderr)
                return 2
            session = VerificationSession(config, plugin_registry=plugin_registry)
            result = session.run()
            loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
            if any(diagnostic.severity.value == "error" for diagnostic in load_diagnostics):
                print("promptabi: cannot build prompt-pack registry while artifact loading has errors", file=sys.stderr)
                return 2
            output_path = Path(args.output).expanduser().resolve() if args.output else None
            base_dir = output_path.parent if output_path is not None else config_path.parent
            registry = build_prompt_pack_registry(loaded_artifacts, result.diagnostics, base_dir=base_dir)
            if output_path is not None:
                write_prompt_pack_registry(output_path, registry)
            if args.format == "json":
                output = prompt_pack_registry_to_json(registry)
            else:
                output = render_prompt_pack_registry_text(registry)
                if args.verbose:
                    output += f"config: {config_path}\n"
                    if output_path is not None:
                        output += f"prompt-pack registry: {output_path}\n"
            exit_code = 0
        except (ConfigError, PromptPackLockError, PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write prompt-pack registry: {exc}", file=sys.stderr)
            return 2
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="prompt-pack registry",
            exit_code=exit_code,
            started_at=started_at,
            metadata={"output_format": args.format, "prompt_pack_count": len(registry.entries)},
        ):
            return 2
        if output_path is None or not args.quiet:
            print(output, end="")
        return exit_code

    if args.command == "verify-training":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            cache_dir = _resolve_cache_dir(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            plugin_registry = _load_cli_plugins(args.plugin)
            if args.manifest:
                if args.config:
                    parser.error("--config cannot be combined with --manifest")
                if args.artifact:
                    parser.error("--artifact is only supported with --config")
                config_path = Path(args.manifest).expanduser().resolve()
                config = build_training_config(
                    config_path,
                    name=args.name,
                    tokenizers=_parse_named_path_values(args.tokenizer, "--tokenizer", parser),
                    chat_templates=_parse_named_path_values(args.chat_template, "--chat-template", parser),
                    checks=TRAINING_WORKFLOW_CHECKS,
                )
            else:
                if args.name:
                    parser.error("--name is only supported with --manifest")
                if args.tokenizer or args.chat_template:
                    parser.error("--tokenizer/--chat-template are only supported with --manifest")
                config_path = Path(args.config).resolve() if args.config else discover_config()
                overrides = _parse_artifact_overrides(args.artifact, parser)
                config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            if args.local_summary is not None and policy_forbids_local_summary(config.policy):
                print("promptabi: organization policy pack forbids --local-summary writes", file=sys.stderr)
                return 2
            result = run_training_verification(config, plugin_registry=plugin_registry)
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare training verification: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": config_path,
                    "cache_dir": cache_dir,
                    "heading": "PromptABI training verification",
                },
                sarif_options=_sarif_options(args, argv),
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        exit_code = _exit_code(result, fail_on=args.fail_on)
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="verify-training",
            exit_code=exit_code,
            started_at=started_at,
            metadata=_verification_summary_metadata(result, output_format=args.format, fail_on=args.fail_on),
        ):
            return 2
        print(output, end="")
        return exit_code

    if args.command == "explain":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            cache_dir = _resolve_cache_dir(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            result = VerificationSession(config).run()
            explanation = explain_diagnostic(
                result,
                fingerprint=args.fingerprint,
                rule_id=args.rule_id,
                index=args.index,
                base_dir=config_path.parent,
            )
        except (ConfigError, ExplainError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot prepare cache directory: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            output = render_explanation_json(explanation)
        else:
            output = render_explanation_text(explanation)
        print(output, end="")
        return 0

    if args.command == "bug-report":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            result = VerificationSession(config).run()
            report = generate_bug_report(
                result,
                config_path=config_path,
                fingerprint=args.fingerprint,
                rule_id=args.rule_id,
                index=args.index,
                expected_behavior=args.expected,
                actual_behavior=args.actual,
                base_dir=config_path.parent,
            )
            output = render_bug_report(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
            else:
                print(output, end="")
        except (ConfigError, BugReportError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write bug report: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "bundle" and args.bundle_command == "create":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            overrides = _parse_artifact_overrides(args.artifact, parser)
            bundle = create_signed_verification_bundle(
                config_path,
                key=args.key,
                key_id=args.key_id,
                artifact_overrides=overrides,
                excerpt_bytes=args.excerpt_bytes,
            )
            write_signed_verification_bundle(args.output, bundle, force=args.force)
        except (ConfigError, VerificationBundleError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write verification bundle: {exc}", file=sys.stderr)
            return 2
        print(
            "wrote signed verification bundle: "
            f"{args.output} ({len(bundle.payload['diagnostics'])} diagnostics, hash {bundle.bundle_hash})"
        )
        return 0

    if args.command == "bundle" and args.bundle_command == "verify":
        try:
            bundle = load_signed_verification_bundle(args.bundle)
            verification = verify_signed_verification_bundle(bundle, key=args.key)
            output = (
                json.dumps(verification.to_dict(), indent=2, sort_keys=True) + "\n"
                if args.format == "json"
                else render_bundle_verification_text(verification)
            )
        except VerificationBundleError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0 if verification.ok else 1

    if args.command == "diagnostics" and args.diagnostics_command == "catalog":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            result = VerificationSession(config).run()
            catalog = build_diagnostic_catalog(result.diagnostics)
            output = (
                render_diagnostic_catalog_text(catalog)
                if args.format == "text"
                else render_diagnostic_catalog_json(catalog)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
            else:
                print(output, end="")
        except (ConfigError, LocalizationError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write diagnostic catalog: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "diagnostics" and args.diagnostics_command == "lsp":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            overrides = _parse_artifact_overrides(args.artifact, parser)
            plugin_registry = _load_cli_plugins(args.plugin)
            report = build_editor_diagnostic_report(
                config_path=config_path,
                artifact_overrides=overrides,
                workspace_root=args.workspace_root,
                plugin_registry=plugin_registry,
            )
            output = (
                render_editor_diagnostic_text(report)
                if args.format == "text"
                else render_editor_diagnostic_json(report)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
            else:
                print(output, end="")
        except (ConfigError, EditorProtocolError, PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write editor diagnostics: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "diff":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            baseline_path = Path(args.baseline).resolve()
            current_path = Path(args.current).resolve()
            plugin_registry = _load_cli_plugins(args.plugin)
            result = diff_config_files(baseline_path, current_path)
        except ConfigError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": current_path if args.verbose else None,
                    "heading": "PromptABI diff",
                },
                sarif_options=_sarif_options(args, argv),
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        if args.format == "text" and args.verbose:
            output = output.replace(f"config: {current_path}\n", f"baseline: {baseline_path}\ncurrent: {current_path}\n")
        exit_code = _exit_code(result, fail_on=args.fail_on)
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="diff",
            exit_code=exit_code,
            started_at=started_at,
            metadata=_verification_summary_metadata(result, output_format=args.format, fail_on=args.fail_on),
        ):
            return 2
        print(output, end="")
        return exit_code

    if args.command == "version-gate":
        try:
            report = run_version_gate(
                Path(args.baseline).resolve(),
                Path(args.current).resolve(),
                allowed_impact=args.allowed_impact,
                policy_path=args.policy,
            )
            output = render_version_gate_json(report) if args.format == "json" else render_version_gate_text(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote version-gate report: {args.output} ({len(report.findings)} findings)")
            else:
                print(output, end="")
        except (ConfigError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write version-gate report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "init":
        try:
            written = scaffold_promptabi_project(
                stack=args.stack,
                output_dir=args.output_dir,
                name=args.name,
                config_filename=args.config,
                force=args.force,
            )
        except InitError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(f"wrote PromptABI {args.stack} scaffold:")
        for path in written:
            print(f"  {path}")
        print(f"next: promptabi verify --config {written[0]}")
        return 0

    if args.command == "prompt-pack" and args.prompt_pack_command == "lock":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            if not args.write and not args.check:
                args.write = True
            config_path = Path(args.config).resolve() if args.config else discover_config()
            plugin_registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            if args.local_summary is not None and policy_forbids_local_summary(config.policy):
                print("promptabi: organization policy pack forbids --local-summary writes", file=sys.stderr)
                return 2
            session = VerificationSession(config, plugin_registry=plugin_registry)
            result = session.run()
            loaded_artifacts, load_diagnostics = session.load_artifacts_with_diagnostics()
            load_failed = any(diagnostic.severity.value == "error" for diagnostic in load_diagnostics)
            lockfile_path = _resolve_prompt_pack_lockfile_path(args.lockfile, config_path)
            lock_diagnostics = ()
            if args.write:
                if load_failed:
                    print("promptabi: cannot write prompt-pack lockfile while artifact loading has errors", file=sys.stderr)
                    return 2
                write_prompt_pack_lockfile(
                    lockfile_path,
                    build_prompt_pack_lockfile(loaded_artifacts, result.diagnostics, base_dir=lockfile_path.parent),
                )
            if args.check:
                try:
                    lock_diagnostics = compare_prompt_pack_lockfile(
                        load_prompt_pack_lockfile(lockfile_path),
                        loaded_artifacts,
                        result.diagnostics,
                        lockfile_path=lockfile_path,
                    )
                except PromptPackLockError as exc:
                    lock_diagnostics = (prompt_pack_lock_error_diagnostic(exc, lockfile_path=lockfile_path),)
            if lock_diagnostics:
                result = VerificationResult(
                    config=result.config,
                    diagnostics=tuple(sorted((*result.diagnostics, *lock_diagnostics), key=lambda item: item.sort_key)),
                )
        except (ConfigError, PromptPackLockError, PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write prompt-pack lockfile: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": config_path if args.verbose else None,
                    "heading": "PromptABI prompt-pack lock",
                },
                sarif_options=_sarif_options(args, argv),
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        if args.format == "text" and args.verbose:
            output += f"prompt-pack lockfile: {lockfile_path}\n"
        exit_code = _exit_code(result, fail_on=args.fail_on)
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="prompt-pack lock",
            exit_code=exit_code,
            started_at=started_at,
            metadata=_verification_summary_metadata(result, output_format=args.format, fail_on=args.fail_on),
        ):
            return 2
        print(output, end="")
        return exit_code

    if args.command == "prompt-pack" and args.prompt_pack_command == "upgrade":
        started_at = time.perf_counter()
        try:
            if args.quiet and args.verbose:
                parser.error("--quiet and --verbose cannot be used together")
            config_path = Path(args.config).resolve() if args.config else discover_config()
            baseline_path = Path(args.baseline_lockfile).expanduser().resolve()
            plugin_registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            if args.local_summary is not None and policy_forbids_local_summary(config.policy):
                print("promptabi: organization policy pack forbids --local-summary writes", file=sys.stderr)
                return 2
            session = VerificationSession(config, plugin_registry=plugin_registry)
            result = session.run()
            loaded_artifacts, _load_diagnostics = session.load_artifacts_with_diagnostics()
            upgrade_diagnostics = compare_prompt_pack_upgrade(
                load_prompt_pack_lockfile(baseline_path),
                loaded_artifacts,
                result.diagnostics,
                baseline_path=baseline_path,
            )
            result = VerificationResult(
                config=result.config,
                diagnostics=tuple(sorted((*result.diagnostics, *upgrade_diagnostics), key=lambda item: item.sort_key)),
            )
        except (ConfigError, PromptPackLockError, PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot compare prompt-pack upgrade: {exc}", file=sys.stderr)
            return 2
        try:
            output = _render_verification_output(
                result,
                args.format,
                plugin_registry=plugin_registry,
                text_kwargs={
                    "verbosity": args.verbose - args.quiet,
                    "config_path": config_path if args.verbose else None,
                    "heading": "PromptABI prompt-pack upgrade",
                },
                sarif_options=_sarif_options(args, argv),
            )
        except PluginError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        if args.format == "text" and args.verbose:
            output += f"baseline prompt-pack lockfile: {baseline_path}\n"
        exit_code = _exit_code(result, fail_on=args.fail_on)
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="prompt-pack upgrade",
            exit_code=exit_code,
            started_at=started_at,
            metadata=_verification_summary_metadata(result, output_format=args.format, fail_on=args.fail_on),
        ):
            return 2
        print(output, end="")
        return exit_code

    if args.command == "corpus" and args.corpus_command == "manifest":
        try:
            if args.output:
                manifest = write_seed_corpus_manifest(args.output, root=args.root)
                print(f"wrote seed corpus manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_seed_corpus_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except SeedCorpusError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "structured-schema-manifest":
        try:
            if args.output:
                manifest = write_structured_schema_corpus_manifest(args.output, root=args.root)
                print(f"wrote structured schema corpus manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_structured_schema_corpus_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except StructuredSchemaCorpusError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "provider-fixture-manifest":
        try:
            if args.output:
                manifest = write_provider_fixture_pack_manifest(args.output, root=args.root)
                print(f"wrote provider fixture pack manifest: {args.output} ({manifest['entry_count']} entries)")
            else:
                manifest = build_provider_fixture_pack_manifest(args.root)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except ProviderFixturePackError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "real-bug-benchmark":
        try:
            if args.output:
                manifest = write_real_bug_benchmark_manifest(args.output, path=args.path)
                print(f"wrote real-bug benchmark manifest: {args.output} ({manifest['case_count']} cases)")
            else:
                manifest = build_real_bug_benchmark_manifest(args.path)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except RealBugBenchmarkError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write real-bug benchmark manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "evaluation-fixture-pack":
        try:
            if args.output:
                manifest = write_evaluation_fixture_pack_manifest(args.output, path=args.path)
                print(f"wrote evaluation fixture pack manifest: {args.output} ({manifest['case_count']} cases)")
            else:
                manifest = build_evaluation_fixture_pack_manifest(args.path)
                print(json.dumps(manifest, indent=2, sort_keys=True))
        except EvaluationFixturePackError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write evaluation fixture pack manifest: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "corpus" and args.corpus_command == "evaluation":
        try:
            report = run_evaluation(args.corpus)
            output = render_evaluation_text(report) if args.format == "text" else render_evaluation_json(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote evaluation report: {args.output} ({len(report.results)} cases)")
            else:
                print(output, end="")
        except EvaluationError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write evaluation report: {exc}", file=sys.stderr)
            return 2
        return 0 if all(result.passed for result in report.results) else 1

    if args.command == "corpus" and args.corpus_command == "evaluation-reproducibility":
        try:
            configs = args.config or None
            report = build_evaluation_reproducibility_report(configs)
            output = (
                render_evaluation_reproducibility_text(report)
                if args.format == "text"
                else render_evaluation_reproducibility_json(report)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote evaluation reproducibility report: {args.output} ({len(report.configs)} configs)")
            else:
                print(output, end="")
        except EvaluationReproducibilityError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write evaluation reproducibility report: {exc}", file=sys.stderr)
            return 2
        return 0 if all(config["reproducibility_status"] == "complete" for config in report.configs) else 1

    if args.command == "corpus" and args.corpus_command == "smt-benchmark":
        try:
            manifest = build_smt_benchmark_manifest(args.path)
            output = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                if args.format == "json"
                else render_smt_benchmark_text(manifest)
            )
            if args.output:
                if args.format == "json":
                    write_smt_benchmark_manifest(args.output, path=args.path)
                else:
                    Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote SMT benchmark manifest: {args.output} ({manifest['case_count']} cases)")
            else:
                print(output, end="")
        except SmtBenchmarkError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write SMT benchmark manifest: {exc}", file=sys.stderr)
            return 2
        return 0 if manifest["all_cases_passed"] else 1

    if args.command == "corpus" and args.corpus_command == "verify":
        try:
            thresholds = CorpusVerificationThresholds(
                min_witness_quality=args.min_witness_quality,
                min_differential_agreement=args.min_differential_agreement,
                max_runtime_seconds=args.max_runtime_seconds,
                max_peak_memory_bytes=(
                    int(args.max_peak_memory_mib * 1024 * 1024)
                    if args.max_peak_memory_mib is not None
                    else None
                ),
            )
            report = run_corpus_verification(
                seed_root=args.seed_root,
                structured_schema_root=args.structured_schema_root,
                provider_fixture_root=args.provider_fixture_root,
                real_bug_benchmark_path=args.real_bug_benchmark,
                evaluation_corpus_path=args.evaluation_corpus,
                evaluation_fixture_pack_path=args.evaluation_fixture_pack,
                smt_benchmark_path=args.smt_benchmark,
                thresholds=thresholds,
            )
            output = (
                render_corpus_verification_json(report)
                if args.format == "json"
                else render_corpus_verification_text(report)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote corpus verification report: {args.output} ({len(report.checks)} checks)")
            else:
                print(output, end="")
        except (
            CorpusVerificationError,
            EvaluationError,
            EvaluationFixturePackError,
            ProviderFixturePackError,
            RealBugBenchmarkError,
            SeedCorpusError,
            SmtBenchmarkError,
            StructuredSchemaCorpusError,
        ) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write corpus verification report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "corpus" and args.corpus_command == "beta-report":
        try:
            report = run_beta_program(args.path)
            output = render_beta_program_json(report) if args.format == "json" else render_beta_program_text(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote beta report: {args.output} ({len(report.results)} cases)")
            else:
                print(output, end="")
        except BetaProgramError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write beta report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.passed else 1

    if args.command == "plugins":
        try:
            registry = _load_cli_plugins(args.plugin)
            output = render_plugin_capabilities(registry, output_format=args.format)
        except (PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "api-docs":
        try:
            manifest = build_public_api_manifest()
            output = (
                render_public_api_manifest_json(manifest)
                if args.format == "json"
                else render_public_api_manifest_markdown(manifest)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
            else:
                print(output, end="")
        except OSError as exc:
            print(f"promptabi: cannot write API docs: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "matrix":
        try:
            registry = _load_cli_plugins(args.plugin)
            matrix = build_compatibility_matrix(plugin_registry=registry)
            if args.format == "json":
                output = render_compatibility_matrix_json(matrix)
            else:
                output = render_compatibility_matrix_text(matrix)
        except (PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "graph":
        try:
            config_path = Path(args.config).resolve() if args.config else discover_config()
            registry = _load_cli_plugins(args.plugin)
            overrides = _parse_artifact_overrides(args.artifact, parser)
            config = load_config(config_path).with_artifact_overrides(overrides, base_dir=Path.cwd())
            graph_report = build_dependency_graph(
                config,
                plugin_registry=registry,
                include_all_checks=args.all_checks,
            )
            if args.format == "json":
                output = render_dependency_graph_json(graph_report)
            elif args.format == "mermaid":
                output = render_dependency_graph_mermaid(graph_report)
            else:
                output = render_dependency_graph_text(graph_report)
        except (ConfigError, PluginError, ValueError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "proofs":
        report = build_supported_proof_catalog()
        output = render_proof_sketch_report_json(report) if args.format == "json" else render_proof_sketch_report_text(report)
        print(output, end="")
        return 0 if report.passed else 1

    if args.command == "solver" and args.solver_command == "replay":
        try:
            replay_file = SolverReplayFile.read_json(args.file)
            report = replay_file.replay()
            output = render_solver_replay_json(report) if args.format == "json" else render_solver_replay_text(report)
        except (OSError, ValueError) as exc:
            print(f"promptabi: cannot replay solver file: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0 if report.ok else 1

    if args.command == "doctor":
        report = run_doctor(
            config_path=args.config,
            cache_dir=args.cache_dir,
            plugin_specs=tuple(args.plugin),
        )
        output = render_doctor_json(report) if args.format == "json" else render_doctor_text(report)
        print(output, end="")
        return 0 if report.ok else 1

    if args.command == "gallery":
        try:
            report = build_gallery(args.root)
            output = render_gallery_json(report) if args.format == "json" else render_gallery_text(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote gallery report: {args.output} ({len(report.entries)} entries)")
            else:
                print(output, end="")
        except (GalleryError, ConfigError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write gallery report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "launch-assets":
        try:
            bundle = write_launch_assets(
                args.output_dir,
                repo_root=args.repo_root,
                benchmark_iterations=args.benchmark_iterations,
                force=args.force,
            )
        except (
            LaunchAssetError,
            EvaluationError,
            RealBugBenchmarkError,
            BetaProgramError,
            ValueError,
        ) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write launch assets: {exc}", file=sys.stderr)
            return 2
        print(render_launch_asset_summary(bundle), end="")
        return 0

    if args.command == "fuzz" and args.fuzz_command == "mutations":
        try:
            report = run_mutation_fuzzing(args.surface or ("all",))
            output = render_mutation_fuzz_text(report) if args.format == "text" else render_mutation_fuzz_json(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote mutation-fuzzing report: {args.output} ({report.mutation_count} mutations)")
            else:
                print(output, end="")
        except MutationFuzzingError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write mutation-fuzzing report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.introduced_violation_count else 1

    if args.command == "paper" and args.paper_command == "reproducibility":
        try:
            package = write_reproducibility_package(
                args.output_dir,
                benchmark_iterations=args.benchmark_iterations,
                force=args.force,
            )
        except ReproducibilityPackageError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write reproducibility package: {exc}", file=sys.stderr)
            return 2
        print(
            "wrote paper reproducibility package: "
            f"{args.output_dir} ({package.manifest['summary']['fixture_file_count']} fixture files)"
        )
        return 0

    if args.command == "release" and args.release_command == "readiness":
        try:
            report = build_release_readiness_report(
                args.repo_root,
                expected_version=args.expected_version,
            )
            output = render_release_readiness_json(report) if args.format == "json" else render_release_readiness_text(report)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote release-readiness report: {args.output} ({len(report.checks)} checks)")
            else:
                print(output, end="")
        except (
            ReleaseReadinessError,
            SeedCorpusError,
            RealBugBenchmarkError,
            ReproducibilityPackageError,
            ValueError,
        ) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write release-readiness report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "release" and args.release_command == "compatibility-audit":
        try:
            report = run_compatibility_audit(
                _parse_candidate_versions(args.candidate_version, parser),
                seed_root=args.seed_root,
                provider_fixture_root=args.provider_fixture_root,
                structured_schema_root=args.structured_schema_root,
                grammar_differential_corpus=args.grammar_differential_corpus,
            )
            output = (
                render_compatibility_audit_json(report)
                if args.format == "json"
                else render_compatibility_audit_text(report)
            )
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"wrote compatibility audit report: {args.output} ({len(report.targets)} targets)")
            else:
                print(output, end="")
        except (
            CompatibilityAuditError,
            CorpusVerificationError,
            EvaluationError,
            ProviderFixturePackError,
            RealBugBenchmarkError,
            SeedCorpusError,
            StructuredSchemaCorpusError,
            ValueError,
        ) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write compatibility audit report: {exc}", file=sys.stderr)
            return 2
        return 0 if report.ok else 1

    if args.command == "usage" and args.usage_command == "summary":
        try:
            report = summarize_local_command_usage(args.path)
            output = render_usage_summary_json(report) if args.format == "json" else render_usage_summary_text(report)
        except UsageAnalyticsError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "usage" and args.usage_command == "privacy":
        print(render_usage_privacy_text(), end="")
        return 0

    if args.command == "maintain" and args.maintain_command == "refresh":
        try:
            refresh = refresh_maintainer_artifacts(
                args.output_dir,
                baseline_dir=args.baseline,
                repo_root=args.repo_root,
                force=args.force,
            )
        except (
            MaintainerToolingError,
            CorpusVerificationError,
            EvaluationError,
            ProviderFixturePackError,
            RealBugBenchmarkError,
            SeedCorpusError,
            StructuredSchemaCorpusError,
            ConfigError,
        ) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write maintainer refresh artifacts: {exc}", file=sys.stderr)
            return 2
        print(
            "wrote maintainer refresh artifacts: "
            f"{refresh.output_dir} ({len(refresh.written_files)} files, diff={refresh.diff['status']})"
        )
        return 0 if refresh.snapshot["corpus_verification"]["ok"] else 1  # type: ignore[index]

    if args.command == "contribute" and args.contribute_command == "validate":
        try:
            report = validate_contributor_infrastructure(args.repo_root)
            output = (
                render_contributor_validation_json(report)
                if args.format == "json"
                else render_contributor_validation_text(report)
            )
        except ContributorValidationError as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0 if report.ok else 1

    if args.command == "minimize":
        try:
            if args.max_steps is not None and args.max_steps <= 0:
                parser.error("--max-steps must be positive")
            kind, value = load_minimization_case(args.case)
            predicate = _minimization_predicate(args, parser)
            result = minimize_repro(value, predicate, kind=kind, max_steps=args.max_steps)
            output = render_minimization_json(result) if args.format == "json" else render_minimization_text(result)
        except (ConfigError, MinimizationError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        print(output, end="")
        return 0

    if args.command == "github-action":
        try:
            if (args.tokenizer or args.chat_template) and not args.training_manifest:
                parser.error("--tokenizer/--chat-template are only supported with --training-manifest")
            run = run_github_action(
                config_path=args.config,
                training_manifest_path=args.training_manifest,
                tokenizers=_parse_named_path_values(args.tokenizer, "--tokenizer", parser),
                chat_templates=_parse_named_path_values(args.chat_template, "--chat-template", parser),
                lockfile_path=args.lockfile,
                cache_dir=args.cache_dir,
                sarif_output=args.sarif_output,
                summary_output=args.summary_output,
                repo_root=args.repo_root,
                fail_on=args.fail_on,
                require_lockfile=args.require_lockfile,
                changed_only=args.changed_only,
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                annotations=args.annotations,
                argv=argv,
            )
        except (ConfigError, GitHubActionError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot write GitHub Action outputs: {exc}", file=sys.stderr)
            return 2
        if run.skipped:
            print("PromptABI GitHub Action: skipped (no configured PromptABI inputs changed)")
        else:
            assert run.result is not None
            errors = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "error")
            warnings = sum(1 for diagnostic in run.result.diagnostics if diagnostic.severity.value == "warning")
            print(
                "PromptABI GitHub Action: "
                f"{'PASS' if run.result.ok else 'FAIL'} "
                f"({errors} errors, {warnings} warnings, SARIF: {run.sarif_path})"
            )
        return run.exit_code

    if args.command == "pre-commit" and args.pre_commit_command == "install":
        try:
            hook_path = install_pre_commit_hook(
                config_path=args.config,
                repo_root=args.repo_root,
                cache_dir=args.cache_dir,
                fail_on=args.fail_on,
                require_lockfile=args.require_lockfile,
                changed_only=not args.all,
                force=args.force,
            )
        except (ConfigError, LocalWorkflowError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot install pre-commit hook: {exc}", file=sys.stderr)
            return 2
        print(f"installed PromptABI pre-commit hook: {hook_path}")
        return 0

    if args.command == "pre-commit" and args.pre_commit_command == "run":
        started_at = time.perf_counter()
        try:
            run = run_local_workflow(
                config_path=args.config,
                repo_root=args.repo_root,
                cache_dir=args.cache_dir,
                fail_on=args.fail_on,
                require_lockfile=args.require_lockfile,
                changed_only=args.changed_only,
                mode=args.mode,
                changed_paths=args.changed_path or None,
                allow_unstaged=args.allow_unstaged,
            )
        except (ConfigError, LocalWorkflowError) as exc:
            print(f"promptabi: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"promptabi: cannot run pre-commit workflow: {exc}", file=sys.stderr)
            return 2
        if not _write_local_summary_if_requested(
            args.local_summary,
            command="pre-commit run",
            exit_code=run.exit_code,
            started_at=started_at,
            metadata={
                "skipped": run.skipped,
                "changed_count": len(run.changed_paths),
                "candidate_count": len(run.candidate_paths),
                "selected_count": len(run.selected_paths),
                "diagnostics_total": len(run.diagnostics),
            },
        ):
            return 2
        print(render_local_workflow_text(run), end="")
        return run.exit_code

    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_artifact_overrides(values: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, location = value.partition("=")
        if not separator or not name or not location:
            parser.error("--artifact values must use NAME=PATH_OR_URI")
        overrides[name] = location
    return overrides


def _parse_named_path_values(
    values: Sequence[str],
    option_name: str,
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error(f"{option_name} values must use NAME=PATH")
        parsed.append((name, path))
    return tuple(parsed)


def _parse_candidate_versions(values: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    versions: dict[str, str] = {}
    for value in values:
        surface, separator, version = value.partition("=")
        if not separator or not surface or not version:
            parser.error("--candidate-version values must use SURFACE=VERSION")
        versions[surface] = version
    return versions


def _minimization_predicate(args, parser: argparse.ArgumentParser):
    if args.oracle == MinimizationOracle.CONTAINS.value:
        if not args.keep_substring:
            parser.error("--keep-substring is required for --oracle contains")
        return contains_oracle(args.keep_substring)
    if args.oracle == MinimizationOracle.DIAGNOSTIC.value:
        missing = [
            flag
            for flag, value in (
                ("--config", args.config),
                ("--artifact-name", args.artifact_name),
                ("--rule-id", args.rule_id),
            )
            if not value
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required for --oracle diagnostic")
        return diagnostic_oracle(
            config_path=args.config,
            artifact_name=args.artifact_name,
            rule_id=args.rule_id,
            case_path=args.artifact_output or f"{args.case}.candidate",
        )
    parser.error(f"unsupported minimization oracle: {args.oracle}")


def _add_github_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sarif-category",
        help="stable GitHub code-scanning SARIF category for this analysis run",
    )
    parser.add_argument(
        "--sarif-checkout-uri-base",
        help="repository checkout root for SARIF uriBaseId locations; defaults to GITHUB_WORKSPACE when set",
    )
    parser.add_argument(
        "--sarif-include-invocation",
        action="store_true",
        help="include deterministic invocation metadata in SARIF output",
    )


def _add_pre_commit_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="path to a PromptABI JSON config; defaults to discovering promptabi.json from the repo root",
    )
    parser.add_argument(
        "--repo-root",
        help="repository root (default: git rev-parse --show-toplevel or cwd)",
    )
    parser.add_argument(
        "--cache-dir",
        help="directory for reusable PromptABI analysis caches",
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "any", "never"),
        default="error",
        help="exit with code 1 at this diagnostic threshold (default: error)",
    )
    parser.add_argument(
        "--require-lockfile",
        action="store_true",
        help="fail if artifacts or diagnostics drift from the PromptABI lockfile",
    )


def _add_local_summary_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-summary",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "append a telemetry-free local command summary JSONL record; optionally choose PATH "
            "(default: PROMPTABI_USAGE_SUMMARY_PATH or local state)"
        ),
    )


def _sarif_options(args, argv: Sequence[str] | None) -> SarifRenderOptions:
    checkout_base = _resolve_checkout_uri_base(args.sarif_checkout_uri_base)
    command_line = None
    if args.sarif_include_invocation:
        words = ["promptabi", *(argv if argv is not None else sys.argv[1:])]
        command_line = " ".join(shlex.quote(str(word)) for word in words)
    return SarifRenderOptions(
        category=args.sarif_category,
        checkout_uri_base=checkout_base,
        include_invocation=args.sarif_include_invocation,
        command_line=command_line,
        working_directory=checkout_base if checkout_base is not None else None,
    )


def _resolve_checkout_uri_base(value: str | None) -> Path | None:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("GITHUB_WORKSPACE")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def _resolve_cache_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("PROMPTABI_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return (Path(xdg_cache_home).expanduser() / "promptabi").resolve()
    return (Path.home() / ".cache" / "promptabi").resolve()


def _resolve_lockfile_path(value: str | None, config_path: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return config_path.with_name("promptabi.lock.json")


def _resolve_prompt_pack_lockfile_path(value: str | None, config_path: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return config_path.with_name("prompt-pack.lock.json")


def _load_cli_plugins(values: Sequence[str]) -> PluginRegistry:
    registry = create_first_party_plugin_registry()
    if values:
        load_plugin_modules(values, registry=registry)
    return registry


def _render_verification_output(
    result,
    output_format: str,
    *,
    plugin_registry: PluginRegistry,
    text_kwargs: dict[str, object],
    sarif_options: SarifRenderOptions | None = None,
) -> str:
    if output_format == "text":
        return render_text(result, **text_kwargs)
    if output_format == "json":
        return render_json(result)
    if output_format == "html":
        return render_html(result)
    if output_format == "sarif":
        return render_sarif(result, options=sarif_options)
    if output_format == "github-annotations":
        checkout_base = sarif_options.checkout_uri_base if sarif_options is not None else None
        return render_github_annotations(result, checkout_uri_base=checkout_base)
    return plugin_registry.render(output_format, result)


def _verification_summary_metadata(result, *, output_format: str, fail_on: str) -> dict[str, object]:
    severities = [diagnostic.severity.value for diagnostic in result.diagnostics]
    return {
        "ok": result.ok,
        "diagnostics_total": len(result.diagnostics),
        "errors": severities.count("error"),
        "warnings": severities.count("warning"),
        "info": severities.count("info"),
        "artifact_count": len(result.config.artifact_bundle.artifacts),
        "check_count": len(result.config.checks),
        "format": output_format,
        "fail_on": fail_on,
    }


def _write_local_summary_if_requested(
    requested_path: str | None,
    *,
    command: str,
    exit_code: int,
    started_at: float,
    metadata: dict[str, object],
) -> bool:
    if requested_path is None:
        return True
    path = requested_path or None
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    try:
        append_local_command_summary(
            path=path,
            command=command,
            exit_code=exit_code,
            duration_ms=duration_ms,
            metadata=metadata,
        )
    except UsageAnalyticsError as exc:
        print(f"promptabi: {exc}", file=sys.stderr)
        return False
    return True


def _exit_code(result, *, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    severities = {diagnostic.severity.value for diagnostic in result.diagnostics}
    if fail_on == "any":
        return 1 if severities else 0
    if fail_on == "warning":
        return 1 if severities.intersection({"error", "warning"}) else 0
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
