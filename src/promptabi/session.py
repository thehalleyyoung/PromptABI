"""Verification session orchestration for PromptABI."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, urlparse

from .artifacts import ArtifactKind, TokenizerArtifact
from .budgets import TokenBudgetFinding, TokenBudgetReport, analyze_token_budget
from .chat_templates import ChatTemplateParseError, parse_hf_tokenizer_config_chat_template
from .config import VerificationConfig, load_config
from .diagnostics import CheckMode, Diagnostic, DiagnosticSeverity, SourceSpan, WitnessStep, WitnessTrace, diagnostic_sort_key
from .enterprise import enterprise_readiness_diagnostics
from .formal import SolverStatus
from .grammar_emptiness import (
    GrammarTokenizerEmptinessReport,
    GrammarTokenizerEmptinessStatus,
    analyze_tokenizer_grammar_emptiness,
)
from .grammar_ambiguity import (
    GrammarTokenizerAmbiguityFinding,
    GrammarTokenizerAmbiguityKind,
    GrammarTokenizerAmbiguityReport,
    analyze_tokenizer_grammar_ambiguity,
)
from .grammar_differential import (
    GrammarDifferentialCaseReport,
    analyze_grammar_differential_corpus,
)
from .parser_compatibility import (
    ParserCompatibilityDirection,
    ParserCompatibilityObservation,
    ParserCompatibilityReport,
    ParserCompatibilityStatus,
    analyze_parser_compatibility,
)
from .first_party_plugins import create_first_party_plugin_registry
from .plugins import PluginRegistry
from .policies import apply_policy_diagnostics
from .provider_fixture_replay import ProviderFixtureReplayCase, ProviderFixtureReplayFinding, analyze_provider_fixture_replay
from .provider_migration import ProviderMigrationFinding, analyze_provider_migration
from .loaders import ArtifactLoadError, ArtifactLoadWarning, ArtifactLoader, LoadedArtifact
from .lockfiles import LOCKFILE_CHECK_MODES
from .role_boundaries import RoleBoundaryForgeryFinding, analyze_role_boundary_nonforgeability
from .static_contracts import StaticContractFinding, StaticContractReport, analyze_static_contracts
from .stop_analysis import (
    StopCollision,
    StopPolicyTokenizerAnalysisReport,
    StopSequenceAnalysis,
    StopTokenIdAnalysis,
    analyze_stop_policy_tokenizer,
)
from .stop_differential import (
    StopDifferentialAbstention,
    StopDifferentialMismatch,
    StopDifferentialReport,
    analyze_stop_differential,
)
from .stop_overreachability import (
    StopOverreachabilityAbstention,
    StopOverreachabilityFinding,
    StopOverreachabilityReport,
    analyze_stop_overreachability,
)
from .tool_serialization import ToolSerializationFinding, analyze_tool_call_serialization
from .tokenizer_drift import TokenizerDriftAbstention, TokenizerDriftFinding, analyze_tokenizer_config_drift
from .tokenizers import TokenizerAdapter, TokenizerError, load_tokenizer


CHECK_MODE_CATALOG: dict[str, tuple[CheckMode, ...]] = {
    "repository-skeleton": (CheckMode.HEURISTIC,),
    "artifact-missing": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-load-failed": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-unpinned": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-weak-pin": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-pin-invalid": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-hash-mismatch": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-missing-hash": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-missing-license": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-missing-source": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-untrusted-source": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-nonreproducible-remote": (CheckMode.SOUND, CheckMode.COMPLETE),
    "artifact-provenance-verified": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-internal-fixture-unsafe": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-local-resource-hash-abstained": (CheckMode.ABSTAINING, CheckMode.COMPLETE),
    "enterprise-local-resource-hash-mismatch": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-local-resource-missing": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-no-network-violation": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-private-index-untrusted": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-readiness-verified": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-solver-sandbox-incomplete": (CheckMode.SOUND, CheckMode.COMPLETE),
    "enterprise-solver-sandbox-unsafe": (CheckMode.SOUND, CheckMode.COMPLETE),
    "lockfile-artifact-added": LOCKFILE_CHECK_MODES,
    "lockfile-artifact-drift": LOCKFILE_CHECK_MODES,
    "lockfile-artifact-missing": LOCKFILE_CHECK_MODES,
    "lockfile-config-drift": LOCKFILE_CHECK_MODES,
    "lockfile-diagnostic-baseline-drift": LOCKFILE_CHECK_MODES,
    "lockfile-library-version-drift": LOCKFILE_CHECK_MODES,
    "lockfile-load-failed": LOCKFILE_CHECK_MODES,
    "lockfile-provider-fixture-drift": LOCKFILE_CHECK_MODES,
    "lockfile-verified": LOCKFILE_CHECK_MODES,
    "role-boundary-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "role-boundary-nonforgeability": (CheckMode.SOUND, CheckMode.BOUNDED),
    "stop-tokenizer-abstained": (CheckMode.ABSTAINING,),
    "stop-tokenizer-alignment": (CheckMode.HEURISTIC,),
    "stop-tokenizer-ambiguous": (CheckMode.HEURISTIC,),
    "stop-tokenizer-collision": (CheckMode.HEURISTIC,),
    "stop-tokenizer-special-interaction": (CheckMode.HEURISTIC,),
    "stop-tokenizer-unreachable": (CheckMode.SOUND,),
    "stop-differential-abstained": (CheckMode.ABSTAINING, CheckMode.HEURISTIC,),
    "stop-differential-agreement": (CheckMode.HEURISTIC,),
    "stop-differential-mismatch": (CheckMode.HEURISTIC,),
    "stop-overreach-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "stop-overreach-content": (CheckMode.SOUND, CheckMode.BOUNDED),
    "stop-overreach-structural": (CheckMode.SOUND, CheckMode.BOUNDED),
    "grammar-tokenizer-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "grammar-tokenizer-ambiguity": (CheckMode.SOUND, CheckMode.BOUNDED),
    "grammar-tokenizer-ambiguity-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "grammar-tokenizer-empty": (CheckMode.SOUND, CheckMode.BOUNDED),
    "grammar-tokenizer-satisfiable": (CheckMode.SOUND, CheckMode.BOUNDED),
    "grammar-differential-abstained": (CheckMode.ABSTAINING, CheckMode.HEURISTIC),
    "grammar-differential-agreement": (CheckMode.HEURISTIC,),
    "grammar-differential-mismatch": (CheckMode.HEURISTIC,),
    "parser-compatibility-abstained": (CheckMode.ABSTAINING, CheckMode.HEURISTIC),
    "parser-compatibility-agreement": (CheckMode.HEURISTIC,),
    "parser-compatibility-mismatch": (CheckMode.HEURISTIC,),
    "provider-fixture-replay": (CheckMode.BOUNDED, CheckMode.HEURISTIC),
    "provider-migration": (CheckMode.BOUNDED, CheckMode.HEURISTIC),
    "tool-schema-ingestion": (CheckMode.SOUND, CheckMode.COMPLETE),
    "tool-serialization": (CheckMode.BOUNDED, CheckMode.HEURISTIC),
    "tokenizer-drift": (CheckMode.SOUND, CheckMode.COMPLETE),
    "tokenizer-drift-abstained": (CheckMode.ABSTAINING, CheckMode.COMPLETE),
    "tokenizer-drift-clean": (CheckMode.SOUND, CheckMode.COMPLETE),
    "token-budget-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "token-budget-context-conflict": (CheckMode.BOUNDED, CheckMode.HEURISTIC),
    "token-budget-framework-truncation": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-invalid": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-must-survive": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-model": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-policy-overflow": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-required-overflow": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-required-truncated": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-segment-overflow": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-total-overflow": (CheckMode.SOUND, CheckMode.BOUNDED),
    "token-budget-truncation-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED),
    "rag-chunk-boundary-drift": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-citation-loss": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-metadata-inflation": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-overlap-accounting": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-payload-truncation": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-template-overhead": (CheckMode.SOUND, CheckMode.BOUNDED),
    "rag-tokenizer-mismatch": (CheckMode.BOUNDED, CheckMode.HEURISTIC),
    "static-contract-abstained": (CheckMode.ABSTAINING, CheckMode.BOUNDED, CheckMode.Z3_BACKED_SMT),
    "static-contract-proved": (CheckMode.SOUND, CheckMode.BOUNDED, CheckMode.Z3_BACKED_SMT),
    "static-contract-unknown": (CheckMode.ABSTAINING, CheckMode.BOUNDED, CheckMode.Z3_BACKED_SMT),
    "static-contract-violation": (CheckMode.SOUND, CheckMode.BOUNDED, CheckMode.Z3_BACKED_SMT),
    "diagnostic-suppressed": (CheckMode.SOUND, CheckMode.COMPLETE),
    "policy-suppression-invalid": (CheckMode.SOUND, CheckMode.COMPLETE),
    "policy-threshold-violation": (CheckMode.SOUND, CheckMode.COMPLETE),
    "check-unknown": (CheckMode.SOUND, CheckMode.COMPLETE),
    "check-failed": (CheckMode.HEURISTIC,),
}


def _artifact_cache_key(loaded_artifacts: Sequence[LoadedArtifact]) -> tuple[tuple[str, str, str | None, str | None, str | None], ...]:
    return tuple(
        (
            loaded.artifact.kind.value,
            loaded.artifact.name,
            loaded.artifact.location.ref_path,
            loaded.actual_sha256 or loaded.manifest_sha256,
            loaded.artifact.provenance.ref_version,
        )
        for loaded in loaded_artifacts
    )


class AnalysisCache:
    """Thread-safe per-session cache for expensive reusable analysis products."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._values: dict[tuple[str, object], object] = {}

    def memoize(self, namespace: str, key: object, factory: Callable[[], object]) -> object:
        cache_key = (namespace, key)
        with self._lock:
            if cache_key in self._values:
                return self._values[cache_key]
            value = factory()
            return self._values.setdefault(cache_key, value)

    def tokenizer(self, artifact: TokenizerArtifact) -> TokenizerAdapter:
        key = (
            artifact.name,
            artifact.location.ref_path,
            artifact.provenance.ref_version,
            artifact.family,
            artifact.added_tokens,
            artifact.metadata,
        )
        return self.memoize("tokenizer-adapter", key, lambda: load_tokenizer(artifact))  # type: ignore[return-value]

    def token_budget_report(
        self,
        config: VerificationConfig,
        loaded_artifacts: tuple[LoadedArtifact, ...],
        tokenizers: tuple[tuple[TokenizerArtifact, TokenizerAdapter], ...],
    ) -> TokenBudgetReport:
        key = (
            config.name,
            config.max_context_tokens,
            _artifact_cache_key(loaded_artifacts),
            tuple((artifact.name, tokenizer.backend.value) for artifact, tokenizer in tokenizers),
        )
        return self.memoize(
            "token-budget-report",
            key,
            lambda: analyze_token_budget(config, loaded_artifacts, tokenizers=tokenizers),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Inputs available to public and built-in verification checks."""

    config: VerificationConfig
    loaded_artifacts: tuple[LoadedArtifact, ...]
    cache: AnalysisCache = field(default_factory=AnalysisCache, compare=False)

    def artifact(self, name: str) -> LoadedArtifact:
        for loaded in self.loaded_artifacts:
            if loaded.artifact.name == name:
                return loaded
        raise KeyError(name)


CheckCallable = Callable[[CheckContext], Iterable[Diagnostic]]


@dataclass(frozen=True, slots=True)
class ScheduledDiagnostic:
    check_ordinal: int
    emission_index: int
    diagnostic: Diagnostic


@dataclass(frozen=True, slots=True)
class CheckDependency:
    artifact_kinds: tuple[ArtifactKind, ...] = ()
    after: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScheduledCheck:
    name: str
    callable: CheckCallable
    dependency: CheckDependency
    modes: tuple[CheckMode, ...]
    ordinal: int


CHECK_DEPENDENCIES: dict[str, CheckDependency] = {
    "repository-skeleton": CheckDependency(artifact_kinds=tuple(ArtifactKind), resources=("repository-summary",)),
    "artifact-provenance": CheckDependency(artifact_kinds=tuple(ArtifactKind)),
    "enterprise-readiness": CheckDependency(resources=("enterprise-config",)),
    "role-boundary-nonforgeability": CheckDependency(artifact_kinds=(ArtifactKind.CHAT_TEMPLATE,)),
    "stop-differential": CheckDependency(
        artifact_kinds=(ArtifactKind.STOP_POLICY, ArtifactKind.PROVIDER_CONFIG),
    ),
    "stop-overreachability": CheckDependency(
        artifact_kinds=(
            ArtifactKind.STOP_POLICY,
            ArtifactKind.SCHEMA,
            ArtifactKind.TOOL_DEFINITION,
            ArtifactKind.PROVIDER_CONFIG,
            ArtifactKind.GRAMMAR,
        ),
    ),
    "stop-tokenizer-analysis": CheckDependency(
        artifact_kinds=(ArtifactKind.STOP_POLICY, ArtifactKind.TOKENIZER),
        resources=("tokenizer-adapter",),
    ),
    "grammar-differential": CheckDependency(artifact_kinds=(ArtifactKind.GRAMMAR,)),
    "grammar-tokenizer-ambiguity": CheckDependency(
        artifact_kinds=(ArtifactKind.TOKENIZER, ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR),
        resources=("tokenizer-adapter", "automata-product"),
    ),
    "grammar-tokenizer-emptiness": CheckDependency(
        artifact_kinds=(ArtifactKind.TOKENIZER, ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR),
        resources=("tokenizer-adapter", "automata-product"),
    ),
    "parser-compatibility": CheckDependency(artifact_kinds=(ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR)),
    "provider-fixture-replay": CheckDependency(artifact_kinds=(ArtifactKind.PROVIDER_CONFIG, ArtifactKind.TOOL_DEFINITION)),
    "provider-migration": CheckDependency(artifact_kinds=(ArtifactKind.PROVIDER_CONFIG, ArtifactKind.TOOL_DEFINITION)),
    "rag-chunking-compatibility": CheckDependency(
        artifact_kinds=(
            ArtifactKind.PROMPT_SEGMENT,
            ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
            ArtifactKind.TOKENIZER,
        ),
        after=("token-budget-model",),
        resources=("token-budget-report", "tokenizer-adapter"),
    ),
    "static-contracts": CheckDependency(artifact_kinds=tuple(ArtifactKind), resources=("z3-query",)),
    "token-budget-model": CheckDependency(
        artifact_kinds=(
            ArtifactKind.PROMPT_SEGMENT,
            ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
            ArtifactKind.TOKENIZER,
        ),
        resources=("token-budget-report", "tokenizer-adapter"),
    ),
    "tool-schema-ingestion": CheckDependency(artifact_kinds=(ArtifactKind.TOOL_DEFINITION,)),
    "tool-serialization": CheckDependency(
        artifact_kinds=(
            ArtifactKind.TOOL_DEFINITION,
            ArtifactKind.PROVIDER_CONFIG,
            ArtifactKind.CHAT_TEMPLATE,
            ArtifactKind.STOP_POLICY,
        ),
    ),
    "tokenizer-config-drift": CheckDependency(artifact_kinds=(ArtifactKind.TOKENIZER,)),
    "tokenizer-drift": CheckDependency(artifact_kinds=(ArtifactKind.TOKENIZER,)),
}


class CheckScheduler:
    """Dependency-aware internal scheduler that keeps output deterministic."""

    def __init__(
        self,
        checks: Mapping[str, CheckCallable],
        *,
        dependencies: Mapping[str, CheckDependency] | None = None,
        modes: Mapping[str, tuple[CheckMode, ...]] | None = None,
    ) -> None:
        self._checks = checks
        self._dependencies = dependencies or CHECK_DEPENDENCIES
        self._modes = modes or CHECK_MODE_CATALOG

    def run(
        self,
        context: CheckContext,
        requested_checks: Sequence[str | CheckCallable],
    ) -> tuple[ScheduledDiagnostic, ...]:
        scheduled = self._resolve(requested_checks)
        selected_names = {check.name for check in scheduled}
        completed: set[str] = set()
        pending = list(scheduled)
        results: list[ScheduledDiagnostic] = []
        while pending:
            ready = [
                check
                for check in pending
                if all(dependency in completed or dependency not in selected_names for dependency in check.dependency.after)
            ]
            if not ready:
                ready = [min(pending, key=lambda check: check.ordinal)]
            ready.sort(key=lambda check: check.ordinal)
            batches = _resource_safe_batches(ready)
            for batch in batches:
                results.extend(_run_batch(context, batch))
                completed.update(check.name for check in batch)
                batch_ordinals = {check.ordinal for check in batch}
                pending = [check for check in pending if check.ordinal not in batch_ordinals]
        return tuple(results)

    def _resolve(self, requested_checks: Sequence[str | CheckCallable]) -> tuple[ScheduledCheck, ...]:
        scheduled: list[ScheduledCheck] = []
        for ordinal, check in enumerate(requested_checks):
            if isinstance(check, str):
                check_name = check
                check_callable = self._checks.get(check)
                if check_callable is None:
                    check_callable = _unknown_check_callable(check_name)
            else:
                check_name = getattr(check, "__name__", "embedded-check")
                check_callable = check
            scheduled.append(
                ScheduledCheck(
                    name=check_name,
                    callable=check_callable,
                    dependency=self._dependencies.get(check_name, CheckDependency()),
                    modes=self._modes.get(check_name, ()),
                    ordinal=ordinal,
                )
            )
        return tuple(scheduled)


def _resource_safe_batches(checks: Sequence[ScheduledCheck]) -> tuple[tuple[ScheduledCheck, ...], ...]:
    batches: list[list[ScheduledCheck]] = []
    batch_resources: list[set[str]] = []
    for check in checks:
        resources = set(check.dependency.resources)
        for index, used in enumerate(batch_resources):
            if resources.isdisjoint(used):
                batches[index].append(check)
                used.update(resources)
                break
        else:
            batches.append([check])
            batch_resources.append(set(resources))
    return tuple(tuple(batch) for batch in batches)


def _run_batch(context: CheckContext, batch: Sequence[ScheduledCheck]) -> tuple[ScheduledDiagnostic, ...]:
    if len(batch) == 1:
        return _run_scheduled_check(context, batch[0])
    max_workers = min(len(batch), 32)
    by_ordinal: dict[int, tuple[ScheduledDiagnostic, ...]] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="promptabi-check") as executor:
        futures = {
            executor.submit(_run_scheduled_check, context, check): check.ordinal
            for check in batch
        }
        for future, ordinal in futures.items():
            by_ordinal[ordinal] = future.result()
    return tuple(
        diagnostic
        for ordinal in sorted(by_ordinal)
        for diagnostic in by_ordinal[ordinal]
    )


def _run_scheduled_check(context: CheckContext, check: ScheduledCheck) -> tuple[ScheduledDiagnostic, ...]:
    try:
        diagnostics = tuple(check.callable(context))
    except Exception as exc:
        diagnostics = (_failed_check_diagnostic(check.name, exc),)
    if check.modes:
        diagnostics = tuple(
            diagnostic if diagnostic.check_modes else replace(diagnostic, check_modes=check.modes)
            for diagnostic in diagnostics
        )
    return tuple(
        ScheduledDiagnostic(
            check_ordinal=check.ordinal,
            emission_index=index,
            diagnostic=diagnostic,
        )
        for index, diagnostic in enumerate(diagnostics)
    )


def _unknown_check_callable(check_name: str) -> CheckCallable:
    def run_unknown_check(context: CheckContext) -> tuple[Diagnostic, ...]:
        del context
        return (_unknown_check_diagnostic(check_name),)

    return run_unknown_check


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
        plugin_registry: PluginRegistry | None = None,
    ) -> None:
        self.config = config
        self.plugin_registry = plugin_registry or create_first_party_plugin_registry()
        self.loader = loader or ArtifactLoader(plugin_registry=self.plugin_registry)
        self.check_dependencies: dict[str, CheckDependency] = dict(CHECK_DEPENDENCIES)
        self.check_modes: dict[str, tuple[CheckMode, ...]] = dict(CHECK_MODE_CATALOG)
        self.checks: dict[str, CheckCallable] = {
            "repository-skeleton": self._repository_skeleton_check,
            "artifact-provenance": self._artifact_provenance_check,
            "enterprise-readiness": self._enterprise_readiness_check,
            "role-boundary-nonforgeability": self._role_boundary_nonforgeability_check,
            "stop-differential": self._stop_differential_check,
            "stop-overreachability": self._stop_overreachability_check,
            "stop-tokenizer-analysis": self._stop_tokenizer_analysis_check,
            "grammar-differential": self._grammar_differential_check,
            "grammar-tokenizer-ambiguity": self._grammar_tokenizer_ambiguity_check,
            "grammar-tokenizer-emptiness": self._grammar_tokenizer_emptiness_check,
            "parser-compatibility": self._parser_compatibility_check,
            "provider-fixture-replay": self._provider_fixture_replay_check,
            "provider-migration": self._provider_migration_check,
            "rag-chunking-compatibility": self._rag_chunking_compatibility_check,
            "static-contracts": self._static_contracts_check,
            "token-budget-model": self._token_budget_model_check,
            "tool-schema-ingestion": self._tool_schema_ingestion_check,
            "tool-serialization": self._tool_serialization_check,
            "tokenizer-config-drift": self._tokenizer_config_drift_check,
            "tokenizer-drift": self._tokenizer_config_drift_check,
        }
        for registration in self.plugin_registry.checks.values():
            self.checks[registration.name] = registration.callable
            self.check_dependencies[registration.name] = CheckDependency(
                artifact_kinds=registration.artifact_kinds,
                after=registration.after,
                resources=registration.resources,
            )
            if registration.modes:
                self.check_modes[registration.name] = registration.modes
        if checks:
            self.checks.update(checks)

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        checks: Mapping[str, CheckCallable] | None = None,
        loader: ArtifactLoader | None = None,
        plugin_registry: PluginRegistry | None = None,
    ) -> "VerificationSession":
        return cls(load_config(path), checks=checks, loader=loader, plugin_registry=plugin_registry)

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

    def load_artifacts_with_diagnostics(self) -> tuple[tuple[LoadedArtifact, ...], tuple[Diagnostic, ...]]:
        """Load configured artifacts and return deterministic non-fatal diagnostics."""

        loaded_artifacts, diagnostics = self._load_artifacts_with_diagnostics()
        return loaded_artifacts, tuple(diagnostics)

    def collect_diagnostics(self, *, checks: Sequence[str | CheckCallable] | None = None) -> tuple[Diagnostic, ...]:
        """Run preflight loading plus selected checks and return sorted diagnostics."""

        loaded_artifacts, diagnostics_tuple = self.load_artifacts_with_diagnostics()
        context = CheckContext(config=self.config, loaded_artifacts=loaded_artifacts)
        scheduled_diagnostics = [
            ScheduledDiagnostic(-1, index, diagnostic)
            for index, diagnostic in enumerate(diagnostics_tuple)
        ]
        scheduled_diagnostics.extend(self._check_diagnostics(context, checks or self.config.checks))
        scheduled_diagnostics.sort(key=_scheduled_diagnostic_sort_key)
        return tuple(item.diagnostic for item in scheduled_diagnostics)

    def run(self, *, checks: Sequence[str | CheckCallable] | None = None) -> VerificationResult:
        diagnostics = self.collect_diagnostics(checks=checks)
        diagnostics = apply_policy_diagnostics(diagnostics, self.config.policy)
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

    def _artifact_provenance_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        for loaded in context.loaded_artifacts:
            artifact = loaded.artifact
            provenance = artifact.provenance
            metadata = dict(artifact.metadata)
            if loaded.resolved and loaded.actual_sha256 is not None and provenance.sha256 is None:
                diagnostics.append(
                    _artifact_provenance_diagnostic(
                        loaded,
                        "artifact-provenance-missing-hash",
                        f"artifact '{artifact.name}' lacks a sha256 pin for its resolved bytes",
                        "sha256",
                        "missing",
                        "Add the computed sha256 to the artifact provenance after reviewing the bytes.",
                        DiagnosticSeverity.WARNING,
                    )
                )
            if provenance.license is None:
                diagnostics.append(
                    _artifact_provenance_diagnostic(
                        loaded,
                        "artifact-provenance-missing-license",
                        f"artifact '{artifact.name}' lacks license metadata",
                        "license",
                        "missing",
                        "Record the artifact license or dataset/model-card license in the PromptABI config.",
                        DiagnosticSeverity.WARNING,
                    )
                )
            if provenance.source is None:
                diagnostics.append(
                    _artifact_provenance_diagnostic(
                        loaded,
                        "artifact-provenance-missing-source",
                        f"artifact '{artifact.name}' lacks an upstream source annotation",
                        "source",
                        "missing",
                        "Record the upstream model repo, fixture origin, or internal source-of-truth URI.",
                        DiagnosticSeverity.WARNING,
                    )
                )
            elif not _metadata_bool(metadata, "trusted_source"):
                diagnostics.append(
                    _artifact_provenance_diagnostic(
                        loaded,
                        "artifact-provenance-untrusted-source",
                        f"artifact '{artifact.name}' source is not explicitly trusted",
                        "trusted_source",
                        "missing",
                        "Set metadata.trusted_source to true only for reviewed internal mirrors or approved upstreams.",
                        DiagnosticSeverity.WARNING,
                    )
                )
            remote_reason = _nonreproducible_remote_reason(loaded)
            if remote_reason is not None:
                diagnostics.append(
                    _artifact_provenance_diagnostic(
                        loaded,
                        "artifact-provenance-nonreproducible-remote",
                        f"remote artifact '{artifact.name}' is not reproducibly downloadable",
                        "remote",
                        remote_reason,
                        "Use hf://...?...revision=<40-hex-commit> or a local mirrored artifact with sha256 provenance.",
                        DiagnosticSeverity.ERROR,
                    )
                )
        if not diagnostics and context.loaded_artifacts:
            diagnostics.append(
                Diagnostic(
                    rule_id="artifact-provenance-verified",
                    severity=DiagnosticSeverity.INFO,
                    message=f"{len(context.loaded_artifacts)} artifact provenance record(s) are pinned and trusted",
                    check_modes=CHECK_MODE_CATALOG["artifact-provenance-verified"],
                    witness=WitnessTrace(
                        summary="PromptABI verified reproducibility metadata for all loaded artifacts.",
                        steps=(
                            WitnessStep(action="load artifacts", output=f"{len(context.loaded_artifacts)} loaded"),
                            WitnessStep(action="verify hashes", output="all resolved artifacts pinned by sha256"),
                            WitnessStep(action="verify license metadata", output="present"),
                            WitnessStep(action="verify trusted-source annotations", output="present"),
                            WitnessStep(action="classify remote downloads", output="reproducible or mirrored"),
                        ),
                        artifacts=tuple(loaded.artifact.to_ref() for loaded in context.loaded_artifacts),
                    ),
                )
            )
        return tuple(diagnostics)

    def _enterprise_readiness_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        artifact_locations = tuple(
            location
            for loaded in context.loaded_artifacts
            if (location := loaded.artifact.location.ref_path) is not None
        )
        artifact_locations = (
            *artifact_locations,
            *(
                location
                for artifact in context.config.artifact_bundle
                if (location := artifact.location.ref_path) is not None
            ),
        )
        return enterprise_readiness_diagnostics(
            context.config.enterprise,
            artifact_locations=tuple(sorted(set(artifact_locations))),
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

    def _stop_tokenizer_analysis_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        tokenizers = [loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.TOKENIZER]
        stop_policies = [
            loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.STOP_POLICY
        ]
        for stop_loaded in stop_policies:
            stop_artifact = stop_loaded.artifact
            for tokenizer_loaded in tokenizers:
                tokenizer_artifact = tokenizer_loaded.artifact
                try:
                    tokenizer = context.cache.tokenizer(tokenizer_artifact)
                    report = analyze_stop_policy_tokenizer(stop_artifact, tokenizer)
                except TokenizerError as exc:
                    diagnostics.append(_stop_tokenizer_abstained_diagnostic(stop_loaded, tokenizer_loaded, exc))
                    continue
                except Exception as exc:
                    diagnostics.append(_stop_tokenizer_abstained_diagnostic(stop_loaded, tokenizer_loaded, exc))
                    continue
                diagnostics.extend(_stop_tokenizer_report_diagnostics(stop_loaded, tokenizer_loaded, report))
        return tuple(diagnostics)

    def _stop_overreachability_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        stop_policies = [
            loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.STOP_POLICY
        ]
        structured_artifacts = [
            loaded.artifact
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind
            in {
                ArtifactKind.SCHEMA,
                ArtifactKind.TOOL_DEFINITION,
                ArtifactKind.PROVIDER_CONFIG,
                ArtifactKind.GRAMMAR,
            }
        ]
        for stop_loaded in stop_policies:
            report = analyze_stop_overreachability(stop_loaded.artifact, structured_artifacts)
            diagnostics.extend(
                _stop_overreachability_finding_diagnostic(stop_loaded, report, finding)
                for finding in report.findings
            )
            diagnostics.extend(
                _stop_overreachability_abstention_diagnostic(stop_loaded, report, abstention)
                for abstention in report.abstentions
            )
        return tuple(diagnostics)

    def _stop_differential_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        stop_policies = [
            loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.STOP_POLICY
        ]
        provider_configs = [
            loaded.artifact
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind is ArtifactKind.PROVIDER_CONFIG
        ]
        for stop_loaded in stop_policies:
            report = analyze_stop_differential(stop_loaded.artifact, provider_configs)
            diagnostics.extend(
                _stop_differential_mismatch_diagnostic(stop_loaded, report, mismatch)
                for mismatch in report.mismatches
            )
            diagnostics.extend(
                _stop_differential_abstention_diagnostic(stop_loaded, report, abstention)
                for abstention in report.abstentions
            )
            if report.matches:
                diagnostics.append(_stop_differential_agreement_diagnostic(stop_loaded, report))
        return tuple(diagnostics)

    def _grammar_tokenizer_emptiness_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        tokenizers = [loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.TOKENIZER]
        grammars = [
            loaded
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind in {ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR}
        ]
        for tokenizer_loaded in tokenizers:
            tokenizer_artifact = tokenizer_loaded.artifact
            try:
                tokenizer = context.cache.tokenizer(tokenizer_artifact)
            except TokenizerError as exc:
                diagnostics.extend(
                    _grammar_tokenizer_abstained_diagnostic(tokenizer_loaded, grammar_loaded, str(exc))
                    for grammar_loaded in grammars
                )
                continue
            for grammar_loaded in grammars:
                report = context.cache.memoize(
                    "grammar-tokenizer-emptiness",
                    (tokenizer_loaded.artifact.name, grammar_loaded.artifact.name, _artifact_cache_key((tokenizer_loaded, grammar_loaded))),
                    lambda tokenizer_artifact=tokenizer_artifact, grammar_loaded=grammar_loaded, tokenizer=tokenizer: analyze_tokenizer_grammar_emptiness(
                        tokenizer_artifact,
                        grammar_loaded.artifact,
                        tokenizer,
                    ),
                )
                diagnostics.append(_grammar_tokenizer_report_diagnostic(tokenizer_loaded, grammar_loaded, report))
        return tuple(diagnostics)

    def _grammar_tokenizer_ambiguity_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        tokenizers = [loaded for loaded in context.loaded_artifacts if loaded.artifact.kind is ArtifactKind.TOKENIZER]
        grammars = [
            loaded
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind in {ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR}
        ]
        for tokenizer_loaded in tokenizers:
            tokenizer_artifact = tokenizer_loaded.artifact
            try:
                tokenizer = context.cache.tokenizer(tokenizer_artifact)
            except TokenizerError as exc:
                diagnostics.extend(
                    _grammar_tokenizer_ambiguity_abstained_diagnostic(tokenizer_loaded, grammar_loaded, str(exc))
                    for grammar_loaded in grammars
                )
                continue
            for grammar_loaded in grammars:
                report = context.cache.memoize(
                    "grammar-tokenizer-ambiguity",
                    (tokenizer_loaded.artifact.name, grammar_loaded.artifact.name, _artifact_cache_key((tokenizer_loaded, grammar_loaded))),
                    lambda tokenizer_artifact=tokenizer_artifact, grammar_loaded=grammar_loaded, tokenizer=tokenizer: analyze_tokenizer_grammar_ambiguity(
                        tokenizer_artifact,
                        grammar_loaded.artifact,
                        tokenizer,
                    ),
                )
                diagnostics.extend(
                    _grammar_tokenizer_ambiguity_report_diagnostics(tokenizer_loaded, grammar_loaded, report)
                )
        return tuple(diagnostics)

    def _grammar_differential_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        fixture_artifacts = [
            loaded
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind is ArtifactKind.GRAMMAR and loaded.source_type == "grammar-differential"
        ]
        for loaded in fixture_artifacts:
            path = loaded.artifact.location.path
            if path is None:
                continue
            report = analyze_grammar_differential_corpus(path)
            diagnostics.extend(
                _grammar_differential_mismatch_diagnostic(loaded, case)
                for case in report.mismatches
            )
            diagnostics.extend(
                _grammar_differential_abstained_diagnostic(loaded, case)
                for case in report.abstentions
            )
            if report.agreements and not report.mismatches and not report.abstentions:
                diagnostics.append(_grammar_differential_agreement_diagnostic(loaded, report.agreements))
        return tuple(diagnostics)

    def _parser_compatibility_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        structured_artifacts = [
            loaded
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind in {ArtifactKind.SCHEMA, ArtifactKind.GRAMMAR}
            and loaded.source_type != "grammar-differential"
        ]
        for loaded in structured_artifacts:
            report = analyze_parser_compatibility(loaded.artifact)
            if report.status is ParserCompatibilityStatus.MISMATCH:
                diagnostics.extend(
                    _parser_compatibility_mismatch_diagnostic(loaded, report, mismatch)
                    for mismatch in report.mismatches
                )
            elif report.status is ParserCompatibilityStatus.ABSTAINED:
                diagnostics.append(_parser_compatibility_abstained_diagnostic(loaded, report))
            else:
                diagnostics.append(_parser_compatibility_agreement_diagnostic(loaded, report))
        return tuple(diagnostics)

    def _tool_schema_ingestion_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        return tuple(
            _tool_schema_ingestion_diagnostic(loaded)
            for loaded in context.loaded_artifacts
            if loaded.artifact.kind is ArtifactKind.TOOL_DEFINITION
            and loaded.source_type == "tool-definition-schema"
        )

    def _tool_serialization_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = analyze_tool_call_serialization(context.loaded_artifacts)
        return tuple(_tool_serialization_diagnostic(finding) for finding in report.findings)

    def _tokenizer_config_drift_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = analyze_tokenizer_config_drift(context.loaded_artifacts)
        diagnostics = [_tokenizer_drift_finding_diagnostic(finding) for finding in report.findings]
        diagnostics.extend(_tokenizer_drift_abstention_diagnostic(abstention) for abstention in report.abstentions)
        if report.compared and not report.findings and not report.abstentions:
            diagnostics.append(_tokenizer_drift_clean_diagnostic(report.compared))
        return tuple(diagnostics)

    def _provider_fixture_replay_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = analyze_provider_fixture_replay(context.loaded_artifacts)
        diagnostics = [_provider_fixture_replay_finding_diagnostic(finding) for finding in report.findings]
        if report.cases:
            diagnostics.extend(_provider_fixture_replay_case_diagnostic(case, report.replay_hash) for case in report.cases)
        return tuple(diagnostics)

    def _provider_migration_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = analyze_provider_migration(context.loaded_artifacts)
        return tuple(_provider_migration_diagnostic(finding) for finding in report.findings)

    def _token_budget_model_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = self._analyze_token_budget_context(context)
        diagnostics = [
            _token_budget_finding_diagnostic(report, finding)
            for finding in report.findings
            if not finding.rule_id.startswith("rag-")
        ]
        if report.reservation is not None:
            diagnostics.append(_token_budget_summary_diagnostic(report))
        return tuple(diagnostics)

    def _rag_chunking_compatibility_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = self._analyze_token_budget_context(context)
        return tuple(
            _token_budget_finding_diagnostic(report, finding)
            for finding in report.findings
            if finding.rule_id.startswith("rag-")
        )

    def _analyze_token_budget_context(self, context: CheckContext) -> TokenBudgetReport:
        tokenizers = []
        for loaded in context.loaded_artifacts:
            if loaded.artifact.kind is not ArtifactKind.TOKENIZER or not isinstance(loaded.artifact, TokenizerArtifact):
                continue
            try:
                tokenizers.append((loaded.artifact, context.cache.tokenizer(loaded.artifact)))
            except TokenizerError:
                continue
        return context.cache.token_budget_report(context.config, context.loaded_artifacts, tuple(tokenizers))

    def _static_contracts_check(self, context: CheckContext) -> tuple[Diagnostic, ...]:
        report = context.cache.memoize(
            "static-contract-z3-queries",
            (context.config.name, context.config.max_context_tokens, _artifact_cache_key(context.loaded_artifacts)),
            lambda: analyze_static_contracts(context.config, context.loaded_artifacts),
        )
        return tuple(_static_contract_finding_diagnostic(report, finding) for finding in report.findings)

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
    ) -> tuple[ScheduledDiagnostic, ...]:
        return CheckScheduler(
            self.checks,
            dependencies=self.check_dependencies,
            modes=self.check_modes,
        ).run(context, checks)

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


def _artifact_provenance_diagnostic(
    loaded: LoadedArtifact,
    rule_id: str,
    message: str,
    field: str,
    output: str,
    suggestion: str,
    severity: DiagnosticSeverity,
) -> Diagnostic:
    artifact = loaded.artifact.to_ref()
    provenance = loaded.artifact.provenance
    steps = [
        WitnessStep(action="classify artifact source", input=loaded.source_type, output="resolved" if loaded.resolved else "metadata-only"),
        WitnessStep(action=f"inspect {field}", input=loaded.artifact.location.ref_path, output=output),
    ]
    if loaded.actual_sha256 is not None:
        steps.append(WitnessStep(action="hash resolved artifact", output=f"sha256={loaded.actual_sha256}"))
    if loaded.manifest_sha256 is not None:
        steps.append(WitnessStep(action="hash artifact manifest", output=f"sha256={loaded.manifest_sha256}"))
    if provenance.source is not None:
        steps.append(WitnessStep(action="read source annotation", output=provenance.source))
    if provenance.license is not None:
        steps.append(WitnessStep(action="read license metadata", output=provenance.license))
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=message,
        artifact=artifact,
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG[rule_id],
        suggestions=(suggestion,),
        witness=WitnessTrace(
            summary="PromptABI audited artifact provenance and supply-chain reproducibility metadata.",
            steps=tuple(steps),
            artifacts=(artifact,),
        ),
        properties=(
            ("field", field),
            ("source_type", loaded.source_type),
            ("resolved", loaded.resolved),
        ),
    )


def _metadata_bool(metadata: dict[str, object], key: str) -> bool:
    value = metadata.get(key)
    return isinstance(value, bool) and value


def _nonreproducible_remote_reason(loaded: LoadedArtifact) -> str | None:
    uri = loaded.artifact.location.uri
    if uri is None:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "memory":
        return None
    if parsed.scheme != "hf":
        return f"unsupported remote scheme {parsed.scheme!r}"
    revision = parse_qs(parsed.query).get("revision", [None])[0] or loaded.artifact.provenance.revision
    if revision is None:
        return "missing immutable Hugging Face revision"
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        return f"movable Hugging Face revision {revision!r}"
    return None


def _scheduled_diagnostic_sort_key(item: ScheduledDiagnostic) -> tuple[object, ...]:
    return (
        diagnostic_sort_key(item.diagnostic),
        item.check_ordinal,
        item.emission_index,
        item.diagnostic.fingerprint,
    )


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


def _stop_tokenizer_abstained_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    exc: Exception,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-tokenizer-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"stop policy '{stop_loaded.artifact.name}' could not be analyzed with tokenizer "
            f"'{tokenizer_loaded.artifact.name}'"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-abstained"],
        suggestions=("Use a local tokenizer artifact supported by PromptABI's tokenizer adapters.",),
        witness=WitnessTrace(
            summary="PromptABI could not construct the concrete tokenizer analysis.",
            steps=(
                WitnessStep(action="load tokenizer", input=tokenizer_loaded.artifact.name, output=str(exc)),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _stop_tokenizer_report_diagnostics(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(
        _stop_unreachable_diagnostic(stop_loaded, tokenizer_loaded, report, token_id)
        for token_id in report.unreachable_token_ids
    )
    diagnostics.extend(
        _stop_collision_diagnostic(stop_loaded, tokenizer_loaded, report, collision)
        for collision in (*report.collisions, *report.normalization_collisions)
    )
    diagnostics.extend(
        _stop_ambiguous_diagnostic(stop_loaded, tokenizer_loaded, report, sequence)
        for sequence in report.lossy_or_normalizing_sequences
    )
    diagnostics.extend(
        _stop_special_interaction_diagnostic(stop_loaded, tokenizer_loaded, report, sequence)
        for sequence in report.special_interactions
    )
    if report.sequences:
        diagnostics.append(_stop_alignment_diagnostic(stop_loaded, tokenizer_loaded, report))
    return tuple(diagnostics)


def _stop_unreachable_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
    token_id: StopTokenIdAnalysis,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-tokenizer-unreachable",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"stop token id {token_id.token_id} from policy '{stop_loaded.artifact.name}' "
            f"is not decodable by tokenizer '{tokenizer_loaded.artifact.name}'"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-unreachable"],
        suggestions=("Remove the token id or verify it belongs to the selected tokenizer revision.",),
        witness=WitnessTrace(
            summary="A configured token-id stop cannot be represented by the selected tokenizer adapter.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action="decode configured stop token id", input=str(token_id.token_id), output=token_id.error),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _stop_collision_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
    collision: StopCollision,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-tokenizer-collision",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"stop sequence {collision.shorter!r} is a {collision.level} {collision.relation} "
            f"collision with {collision.longer!r}"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-collision"],
        suggestions=("Prefer non-overlapping stop strings, or make the intended precedence explicit in tests.",),
        witness=WitnessTrace(
            summary="Two configured stop strings overlap under string, byte, token, or normalized surfaces.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action=f"compare {collision.level} surfaces", input=collision.shorter, output=collision.witness),
                WitnessStep(action="classify collision", input=collision.longer, output=collision.relation),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _stop_ambiguous_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
    sequence: StopSequenceAnalysis,
) -> Diagnostic:
    reason = (
        f"normalizes to {sequence.normalized_sequence!r}"
        if sequence.normalization_changed
        else f"decodes as {sequence.decoded_text!r}"
    )
    return Diagnostic(
        rule_id="stop-tokenizer-ambiguous",
        severity=DiagnosticSeverity.WARNING,
        message=f"stop sequence {sequence.stop_sequence!r} is tokenizer-sensitive: {reason}",
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-ambiguous"],
        suggestions=("Verify whether the provider matches stops before or after tokenizer normalization/decoding.",),
        witness=WitnessTrace(
            summary="The stop string's configured surface differs from a tokenizer-derived surface.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action="encode stop string", input=sequence.stop_sequence, output=sequence.token_summary()),
                WitnessStep(action="decode stop token ids", input=str(sequence.token_ids), output=sequence.decoded_text),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _stop_special_interaction_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
    sequence: StopSequenceAnalysis,
) -> Diagnostic:
    details = []
    if sequence.special_token_ids:
        details.append(f"special ids={sequence.special_token_ids}")
    if sequence.added_token_ids:
        details.append(f"added ids={sequence.added_token_ids}")
    return Diagnostic(
        rule_id="stop-tokenizer-special-interaction",
        severity=DiagnosticSeverity.WARNING,
        message=f"stop sequence {sequence.stop_sequence!r} intersects tokenizer special/added tokens",
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-special-interaction"],
        suggestions=("Confirm whether the runtime stop matcher treats added and special tokens as text or token ids.",),
        witness=WitnessTrace(
            summary="A configured stop string tokenizes through tokenizer control-token machinery.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action="encode stop string", input=sequence.stop_sequence, output=sequence.token_summary()),
                WitnessStep(action="classify token flags", output=", ".join(details)),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _stop_alignment_diagnostic(
    stop_loaded: LoadedArtifact,
    tokenizer_loaded: LoadedArtifact,
    report: StopPolicyTokenizerAnalysisReport,
) -> Diagnostic:
    alignment = "; ".join(
        f"{sequence.stop_sequence!r}: bytes={sequence.utf8_bytes}, ids={sequence.token_ids}"
        for sequence in report.sequences
    )
    return Diagnostic(
        rule_id="stop-tokenizer-alignment",
        severity=DiagnosticSeverity.INFO,
        message=f"stop policy '{stop_loaded.artifact.name}' has tokenizer alignment metadata",
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-tokenizer-alignment"],
        witness=WitnessTrace(
            summary="PromptABI encoded configured stop strings with the selected tokenizer.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action="encode stop strings", output=alignment),
            ),
            artifacts=(stop_loaded.artifact.to_ref(), tokenizer_loaded.artifact.to_ref()),
        ),
    )


def _grammar_differential_mismatch_diagnostic(
    loaded: LoadedArtifact,
    case: GrammarDifferentialCaseReport,
) -> Diagnostic:
    mismatches = case.mismatches
    sample_summary = "; ".join(
        f"{item.sample.text!r}: expected={item.sample.expected_accepts}, promptabi={item.promptabi_accepts}"
        for item in mismatches
    )
    return Diagnostic(
        rule_id="grammar-differential-mismatch",
        severity=DiagnosticSeverity.ERROR,
        message=(
            f"grammar differential case '{case.case_id}' disagrees with recorded "
            f"{case.backend_family} semantics"
        ),
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-differential-mismatch"],
        suggestions=("Inspect the reduced backend fixture and either fix the grammar model or mark unsupported semantics as an abstention.",),
        witness=WitnessTrace(
            summary="PromptABI replayed hand-labeled backend membership samples and found a disagreement.",
            steps=(
                WitnessStep(action="select backend fixture", input=case.backend_family, output=case.case_id),
                WitnessStep(action="ingest grammar", input=case.declared_type, output=", ".join(case.features)),
                WitnessStep(action="compare membership labels", output=sample_summary),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _grammar_differential_abstained_diagnostic(
    loaded: LoadedArtifact,
    case: GrammarDifferentialCaseReport,
) -> Diagnostic:
    return Diagnostic(
        rule_id="grammar-differential-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=f"grammar differential case '{case.case_id}' is outside local replay semantics",
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-differential-abstained"],
        suggestions=("Reduce the fixture to JSON Schema, regex, choices, or finite literal grammar semantics.",),
        witness=WitnessTrace(
            summary="PromptABI abstained rather than overclaim backend equivalence.",
            steps=(
                WitnessStep(action="select backend fixture", input=case.backend_family, output=case.case_id),
                WitnessStep(action="evaluate supported fragment", input=case.declared_type, output=case.reason),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _grammar_differential_agreement_diagnostic(
    loaded: LoadedArtifact,
    agreements: tuple[GrammarDifferentialCaseReport, ...],
) -> Diagnostic:
    families = ", ".join(sorted({case.backend_family for case in agreements}))
    return Diagnostic(
        rule_id="grammar-differential-agreement",
        severity=DiagnosticSeverity.INFO,
        message=f"grammar differential corpus agrees on {len(agreements)} recorded backend cases",
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-differential-agreement"],
        witness=WitnessTrace(
            summary="PromptABI replayed recorded backend grammar-semantics labels without disagreement.",
            steps=(
                WitnessStep(action="load fixture families", output=families),
                WitnessStep(action="compare accepted and rejected samples", output=f"{len(agreements)} cases"),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _parser_compatibility_mismatch_diagnostic(
    loaded: LoadedArtifact,
    report: ParserCompatibilityReport,
    mismatch: ParserCompatibilityObservation,
) -> Diagnostic:
    direction = mismatch.direction or ParserCompatibilityDirection.PARSER_BROADER
    if direction is ParserCompatibilityDirection.PARSER_BROADER:
        suggestion = "Tighten the application parser or make the grammar/schema accept exactly the parser-admitted envelope."
    else:
        suggestion = "Tighten the grammar or update the application parser so generated outputs are parseable at runtime."
    return Diagnostic(
        rule_id="parser-compatibility-mismatch",
        severity=DiagnosticSeverity.ERROR,
        message=(
            f"structured-output artifact '{loaded.artifact.name}' has {direction.value} "
            f"parser compatibility for {report.parser_format}"
        ),
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["parser-compatibility-mismatch"],
        suggestions=(
            suggestion,
            "Add representative parser_compatibility samples for both accepted and rejected runtime strings.",
        ),
        witness=WitnessTrace(
            summary="PromptABI replayed a concrete structured-output string through both grammar membership and the declared parser model.",
            steps=(
                WitnessStep(action="select parser model", input=report.parser_format, output=", ".join(report.assumptions)),
                WitnessStep(action="select bounded sample", input=mismatch.sample.source, output=mismatch.sample.text),
                WitnessStep(action="evaluate grammar membership", input=report.grammar_kind, output=str(mismatch.grammar_accepts)),
                WitnessStep(action="evaluate parser acceptance", input=report.parser_format, output=str(mismatch.parser_accepts)),
                WitnessStep(action="classify disagreement", output=direction.value),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _parser_compatibility_abstained_diagnostic(
    loaded: LoadedArtifact,
    report: ParserCompatibilityReport,
) -> Diagnostic:
    return Diagnostic(
        rule_id="parser-compatibility-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=f"structured-output artifact '{loaded.artifact.name}' is outside parser compatibility replay",
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["parser-compatibility-abstained"],
        suggestions=(
            "Declare metadata.parser_format for non-schema artifacts and provide parser_compatibility samples.",
            "Use JSON Schema, regex, finite literal grammar, markdown-fence, XML tool-call, or custom-delimited parser fixtures.",
        ),
        witness=WitnessTrace(
            summary="PromptABI abstained instead of guessing application-parser equivalence.",
            steps=(
                WitnessStep(action="select structured artifact", input=loaded.artifact.name, output=report.grammar_kind),
                WitnessStep(action="select parser model", input=report.parser_format, output=report.reason),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _parser_compatibility_agreement_diagnostic(
    loaded: LoadedArtifact,
    report: ParserCompatibilityReport,
) -> Diagnostic:
    return Diagnostic(
        rule_id="parser-compatibility-agreement",
        severity=DiagnosticSeverity.INFO,
        message=(
            f"structured-output artifact '{loaded.artifact.name}' agrees with declared "
            f"{report.parser_format} parser on {len(report.observations)} bounded sample(s)"
        ),
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["parser-compatibility-agreement"],
        witness=WitnessTrace(
            summary="Concrete grammar witnesses and declared parser samples agreed under heuristic replay.",
            steps=(
                WitnessStep(action="select parser model", input=report.parser_format, output=", ".join(report.assumptions)),
                WitnessStep(action="compare bounded samples", output=f"{len(report.observations)} agreement(s)"),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _tool_schema_ingestion_diagnostic(loaded: LoadedArtifact) -> Diagnostic:
    metadata = dict(loaded.metadata)
    provider = str(metadata.get("provider_family", "unknown"))
    tool_count = int(metadata.get("tool_count", 0))
    tool_names = tuple(metadata.get("tool_names", ()))
    closed = tuple(metadata.get("closed_tool_names", ()))
    encodings = tuple(metadata.get("argument_encodings", ()))
    issue_count = int(metadata.get("issue_count", 0))
    return Diagnostic(
        rule_id="tool-schema-ingestion",
        severity=DiagnosticSeverity.INFO,
        message=(
            f"tool-definition artifact '{loaded.artifact.name}' ingested {tool_count} "
            f"{provider} tool schema(s)"
        ),
        artifact=loaded.artifact.to_ref(),
        span=_artifact_span(loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["tool-schema-ingestion"],
        witness=WitnessTrace(
            summary="PromptABI normalized provider/framework tool definitions into typed function schemas.",
            steps=(
                WitnessStep(action="select tool envelope family", output=provider),
                WitnessStep(action="extract tool names", output=", ".join(tool_names) or "<none>"),
                WitnessStep(action="classify argument encodings", output=", ".join(encodings) or "<none>"),
                WitnessStep(action="identify closed parameter schemas", output=", ".join(closed) or "<none>"),
                WitnessStep(action="record ingestion issues", output=str(issue_count)),
            ),
            artifacts=(loaded.artifact.to_ref(),),
        ),
    )


def _static_contract_finding_diagnostic(report: StaticContractReport, finding: StaticContractFinding) -> Diagnostic:
    del report
    if finding.severity == "error":
        rule_id = "static-contract-violation"
        severity = DiagnosticSeverity.ERROR
        summary = "PromptABI extracted a concrete counterexample for a satisfiable finite static contract."
    elif finding.status.value == "unknown" or finding.result is None:
        rule_id = "static-contract-abstained" if finding.name == "static-contract-abstained" else "static-contract-unknown"
        severity = DiagnosticSeverity.WARNING
        summary = "PromptABI abstained instead of solving a static contract outside the available finite fragment."
    else:
        rule_id = "static-contract-proved"
        severity = DiagnosticSeverity.INFO
        summary = "PromptABI discharged a finite static contract over discrete artifact facts."

    steps: list[WitnessStep] = []
    if finding.problem is not None:
        steps.append(
            WitnessStep(
                action="lower finite contract",
                input=finding.name,
                output=f"{len(finding.problem.variables)} variables, {len(finding.problem.constraints)} constraints",
            )
        )
    if finding.result is not None:
        steps.append(
            WitnessStep(
                action="solve finite contract",
                input=finding.result.backend.value,
                output=finding.result.status.value,
            )
        )
        steps.append(
            WitnessStep(
                action="classify SMT diagnostic",
                input=finding.result.conclusion.value,
                output=_static_contract_outcome(finding),
            )
        )
        if finding.result.assignment:
            assignment = ", ".join(f"{key}={value!r}" for key, value in sorted(finding.result.assignment.items()))
            model_action = "extract Z3 model" if finding.result.backend.value == "z3" else "extract finite model"
            steps.append(WitnessStep(action=model_action, output=assignment))
            steps.append(WitnessStep(action="record concrete counterexample", output=assignment))
        if finding.result.unsat_core:
            core_action = "extract minimized Z3 unsat core" if finding.result.backend.value == "z3" else "extract minimal unsat core"
            steps.append(WitnessStep(action=core_action, output=", ".join(finding.result.unsat_core)))
        if finding.result.reason is not None:
            steps.append(WitnessStep(action="record solver abstention reason", output=finding.result.reason))
    steps.extend(
        WitnessStep(action="record contract evidence", input=key, output=value)
        for key, value in finding.evidence
    )
    if not steps:
        steps.append(WitnessStep(action="inspect artifacts", output="no finite cross-artifact obligation"))

    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=finding.message,
        check_modes=CHECK_MODE_CATALOG[rule_id],
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(summary=summary, steps=tuple(steps)),
    )


def _static_contract_outcome(finding: StaticContractFinding) -> str:
    if finding.result is None or finding.status is SolverStatus.UNKNOWN:
        return "abstention outside the finite modeled fragment"
    if finding.result.sat:
        if finding.severity == "error":
            return "concrete counterexample witness"
        return "proof of incompatibility"
    if finding.result.unsat:
        if finding.severity == "error":
            return "proof of incompatibility"
        return "proof of safety"
    return "abstention outside the finite modeled fragment"


def _token_budget_finding_diagnostic(report: TokenBudgetReport, finding: TokenBudgetFinding) -> Diagnostic:
    severity = (
        DiagnosticSeverity.ERROR
        if finding.severity == "error"
        else DiagnosticSeverity.WARNING
        if finding.severity == "warning"
        else DiagnosticSeverity.INFO
    )
    steps = [
        WitnessStep(action="select context budget", output=report.budget_source or "missing"),
    ]
    if report.reservation is not None:
        steps.extend(
            [
                WitnessStep(
                    action="compute reserved context tokens",
                    input=str(report.reservation.max_context_tokens),
                    output=str(report.reservation.reserved_total),
                ),
                WitnessStep(action="compute modeled input budget", output=str(report.reservation.input_budget_tokens)),
            ]
        )
    steps.extend(
        WitnessStep(action="inspect budget evidence", input=key, output=value)
        for key, value in finding.evidence
    )
    if report.must_survive_proof is not None and finding.rule_id == "token-budget-required-truncated":
        proof = report.must_survive_proof
        steps.extend(
            WitnessStep(action="prove must-survive field", input=key, output=value)
            for key, value in proof.to_metadata()
        )
    return Diagnostic(
        rule_id=finding.rule_id,
        severity=severity,
        message=finding.message,
        check_modes=CHECK_MODE_CATALOG[finding.rule_id],
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(
            summary="PromptABI modeled finite context-window arithmetic and framework truncation decisions.",
            steps=tuple(steps),
        ),
    )


def _token_budget_summary_diagnostic(report: TokenBudgetReport) -> Diagnostic:
    assert report.reservation is not None
    total = report.total_prompt_tokens
    required = report.required_prompt_tokens
    unknown = ", ".join(segment.name for segment in report.unknown_segments) or "<none>"
    known = ", ".join(
        f"{segment.name}={segment.total_tokens} ({segment.source})" for segment in report.known_segments
    ) or "<none>"
    truncation = report.truncation
    kept = ", ".join(segment.name for segment in truncation.kept_segments) if truncation is not None else "<not modeled>"
    dropped = ", ".join(segment.name for segment in truncation.dropped_segments) if truncation is not None else "<not modeled>"
    proof = report.must_survive_proof
    proof_status = proof.status if proof is not None else "not modeled"
    proof_detail = "<none>"
    if proof is not None:
        proof_detail = "; ".join(f"{key}={value}" for key, value in proof.to_metadata())
    visualization = report.visualization
    visualization_text = visualization.render_text() if visualization is not None else "<not modeled>"
    properties = (
        (("token_budget_visualization", visualization.to_dict()),)
        if visualization is not None
        else ()
    )
    return Diagnostic(
        rule_id="token-budget-model",
        severity=DiagnosticSeverity.INFO,
        message=(
            f"context budget '{report.budget_source}' leaves "
            f"{report.reservation.input_budget_tokens} input token(s) after reservations"
        ),
        check_modes=CHECK_MODE_CATALOG["token-budget-model"],
        witness=WitnessTrace(
            summary="PromptABI constructed a named prompt-segment budget model from real artifacts.",
            steps=(
                WitnessStep(action="select framework", input=report.framework or "config", output=report.strategy or "none"),
                WitnessStep(
                    action="reserve output/tool/generation/special tokens",
                    input=str(report.reservation.max_context_tokens),
                    output=str(report.reservation.reserved_total),
                ),
                WitnessStep(action="compute input budget", output=str(report.reservation.input_budget_tokens)),
                WitnessStep(action="sum required segment tokens", output=str(required) if required is not None else "unknown"),
                WitnessStep(action="sum all segment tokens", output=str(total) if total is not None else "unknown"),
                WitnessStep(action="record known segment counts", output=known),
                WitnessStep(action="record unknown segment counts", output=unknown),
                WitnessStep(action="simulate framework truncation", input=report.framework or "config", output=report.strategy or "none"),
                WitnessStep(action="record kept segments", output=kept or "<none>"),
                WitnessStep(action="record dropped segments", output=dropped or "<none>"),
                WitnessStep(action="prove must-survive prompt segments", output=proof_status),
                WitnessStep(action="record must-survive proof", output=proof_detail),
                WitnessStep(action="render token-budget visualization", output=visualization_text),
            ),
        ),
        properties=properties,
    )


def _tool_serialization_diagnostic(finding: ToolSerializationFinding) -> Diagnostic:
    severity = (
        DiagnosticSeverity.ERROR
        if finding.severity == "error"
        else DiagnosticSeverity.WARNING
        if finding.severity == "warning"
        else DiagnosticSeverity.INFO
    )
    context_steps = []
    if finding.provider_name is not None:
        context_steps.append(WitnessStep(action="select provider contract", input=finding.provider_name))
    if finding.tool_artifact_name is not None:
        context_steps.append(WitnessStep(action="select tool schema", input=finding.tool_artifact_name))
    if finding.template_name is not None:
        context_steps.append(WitnessStep(action="select chat template", input=finding.template_name))
    if finding.stop_policy_name is not None:
        context_steps.append(WitnessStep(action="select stop policy", input=finding.stop_policy_name))
    evidence_steps = tuple(
        WitnessStep(action="compare serialization field", input=key, output=value)
        for key, value in finding.evidence
    )
    return Diagnostic(
        rule_id="tool-serialization",
        severity=severity,
        message=f"{finding.kind.value}: {finding.message}",
        span=finding.span,
        check_modes=CHECK_MODE_CATALOG["tool-serialization"],
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(
            summary="A bounded recorded tool-call serialization contract disagreed across selected artifacts.",
            steps=tuple(context_steps) + evidence_steps,
        ),
    )


def _tokenizer_drift_finding_diagnostic(finding: TokenizerDriftFinding) -> Diagnostic:
    severity = DiagnosticSeverity.ERROR if finding.breaking else DiagnosticSeverity.WARNING
    revision_step = ()
    if finding.baseline_revision is not None or finding.current_revision is not None:
        revision_step = (
            WitnessStep(
                action="compare artifact revisions",
                input=finding.baseline_revision or "<unversioned baseline>",
                output=finding.current_revision or "<unversioned current>",
            ),
        )
    return Diagnostic(
        rule_id="tokenizer-drift",
        severity=severity,
        message=(
            f"{finding.kind.value} in {finding.field}: "
            f"{_value_summary(finding.baseline)} -> {_value_summary(finding.current)}"
        ),
        span=SourceSpan(path=finding.current_path),
        check_modes=CHECK_MODE_CATALOG["tokenizer-drift"],
        suggestions=(
            "Review the tokenizer/config revision change before accepting the new baseline.",
            "Update the drift baseline only after downstream templates, stop policies, and parsers are verified.",
        ),
        properties=(
            ("baseline", finding.baseline),
            ("current", finding.current),
            ("field", finding.field),
            ("kind", finding.kind.value),
        ),
        witness=WitnessTrace(
            summary="A real tokenizer/config snapshot changed relative to the pinned drift baseline.",
            steps=(
                WitnessStep(action="load baseline tokenizer snapshot", input=finding.baseline_path),
                WitnessStep(action="load current tokenizer snapshot", input=finding.current_path),
                *revision_step,
                WitnessStep(action="compare drift field", input=finding.field, output=finding.kind.value),
                WitnessStep(action="record baseline value", output=_value_summary(finding.baseline)),
                WitnessStep(action="record current value", output=_value_summary(finding.current)),
            ),
        ),
    )


def _tokenizer_drift_abstention_diagnostic(abstention: TokenizerDriftAbstention) -> Diagnostic:
    return Diagnostic(
        rule_id="tokenizer-drift-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=f"tokenizer artifact '{abstention.artifact_name}' could not be checked for config drift",
        span=SourceSpan(path=abstention.path) if abstention.path is not None else None,
        check_modes=CHECK_MODE_CATALOG["tokenizer-drift-abstained"],
        suggestions=("Use a local tokenizer file/directory and a readable metadata.drift_baseline_path.",),
        witness=WitnessTrace(
            summary="PromptABI abstained instead of guessing drift from an unavailable or malformed baseline.",
            steps=(WitnessStep(action="load tokenizer drift baseline", input=abstention.path, output=abstention.reason),),
        ),
    )


def _tokenizer_drift_clean_diagnostic(compared: tuple[tuple[str, str], ...]) -> Diagnostic:
    pairs = "; ".join(f"{baseline} -> {current}" for baseline, current in compared)
    return Diagnostic(
        rule_id="tokenizer-drift-clean",
        severity=DiagnosticSeverity.INFO,
        message=f"tokenizer/config drift baseline matches {len(compared)} current artifact(s)",
        check_modes=CHECK_MODE_CATALOG["tokenizer-drift-clean"],
        suggestions=(),
        witness=WitnessTrace(
            summary="PromptABI compared real tokenizer/config snapshots and found no contract-relevant drift.",
            steps=(WitnessStep(action="compare tokenizer drift baselines", output=pairs),),
        ),
    )


def _provider_migration_diagnostic(finding: ProviderMigrationFinding) -> Diagnostic:
    severity = (
        DiagnosticSeverity.ERROR
        if finding.severity == "error"
        else DiagnosticSeverity.WARNING
        if finding.severity == "warning"
        else DiagnosticSeverity.INFO
    )
    context_steps = [
        WitnessStep(action="select source provider", input=finding.source_artifact_name, output=finding.source_provider),
    ]
    if finding.target_artifact_name is not None:
        context_steps.append(
            WitnessStep(action="select target provider", input=finding.target_artifact_name, output=finding.target_provider)
        )
    else:
        context_steps.append(WitnessStep(action="select target provider", input=finding.target_provider))
    evidence_steps = tuple(
        WitnessStep(action="compare provider migration field", input=key, output=value)
        for key, value in finding.evidence
    )
    return Diagnostic(
        rule_id="provider-migration",
        severity=severity,
        message=f"{finding.kind.value}: {finding.message}",
        span=finding.span,
        check_modes=CHECK_MODE_CATALOG["provider-migration"],
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(
            summary="A bounded recorded provider migration contract disagreed across source and target fixtures.",
            steps=tuple(context_steps) + evidence_steps,
        ),
    )


def _provider_fixture_replay_finding_diagnostic(finding: ProviderFixtureReplayFinding) -> Diagnostic:
    severity = (
        DiagnosticSeverity.ERROR
        if finding.severity == "error"
        else DiagnosticSeverity.WARNING
        if finding.severity == "warning"
        else DiagnosticSeverity.INFO
    )
    evidence_steps = tuple(
        WitnessStep(action="replay provider fixture field", input=key, output=value)
        for key, value in finding.evidence
    )
    return Diagnostic(
        rule_id="provider-fixture-replay",
        severity=severity,
        message=f"{finding.kind.value}: {finding.message}",
        span=finding.span,
        check_modes=CHECK_MODE_CATALOG["provider-fixture-replay"],
        suggestions=(finding.suggestion,),
        witness=WitnessTrace(
            summary="A recorded provider fixture pack failed deterministic offline replay.",
            steps=(
                WitnessStep(action="select provider fixture", input=finding.artifact_name, output=finding.provider),
                *evidence_steps,
            ),
        ),
    )


def _provider_fixture_replay_case_diagnostic(case: ProviderFixtureReplayCase, corpus_hash: str) -> Diagnostic:
    return Diagnostic(
        rule_id="provider-fixture-replay",
        severity=DiagnosticSeverity.INFO,
        message=(
            f"provider fixture '{case.artifact_name}' replayed {len(case.surfaces)} offline "
            f"surface(s) for {case.provider_family}"
        ),
        check_modes=CHECK_MODE_CATALOG["provider-fixture-replay"],
        witness=WitnessTrace(
            summary="PromptABI replayed the recorded provider API contract without contacting a third-party API.",
            steps=(
                WitnessStep(action="select provider fixture", input=case.artifact_name, output=case.provider),
                WitnessStep(action="replay provider fixture pack", input=", ".join(case.surfaces), output=case.replay_hash),
                WitnessStep(action="replay request fields", output=", ".join(case.request_fields)),
                WitnessStep(action="replay response fields", output=", ".join(case.response_fields)),
                WitnessStep(action="replay stop behavior", output=", ".join(case.stop_sequences)),
                WitnessStep(action="replay labeled edge cases", output=", ".join(case.edge_cases)),
                WitnessStep(action="record corpus replay hash", output=corpus_hash),
            ),
        ),
    )


def _stop_overreachability_finding_diagnostic(
    stop_loaded: LoadedArtifact,
    report: StopOverreachabilityReport,
    finding: StopOverreachabilityFinding,
) -> Diagnostic:
    rule_id = f"stop-overreach-{finding.category}"
    severity = DiagnosticSeverity.ERROR
    region = finding.region
    return Diagnostic(
        rule_id=rule_id,
        severity=severity,
        message=(
            f"stop sequence {finding.stop_sequence!r} can fire in {region.kind} region "
            f"'{region.name}' at {region.path}"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG[rule_id],
        suggestions=(
            "Use a stop delimiter that cannot occur in valid structured output, or require parser-aware completion.",
            "Prefer grammar/tool-call termination over raw substring stops for structured outputs.",
        ),
        witness=WitnessTrace(
            summary=(
                "A bounded valid structured-output witness contains the configured stop before "
                "the parser has reached a complete safe state."
            ),
            steps=(
                WitnessStep(action="select stop policy", input=report.stop_policy_name, output=report.bound),
                WitnessStep(action="build structured-output region", input=region.kind, output=region.description),
                WitnessStep(
                    action="locate stop firing point",
                    input=finding.stop_sequence,
                    output=finding.firing_point,
                ),
                WitnessStep(action="record parser state at truncation", output=finding.resulting_state),
                WitnessStep(action="show valid output prefix through stop", output=finding.valid_output_prefix),
                WitnessStep(action="show runtime-truncated prefix", output=finding.truncated_prefix),
                WitnessStep(
                    action="show resulting malformed or prematurely accepted structure",
                    output=finding.resulting_structure,
                ),
            ),
            artifacts=(stop_loaded.artifact.to_ref(),),
        ),
    )


def _stop_overreachability_abstention_diagnostic(
    stop_loaded: LoadedArtifact,
    report: StopOverreachabilityReport,
    abstention: StopOverreachabilityAbstention,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-overreach-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"structured artifact '{abstention.artifact_name}' is outside the bounded "
            f"stop-overreachability fragment"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-overreach-abstained"],
        suggestions=("Add a minimized JSON Schema or tool-parameter fixture in the supported object/string fragment.",),
        witness=WitnessTrace(
            summary="PromptABI did not claim content overreachability for an unsupported structured artifact.",
            steps=(
                WitnessStep(action="select stop policy", input=report.stop_policy_name, output=report.bound),
                WitnessStep(action="abstain on structured artifact", input=abstention.artifact_name, output=abstention.reason),
            ),
            artifacts=(stop_loaded.artifact.to_ref(),),
        ),
    )


def _stop_differential_mismatch_diagnostic(
    stop_loaded: LoadedArtifact,
    report: StopDifferentialReport,
    mismatch: StopDifferentialMismatch,
) -> Diagnostic:
    expected = mismatch.expected
    actual = mismatch.actual
    return Diagnostic(
        rule_id="stop-differential-mismatch",
        severity=DiagnosticSeverity.ERROR,
        message=(
            f"recorded stop trace '{mismatch.case.name}' disagrees with PromptABI's "
            f"{mismatch.case.family} stop simulator on {', '.join(mismatch.fields)}"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-differential-mismatch"],
        suggestions=(
            "Inspect whether the configured stop policy matches the runtime family captured by the fixture.",
            "Keep provider fixture traces version-pinned when stop matching behavior changes.",
        ),
        witness=WitnessTrace(
            summary="A recorded CPU-only provider/framework stop trace diverges from the local stop simulator.",
            steps=(
                WitnessStep(action="select stop policy", input=report.stop_policy_name, output=mismatch.case.family),
                WitnessStep(action="replay text chunks", input=mismatch.case.name, output=repr(mismatch.case.chunks)),
                WitnessStep(
                    action="compare stopped flag",
                    input=str(expected.stopped),
                    output=str(actual.stopped),
                ),
                WitnessStep(action="compare output text", input=repr(expected.output), output=repr(actual.output)),
                WitnessStep(
                    action="compare matched stop",
                    input=str(expected.matched_stop),
                    output=str(actual.matched_stop),
                ),
                WitnessStep(
                    action="compare stop inclusion",
                    input=str(expected.include_stop_in_output),
                    output=str(actual.include_stop_in_output),
                ),
            ),
            artifacts=(stop_loaded.artifact.to_ref(),),
        ),
    )


def _stop_differential_abstention_diagnostic(
    stop_loaded: LoadedArtifact,
    report: StopDifferentialReport,
    abstention: StopDifferentialAbstention,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-differential-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=f"provider fixture '{abstention.artifact_name}' cannot be replayed for stop differential testing",
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-differential-abstained"],
        suggestions=("Add a local provider-config JSON fixture with stop_trace or stop_traces entries.",),
        witness=WitnessTrace(
            summary="PromptABI abstained instead of guessing provider stop behavior without a supported trace.",
            steps=(
                WitnessStep(action="select stop policy", input=report.stop_policy_name),
                WitnessStep(action="load provider fixture", input=abstention.artifact_name, output=abstention.reason),
            ),
            artifacts=(stop_loaded.artifact.to_ref(),),
        ),
    )


def _stop_differential_agreement_diagnostic(
    stop_loaded: LoadedArtifact,
    report: StopDifferentialReport,
) -> Diagnostic:
    return Diagnostic(
        rule_id="stop-differential-agreement",
        severity=DiagnosticSeverity.INFO,
        message=(
            f"stop policy '{stop_loaded.artifact.name}' matches {len(report.matches)} "
            f"recorded stop trace(s)"
        ),
        artifact=stop_loaded.artifact.to_ref(),
        span=_artifact_span(stop_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["stop-differential-agreement"],
        witness=WitnessTrace(
            summary="Recorded provider/framework traces agreed with the bounded local stop simulator.",
            steps=(
                WitnessStep(action="select stop policy", input=report.stop_policy_name),
                WitnessStep(action="replay stop traces", output=f"{len(report.matches)} matched"),
            ),
            artifacts=(stop_loaded.artifact.to_ref(),),
        ),
    )


def _grammar_tokenizer_report_diagnostic(
    tokenizer_loaded: LoadedArtifact,
    grammar_loaded: LoadedArtifact,
    report: GrammarTokenizerEmptinessReport,
) -> Diagnostic:
    if report.status is GrammarTokenizerEmptinessStatus.SATISFIABLE:
        assert report.witness is not None
        return Diagnostic(
            rule_id="grammar-tokenizer-satisfiable",
            severity=DiagnosticSeverity.INFO,
            message=(
                f"grammar '{grammar_loaded.artifact.name}' has a tokenizer-compatible witness "
                f"under tokenizer '{tokenizer_loaded.artifact.name}'"
            ),
            artifact=grammar_loaded.artifact.to_ref(),
            span=_artifact_span(grammar_loaded.artifact),
            check_modes=CHECK_MODE_CATALOG["grammar-tokenizer-satisfiable"],
            witness=WitnessTrace(
                summary="A bounded grammar witness survived tokenizer encode-normalize-decode assumptions.",
                steps=(
                    WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                    WitnessStep(action="compile bounded grammar", input=grammar_loaded.artifact.name, output=report.grammar_kind),
                    WitnessStep(action="enumerate grammar witnesses", output=f"{report.checked_candidates} checked"),
                    WitnessStep(action="encode grammar witness", input=report.witness.grammar_text, output=str(report.witness.token_ids)),
                    WitnessStep(action="decode token path", input=str(report.witness.token_ids), output=report.witness.decoded_text),
                    WitnessStep(action="accept decoded text", output="accepted by grammar automaton"),
                ),
                artifacts=(tokenizer_loaded.artifact.to_ref(), grammar_loaded.artifact.to_ref()),
            ),
        )
    if report.status is GrammarTokenizerEmptinessStatus.EMPTY:
        first_attempt = report.attempts[0] if report.attempts else None
        steps = [
            WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
            WitnessStep(action="compile bounded grammar", input=grammar_loaded.artifact.name, output=report.grammar_kind),
            WitnessStep(action="enumerate grammar witnesses", output=f"{report.checked_candidates} checked"),
        ]
        if first_attempt is not None:
            steps.extend(
                [
                    WitnessStep(action="encode grammar witness", input=first_attempt.grammar_text, output=str(first_attempt.token_ids)),
                    WitnessStep(action="decode token path", input=str(first_attempt.token_ids), output=first_attempt.decoded_text),
                    WitnessStep(action="reject decoded text", output=first_attempt.reason),
                ]
            )
        else:
            steps.append(WitnessStep(action="prove no accepting grammar path", output=report.reason))
        return Diagnostic(
            rule_id="grammar-tokenizer-empty",
            severity=DiagnosticSeverity.ERROR,
            message=(
                f"grammar '{grammar_loaded.artifact.name}' is empty under tokenizer "
                f"'{tokenizer_loaded.artifact.name}' assumptions"
            ),
            artifact=grammar_loaded.artifact.to_ref(),
            span=_artifact_span(grammar_loaded.artifact),
            check_modes=CHECK_MODE_CATALOG["grammar-tokenizer-empty"],
            suggestions=(
                "Verify that the constrained-decoding backend tokenizes grammar literals with the same tokenizer settings.",
                "Avoid tokenizer normalization or added-token rules that rewrite required grammar terminals.",
            ),
            witness=WitnessTrace(
                summary=report.reason or "No bounded grammar witness was tokenizer-compatible.",
                steps=tuple(steps),
                artifacts=(tokenizer_loaded.artifact.to_ref(), grammar_loaded.artifact.to_ref()),
            ),
        )
    return _grammar_tokenizer_abstained_diagnostic(tokenizer_loaded, grammar_loaded, report.reason or "unsupported grammar product")


def _grammar_tokenizer_ambiguity_report_diagnostics(
    tokenizer_loaded: LoadedArtifact,
    grammar_loaded: LoadedArtifact,
    report: GrammarTokenizerAmbiguityReport,
) -> tuple[Diagnostic, ...]:
    if report.abstained:
        return (
            _grammar_tokenizer_ambiguity_abstained_diagnostic(
                tokenizer_loaded,
                grammar_loaded,
                report.reason or "unsupported grammar ambiguity product",
            ),
        )
    return tuple(
        _grammar_tokenizer_ambiguity_diagnostic(tokenizer_loaded, grammar_loaded, report, finding)
        for finding in report.findings
    )


def _grammar_tokenizer_ambiguity_diagnostic(
    tokenizer_loaded: LoadedArtifact,
    grammar_loaded: LoadedArtifact,
    report: GrammarTokenizerAmbiguityReport,
    finding: GrammarTokenizerAmbiguityFinding,
) -> Diagnostic:
    severity = (
        DiagnosticSeverity.ERROR
        if finding.kind
        in {
            GrammarTokenizerAmbiguityKind.TOKEN_PATH_CONFLICT,
            GrammarTokenizerAmbiguityKind.DECODED_TEXT_CONFLICT,
        }
        else DiagnosticSeverity.WARNING
    )
    return Diagnostic(
        rule_id="grammar-tokenizer-ambiguity",
        severity=severity,
        message=(
            f"grammar '{grammar_loaded.artifact.name}' has {finding.kind.value} under "
            f"tokenizer '{tokenizer_loaded.artifact.name}'"
        ),
        artifact=grammar_loaded.artifact.to_ref(),
        span=_artifact_span(grammar_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-tokenizer-ambiguity"],
        suggestions=(
            "Canonicalize structured outputs before parsing or make the grammar/tokenizer backend reject ambiguous spellings.",
            "Avoid tokenizer normalization or added tokens that collapse distinct schema values or parser states.",
        ),
        witness=WitnessTrace(
            summary=finding.reason,
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name, output=report.tokenizer_backend),
                WitnessStep(action="compile bounded grammar", input=grammar_loaded.artifact.name, output=report.grammar_kind),
                WitnessStep(action="enumerate JSON grammar variants", output=f"{report.checked_candidates} checked"),
                WitnessStep(action="encode first grammar text", input=finding.grammar_text, output=str(finding.token_ids)),
                WitnessStep(action="encode second grammar text", input=finding.other_grammar_text, output=str(finding.other_token_ids)),
                WitnessStep(action="compare structured values", input=finding.structured_value, output=finding.other_structured_value),
                WitnessStep(action="compare decoded text", input=finding.decoded_text, output=finding.other_decoded_text),
            ),
            artifacts=(tokenizer_loaded.artifact.to_ref(), grammar_loaded.artifact.to_ref()),
        ),
    )


def _grammar_tokenizer_ambiguity_abstained_diagnostic(
    tokenizer_loaded: LoadedArtifact,
    grammar_loaded: LoadedArtifact,
    reason: str,
) -> Diagnostic:
    return Diagnostic(
        rule_id="grammar-tokenizer-ambiguity-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"grammar '{grammar_loaded.artifact.name}' could not be checked for tokenizer ambiguity "
            f"against tokenizer '{tokenizer_loaded.artifact.name}'"
        ),
        artifact=grammar_loaded.artifact.to_ref(),
        span=_artifact_span(grammar_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-tokenizer-ambiguity-abstained"],
        suggestions=("Use a local JSON Schema artifact in PromptABI's bounded ambiguity fragment.",),
        witness=WitnessTrace(
            summary="PromptABI abstained instead of guessing tokenizer x grammar ambiguity outside the supported product.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name),
                WitnessStep(action="compile bounded grammar", input=grammar_loaded.artifact.name, output=reason),
            ),
            artifacts=(tokenizer_loaded.artifact.to_ref(), grammar_loaded.artifact.to_ref()),
        ),
    )


def _grammar_tokenizer_abstained_diagnostic(
    tokenizer_loaded: LoadedArtifact,
    grammar_loaded: LoadedArtifact,
    reason: str,
) -> Diagnostic:
    return Diagnostic(
        rule_id="grammar-tokenizer-abstained",
        severity=DiagnosticSeverity.WARNING,
        message=(
            f"grammar '{grammar_loaded.artifact.name}' could not be checked against tokenizer "
            f"'{tokenizer_loaded.artifact.name}'"
        ),
        artifact=grammar_loaded.artifact.to_ref(),
        span=_artifact_span(grammar_loaded.artifact),
        check_modes=CHECK_MODE_CATALOG["grammar-tokenizer-abstained"],
        suggestions=("Use a local JSON Schema artifact in PromptABI's bounded supported subset.",),
        witness=WitnessTrace(
            summary="PromptABI abstained instead of guessing tokenizer x grammar emptiness outside the supported product.",
            steps=(
                WitnessStep(action="select tokenizer", input=tokenizer_loaded.artifact.name),
                WitnessStep(action="compile bounded grammar", input=grammar_loaded.artifact.name, output=reason),
            ),
            artifacts=(tokenizer_loaded.artifact.to_ref(), grammar_loaded.artifact.to_ref()),
        ),
    )


def _value_summary(value: object) -> str:
    text = repr(value)
    if len(text) > 180:
        return text[:177] + "..."
    return text


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
