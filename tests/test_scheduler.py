import threading

import promptabi.session as session_module
from promptabi import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    CheckContext,
    CheckMode,
    Diagnostic,
    DiagnosticSeverity,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    VerificationConfig,
    VerificationSession,
    WitnessTrace,
)


def test_scheduler_parallelizes_independent_embedded_checks() -> None:
    barrier = threading.Barrier(2)

    def first_check(context: CheckContext):
        del context
        barrier.wait(timeout=2)
        return (
            Diagnostic(
                rule_id="scheduler-parallel",
                severity=DiagnosticSeverity.INFO,
                message="first completed",
                check_modes=(CheckMode.HEURISTIC,),
            ),
        )

    def second_check(context: CheckContext):
        del context
        barrier.wait(timeout=2)
        return (
            Diagnostic(
                rule_id="scheduler-parallel",
                severity=DiagnosticSeverity.INFO,
                message="second completed",
                check_modes=(CheckMode.HEURISTIC,),
            ),
        )

    config = VerificationConfig(name="parallel-scheduler", checks=())

    diagnostics = VerificationSession(config).collect_diagnostics(checks=(first_check, second_check))

    assert [diagnostic.message for diagnostic in diagnostics] == ["first completed", "second completed"]
    assert all(diagnostic.rule_id == "scheduler-parallel" for diagnostic in diagnostics)


def test_scheduler_preserves_deterministic_tie_order() -> None:
    def alpha(context: CheckContext):
        del context
        return (
            Diagnostic(
                rule_id="same-sort-key",
                severity=DiagnosticSeverity.WARNING,
                message="same message",
                witness=WitnessTrace(summary="alpha"),
            ),
        )

    def beta(context: CheckContext):
        del context
        return (
            Diagnostic(
                rule_id="same-sort-key",
                severity=DiagnosticSeverity.WARNING,
                message="same message",
                witness=WitnessTrace(summary="beta"),
            ),
        )

    config = VerificationConfig(name="deterministic-scheduler", checks=())
    session = VerificationSession(config)
    expected = ("alpha", "beta")

    for _ in range(20):
        diagnostics = session.collect_diagnostics(checks=(alpha, beta))
        assert tuple(diagnostic.witness.summary for diagnostic in diagnostics if diagnostic.witness) == expected


def test_scheduler_reuses_token_budget_cache_between_budget_and_rag_checks(monkeypatch) -> None:
    calls = 0
    original = session_module.analyze_token_budget

    def counting_analyze_token_budget(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(session_module, "analyze_token_budget", counting_analyze_token_budget)
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="rag",
        location=ArtifactLocation(uri="memory://rag"),
        segments=(
            PromptSegment("system", role="system", required=True, token_count=15),
            PromptSegment("retrieval", role="retrieval", required=False, token_count=40),
            PromptSegment("question", role="user", required=True, token_count=12),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="budget",
        location=ArtifactLocation(uri="memory://budget"),
        framework="custom-rag-pipeline",
        strategy="priority",
        drop_roles=("retrieval",),
        max_context_tokens=50,
        preserve_system=True,
    )
    config = VerificationConfig(
        name="cached-budget",
        checks=("token-budget-model", "rag-chunking-compatibility"),
        artifact_bundle=ArtifactBundle((segments, budget)),
    )

    diagnostics = VerificationSession(config).collect_diagnostics()

    assert calls == 1
    assert {diagnostic.rule_id for diagnostic in diagnostics} >= {
        "token-budget-model",
        "token-budget-framework-truncation",
    }
