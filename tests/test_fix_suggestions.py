import json

from promptabi import (
    ArtifactRef,
    Diagnostic,
    DiagnosticClusterStrategy,
    DiagnosticSeverity,
    FixBlastRadius,
    FixCompatibility,
    FixSafety,
    rank_fix_suggestions,
    render_diagnostic_clusters_json,
    render_diagnostic_clusters_text,
)
from promptabi.diagnostic_clustering import build_diagnostic_clusters


def test_fix_suggestion_ranking_prefers_safe_compatible_low_blast_radius_fixes() -> None:
    diagnostics = (
        Diagnostic(
            rule_id="artifact-unpinned",
            severity=DiagnosticSeverity.WARNING,
            message="tokenizer is not pinned by sha256",
            artifact=ArtifactRef(kind="tokenizer", name="dev-tokenizer", path="tokenizer.json"),
            suggestions=(
                "Rewrite the chat template and stop policy to avoid this tokenizer.",
                "Add a sha256 pin and refresh the lockfile after review.",
                "Suppress this diagnostic as accepted risk.",
            ),
        ),
    )

    ranked = rank_fix_suggestions(diagnostics)

    assert [item.text for item in ranked] == [
        "Add a sha256 pin and refresh the lockfile after review.",
        "Rewrite the chat template and stop policy to avoid this tokenizer.",
        "Suppress this diagnostic as accepted risk.",
    ]
    assert ranked[0].safety is FixSafety.HIGH
    assert ranked[0].compatibility is FixCompatibility.HIGH
    assert ranked[0].blast_radius is FixBlastRadius.LOW
    assert ranked[0].changes_user_visible_prompt_behavior is False
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_fix_suggestion_ranking_honors_explicit_diagnostic_metadata() -> None:
    diagnostics = (
        Diagnostic(
            rule_id="template-boundary",
            severity=DiagnosticSeverity.ERROR,
            message="role delimiter can be forged",
            suggestions=("Escape user content before rendering role delimiters.",),
            properties=(
                ("fix_safety", "high"),
                ("fix_compatibility", "low"),
                ("fix_blast_radius", "high"),
                ("fix_changes_user_visible_prompt_behavior", "true"),
            ),
        ),
    )

    suggestion = rank_fix_suggestions(diagnostics)[0]

    assert suggestion.safety is FixSafety.HIGH
    assert suggestion.compatibility is FixCompatibility.LOW
    assert suggestion.blast_radius is FixBlastRadius.HIGH
    assert suggestion.changes_user_visible_prompt_behavior is True
    assert "changes user-visible prompt behavior" in suggestion.rationale


def test_diagnostic_clusters_expose_ranked_suggestions_in_text_and_json() -> None:
    diagnostics = (
        Diagnostic(
            rule_id="stop-overreach-content",
            severity=DiagnosticSeverity.ERROR,
            message="Stop sequence END can fire inside JSON.",
            suggestions=(
                "Change the provider stop sequence and update parser tests.",
                "Document the stop policy limitation only.",
            ),
            properties=(("root_cause_id", "stop-policy:end-in-json"),),
        ),
        Diagnostic(
            rule_id="parser-compatibility-mismatch",
            severity=DiagnosticSeverity.WARNING,
            message="Parser expects complete JSON after provider stop handling.",
            suggestions=("Change the provider stop sequence and update parser tests.",),
            properties=(("root_cause_id", "stop-policy:end-in-json"),),
        ),
    )

    report = build_diagnostic_clusters(
        diagnostics,
        strategies=(DiagnosticClusterStrategy.ROOT_CAUSE,),
        min_cluster_size=2,
    )

    text = render_diagnostic_clusters_text(report)
    payload = json.loads(render_diagnostic_clusters_json(report))
    ranked = payload["clusters"][0]["ranked_suggestions"]

    assert report.clusters[0].suggestions[0] == "Document the stop policy limitation only."
    assert "suggestion[1]: Document the stop policy limitation only." in text
    assert ranked[0]["safety"] == "medium"
    assert ranked[0]["blast_radius"] == "low"
    assert ranked[1]["diagnostic_count"] == 2
