import json

from promptabi import (
    ArtifactRef,
    Diagnostic,
    DiagnosticClusterStrategy,
    DiagnosticSeverity,
    WitnessTrace,
    build_diagnostic_clusters,
    diagnostic_clusters,
    render_diagnostic_clusters_json,
    render_diagnostic_clusters_text,
)
from promptabi.cli import main


def test_diagnostic_clustering_groups_real_triage_dimensions() -> None:
    shared_witness = WitnessTrace(
        summary="valid JSON string is truncated by provider stop",
        rendered_strings=('{"answer": "END"}',),
        parser_states=("inside-json-string",),
        solver_assignments=({"stop": "END", "parser_state": "inside-string"},),
        minimal_fixes=("Use a stop sequence that cannot appear inside JSON strings.",),
    )
    diagnostics = (
        Diagnostic(
            rule_id="stop-overreach-content",
            severity=DiagnosticSeverity.ERROR,
            message="Stop sequence END can fire inside a JSON string.",
            artifact=ArtifactRef(kind="stop-policy", name="openai-stop", path="stop.json"),
            witness=shared_witness,
            suggestions=("Use a stop sequence that cannot appear inside JSON strings.",),
            properties=(
                ("root_cause_id", "stop-policy:end-in-json-string"),
                ("provider_family", "openai-compatible"),
                ("provider_behavior", "string stop before JSON parser completes"),
                ("source_artifact", "stop-policy:openai-stop"),
                ("target_artifact", "schema:answer-json"),
            ),
        ),
        Diagnostic(
            rule_id="parser-compatibility-mismatch",
            severity=DiagnosticSeverity.WARNING,
            message="Application parser expects complete JSON after provider stop handling.",
            artifact=ArtifactRef(kind="schema", name="answer-json", path="schema.json"),
            witness=shared_witness,
            suggestions=("Use a stop sequence that cannot appear inside JSON strings.",),
            properties=(
                ("root_cause_id", "stop-policy:end-in-json-string"),
                ("provider_family", "openai-compatible"),
                ("provider_behavior", "string stop before JSON parser completes"),
                ("source_artifact", "stop-policy:openai-stop"),
                ("target_artifact", "schema:answer-json"),
            ),
        ),
        Diagnostic(
            rule_id="artifact-unpinned",
            severity=DiagnosticSeverity.INFO,
            message="Tokenizer artifact is not pinned.",
            artifact=ArtifactRef(kind="tokenizer", name="dev-tokenizer", path="tokenizer.json"),
        ),
    )

    report = build_diagnostic_clusters(diagnostics)

    strategies = {cluster.strategy for cluster in report.clusters}
    assert DiagnosticClusterStrategy.ROOT_CAUSE in strategies
    assert DiagnosticClusterStrategy.ARTIFACT_EDGE in strategies
    assert DiagnosticClusterStrategy.PROVIDER_BEHAVIOR in strategies
    assert DiagnosticClusterStrategy.SHARED_WITNESS in strategies
    assert report.total_diagnostics == 3
    assert report.clustered_diagnostic_count == 2
    assert report.unclustered_diagnostic_count == 1
    root_cause = next(cluster for cluster in report.clusters if cluster.strategy is DiagnosticClusterStrategy.ROOT_CAUSE)
    assert root_cause.worst_severity is DiagnosticSeverity.ERROR
    assert root_cause.rules == ("parser-compatibility-mismatch", "stop-overreach-content")
    assert root_cause.suggestions == ("Use a stop sequence that cannot appear inside JSON strings.",)

    text = render_diagnostic_clusters_text(report)
    payload = json.loads(render_diagnostic_clusters_json(report))
    assert "PromptABI diagnostic clusters" in text
    assert payload["clustered_diagnostic_count"] == 2
    assert payload["unclustered_diagnostic_count"] == 1


def test_diagnostic_clustering_handles_non_json_metadata_without_crashing() -> None:
    diagnostics = (
        Diagnostic(
            rule_id="static-contract-violation",
            severity=DiagnosticSeverity.ERROR,
            message="SMT model violates a role invariant.",
            witness=WitnessTrace(
                summary="solver assignment uses a non-JSON object from a plugin",
                solver_assignments=({"plugin_value": {"assistant", "user"}},),
            ),
        ),
    )

    rendered = diagnostic_clusters(
        diagnostics,
        strategies=(DiagnosticClusterStrategy.SHARED_WITNESS.value,),
        min_cluster_size=1,
        output_format="json",
    )

    payload = json.loads(rendered)
    assert payload["cluster_count"] == 1
    assert payload["clusters"][0]["strategy"] == "shared-witness"


def test_diagnostics_cluster_cli_reports_clusters_from_real_config(capsys) -> None:
    exit_code = main(
        [
            "diagnostics",
            "cluster",
            "--config",
            "examples/minimal/promptabi.json",
            "--strategy",
            "rule",
            "--min-size",
            "1",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["total_diagnostics"] >= 1
    assert payload["clusters"][0]["strategy"] == "rule"
    assert payload["clusters"][0]["members"][0]["rule_id"] == "repository-skeleton"
    assert captured.err == ""
