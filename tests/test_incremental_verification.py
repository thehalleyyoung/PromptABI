import json

from promptabi.artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    SchemaArtifact,
)
from promptabi.cli import main
from promptabi.incremental import plan_incremental_checks
from promptabi.loaders import LoadedArtifact
from promptabi import VerificationConfig, VerificationSession


def test_incremental_cli_reuses_cached_unchanged_check_results(tmp_path, capsys) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {"answer": {"type": "string"}}}', encoding="utf-8")
    config = tmp_path / "promptabi.json"
    config.write_text(
        json.dumps(
            {
                "name": "incremental-cli",
                "checks": ["parser-compatibility", "role-boundary-nonforgeability"],
                "artifacts": {"schema": {"kind": "schema", "path": "schema.json"}},
            }
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"

    first_exit = main(
        [
            "verify",
            "--config",
            str(config),
            "--cache-dir",
            str(cache_dir),
            "--changed-path",
            "schema.json",
            "--format",
            "json",
        ]
    )
    first = json.loads(capsys.readouterr().out)

    second_exit = main(
        [
            "verify",
            "--config",
            str(config),
            "--cache-dir",
            str(cache_dir),
            "--changed-path",
            "schema.json",
            "--format",
            "json",
        ]
    )
    second = json.loads(capsys.readouterr().out)

    assert first_exit == 0
    assert second_exit == 0
    assert {diagnostic["rule_id"] for diagnostic in first["diagnostics"]} >= {
        "incremental-cache-miss",
        "parser-compatibility-agreement",
    }
    assert {diagnostic["rule_id"] for diagnostic in second["diagnostics"]} >= {
        "incremental-check-reused",
        "parser-compatibility-agreement",
    }
    reused = next(
        diagnostic
        for diagnostic in second["diagnostics"]
        if diagnostic["rule_id"] == "incremental-check-reused"
    )
    assert reused["properties"]["check"] == "role-boundary-nonforgeability"
    assert reused["properties"]["changed_artifacts"] == ["schema"]
    assert reused["properties"]["selected_checks"] == ["parser-compatibility"]
    assert reused["properties"]["skipped_checks"] == ["role-boundary-nonforgeability"]


def test_incremental_planner_forces_full_run_when_config_changes(tmp_path) -> None:
    config_path = tmp_path / "promptabi.json"
    config_path.write_text("{}", encoding="utf-8")
    schema = SchemaArtifact(
        kind=ArtifactKind.SCHEMA,
        name="schema",
        location=ArtifactLocation(path=str((tmp_path / "schema.json").resolve())),
    )
    session = VerificationSession(
        VerificationConfig(
            name="config-change",
            checks=("parser-compatibility", "role-boundary-nonforgeability"),
            artifact_bundle=ArtifactBundle((schema,)),
        )
    )

    plan = plan_incremental_checks(
        session,
        changed_paths=(config_path,),
        config_path=config_path,
        loaded_artifacts=(LoadedArtifact(schema, "json-schema", pinned=False, resolved=False),),
    )

    assert plan.full_run
    assert plan.full_run_reason == "config changed"
    assert plan.selected_checks == ("parser-compatibility", "role-boundary-nonforgeability")
    assert plan.skipped_checks == ()


def test_incremental_planner_closes_dependent_checks_for_prompt_budget(tmp_path) -> None:
    prompt_path = (tmp_path / "segments.json").resolve()
    budget_path = (tmp_path / "budget.json").resolve()
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="segments",
        location=ArtifactLocation(path=str(prompt_path)),
        segments=(PromptSegment("system", role="system", required=True, token_count=8),),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="budget",
        location=ArtifactLocation(path=str(budget_path)),
        framework="custom",
        strategy="priority",
        max_context_tokens=16,
    )
    session = VerificationSession(
        VerificationConfig(
            name="dependency-closure",
            checks=("rag-chunking-compatibility", "token-budget-model", "parser-compatibility"),
            artifact_bundle=ArtifactBundle((segments, budget)),
        )
    )

    plan = plan_incremental_checks(
        session,
        changed_paths=(prompt_path,),
        config_path=tmp_path / "promptabi.json",
        loaded_artifacts=(
            LoadedArtifact(segments, "prompt-segments", pinned=False, resolved=False),
            LoadedArtifact(budget, "framework-truncation", pinned=False, resolved=False),
        ),
    )

    assert plan.changed_artifacts == ("segments",)
    assert plan.changed_kinds == (ArtifactKind.PROMPT_SEGMENT,)
    assert plan.selected_checks == ("rag-chunking-compatibility", "token-budget-model")
    assert plan.skipped_checks == ("parser-compatibility",)
