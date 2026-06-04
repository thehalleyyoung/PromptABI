import json
from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    TokenBudgetReservation,
    VerificationConfig,
    analyze_token_budget,
)
from promptabi.cli import main
from promptabi.loaders import ArtifactLoader


def test_token_budget_reservation_models_context_arithmetic() -> None:
    reservation = TokenBudgetReservation(
        max_context_tokens=100,
        reserve_output_tokens=20,
        reserved_tool_tokens=7,
        generation_prompt_tokens=3,
        special_token_overhead=5,
    )

    assert reservation.reserved_total == 35
    assert reservation.input_budget_tokens == 65
    assert dict(reservation.to_metadata())["input_budget_tokens"] == 65


def test_budget_analyzer_reports_required_overflow_from_declared_counts() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="segments",
        location=ArtifactLocation(uri="memory://segments"),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=60),
            PromptSegment("user-request", role="user", required=True, token_count=30),
            PromptSegment("retrieval", role="user", required=False, max_tokens=50),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="budget",
        location=ArtifactLocation(uri="memory://budget"),
        framework="vllm",
        max_context_tokens=100,
        reserve_output_tokens=20,
        reserved_tool_tokens=5,
    )
    config = VerificationConfig(name="budget", artifact_bundle=())

    report = analyze_token_budget(
        config,
        (
            ArtifactLoader().load(segments),
            ArtifactLoader().load(budget),
        ),
    )

    assert report.reservation is not None
    assert report.reservation.input_budget_tokens == 75
    assert report.required_prompt_tokens == 90
    assert [finding.rule_id for finding in report.findings] == [
        "token-budget-required-overflow",
        "token-budget-total-overflow",
    ]


def test_prompt_segment_loader_merges_real_messages_with_declared_segments(tmp_path: Path) -> None:
    messages = tmp_path / "messages.json"
    messages.write_text(
        json.dumps(
            [
                {"role": "system", "content": "System policy."},
                {"role": "user", "content": "Question."},
            ]
        ),
        encoding="utf-8",
    )
    artifact = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="messages",
        location=ArtifactLocation(path=str(messages)),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=12),
            PromptSegment("user-request", role="user", required=True),
        ),
    )

    loaded = ArtifactLoader().load(artifact)

    assert loaded.source_type == "prompt-segments"
    assert isinstance(loaded.artifact, PromptSegmentArtifact)
    assert loaded.artifact.segments[0].content == "System policy."
    assert loaded.artifact.segments[1].content == "Question."
    assert dict(loaded.metadata)["segment_count"] == 2


def test_verify_token_budget_example_reports_real_budget_overflow(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/token-budget/promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]]

    assert exit_code == 1
    assert "token-budget-context-conflict" in rule_ids
    assert "token-budget-required-overflow" in rule_ids
    assert "token-budget-model" in rule_ids
    overflow = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["rule_id"] == "token-budget-required-overflow"
    )
    assert "required prompt segments need 74 token(s)" in overflow["message"]
    assert overflow["check_modes"] == ["bounded", "sound"]
