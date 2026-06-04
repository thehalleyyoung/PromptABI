import json
from pathlib import Path

from promptabi import (
    ArtifactKind,
    ArtifactLocation,
    ByteLevelTokenizer,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    TokenBudgetReservation,
    TokenizerArtifact,
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
        "token-budget-required-truncated",
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


def test_langchain_default_truncation_preserves_system_but_drops_old_required_message() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="conversation",
        location=ArtifactLocation(uri="memory://conversation"),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=18),
            PromptSegment("early-user-task", role="user", required=True, token_count=26),
            PromptSegment("latest-user-task", role="user", required=True, token_count=20),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="langchain-memory",
        location=ArtifactLocation(uri="memory://langchain-memory"),
        framework="langchain",
        max_context_tokens=50,
    )

    report = analyze_token_budget(
        VerificationConfig(name="langchain-budget", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
    )

    assert report.policy is not None
    assert report.policy.strategy == "oldest-message"
    assert report.truncation is not None
    assert [segment.name for segment in report.truncation.kept_segments] == [
        "system-policy",
        "latest-user-task",
    ]
    assert report.must_survive_proof is not None
    assert report.must_survive_proof.status == "violated"
    assert report.must_survive_proof.dropped_segments == ("early-user-task",)
    assert [segment.name for segment in report.must_survive_proof.minimal_counterexample] == [
        "system-policy",
        "early-user-task",
        "latest-user-task",
    ]
    assert [finding.rule_id for finding in report.findings] == [
        "token-budget-required-overflow",
        "token-budget-total-overflow",
        "token-budget-required-truncated",
    ]
    truncated = next(finding for finding in report.findings if finding.rule_id == "token-budget-required-truncated")
    assert dict(truncated.evidence)["dropped_required"] == "early-user-task"


def test_custom_rag_priority_policy_drops_retrieval_before_required_segments() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="rag",
        location=ArtifactLocation(uri="memory://rag"),
        segments=(
            PromptSegment("system-policy", role="system", required=True, token_count=15),
            PromptSegment("retrieval-a", role="retrieval", required=False, token_count=30),
            PromptSegment("retrieval-b", role="retrieval", required=False, token_count=20),
            PromptSegment("question", role="user", required=True, token_count=14),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="rag-budget",
        location=ArtifactLocation(uri="memory://rag-budget"),
        framework="custom-rag-pipeline",
        strategy="priority",
        drop_roles=("retrieval",),
        max_context_tokens=55,
        preserve_system=True,
    )

    report = analyze_token_budget(
        VerificationConfig(name="rag-budget", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
    )

    assert report.truncation is not None
    assert [segment.name for segment in report.truncation.dropped_segments] == ["retrieval-a"]
    assert [finding.rule_id for finding in report.findings] == [
        "token-budget-total-overflow",
        "token-budget-framework-truncation",
    ]
    optional_drop = next(finding for finding in report.findings if finding.rule_id == "token-budget-framework-truncation")
    assert "retrieval-a" in optional_drop.message


def test_vllm_left_truncation_uses_declaration_order_not_segment_name_order() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="ordered",
        location=ArtifactLocation(uri="memory://ordered"),
        segments=(
            PromptSegment("z-first", role="user", required=False, token_count=25),
            PromptSegment("a-second", role="user", required=True, token_count=20),
            PromptSegment("m-third", role="assistant", required=True, token_count=20),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="vllm-budget",
        location=ArtifactLocation(uri="memory://vllm-budget"),
        framework="vllm",
        max_context_tokens=45,
    )

    report = analyze_token_budget(
        VerificationConfig(name="vllm-budget", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
    )

    assert report.policy is not None
    assert report.policy.strategy == "left"
    assert report.truncation is not None
    assert [segment.name for segment in report.truncation.dropped_segments] == ["z-first"]
    assert [segment.name for segment in report.truncation.kept_segments] == ["a-second", "m-third"]
    assert report.must_survive_proof is not None
    assert report.must_survive_proof.status == "proven"
    assert report.must_survive_proof.survived_segments == ("a-second", "m-third")
    assert report.visualization is not None
    visualization = report.visualization.to_dict()
    assert visualization["input_budget_tokens"] == 45
    assert visualization["dropped_fields"] == ["z-first"]
    assert visualization["must_survive_status"] == "proven"
    rows = visualization["rows"]
    assert [row["name"] for row in rows] == ["z-first", "a-second", "m-third"]
    assert [row["status"] for row in rows] == ["dropped", "kept", "kept"]
    assert [row["survival"] for row in rows] == ["optional-dropped", "guaranteed", "guaranteed"]
    assert [(row["start_token"], row["end_token"]) for row in rows] == [(0, 25), (25, 45), (45, 65)]


def test_must_survive_counterexample_is_minimized_for_real_truncation_policy() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="chat",
        location=ArtifactLocation(uri="memory://chat"),
        segments=(
            PromptSegment("optional-preface", role="user", required=False, token_count=12),
            PromptSegment("must-survive-system", role="system", required=True, token_count=30),
            PromptSegment("optional-retrieval", role="retrieval", required=False, token_count=45),
            PromptSegment("latest-question", role="user", required=True, token_count=25),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="transformers-budget",
        location=ArtifactLocation(uri="memory://transformers-budget"),
        framework="transformers",
        max_context_tokens=50,
    )

    report = analyze_token_budget(
        VerificationConfig(name="must-survive", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
    )

    assert report.must_survive_proof is not None
    assert report.must_survive_proof.status == "violated"
    assert report.must_survive_proof.dropped_segments == ("must-survive-system",)
    assert [segment.name for segment in report.must_survive_proof.minimal_counterexample] == [
        "must-survive-system",
        "latest-question",
    ]
    assert report.must_survive_proof.counterexample_tokens == 55
    truncated = next(finding for finding in report.findings if finding.rule_id == "token-budget-required-truncated")
    evidence = dict(truncated.evidence)
    assert evidence["minimal_counterexample"] == "must-survive-system=30, latest-question=25"
    assert evidence["minimal_counterexample_tokens"] == "55"


def test_rag_chunking_compatibility_reports_real_chunk_contract_failures() -> None:
    segments = PromptSegmentArtifact(
        kind=ArtifactKind.PROMPT_SEGMENT,
        name="rag",
        location=ArtifactLocation(uri="memory://rag"),
        segments=(
            PromptSegment(
                "system",
                role="system",
                required=True,
                token_count=10,
            ),
            PromptSegment(
                "chunk-a",
                role="retrieval",
                token_count=30,
                max_tokens=40,
                chunk_id="a",
                document_id="doc-1",
                chunk_tokenizer="embedder-v1",
                source_start=0,
                source_end=30,
                chunk_start=1,
                chunk_end=30,
                citation="doc-1#0",
                citation_required=True,
                metadata_tokens=20,
                template_overhead_tokens=8,
                retrieval_payload_limit_tokens=45,
            ),
            PromptSegment(
                "chunk-b",
                role="retrieval",
                token_count=25,
                max_tokens=35,
                chunk_id="b",
                document_id="doc-1",
                chunk_tokenizer="serving-tokenizer",
                source_start=25,
                source_end=50,
                expected_overlap_tokens=8,
                actual_overlap_tokens=5,
                citation_required=True,
                metadata_tokens=2,
                template_overhead_tokens=4,
                retrieval_payload_limit_tokens=80,
            ),
            PromptSegment("question", role="user", required=True, token_count=10),
        ),
    )
    budget = FrameworkTruncationConfigArtifact(
        kind=ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG,
        name="rag-budget",
        location=ArtifactLocation(uri="memory://rag-budget"),
        framework="custom-rag",
        strategy="priority",
        drop_roles=("retrieval",),
        max_context_tokens=80,
        preserve_system=True,
    )
    tokenizer_artifact = TokenizerArtifact(
        kind=ArtifactKind.TOKENIZER,
        name="serving-tokenizer",
        family="byte-level",
        location=ArtifactLocation(uri="memory://tokenizer"),
    )

    report = analyze_token_budget(
        VerificationConfig(name="rag", artifact_bundle=()),
        (ArtifactLoader().load(segments), ArtifactLoader().load(budget)),
        tokenizers=((tokenizer_artifact, ByteLevelTokenizer()),),
    )

    rag_rule_ids = [finding.rule_id for finding in report.findings if finding.rule_id.startswith("rag-")]
    assert rag_rule_ids == [
        "rag-tokenizer-mismatch",
        "rag-chunk-boundary-drift",
        "rag-citation-loss",
        "rag-payload-truncation",
        "rag-payload-truncation",
        "rag-metadata-inflation",
        "rag-template-overhead",
        "rag-citation-loss",
        "rag-overlap-accounting",
    ]
    assert report.truncation is not None
    assert [segment.name for segment in report.truncation.dropped_segments] == ["chunk-a"]


def test_verify_rag_chunking_example_uses_dedicated_check(capsys) -> None:
    exit_code = main(["verify", "--config", "examples/rag-chunking/promptabi.json", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rule_ids = [diagnostic["rule_id"] for diagnostic in payload["diagnostics"]]

    assert exit_code == 1
    assert "rag-citation-loss" in rule_ids
    assert "rag-overlap-accounting" in rule_ids
    assert "rag-template-overhead" in rule_ids
    assert "token-budget-model" not in rule_ids


def test_token_budget_visualizer_renders_through_text_json_and_sarif(capsys) -> None:
    exit_code = main(
        [
            "verify",
            "--config",
            "examples/token-budget/promptabi.json",
            "--format",
            "text",
            "--fail-on",
            "never",
        ]
    )
    text_output = capsys.readouterr().out

    assert exit_code == 0
    assert "render token-budget visualization" in text_output
    assert "budget=runtime-budget framework=vllm:left" in text_output
    assert "dropped=system-policy" in text_output
    assert "boundary=60" in text_output

    exit_code = main(
        [
            "verify",
            "--config",
            "examples/token-budget/promptabi.json",
            "--format",
            "json",
            "--fail-on",
            "never",
        ]
    )
    json_payload = json.loads(capsys.readouterr().out)
    summary = next(
        diagnostic
        for diagnostic in json_payload["diagnostics"]
        if diagnostic["rule_id"] == "token-budget-model"
    )
    json_visualization = summary["properties"]["token_budget_visualization"]

    assert exit_code == 0
    assert json_visualization["input_budget_tokens"] == 60
    assert json_visualization["total_prompt_tokens"] == 114
    assert json_visualization["dropped_fields"] == ["system-policy"]
    assert [row["name"] for row in json_visualization["rows"]] == [
        "system-policy",
        "retrieval-context",
        "user-request",
    ]
    assert [row["status"] for row in json_visualization["rows"]] == ["dropped", "kept", "kept"]

    exit_code = main(
        [
            "verify",
            "--config",
            "examples/token-budget/promptabi.json",
            "--format",
            "sarif",
            "--fail-on",
            "never",
        ]
    )
    sarif_payload = json.loads(capsys.readouterr().out)
    sarif_summary = next(
        result
        for result in sarif_payload["runs"][0]["results"]
        if result["ruleId"] == "token-budget-model"
    )
    sarif_visualization = sarif_summary["properties"]["token_budget_visualization"]

    assert exit_code == 0
    assert sarif_visualization["must_survive_status"] == "violated"
    assert sarif_visualization["rows"][0]["survival"] == "violated"
