"""Tests for RLHF judge-prompt privacy checks (step 264)."""

from __future__ import annotations

from promptabi.judge_prompt_privacy import (
    JudgePrompt,
    JudgePrivacyFindingKind,
    check_judge_privacy,
    render_judge_privacy_text,
)


def test_clean_prompt_is_private() -> None:
    prompt = JudgePrompt(
        rubric="Score helpfulness 1-5.",
        transcript="User asked about weather. Assistant gave forecast.",
    )
    assert check_judge_privacy(prompt).private


def test_raw_private_value_leak() -> None:
    prompt = JudgePrompt(
        rubric="Score it.",
        transcript="user id 12345 said hi",
        private_field_values=("12345",),
    )
    result = check_judge_privacy(prompt)
    assert any(
        f.kind is JudgePrivacyFindingKind.PRIVATE_FIELD_LEAK for f in result.findings
    )


def test_redacted_value_is_ok() -> None:
    prompt = JudgePrompt(
        rubric="Score it.",
        transcript="user id [REDACTED] said hi",
        private_field_values=("[REDACTED]",),
    )
    # The literal value is redaction-marked, so it should not be flagged.
    assert check_judge_privacy(prompt).private


def test_email_pattern_leak() -> None:
    prompt = JudgePrompt(
        rubric="Score it.",
        transcript="contact me at alice@example.com please",
    )
    result = check_judge_privacy(prompt)
    assert any(f.kind is JudgePrivacyFindingKind.PATTERN_LEAK for f in result.findings)


def test_ssn_pattern_leak() -> None:
    prompt = JudgePrompt(rubric="r", transcript="ssn 123-45-6789")
    result = check_judge_privacy(prompt)
    assert not result.private


def test_render_text_smoke() -> None:
    result = check_judge_privacy(JudgePrompt("r", "t"))
    assert "privacy" in render_judge_privacy_text(result)
