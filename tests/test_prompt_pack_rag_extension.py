"""Tests for prompt-pack RAG extension points (step 251)."""

from __future__ import annotations

from promptabi.prompt_pack_rag_extension import (
    ExtensionPoint,
    PackTemplate,
    RagFindingKind,
    render_rag_verification_text,
    verify_extension_points,
)

SAFE_TEMPLATE = PackTemplate(
    source=(
        "<|im_start|>system\nUse this context:\n{{retrieved_docs}}<|im_end|>\n"
        "<|im_start|>user\n{{user_message}}<|im_end|>\n"
    )
)


def test_safe_extension_points() -> None:
    points = (
        ExtensionPoint("retrieved_docs", "{{retrieved_docs}}", sanitizer="json_escape"),
        ExtensionPoint("user_message", "{{user_message}}", sanitizer="json_escape"),
    )
    result = verify_extension_points(SAFE_TEMPLATE, points)
    assert result.safe, result.findings


def test_missing_slot() -> None:
    result = verify_extension_points(
        SAFE_TEMPLATE,
        (ExtensionPoint("ghost", "{{ghost}}", sanitizer="x"),),
    )
    assert any(f.kind is RagFindingKind.SLOT_NOT_FOUND for f in result.findings)


def test_unsanitized_slot() -> None:
    result = verify_extension_points(
        SAFE_TEMPLATE,
        (ExtensionPoint("retrieved_docs", "{{retrieved_docs}}"),),
    )
    assert any(f.kind is RagFindingKind.SLOT_UNSANITIZED for f in result.findings)


def test_slot_in_control_region() -> None:
    template = PackTemplate(source="<|im_start|>{{retrieved_docs}} body")
    result = verify_extension_points(
        template,
        (ExtensionPoint("retrieved_docs", "{{retrieved_docs}}", sanitizer="x"),),
    )
    assert any(f.kind is RagFindingKind.SLOT_IN_CONTROL_REGION for f in result.findings)


def test_duplicate_slot() -> None:
    points = (
        ExtensionPoint("retrieved_docs", "{{retrieved_docs}}", sanitizer="x"),
        ExtensionPoint("retrieved_docs", "{{retrieved_docs}}", sanitizer="x"),
    )
    result = verify_extension_points(SAFE_TEMPLATE, points)
    assert any(f.kind is RagFindingKind.DUPLICATE_SLOT for f in result.findings)


def test_render_text_smoke() -> None:
    result = verify_extension_points(SAFE_TEMPLATE, ())
    assert "RAG" in render_rag_verification_text(result)
