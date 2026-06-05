"""Tests for prompt-pack example certification (step 246)."""

from __future__ import annotations

from promptabi.artifacts import (
    ArtifactKind,
    ArtifactLocation,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
)
from promptabi.prompt_pack_example_certification import (
    ExampleFindingKind,
    PackExample,
    certify_examples,
    render_example_report_text,
)

LOCATION = ArtifactLocation(path="/tmp/pack.json")


def _pack() -> PromptPackArtifact:
    return PromptPackArtifact(
        kind=ArtifactKind.PROMPT_PACK,
        name="pack",
        location=LOCATION,
        pack_name="support-pack",
        pack_version="1.0.0",
        exported_templates=(
            PromptPackTemplate(
                name="chat",
                template="{{ messages }}",
                roles=("system", "user", "assistant"),
                supported_model_families=("llama", "mistral"),
            ),
            PromptPackTemplate(
                name="tool_chat",
                template="{{ messages }}{{ tools }}",
                roles=("system", "user", "assistant", "tool"),
                supported_model_families=("llama",),
            ),
        ),
        expected_roles=("system", "user"),
        tool_schemas=(
            PromptPackToolSchema(name="search", required=True),
            PromptPackToolSchema(name="lookup", required=False),
        ),
        stop_policies=(PromptPackStopPolicy(name="default"),),
        supported_model_families=("llama", "mistral"),
    )


def test_valid_examples_certified() -> None:
    examples = (
        PackExample("basic", "llama", "chat", roles_used=("user", "assistant")),
        PackExample(
            "with_tool",
            "llama",
            "tool_chat",
            roles_used=("user", "assistant", "tool"),
            tools_called=("search",),
        ),
    )
    report = certify_examples(_pack(), examples)
    assert report.certified
    assert report.examples_checked == 2


def test_unsupported_family_rejected() -> None:
    examples = (PackExample("gpt_ex", "gpt", "chat", roles_used=("user",)),)
    report = certify_examples(_pack(), examples)
    assert not report.certified
    assert any(
        f.kind is ExampleFindingKind.UNSUPPORTED_MODEL_FAMILY for f in report.findings
    )


def test_template_family_mismatch_rejected() -> None:
    # mistral is a pack family but tool_chat template only supports llama
    examples = (
        PackExample("mismatch", "mistral", "tool_chat", roles_used=("user",)),
    )
    report = certify_examples(_pack(), examples)
    assert any(
        f.kind is ExampleFindingKind.TEMPLATE_FAMILY_MISMATCH for f in report.findings
    )


def test_unknown_template_rejected() -> None:
    examples = (PackExample("ghost", "llama", "nope", roles_used=("user",)),)
    report = certify_examples(_pack(), examples)
    assert any(
        f.kind is ExampleFindingKind.UNKNOWN_TEMPLATE for f in report.findings
    )


def test_undeclared_role_and_tool_rejected() -> None:
    examples = (
        PackExample(
            "bad",
            "llama",
            "chat",
            roles_used=("user", "root"),
            tools_called=("exfiltrate",),
        ),
    )
    report = certify_examples(_pack(), examples)
    kinds = {f.kind for f in report.findings}
    assert ExampleFindingKind.UNDECLARED_ROLE in kinds
    assert ExampleFindingKind.UNDECLARED_TOOL in kinds


def test_render_text() -> None:
    report = certify_examples(
        _pack(), (PackExample("gpt_ex", "gpt", "chat"),)
    )
    assert "REJECTED" in render_example_report_text(report)
