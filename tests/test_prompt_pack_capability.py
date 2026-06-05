"""Tests for prompt-pack capability signatures (step 240)."""

from __future__ import annotations

import json

from promptabi.artifacts import (
    ArtifactKind,
    ArtifactLocation,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
)
from promptabi.prompt_pack_capability import (
    CapabilityFindingKind,
    CapabilityRequirement,
    derive_capability_signature,
    match_capability,
    render_capability_json,
    render_capability_match_text,
)

LOCATION = ArtifactLocation(path="/tmp/pack.json")


def _pack(version: str = "1.0.0") -> PromptPackArtifact:
    return PromptPackArtifact(
        kind=ArtifactKind.PROMPT_PACK,
        name="pack",
        location=LOCATION,
        pack_name="support-pack",
        pack_version=version,
        exported_templates=(
            PromptPackTemplate(
                name="chat",
                template="{{ messages }}",
                roles=("system", "user", "assistant"),
                supported_model_families=("llama",),
            ),
        ),
        expected_roles=("system", "user"),
        tool_schemas=(
            PromptPackToolSchema(name="search", required=True),
            PromptPackToolSchema(name="optional_tool", required=False),
        ),
        stop_policies=(PromptPackStopPolicy(name="default"),),
        supported_model_families=("llama", "mistral"),
    )


def test_signature_collects_capabilities() -> None:
    sig = derive_capability_signature(_pack())
    assert sig.templates == ("chat",)
    assert set(sig.roles) == {"system", "user", "assistant"}
    assert sig.tools == ("optional_tool", "search")
    assert sig.required_tools == ("search",)
    assert sig.stop_policies == ("default",)
    assert set(sig.model_families) == {"llama", "mistral"}


def test_digest_is_stable_and_order_independent() -> None:
    sig1 = derive_capability_signature(_pack())
    sig2 = derive_capability_signature(_pack())
    assert sig1.digest == sig2.digest


def test_matching_requirement_is_satisfied() -> None:
    sig = derive_capability_signature(_pack())
    req = CapabilityRequirement(
        templates=("chat",),
        roles=("system", "assistant"),
        tools=("search",),
        stop_policies=("default",),
        model_families=("llama",),
    )
    match = match_capability(sig, req)
    assert match.satisfied
    assert not match.findings


def test_missing_capabilities_reported() -> None:
    sig = derive_capability_signature(_pack())
    req = CapabilityRequirement(
        templates=("nonexistent",),
        roles=("tool",),
        tools=("calculator",),
        stop_policies=("strict",),
        model_families=("gpt",),
    )
    match = match_capability(sig, req)
    assert not match.satisfied
    kinds = {f.kind for f in match.findings}
    assert CapabilityFindingKind.MISSING_TEMPLATE in kinds
    assert CapabilityFindingKind.MISSING_ROLE in kinds
    assert CapabilityFindingKind.MISSING_TOOL in kinds
    assert CapabilityFindingKind.MISSING_STOP_POLICY in kinds
    assert CapabilityFindingKind.MISSING_MODEL_FAMILY in kinds


def test_render_round_trips() -> None:
    sig = derive_capability_signature(_pack())
    payload = json.loads(render_capability_json(sig))
    assert payload["pack_name"] == "support-pack"
    assert "digest" in payload
    match = match_capability(sig, CapabilityRequirement(templates=("chat",)))
    assert "SATISFIED" in render_capability_match_text(match)
