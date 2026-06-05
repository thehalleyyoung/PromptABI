"""Tests for prompt-pack upgrade compatibility certification (step 241)."""

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
from promptabi.prompt_pack_capability import derive_capability_signature
from promptabi.prompt_pack_upgrade_certificate import (
    UpgradeFindingKind,
    UpgradeImpact,
    certify_upgrade,
    render_upgrade_json,
    render_upgrade_text,
)

LOCATION = ArtifactLocation(path="/tmp/pack.json")


def _pack(
    version: str,
    *,
    templates=("chat",),
    tools=(("search", True),),
    stop=("default",),
    families=("llama",),
) -> PromptPackArtifact:
    return PromptPackArtifact(
        kind=ArtifactKind.PROMPT_PACK,
        name="pack",
        location=LOCATION,
        pack_name="support-pack",
        pack_version=version,
        exported_templates=tuple(
            PromptPackTemplate(name=t, template="{{ x }}", roles=("system", "user"))
            for t in templates
        ),
        tool_schemas=tuple(
            PromptPackToolSchema(name=name, required=req) for name, req in tools
        ),
        stop_policies=tuple(PromptPackStopPolicy(name=s) for s in stop),
        supported_model_families=families,
    )


def _sig(pack):
    return derive_capability_signature(pack)


def test_pure_addition_is_minor_and_compatible() -> None:
    old = _sig(_pack("1.0.0"))
    new = _sig(_pack("1.1.0", templates=("chat", "summary")))
    cert = certify_upgrade(old, new)
    assert cert.compatible
    assert cert.impact is UpgradeImpact.MINOR
    assert any(
        f.kind is UpgradeFindingKind.ADDED_CAPABILITY for f in cert.findings
    )


def test_removed_template_is_breaking_major() -> None:
    old = _sig(_pack("1.0.0", templates=("chat", "summary")))
    new = _sig(_pack("2.0.0", templates=("chat",)))
    cert = certify_upgrade(old, new)
    assert not cert.compatible
    assert cert.impact is UpgradeImpact.MAJOR
    assert any(
        f.kind is UpgradeFindingKind.REMOVED_TEMPLATE and f.breaking
        for f in cert.findings
    )


def test_removed_required_tool_is_breaking() -> None:
    old = _sig(_pack("1.0.0", tools=(("search", True), ("calc", True))))
    new = _sig(_pack("2.0.0", tools=(("search", True),)))
    cert = certify_upgrade(old, new)
    assert not cert.compatible
    assert any(
        f.kind is UpgradeFindingKind.REMOVED_REQUIRED_TOOL for f in cert.findings
    )


def test_removed_optional_tool_is_not_breaking() -> None:
    old = _sig(_pack("1.0.0", tools=(("search", True), ("extra", False))))
    new = _sig(_pack("1.1.0", tools=(("search", True),)))
    cert = certify_upgrade(old, new)
    assert cert.compatible
    assert cert.impact is UpgradeImpact.MINOR
    assert any(
        f.kind is UpgradeFindingKind.REMOVED_OPTIONAL_TOOL for f in cert.findings
    )


def test_version_understatement_flagged() -> None:
    # breaking change shipped as a patch bump
    old = _sig(_pack("1.0.0", templates=("chat", "summary")))
    new = _sig(_pack("1.0.1", templates=("chat",)))
    cert = certify_upgrade(old, new)
    assert any(
        f.kind is UpgradeFindingKind.VERSION_UNDERSTATES_IMPACT for f in cert.findings
    )


def test_identical_packs_are_patch() -> None:
    old = _sig(_pack("1.0.0"))
    new = _sig(_pack("1.0.0"))
    cert = certify_upgrade(old, new)
    assert cert.impact is UpgradeImpact.PATCH
    assert cert.compatible


def test_render_round_trips() -> None:
    old = _sig(_pack("1.0.0"))
    new = _sig(_pack("1.1.0", templates=("chat", "summary")))
    cert = certify_upgrade(old, new)
    payload = json.loads(render_upgrade_json(cert))
    assert payload["impact"] == "minor"
    assert "upgrade certificate" in render_upgrade_text(cert)
