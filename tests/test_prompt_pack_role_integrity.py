"""Tests for prompt-pack role non-forgeability (step 247)."""

from __future__ import annotations

from promptabi.prompt_pack_role_integrity import (
    ConsumerInputChannel,
    PackRoleModel,
    RoleIntegrityFindingKind,
    RoleMarker,
    prove_role_nonforgeability,
    render_role_integrity_text,
)

MARKERS = (
    RoleMarker("system", "<|system|>", privileged=True),
    RoleMarker("assistant", "<|assistant|>", privileged=True),
    RoleMarker("tool", "<|tool|>", privileged=True),
    RoleMarker("user", "<|user|>", privileged=False),
)


def test_escaped_channel_is_nonforgeable() -> None:
    model = PackRoleModel(
        markers=MARKERS,
        channels=(
            ConsumerInputChannel(
                "user",
                escaped_markers=("<|system|>", "<|assistant|>", "<|tool|>"),
            ),
        ),
    )
    report = prove_role_nonforgeability(model)
    assert report.nonforgeable
    assert report.privileged_roles == ("assistant", "system", "tool")


def test_isolated_channel_is_nonforgeable() -> None:
    model = PackRoleModel(
        markers=MARKERS,
        channels=(ConsumerInputChannel("tool", structurally_isolated=True),),
    )
    report = prove_role_nonforgeability(model)
    assert report.nonforgeable


def test_unescaped_marker_is_forgeable() -> None:
    model = PackRoleModel(
        markers=MARKERS,
        channels=(
            ConsumerInputChannel("user", escaped_markers=("<|system|>",)),
        ),
    )
    report = prove_role_nonforgeability(model)
    assert not report.nonforgeable
    forgeable = [
        f for f in report.findings
        if f.kind is RoleIntegrityFindingKind.FORGEABLE_ROLE
    ]
    forged_roles = {f.target_role for f in forgeable}
    assert forged_roles == {"assistant", "tool"}
    # witness actually contains the marker that would be injected
    for f in forgeable:
        assert f.marker in f.forging_witness


def test_completely_unescaped_channel_forges_all() -> None:
    model = PackRoleModel(
        markers=MARKERS,
        channels=(ConsumerInputChannel("user"),),
    )
    report = prove_role_nonforgeability(model)
    forged = {
        f.target_role
        for f in report.findings
        if f.kind is RoleIntegrityFindingKind.FORGEABLE_ROLE
    }
    assert forged == {"system", "assistant", "tool"}


def test_undeclared_channel_role_flagged() -> None:
    model = PackRoleModel(
        markers=MARKERS,
        channels=(
            ConsumerInputChannel("ghost", structurally_isolated=True),
        ),
    )
    report = prove_role_nonforgeability(model)
    assert any(
        f.kind is RoleIntegrityFindingKind.UNDECLARED_CHANNEL_ROLE
        for f in report.findings
    )


def test_render_text() -> None:
    model = PackRoleModel(markers=MARKERS, channels=(ConsumerInputChannel("user"),))
    report = prove_role_nonforgeability(model)
    text = render_role_integrity_text(report)
    assert "FORGEABLE" in text
