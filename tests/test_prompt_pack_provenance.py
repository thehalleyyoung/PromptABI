"""Tests for prompt-pack provenance vs model registries (step 254)."""

from __future__ import annotations

from promptabi.prompt_pack_provenance import (
    ModelRegistrySnapshot,
    PackProvenance,
    PinnedModel,
    ProvenanceFindingKind,
    RegistryModel,
    render_provenance_text,
    verify_provenance,
)

SNAPSHOT = ModelRegistrySnapshot(
    registry="hf",
    models=(
        RegistryModel("meta-llama/Llama-3.1-8B-Instruct", "rev1", "d-llama"),
        RegistryModel("Qwen/Qwen2.5-7B-Instruct", "rev2", "d-qwen"),
    ),
)

PROVENANCE = PackProvenance(
    pack="support-triage",
    version="1.2.0",
    pack_digest="d-pack",
    registry="hf",
    models=(
        PinnedModel("meta-llama/Llama-3.1-8B-Instruct", "rev1", "d-llama"),
        PinnedModel("Qwen/Qwen2.5-7B-Instruct", "rev2", "d-qwen"),
    ),
)


def test_valid_chain() -> None:
    result = verify_provenance(PROVENANCE, SNAPSHOT, "d-pack")
    assert result.valid, result.findings


def test_pack_digest_mismatch() -> None:
    result = verify_provenance(PROVENANCE, SNAPSHOT, "tampered")
    assert any(
        f.kind is ProvenanceFindingKind.PACK_DIGEST_MISMATCH for f in result.findings
    )


def test_model_not_in_registry() -> None:
    prov = PackProvenance(
        "p", "1.0", "d-pack", "hf", (PinnedModel("ghost/model", "r", "d"),)
    )
    result = verify_provenance(prov, SNAPSHOT, "d-pack")
    assert any(
        f.kind is ProvenanceFindingKind.MODEL_NOT_IN_REGISTRY for f in result.findings
    )


def test_revision_mismatch_unless_known_good() -> None:
    bad = PackProvenance(
        "p",
        "1.0",
        "d-pack",
        "hf",
        (PinnedModel("Qwen/Qwen2.5-7B-Instruct", "rev2", "d-old"),),
    )
    result = verify_provenance(bad, SNAPSHOT, "d-pack")
    assert any(
        f.kind is ProvenanceFindingKind.MODEL_REVISION_MISMATCH for f in result.findings
    )

    okay = PackProvenance(
        "p",
        "1.0",
        "d-pack",
        "hf",
        (PinnedModel("Qwen/Qwen2.5-7B-Instruct", "rev2", "d-old", known_good_prior=True),),
    )
    assert verify_provenance(okay, SNAPSHOT, "d-pack").valid


def test_no_models_pinned() -> None:
    prov = PackProvenance("p", "1.0", "d-pack", "hf", ())
    result = verify_provenance(prov, SNAPSHOT, "d-pack")
    assert any(
        f.kind is ProvenanceFindingKind.NO_MODELS_PINNED for f in result.findings
    )


def test_render_text_smoke() -> None:
    result = verify_provenance(PROVENANCE, SNAPSHOT, "d-pack")
    assert "provenance" in render_provenance_text(result)
