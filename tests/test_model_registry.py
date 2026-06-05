import json

from promptabi import (
    MODEL_REGISTRY_MANIFEST_VERSION,
    ModelRegistryKind,
    build_model_registry_publication,
    load_model_registry_targets,
    render_model_registry_publication_json,
    render_model_registry_publication_text,
)
from promptabi.cli import main


def test_model_registry_targets_cover_deployment_registry_families() -> None:
    targets = load_model_registry_targets("examples/model-registries/targets.json")

    assert {target.kind for target in targets} == {
        ModelRegistryKind.HUGGING_FACE_HUB,
        ModelRegistryKind.INTERNAL,
        ModelRegistryKind.MLFLOW,
        ModelRegistryKind.ARTIFACT_REPOSITORY,
    }
    assert all(target.instructions for target in targets)


def test_model_registry_publication_uses_real_verification_evidence() -> None:
    publication = build_model_registry_publication(
        "examples/model-registries/promptabi.json",
        targets="examples/model-registries/targets.json",
        bundle_key="registry-test-key",
        bundle_key_id="registry-test",
    )
    payload = json.loads(render_model_registry_publication_json(publication))

    assert publication.ok is True
    assert payload["manifest_version"] == MODEL_REGISTRY_MANIFEST_VERSION
    assert payload["registry_evidence"]["signed_bundle"]["available"] is True
    assert payload["registry_evidence"]["signed_bundle"]["signing_key_id"] == "registry-test"
    assert payload["registry_evidence"]["reproducibility_hash"]
    assert len(payload["targets"]) == 4
    assert "huggingface-cli upload" in json.dumps(payload)
    assert "mlflow artifacts log-artifact" in json.dumps(payload)
    assert "registryctl attach-attestation" in json.dumps(payload)
    assert "oras attach" in json.dumps(payload)


def test_model_registry_publication_blocks_unsigned_evidence() -> None:
    publication = build_model_registry_publication(
        "examples/model-registries/promptabi.json",
        targets="examples/model-registries/targets.json",
    )

    assert publication.ok is False
    assert publication.publish_blockers == (
        "signed verification bundle evidence is required for registry publication",
    )


def test_model_registry_cli_renders_json_and_text(capsys) -> None:
    exit_code = main(
        [
            "model-registry",
            "--config",
            "examples/model-registries/promptabi.json",
            "--targets",
            "examples/model-registries/targets.json",
            "--bundle-key",
            "registry-cli-key",
            "--bundle-key-id",
            "registry-cli",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["registry_evidence"]["signed_bundle"]["signing_key_id"] == "registry-cli"
    assert captured.err == ""

    text_publication = build_model_registry_publication(
        "examples/model-registries/promptabi.json",
        targets="examples/model-registries/targets.json",
        bundle_key="registry-text-key",
    )
    rendered_text = render_model_registry_publication_text(text_publication)
    assert "PromptABI model-registry publication" in rendered_text
    assert "hugging-face-hub hf-hub-model-card" in rendered_text
