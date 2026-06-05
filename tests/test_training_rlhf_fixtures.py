import json
from pathlib import Path

from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader
from promptabi.session import VerificationSession


FIXTURE_DIR = Path("fixtures/training_rlhf").resolve()


def _config_for_fixture(tmp_path: Path, manifest_name: str) -> Path:
    config_path = tmp_path / f"{manifest_name}.promptabi.json"
    config_path.write_text(
        json.dumps(
            {
                "name": f"training-rlhf-{manifest_name}",
                "checks": ["static-contracts"],
                "artifacts": {
                    manifest_name: {
                        "kind": "training-manifest",
                        "path": str(FIXTURE_DIR / f"{manifest_name}.json"),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def test_rlhf_dpo_fixture_pack_loads_preference_metadata() -> None:
    config = load_config(FIXTURE_DIR / "promptabi.json")
    loader = ArtifactLoader()
    loaded = [loader.load(artifact) for artifact in config.artifact_bundle.artifacts]

    metadata_by_name = {artifact.artifact.name: dict(artifact.metadata) for artifact in loaded}

    assert set(metadata_by_name) == {
        "clean-dpo",
        "mask-truncation-defect",
        "prompt-hash-mismatch",
        "role-boundary-packing-defect",
    }
    assert all(metadata["preference_dataset_count"] == 1 for metadata in metadata_by_name.values())
    assert all(metadata["preference_pair_count"] == 1 for metadata in metadata_by_name.values())


def test_clean_dpo_fixture_proves_shared_prompt_layout_mask_and_packing(tmp_path: Path) -> None:
    result = VerificationSession(load_config(_config_for_fixture(tmp_path, "clean-dpo"))).run()
    preference_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if dict(diagnostic.properties).get("preference_pair_count") == "1"
    ]

    assert result.ok
    assert len(preference_diagnostics) == 1
    assert preference_diagnostics[0].rule_id == "static-contract-proved"
    assert "all declared chosen/rejected preference pairs share prompt prefix" in preference_diagnostics[0].message


def test_rlhf_dpo_fixtures_catch_known_alignment_mask_truncation_and_role_defects(tmp_path: Path) -> None:
    expected_reasons = {
        "prompt-hash-mismatch": {"prompt-prefix-hash-mismatch"},
        "mask-truncation-defect": {"mask-policy-mismatch", "preference-branch-truncated"},
        "role-boundary-packing-defect": {
            "role-layout-mismatch",
            "prompt-prefix-token-length-mismatch",
            "response-start-token-mismatch",
            "packed-example-boundary-mismatch",
        },
    }

    for manifest_name, reasons in expected_reasons.items():
        result = VerificationSession(load_config(_config_for_fixture(tmp_path, manifest_name))).run()
        violations = [diagnostic for diagnostic in result.diagnostics if diagnostic.rule_id == "static-contract-violation"]

        assert not result.ok, manifest_name
        assert len(violations) == 1
        properties = dict(violations[0].properties)
        assert properties["pair_id"]
        assert reasons.issubset(set(str(properties["reasons"]).split(", ")))
