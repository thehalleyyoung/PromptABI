from pathlib import Path

import pytest

from promptabi.artifacts import ArtifactKind, ArtifactLocation, TrainingManifestArtifact
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoadError, ArtifactLoader
from promptabi.session import VerificationSession
from promptabi.training_data_loaders import (
    REQUIRED_DATA_LOADER_FAMILIES,
    analyze_data_loader_adapters,
)


FIXTURE_ROOT = Path("fixtures/training_data_loaders")
MANIFEST = FIXTURE_ROOT / "training_loaders.training-manifest.json"


def test_training_data_loader_adapter_fixtures_cover_required_families() -> None:
    artifact = TrainingManifestArtifact(
        kind=ArtifactKind.TRAINING_MANIFEST,
        name="training-loaders",
        location=ArtifactLocation(path=str(MANIFEST)),
    )

    loaded = ArtifactLoader().load(artifact)
    metadata = dict(loaded.metadata)

    assert metadata["data_loader_adapter_count"] == 8
    assert metadata["data_loader_adapter_required_families_complete"] is True
    assert metadata["data_loader_adapter_families"] == REQUIRED_DATA_LOADER_FAMILIES
    assert metadata["data_loader_adapter_sample_count"] == 16
    assert metadata["data_loader_adapter_private_material"] is False


def test_training_data_loader_adapter_report_reads_real_fixture_files() -> None:
    import json

    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = analyze_data_loader_adapters(raw, base_dir=MANIFEST.parent)
    probes = {probe.fixture.name: probe for probe in report.probes}

    assert probes["jsonl-chat"].source_type == "jsonl-sample"
    assert probes["jsonl-chat"].observed_count == 2
    assert "messages" in probes["jsonl-chat"].observed_fields
    assert probes["parquet-chat-metadata"].source_type == "parquet-metadata"
    assert probes["hf-datasets-chat"].source_type == "huggingface-datasets-metadata"
    assert probes["trl-dpo"].fixture.preference_fields == ("chosen", "rejected")
    assert probes["openrlhf-preference"].observed_fields == (
        "chosen",
        "chosen_score",
        "prompt",
        "rejected",
        "rejected_score",
    )


def test_training_data_loader_adapter_manifest_runs_real_training_checks() -> None:
    result = VerificationSession(load_config(FIXTURE_ROOT / "promptabi.json")).run()

    assert result.ok
    assert any(diagnostic.rule_id == "training-invalid-interface-verified" for diagnostic in result.diagnostics)


def test_training_data_loader_adapter_rejects_bad_declared_fields(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"prompt":"hi","chosen":"ok","rejected":"bad"}\n', encoding="utf-8")
    manifest = tmp_path / "training.json"
    manifest.write_text(
        """
{
  "dataset_format": "bad-loader",
  "datasets": [],
  "metadata": {
    "data_loader_adapters": [
      {
        "name": "bad-jsonl",
        "family": "jsonl",
        "path": "sample.jsonl",
        "kind": "supervised",
        "content_fields": ["messages"],
        "example_count": 1
      }
    ]
  }
}
""",
        encoding="utf-8",
    )
    artifact = TrainingManifestArtifact(
        kind=ArtifactKind.TRAINING_MANIFEST,
        name="bad-training",
        location=ArtifactLocation(path=str(manifest)),
    )

    with pytest.raises(ArtifactLoadError, match="could not be parsed") as exc_info:
        ArtifactLoader().load(artifact)

    assert "missing declared fields: messages" in exc_info.value.steps[0][2]
