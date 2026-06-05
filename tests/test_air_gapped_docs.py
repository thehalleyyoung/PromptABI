import json
from pathlib import Path

from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "air-gapped.md"


def test_air_gapped_guide_is_linked_and_covers_required_surfaces() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "Air-gapped installation: air-gapped.md" in mkdocs
    assert "air-gapped installation guide" in readme
    assert "air-gapped installation" in quickstart
    assert "air-gapped installation" in docs_index

    for phrase in (
        "vendor/wheelhouse",
        "z3-solver==4.15.4",
        "fixtures/seed_corpus",
        "fixtures/provider_fixture_packs",
        "prompt-pack-mirror.json",
        "provider-fixture-manifest",
        "paper_artifact/reproduction_commands.sh",
        "--no-index --find-links",
        "artifact-provenance",
    ):
        assert phrase in guide


def test_air_gapped_documented_gates_execute_against_real_artifacts(tmp_path: Path, capsys) -> None:
    provider_manifest = tmp_path / "provider-fixture-manifest.json"
    assert main(["corpus", "provider-fixture-manifest", "--output", str(provider_manifest)]) == 0
    provider_output = capsys.readouterr()
    provider_payload = json.loads(provider_manifest.read_text(encoding="utf-8"))
    assert provider_output.err == ""
    assert provider_payload["entry_count"] >= 6
    assert all(entry["download_required"] is False for entry in provider_payload["entries"])
    assert all(entry["secrets_included"] is False for entry in provider_payload["entries"])

    mirror_dir = tmp_path / "prompt-pack-mirror"
    assert (
        main(
            [
                "prompt-pack",
                "mirror",
                "build",
                "--config",
                "examples/prompt-packs/promptabi.json",
                "--mirror-dir",
                str(mirror_dir),
                "--format",
                "json",
            ]
        )
        == 0
    )
    mirror_payload = json.loads(capsys.readouterr().out)
    assert mirror_payload["mirror_version"] == 1
    assert (mirror_dir / "prompt-pack-mirror.json").is_file()

    assert (
        main(
            [
                "prompt-pack",
                "mirror",
                "verify",
                "--manifest",
                str(mirror_dir / "prompt-pack-mirror.json"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    mirror_verification = json.loads(capsys.readouterr().out)
    assert mirror_verification["ok"] is True
    assert mirror_verification["diagnostics"][0]["rule_id"] == "prompt-pack-mirror-verified"

    artifact_dir = tmp_path / "paper_artifact"
    assert main(["paper", "reproducibility", "--output-dir", str(artifact_dir), "--benchmark-iterations", "1"]) == 0
    artifact_output = capsys.readouterr()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    fixture_hashes = json.loads((artifact_dir / "fixture_hashes.json").read_text(encoding="utf-8"))
    environment = json.loads((artifact_dir / "environment.json").read_text(encoding="utf-8"))
    assert artifact_output.err == ""
    assert manifest["summary"]["fixture_file_count"] == fixture_hashes["summary"]["file_count"]
    assert environment["solver"]["reproduction_pin"].startswith("z3-solver==")
