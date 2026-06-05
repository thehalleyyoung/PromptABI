import json
from pathlib import Path

import promptabi
from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_maintainer_health_validates_repository() -> None:
    report = promptabi.validate_maintainer_health(REPO_ROOT)

    assert report.ok
    assert report.metrics["triage_label_count"] >= len(promptabi.REQUIRED_TRIAGE_LABELS)
    assert report.metrics["fixture_json_files"] > 0
    assert report.metrics["verification_config_count"] >= 10
    assert {item.id for item in report.rotation_roles} == {
        "rotation-release-captain",
        "rotation-corpus-steward",
        "rotation-triage-lead",
    }
    assert "docs/maintainer-health.md" in report.checked_paths


def test_maintainer_health_cli_outputs_stable_json(capsys) -> None:
    exit_code = main(["maintain", "health", "--repo-root", str(REPO_ROOT), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["manifest_version"] == promptabi.MAINTAINER_HEALTH_VERSION
    assert payload["rotation_roles"][0]["id"] == "rotation-release-captain"
    assert payload["metrics"]["triage_label_count"] >= len(promptabi.REQUIRED_TRIAGE_LABELS)


def test_maintainer_health_public_api_renders_text() -> None:
    text = promptabi.maintainer_health(REPO_ROOT, output_format="text")

    assert "PromptABI maintainer health: PASS" in text
    assert "rotation-release-captain: Release captain" in text
    assert "corpus-provenance-review: Provenance review" in text


def test_maintainer_health_reports_missing_doc(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs" / "contributing").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github").mkdir(exist_ok=True)
    (repo / "docs" / "governance.md").write_text("governance", encoding="utf-8")
    (repo / "docs" / "contributing" / "corpus-contributions.md").write_text("corpus", encoding="utf-8")
    (repo / ".github" / "labels.yml").write_text("", encoding="utf-8")
    (repo / "mkdocs.yml").write_text("", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text("", encoding="utf-8")

    report = promptabi.validate_maintainer_health(repo)

    assert not report.ok
    assert "docs/maintainer-health.md: missing maintainer-health document" in report.issues
    assert ".github/labels.yml: missing label status: needs-triage" in report.issues
