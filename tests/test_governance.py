import json
from pathlib import Path

import promptabi
from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_governance_policy_validates_repository() -> None:
    report = promptabi.validate_governance(REPO_ROOT)

    assert report.ok
    assert report.release_blockers == promptabi.REQUIRED_RELEASE_BLOCKERS
    assert "docs/governance.md" in report.checked_paths
    assert {principle.id for principle in report.principles} == {
        "checker-acceptance",
        "proof-standards",
        "corpus-licensing",
        "security-disclosure",
        "release-regressions",
    }


def test_governance_cli_outputs_stable_json(capsys) -> None:
    exit_code = main(["governance", "--repo-root", str(REPO_ROOT), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["manifest_version"] == promptabi.GOVERNANCE_POLICY_VERSION
    assert payload["principle_count"] == 5
    assert "unsound-safe-result" in payload["release_blockers"]
    assert payload["principles"][0]["id"] == "checker-acceptance"


def test_governance_public_api_renders_text() -> None:
    text = promptabi.governance_policy(REPO_ROOT, output_format="text")

    assert "PromptABI governance: PASS" in text
    assert "checker-acceptance: Checker acceptance criteria" in text
    assert "witness-replay-failure" in text


def test_governance_validation_reports_missing_doc(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs" / "contributing").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "docs" / "contributing" / "checker-design.md").write_text(
        "supported fragment abstention witness release-blocking",
        encoding="utf-8",
    )
    (repo / "docs" / "contributing" / "corpus-contributions.md").write_text(
        "license provenance no secrets release-blocking",
        encoding="utf-8",
    )
    (repo / "docs" / "security-model.md").write_text(
        "Responsible disclosure private security advisory sanitized",
        encoding="utf-8",
    )
    (repo / "mkdocs.yml").write_text("governance.md\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "tests/test_governance.py docs/governance.md promptabi governance --format text",
        encoding="utf-8",
    )

    report = promptabi.validate_governance(repo)

    assert not report.ok
    assert any("docs/governance.md: missing governance document" == issue for issue in report.issues)
