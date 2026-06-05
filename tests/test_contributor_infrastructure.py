import json
from pathlib import Path

import promptabi
from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_contributor_validation_passes_against_repository() -> None:
    report = promptabi.validate_contributor_infrastructure(REPO_ROOT)

    assert report.ok
    assert report.issue_count == 0
    assert ".github/ISSUE_TEMPLATE/bug_report.yml" in report.checked_paths
    assert ".github/labels.yml" in report.checked_paths
    assert "good first issue" in report.labels
    assert "checker_proposal.yml" in report.issue_templates


def test_contributor_validation_cli_outputs_stable_json(capsys) -> None:
    exit_code = main(["contribute", "validate", "--repo-root", str(REPO_ROOT), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["issue_count"] == 0
    assert "docs/contributing/plugin-author-guide.md" in payload["checked_paths"]
    assert "area: plugin" in payload["labels"]


def test_contributor_guides_are_linked_and_actionable() -> None:
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for relative_path in (
        "contributing/plugin-author-guide.md",
        "contributing/checker-design.md",
        "contributing/corpus-contributions.md",
    ):
        assert relative_path in mkdocs
        assert f"docs/{relative_path}" in contributing

    assert "promptabi contribute validate" in contributing
    assert "tests/test_contributor_infrastructure.py" in (
        REPO_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")


def test_contributor_validation_reports_missing_required_label(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".github" / "ISSUE_TEMPLATE").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "docs" / "contributing").mkdir(parents=True)

    for template_name in promptabi.REQUIRED_ISSUE_TEMPLATES:
        (repo / ".github" / "ISSUE_TEMPLATE" / template_name).write_text(
            "\n".join(
                (
                    "name: Template",
                    "description: Example",
                    "title: Example",
                    "labels: ['type: bug']",
                    "body:",
                    "  - type: textarea",
                    "    id: details",
                    "    validations:",
                    "      required: true",
                    "",
                )
            ),
            encoding="utf-8",
        )
    (repo / ".github" / "ISSUE_TEMPLATE" / "config.yml").write_text(
        "blank_issues_enabled: false\n",
        encoding="utf-8",
    )
    (repo / ".github" / "labels.yml").write_text(
        "- name: \"help wanted\"\n",
        encoding="utf-8",
    )
    (repo / "CONTRIBUTING.md").write_text(
        "deterministic CPU-only promptabi contribute validate\n",
        encoding="utf-8",
    )
    (repo / "docs" / "contributing" / "plugin-author-guide.md").write_text(
        "PluginRegistry privacy tests/test_contributor_infrastructure.py\n",
        encoding="utf-8",
    )
    (repo / "docs" / "contributing" / "checker-design.md").write_text(
        "CheckMode witness abstention\n",
        encoding="utf-8",
    )
    (repo / "docs" / "contributing" / "corpus-contributions.md").write_text(
        "provenance license no secrets\n",
        encoding="utf-8",
    )
    (repo / "mkdocs.yml").write_text(
        "\n".join(
            (
                "contributing/plugin-author-guide.md",
                "contributing/checker-design.md",
                "contributing/corpus-contributions.md",
            )
        ),
        encoding="utf-8",
    )
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "\n".join(
            (
                "python -m pytest tests/test_contributor_infrastructure.py",
                "docs/contributing/**",
                ".github/ISSUE_TEMPLATE/**",
                ".github/labels.yml",
            )
        ),
        encoding="utf-8",
    )

    report = promptabi.validate_contributor_infrastructure(repo)

    assert not report.ok
    assert any(issue.check == "required-label" and "good first issue" in issue.message for issue in report.issues)
