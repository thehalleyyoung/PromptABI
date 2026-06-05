import json
import shutil
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.release import (
    ReleaseReadinessStatus,
    build_release_readiness_report,
    render_release_readiness_json,
    render_release_readiness_text,
)


def test_release_readiness_gate_passes_against_live_repository() -> None:
    report = build_release_readiness_report(expected_version="1.0.0")

    assert report.ok
    assert report.version == "1.0.0"
    assert {check.name for check in report.checks} == {
        "version-metadata",
        "changelog",
        "readme",
        "stable-cli",
        "github-action-and-release",
        "docs-site",
        "seed-corpus",
        "formal-checks",
        "real-bug-benchmark",
        "reproducibility-package",
        "paper-preprint",
    }
    assert all(check.status is ReleaseReadinessStatus.PASS for check in report.checks)


def test_release_readiness_renderers_and_public_api_are_stable() -> None:
    report = build_release_readiness_report(expected_version="1.0.0")
    text = render_release_readiness_text(report)
    payload = json.loads(render_release_readiness_json(report))
    api_payload = json.loads(promptabi.release_readiness(output_format="json"))

    assert "PromptABI release readiness" in text
    assert "stable-cli: PASS" in text
    assert payload["ok"] is True
    assert payload["version"] == "1.0.0"
    assert api_payload["checks"] == payload["checks"]


def test_release_readiness_cli_returns_success_and_writes_json(tmp_path: Path, capsys) -> None:
    output = tmp_path / "release-readiness.json"
    exit_code = main(["release", "readiness", "--format", "json", "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "wrote release-readiness report" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_release_readiness_fails_on_version_mismatch_without_mutating_repo() -> None:
    report = build_release_readiness_report(expected_version="9.9.9")
    by_name = {check.name: check for check in report.checks}

    assert not report.ok
    assert by_name["version-metadata"].status is ReleaseReadinessStatus.FAIL
    assert by_name["changelog"].status is ReleaseReadinessStatus.FAIL
    assert all(check.passed for name, check in by_name.items() if name not in {"version-metadata", "changelog"})


def test_release_readiness_detects_missing_changelog_entry_in_copy(tmp_path: Path) -> None:
    repo = Path.cwd()
    copied = tmp_path / "repo"
    shutil.copytree(
        repo,
        copied,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc"),
    )
    changelog = copied / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\nPromptABI follows semantic versioning.\n", encoding="utf-8")

    report = build_release_readiness_report(copied, expected_version="1.0.0")
    by_name = {check.name: check for check in report.checks}

    assert not report.ok
    assert by_name["changelog"].status is ReleaseReadinessStatus.FAIL
