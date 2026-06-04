import json
import shutil
import subprocess
from pathlib import Path

from promptabi.cli import main
from promptabi.github_action import changed_promptabi_paths, relevant_promptabi_paths


def _copy_minimal_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    shutil.copytree("examples/minimal", project)
    return project


def _write_lockfile(config: Path, lockfile: Path, capsys) -> None:
    exit_code = main(
        [
            "verify",
            "--config",
            str(config),
            "--lockfile",
            str(lockfile),
            "--write-lockfile",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""


def test_github_action_cli_writes_sarif_summary_outputs_and_enforces_lockfile(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = _copy_minimal_project(tmp_path)
    config = project / "promptabi.json"
    lockfile = project / "promptabi.lock.json"
    _write_lockfile(config, lockfile, capsys)
    lock_payload = json.loads(lockfile.read_text(encoding="utf-8"))
    assert {artifact["location"] for artifact in lock_payload["artifacts"]} == {
        "answer.schema.json",
        "messages.json",
        "tools.json",
    }
    sarif = tmp_path / "promptabi.sarif"
    summary = tmp_path / "summary.md"
    output = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    exit_code = main(
        [
            "github-action",
            "--config",
            str(config),
            "--lockfile",
            str(lockfile),
            "--require-lockfile",
            "--repo-root",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--sarif-output",
            str(sarif),
            "--summary-output",
            str(summary),
            "--annotations",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI GitHub Action: PASS" in captured.out
    assert sarif.is_file()
    payload = json.loads(sarif.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["automationDetails"]["id"] == "promptabi/"
    assert "originalUriBaseIds" in run
    assert any(result["ruleId"] == "lockfile-verified" for result in run["results"])
    assert "PromptABI verification" in summary.read_text(encoding="utf-8")
    outputs = output.read_text(encoding="utf-8")
    assert "skipped=false" in outputs
    assert "ok=true" in outputs
    assert f"sarif={sarif}" in outputs


def test_github_action_changed_only_skips_unrelated_committed_changes(
    tmp_path: Path,
    capsys,
) -> None:
    project = _copy_minimal_project(tmp_path)
    config = project / "promptabi.json"
    lockfile = project / "promptabi.lock.json"
    _write_lockfile(config, lockfile, capsys)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "promptabi@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "PromptABI"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "README.md").write_text("# unrelated\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    sarif = tmp_path / "skip.sarif"
    summary = tmp_path / "skip.md"

    exit_code = main(
        [
            "github-action",
            "--config",
            str(config),
            "--lockfile",
            str(lockfile),
            "--require-lockfile",
            "--repo-root",
            str(tmp_path),
            "--changed-only",
            "--base-ref",
            base,
            "--head-ref",
            head,
            "--sarif-output",
            str(sarif),
            "--summary-output",
            str(summary),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "skipped" in captured.out
    assert json.loads(sarif.read_text(encoding="utf-8"))["runs"][0]["results"] == []
    assert "No configured PromptABI artifact changed" in summary.read_text(encoding="utf-8")
    assert changed_promptabi_paths(repo_root=tmp_path, base_ref=base, head_ref=head) == ("README.md",)


def test_relevant_promptabi_paths_include_config_lockfile_and_local_artifacts(tmp_path: Path) -> None:
    paths = relevant_promptabi_paths(
        config_path=tmp_path / "promptabi.json",
        lockfile_path=tmp_path / "promptabi.lock.json",
        artifact_paths=[str(tmp_path / "schema.json"), None],
        repo_root=tmp_path,
    )

    assert paths == ("promptabi.json", "promptabi.lock.json", "schema.json")
