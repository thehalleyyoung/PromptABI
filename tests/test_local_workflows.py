import os
import shutil
import stat
import subprocess
from pathlib import Path

from promptabi.cli import main
from promptabi.local_workflows import is_promptabi_candidate_path


def _copy_minimal_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    shutil.copytree("examples/minimal", project)
    return project


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "promptabi@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "PromptABI"], cwd=repo, check=True)


def _commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, stdout=subprocess.PIPE)


def test_pre_commit_install_writes_executable_python_module_hook(tmp_path: Path, capsys) -> None:
    project = _copy_minimal_project(tmp_path)
    _init_git_repo(project)

    exit_code = main(["pre-commit", "install", "--config", str(project / "promptabi.json"), "--repo-root", str(project)])

    captured = capsys.readouterr()
    hook = project / ".git" / "hooks" / "pre-commit"
    content = hook.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "installed PromptABI pre-commit hook" in captured.out
    assert "PromptABI managed pre-commit hook" in content
    assert "-m promptabi pre-commit run" in content
    assert "--changed-only" in content
    assert os.access(hook, os.X_OK)
    assert hook.stat().st_mode & stat.S_IXUSR
    assert captured.err == ""


def test_pre_commit_run_skips_unrelated_staged_changes(tmp_path: Path, capsys) -> None:
    project = _copy_minimal_project(tmp_path)
    _init_git_repo(project)
    _commit_all(project, "initial")
    (project / "README.md").write_text("# docs only\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=project, check=True)

    exit_code = main(
        [
            "pre-commit",
            "run",
            "--config",
            str(project / "promptabi.json"),
            "--repo-root",
            str(project),
            "--changed-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PromptABI pre-commit: skipped" in captured.out
    assert "changed paths checked for relevance: 1" in captured.out
    assert captured.err == ""


def test_pre_commit_run_blocks_staged_configured_artifact_drift(tmp_path: Path, capsys) -> None:
    project = _copy_minimal_project(tmp_path)
    _init_git_repo(project)
    _commit_all(project, "initial")
    schema = project / "answer.schema.json"
    schema.write_text('{"type": "object", "required": ["answer"]}\n', encoding="utf-8")
    subprocess.run(["git", "add", "answer.schema.json"], cwd=project, check=True)

    exit_code = main(
        [
            "pre-commit",
            "run",
            "--config",
            str(project / "promptabi.json"),
            "--repo-root",
            str(project),
            "--changed-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PromptABI pre-commit: FAIL" in captured.out
    assert "changed PromptABI inputs:" in captured.out
    assert "answer.schema.json" in captured.out
    assert captured.err == ""


def test_pre_commit_run_fails_closed_when_selected_staged_artifact_has_unstaged_edits(
    tmp_path: Path,
    capsys,
) -> None:
    project = _copy_minimal_project(tmp_path)
    _init_git_repo(project)
    _commit_all(project, "initial")
    schema = project / "answer.schema.json"
    schema.write_text('{"type": "object"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "answer.schema.json"], cwd=project, check=True)
    schema.write_text('{"type": "array"}\n', encoding="utf-8")

    exit_code = main(
        [
            "pre-commit",
            "run",
            "--config",
            str(project / "promptabi.json"),
            "--repo-root",
            str(project),
            "--changed-only",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "working-tree copies differ" in captured.err
    assert "answer.schema.json" in captured.err


def test_promptabi_candidate_classifier_covers_local_artifact_families() -> None:
    candidates = [
        "promptabi.json",
        "schemas/answer.schema.json",
        "models/tokenizer_config.json",
        "prompts/chat_template.jinja",
        "tools/weather-tools.json",
        "training/training-manifest.json",
        "grammars/tool_call.ebnf",
    ]

    assert all(is_promptabi_candidate_path(path) for path in candidates)
    assert not is_promptabi_candidate_path("docs/README.md")
