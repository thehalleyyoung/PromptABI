import json
import shutil
import subprocess
from pathlib import Path

from promptabi.cli import main
from promptabi.github_action import changed_promptabi_paths, relevant_promptabi_paths, training_pr_relevant_paths


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


def test_training_pr_relevant_paths_include_manifest_dataset_and_alignment_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "sft.training-manifest.json"
    dataset = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer_config.json"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    tokenizer.write_text('{"chat_template": "{{ messages }}"}', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "dataset_format": "chat-jsonl",
                "datasets": [
                    {
                        "name": "sft",
                        "kind": "supervised",
                        "path": dataset.name,
                        "format": "chat-jsonl",
                    }
                ],
                "loss_mask_policy": {"strategy": "assistant-only", "target_roles": ["assistant"]},
                "packing_window": {"strategy": "sample-packing", "max_tokens": 128},
                "chat_template_version": {
                    "name": "quickstart-template",
                    "tokenizer_name": "quickstart-tokenizer",
                    "add_generation_prompt": False,
                },
                "pipeline_stages": [
                    {
                        "stage": "training",
                        "tokenizer_name": "quickstart-tokenizer",
                        "chat_template_name": "quickstart-template",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    paths = training_pr_relevant_paths(
        manifest_path=manifest,
        tokenizers=(("quickstart-tokenizer", tokenizer),),
        chat_templates=(("quickstart-template", tokenizer),),
    )

    assert paths == tuple(sorted({manifest.resolve().as_posix(), dataset.resolve().as_posix(), tokenizer.resolve().as_posix()}))


def test_github_action_training_manifest_writes_sarif_summary_and_watches_dataset_changes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    shutil.copytree("examples/end-to-end/training-quickstart", project)
    manifest = project / "fixed.training-manifest.json"
    tokenizer = project / "tokenizer_config.json"
    sarif = tmp_path / "training.sarif"
    summary = tmp_path / "training.md"
    output = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    exit_code = main(
        [
            "github-action",
            "--training-manifest",
            str(manifest),
            "--tokenizer",
            f"quickstart-tokenizer={tokenizer}",
            "--chat-template",
            f"quickstart-chat-template={tokenizer}",
            "--repo-root",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--sarif-output",
            str(sarif),
            "--summary-output",
            str(summary),
            "--fail-on",
            "error",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(sarif.read_text(encoding="utf-8"))
    rule_ids = {result["ruleId"] for result in payload["runs"][0]["results"]}
    watched = summary.read_text(encoding="utf-8")
    outputs = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "PromptABI GitHub Action: PASS" in captured.out
    assert "training-workflow-verified" in rule_ids
    assert "project/fixed.training-manifest.json" in watched
    assert "project/train.jsonl" in watched
    assert "skipped=false" in outputs
    assert "ok=true" in outputs


def test_training_data_workflow_example_uses_dedicated_training_gate() -> None:
    workflow = Path(".github/workflows/promptabi-training-data.yml").read_text(encoding="utf-8")
    action = Path(".github/actions/promptabi/action.yml").read_text(encoding="utf-8")

    assert "**/*.training-manifest.json" in workflow
    assert "**/*.jsonl" in workflow
    assert "training-manifest: examples/end-to-end/training-quickstart/fixed.training-manifest.json" in workflow
    assert 'require-lockfile: "false"' in workflow
    assert "--training-manifest" in action
    assert "promptabi verify-training" in action
    assert '[[ -z "$PROMPTABI_TRAINING_MANIFEST" && "$PROMPTABI_REQUIRE_LOCKFILE" == "true" ]]' in action
