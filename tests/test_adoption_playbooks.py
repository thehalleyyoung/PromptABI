import argparse
import json
from pathlib import Path

from promptabi.adoption_playbooks import (
    ADOPTION_AUDIENCES,
    AdoptionPlaybookError,
    build_adoption_playbook_report,
    render_adoption_playbooks_json,
    render_adoption_playbooks_markdown,
    render_adoption_playbooks_text,
    write_adoption_playbooks,
)
from promptabi.cli import build_parser, main
from promptabi.evaluation import run_evaluation
from promptabi.real_bug_benchmarks import build_real_bug_benchmark_manifest


def test_adoption_playbooks_are_backed_by_live_evidence() -> None:
    report = build_adoption_playbook_report()
    real_bug_manifest = build_real_bug_benchmark_manifest()
    evaluation = run_evaluation().to_dict()

    assert tuple(playbook.audience for playbook in report.playbooks) == ADOPTION_AUDIENCES
    assert report.evidence["real_bug_cases"] == real_bug_manifest["case_count"]
    assert report.evidence["real_bug_manifest_sha256"] == real_bug_manifest["manifest_sha256"]
    assert report.evidence["evaluation_cases"] == evaluation["case_count"]
    assert report.evidence["evaluation_precision"] == evaluation["score"]["precision"]
    assert report.evidence["all_real_bug_cases_passed"] is True
    assert report.evidence["evaluation_passed"] is True
    assert "no model weights" in str(report.evidence["privacy_posture"])


def test_adoption_playbook_renderers_and_commands_are_real() -> None:
    report = build_adoption_playbook_report()
    text = render_adoption_playbooks_text(report)
    markdown = render_adoption_playbooks_markdown(report)
    payload = json.loads(render_adoption_playbooks_json(report))
    parser = build_parser()

    assert "PromptABI adoption playbooks" in text
    assert "Startup launch gate" in markdown
    assert payload["report_sha256"] == report.report_sha256
    assert payload["audience_count"] == 5
    for playbook in report.playbooks:
        for command in playbook.commands:
            assert command.startswith("promptabi ")
            _assert_cli_path_exists(parser, command)


def test_adoption_playbook_writer_and_cli_create_expected_files(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "adoption"
    bundle = write_adoption_playbooks(output_dir)

    assert sorted(path.name for path in bundle.written_files) == [
        "README.md",
        "adoption-playbooks.json",
        "enterprise-ai-platforms.md",
        "model-hosting-providers.md",
        "open-source-agent-projects.md",
        "research-labs.md",
        "startups.md",
    ]
    assert json.loads((output_dir / "adoption-playbooks.json").read_text(encoding="utf-8"))["report_sha256"]
    assert "Provider compatibility lab" in (output_dir / "model-hosting-providers.md").read_text(encoding="utf-8")

    try:
        write_adoption_playbooks(output_dir)
    except AdoptionPlaybookError as exc:
        assert "pass --force" in str(exc)
    else:
        raise AssertionError("expected existing adoption playbook directory to require --force")

    cli_dir = tmp_path / "cli-adoption"
    exit_code = main(["adoption-playbooks", "--output-dir", str(cli_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "PromptABI adoption playbooks" in captured.out
    assert "audiences: 5" in captured.out
    assert (cli_dir / "adoption-playbooks.json").is_file()


def _assert_cli_path_exists(parser: argparse.ArgumentParser, command: str) -> None:
    tokens = command.split()[1:]
    current = parser
    for token in tokens:
        subparsers = [
            action
            for action in current._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        if not subparsers:
            return
        choices = subparsers[0].choices
        if token.startswith("-"):
            return
        assert token in choices
        current = choices[token]
