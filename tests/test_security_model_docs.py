from pathlib import Path

from promptabi.cli import build_parser, main
from promptabi.usage_analytics import privacy_guarantees


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_security_model_covers_real_privacy_and_fixture_guarantees() -> None:
    security_model = (REPO_ROOT / "docs" / "security-model.md").read_text(encoding="utf-8")
    guarantees = " ".join(privacy_guarantees())

    required_phrases = [
        "structural prompt-interface risks",
        "Non-goals",
        "Secret handling and artifact privacy",
        "Solver-input privacy",
        "Provider fixture safety",
        "Responsible disclosure workflow",
        "promptabi bug-report",
        "promptabi minimize",
        "promptabi usage privacy",
        "credential-like",
        "does not send these formulas to a remote solver",
    ]
    for phrase in required_phrases:
        assert phrase in security_model

    assert "No telemetry is sent" in guarantees
    assert "Prompts, schemas, configs, constraints, witnesses" in guarantees
    assert "No telemetry is sent" in security_model or "not telemetry" in security_model


def test_security_model_mentions_only_existing_security_workflow_commands(capsys) -> None:
    parser = build_parser()
    command_names = set(parser._subparsers._group_actions[0].choices)

    for command in ("verify", "explain", "bug-report", "minimize", "usage"):
        assert command in command_names

    exit_code = main(["usage", "privacy"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No telemetry is sent" in captured.out
    assert "network sends" in captured.out
    assert captured.err == ""


def test_security_model_is_linked_from_docs_and_readme() -> None:
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Security model: security-model.md" in mkdocs
    assert "security model" in docs_index.lower()
    assert "security model" in quickstart.lower()
    assert "security model" in readme.lower()
