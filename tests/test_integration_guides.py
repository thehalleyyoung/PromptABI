import re
from pathlib import Path

from promptabi.cli import main
from promptabi.init import available_stacks


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "integrations.md"


def test_integration_guide_is_linked_and_covers_required_stacks() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Integration guides: integrations.md" in mkdocs
    for heading in (
        "LangChain RAG",
        "LlamaIndex agents",
        "vLLM OpenAI-compatible servers",
        "llama.cpp and Ollama local servers",
        "Hugging Face Transformers",
        "OpenAI-compatible servers",
        "LiteLLM routers",
        "MCP tools",
        "Custom agent frameworks",
        "Training and fine-tuning data pipelines",
    ):
        assert f"## {heading}" in guide

    for stack in available_stacks():
        assert f"promptabi init --stack {stack}" in guide


def test_integration_guide_references_real_repository_artifacts() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    referenced_paths = sorted(set(re.findall(r"(?:examples|fixtures)/[-_./A-Za-z0-9]+\.json", guide)))

    assert referenced_paths
    assert "examples/minimal/promptabi.json" in referenced_paths
    assert "fixtures/provider_migration/litellm-target.json" in referenced_paths
    for relative_path in referenced_paths:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_integration_guide_smoke_commands_execute_against_real_cli(capsys) -> None:
    commands = (
        ["verify", "--config", "examples/minimal/promptabi.json"],
        ["verify", "--config", "examples/rag-chunking/promptabi.json", "--fail-on", "never"],
        ["verify", "--config", "examples/role-boundary/unsafe.promptabi.json", "--fail-on", "never"],
        ["verify", "--config", "fixtures/provider_migration/promptabi.json", "--fail-on", "never"],
    )

    for command in commands:
        exit_code = main(command)
        captured = capsys.readouterr()

        assert exit_code == 0, command
        assert "Traceback" not in captured.err


def test_integration_guide_scaffold_command_matches_real_stack(tmp_path, capsys) -> None:
    output_dir = tmp_path / "langchain-rag"

    init_exit = main(["init", "--stack", "langchain-rag", "--output-dir", str(output_dir)])
    init_output = capsys.readouterr()
    verify_exit = main(["verify", "--config", str(output_dir / "promptabi.json"), "--fail-on", "never"])
    verify_output = capsys.readouterr()

    assert init_exit == 0
    assert "wrote PromptABI langchain-rag scaffold" in init_output.out
    assert verify_exit == 0
    assert "PromptABI verification: langchain-rag-promptabi" in verify_output.out
