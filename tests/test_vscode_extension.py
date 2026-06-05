import json
import shutil
import subprocess
from pathlib import Path

from promptabi.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = REPO_ROOT / "editors" / "vscode"


def test_vscode_extension_manifest_contributes_real_promptabi_commands() -> None:
    manifest = json.loads((EXTENSION_ROOT / "package.json").read_text(encoding="utf-8"))

    commands = {command["command"] for command in manifest["contributes"]["commands"]}
    assert {
        "promptabi.quickCheck",
        "promptabi.explainDiagnostic",
        "promptabi.previewWitness",
    } <= commands
    assert "workspaceContains:**/promptabi.json" in manifest["activationEvents"]
    assert manifest["contributes"]["configuration"]["properties"]["promptabi.executable"]["default"] == "promptabi"


def test_vscode_extension_javascript_is_syntax_checked_when_node_is_available() -> None:
    node = shutil.which("node")
    if node is None:
        return

    for relative in ("extension.js", "src/promptabiCli.js"):
        subprocess.run([node, "--check", str(EXTENSION_ROOT / relative)], check=True)


def test_vscode_cli_helper_parses_real_role_boundary_witness(capsys) -> None:
    exit_code = main(
        [
            "diagnostics",
            "lsp",
            "--config",
            "examples/role-boundary/unsafe.promptabi.json",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    previews = [
        diagnostic["data"]["witness"]["steps"]
        for document in payload["documents"]
        for diagnostic in document["params"]["diagnostics"]
        if diagnostic["code"] == "role-boundary-nonforgeability" and "witness" in diagnostic["data"]
    ]
    assert previews
    assert any(
        step["action"] == "tokenize forged excerpt" and "<|im_start|>" in step["output"]
        for steps in previews
        for step in steps
    )

    helper_source = (EXTENSION_ROOT / "src" / "promptabiCli.js").read_text(encoding="utf-8")
    assert "collectWitnessPreviews" in helper_source
    assert "renderWitnessMarkdown" in helper_source
