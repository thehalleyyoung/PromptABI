import json
import importlib.util
from pathlib import Path

from promptabi.agent_frameworks import (
    AgentFrameworkIntegrationError,
    load_agent_prompt_pack_assembly,
    render_agent_prompt_pack_plan,
    write_agent_promptabi_config,
)
from promptabi.cli import main


EXAMPLE_ROOT = Path("examples/agent-frameworks")


def _verify_generated(spec_name: str, tmp_path: Path, capsys) -> tuple[int, dict[str, object]]:
    assembly = load_agent_prompt_pack_assembly(EXAMPLE_ROOT / spec_name)
    config_path = tmp_path / f"{assembly.name}.promptabi.json"
    write_agent_promptabi_config(assembly, config_path)

    exit_code = main(["verify", "--config", str(config_path), "--format", "json"])
    captured = capsys.readouterr()

    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def _rule_ids(payload: dict[str, object], *, severity: str | None = None) -> set[str]:
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, list)
    return {
        str(diagnostic["rule_id"])
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict) and (severity is None or diagnostic["severity"] == severity)
    }


def test_dynamic_agent_prompt_pack_safe_example_generates_verified_config(tmp_path, capsys) -> None:
    exit_code, payload = _verify_generated("safe.agent-prompt-pack.json", tmp_path, capsys)

    assert exit_code == 0
    assert _rule_ids(payload) == {"prompt-pack-verified"}


def test_dynamic_agent_prompt_pack_buggy_example_reports_runtime_drift(tmp_path, capsys) -> None:
    exit_code, payload = _verify_generated("buggy.agent-prompt-pack.json", tmp_path, capsys)

    assert exit_code == 1
    assert {
        "prompt-pack-model-family-unsupported",
        "prompt-pack-stop-missing",
        "prompt-pack-tool-missing",
    } <= _rule_ids(payload, severity="error")


def test_dynamic_agent_example_app_writes_real_promptabi_config(tmp_path, capsys) -> None:
    script_path = EXAMPLE_ROOT / "dynamic_support_agent.py"
    spec = importlib.util.spec_from_file_location("dynamic_support_agent", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "generated.promptabi.json"
    exit_code = module.main([str(EXAMPLE_ROOT / "safe.agent-prompt-pack.json"), "--write-config", str(output), "--preview"])
    captured = capsys.readouterr()
    generated = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "langchain-runnable-graph" in captured.out
    assert "Prompt preview" in captured.out
    assert generated["checks"] == ["prompt-pack-contracts"]
    assert generated["artifacts"]["support-pack"]["kind"] == "prompt-pack"


def test_dynamic_agent_prompt_pack_loader_rejects_unverifiable_assembly() -> None:
    broken = {
        "name": "broken",
        "framework": "custom-agent",
        "prompt_pack": {
            "path": "../prompt-packs/support.prompt-pack.json",
            "template": "support-chat",
            "version": "1.0.0",
        },
        "provider": "openai",
        "model_family": "openai-compatible",
        "segments": [{"name": "user-request", "role": "user", "required": True}],
        "dynamic_context_sources": [],
        "tools": ["refund_user"],
        "stops": ["</tool_call>"],
    }

    try:
        from promptabi.agent_frameworks import agent_prompt_pack_assembly_from_mapping

        agent_prompt_pack_assembly_from_mapping(broken, base_dir=EXAMPLE_ROOT)
    except AgentFrameworkIntegrationError as exc:
        assert "required region" in str(exc) or "omits role" in str(exc)
    else:
        raise AssertionError("unverifiable dynamic assembly should fail before config generation")
