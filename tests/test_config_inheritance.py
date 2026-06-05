import json
from pathlib import Path

import pytest

from promptabi.config import ConfigError, load_config
from promptabi.loaders import load_artifact
from promptabi.static_contracts import analyze_static_contracts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_config_extends_resolves_parent_paths_and_tracks_obligation_lineage(tmp_path: Path) -> None:
    parent_dir = tmp_path / "base"
    child_dir = tmp_path / "service"
    schema = parent_dir / "schema.json"
    prompt = child_dir / "prompt.json"
    _write_json(schema, {})
    _write_json(
        prompt,
        {
            "segments": [
                {"name": "system", "role": "system", "required": True, "token_count": 12},
                {"name": "user", "role": "user", "required": True, "token_count": 4},
            ]
        },
    )
    base_config = parent_dir / "base.promptabi.json"
    child_config = child_dir / "promptabi.json"
    _write_json(
        base_config,
        {
            "name": "base",
            "checks": ["z3-static-contracts"],
            "max_context_tokens": 64,
            "artifacts": {"schema": "schema.json"},
        },
    )
    _write_json(
        child_config,
        {
            "name": "child",
            "extends": "../base/base.promptabi.json",
            "checks": ["token-budget-model"],
            "artifacts": {
                "prompt": {
                    "kind": "prompt-segment",
                    "path": "prompt.json",
                    "segments": [{"name": "declared", "role": "system", "required": True, "token_count": 1}],
                }
            },
        },
    )

    config = load_config(child_config)

    assert config.name == "child"
    assert config.max_context_tokens == 64
    assert config.artifacts["schema"] == str(schema.resolve())
    assert config.artifacts["prompt"] == str(prompt.resolve())
    assert config.checks == ("token-budget-model", "z3-static-contracts")
    assert {Path(source.path).name for source in config.inheritance_sources} == {
        "base.promptabi.json",
        "promptabi.json",
    }
    lineage = [item.to_dict() for item in config.proof_obligation_lineage]
    assert {
        (Path(item["source_path"]).name, item["field"], item["obligation"])
        for item in lineage
    } >= {
        ("base.promptabi.json", "max_context_tokens", "prompt-segment-budget"),
        ("base.promptabi.json", "artifacts.schema", "grammar-tokenizer-emptiness"),
        ("promptabi.json", "artifacts.prompt", "prompt-segment-budget"),
    }


def test_config_extends_child_artifact_override_is_marked(tmp_path: Path) -> None:
    parent_prompt = tmp_path / "parent" / "prompt.json"
    child_prompt = tmp_path / "child" / "prompt.json"
    _write_json(parent_prompt, {"segments": [{"name": "old", "required": True, "token_count": 9}]})
    _write_json(child_prompt, {"segments": [{"name": "new", "required": True, "token_count": 3}]})
    base_config = tmp_path / "parent" / "base.promptabi.json"
    child_config = tmp_path / "child" / "promptabi.json"
    _write_json(
        base_config,
        {
            "name": "base",
            "artifacts": {
                "prompt": {
                    "kind": "prompt-segment",
                    "path": "prompt.json",
                    "segments": [{"name": "old", "required": True, "token_count": 9}],
                }
            },
        },
    )
    _write_json(
        child_config,
        {
            "name": "child",
            "extends": "../parent/base.promptabi.json",
            "artifacts": {
                "prompt": {
                    "kind": "prompt-segment",
                    "path": "prompt.json",
                    "segments": [{"name": "new", "required": True, "token_count": 3}],
                }
            },
        },
    )

    config = load_config(child_config)

    assert config.artifacts["prompt"] == str(child_prompt.resolve())
    assert any(
        item.field == "artifacts.prompt"
        and item.status == "override"
        and Path(item.source_path).name == "promptabi.json"
        for item in config.proof_obligation_lineage
    )


def test_config_extends_detects_cycles(tmp_path: Path) -> None:
    a_config = tmp_path / "a.json"
    b_config = tmp_path / "b.json"
    _write_json(a_config, {"name": "a", "extends": "b.json"})
    _write_json(b_config, {"name": "b", "extends": "a.json"})

    with pytest.raises(ConfigError, match="cycle"):
        load_config(a_config)


def test_static_contract_findings_include_inherited_obligation_lineage(tmp_path: Path) -> None:
    prompt = tmp_path / "child" / "prompt.json"
    _write_json(prompt, {"segments": [{"name": "must", "required": True, "token_count": 20}]})
    base_config = tmp_path / "base" / "base.promptabi.json"
    child_config = tmp_path / "child" / "promptabi.json"
    _write_json(base_config, {"name": "base", "max_context_tokens": 10})
    _write_json(
        child_config,
        {
            "name": "child",
            "extends": "../base/base.promptabi.json",
            "artifacts": {
                "prompt": {
                    "kind": "prompt-segment",
                    "path": "prompt.json",
                    "segments": [{"name": "must", "required": True, "token_count": 20}],
                }
            },
        },
    )
    config = load_config(child_config)
    loaded = tuple(load_artifact(artifact) for artifact in config.artifact_bundle)

    report = analyze_static_contracts(config, loaded, prefer_z3=False)
    budget = next(finding for finding in report.findings if finding.name == "prompt-segment-survival-violation")
    evidence = dict(budget.evidence)

    assert budget.severity == "error"
    assert "base.promptabi.json" in evidence["config_inheritance"]
    assert "promptabi.json" in evidence["config_inheritance"]
    assert "max_context_tokens=10" in evidence["obligation_lineage"]
    assert "artifacts.prompt" in evidence["obligation_lineage"]
