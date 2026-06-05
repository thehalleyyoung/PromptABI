import json
from pathlib import Path

from promptabi import (
    PromptPackModuleFindingKind,
    resolve_prompt_pack_modules,
)
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader


EXAMPLE_CONFIG = Path("examples/prompt-pack-modules/promptabi.json")


def _load(config_path: Path):
    config = load_config(config_path)
    return tuple(ArtifactLoader().load(artifact) for artifact in config.artifact_bundle)


def test_clean_module_program_links_with_deterministic_order() -> None:
    graph = resolve_prompt_pack_modules(_load(EXAMPLE_CONFIG))

    assert graph.ok
    assert graph.findings == ()
    assert graph.link_order == ("base-pack", "app-pack")
    app = next(module for module in graph.modules if module.pack_name == "app-pack")
    assert "template:chat-core" in {edge.symbol for edge in app.imports}
    base = next(module for module in graph.modules if module.pack_name == "base-pack")
    assert "tool:search_docs" in base.exports


def test_unresolved_symbol_is_reported(tmp_path: Path) -> None:
    app = json.loads(Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text())
    app["imports"].append({"pack": "base-pack", "symbol": "template:does-not-exist"})
    _materialize(tmp_path, app_override=app)

    graph = resolve_prompt_pack_modules(_load(tmp_path / "promptabi.json"))

    assert not graph.ok
    kinds = {finding.kind for finding in graph.findings}
    assert PromptPackModuleFindingKind.UNRESOLVED_SYMBOL in kinds
    assert graph.link_order == ()


def test_unresolved_pack_is_reported(tmp_path: Path) -> None:
    app = json.loads(Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text())
    app["imports"].append({"pack": "ghost-pack", "symbol": "template:chat-core"})
    _materialize(tmp_path, app_override=app)

    graph = resolve_prompt_pack_modules(_load(tmp_path / "promptabi.json"))

    assert any(
        finding.kind is PromptPackModuleFindingKind.UNRESOLVED_PACK for finding in graph.findings
    )


def test_version_incompatible_import_is_reported(tmp_path: Path) -> None:
    app = json.loads(Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text())
    app["imports"][0]["min_version"] = "2.0.0"  # base-pack is only 1.2.0
    _materialize(tmp_path, app_override=app)

    graph = resolve_prompt_pack_modules(_load(tmp_path / "promptabi.json"))

    version_findings = [
        finding
        for finding in graph.findings
        if finding.kind is PromptPackModuleFindingKind.VERSION_INCOMPATIBLE
    ]
    assert version_findings
    assert version_findings[0].witness.minimal_fixes


def test_import_cycle_is_detected(tmp_path: Path) -> None:
    base = json.loads(Path("examples/prompt-pack-modules/base.prompt-pack.json").read_text())
    app = json.loads(Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text())
    # Make base import from app so the graph forms a cycle.
    base["imports"] = [{"pack": "app-pack", "symbol": "template:app-flow"}]
    _materialize(tmp_path, base_override=base, app_override=app)

    graph = resolve_prompt_pack_modules(_load(tmp_path / "promptabi.json"))

    assert any(
        finding.kind is PromptPackModuleFindingKind.IMPORT_CYCLE for finding in graph.findings
    )
    assert graph.link_order == ()


def test_duplicate_export_across_packs_is_reported(tmp_path: Path) -> None:
    base = json.loads(Path("examples/prompt-pack-modules/base.prompt-pack.json").read_text())
    app = json.loads(Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text())
    # Export the same template name from both packs.
    app["exported_templates"][0]["name"] = "chat-core"
    app.pop("imports", None)
    _materialize(tmp_path, base_override=base, app_override=app)

    graph = resolve_prompt_pack_modules(_load(tmp_path / "promptabi.json"))

    assert any(
        finding.kind is PromptPackModuleFindingKind.DUPLICATE_EXPORT for finding in graph.findings
    )


def test_prompt_pack_modules_cli(capsys) -> None:
    exit_code = main(
        ["prompt-pack-modules", "--config", str(EXAMPLE_CONFIG), "--format", "json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["version"] == "promptabi.prompt-pack-modules.v1"
    assert payload["link_order"] == ["base-pack", "app-pack"]


def _materialize(tmp_path: Path, *, base_override=None, app_override=None) -> None:
    base = base_override if base_override is not None else json.loads(
        Path("examples/prompt-pack-modules/base.prompt-pack.json").read_text()
    )
    app = app_override if app_override is not None else json.loads(
        Path("examples/prompt-pack-modules/app.prompt-pack.json").read_text()
    )
    (tmp_path / "base.prompt-pack.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "app.prompt-pack.json").write_text(json.dumps(app), encoding="utf-8")
    (tmp_path / "promptabi.json").write_text(
        Path("examples/prompt-pack-modules/promptabi.json").read_text(), encoding="utf-8"
    )
