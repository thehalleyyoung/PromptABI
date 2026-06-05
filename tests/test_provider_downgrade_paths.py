import json
from pathlib import Path

from promptabi import (
    DowngradePathStatus,
    verify_provider_downgrade_paths,
)
from promptabi.cli import main
from promptabi.config import load_config
from promptabi.loaders import ArtifactLoader


EXAMPLE_CONFIG = Path("examples/provider-downgrade-paths/promptabi.json")


def _load(config_path: Path):
    config = load_config(config_path)
    return tuple(ArtifactLoader().load(artifact) for artifact in config.artifact_bundle)


def test_complete_downgrade_plan_is_verified_against_real_losses() -> None:
    report = verify_provider_downgrade_paths(_load(EXAMPLE_CONFIG))
    by_pair = {
        (v.source_artifact, v.target_artifact): v for v in report.verifications
    }

    anthropic = by_pair[("openai-source", "anthropic-target")]
    assert anthropic.status is DowngradePathStatus.VERIFIED
    assert anthropic.verified
    assert anthropic.uncovered_losses == ()
    # The plan mitigates real, analyzer-derived capability losses.
    assert "request-field-loss" in anthropic.covered_losses
    assert "structured-output-mismatch" in anthropic.covered_losses
    assert anthropic.spurious_mitigations == ()


def test_incomplete_downgrade_plan_blocks_release() -> None:
    report = verify_provider_downgrade_paths(_load(EXAMPLE_CONFIG))
    by_pair = {
        (v.source_artifact, v.target_artifact): v for v in report.verifications
    }

    ollama = by_pair[("openai-source", "ollama-target")]
    assert ollama.status is DowngradePathStatus.INCOMPLETE
    assert not ollama.verified
    # Only request-field-loss was mitigated; the rest remain uncovered.
    assert ollama.covered_losses == ("request-field-loss",)
    assert "tool-id-mismatch" in ollama.uncovered_losses
    assert "context-limit-regression" in ollama.uncovered_losses
    assert ollama.witness.minimal_fixes

    assert not report.ok
    assert ollama in report.blocking
    assert report.downgrades_checked == 2


def test_undeclared_plan_is_flagged_when_capabilities_are_dropped(tmp_path: Path) -> None:
    # Remove the downgrade_plans entirely; every downgrade becomes undeclared.
    source = json.loads((Path("examples/provider-downgrade-paths/openai-source.json")).read_text())
    source["provider_migration"].pop("downgrade_plans", None)
    src_path = tmp_path / "openai-source.json"
    src_path.write_text(json.dumps(source), encoding="utf-8")
    for name in ("anthropic-target.json", "ollama-target.json", "promptabi.json"):
        (tmp_path / name).write_text(
            Path(f"examples/provider-downgrade-paths/{name}").read_text(), encoding="utf-8"
        )

    report = verify_provider_downgrade_paths(_load(tmp_path / "promptabi.json"))
    statuses = {v.status for v in report.verifications if v.status is not DowngradePathStatus.NO_DOWNGRADE}

    assert statuses == {DowngradePathStatus.UNDECLARED}
    assert not report.ok


def test_spurious_mitigation_is_surfaced(tmp_path: Path) -> None:
    source = json.loads((Path("examples/provider-downgrade-paths/openai-source.json")).read_text())
    source["provider_migration"]["downgrade_plans"]["anthropic-target"]["mitigations"].append(
        {"loss": "tool-id-mismatch", "fallback": "synthesize tool ids (does not actually occur for this pair)"}
    )
    src_path = tmp_path / "openai-source.json"
    src_path.write_text(json.dumps(source), encoding="utf-8")
    for name in ("anthropic-target.json", "ollama-target.json", "promptabi.json"):
        (tmp_path / name).write_text(
            Path(f"examples/provider-downgrade-paths/{name}").read_text(), encoding="utf-8"
        )

    report = verify_provider_downgrade_paths(_load(tmp_path / "promptabi.json"))
    anthropic = next(
        v for v in report.verifications if v.target_artifact == "anthropic-target"
    )

    assert "tool-id-mismatch" in anthropic.spurious_mitigations


def test_provider_downgrade_cli(capsys) -> None:
    exit_code = main(
        ["provider-downgrade", "--config", str(EXAMPLE_CONFIG), "--format", "json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1  # the ollama path is intentionally incomplete
    assert payload["version"] == "promptabi.provider-downgrade-paths.v1"
    assert payload["downgrades_checked"] == 2
    assert payload["blocking"] == 1
    assert payload["ok"] is False
