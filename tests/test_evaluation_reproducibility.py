import json
import shutil
from pathlib import Path

import promptabi
from promptabi.cli import main
from promptabi.evaluation_reproducibility import (
    build_evaluation_reproducibility_report,
    render_evaluation_reproducibility_json,
    render_evaluation_reproducibility_text,
)


ROOT = Path(__file__).resolve().parents[1]
SAFE_CONFIG = ROOT / "examples" / "evaluation-harness" / "safe.promptabi.json"


def test_evaluation_reproducibility_pins_all_harness_surfaces() -> None:
    report = build_evaluation_reproducibility_report([SAFE_CONFIG])
    payload = report.to_dict()
    config = payload["configs"][0]
    surfaces = config["surfaces"]

    assert payload["manifest_version"] == promptabi.EVALUATION_REPRODUCIBILITY_VERSION
    assert payload["manifest_sha256"]
    assert config["reproducibility_status"] == "complete"
    assert set(surfaces) == {
        "parser_contracts",
        "prompt_rendering",
        "provider_fixtures",
        "stop_policies",
        "tokenizer_versions",
    }
    assert all(surface["complete"] is True for surface in surfaces.values())
    assert all(surface["surface_sha256"] for surface in surfaces.values())
    assert surfaces["prompt_rendering"]["chat_templates"][0]["actual_sha256"]
    assert surfaces["tokenizer_versions"]["harness_benchmark_tokenizers"][0]["benchmark_tokenizer"]["name"] == "byte-bpe"
    assert surfaces["provider_fixtures"]["provider_configs"][0]["provider"] == "openai-compatible"
    assert surfaces["stop_policies"]["stop_policies"][0]["stop_sequences"] == ["</answer>"]
    assert surfaces["parser_contracts"]["schemas"][0]["dialect"] == "json-schema"


def test_evaluation_reproducibility_renderers_and_cli(capsys) -> None:
    report = build_evaluation_reproducibility_report([SAFE_CONFIG])
    text = render_evaluation_reproducibility_text(report)
    payload = json.loads(render_evaluation_reproducibility_json(report))

    assert "PromptABI evaluation reproducibility" in text
    assert payload["configs"][0]["reproducibility_status"] == "complete"

    exit_code = main(["corpus", "evaluation-reproducibility", "--config", str(SAFE_CONFIG), "--format", "json"])
    captured = capsys.readouterr()
    cli_payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert cli_payload["manifest_sha256"] == payload["manifest_sha256"]


def test_public_api_evaluation_reproducibility() -> None:
    report = promptabi.evaluation_reproducibility([SAFE_CONFIG])
    rendered = promptabi.evaluation_reproducibility([SAFE_CONFIG], output_format="json")

    assert isinstance(report, promptabi.EvaluationReproducibilityReport)
    assert json.loads(rendered)["config_count"] == 1


def test_evaluation_reproducibility_hash_changes_when_stop_policy_changes(tmp_path: Path) -> None:
    copied = tmp_path / "evaluation-harness"
    shutil.copytree(ROOT / "examples" / "evaluation-harness", copied)
    config = copied / "safe.promptabi.json"

    before = build_evaluation_reproducibility_report([config])
    stop_policy = copied / "stop-policy.json"
    payload = json.loads(stop_policy.read_text(encoding="utf-8"))
    payload["stop"] = ["</answer>", "<END>"]
    stop_policy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    after = build_evaluation_reproducibility_report([config])

    before_surface = before.to_dict()["configs"][0]["surfaces"]["stop_policies"]["surface_sha256"]
    after_surface = after.to_dict()["configs"][0]["surfaces"]["stop_policies"]["surface_sha256"]
    assert before.manifest_sha256 != after.manifest_sha256
    assert before_surface != after_surface
