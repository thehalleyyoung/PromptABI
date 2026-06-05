import json

from promptabi.cli import main
from promptabi.gallery import build_gallery, render_gallery_json, render_gallery_text


def test_gallery_runs_curated_configs_against_real_verifier() -> None:
    report = build_gallery("examples/gallery")
    entries = {entry.id: entry for entry in report.entries}

    assert report.ok
    assert set(entries) == {
        "audited-accepted-risk",
        "minimal-openai-tools",
        "rag-budget-fixed",
        "safe-chatml-boundary",
        "training-serving-static",
    }
    assert {"PASS", "PINNED", "OFFLINE", "LOCKFILE"}.issubset(entries["minimal-openai-tools"].badges)
    assert {"PASS", "PINNED", "OFFLINE"}.issubset(entries["safe-chatml-boundary"].badges)
    assert {"PASS", "PINNED", "OFFLINE", "SOUND", "SMT-CAPABLE"}.issubset(
        entries["training-serving-static"].badges
    )
    assert entries["training-serving-static"].proof_summaries
    assert all(summary.actual_backend != "z3-backed-smt" for summary in entries["training-serving-static"].proof_summaries)
    assert entries["audited-accepted-risk"].accepted_risks[0].owner == "platform-verification"
    assert "documentation/demo scaffolding" in entries["audited-accepted-risk"].accepted_risks[0].explanation


def test_gallery_renderers_are_deterministic_and_truthful() -> None:
    report = build_gallery("examples/gallery")

    payload = json.loads(render_gallery_json(report))
    assert payload["ok"] is True
    assert payload["summary"]["entries"] == 5
    static = next(entry for entry in payload["entries"] if entry["id"] == "training-serving-static")
    assert static["proof_summaries"]
    assert all("actual_backend" in summary for summary in static["proof_summaries"])

    text = render_gallery_text(report)
    assert "PromptABI verified configuration gallery" in text
    assert "minimal-openai-tools: Pinned OpenAI-style tool contract [PASS] [PINNED]" in text
    assert "proof: static-contract-proved ->" in text
    assert "accepted risk (platform-verification):" in text


def test_gallery_cli_outputs_json_and_writes_files(tmp_path, capsys) -> None:
    output = tmp_path / "gallery.json"

    exit_code = main(["gallery", "--root", "examples/gallery", "--format", "json", "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"wrote gallery report: {output} (5 entries)\n"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["passing"] == 5
