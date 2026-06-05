import json
from pathlib import Path

from promptabi.bug_gallery import (
    build_public_bug_gallery,
    render_public_bug_gallery_json,
    render_public_bug_gallery_markdown,
    render_public_bug_gallery_text,
)
from promptabi.cli import main
from promptabi.real_bug_benchmarks import REQUIRED_REAL_BUG_CATEGORIES


def test_public_bug_gallery_replays_real_cases_with_sanitized_witnesses() -> None:
    report = build_public_bug_gallery()

    assert report.version == 1
    assert report.all_replayed is True
    assert len(report.entries) >= 7
    assert set(report.categories) == set(REQUIRED_REAL_BUG_CATEGORIES)
    assert len(report.report_sha256) == 64
    for entry in report.entries:
        assert entry.replayed is True
        assert entry.public_reference.startswith("https://github.com/")
        assert entry.root_cause
        assert entry.sanitized_artifacts
        assert all(len(artifact.sha256) == 64 for artifact in entry.sanitized_artifacts)
        assert entry.witness.rule_ids
        assert entry.witness.evidence
        assert entry.witness.minimized_repro["observed_rule_ids"]
        assert len(entry.witness.replay_hash) == 64
        assert entry.fixes
        assert entry.upstream_patches


def test_public_bug_gallery_renderers_are_deterministic() -> None:
    report = build_public_bug_gallery()
    payload = json.loads(render_public_bug_gallery_json(report))

    assert payload["summary"]["all_replayed"] is True
    assert payload["summary"]["entries"] == len(report.entries)
    assert payload["report_sha256"] == report.report_sha256
    assert payload["entries"][0]["minimized_witness"]["replay_hash"]

    markdown = render_public_bug_gallery_markdown(report)
    assert "PromptABI public bug gallery" in markdown
    assert "Root cause" in markdown
    assert "replay hash" in markdown

    text = render_public_bug_gallery_text(report)
    assert "PromptABI public bug gallery" in text
    assert "all replayed: true" in text


def test_public_bug_gallery_cli_outputs_and_writes(tmp_path: Path, capsys) -> None:
    exit_code = main(["corpus", "bug-gallery", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["summary"]["all_replayed"] is True

    output = tmp_path / "bug-gallery.md"
    exit_code = main(["corpus", "bug-gallery", "--format", "markdown", "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert f"wrote public bug gallery: {output}" in captured.out
    assert "PromptABI public bug gallery" in output.read_text(encoding="utf-8")
