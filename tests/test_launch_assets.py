import json
from pathlib import Path

from promptabi.cli import main
from promptabi.launch_assets import LaunchAssetError, build_launch_asset_payloads, write_launch_assets


def test_launch_assets_are_generated_from_real_reports() -> None:
    payloads, gif_bytes, manifest = build_launch_asset_payloads(benchmark_iterations=1)

    assert manifest["summary"]["real_bug_cases"] >= 7
    assert manifest["summary"]["all_real_bug_cases_passed"] is True
    assert manifest["summary"]["evaluation_precision"] == 1.0
    assert manifest["summary"]["evaluation_recall"] == 1.0
    assert manifest["summary"]["benchmark_cases"] == 8
    assert manifest["summary"]["upstream_issue_count"] >= 1
    assert "PromptABI comparison" in payloads["comparison.md"]
    assert "flowchart LR" in payloads["architecture.mmd"]
    assert "promptabi verify --config examples/role-boundary/unsafe.promptabi.json" in payloads["demo-script.md"]
    assert "<svg" in payloads["benchmark-chart.svg"]
    assert "PromptABI bug gallery" in payloads["bug-gallery.md"]
    assert "Hacker News title" in payloads["positioning.md"]
    assert gif_bytes.startswith(b"GIF89a")


def test_launch_assets_writer_and_cli_create_expected_files(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "launch"
    bundle = write_launch_assets(output_dir, benchmark_iterations=1)

    assert sorted(path.name for path in bundle.written_files) == [
        "architecture.mmd",
        "benchmark-chart.svg",
        "benchmark-data.json",
        "bug-gallery.md",
        "comparison.md",
        "demo-script.md",
        "demo.gif",
        "launch-manifest.json",
        "positioning.md",
    ]
    assert (output_dir / "demo.gif").read_bytes().startswith(b"GIF89a")
    assert json.loads((output_dir / "launch-manifest.json").read_text(encoding="utf-8"))["asset_payload_sha256"]

    try:
        write_launch_assets(output_dir, benchmark_iterations=1)
    except LaunchAssetError as exc:
        assert "pass --force" in str(exc)
    else:
        raise AssertionError("expected existing launch asset directory to require --force")

    cli_dir = tmp_path / "cli-launch"
    exit_code = main(["launch-assets", "--output-dir", str(cli_dir), "--benchmark-iterations", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "PromptABI launch assets" in captured.out
    assert "real-bug cases:" in captured.out
    assert (cli_dir / "comparison.md").is_file()
