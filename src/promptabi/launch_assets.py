"""Launch asset generation backed by real PromptABI reports."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .benchmarks import repo_root as default_repo_root
from .benchmarks import run_benchmarks
from .beta import run_beta_program
from .bug_gallery import build_public_bug_gallery, render_public_bug_gallery_markdown
from .comparative_studies import build_comparative_study_report, render_comparative_study_markdown
from .evaluation import run_evaluation
from .real_bug_benchmarks import build_real_bug_benchmark_manifest


LAUNCH_ASSET_VERSION = 1
TEXT_ASSET_FILENAMES = (
    "comparison.md",
    "architecture.mmd",
    "demo-script.md",
    "benchmark-chart.svg",
    "benchmark-data.json",
    "bug-gallery.md",
    "positioning.md",
    "launch-manifest.json",
)
LAUNCH_ASSET_FILENAMES = (*TEXT_ASSET_FILENAMES, "demo.gif")
_DEMO_GIF_BYTES = base64.b64decode(
    # 1x1 transparent GIF89a placeholder. The accompanying demo-script.md is the
    # replay source for producing a higher-fidelity terminal capture.
    "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
)


class LaunchAssetError(ValueError):
    """Raised when launch assets cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class LaunchAssetBundle:
    """Generated launch assets and the real reports behind their claims."""

    output_dir: Path
    manifest: dict[str, object]
    written_files: tuple[Path, ...]


def build_launch_asset_payloads(
    *,
    repo_root: str | Path | None = None,
    benchmark_iterations: int = 1,
) -> tuple[dict[str, str], bytes, dict[str, object]]:
    """Build launch-facing Markdown/SVG/JSON assets from live PromptABI code paths."""

    if benchmark_iterations <= 0:
        raise LaunchAssetError("benchmark_iterations must be positive")
    root = Path(repo_root).resolve() if repo_root is not None else default_repo_root()
    real_bug_path = root / "fixtures" / "real_bug_benchmarks" / "benchmark.json"
    real_bug_manifest = build_real_bug_benchmark_manifest(real_bug_path)
    bug_gallery_report = build_public_bug_gallery(real_bug_path)
    bug_gallery = bug_gallery_report.to_dict()
    evaluation = run_evaluation(root / "fixtures" / "evaluation" / "labeled_corpus.json").to_dict()
    comparative_study_report = build_comparative_study_report(
        evaluation_corpus_path=root / "fixtures" / "evaluation" / "labeled_corpus.json",
        real_bug_benchmark_path=real_bug_path,
    )
    comparative_study = comparative_study_report.to_dict()
    beta = run_beta_program(root / "fixtures" / "beta" / "beta_program.json").to_dict()
    benchmarks = [result.to_dict() for result in run_benchmarks(("all",), iterations=benchmark_iterations, root=root)]
    evidence = {
        "real_bug_manifest": real_bug_manifest,
        "evaluation": evaluation,
        "comparative_study": comparative_study,
        "comparative_study_report": comparative_study_report,
        "beta": beta,
        "benchmarks": benchmarks,
        "bug_gallery": bug_gallery,
    }
    payloads = {
        "comparison.md": _render_comparison(evidence),
        "architecture.mmd": _render_architecture(),
        "demo-script.md": _render_demo_script(evidence),
        "benchmark-chart.svg": _render_benchmark_chart(benchmarks),
        "benchmark-data.json": json.dumps({"benchmarks": benchmarks}, indent=2, sort_keys=True) + "\n",
        "bug-gallery.md": render_public_bug_gallery_markdown(bug_gallery_report),
        "positioning.md": _render_positioning(evidence),
    }
    manifest = _build_manifest(payloads=payloads, evidence=evidence, benchmark_iterations=benchmark_iterations)
    payloads["launch-manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return payloads, _DEMO_GIF_BYTES, manifest


def write_launch_assets(
    output_dir: str | Path = "launch_assets",
    *,
    repo_root: str | Path | None = None,
    benchmark_iterations: int = 1,
    force: bool = False,
) -> LaunchAssetBundle:
    """Write generated launch assets, refusing unsafe overwrites by default."""

    destination = Path(output_dir)
    _prepare_output_dir(destination, force=force)
    payloads, gif_bytes, manifest = build_launch_asset_payloads(
        repo_root=repo_root,
        benchmark_iterations=benchmark_iterations,
    )
    written: list[Path] = []
    for filename in TEXT_ASSET_FILENAMES:
        path = destination / filename
        path.write_text(payloads[filename], encoding="utf-8")
        written.append(path)
    gif_path = destination / "demo.gif"
    gif_path.write_bytes(gif_bytes)
    written.append(gif_path)
    return LaunchAssetBundle(output_dir=destination, manifest=manifest, written_files=tuple(written))


def render_launch_asset_summary(bundle: LaunchAssetBundle) -> str:
    """Render a concise terminal summary for generated launch assets."""

    summary = bundle.manifest["summary"]  # type: ignore[index]
    return (
        "PromptABI launch assets\n"
        f"output: {bundle.output_dir}\n"
        f"files: {len(bundle.written_files)}\n"
        f"real-bug cases: {summary['real_bug_cases']}\n"
        f"evaluation cases: {summary['evaluation_cases']}\n"
        f"benchmark cases: {summary['benchmark_cases']}\n"
        f"beta projects: {summary['beta_projects']}\n"
        f"manifest: {bundle.output_dir / 'launch-manifest.json'}\n"
    )


def _build_manifest(
    *,
    payloads: dict[str, str],
    evidence: dict[str, Any],
    benchmark_iterations: int,
) -> dict[str, object]:
    real_bug_manifest = evidence["real_bug_manifest"]
    evaluation = evidence["evaluation"]
    beta = evidence["beta"]
    benchmarks = evidence["benchmarks"]
    bug_gallery = evidence["bug_gallery"]
    comparative_study = evidence["comparative_study"]
    manifest: dict[str, object] = {
        "manifest_version": LAUNCH_ASSET_VERSION,
        "promptabi_version": __version__,
        "purpose": "Launch assets generated from real PromptABI benchmark, evaluation, beta, and real-bug replay code paths.",
        "benchmark_iterations": benchmark_iterations,
        "files": list(LAUNCH_ASSET_FILENAMES),
        "summary": {
            "real_bug_cases": real_bug_manifest["case_count"],
            "public_bug_gallery_entries": bug_gallery["summary"]["entries"],
            "real_bug_categories": real_bug_manifest["categories"],
            "all_real_bug_cases_passed": real_bug_manifest["all_cases_passed"],
            "evaluation_cases": evaluation["case_count"],
            "evaluation_precision": evaluation["score"]["precision"],
            "evaluation_recall": evaluation["score"]["recall"],
            "benchmark_cases": len(benchmarks),
            "beta_projects": beta["project_count"],
            "upstream_issue_count": beta["upstream_issue_count"],
            "comparative_baselines": len(comparative_study["baselines"]),
            "comparative_study_passed": comparative_study["passed"],
        },
        "source_reports": {
            "real_bug_manifest_sha256": real_bug_manifest["manifest_sha256"],
            "public_bug_gallery_sha256": bug_gallery["report_sha256"],
            "evaluation_passed": evaluation["passed"],
            "beta_passed": beta["passed"],
            "benchmark_names": [row["benchmark"] for row in benchmarks],
        },
    }
    manifest["asset_payload_sha256"] = _stable_json_hash(
        {
            "text_assets": {name: payloads[name] for name in sorted(payloads)},
            "gif_sha256": _stable_json_hash({"gif_bytes": _DEMO_GIF_BYTES.hex()}),
            "summary": manifest["summary"],
        }
    )
    return manifest


def _render_comparison(evidence: dict[str, Any]) -> str:
    return render_comparative_study_markdown(evidence["comparative_study_report"])


def _render_architecture() -> str:
    return """%% PromptABI launch architecture diagram.
flowchart LR
  A[PromptABI config] --> B[Artifact loaders]
  B --> C[Chat templates]
  B --> D[Tokenizers]
  B --> E[Stops / grammars / tools]
  B --> F[Provider, RAG, training contracts]
  C --> G[Automata + transducer products]
  D --> G
  E --> G
  F --> H[Z3-backed finite contracts]
  G --> I[Diagnostics with witnesses]
  H --> I
  I --> J[CLI / JSON / SARIF / HTML / editor protocol]
  I --> K[Real-bug benchmark + paper artifact]
"""


def _render_demo_script(evidence: dict[str, Any]) -> str:
    return (
        "# Demo GIF source script\n\n"
        "Record these deterministic, CPU-only commands into a terminal GIF for launch pages:\n\n"
        "```bash\n"
        "python -m pip install -e \".[dev,grammars,solver,tokenizers]\"\n"
        "promptabi verify --config examples/role-boundary/unsafe.promptabi.json --fail-on never\n"
        "promptabi explain --config examples/role-boundary/unsafe.promptabi.json --index 1\n"
        "promptabi corpus real-bug-benchmark --output /tmp/promptabi-real-bugs.json\n"
        "promptabi corpus evaluation --format text\n"
        "promptabi launch-assets --output-dir launch_assets --force\n"
        "```\n\n"
        f"The accompanying placeholder `demo.gif` is valid GIF89a; regenerate it from this script for release. "
        f"The evidence behind the script currently reports {evidence['real_bug_manifest']['case_count']} real-bug cases "
        f"and {evidence['evaluation']['case_count']} labeled evaluation cases.\n"
    )


def _render_benchmark_chart(benchmarks: list[dict[str, Any]]) -> str:
    width = 900
    row_height = 34
    chart_height = 80 + row_height * len(benchmarks)
    max_rps = max((float(row["runs_per_second"]) for row in benchmarks), default=1.0)
    rows = []
    for index, row in enumerate(benchmarks):
        y = 54 + index * row_height
        label = html.escape(str(row["benchmark"]))
        rps = float(row["runs_per_second"])
        bar_width = 1 if max_rps <= 0 else max(1, int((rps / max_rps) * 520))
        rows.append(
            f'<text x="24" y="{y + 17}" font-size="13">{label}</text>'
            f'<rect x="286" y="{y}" width="{bar_width}" height="20" fill="#2563eb" rx="3"/>'
            f'<text x="{300 + bar_width}" y="{y + 16}" font-size="12">{rps:.2f} runs/s</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" '
        f'height="{chart_height}" viewBox="0 0 {width} {chart_height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<text x="24" y="28" font-size="20" font-weight="700">PromptABI CPU-only benchmark chart</text>\n'
        '<text x="24" y="48" font-size="12" fill="#475569">Generated from live benchmark code paths; values vary by machine.</text>\n'
        + "\n".join(rows)
        + "\n</svg>\n"
    )


def _render_positioning(evidence: dict[str, Any]) -> str:
    beta = evidence["beta"]
    real_bug_manifest = evidence["real_bug_manifest"]
    evaluation = evidence["evaluation"]
    return (
        "# Launch positioning\n\n"
        "## Hacker News title\n\n"
        "PromptABI: static CI for the tokenizer/template/tool-calling boundary of LLM apps\n\n"
        "## GitHub social preview\n\n"
        "Prove LLM interface bugs before inference: role delimiters, stops, grammars, tools, providers, RAG budgets, "
        "training masks, lockfiles, SARIF, and real-bug benchmarks.\n\n"
        "## Proof points\n\n"
        f"- {real_bug_manifest['case_count']} real-bug reductions across "
        f"{len(real_bug_manifest['categories'])} categories replay offline.\n"
        f"- {evaluation['case_count']} labeled evaluation cases report precision={evaluation['score']['precision']} "
        f"and recall={evaluation['score']['recall']}.\n"
        f"- Beta replay covers {beta['project_count']} open-source-style projects and "
        f"{beta['upstream_issue_count']} upstream issue records.\n"
        "- CPU-only: no model weights, provider calls, prompt upload, or telemetry required.\n"
    )


def _prepare_output_dir(destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise LaunchAssetError(f"output path exists and is not a directory: {destination}")
        existing = {path.name for path in destination.iterdir() if not path.name.startswith(".")}
        unexpected = existing.difference(LAUNCH_ASSET_FILENAMES)
        if existing and (unexpected or not force):
            detail = ", ".join(sorted(existing))
            raise LaunchAssetError(
                f"output directory is not empty: {destination} ({detail}); pass --force to overwrite launch assets"
            )
    destination.mkdir(parents=True, exist_ok=True)


def _stable_json_hash(value: object) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
