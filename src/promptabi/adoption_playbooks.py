"""Evidence-backed adoption playbooks for PromptABI user segments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .benchmarks import repo_root as default_repo_root
from .beta import run_beta_program
from .comparative_studies import build_comparative_study_report
from .evaluation import run_evaluation
from .real_bug_benchmarks import build_real_bug_benchmark_manifest


ADOPTION_PLAYBOOK_VERSION = 1
ADOPTION_AUDIENCES = (
    "startups",
    "research-labs",
    "enterprise-ai-platforms",
    "model-hosting-providers",
    "open-source-agent-projects",
)


class AdoptionPlaybookError(ValueError):
    """Raised when adoption playbooks cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class AdoptionPlaybook:
    """One evidence-backed rollout plan for a PromptABI adopter segment."""

    audience: str
    title: str
    primary_goal: str
    first_week: tuple[str, ...]
    production_gate: tuple[str, ...]
    commands: tuple[str, ...]
    success_metrics: tuple[str, ...]
    evidence_links: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "audience": self.audience,
            "title": self.title,
            "primary_goal": self.primary_goal,
            "first_week": list(self.first_week),
            "production_gate": list(self.production_gate),
            "commands": list(self.commands),
            "success_metrics": list(self.success_metrics),
            "evidence_links": list(self.evidence_links),
        }


@dataclass(frozen=True, slots=True)
class AdoptionPlaybookReport:
    """All adoption playbooks plus the live evidence behind them."""

    evidence: dict[str, object]
    playbooks: tuple[AdoptionPlaybook, ...]
    report_sha256: str

    @property
    def audience_count(self) -> int:
        return len(self.playbooks)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": ADOPTION_PLAYBOOK_VERSION,
            "promptabi_version": __version__,
            "evidence": self.evidence,
            "audience_count": self.audience_count,
            "playbooks": [playbook.to_dict() for playbook in self.playbooks],
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True, slots=True)
class AdoptionPlaybookBundle:
    """Files written by the adoption-playbook writer."""

    output_dir: Path
    report: AdoptionPlaybookReport
    written_files: tuple[Path, ...]


def build_adoption_playbook_report(*, repo_root: str | Path | None = None) -> AdoptionPlaybookReport:
    """Build adoption playbooks from live PromptABI benchmark and evaluation evidence."""

    root = Path(repo_root).resolve() if repo_root is not None else default_repo_root()
    real_bug_path = root / "fixtures" / "real_bug_benchmarks" / "benchmark.json"
    evaluation_path = root / "fixtures" / "evaluation" / "labeled_corpus.json"
    beta_path = root / "fixtures" / "beta" / "beta_program.json"
    real_bug_manifest = build_real_bug_benchmark_manifest(real_bug_path)
    evaluation = run_evaluation(evaluation_path).to_dict()
    comparative = build_comparative_study_report(
        evaluation_corpus_path=evaluation_path,
        real_bug_benchmark_path=real_bug_path,
    ).to_dict()
    beta = run_beta_program(beta_path).to_dict()
    evidence = {
        "real_bug_cases": real_bug_manifest["case_count"],
        "real_bug_categories": real_bug_manifest["categories"],
        "all_real_bug_cases_passed": real_bug_manifest["all_cases_passed"],
        "real_bug_manifest_sha256": real_bug_manifest["manifest_sha256"],
        "evaluation_cases": evaluation["case_count"],
        "evaluation_precision": evaluation["score"]["precision"],
        "evaluation_recall": evaluation["score"]["recall"],
        "evaluation_passed": evaluation["passed"],
        "comparative_baselines": len(comparative["baselines"]),
        "comparative_study_passed": comparative["passed"],
        "beta_projects": beta["project_count"],
        "upstream_issue_count": beta["upstream_issue_count"],
        "privacy_posture": "local CPU-only evidence; no model weights, provider calls, prompts, schemas, or witnesses are uploaded",
        "source_paths": {
            "real_bug_benchmark": _repo_relative(root, real_bug_path),
            "evaluation_corpus": _repo_relative(root, evaluation_path),
            "beta_program": _repo_relative(root, beta_path),
        },
    }
    playbooks = _build_playbooks(evidence)
    payload = {
        "manifest_version": ADOPTION_PLAYBOOK_VERSION,
        "promptabi_version": __version__,
        "evidence": evidence,
        "playbooks": [playbook.to_dict() for playbook in playbooks],
    }
    return AdoptionPlaybookReport(
        evidence=evidence,
        playbooks=playbooks,
        report_sha256=_stable_json_hash(payload),
    )


def render_adoption_playbooks_json(report: AdoptionPlaybookReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_adoption_playbooks_text(report: AdoptionPlaybookReport) -> str:
    """Render a compact terminal adoption report."""

    lines = [
        "PromptABI adoption playbooks",
        f"audiences: {report.audience_count}",
        (
            "evidence: "
            f"{report.evidence['real_bug_cases']} real-bug cases, "
            f"{report.evidence['evaluation_cases']} evaluation cases, "
            f"{report.evidence['comparative_baselines']} comparative baselines"
        ),
        f"privacy: {report.evidence['privacy_posture']}",
        f"report-sha256: {report.report_sha256}",
        "",
    ]
    for playbook in report.playbooks:
        lines.extend(
            [
                f"{playbook.title}",
                f"  goal: {playbook.primary_goal}",
                f"  first command: {playbook.commands[0]}",
                f"  production gate: {playbook.production_gate[0]}",
                f"  success metric: {playbook.success_metrics[0]}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_adoption_playbooks_markdown(report: AdoptionPlaybookReport) -> str:
    """Render playbooks as a concise Markdown handbook."""

    lines = [
        "# PromptABI adoption playbooks",
        "",
        (
            "These playbooks are generated from live repository evidence: "
            f"{report.evidence['real_bug_cases']} replayed real-bug reductions, "
            f"{report.evidence['evaluation_cases']} labeled evaluation cases, "
            f"{report.evidence['comparative_baselines']} comparative baselines, "
            f"and {report.evidence['beta_projects']} beta-style projects."
        ),
        "",
        f"Privacy posture: {report.evidence['privacy_posture']}.",
        "",
    ]
    for playbook in report.playbooks:
        lines.extend(_render_playbook_markdown(playbook))
    lines.extend(
        [
            "## Evidence manifest",
            "",
            f"- Real-bug manifest hash: `{report.evidence['real_bug_manifest_sha256']}`",
            f"- Evaluation precision/recall: `{report.evidence['evaluation_precision']}` / `{report.evidence['evaluation_recall']}`",
            f"- Report hash: `{report.report_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_adoption_playbooks(
    output_dir: str | Path = "adoption_playbooks",
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> AdoptionPlaybookBundle:
    """Write Markdown playbooks and a deterministic JSON manifest."""

    destination = Path(output_dir)
    report = build_adoption_playbook_report(repo_root=repo_root)
    filenames = _playbook_filenames(report)
    _prepare_output_dir(destination, expected_filenames=filenames, force=force)
    written: list[Path] = []
    manifest_path = destination / "adoption-playbooks.json"
    manifest_path.write_text(render_adoption_playbooks_json(report), encoding="utf-8")
    written.append(manifest_path)
    index_path = destination / "README.md"
    index_path.write_text(render_adoption_playbooks_markdown(report), encoding="utf-8")
    written.append(index_path)
    for playbook in report.playbooks:
        path = destination / f"{playbook.audience}.md"
        path.write_text("\n".join(_render_playbook_markdown(playbook)), encoding="utf-8")
        written.append(path)
    return AdoptionPlaybookBundle(output_dir=destination, report=report, written_files=tuple(written))


def render_adoption_playbook_summary(bundle: AdoptionPlaybookBundle) -> str:
    return (
        "PromptABI adoption playbooks\n"
        f"output: {bundle.output_dir}\n"
        f"files: {len(bundle.written_files)}\n"
        f"audiences: {bundle.report.audience_count}\n"
        f"real-bug cases: {bundle.report.evidence['real_bug_cases']}\n"
        f"evaluation cases: {bundle.report.evidence['evaluation_cases']}\n"
        f"manifest: {bundle.output_dir / 'adoption-playbooks.json'}\n"
    )


def _build_playbooks(evidence: dict[str, object]) -> tuple[AdoptionPlaybook, ...]:
    real_bug_cases = str(evidence["real_bug_cases"])
    evaluation_cases = str(evidence["evaluation_cases"])
    upstream_issues = str(evidence["upstream_issue_count"])
    return (
        AdoptionPlaybook(
            audience="startups",
            title="Startup launch gate",
            primary_goal="Ship LLM features quickly without letting template, stop, tool, or provider drift escape CI.",
            first_week=(
                "Run the minimal verifier on one production-like config and fix every error before expanding scope.",
                "Add the GitHub Action with lockfile enforcement after the first clean local run.",
                "Use the bug gallery to teach the team which structural failures are caught before inference.",
            ),
            production_gate=(
                "Require `promptabi github-action --config examples/minimal/promptabi.json --require-lockfile` on pull requests touching prompt-interface artifacts.",
                "Treat new error diagnostics as release blockers and warnings as explicit launch-risk review items.",
            ),
            commands=(
                "promptabi verify --config examples/minimal/promptabi.json",
                "promptabi github-action --config examples/minimal/promptabi.json --require-lockfile",
                "promptabi corpus bug-gallery --format markdown",
            ),
            success_metrics=(
                f"CI proves the same classes covered by {real_bug_cases} replayed real-bug reductions before launch.",
                "Every accepted risk has an owner, expiration, and unchanged witness digest.",
            ),
            evidence_links=("real_bug_cases", "evaluation_precision", "comparative_baselines"),
        ),
        AdoptionPlaybook(
            audience="research-labs",
            title="Research reproducibility lane",
            primary_goal="Publish evaluations and papers with prompt rendering, tokenizer, stop, parser, and fixture assumptions pinned.",
            first_week=(
                "Run the evaluation harness verifier on benchmark prompts and parsers.",
                "Generate the paper reproducibility package and check that fixture hashes match the repository state.",
                "Replay comparative and adversarial corpora to document blind spots and abstentions.",
            ),
            production_gate=(
                "Require `promptabi corpus evaluation-reproducibility --config examples/evaluation-harness/safe.promptabi.json` before publishing scores.",
                "Archive the generated paper artifact next to result tables.",
            ),
            commands=(
                "promptabi corpus evaluation --format text",
                "promptabi corpus evaluation-reproducibility --config examples/evaluation-harness/safe.promptabi.json --format json",
                "promptabi paper reproducibility --output-dir paper_artifact --force",
            ),
            success_metrics=(
                f"Benchmark claims are replayed over {evaluation_cases} labeled evaluation cases with pinned parser/tokenizer contracts.",
                "Abstentions and unsupported fragments are listed beside scores rather than hidden in prose.",
            ),
            evidence_links=("evaluation_cases", "evaluation_recall", "source_paths"),
        ),
        AdoptionPlaybook(
            audience="enterprise-ai-platforms",
            title="Enterprise platform control plane",
            primary_goal="Standardize offline verification, approved policy packs, private indexes, and audit bundles across teams.",
            first_week=(
                "Declare strict no-network mirrors, policy packs, provider fixtures, and solver resource limits in a platform config.",
                "Create signed verification bundles for one representative app, one RAG workflow, and one training workflow.",
                "Publish a team dashboard view with accepted suppressions and drift warnings.",
            ),
            production_gate=(
                "Require `promptabi bundle create --config examples/minimal/promptabi.json --output promptabi.bundle.json` for release evidence.",
                "Reject deploys whose bundles are older than the current lockfile or platform policy pack.",
            ),
            commands=(
                "promptabi verify --config examples/minimal/promptabi.json --fail-on error",
                "promptabi bundle create --config examples/minimal/promptabi.json --output promptabi.bundle.json",
                "promptabi dashboard --config examples/role-boundary/unsafe.promptabi.json --history .promptabi/team-dashboard.jsonl --record",
            ),
            success_metrics=(
                "All private prompt-interface evidence stays local while bundles expose hashes, source spans, diagnostics, and solver metadata.",
                "Platform dashboards trend structural risk without storing raw prompts or schemas.",
            ),
            evidence_links=("privacy_posture", "beta_projects", "upstream_issue_count"),
        ),
        AdoptionPlaybook(
            audience="model-hosting-providers",
            title="Provider compatibility lab",
            primary_goal="Prove hosted tokenizer, template, tool-call, streaming, and stop semantics remain compatible across API and model updates.",
            first_week=(
                "Replay provider conformance fixtures for JSON mode, tool-call streaming, stop handling, and error envelopes.",
                "Run provider-migration checks against representative OpenAI-compatible, Anthropic-style, Gemini-style, and local-server configs.",
                "Publish non-sensitive compatibility notes and fixed-version metadata for downstream users.",
            ),
            production_gate=(
                "Require provider fixture conformance and drift-bisect reports before changing tokenizer/template/provider revisions.",
                "Attach upstream issue links and local workarounds when a compatibility regression is found.",
            ),
            commands=(
                "promptabi corpus provider-conformance --format text",
                "promptabi diff promptabi.baseline.json promptabi.json",
                "promptabi release drift-bisect --surface provider --baseline provider-r0 --revision provider-r1=provider-r1",
            ),
            success_metrics=(
                f"Provider-facing failures are checked against {upstream_issues} upstream issue records and local workaround metadata.",
                "Every public compatibility badge is backed by replayable fixture hashes rather than provider calls.",
            ),
            evidence_links=("upstream_issue_count", "comparative_study_passed", "real_bug_categories"),
        ),
        AdoptionPlaybook(
            audience="open-source-agent-projects",
            title="Open-source agent maintainer loop",
            primary_goal="Make prompt-pack, tool-schema, and contributor changes safe for external users and plugin authors.",
            first_week=(
                "Add a prompt-pack contract for reusable templates and tool schemas.",
                "Generate contributor validation output so sanitized bug fixtures and minimized witnesses are easy to submit.",
                "Run adoption playbooks in CI to keep onboarding instructions synchronized with working commands.",
            ),
            production_gate=(
                "Require prompt-pack upgrade preservation and contributor validation on pull requests that change public agent prompts.",
                "Publish registry metadata without raw prompt contents so users can mirror and verify packs offline.",
            ),
            commands=(
                "promptabi prompt-pack upgrade --config examples/prompt-packs/promptabi.json --baseline-lockfile /tmp/prompt-pack.lock.json",
                "promptabi prompt-pack registry --config examples/prompt-packs/promptabi.json --format text",
                "promptabi contribute validate --format text",
            ),
            success_metrics=(
                "New contributors can reproduce a failing structural bug with minimized, secret-free artifacts.",
                "Prompt-pack upgrades preserve role, stop, schema, and budget guarantees before release.",
            ),
            evidence_links=("real_bug_manifest_sha256", "privacy_posture", "comparative_baselines"),
        ),
    )


def _render_playbook_markdown(playbook: AdoptionPlaybook) -> list[str]:
    lines = [
        f"## {playbook.title}",
        "",
        f"**Audience:** `{playbook.audience}`",
        "",
        f"**Goal:** {playbook.primary_goal}",
        "",
        "### First week",
        "",
    ]
    lines.extend(f"1. {item}" for item in playbook.first_week)
    lines.extend(["", "### Production gate", ""])
    lines.extend(f"1. {item}" for item in playbook.production_gate)
    lines.extend(["", "### Commands", "", "```bash"])
    lines.extend(playbook.commands)
    lines.extend(["```", "", "### Success metrics", ""])
    lines.extend(f"- {item}" for item in playbook.success_metrics)
    lines.extend(["", "### Evidence keys", ""])
    lines.extend(f"- `{item}`" for item in playbook.evidence_links)
    lines.append("")
    return lines


def _playbook_filenames(report: AdoptionPlaybookReport) -> frozenset[str]:
    return frozenset({"adoption-playbooks.json", "README.md", *(f"{item.audience}.md" for item in report.playbooks)})


def _prepare_output_dir(destination: Path, *, expected_filenames: frozenset[str], force: bool) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise AdoptionPlaybookError(f"output path exists and is not a directory: {destination}")
        existing = {path.name for path in destination.iterdir() if not path.name.startswith(".")}
        unexpected = existing.difference(expected_filenames)
        if existing and (unexpected or not force):
            detail = ", ".join(sorted(existing))
            raise AdoptionPlaybookError(
                f"output directory is not empty: {destination} ({detail}); pass --force to overwrite adoption playbooks"
            )
    destination.mkdir(parents=True, exist_ok=True)


def _repo_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
