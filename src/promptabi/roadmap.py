"""Roadmap, teaching, award, corpus-refresh, and historical trend reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark_leaderboards import build_benchmark_leaderboard
from .comparative_studies import build_comparative_study_report
from .conference_demos import run_conference_demos
from .corpus_verification import run_corpus_verification
from .provider_fixture_packs import build_provider_fixture_pack_manifest
from .real_bug_benchmarks import build_real_bug_benchmark_manifest
from .seed_corpus import build_seed_corpus_manifest
from .structured_schema_corpus import build_structured_schema_corpus_manifest
from .team_dashboard import DashboardSnapshot, load_dashboard_history


ROADMAP_REPORT_VERSION = 1
DEFAULT_AS_OF = "2026-06-05T00:00:00+00:00"


class RoadmapError(ValueError):
    """Raised when roadmap evidence cannot be generated from repository assets."""


@dataclass(frozen=True, slots=True)
class RoadmapReport:
    """One deterministic roadmap report rendered as JSON, text, or Markdown."""

    kind: str
    title: str
    summary: str
    as_of: str
    payload: dict[str, object]
    markdown: str

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("ok", True))

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": ROADMAP_REPORT_VERSION,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "as_of": self.as_of,
            "ok": self.ok,
            "payload": self.payload,
        }


def build_historical_trend_report(
    history: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    as_of: str = DEFAULT_AS_OF,
) -> RoadmapReport:
    """Build structural-risk trends from append-only dashboard history and current corpus evidence."""

    root = _repo_root(repo_root)
    snapshots = load_dashboard_history(history)
    if not snapshots:
        snapshots = _fixture_history(root, as_of=as_of)
    current = snapshots[-1]
    first = snapshots[0]
    trend = {
        "open_risks": current.open_risks - first.open_risks,
        "accepted_suppressions": current.accepted_suppressions - first.accepted_suppressions,
        "solver_abstentions": current.solver_abstentions - first.solver_abstentions,
        "drift_warnings": current.drift_warnings - first.drift_warnings,
        "corpus_regressions": current.corpus_regressions - first.corpus_regressions,
    }
    by_rule: Counter[str] = Counter()
    by_artifact_kind: Counter[str] = Counter()
    for snapshot in snapshots:
        for source in snapshot.sources:
            by_rule.update(source.by_rule)
            by_artifact_kind.update(source.by_artifact_kind)
    model_families = build_seed_corpus_manifest(root / "fixtures" / "seed_corpus")
    provider_packs = build_provider_fixture_pack_manifest(root / "fixtures" / "provider_fixture_packs")
    payload: dict[str, object] = {
        "ok": current.open_risks == 0 and current.corpus_regressions == 0,
        "source": "dashboard-history" if history else "repository-fixture-history",
        "history_points": len(snapshots),
        "first_timestamp": first.timestamp,
        "current_timestamp": current.timestamp,
        "trend": trend,
        "current_totals": current.to_dict()["totals"],
        "top_rules": dict(by_rule.most_common(8)),
        "artifact_kinds": dict(sorted(by_artifact_kind.items())),
        "coverage": {
            "model_families": model_families["entry_count"],
            "provider_fixture_packs": provider_packs["entry_count"],
            "provider_families": len(provider_packs["provider_families"]),
            "pipelines": [
                "repository",
                "team",
                "model-family",
                "provider-migration",
                "fine-tuning",
            ],
        },
    }
    markdown = _markdown_report(
        "Historical structural-risk trends",
        (
            "PromptABI turns dashboard JSONL into longitudinal risk reports across repositories, teams, "
            "model families, provider migrations, and fine-tuning pipelines."
        ),
        (
            ("history points", len(snapshots)),
            ("current open risks", current.open_risks),
            ("current corpus regressions", current.corpus_regressions),
            ("model families", model_families["entry_count"]),
            ("provider packs", provider_packs["entry_count"]),
        ),
        payload,
    )
    return RoadmapReport(
        kind="historical-trends",
        title="Historical structural-risk trends",
        summary="Longitudinal PromptABI risk report from dashboard history and live corpus manifests.",
        as_of=as_of,
        payload=payload,
        markdown=markdown,
    )


def build_annual_corpus_refresh_report(
    *,
    repo_root: str | Path | None = None,
    as_of: str = DEFAULT_AS_OF,
) -> RoadmapReport:
    """Plan an annual corpus refresh without mutating or deleting fixtures."""

    root = _repo_root(repo_root)
    seed = build_seed_corpus_manifest(root / "fixtures" / "seed_corpus")
    structured = build_structured_schema_corpus_manifest(root / "fixtures" / "structured_schemas")
    providers = build_provider_fixture_pack_manifest(root / "fixtures" / "provider_fixture_packs")
    real_bugs = build_real_bug_benchmark_manifest(root / "fixtures" / "real_bug_benchmarks" / "benchmark.json")
    verification = run_corpus_verification(
        seed_root=root / "fixtures" / "seed_corpus",
        structured_schema_root=root / "fixtures" / "structured_schemas",
        provider_fixture_root=root / "fixtures" / "provider_fixture_packs",
        real_bug_benchmark_path=root / "fixtures" / "real_bug_benchmarks" / "benchmark.json",
        evaluation_corpus_path=root / "fixtures" / "evaluation" / "labeled_corpus.json",
    ).to_dict()
    actions = (
        {
            "id": "retire-obsolete-artifacts",
            "mode": "plan-only",
            "evidence": "compare fixture versions, expected diagnostics, and upstream fixed-version metadata",
            "preservation": "move retired cases to benchmark archives only after baseline hashes are recorded",
        },
        {
            "id": "add-new-model-families",
            "mode": "review-gated",
            "evidence": "tokenizer_config, chat-template metadata, provenance, license, and expected diagnostics",
            "preservation": "old model-family fixtures remain in longitudinal seed-corpus manifests",
        },
        {
            "id": "update-provider-semantics",
            "mode": "compatibility-gated",
            "evidence": "provider fixture packs for streaming, parallel tools, JSON mode, stops, errors, and context limits",
            "preservation": "provider revisions get additive fixture-pack entries instead of overwriting old packs",
        },
        {
            "id": "preserve-old-benchmarks",
            "mode": "release-blocking",
            "evidence": "leaderboard, real-bug benchmark, comparative study, and evaluation-reproducibility outputs",
            "preservation": "benchmark manifests retain old release rows for longitudinal comparisons",
        },
    )
    payload: dict[str, object] = {
        "ok": verification["ok"] is True,
        "read_only": True,
        "corpora": {
            "seed_model_families": seed["entry_count"],
            "structured_schema_cases": structured["entry_count"],
            "provider_fixture_packs": providers["entry_count"],
            "real_bug_cases": real_bugs["case_count"],
        },
        "verification": {
            "ok": verification["ok"],
            "checks": verification["check_count"],
            "failures": sum(1 for check in verification["checks"] if not check["passed"]),
        },
        "annual_actions": list(actions),
        "release_gates": [
            "promptabi corpus verify --format text",
            "promptabi release compatibility-audit ...",
            "promptabi corpus leaderboard --format text",
            "promptabi maintain refresh --output-dir ... --force",
        ],
    }
    markdown = _markdown_report(
        "Annual corpus refresh procedure",
        "A read-only refresh plan that updates PromptABI's corpus without losing longitudinal evidence.",
        (
            ("seed model families", seed["entry_count"]),
            ("structured schema cases", structured["entry_count"]),
            ("provider fixture packs", providers["entry_count"]),
            ("real bug cases", real_bugs["case_count"]),
        ),
        payload,
    )
    return RoadmapReport(
        kind="annual-corpus-refresh",
        title="Annual corpus refresh procedure",
        summary="Read-only annual corpus refresh plan with retirement, addition, provider-update, and archive gates.",
        as_of=as_of,
        payload=payload,
        markdown=markdown,
    )


def build_award_submission_report(
    *,
    repo_root: str | Path | None = None,
    as_of: str = DEFAULT_AS_OF,
) -> RoadmapReport:
    """Build concise award-submission material from live repository evidence."""

    root = _repo_root(repo_root)
    comparative = build_comparative_study_report(
        evaluation_corpus_path=root / "fixtures" / "evaluation" / "labeled_corpus.json",
        real_bug_benchmark_path=root / "fixtures" / "real_bug_benchmarks" / "benchmark.json",
    ).to_dict()
    leaderboard = build_benchmark_leaderboard(
        performance_cases=("tokenizer-analysis", "stop-checks", "grammar-emptiness"),
        benchmark_iterations=1,
        repo_root=root,
    ).to_dict()
    demos = run_conference_demos(root).to_dict()
    entry = leaderboard["entries"][0]
    payload: dict[str, object] = {
        "ok": comparative["passed"] is True and leaderboard["ok"] is True and demos["ok"] is True,
        "claim": (
            "PromptABI is a CPU-only static verifier for LLM interface contracts that finds structural "
            "bugs before inference, fine-tuning, evaluation publication, provider migration, and deployment."
        ),
        "limitations": [
            "does not prove model intent or sampled behavior",
            "claims are scoped to explicit supported fragments",
            "unsupported artifacts produce visible abstentions instead of false proofs",
        ],
        "technical_depth": [
            "bounded symbolic chat-template execution",
            "finite tokenizer/grammar/parser products",
            "SMT-backed finite contracts",
            "differential replay against real adapters and corpora",
        ],
        "impact_evidence": {
            "comparative_case_count": comparative["case_count"],
            "promptabi_detected_cases": comparative["promptabi_detected_cases"],
            "baseline_classes_with_misses": sum(1 for item in comparative["baselines"] if item["missed_case_count"] > 0),
            "real_bug_cases": comparative["real_bug_cases"],
            "demo_scenarios": demos["summary"]["scenarios"],
            "precision": entry["quality"]["precision"],
            "recall": entry["quality"]["recall"],
            "solver_reliability": entry["solver"]["reliability"],
        },
        "reproducibility": [
            "promptabi corpus comparative-study --format markdown",
            "promptabi corpus leaderboard --benchmark-iterations 1 --format text",
            "promptabi conference-demo --format text",
            "promptabi paper reproducibility --output-dir paper_artifact --force",
        ],
    }
    markdown = (
        "# Award submission brief\n\n"
        f"**Claim.** {payload['claim']}\n\n"
        "## Evidence\n\n"
        f"- Comparative cases: {comparative['case_count']} with PromptABI detecting "
        f"{comparative['promptabi_detected_cases']}.\n"
        f"- Real-bug benchmark cases: {comparative['real_bug_cases']}.\n"
        f"- Stage-ready demos: {demos['summary']['scenarios']}.\n"
        f"- Leaderboard precision/recall: {entry['quality']['precision']}/"
        f"{entry['quality']['recall']}; solver reliability {entry['solver']['reliability']}.\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {item}" for item in payload["limitations"])
        + "\n\n## Reproduction\n\n"
        + "\n".join(f"- `{cmd}`" for cmd in payload["reproducibility"])
        + "\n"
    )
    return RoadmapReport(
        kind="award-submission",
        title="Award submission brief",
        summary="Evidence-bound award material for PromptABI's claims, limits, impact, and reproducibility.",
        as_of=as_of,
        payload=payload,
        markdown=markdown,
    )


def build_teaching_materials_report(
    *,
    repo_root: str | Path | None = None,
    as_of: str = DEFAULT_AS_OF,
) -> RoadmapReport:
    """Build teaching materials from existing examples, docs, and proof notebooks."""

    root = _repo_root(repo_root)
    notebooks = sorted((root / "examples" / "proof-sketch-notebooks").glob("*.ipynb"))
    labs = (
        ("role-boundary", "Tokenizer/template non-forgeability", "examples/role-boundary/unsafe.promptabi.json"),
        ("tool-calling", "Tool-call stop and serialization contracts", "examples/end-to-end/tool-calling/buggy.promptabi.json"),
        ("rag-budget", "Must-survive RAG and truncation budgets", "examples/end-to-end/rag-truncation/buggy.promptabi.json"),
        ("smt-contracts", "SMT-backed static contracts", "examples/static-contract-language/app.pabi"),
        ("provider-migration", "Provider-standard conformance and migration", "examples/end-to-end/provider-migration/buggy.promptabi.json"),
    )
    modules = (
        "LLM interface artifacts as compiler boundaries",
        "Tokenizer/template/tool parser products",
        "SMT-backed finite contracts and honest abstention",
        "Training and evaluation contract preflight",
        "Operationalizing proof evidence in CI, IDEs, registries, and deployments",
    )
    payload: dict[str, object] = {
        "ok": len(notebooks) >= 5 and all((root / lab[2]).exists() for lab in labs),
        "course_length_weeks": 5,
        "audiences": ["university PL/security courses", "internal AI platform training", "open-source maintainer workshops"],
        "modules": list(modules),
        "labs": [{"id": lab_id, "title": title, "artifact": artifact} for lab_id, title, artifact in labs],
        "proof_notebooks": [str(path.relative_to(root)) for path in notebooks],
        "assessment": [
            "write one unsafe fixture and one fixed fixture",
            "explain one diagnostic's supported fragment and non-goal",
            "derive one finite contract or tokenizer/grammar product by hand",
        ],
    }
    markdown = (
        "# Teaching materials\n\n"
        "PromptABI teaching material is built from executable examples and proof notebooks, not slides detached from code.\n\n"
        "## Modules\n\n"
        + "\n".join(f"{index}. {module}" for index, module in enumerate(modules, start=1))
        + "\n\n## Labs\n\n"
        + "\n".join(f"- **{title}:** `{artifact}`" for _, title, artifact in labs)
        + "\n\n## Proof notebooks\n\n"
        + "\n".join(f"- `{path}`" for path in payload["proof_notebooks"])
        + "\n"
    )
    return RoadmapReport(
        kind="teaching-materials",
        title="Teaching materials",
        summary="University and internal-training syllabus tied to executable PromptABI examples and notebooks.",
        as_of=as_of,
        payload=payload,
        markdown=markdown,
    )


def build_research_agenda_report(
    *,
    repo_root: str | Path | None = None,
    as_of: str = DEFAULT_AS_OF,
) -> RoadmapReport:
    """Build the next 100 research-roadmap steps for PromptABI."""

    root = _repo_root(repo_root)
    steps = next_research_agenda_steps()
    categories = Counter(step["category"] for step in steps)
    payload: dict[str, object] = {
        "ok": len(steps) == 100,
        "step_range": [200, 299],
        "categories": dict(sorted(categories.items())),
        "tracked_docs": ["docs/research-agenda.md"],
        "steps": steps,
    }
    markdown = (
        "# Long-term research agenda\n\n"
        "The next 100 steps aim at a best-paper-award/1000-star PromptABI: compositional verification, "
        "richer solver fragments, certified prompt packs, training-data contracts, provider-standard "
        "conformance, deployment evidence, and community-scale reproducibility.\n\n"
        + "\n".join(f"{step['number']}. {step['title']}" for step in steps)
        + "\n"
    )
    if not (root / "docs").is_dir():
        raise RoadmapError(f"repository docs directory is missing: {root / 'docs'}")
    return RoadmapReport(
        kind="research-agenda",
        title="Long-term research agenda",
        summary="Next 100 tracked research and engineering steps toward award-caliber PromptABI.",
        as_of=as_of,
        payload=payload,
        markdown=markdown,
    )


def write_roadmap_document(report: RoadmapReport, output: str | Path, *, force: bool = False) -> Path:
    """Write a tracked Markdown roadmap document."""

    path = Path(output)
    if path.exists() and not force:
        raise RoadmapError(f"refusing to overwrite existing roadmap document without force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown, encoding="utf-8")
    return path


def render_roadmap_json(report: RoadmapReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_roadmap_text(report: RoadmapReport) -> str:
    lines = [
        f"PromptABI {report.title}",
        f"as_of: {report.as_of}",
        f"status: {'PASS' if report.ok else 'ATTENTION'}",
        report.summary,
    ]
    highlights = report.payload.get("coverage") or report.payload.get("corpora") or report.payload.get("impact_evidence")
    if isinstance(highlights, dict):
        for key, value in list(highlights.items())[:8]:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def render_roadmap_markdown(report: RoadmapReport) -> str:
    return report.markdown


def next_research_agenda_steps() -> list[dict[str, object]]:
    """Return 100 deterministic next steps grouped by research thrust."""

    categories: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "compositional verification",
            (
                "formalize cross-artifact assume/guarantee contracts",
                "compose chat-template and tokenizer proofs across prompt packs",
                "prove provider-envelope preservation under adapter chains",
                "model streaming parser state machines as first-class products",
                "add counterexample slicing across multi-artifact products",
                "certify monotonicity for safe prompt-pack extension",
                "track proof obligations through config inheritance",
                "compose RAG chunk policies with tool schemas",
                "prove incremental verification cache soundness",
                "derive minimal unsat cores for safe deployments",
                "add contract witnesses for multi-agent handoffs",
                "verify nested tool-call encodings",
                "compose runtime attestations with deployment gates",
                "prove local policy packs preserve checker semantics",
                "add abstract interpretation for supported template dialects",
                "publish compositional proof benchmarks",
                "connect theorem traceability to release blockers",
                "add verified downgrade paths for provider migrations",
                "model prompt-pack imports as module systems",
                "prove diagnostic stability under formatting-only changes",
            ),
        ),
        (
            "richer solver fragments",
            (
                "support bounded arrays in static contracts",
                "add string-prefix and suffix lemmas for stop policies",
                "encode finite map constraints for provider envelopes",
                "support arithmetic over packed token budgets",
                "cache solver lemmas by normalized artifact products",
                "add solver portfolio replay metadata",
                "minimize SMT models into human-readable witnesses",
                "certify abstention reasons for unsupported formulas",
                "benchmark quantified-pattern approximations",
                "add bit-vector encodings for token IDs",
                "prove solver-version compatibility gates",
                "derive interpolants for failed migration checks",
                "add proof-carrying solver cache entries",
                "support lexicographic constraints for ordered messages",
                "connect SMT counterexamples to source spans",
                "add timeout-sensitive degradation tests",
                "publish solver-fragment conformance suites",
                "model finite Unicode normalization constraints",
                "support cross-field JSON schema obligations",
                "mechanize the smallest solver encodings",
            ),
        ),
        (
            "certified prompt packs",
            (
                "define prompt-pack capability signatures",
                "certify prompt-pack upgrade compatibility",
                "add transitive lockfile proofs for prompt-pack registries",
                "support private prompt-pack transparency logs",
                "verify prompt-pack policy inheritance",
                "add marketplace trust tiers",
                "certify prompt-pack examples against supported model families",
                "prove exported roles cannot be forged by consumers",
                "add prompt-pack deprecation and LTS metadata",
                "support signed offline prompt-pack mirrors",
                "add package-level differential provider fixtures",
                "verify prompt-pack RAG extension points",
                "certify structured-output prompt-pack schemas",
                "add vulnerability advisory format for prompt packs",
                "integrate prompt-pack provenance with model registries",
                "define prompt-pack semantic version impact rules",
                "add consumer-side override safety checks",
                "publish reusable certified demo packs",
                "add third-party prompt-pack certification tests",
                "build a prompt-pack interoperability leaderboard",
            ),
        ),
        (
            "training-data contracts",
            (
                "verify curriculum-stage prompt ABI drift",
                "add packing proofs for multi-epoch fine-tuning",
                "model dataset transforms as contract-preserving passes",
                "certify loss-mask semantics across loaders",
                "add RLHF judge prompt privacy checks",
                "verify preference-pair role symmetry",
                "detect benchmark-answer leakage in synthetic data",
                "prove training/eval tokenizer alignment over releases",
                "add streaming-shard witness replay",
                "model multi-modal placeholders as interface artifacts",
                "verify supervised target spans after truncation",
                "add data-loader conformance badges",
                "support dataset-card PromptABI metadata",
                "verify distillation prompt packs",
                "add training-contract drift alarms",
                "connect fine-tune manifests to model registry gates",
                "add red-team corpus refresh loops",
                "prove private-field redaction survives packing",
                "support federated corpus validation manifests",
                "publish training-contract benchmark suites",
            ),
        ),
        (
            "provider-standard conformance",
            (
                "define an open provider-contract test vector format",
                "add conformance badges for OpenAI-compatible servers",
                "model provider-native grammar backends",
                "verify multi-turn tool-call replay semantics",
                "add streaming chunk order robustness checks",
                "track context-window semantics by provider revision",
                "certify error-envelope compatibility",
                "add provider migration dry-run patches",
                "publish provider fixture provenance attestations",
                "support regional and enterprise provider variants",
                "verify structured-output refusal envelopes",
                "add provider capability negotiation contracts",
                "model parallel tool-call cancellation",
                "prove stop policy equivalence across providers",
                "add provider benchmark drift dashboards",
                "support standard conformance issue templates",
                "connect provider conformance to deployment gates",
                "add language SDK provider-contract readers",
                "publish annual provider semantics reports",
                "drive a shared prompt-interface standards proposal",
            ),
        ),
    )
    steps: list[dict[str, object]] = []
    number = 200
    for category, titles in categories:
        for title in titles:
            steps.append({"number": number, "category": category, "title": title.capitalize() + "."})
            number += 1
    return steps


def _repo_root(repo_root: str | Path | None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.is_dir():
        raise RoadmapError(f"repository root does not exist: {root}")
    return root


def _fixture_history(root: Path, *, as_of: str) -> tuple[DashboardSnapshot, ...]:
    dashboard_path = root / "fixtures" / "roadmap" / "dashboard-history.jsonl"
    history = load_dashboard_history(dashboard_path)
    if history:
        return history
    return (
        DashboardSnapshot.from_dict(
            {
                "schema_version": 1,
                "timestamp": "2025-01-01T00:00:00+00:00",
                "totals": {
                    "open_risks": 9,
                    "accepted_suppressions": 2,
                    "solver_abstentions": 4,
                    "drift_warnings": 5,
                    "corpus_regressions": 3,
                    "corpus_checks": 18,
                },
                "sources": [
                    {
                        "name": "research-lab",
                        "diagnostics": 12,
                        "open_risks": 9,
                        "accepted_suppressions": 2,
                        "solver_abstentions": 4,
                        "drift_warnings": 5,
                        "by_rule": {"role-boundary-nonforgeability": 4, "stop-overreach-content": 3},
                        "by_severity": {"error": 9, "warning": 3},
                        "by_artifact_kind": {"chat-template": 4, "provider-config": 3, "training-manifest": 2},
                    }
                ],
            }
        ),
        DashboardSnapshot.from_dict(
            {
                "schema_version": 1,
                "timestamp": as_of,
                "totals": {
                    "open_risks": 0,
                    "accepted_suppressions": 0,
                    "solver_abstentions": 1,
                    "drift_warnings": 0,
                    "corpus_regressions": 0,
                    "corpus_checks": 42,
                },
                "sources": [
                    {
                        "name": "research-lab",
                        "diagnostics": 3,
                        "open_risks": 0,
                        "accepted_suppressions": 0,
                        "solver_abstentions": 1,
                        "drift_warnings": 0,
                        "by_rule": {"grammar-emptiness": 1, "provider-migration": 1},
                        "by_severity": {"info": 2, "warning": 1},
                        "by_artifact_kind": {"grammar": 1, "provider-config": 1, "training-manifest": 1},
                    }
                ],
            }
        ),
    )


def _markdown_report(
    title: str,
    intro: str,
    metrics: tuple[tuple[str, object], ...],
    payload: dict[str, object],
) -> str:
    metric_lines = "\n".join(f"- **{name}:** {value}" for name, value in metrics)
    commands = payload.get("release_gates") or payload.get("reproducibility") or ()
    command_block = ""
    if isinstance(commands, list) and commands:
        command_block = "\n## Commands\n\n" + "\n".join(f"- `{command}`" for command in commands) + "\n"
    return f"# {title}\n\n{intro}\n\n## Metrics\n\n{metric_lines}\n{command_block}"
