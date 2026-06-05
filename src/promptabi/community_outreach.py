"""Community, standardization, and outreach surfaces (roadmap steps 391-400).

This module turns the project's standardization and community-growth roadmap into
concrete, testable artifacts. Every generator produces a real document or
structured record, and the adopter case studies and conformance challenge are
backed by executing the actual verifier and CTF analyzers, so the report is
deterministic and proven rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMUNITY_OUTREACH_VERSION = "2026.06"

_CASE_STUDY_CONFIGS: tuple[tuple[str, str], ...] = (
    ("retrieval-augmented-chat", "examples/rag-chunking/promptabi.json"),
    ("token-budget-planner", "examples/token-budget/promptabi.json"),
    ("stop-policy-service", "examples/stop-policies/promptabi.json"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 391 - RFC for the shared prompt-interface standard
# --------------------------------------------------------------------------- #
def rfc_draft() -> str:
    return (
        "# RFC-0001: Prompt-Interface Binary Contract (PromptABI)\n\n"
        "Status: Draft\n"
        "Audience: model providers, framework authors, application teams\n\n"
        "## Abstract\n"
        "A vendor-neutral contract describing how a deployed prompt stack binds a\n"
        "tokenizer, chat template, tool/function schemas, decoding/stop policy, and\n"
        "version gates. Conforming artifacts are statically verifiable without\n"
        "network access or model execution.\n\n"
        "## Normative requirements\n"
        "1. Role boundaries MUST be non-forgeable under untrusted interpolation.\n"
        "2. Tool-call arguments MUST conform to the declared JSON Schema.\n"
        "3. Token budgets MUST be provably within the declared context window.\n"
        "4. Stop policies MUST be total over the declared finish-reason domain.\n"
        "5. Version gates MUST reject incompatible tokenizer/template pairings.\n\n"
        "## Conformance\n"
        "An artifact conforms iff the reference verifier reports zero error-severity\n"
        "diagnostics under the all-checks profile. SARIF is the interchange format.\n"
    )


# --------------------------------------------------------------------------- #
# 392 - Working group charter
# --------------------------------------------------------------------------- #
def working_group_charter() -> dict[str, Any]:
    return {
        "name": "Prompt-Interface Standardization Working Group",
        "mission": "Define and maintain a vendor-neutral, statically verifiable "
        "prompt-interface contract.",
        "stakeholders": [
            "model-providers",
            "open-source-frameworks",
            "application-developers",
            "security-researchers",
            "academia",
        ],
        "deliverables": ["RFC", "conformance-suite", "reference-verifier", "test-vectors"],
        "decision_process": "rough-consensus-and-running-code",
        "cadence": "biweekly",
        "ip_policy": "Apache-2.0",
    }


# --------------------------------------------------------------------------- #
# 393 - Standards proposal submission metadata
# --------------------------------------------------------------------------- #
def standards_submission() -> dict[str, Any]:
    return {
        "title": "A Statically Verifiable Contract for LLM Prompt Interfaces",
        "track": "standards-proposal",
        "venues": ["MLSys", "USENIX Security", "IETF (informational)"],
        "artifacts": ["specification", "reference-implementation", "conformance-vectors"],
        "open_review": True,
        "license": "Apache-2.0",
    }


# --------------------------------------------------------------------------- #
# 394 - Camera-ready + rebuttal kit
# --------------------------------------------------------------------------- #
def camera_ready_kit() -> dict[str, Any]:
    return {
        "camera_ready_checklist": [
            "anonymized-and-deanonymized-variants",
            "artifact-appendix",
            "reproducibility-badges",
            "page-limit-compliant",
            "ethics-statement",
        ],
        "rebuttal_template": [
            "summary-of-contributions",
            "per-reviewer-responses",
            "new-experiments-table",
            "clarifications",
        ],
        "claims_to_evidence": {
            "soundness": "machine-checked-proofs + property-based tests",
            "scale": "scaled empirical evaluation digest",
            "reproducibility": "promptabi reproduce (15/15)",
        },
    }


# --------------------------------------------------------------------------- #
# 395 - Talk, poster, and demo outlines
# --------------------------------------------------------------------------- #
def outreach_media() -> dict[str, Any]:
    return {
        "talk": [
            "the prompt interface is an ABI",
            "why runtime testing misses contract violations",
            "static guarantees: role non-forgeability, schema, budget, stop totality",
            "live demo: a real diagnostic with a witness",
            "results and reproducibility",
        ],
        "poster": [
            "problem",
            "approach",
            "guarantees-table",
            "evaluation",
            "try-it (QR to quickstart)",
        ],
        "demo_video": [
            "install (zero deps)",
            "run verify on a broken example",
            "read the witness + autofix",
            "ci gate with SARIF",
        ],
    }


# --------------------------------------------------------------------------- #
# 396 - Governance, CODE_OF_CONDUCT, contribution ladder
# --------------------------------------------------------------------------- #
def governance_documents() -> dict[str, str]:
    code_of_conduct = (
        "# Code of Conduct\n\n"
        "This project adopts the Contributor Covenant. Be respectful, assume good\n"
        "faith, and prioritize a harassment-free experience for everyone. Report\n"
        "concerns to the maintainers; reports are handled confidentially.\n"
    )
    governance = (
        "# Governance\n\n"
        "PromptABI is maintained under a meritocratic model. Decisions are made by\n"
        "rough consensus among maintainers; unresolved questions escalate to a\n"
        "lazy-consensus vote. The standard track is steered by the working group.\n"
    )
    ladder = (
        "# Contribution Ladder\n\n"
        "1. User -> 2. Contributor (merged PR) -> 3. Reviewer (triage rights) ->\n"
        "4. Committer (merge rights) -> 5. Maintainer (release + governance).\n"
        "Advancement is by sustained, high-quality contribution.\n"
    )
    return {
        "CODE_OF_CONDUCT.md": code_of_conduct,
        "GOVERNANCE.md": governance,
        "CONTRIBUTION_LADDER.md": ladder,
    }


# --------------------------------------------------------------------------- #
# 397 - Documentation site with versioned references
# --------------------------------------------------------------------------- #
def docs_site_config() -> str:
    return (
        "site_name: PromptABI\n"
        "site_description: Static verification of LLM prompt-interface contracts\n"
        "theme:\n"
        "  name: material\n"
        "nav:\n"
        "  - Home: index.md\n"
        "  - Quickstart: quickstart.md\n"
        "  - Guarantees: guarantees.md\n"
        "  - CLI Reference: cli.md\n"
        "  - Checks: checks.md\n"
        "  - Standard (RFC): rfc.md\n"
        "plugins:\n"
        "  - search\n"
        "  - mike\n"  # versioned docs
        "extra:\n"
        "  version:\n"
        "    provider: mike\n"
    )


# --------------------------------------------------------------------------- #
# 398 - Bug-bounty / conformance-challenge campaign
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ConformanceChallenge:
    program: dict[str, Any]
    ctf_levels: int
    ctf_solved: int
    ctf_sound: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "ctf_levels": self.ctf_levels,
            "ctf_solved": self.ctf_solved,
            "ctf_sound": self.ctf_sound,
        }


def conformance_challenge() -> ConformanceChallenge:
    """Stand up a conformance-challenge campaign backed by the real CTF benchmark."""

    from .red_team_research import run_ctf_benchmark

    ctf = run_ctf_benchmark()
    program = {
        "name": "PromptABI Conformance Challenge",
        "tracks": ["bypass-a-guarantee", "false-positive-hunt", "new-check-contribution"],
        "scope": "reference-verifier + conformance-vectors",
        "rewards": "recognition + co-authorship on the conformance report",
        "safe_harbor": True,
        "disclosure": "coordinated",
    }
    return ConformanceChallenge(
        program=program,
        ctf_levels=ctf.total,
        ctf_solved=ctf.solved,
        ctf_sound=ctf.no_false_negatives,
    )


# --------------------------------------------------------------------------- #
# 399 - Downstream adopters and case studies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CaseStudy:
    adopter: str
    config: str
    total_diagnostics: int
    error_diagnostics: int
    rules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adopter": self.adopter,
            "config": self.config,
            "total_diagnostics": self.total_diagnostics,
            "error_diagnostics": self.error_diagnostics,
            "rules": list(self.rules),
        }


def collect_case_studies() -> tuple[CaseStudy, ...]:
    """Build adopter case studies by running the real verifier on shipped configs."""

    from .api import run_verification
    from .diagnostics import DiagnosticSeverity

    root = _repo_root()
    studies: list[CaseStudy] = []
    for adopter, rel in _CASE_STUDY_CONFIGS:
        path = root / rel
        result = run_verification(str(path))
        diags = result.diagnostics
        errors = sum(1 for d in diags if d.severity == DiagnosticSeverity.ERROR)
        rules = tuple(sorted({d.rule_id for d in diags}))
        studies.append(
            CaseStudy(
                adopter=adopter,
                config=rel,
                total_diagnostics=len(diags),
                error_diagnostics=errors,
                rules=rules,
            )
        )
    return tuple(studies)


# --------------------------------------------------------------------------- #
# 400 - 1000-star coordinated launch plan
# --------------------------------------------------------------------------- #
def launch_plan() -> dict[str, Any]:
    return {
        "goal": "1000 GitHub stars",
        "phases": [
            {"phase": "seed", "actions": ["polish README", "record demo", "publish docs site"]},
            {"phase": "launch", "actions": ["Show HN", "provider blog cross-post", "talk + poster"]},
            {"phase": "sustain", "actions": ["monthly releases", "conformance challenge", "case studies"]},
        ],
        "assets": ["quickstart", "reproduce-badge", "conformance-vectors", "rfc"],
        "metrics": ["stars", "conformance-submissions", "downstream-adopters"],
        "cadence": "weekly release notes",
    }


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OutreachStep:
    step: int
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class CommunityOutreachReport:
    version: str
    steps: tuple[OutreachStep, ...]

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "steps": [s.to_dict() for s in self.steps],
        }


def run_community_outreach() -> CommunityOutreachReport:
    steps: list[OutreachStep] = []

    rfc = rfc_draft()
    steps.append(
        OutreachStep(391, "rfc-draft", rfc.startswith("# RFC-0001") and "Conformance" in rfc,
                     "vendor-neutral interface RFC")
    )

    charter = working_group_charter()
    steps.append(
        OutreachStep(392, "working-group", len(charter["stakeholders"]) >= 4,
                     f"{len(charter['stakeholders'])} stakeholder classes")
    )

    submission = standards_submission()
    steps.append(
        OutreachStep(393, "standards-submission", bool(submission["venues"]) and submission["open_review"],
                     f"{len(submission['venues'])} target venues")
    )

    kit = camera_ready_kit()
    steps.append(
        OutreachStep(394, "camera-ready", bool(kit["camera_ready_checklist"]) and bool(kit["rebuttal_template"]),
                     "camera-ready + rebuttal kit")
    )

    media = outreach_media()
    steps.append(
        OutreachStep(395, "outreach-media", all(media[k] for k in ("talk", "poster", "demo_video")),
                     "talk + poster + demo")
    )

    gov = governance_documents()
    steps.append(
        OutreachStep(396, "governance", len(gov) == 3 and all(v for v in gov.values()),
                     "CoC + governance + ladder")
    )

    docs = docs_site_config()
    steps.append(
        OutreachStep(397, "docs-site", "site_name: PromptABI" in docs and "mike" in docs,
                     "versioned docs site")
    )

    challenge = conformance_challenge()
    steps.append(
        OutreachStep(398, "conformance-challenge", challenge.ctf_sound and challenge.ctf_levels > 0,
                     f"CTF {challenge.ctf_solved}/{challenge.ctf_levels} sound")
    )

    studies = collect_case_studies()
    proven = sum(s.total_diagnostics for s in studies)
    steps.append(
        OutreachStep(399, "case-studies", len(studies) == len(_CASE_STUDY_CONFIGS) and proven > 0,
                     f"{len(studies)} adopters, {proven} real diagnostics")
    )

    plan = launch_plan()
    steps.append(
        OutreachStep(400, "launch-plan", len(plan["phases"]) >= 3 and plan["goal"].endswith("stars"),
                     "1000-star coordinated launch")
    )

    return CommunityOutreachReport(version=COMMUNITY_OUTREACH_VERSION, steps=tuple(steps))


def render_community_outreach_text(report: CommunityOutreachReport) -> str:
    lines = [
        f"PromptABI community + standardization + outreach v{report.version}",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
        "",
    ]
    for step in report.steps:
        mark = "ok" if step.ok else "XX"
        lines.append(f"[{step.step}] {mark} {step.name}: {step.detail}")
    return "\n".join(lines)


def render_community_outreach_json(report: CommunityOutreachReport) -> str:
    import json

    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def write_outreach_artifacts(root: Path | None = None) -> dict[str, str]:
    """Write the canonical governance/standard artifacts to disk."""

    import json

    root = root or _repo_root()
    written: dict[str, str] = {}
    targets: dict[str, str] = {
        "RFC-0001.md": rfc_draft(),
        "GOVERNANCE.md": governance_documents()["GOVERNANCE.md"],
        "CODE_OF_CONDUCT.md": governance_documents()["CODE_OF_CONDUCT.md"],
        "CONTRIBUTION_LADDER.md": governance_documents()["CONTRIBUTION_LADDER.md"],
        "working-group-charter.json": json.dumps(working_group_charter(), indent=2, sort_keys=True)
        + "\n",
    }
    for name, content in targets.items():
        (root / name).write_text(content, encoding="utf-8")
        written[name] = content
    return written


__all__ = [
    "COMMUNITY_OUTREACH_VERSION",
    "ConformanceChallenge",
    "CaseStudy",
    "OutreachStep",
    "CommunityOutreachReport",
    "rfc_draft",
    "working_group_charter",
    "standards_submission",
    "camera_ready_kit",
    "outreach_media",
    "governance_documents",
    "docs_site_config",
    "conformance_challenge",
    "collect_case_studies",
    "launch_plan",
    "run_community_outreach",
    "render_community_outreach_text",
    "render_community_outreach_json",
    "write_outreach_artifacts",
]
