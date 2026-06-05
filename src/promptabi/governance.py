"""Governance policy for accepting and releasing PromptABI checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


GOVERNANCE_POLICY_VERSION = 1
REQUIRED_GOVERNANCE_DOCS = (
    "docs/governance.md",
    "docs/contributing/checker-design.md",
    "docs/contributing/corpus-contributions.md",
    "docs/security-model.md",
)
REQUIRED_RELEASE_BLOCKERS = (
    "unsound-safe-result",
    "missing-abstention",
    "witness-replay-failure",
    "secret-bearing-fixture",
    "license-incompatible-corpus",
    "regression-on-labeled-real-bug",
)


@dataclass(frozen=True, slots=True)
class GovernancePrinciple:
    """One policy principle that reviewers can enforce."""

    id: str
    title: str
    standard: str
    evidence: tuple[str, ...]
    release_blocking_when: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "standard": self.standard,
            "evidence": list(self.evidence),
            "release_blocking_when": list(self.release_blocking_when),
        }


@dataclass(frozen=True, slots=True)
class GovernanceReport:
    """Repository-level validation of the documented governance contract."""

    repo_root: Path
    checked_paths: tuple[str, ...]
    principles: tuple[GovernancePrinciple, ...]
    release_blockers: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": GOVERNANCE_POLICY_VERSION,
            "ok": self.ok,
            "checked_paths": list(self.checked_paths),
            "principle_count": len(self.principles),
            "release_blockers": list(self.release_blockers),
            "issues": list(self.issues),
            "principles": [principle.to_dict() for principle in self.principles],
        }


class GovernanceError(ValueError):
    """Raised when governance validation cannot inspect a repository."""


def build_governance_principles() -> tuple[GovernancePrinciple, ...]:
    """Return PromptABI's stable checker and release governance policy."""

    return (
        GovernancePrinciple(
            id="checker-acceptance",
            title="Checker acceptance criteria",
            standard=(
                "A new checker must state the structural property, supported fragment, "
                "guarantee mode, artifact inputs, source-span obligations, and abstention boundary."
            ),
            evidence=(
                "design note in docs/contributing/checker-design.md or a linked proposal",
                "typed public API or clearly internal module boundary",
                "text/JSON diagnostic snapshots with deterministic ordering",
                "safe, unsafe, ambiguous, unsupported, and malformed fixtures where applicable",
            ),
            release_blocking_when=(
                "a checker emits a safe result outside its supported fragment",
                "a malformed artifact silently passes instead of diagnosing or abstaining",
            ),
        ),
        GovernancePrinciple(
            id="proof-standards",
            title="Proof and evidence standards",
            standard=(
                "Sound or complete claims require executable witnesses, replayable products, "
                "property tests, and differential evidence whenever external library semantics are modeled."
            ),
            evidence=(
                "witness replay tests or executable specs",
                "property-based or generated cases for proof obligations",
                "solver replay files for SMT-backed findings when private artifacts cannot be shared",
                "differential fixtures against real tokenizer, provider, grammar, or framework behavior",
            ),
            release_blocking_when=(
                "a counterexample witness cannot be replayed",
                "a theorem-to-test traceability link is missing for a central proof claim",
            ),
        ),
        GovernancePrinciple(
            id="corpus-licensing",
            title="Corpus licensing and provenance",
            standard=(
                "Every corpus entry must be reproducible offline, license-compatible, pinned to a revision "
                "or synthetic generator, and free of secrets, private prompts, and customer data."
            ),
            evidence=(
                "provenance, license, revision/hash, and expected diagnostics",
                "sanitization statement for real-world fixtures",
                "non-sensitive minimized witness or hash-only proof summary",
                "fixture-pack validation and corpus replay command",
            ),
            release_blocking_when=(
                "a fixture lacks license provenance",
                "a fixture contains credential-like or private payloads",
            ),
        ),
        GovernancePrinciple(
            id="security-disclosure",
            title="Security disclosure workflow",
            standard=(
                "Sensitive structural vulnerabilities are reported privately first; public issues must use "
                "minimized, sanitized artifacts and avoid raw private prompts or provider credentials."
            ),
            evidence=(
                "docs/security-model.md responsible disclosure workflow",
                "sanitized bug-report output when upstreaming third-party findings",
                "private advisory path for PromptABI vulnerabilities",
            ),
            release_blocking_when=(
                "a release artifact leaks secrets or private witness text",
                "a known sensitive vulnerability lacks a disclosure path or owner",
            ),
        ),
        GovernancePrinciple(
            id="release-regressions",
            title="Release-blocking regressions",
            standard=(
                "A release is blocked by unsound safe results, missing abstentions, witness replay failures, "
                "privacy regressions, license-incompatible corpus entries, or labeled real-bug regressions."
            ),
            evidence=(
                "targeted tests for touched checkers",
                "corpus verification or focused conformance suite for affected surfaces",
                "release readiness report before tagged releases",
            ),
            release_blocking_when=REQUIRED_RELEASE_BLOCKERS,
        ),
    )


def validate_governance(repo_root: str | Path | None = None) -> GovernanceReport:
    """Validate that repository docs expose and enforce the governance policy."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    if not root.exists():
        raise GovernanceError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise GovernanceError(f"repository root is not a directory: {root}")

    principles = build_governance_principles()
    issues: list[str] = []
    checked_paths: list[str] = []
    texts: dict[str, str] = {}
    for relative in REQUIRED_GOVERNANCE_DOCS:
        checked_paths.append(relative)
        path = root / relative
        if not path.is_file():
            issues.append(f"{relative}: missing governance document")
            continue
        texts[relative] = path.read_text(encoding="utf-8")

    governance_text = texts.get("docs/governance.md", "")
    for principle in principles:
        if principle.id not in governance_text:
            issues.append(f"docs/governance.md: missing principle id {principle.id}")
        if principle.title.lower() not in governance_text.lower():
            issues.append(f"docs/governance.md: missing principle title {principle.title}")
    for blocker in REQUIRED_RELEASE_BLOCKERS:
        if blocker not in governance_text:
            issues.append(f"docs/governance.md: missing release blocker {blocker}")

    expected_terms = {
        "docs/contributing/checker-design.md": ("supported fragment", "abstention", "witness", "release-blocking"),
        "docs/contributing/corpus-contributions.md": ("license", "provenance", "no secrets", "release-blocking"),
        "docs/security-model.md": ("Responsible disclosure", "private security advisory", "sanitized"),
    }
    for relative, terms in expected_terms.items():
        lower_text = texts.get(relative, "").lower()
        for term in terms:
            if term.lower() not in lower_text:
                issues.append(f"{relative}: missing governance term {term}")

    mkdocs = root / "mkdocs.yml"
    checked_paths.append("mkdocs.yml")
    if not mkdocs.is_file():
        issues.append("mkdocs.yml: missing docs navigation")
    elif "governance.md" not in mkdocs.read_text(encoding="utf-8"):
        issues.append("mkdocs.yml: missing governance docs nav entry")

    ci = root / ".github" / "workflows" / "ci.yml"
    checked_paths.append(".github/workflows/ci.yml")
    if not ci.is_file():
        issues.append(".github/workflows/ci.yml: missing CI workflow")
    else:
        ci_text = ci.read_text(encoding="utf-8")
        for required in ("tests/test_governance.py", "docs/governance.md", "promptabi governance --format text"):
            if required not in ci_text:
                issues.append(f".github/workflows/ci.yml: missing {required}")

    return GovernanceReport(
        repo_root=root,
        checked_paths=tuple(sorted(checked_paths)),
        principles=principles,
        release_blockers=REQUIRED_RELEASE_BLOCKERS,
        issues=tuple(issues),
    )


def render_governance_json(report: GovernanceReport) -> str:
    """Render a governance report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_governance_text(report: GovernanceReport) -> str:
    """Render a governance report for maintainers and release logs."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"PromptABI governance: {status}",
        f"principles: {len(report.principles)}",
        f"release blockers: {', '.join(report.release_blockers)}",
    ]
    for principle in report.principles:
        lines.extend(
            (
                "",
                f"{principle.id}: {principle.title}",
                f"  standard: {principle.standard}",
                f"  evidence: {'; '.join(principle.evidence)}",
                f"  blocks: {'; '.join(principle.release_blocking_when)}",
            )
        )
    if report.issues:
        lines.append("")
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in report.issues)
    return "\n".join(lines) + "\n"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
