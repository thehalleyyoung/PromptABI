"""Release-readiness and maintenance release gates for PromptABI artifacts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ._version import __version__
from .compatibility_audit import (
    CompatibilityAuditError,
    CompatibilityAuditReport,
    run_compatibility_audit,
)
from .compatibility_matrix import build_compatibility_matrix
from .diagnostics import CheckMode
from .proof_sketches import build_supported_proof_catalog
from .real_bug_benchmarks import build_real_bug_benchmark_manifest
from .reproducibility import ReproducibilityInputs, build_reproducibility_package
from .seed_corpus import build_seed_corpus_manifest
from .theorem_traceability import build_theorem_traceability_report


DEFAULT_RELEASE_VERSION = "1.0.0"
RELEASE_READINESS_VERSION = 1
LTS_RELEASE_PLAN_VERSION = 1
LTS_ITEM_CATEGORIES = ("checker_fix", "security_patch", "corpus_update", "compatibility_metadata")
DEFAULT_LTS_CANDIDATE_VERSIONS = {
    "tokenizer": "seed-v1",
    "template": "seed-v1",
    "provider": "provider-fixtures-v1",
    "grammar": "grammar-differential-v1",
    "framework": "structured-schemas-v1",
}
_SEMVER_RE = re.compile(r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


class ReleaseReadinessStatus(StrEnum):
    """Outcome for one release-readiness check."""

    PASS = "pass"
    FAIL = "fail"


class LTSReleaseStatus(StrEnum):
    """Outcome for one long-term support release planning gate."""

    PASS = "pass"
    FAIL = "fail"
    ABSTAINED = "abstained"


class LTSReleaseRisk(StrEnum):
    """Backport risk assigned by deterministic release automation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ReleaseReadinessCheck:
    """One deterministic release gate with machine-readable evidence."""

    name: str
    status: ReleaseReadinessStatus
    summary: str
    evidence: tuple[tuple[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is ReleaseReadinessStatus.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": {key: value for key, value in self.evidence},
        }


@dataclass(frozen=True, slots=True)
class ReleaseReadinessReport:
    """Complete 1.0 readiness report for release automation and humans."""

    version: str
    repository_root: Path
    checks: tuple[ReleaseReadinessCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": RELEASE_READINESS_VERSION,
            "version": self.version,
            "repository_root": str(self.repository_root),
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }


class ReleaseReadinessError(ValueError):
    """Raised when release-readiness inputs cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class LTSMaintenanceItem:
    """One requested maintenance item for an LTS release train."""

    category: str
    item_id: str
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = {"category": self.category, "id": self.item_id}
        if self.summary:
            payload["summary"] = self.summary
        return payload


@dataclass(frozen=True, slots=True)
class LTSBackportDecision:
    """Automated routing decision for a requested LTS item."""

    item: LTSMaintenanceItem
    lane: str
    risk: LTSReleaseRisk
    status: LTSReleaseStatus
    evidence: tuple[tuple[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is LTSReleaseStatus.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "item": self.item.to_dict(),
            "lane": self.lane,
            "risk": self.risk.value,
            "status": self.status.value,
            "evidence": {key: value for key, value in self.evidence},
        }


@dataclass(frozen=True, slots=True)
class LTSReleaseCheck:
    """One machine-readable LTS release gate."""

    name: str
    status: LTSReleaseStatus
    summary: str
    evidence: tuple[tuple[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is LTSReleaseStatus.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": {key: value for key, value in self.evidence},
        }


@dataclass(frozen=True, slots=True)
class LTSReleasePlan:
    """Long-term support train plan backed by live repository evidence."""

    series: str
    base_version: str
    target_version: str
    repository_root: Path
    candidate_versions: dict[str, str]
    items: tuple[LTSMaintenanceItem, ...]
    decisions: tuple[LTSBackportDecision, ...]
    checks: tuple[LTSReleaseCheck, ...]
    manifest_sha256: str

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks) and all(decision.passed for decision in self.decisions)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": LTS_RELEASE_PLAN_VERSION,
            "ok": self.ok,
            "series": self.series,
            "base_version": self.base_version,
            "target_version": self.target_version,
            "repository_root": str(self.repository_root),
            "candidate_versions": dict(sorted(self.candidate_versions.items())),
            "items": [item.to_dict() for item in self.items],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "checks": [check.to_dict() for check in self.checks],
            "manifest_sha256": self.manifest_sha256,
        }


class LTSReleaseError(ValueError):
    """Raised when LTS release automation inputs are malformed."""


def build_release_readiness_report(
    repo_root: str | Path | None = None,
    *,
    expected_version: str = DEFAULT_RELEASE_VERSION,
) -> ReleaseReadinessReport:
    """Run the PromptABI release gate against live repository assets."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.exists():
        raise ReleaseReadinessError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise ReleaseReadinessError(f"repository root is not a directory: {root}")
    checks = (
        _version_metadata_check(root, expected_version),
        _changelog_check(root, expected_version),
        _readme_check(root),
        _cli_check(),
        _github_action_check(root),
        _docs_site_check(root),
        _seed_corpus_check(root),
        _formal_checks_check(),
        _theorem_traceability_check(root),
        _real_bug_benchmark_check(root),
        _reproducibility_check(root),
        _paper_check(root),
    )
    return ReleaseReadinessReport(version=expected_version, repository_root=root, checks=checks)


def lts_item_from_string(value: str) -> LTSMaintenanceItem:
    """Parse CATEGORY:ID[:SUMMARY] declarations used by CLI and workflows."""

    category, separator, remainder = value.partition(":")
    if not separator or not category or not remainder:
        raise LTSReleaseError("LTS item must use CATEGORY:ID or CATEGORY:ID:SUMMARY")
    item_id, _, summary = remainder.partition(":")
    if not item_id:
        raise LTSReleaseError("LTS item id cannot be empty")
    normalized_category = category.strip().lower().replace("-", "_")
    if normalized_category not in LTS_ITEM_CATEGORIES:
        raise LTSReleaseError(
            "unknown LTS item category "
            f"{category!r}; expected one of {', '.join(LTS_ITEM_CATEGORIES)}"
        )
    normalized_id = item_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized_id):
        raise LTSReleaseError(f"LTS item id contains unsupported characters: {item_id!r}")
    return LTSMaintenanceItem(category=normalized_category, item_id=normalized_id, summary=summary.strip())


def build_lts_release_plan(
    items: tuple[LTSMaintenanceItem, ...],
    *,
    series: str,
    base_version: str,
    target_version: str,
    repo_root: str | Path | None = None,
    candidate_versions: dict[str, str] | None = None,
) -> LTSReleasePlan:
    """Build a deterministic LTS release plan using local fixture-backed evidence."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.exists():
        raise LTSReleaseError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise LTSReleaseError(f"repository root is not a directory: {root}")
    if not items:
        raise LTSReleaseError("at least one LTS maintenance item is required")
    _validate_semver("base version", base_version)
    _validate_semver("target version", target_version)
    _validate_lts_series(series, base_version, target_version)

    normalized_versions = dict(sorted((candidate_versions or DEFAULT_LTS_CANDIDATE_VERSIONS).items()))
    normalized_items = tuple(sorted(items, key=lambda item: (item.category, item.item_id, item.summary)))
    compatibility = run_compatibility_audit(normalized_versions)
    seed_manifest = build_seed_corpus_manifest(root / "fixtures" / "seed_corpus")
    bug_manifest = build_real_bug_benchmark_manifest(root / "fixtures" / "real_bug_benchmarks" / "benchmark.json")
    matrix = build_compatibility_matrix(include_plugins=False)
    pyproject = _read_pyproject(root)

    decisions = tuple(
        _lts_decision(item, root=root, seed_manifest=seed_manifest, bug_manifest=bug_manifest, matrix=matrix)
        for item in normalized_items
    )
    checks = (
        _lts_version_check(series, base_version, target_version),
        _lts_category_coverage_check(normalized_items),
        _lts_compatibility_check(compatibility),
        _lts_corpus_check(seed_manifest, bug_manifest),
        _lts_metadata_check(pyproject, matrix),
    )
    manifest_sha256 = _lts_manifest_hash(
        series=series,
        base_version=base_version,
        target_version=target_version,
        candidate_versions=normalized_versions,
        items=normalized_items,
        decisions=decisions,
        checks=checks,
    )
    return LTSReleasePlan(
        series=series,
        base_version=base_version,
        target_version=target_version,
        repository_root=root,
        candidate_versions=normalized_versions,
        items=normalized_items,
        decisions=decisions,
        checks=checks,
        manifest_sha256=manifest_sha256,
    )


def render_release_readiness_json(report: ReleaseReadinessReport) -> str:
    """Render release readiness as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_lts_release_plan_json(plan: LTSReleasePlan) -> str:
    """Render an LTS release plan as stable JSON."""

    return json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"


def render_release_readiness_text(report: ReleaseReadinessReport) -> str:
    """Render release readiness as concise CLI text."""

    lines = [
        "PromptABI release readiness",
        f"version: {report.version}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
    ]
    for check in report.checks:
        lines.append("")
        lines.append(f"{check.name}: {check.status.value.upper()}")
        lines.append(f"  {check.summary}")
        for key, value in check.evidence:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def render_lts_release_plan_text(plan: LTSReleasePlan) -> str:
    """Render an LTS release plan for release maintainers."""

    lines = [
        "PromptABI LTS release plan",
        f"series: {plan.series}",
        f"base: {plan.base_version}",
        f"target: {plan.target_version}",
        f"status: {'PASS' if plan.ok else 'FAIL'}",
        f"manifest_sha256: {plan.manifest_sha256}",
        "",
        "checks:",
    ]
    for check in plan.checks:
        lines.append(f"- {check.name}: {check.status.value.upper()} {check.summary}")
        for key, value in check.evidence:
            lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("backports:")
    for decision in plan.decisions:
        lines.append(
            f"- {decision.item.category}:{decision.item.item_id} -> {decision.lane} "
            f"risk={decision.risk.value} status={decision.status.value.upper()}"
        )
        for key, value in decision.evidence:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _version_metadata_check(root: Path, expected_version: str) -> ReleaseReadinessCheck:
    pyproject = _read_pyproject(root)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return _fail("version-metadata", "pyproject.toml is missing [project] metadata")
    pyproject_version = project.get("version")
    classifiers = project.get("classifiers", ())
    stable_classifier = "Development Status :: 5 - Production/Stable"
    passed = (
        pyproject_version == expected_version
        and __version__ == expected_version
        and isinstance(classifiers, list)
        and stable_classifier in classifiers
    )
    return _check(
        "version-metadata",
        passed,
        "pyproject, importable version, and maturity classifier are release-stable",
        "version metadata is not ready for a 1.0 release",
        (
            ("pyproject_version", pyproject_version),
            ("package_version", __version__),
            ("required_classifier", stable_classifier),
        ),
    )


def _changelog_check(root: Path, expected_version: str) -> ReleaseReadinessCheck:
    changelog = _read_text(root / "CHANGELOG.md")
    header = f"## {expected_version}"
    passed = header in changelog and "semantic versioning" in changelog and "release-readiness" in changelog
    return _check(
        "changelog",
        passed,
        "CHANGELOG documents the 1.0 release contract and readiness gate",
        "CHANGELOG is missing the 1.0 release entry or readiness note",
        (("header", header),),
    )


def _readme_check(root: Path) -> ReleaseReadinessCheck:
    readme = _read_text(root / "README.md")
    required_tokens = (
        "promptabi verify",
        "promptabi github-action",
        "promptabi corpus verify",
        "promptabi paper reproducibility",
        "promptabi release readiness",
        "Z3-backed",
        "CPU-only",
    )
    missing = tuple(token for token in required_tokens if token not in readme)
    return _check(
        "readme",
        not missing,
        "README demonstrates core verification, CI, corpus, paper, and release workflows",
        "README is missing release-critical command anchors",
        (("missing", list(missing)),),
    )


def _cli_check() -> ReleaseReadinessCheck:
    from .cli import build_parser

    parser = build_parser()
    required_paths = (
        ("verify",),
        ("explain",),
        ("diff",),
        ("init",),
        ("github-action",),
        ("api-docs",),
        ("matrix",),
        ("proofs",),
        ("doctor",),
        ("corpus", "verify"),
        ("corpus", "real-bug-benchmark"),
        ("corpus", "evaluation"),
        ("paper", "reproducibility"),
        ("release", "readiness"),
        ("release", "compatibility-audit"),
        ("release", "lts-plan"),
    )
    missing = tuple(" ".join(path) for path in required_paths if not _parser_accepts(parser, path))
    return _check(
        "stable-cli",
        not missing,
        "stable CLI exposes release-critical verification, corpus, GitHub, paper, and release commands",
        "stable CLI is missing release-critical commands",
        (("missing", list(missing)),),
    )


def _github_action_check(root: Path) -> ReleaseReadinessCheck:
    required_paths = (
        ".github/actions/promptabi/action.yml",
        ".github/workflows/promptabi.yml",
        ".github/workflows/release.yml",
    )
    missing = tuple(path for path in required_paths if not (root / path).exists())
    action = _read_text(root / ".github/actions/promptabi/action.yml") if not missing else ""
    release = _read_text(root / ".github/workflows/release.yml") if not missing else ""
    required_tokens = ("promptabi github-action", "python -m build", "pypa/gh-action-pypi-publish")
    missing_tokens = tuple(token for token in required_tokens if token not in f"{action}\n{release}")
    return _check(
        "github-action-and-release",
        not missing and not missing_tokens,
        "GitHub Action, PromptABI workflow, and signed release workflow are present",
        "GitHub release/Action assets are incomplete",
        (("missing_paths", list(missing)), ("missing_tokens", list(missing_tokens))),
    )


def _docs_site_check(root: Path) -> ReleaseReadinessCheck:
    mkdocs = _read_text(root / "mkdocs.yml")
    required_docs = (
        "docs/index.md",
        "docs/quickstart.md",
        "docs/checks.md",
        "docs/security-model.md",
        "docs/public-api.md",
        "docs/concepts/static-contracts.md",
    )
    missing = tuple(path for path in required_docs if not (root / path).exists())
    nav_tokens = ("Public API", "Security model", "Static contracts")
    missing_nav = tuple(token for token in nav_tokens if token not in mkdocs)
    return _check(
        "docs-site",
        not missing and not missing_nav,
        "MkDocs site contains quickstart, API, security, checks, and formal concept docs",
        "documentation site is missing release-critical pages or navigation",
        (("missing_docs", list(missing)), ("missing_nav", list(missing_nav))),
    )


def _seed_corpus_check(root: Path) -> ReleaseReadinessCheck:
    manifest = build_seed_corpus_manifest(root / "fixtures" / "seed_corpus")
    families = tuple(manifest.get("families", ()))
    passed = int(manifest.get("entry_count", 0)) >= 10 and len(families) >= 8 and bool(manifest.get("manifest_sha256"))
    return _check(
        "seed-corpus",
        passed,
        "seed corpus manifest builds with broad tokenizer/template family coverage",
        "seed corpus is too small or missing deterministic manifest evidence",
        (
            ("entry_count", manifest.get("entry_count")),
            ("family_count", len(families)),
            ("manifest_sha256", manifest.get("manifest_sha256")),
        ),
    )


def _formal_checks_check() -> ReleaseReadinessCheck:
    matrix = build_compatibility_matrix(include_plugins=False)
    entries = {entry.check: entry for entry in matrix.entries}
    required_checks = (
        "role-boundary-nonforgeability",
        "stop-overreachability",
        "grammar-tokenizer-emptiness",
        "static-contracts",
        "token-budget-model",
    )
    missing = tuple(check for check in required_checks if check not in entries)
    static_modes = set(entries["static-contracts"].modes) if "static-contracts" in entries else set()
    proofs = build_supported_proof_catalog()
    proof_ids = {sketch.property_id for sketch in proofs.sketches}
    required_proofs = (
        "role-boundary-nonforgeability",
        "stop-overreachability",
        "grammar-tokenizer-emptiness",
        "must-survive-budget",
        "z3-backed-finite-contract",
    )
    missing_proofs = tuple(proof for proof in required_proofs if proof not in proof_ids)
    passed = (
        not missing
        and not missing_proofs
        and proofs.passed
        and CheckMode.Z3_BACKED_SMT in static_modes
        and CheckMode.BOUNDED in static_modes
    )
    return _check(
        "formal-checks",
        passed,
        "core formal, bounded, and Z3-backed checks are registered with proof sketches",
        "core formal/Z3-backed checks are not release-ready",
        (
            ("missing_checks", list(missing)),
            ("static_contract_modes", sorted(mode.value for mode in static_modes)),
            ("missing_proofs", list(missing_proofs)),
        ),
    )


def _theorem_traceability_check(root: Path) -> ReleaseReadinessCheck:
    report = build_theorem_traceability_report(root)
    failed = tuple(trace.property_id for trace in report.traces if not trace.passed)
    return _check(
        "theorem-traceability",
        report.passed,
        "each core proof claim maps to executable specs, property tests, corpus fixtures, and release gates",
        "theorem-to-test traceability has missing or stale evidence",
        (
            ("theorem_count", report.theorem_count),
            ("failed", list(failed)),
        ),
    )


def _real_bug_benchmark_check(root: Path) -> ReleaseReadinessCheck:
    manifest = build_real_bug_benchmark_manifest(root / "fixtures" / "real_bug_benchmarks" / "benchmark.json")
    passed = bool(manifest.get("all_cases_passed")) and int(manifest.get("case_count", 0)) >= 7
    return _check(
        "real-bug-benchmark",
        passed,
        "real-bug benchmark replays against live analyzers and passes",
        "real-bug benchmark replay is incomplete or failing",
        (
            ("case_count", manifest.get("case_count")),
            ("categories", manifest.get("categories")),
            ("manifest_sha256", manifest.get("manifest_sha256")),
        ),
    )


def _reproducibility_check(root: Path) -> ReleaseReadinessCheck:
    package = build_reproducibility_package(
        inputs=ReproducibilityInputs(repository_root=root),
        benchmark_iterations=1,
    )
    summary = package.manifest["summary"]  # type: ignore[index]
    evaluation = package.expected_tables["evaluation"]  # type: ignore[index]
    passed = bool(evaluation["passed"]) and int(summary["fixture_file_count"]) > 20 and bool(package.manifest["artifact_sha256"])
    return _check(
        "reproducibility-package",
        passed,
        "paper reproducibility bundle builds frozen fixtures, expected tables, and commands",
        "paper reproducibility bundle failed release thresholds",
        (
            ("fixture_file_count", summary["fixture_file_count"]),
            ("evaluation_passed", evaluation["passed"]),
            ("artifact_sha256", package.manifest["artifact_sha256"]),
        ),
    )


def _paper_check(root: Path) -> ReleaseReadinessCheck:
    tex_path = root / "tool_paper.tex"
    pdf_path = root / "tool_paper.pdf"
    tex = _read_text(tex_path)
    required_tokens = (
        r"\section{Evaluation Design}",
        "promptabi corpus verify",
        "paper reproducibility",
        "promptabi release readiness",
        "real-bug benchmark",
        "Z3",
    )
    missing = tuple(token for token in required_tokens if token not in tex)
    page_count = _pdf_page_count(pdf_path)
    page_ok = page_count is None or 20 < page_count < 40
    return _check(
        "paper-preprint",
        pdf_path.exists() and not missing and page_ok,
        "paper preprint source/PDF exist, describe deployed experiments, and stay within page bounds",
        "paper preprint is missing release-critical evidence or page bounds",
        (("missing", list(missing)), ("pdf_exists", pdf_path.exists()), ("pdf_pages", page_count)),
    )


def _lts_version_check(series: str, base_version: str, target_version: str) -> LTSReleaseCheck:
    base_major_minor = ".".join(base_version.split(".")[:2])
    target_major_minor = ".".join(target_version.split(".")[:2])
    passed = series == base_major_minor == target_major_minor
    return _lts_check(
        "version-window",
        passed,
        "base and target versions stay inside the requested LTS series",
        "base/target versions do not both belong to the requested LTS series",
        (("series", series), ("base_major_minor", base_major_minor), ("target_major_minor", target_major_minor)),
    )


def _lts_category_coverage_check(items: tuple[LTSMaintenanceItem, ...]) -> LTSReleaseCheck:
    present = tuple(sorted({item.category for item in items}))
    missing = tuple(category for category in LTS_ITEM_CATEGORIES if category not in present)
    return _lts_check(
        "maintenance-category-coverage",
        not missing,
        "LTS train includes checker fixes, security patches, corpus updates, and compatibility metadata",
        "LTS train is missing required maintenance categories",
        (("present", list(present)), ("missing", list(missing))),
    )


def _lts_compatibility_check(report: CompatibilityAuditReport) -> LTSReleaseCheck:
    abstained = tuple(target.surface for target in report.targets if target.status.value == "abstained")
    failed = tuple(target.surface for target in report.targets if target.status.value == "fail")
    return _lts_check(
        "compatibility-metadata",
        report.ok,
        "candidate compatibility metadata is backed by pinned local fixture replays",
        "candidate compatibility metadata is missing pinned fixture coverage or failed replay",
        (
            ("coverage_count", report.coverage_count),
            ("abstained", list(abstained)),
            ("failed", list(failed)),
        ),
    )


def _lts_corpus_check(seed_manifest: dict[str, object], bug_manifest: dict[str, object]) -> LTSReleaseCheck:
    seed_count = int(seed_manifest.get("entry_count", 0))
    bug_count = int(bug_manifest.get("case_count", 0))
    passed = seed_count >= 10 and bug_count >= 7 and bool(seed_manifest.get("manifest_sha256")) and bool(bug_manifest.get("manifest_sha256"))
    return _lts_check(
        "corpus-update-evidence",
        passed,
        "LTS corpus updates are tied to deterministic seed and real-bug benchmark manifests",
        "LTS corpus evidence is too small or missing deterministic hashes",
        (
            ("seed_entry_count", seed_count),
            ("seed_manifest_sha256", seed_manifest.get("manifest_sha256")),
            ("real_bug_case_count", bug_count),
            ("real_bug_manifest_sha256", bug_manifest.get("manifest_sha256")),
        ),
    )


def _lts_metadata_check(pyproject: dict[str, object], matrix) -> LTSReleaseCheck:
    project = pyproject.get("project") if isinstance(pyproject, dict) else None
    package_name = project.get("name") if isinstance(project, dict) else None
    matrix_checks = tuple(sorted(entry.check for entry in matrix.entries))
    required = {"role-boundary-nonforgeability", "stop-overreachability", "grammar-tokenizer-emptiness", "static-contracts"}
    missing = tuple(sorted(required.difference(matrix_checks)))
    return _lts_check(
        "release-metadata",
        package_name == "promptabi" and not missing,
        "package metadata and compatibility matrix expose LTS-relevant checker identities",
        "package metadata or compatibility matrix cannot support LTS metadata",
        (("package_name", package_name), ("missing_checks", list(missing)), ("matrix_check_count", len(matrix_checks))),
    )


def _lts_decision(item: LTSMaintenanceItem, *, root: Path, seed_manifest: dict[str, object], bug_manifest: dict[str, object], matrix) -> LTSBackportDecision:
    matrix_checks = {entry.check for entry in matrix.entries}
    if item.category == "checker_fix":
        known = item.item_id in matrix_checks
        return LTSBackportDecision(
            item=item,
            lane="checker-backport",
            risk=LTSReleaseRisk.MEDIUM,
            status=LTSReleaseStatus.PASS if known else LTSReleaseStatus.ABSTAINED,
            evidence=(("registered_check", known), ("matrix_check_count", len(matrix_checks))),
        )
    if item.category == "security_patch":
        security_docs = ("docs/security-model.md", "SECURITY.md")
        present = tuple(path for path in security_docs if (root / path).exists())
        return LTSBackportDecision(
            item=item,
            lane="security-hotfix",
            risk=LTSReleaseRisk.HIGH,
            status=LTSReleaseStatus.PASS if present else LTSReleaseStatus.ABSTAINED,
            evidence=(("security_evidence_paths", list(present)),),
        )
    if item.category == "corpus_update":
        return LTSBackportDecision(
            item=item,
            lane="corpus-refresh",
            risk=LTSReleaseRisk.LOW,
            status=LTSReleaseStatus.PASS,
            evidence=(
                ("seed_manifest_sha256", seed_manifest.get("manifest_sha256")),
                ("real_bug_manifest_sha256", bug_manifest.get("manifest_sha256")),
            ),
        )
    return LTSBackportDecision(
        item=item,
        lane="compatibility-metadata",
        risk=LTSReleaseRisk.LOW,
        status=LTSReleaseStatus.PASS,
        evidence=(("candidate_surfaces", list(DEFAULT_LTS_CANDIDATE_VERSIONS)),),
    )


def _lts_check(
    name: str,
    passed: bool,
    pass_summary: str,
    fail_summary: str,
    evidence: tuple[tuple[str, object], ...] = (),
) -> LTSReleaseCheck:
    return LTSReleaseCheck(
        name=name,
        status=LTSReleaseStatus.PASS if passed else LTSReleaseStatus.FAIL,
        summary=pass_summary if passed else fail_summary,
        evidence=evidence,
    )


def _validate_semver(label: str, value: str) -> None:
    if _SEMVER_RE.fullmatch(value) is None:
        raise LTSReleaseError(f"{label} is not a semantic version: {value}")


def _validate_lts_series(series: str, base_version: str, target_version: str) -> None:
    if re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", series) is None:
        raise LTSReleaseError(f"LTS series must be MAJOR.MINOR: {series}")
    base_parts = tuple(int(part) for part in base_version.split(".")[:3])
    target_parts = tuple(int(part) for part in target_version.split(".")[:3])
    if target_parts < base_parts:
        raise LTSReleaseError(f"target version {target_version} is older than base version {base_version}")


def _lts_manifest_hash(
    *,
    series: str,
    base_version: str,
    target_version: str,
    candidate_versions: dict[str, str],
    items: tuple[LTSMaintenanceItem, ...],
    decisions: tuple[LTSBackportDecision, ...],
    checks: tuple[LTSReleaseCheck, ...],
) -> str:
    payload = {
        "series": series,
        "base_version": base_version,
        "target_version": target_version,
        "candidate_versions": dict(sorted(candidate_versions.items())),
        "items": [item.to_dict() for item in items],
        "decisions": [decision.to_dict() for decision in decisions],
        "checks": [check.to_dict() for check in checks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
    missing = tuple(token for token in required_tokens if token not in tex)
    page_count = _pdf_page_count(pdf_path)
    page_ok = page_count is None or 20 < page_count < 40
    return _check(
        "paper-preprint",
        pdf_path.exists() and not missing and page_ok,
        "paper preprint source/PDF exist, describe deployed experiments, and stay within page bounds",
        "paper preprint is missing release-critical evidence or page bounds",
        (("missing", list(missing)), ("pdf_exists", pdf_path.exists()), ("pdf_pages", page_count)),
    )


def _check(
    name: str,
    passed: bool,
    pass_summary: str,
    fail_summary: str,
    evidence: tuple[tuple[str, object], ...] = (),
) -> ReleaseReadinessCheck:
    return ReleaseReadinessCheck(
        name=name,
        status=ReleaseReadinessStatus.PASS if passed else ReleaseReadinessStatus.FAIL,
        summary=pass_summary if passed else fail_summary,
        evidence=evidence,
    )


def _fail(name: str, summary: str) -> ReleaseReadinessCheck:
    return ReleaseReadinessCheck(name=name, status=ReleaseReadinessStatus.FAIL, summary=summary)


def _read_pyproject(root: Path) -> dict[str, object]:
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ReleaseReadinessError(f"missing pyproject.toml under {root}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseReadinessError(f"cannot parse pyproject.toml: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _parser_accepts(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args((*path, "--help"))
    except SystemExit as exc:
        return exc.code == 0
    return True


def _pdf_page_count(path: Path) -> int | None:
    if not path.exists():
        return None
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is not None:
        try:
            completed = subprocess.run(
                [pdfinfo, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
            if match is not None:
                return int(match.group(1))
    try:
        data = path.read_bytes()
    except OSError:
        return None
    matches = re.findall(rb"/Type\s*/Page\b", data)
    if matches:
        return len(matches)
    return None
