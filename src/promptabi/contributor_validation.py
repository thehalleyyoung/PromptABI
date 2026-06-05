"""Contributor-facing repository infrastructure validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


CONTRIBUTOR_VALIDATION_VERSION = 1
REQUIRED_ISSUE_TEMPLATES = (
    "bug_report.yml",
    "checker_proposal.yml",
    "corpus_fixture.yml",
    "plugin_request.yml",
)
REQUIRED_LABELS = (
    "good first issue",
    "help wanted",
    "area: checker",
    "area: corpus",
    "area: plugin",
    "type: bug",
    "type: docs",
    "type: proposal",
)
REQUIRED_GUIDE_PATHS = (
    "CONTRIBUTING.md",
    "docs/contributing/plugin-author-guide.md",
    "docs/contributing/checker-design.md",
    "docs/contributing/corpus-contributions.md",
)
REQUIRED_CI_TEST_TARGET = "tests/test_contributor_infrastructure.py"


class ContributorValidationError(ValueError):
    """Raised when contributor validation cannot inspect a repository."""


@dataclass(frozen=True, slots=True)
class ContributorValidationIssue:
    """One failed contributor-infrastructure validation check."""

    path: str
    check: str
    message: str


@dataclass(frozen=True, slots=True)
class ContributorValidationReport:
    """Deterministic validation report for contributor-facing infrastructure."""

    repo_root: Path
    issue_count: int
    checked_paths: tuple[str, ...]
    labels: tuple[str, ...]
    issue_templates: tuple[str, ...]
    issues: tuple[ContributorValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return self.issue_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": CONTRIBUTOR_VALIDATION_VERSION,
            "ok": self.ok,
            "issue_count": self.issue_count,
            "checked_paths": list(self.checked_paths),
            "labels": list(self.labels),
            "issue_templates": list(self.issue_templates),
            "issues": [
                {
                    "path": issue.path,
                    "check": issue.check,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }


def validate_contributor_infrastructure(repo_root: str | Path | None = None) -> ContributorValidationReport:
    """Validate GitHub templates, labels, contribution docs, and CI contributor gates."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    if not root.exists():
        raise ContributorValidationError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise ContributorValidationError(f"repository root is not a directory: {root}")

    issues: list[ContributorValidationIssue] = []
    checked_paths: set[str] = set()
    checked_templates: list[str] = []

    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    checked_paths.add(".github/ISSUE_TEMPLATE")
    if not template_dir.is_dir():
        issues.append(_issue(".github/ISSUE_TEMPLATE", "exists", "issue-template directory is missing"))
    for template_name in REQUIRED_ISSUE_TEMPLATES:
        relative = f".github/ISSUE_TEMPLATE/{template_name}"
        checked_paths.add(relative)
        checked_templates.append(template_name)
        template = template_dir / template_name
        if not template.is_file():
            issues.append(_issue(relative, "exists", "required issue template is missing"))
            continue
        text = template.read_text(encoding="utf-8")
        _validate_issue_template(relative, text, issues)

    config = root / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    checked_paths.add(".github/ISSUE_TEMPLATE/config.yml")
    if not config.is_file():
        issues.append(_issue(".github/ISSUE_TEMPLATE/config.yml", "exists", "issue template chooser config is missing"))
    elif "blank_issues_enabled: false" not in config.read_text(encoding="utf-8"):
        issues.append(
            _issue(
                ".github/ISSUE_TEMPLATE/config.yml",
                "blank-issues-disabled",
                "blank issues should be disabled so contributors choose a structured path",
            )
        )

    labels_path = root / ".github" / "labels.yml"
    checked_paths.add(".github/labels.yml")
    labels: tuple[str, ...] = ()
    if not labels_path.is_file():
        issues.append(_issue(".github/labels.yml", "exists", "label manifest is missing"))
    else:
        labels = _extract_label_names(labels_path.read_text(encoding="utf-8"))
        for required in REQUIRED_LABELS:
            if required not in labels:
                issues.append(_issue(".github/labels.yml", "required-label", f"missing label: {required}"))

    for relative in REQUIRED_GUIDE_PATHS:
        checked_paths.add(relative)
        path = root / relative
        if not path.is_file():
            issues.append(_issue(relative, "exists", "required contributor guide is missing"))
            continue
        text = path.read_text(encoding="utf-8")
        _validate_guide(relative, text, issues)

    mkdocs = root / "mkdocs.yml"
    checked_paths.add("mkdocs.yml")
    if not mkdocs.is_file():
        issues.append(_issue("mkdocs.yml", "exists", "MkDocs config is missing"))
    else:
        mkdocs_text = mkdocs.read_text(encoding="utf-8")
        for relative in REQUIRED_GUIDE_PATHS[1:]:
            doc_relative = relative.removeprefix("docs/")
            if doc_relative not in mkdocs_text:
                issues.append(_issue("mkdocs.yml", "nav", f"missing docs nav entry for {doc_relative}"))

    ci = root / ".github" / "workflows" / "ci.yml"
    checked_paths.add(".github/workflows/ci.yml")
    if not ci.is_file():
        issues.append(_issue(".github/workflows/ci.yml", "exists", "CI workflow is missing"))
    else:
        ci_text = ci.read_text(encoding="utf-8")
        if REQUIRED_CI_TEST_TARGET not in ci_text:
            issues.append(
                _issue(
                    ".github/workflows/ci.yml",
                    "contributor-validation",
                    f"CI must run `{REQUIRED_CI_TEST_TARGET}`",
                )
            )
        for path_fragment in ("docs/contributing/**", ".github/ISSUE_TEMPLATE/**", ".github/labels.yml"):
            if path_fragment not in ci_text:
                issues.append(
                    _issue(
                        ".github/workflows/ci.yml",
                        "path-trigger",
                        f"CI must trigger on {path_fragment}",
                    )
                )

    return ContributorValidationReport(
        repo_root=root,
        issue_count=len(issues),
        checked_paths=tuple(sorted(checked_paths)),
        labels=labels,
        issue_templates=tuple(checked_templates),
        issues=tuple(issues),
    )


def render_contributor_validation_json(report: ContributorValidationReport) -> str:
    """Render a contributor-validation report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_contributor_validation_text(report: ContributorValidationReport) -> str:
    """Render a contributor-validation report for humans and CI logs."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"PromptABI contributor infrastructure: {status}",
        f"checked paths: {len(report.checked_paths)}",
        f"issue templates: {', '.join(report.issue_templates)}",
        f"labels: {len(report.labels)}",
    ]
    if report.issues:
        lines.append("issues:")
        for issue in report.issues:
            lines.append(f"- {issue.path}: {issue.check}: {issue.message}")
    else:
        lines.append("all structured contribution paths are present and CI-gated")
    return "\n".join(lines) + "\n"


def _validate_issue_template(relative: str, text: str, issues: list[ContributorValidationIssue]) -> None:
    for key in ("name:", "description:", "title:", "labels:", "body:"):
        if key not in text:
            issues.append(_issue(relative, "template-shape", f"missing `{key}`"))
    if "type: textarea" not in text:
        issues.append(_issue(relative, "template-fields", "template needs at least one textarea field"))
    if "validations:" not in text or "required: true" not in text:
        issues.append(_issue(relative, "required-fields", "template must require at least one field"))


def _validate_guide(relative: str, text: str, issues: list[ContributorValidationIssue]) -> None:
    required_terms = {
        "CONTRIBUTING.md": ("deterministic", "CPU-only", "promptabi contribute validate"),
        "docs/contributing/plugin-author-guide.md": ("PluginRegistry", "privacy", "tests/test_contributor_infrastructure.py"),
        "docs/contributing/checker-design.md": ("CheckMode", "witness", "abstention"),
        "docs/contributing/corpus-contributions.md": ("provenance", "license", "no secrets"),
    }[relative]
    lower = text.lower()
    for term in required_terms:
        if term.lower() not in lower:
            issues.append(_issue(relative, "guide-content", f"missing required guidance term: {term}"))


def _extract_label_names(text: str) -> tuple[str, ...]:
    names = []
    for match in re.finditer(r"(?m)^\s*-\s+name:\s*['\"]?([^'\"\n]+)['\"]?\s*$", text):
        names.append(match.group(1).strip())
    return tuple(sorted(set(names)))


def _issue(path: str, check: str, message: str) -> ContributorValidationIssue:
    return ContributorValidationIssue(path=path, check=check, message=message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
