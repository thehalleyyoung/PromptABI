"""Maintainer refresh and project-health tooling for PromptABI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .corpus_verification import render_corpus_verification_json, run_corpus_verification
from .real_bug_benchmarks import build_real_bug_benchmark_manifest
from .seed_corpus import build_seed_corpus_manifest
from .session import VerificationSession
from .structured_schema_corpus import build_structured_schema_corpus_manifest
from .provider_fixture_packs import build_provider_fixture_pack_manifest


MAINTAINER_REFRESH_VERSION = 1
MAINTAINER_HEALTH_VERSION = 1
MAINTAINER_FILENAMES = (
    "seed-corpus.manifest.json",
    "structured-schemas.manifest.json",
    "provider-fixtures.manifest.json",
    "real-bug-benchmark.manifest.json",
    "corpus-verification.json",
    "expected-diagnostics.json",
    "maintainer-snapshot.json",
    "corpus-diff.json",
    "release-notes.md",
)
REQUIRED_MAINTAINER_HEALTH_DOCS = (
    "docs/maintainer-health.md",
    "docs/governance.md",
    "docs/contributing/corpus-contributions.md",
)
REQUIRED_TRIAGE_LABELS = (
    "status: needs-triage",
    "status: release-blocking",
    "priority: high",
    "area: corpus",
    "area: checker",
    "type: bug",
)


class MaintainerToolingError(ValueError):
    """Raised when maintainer refresh artifacts cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class MaintainerRefresh:
    """All generated maintainer artifacts for one repository state."""

    output_dir: Path
    snapshot: dict[str, object]
    diff: dict[str, object]
    release_notes: str
    written_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class MaintainerHealthItem:
    """One maintainer-health expectation with concrete review evidence."""

    id: str
    title: str
    detail: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class MaintainerHealthReport:
    """Validation report for maintainer rotation, triage, releases, corpus review, and health metrics."""

    repo_root: Path
    checked_paths: tuple[str, ...]
    rotation_roles: tuple[MaintainerHealthItem, ...]
    triage_labels: tuple[MaintainerHealthItem, ...]
    release_checklist: tuple[MaintainerHealthItem, ...]
    corpus_review: tuple[MaintainerHealthItem, ...]
    metrics: dict[str, object]
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": MAINTAINER_HEALTH_VERSION,
            "ok": self.ok,
            "checked_paths": list(self.checked_paths),
            "rotation_roles": [item.to_dict() for item in self.rotation_roles],
            "triage_labels": [item.to_dict() for item in self.triage_labels],
            "release_checklist": [item.to_dict() for item in self.release_checklist],
            "corpus_review": [item.to_dict() for item in self.corpus_review],
            "metrics": dict(sorted(self.metrics.items())),
            "issues": list(self.issues),
        }


def refresh_maintainer_artifacts(
    output_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> MaintainerRefresh:
    """Regenerate model/corpus manifests, expected diagnostics, diffs, and release notes."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    destination = Path(output_dir)
    _prepare_output_dir(destination, force=force)

    seed_manifest = build_seed_corpus_manifest(root / "fixtures" / "seed_corpus")
    structured_manifest = build_structured_schema_corpus_manifest(root / "fixtures" / "structured_schemas")
    provider_manifest = build_provider_fixture_pack_manifest(root / "fixtures" / "provider_fixture_packs")
    real_bug_manifest = build_real_bug_benchmark_manifest(root / "fixtures" / "real_bug_benchmarks" / "benchmark.json")
    corpus_verification = run_corpus_verification(
        seed_root=root / "fixtures" / "seed_corpus",
        structured_schema_root=root / "fixtures" / "structured_schemas",
        provider_fixture_root=root / "fixtures" / "provider_fixture_packs",
        real_bug_benchmark_path=root / "fixtures" / "real_bug_benchmarks" / "benchmark.json",
        evaluation_corpus_path=root / "fixtures" / "evaluation" / "labeled_corpus.json",
    ).to_dict()
    expected_diagnostics = collect_expected_diagnostics(root)
    snapshot = build_maintainer_snapshot(
        seed_manifest=seed_manifest,
        structured_manifest=structured_manifest,
        provider_manifest=provider_manifest,
        real_bug_manifest=real_bug_manifest,
        corpus_verification=corpus_verification,
        expected_diagnostics=expected_diagnostics,
        repo_root=root,
    )
    baseline_snapshot = _load_baseline_snapshot(baseline_dir)
    diff = diff_maintainer_snapshots(baseline_snapshot, snapshot)
    release_notes = render_maintainer_release_notes(diff, snapshot)

    payloads = {
        "seed-corpus.manifest.json": _json_dump(seed_manifest),
        "structured-schemas.manifest.json": _json_dump(structured_manifest),
        "provider-fixtures.manifest.json": _json_dump(provider_manifest),
        "real-bug-benchmark.manifest.json": _json_dump(real_bug_manifest),
        "corpus-verification.json": _json_dump(corpus_verification),
        "expected-diagnostics.json": _json_dump(expected_diagnostics),
        "maintainer-snapshot.json": _json_dump(snapshot),
        "corpus-diff.json": _json_dump(diff),
        "release-notes.md": release_notes,
    }
    written = []
    for name in MAINTAINER_FILENAMES:
        path = destination / name
        path.write_text(payloads[name], encoding="utf-8")
        written.append(path)
    return MaintainerRefresh(
        output_dir=destination,
        snapshot=snapshot,
        diff=diff,
        release_notes=release_notes,
        written_files=tuple(written),
    )


def build_maintainer_health_items() -> dict[str, tuple[MaintainerHealthItem, ...]]:
    """Return the maintainer-health contract that docs and CI must expose."""

    return {
        "rotation_roles": (
            MaintainerHealthItem(
                id="rotation-release-captain",
                title="Release captain",
                detail="Owns the release checklist, version gates, and final release-blocker decision for one train.",
                evidence=("release readiness report", "governance report", "maintainer health report"),
            ),
            MaintainerHealthItem(
                id="rotation-corpus-steward",
                title="Corpus steward",
                detail="Reviews fixture provenance, labels expected diagnostics, and coordinates annual corpus refreshes.",
                evidence=("corpus review checklist", "fixture manifests", "sanitization notes"),
            ),
            MaintainerHealthItem(
                id="rotation-triage-lead",
                title="Triage lead",
                detail="Keeps issues labeled, escalates release blockers, and routes contributor-ready work.",
                evidence=("triage labels", "issue template labels", "weekly triage notes"),
            ),
        ),
        "triage_labels": (
            MaintainerHealthItem(
                id="label-status-needs-triage",
                title="status: needs-triage",
                detail="New reports that need a maintainer to classify affected surface, severity, and reproducibility.",
                evidence=("status: needs-triage",),
            ),
            MaintainerHealthItem(
                id="label-status-release-blocking",
                title="status: release-blocking",
                detail="Bugs that match governance release blockers or invalidate paper/release claims.",
                evidence=("status: release-blocking",),
            ),
            MaintainerHealthItem(
                id="label-priority-high",
                title="priority: high",
                detail="Urgent correctness, privacy, or compatibility work that should preempt normal roadmap items.",
                evidence=("priority: high",),
            ),
            MaintainerHealthItem(
                id="label-area-corpus",
                title="area: corpus",
                detail="Fixture, benchmark, provenance, corpus refresh, and expected-diagnostic work.",
                evidence=("area: corpus",),
            ),
        ),
        "release_checklist": (
            MaintainerHealthItem(
                id="release-governance-gate",
                title="Governance gate",
                detail="Run governance and maintainer-health validators before tagging a release.",
                evidence=("promptabi governance --format text", "promptabi maintain health --format text"),
            ),
            MaintainerHealthItem(
                id="release-corpus-gate",
                title="Corpus gate",
                detail="Run affected corpus/conformance suites and refresh maintainer artifacts when expected diagnostics change.",
                evidence=("promptabi corpus verify --format text", "promptabi maintain refresh --output-dir"),
            ),
            MaintainerHealthItem(
                id="release-privacy-gate",
                title="Privacy gate",
                detail="Confirm release artifacts, witnesses, bug reports, and fixture packs contain no secrets or private prompts.",
                evidence=("secret-bearing-fixture", "witness privacy modes", "sanitized bug reports"),
            ),
        ),
        "corpus_review": (
            MaintainerHealthItem(
                id="corpus-provenance-review",
                title="Provenance review",
                detail="Every fixture addition records source, license, revision/hash, expected diagnostics, and sanitization status.",
                evidence=("license", "provenance", "expected diagnostics", "no secrets"),
            ),
            MaintainerHealthItem(
                id="corpus-regression-review",
                title="Regression review",
                detail="Changed fixtures must preserve or intentionally update labeled real-bug coverage and witness replay.",
                evidence=("regression-on-labeled-real-bug", "witness-replay-failure"),
            ),
            MaintainerHealthItem(
                id="corpus-refresh-cadence",
                title="Refresh cadence",
                detail="At least annual corpus refreshes add new model/provider/framework semantics while preserving old benchmarks.",
                evidence=("annual corpus refresh", "longitudinal benchmark"),
            ),
        ),
    }


def validate_maintainer_health(repo_root: str | Path | None = None) -> MaintainerHealthReport:
    """Validate maintainer rotation, triage, release, corpus-review, and health-metric coverage."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    if not root.exists():
        raise MaintainerToolingError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise MaintainerToolingError(f"repository root is not a directory: {root}")

    items = build_maintainer_health_items()
    issues: list[str] = []
    checked_paths: list[str] = []
    texts: dict[str, str] = {}
    for relative in REQUIRED_MAINTAINER_HEALTH_DOCS:
        checked_paths.append(relative)
        path = root / relative
        if not path.is_file():
            issues.append(f"{relative}: missing maintainer-health document")
            continue
        texts[relative] = path.read_text(encoding="utf-8")

    health_text = texts.get("docs/maintainer-health.md", "")
    for category_items in items.values():
        for item in category_items:
            if item.id not in health_text:
                issues.append(f"docs/maintainer-health.md: missing item id {item.id}")
            if item.title.lower() not in health_text.lower():
                issues.append(f"docs/maintainer-health.md: missing item title {item.title}")
            for evidence in item.evidence:
                if evidence.lower() not in health_text.lower():
                    issues.append(f"docs/maintainer-health.md: missing evidence term {evidence}")

    labels_path = root / ".github" / "labels.yml"
    checked_paths.append(".github/labels.yml")
    if not labels_path.is_file():
        issues.append(".github/labels.yml: missing triage labels")
        label_text = ""
    else:
        label_text = labels_path.read_text(encoding="utf-8")
        for label in REQUIRED_TRIAGE_LABELS:
            if f'name: "{label}"' not in label_text and f"name: '{label}'" not in label_text and f"name: {label}" not in label_text:
                issues.append(f".github/labels.yml: missing label {label}")

    mkdocs = root / "mkdocs.yml"
    checked_paths.append("mkdocs.yml")
    if not mkdocs.is_file():
        issues.append("mkdocs.yml: missing docs navigation")
    elif "maintainer-health.md" not in mkdocs.read_text(encoding="utf-8"):
        issues.append("mkdocs.yml: missing maintainer health docs nav entry")

    ci = root / ".github" / "workflows" / "ci.yml"
    checked_paths.append(".github/workflows/ci.yml")
    if not ci.is_file():
        issues.append(".github/workflows/ci.yml: missing CI workflow")
    else:
        ci_text = ci.read_text(encoding="utf-8")
        for required in ("tests/test_maintainer_health.py", "docs/maintainer-health.md", "promptabi maintain health --format text"):
            if required not in ci_text:
                issues.append(f".github/workflows/ci.yml: missing {required}")

    metrics = _maintainer_health_metrics(root, label_text)
    return MaintainerHealthReport(
        repo_root=root,
        checked_paths=tuple(sorted(checked_paths)),
        rotation_roles=items["rotation_roles"],
        triage_labels=items["triage_labels"],
        release_checklist=items["release_checklist"],
        corpus_review=items["corpus_review"],
        metrics=metrics,
        issues=tuple(issues),
    )


def render_maintainer_health_json(report: MaintainerHealthReport) -> str:
    """Render maintainer health as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_maintainer_health_text(report: MaintainerHealthReport) -> str:
    """Render maintainer health for release logs and CI summaries."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"PromptABI maintainer health: {status}",
        f"rotation roles: {len(report.rotation_roles)}",
        f"triage labels: {len(REQUIRED_TRIAGE_LABELS)} required",
        f"release checklist items: {len(report.release_checklist)}",
        f"corpus review gates: {len(report.corpus_review)}",
        "metrics:",
    ]
    for key, value in sorted(report.metrics.items()):
        lines.append(f"  {key}: {value}")
    sections = (
        ("Rotation roles", report.rotation_roles),
        ("Triage labels", report.triage_labels),
        ("Release checklist", report.release_checklist),
        ("Corpus review", report.corpus_review),
    )
    for title, category_items in sections:
        lines.extend(("", title))
        for item in category_items:
            lines.append(f"  {item.id}: {item.title}")
            lines.append(f"    {item.detail}")
    if report.issues:
        lines.extend(("", "Issues:"))
        lines.extend(f"- {issue}" for issue in report.issues)
    return "\n".join(lines) + "\n"


def collect_expected_diagnostics(repo_root: str | Path | None = None) -> dict[str, object]:
    """Run real verification configs and record portable expected diagnostics."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    configs = _discover_verification_configs(root)
    entries = []
    diagnostic_total = 0
    for config_path in configs:
        result = VerificationSession.from_config_file(config_path).run()
        config_name = _relative_path(config_path, root)
        diagnostics = [
            _diagnostic_entry(diagnostic, root, config_name=config_name, diagnostic_index=index)
            for index, diagnostic in enumerate(result.diagnostics, start=1)
        ]
        diagnostic_total += len(diagnostics)
        entries.append(
            {
                "config": config_name,
                "ok": result.ok,
                "diagnostic_count": len(diagnostics),
                "rule_ids": sorted({str(item["rule_id"]) for item in diagnostics}),
                "diagnostics": diagnostics,
                "baseline_sha256": _stable_json_hash(diagnostics),
            }
        )
    payload: dict[str, object] = {
        "manifest_version": MAINTAINER_REFRESH_VERSION,
        "config_count": len(entries),
        "diagnostic_count": diagnostic_total,
        "configs": entries,
    }
    payload["baseline_sha256"] = _stable_json_hash(payload)
    return payload


def build_maintainer_snapshot(
    *,
    seed_manifest: dict[str, object],
    structured_manifest: dict[str, object],
    provider_manifest: dict[str, object],
    real_bug_manifest: dict[str, object],
    corpus_verification: dict[str, object],
    expected_diagnostics: dict[str, object],
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Build a compact snapshot used for corpus-diff and release-note generation."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    corpora = {
        "seed-corpus": _corpus_summary(seed_manifest, count_key="entry_count"),
        "structured-schemas": _corpus_summary(structured_manifest, count_key="entry_count"),
        "provider-fixtures": _corpus_summary(provider_manifest, count_key="entry_count"),
        "real-bug-benchmark": _corpus_summary(real_bug_manifest, count_key="case_count"),
    }
    snapshot: dict[str, object] = {
        "manifest_version": MAINTAINER_REFRESH_VERSION,
        "promptabi_version": __version__,
        "repository_root": ".",
        "purpose": (
            "Maintainer refresh snapshot for model artifact updates, corpus diffs, "
            "provider fixture refreshes, expected diagnostics, and release notes."
        ),
        "corpora": corpora,
        "corpus_verification": {
            "ok": corpus_verification["ok"],
            "check_count": corpus_verification["check_count"],
            "coverage_count": corpus_verification["coverage_count"],
            "checks": [
                {
                    "name": check["name"],
                    "passed": check["passed"],
                    "coverage_count": check["coverage_count"],
                    "expected_count": check["expected_count"],
                    "failures": check["failures"],
                }
                for check in corpus_verification["checks"]  # type: ignore[index]
            ],
        },
        "expected_diagnostics": {
            "config_count": expected_diagnostics["config_count"],
            "diagnostic_count": expected_diagnostics["diagnostic_count"],
            "baseline_sha256": expected_diagnostics["baseline_sha256"],
            "configs": [
                {
                    "config": entry["config"],
                    "ok": entry["ok"],
                    "diagnostic_count": entry["diagnostic_count"],
                    "rule_ids": entry["rule_ids"],
                    "baseline_sha256": entry["baseline_sha256"],
                }
                for entry in expected_diagnostics["configs"]  # type: ignore[index]
            ],
        },
        "generated_paths": {
            "seed_corpus": _relative_path(root / "fixtures" / "seed_corpus", root),
            "structured_schemas": _relative_path(root / "fixtures" / "structured_schemas", root),
            "provider_fixture_packs": _relative_path(root / "fixtures" / "provider_fixture_packs", root),
            "real_bug_benchmark": _relative_path(root / "fixtures" / "real_bug_benchmarks" / "benchmark.json", root),
        },
    }
    snapshot["snapshot_sha256"] = _stable_json_hash(snapshot)
    return snapshot


def diff_maintainer_snapshots(
    baseline: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    """Compare two maintainer snapshots and report corpus/check/diagnostic changes."""

    if baseline is None:
        diff: dict[str, object] = {
            "manifest_version": MAINTAINER_REFRESH_VERSION,
            "status": "initial",
            "summary": "no baseline snapshot supplied",
            "corpora": _initial_corpus_diff(current),
            "diagnostics": _initial_diagnostic_diff(current),
            "verification_checks": _verification_check_changes(None, current),
        }
    else:
        corpus_changes = _corpus_changes(baseline, current)
        diagnostic_changes = _diagnostic_changes(baseline, current)
        check_changes = _verification_check_changes(baseline, current)
        changed = bool(corpus_changes or diagnostic_changes or check_changes)
        diff = {
            "manifest_version": MAINTAINER_REFRESH_VERSION,
            "status": "changed" if changed else "unchanged",
            "summary": "maintainer artifacts changed" if changed else "no maintainer artifact changes",
            "corpora": corpus_changes,
            "diagnostics": diagnostic_changes,
            "verification_checks": check_changes,
        }
    diff["diff_sha256"] = _stable_json_hash(diff)
    return diff


def render_maintainer_release_notes(diff: dict[str, object], snapshot: dict[str, object]) -> str:
    """Render concise release notes from check, corpus, and diagnostic-baseline changes."""

    lines = [
        "# PromptABI maintainer release notes",
        "",
        f"- Status: **{diff['status']}**",
        f"- PromptABI version: `{snapshot['promptabi_version']}`",
        f"- Corpus gate: **{'PASS' if snapshot['corpus_verification']['ok'] else 'FAIL'}** "
        f"over {snapshot['corpus_verification']['coverage_count']} replayed item(s)",  # type: ignore[index]
        f"- Expected diagnostics: {snapshot['expected_diagnostics']['diagnostic_count']} diagnostic(s) "
        f"across {snapshot['expected_diagnostics']['config_count']} config(s)",  # type: ignore[index]
        "",
    ]
    initial_line_count = len(lines)
    corpora = diff.get("corpora")
    if isinstance(corpora, list) and corpora:
        lines.extend(["## Corpus and fixture changes", ""])
        for change in corpora:
            lines.append(f"- `{change['corpus']}`: {change['summary']}")
        lines.append("")
    diagnostics = diff.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        lines.extend(["## Expected diagnostic changes", ""])
        for change in diagnostics:
            lines.append(f"- `{change['config']}`: {change['summary']}")
        lines.append("")
    checks = diff.get("verification_checks")
    if isinstance(checks, list) and checks:
        lines.extend(["## Release-gate check changes", ""])
        for change in checks:
            lines.append(f"- `{change['check']}`: {change['summary']}")
        lines.append("")
    if len(lines) == initial_line_count:
        lines.extend(["No corpus, fixture, diagnostic, or release-gate changes were detected.", ""])
    lines.append("Generated by `promptabi maintain refresh` from local, secret-free repository artifacts.")
    return "\n".join(lines) + "\n"


def _prepare_output_dir(destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise MaintainerToolingError(f"output path exists and is not a directory: {destination}")
        existing = {path.name for path in destination.iterdir() if not path.name.startswith(".")}
        unexpected = existing.difference(MAINTAINER_FILENAMES)
        if existing and (unexpected or not force):
            detail = ", ".join(sorted(existing))
            raise MaintainerToolingError(
                f"output directory is not empty: {destination} ({detail}); pass --force to overwrite maintainer files"
            )
    destination.mkdir(parents=True, exist_ok=True)


def _load_baseline_snapshot(baseline_dir: str | Path | None) -> dict[str, object] | None:
    if baseline_dir is None:
        return None
    path = Path(baseline_dir) / "maintainer-snapshot.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MaintainerToolingError(f"baseline snapshot does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MaintainerToolingError(f"baseline snapshot is not valid JSON: {path}:{exc.lineno}:{exc.colno}") from exc
    if not isinstance(payload, dict):
        raise MaintainerToolingError(f"baseline snapshot must contain a JSON object: {path}")
    return payload


def _discover_verification_configs(root: Path) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for directory in (root / "examples", root / "fixtures"):
        if not directory.is_dir():
            continue
        candidates.update(directory.rglob("promptabi.json"))
        candidates.update(directory.rglob("*.promptabi.json"))
    return tuple(
        sorted(
            path
            for path in candidates
            if path.is_file()
            and not path.name.endswith(".lock.json")
            and ".manifest." not in path.name
            and "maintainer" not in path.parts
        )
    )


def _diagnostic_entry(
    diagnostic: Any,
    root: Path,
    *,
    config_name: str,
    diagnostic_index: int,
) -> dict[str, object]:
    span = _span_payload(diagnostic.span, root)
    artifact = _artifact_payload(diagnostic.artifact, root)
    payload: dict[str, object] = {
        "rule_id": diagnostic.rule_id,
        "severity": diagnostic.severity.value,
        "message": diagnostic.message,
        "check_modes": [mode.value for mode in diagnostic.check_modes],
        "config": config_name,
        "index": diagnostic_index,
        "span": span,
        "artifact": artifact,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def _span_payload(span: Any, root: Path) -> dict[str, object] | None:
    if span is None:
        return None
    payload = {
        "path": _relative_path(Path(span.path), root),
        "start_line": span.start_line,
        "start_column": span.start_column,
    }
    if span.end_line is not None:
        payload["end_line"] = span.end_line
    if span.end_column is not None:
        payload["end_column"] = span.end_column
    return payload


def _artifact_payload(artifact: Any, root: Path) -> dict[str, object] | None:
    if artifact is None:
        return None
    payload = artifact.to_dict()
    path = payload.get("path")
    if isinstance(path, str):
        payload["path"] = _relative_path(Path(path), root)
    return payload


def _corpus_summary(manifest: dict[str, object], *, count_key: str) -> dict[str, object]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        entries = []
    entry_hashes = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            entry_hashes[entry["id"]] = entry.get("fixture_sha256") or entry.get("pack_sha256") or entry.get("config_sha256")
    return {
        "count": manifest[count_key],
        "manifest_sha256": manifest["manifest_sha256"],
        "entries": dict(sorted(entry_hashes.items())),
    }


def _initial_corpus_diff(current: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "corpus": name,
            "change": "initial",
            "summary": f"{corpus['count']} tracked item(s), manifest {corpus['manifest_sha256']}",
            "added": sorted(corpus["entries"]),
            "removed": [],
            "changed": [],
        }
        for name, corpus in _snapshot_corpora(current).items()
    ]


def _initial_diagnostic_diff(current: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "config": entry["config"],
            "change": "initial",
            "summary": f"{entry['diagnostic_count']} expected diagnostic(s)",
            "added": list(entry["rule_ids"]),
            "removed": [],
            "changed": [],
        }
        for entry in _snapshot_diagnostic_configs(current).values()
    ]


def _corpus_changes(baseline: dict[str, object], current: dict[str, object]) -> list[dict[str, object]]:
    old = _snapshot_corpora(baseline)
    new = _snapshot_corpora(current)
    changes = []
    for name in sorted(set(old) | set(new)):
        before = old.get(name, {"entries": {}, "manifest_sha256": None, "count": 0})
        after = new.get(name, {"entries": {}, "manifest_sha256": None, "count": 0})
        before_entries = before["entries"]
        after_entries = after["entries"]
        added = sorted(set(after_entries) - set(before_entries))
        removed = sorted(set(before_entries) - set(after_entries))
        changed = sorted(
            entry_id for entry_id in set(before_entries) & set(after_entries)
            if before_entries[entry_id] != after_entries[entry_id]
        )
        if added or removed or changed or before.get("manifest_sha256") != after.get("manifest_sha256"):
            changes.append(
                {
                    "corpus": name,
                    "change": "changed",
                    "summary": (
                        f"{before.get('count', 0)} -> {after.get('count', 0)} item(s); "
                        f"{len(added)} added, {len(removed)} removed, {len(changed)} changed"
                    ),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                    "old_manifest_sha256": before.get("manifest_sha256"),
                    "new_manifest_sha256": after.get("manifest_sha256"),
                }
            )
    return changes


def _diagnostic_changes(baseline: dict[str, object], current: dict[str, object]) -> list[dict[str, object]]:
    old = _snapshot_diagnostic_configs(baseline)
    new = _snapshot_diagnostic_configs(current)
    changes = []
    for config in sorted(set(old) | set(new)):
        before = old.get(config, {"rule_ids": [], "baseline_sha256": None, "diagnostic_count": 0})
        after = new.get(config, {"rule_ids": [], "baseline_sha256": None, "diagnostic_count": 0})
        added = sorted(set(after["rule_ids"]) - set(before["rule_ids"]))
        removed = sorted(set(before["rule_ids"]) - set(after["rule_ids"]))
        changed = []
        if before.get("baseline_sha256") != after.get("baseline_sha256") and not (added or removed):
            changed = ["fingerprint-or-message"]
        if added or removed or changed:
            changes.append(
                {
                    "config": config,
                    "change": "changed",
                    "summary": (
                        f"{before.get('diagnostic_count', 0)} -> {after.get('diagnostic_count', 0)} diagnostic(s); "
                        f"{len(added)} rule(s) added, {len(removed)} removed, {len(changed)} changed"
                    ),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                    "old_baseline_sha256": before.get("baseline_sha256"),
                    "new_baseline_sha256": after.get("baseline_sha256"),
                }
            )
    return changes


def _verification_check_changes(
    baseline: dict[str, object] | None,
    current: dict[str, object],
) -> list[dict[str, object]]:
    new = _snapshot_checks(current)
    if baseline is None:
        return [
            {
                "check": name,
                "change": "initial",
                "summary": f"{check['coverage_count']}/{check['expected_count']} coverage; passed={check['passed']}",
            }
            for name, check in new.items()
        ]
    old = _snapshot_checks(baseline)
    changes = []
    for name in sorted(set(old) | set(new)):
        before = old.get(name)
        after = new.get(name)
        if before != after:
            changes.append(
                {
                    "check": name,
                    "change": "changed",
                    "summary": f"{_check_summary(before)} -> {_check_summary(after)}",
                }
            )
    return changes


def _snapshot_corpora(snapshot: dict[str, object]) -> dict[str, dict[str, Any]]:
    corpora = snapshot.get("corpora")
    return corpora if isinstance(corpora, dict) else {}


def _snapshot_diagnostic_configs(snapshot: dict[str, object]) -> dict[str, dict[str, Any]]:
    diagnostics = snapshot.get("expected_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    configs = diagnostics.get("configs")
    if not isinstance(configs, list):
        return {}
    return {entry["config"]: entry for entry in configs if isinstance(entry, dict) and isinstance(entry.get("config"), str)}


def _snapshot_checks(snapshot: dict[str, object]) -> dict[str, dict[str, Any]]:
    verification = snapshot.get("corpus_verification")
    if not isinstance(verification, dict):
        return {}
    checks = verification.get("checks")
    if not isinstance(checks, list):
        return {}
    return {check["name"]: check for check in checks if isinstance(check, dict) and isinstance(check.get("name"), str)}


def _check_summary(check: dict[str, Any] | None) -> str:
    if check is None:
        return "missing"
    return f"{check['coverage_count']}/{check['expected_count']} coverage; passed={check['passed']}"


def _maintainer_health_metrics(root: Path, label_text: str) -> dict[str, object]:
    fixture_root = root / "fixtures"
    docs_root = root / "docs"
    test_root = root / "tests"
    labels = [line for line in label_text.splitlines() if line.strip().startswith("- name:")]
    fixture_json_count = 0
    if fixture_root.is_dir():
        fixture_json_count = sum(1 for path in fixture_root.rglob("*.json") if path.is_file())
    return {
        "docs_markdown_files": sum(1 for path in docs_root.rglob("*.md") if path.is_file()) if docs_root.is_dir() else 0,
        "fixture_json_files": fixture_json_count,
        "python_test_files": sum(1 for path in test_root.glob("test_*.py") if path.is_file()) if test_root.is_dir() else 0,
        "triage_label_count": len(labels),
        "verification_config_count": len(_discover_verification_configs(root)),
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_dump(payload: dict[str, object] | list[object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
