"""Curated public bug gallery built from replayed real-bug benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .diagnostics import UpstreamIssueLink
from .real_bug_benchmarks import (
    RealBugBenchmarkCase,
    RealBugBenchmarkError,
    RealBugBenchmarkResult,
    load_real_bug_benchmark_suite,
)


PUBLIC_BUG_GALLERY_VERSION = 1


class PublicBugGalleryError(ValueError):
    """Raised when a public bug-gallery report cannot be built."""


@dataclass(frozen=True, slots=True)
class PublicBugGalleryArtifact:
    """A non-sensitive artifact summary for one public bug-gallery case."""

    kind: str
    reference: str
    sha256: str
    redaction: str = "sanitized-reduction"

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "reference": self.reference,
            "sha256": self.sha256,
            "redaction": self.redaction,
        }


@dataclass(frozen=True, slots=True)
class PublicBugGalleryWitness:
    """A minimized, non-secret witness summary derived from an actual replay."""

    rule_ids: tuple[str, ...]
    evidence: str
    minimized_repro: dict[str, object]
    replay_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_ids": list(self.rule_ids),
            "evidence": self.evidence,
            "minimized_repro": self.minimized_repro,
            "replay_hash": self.replay_hash,
        }


@dataclass(frozen=True, slots=True)
class PublicBugGalleryEntry:
    """One public, replayed, sanitized bug-gallery entry."""

    case_id: str
    category: str
    display_name: str
    bug_class: str
    public_reference: str
    source_kind: str
    labels: tuple[str, ...]
    replayed: bool
    root_cause: str
    sanitized_artifacts: tuple[PublicBugGalleryArtifact, ...]
    witness: PublicBugGalleryWitness
    fixes: tuple[str, ...]
    upstream_patches: tuple[UpstreamIssueLink, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "category": self.category,
            "display_name": self.display_name,
            "bug_class": self.bug_class,
            "public_reference": self.public_reference,
            "source_kind": self.source_kind,
            "labels": list(self.labels),
            "replayed": self.replayed,
            "root_cause": self.root_cause,
            "sanitized_artifacts": [artifact.to_dict() for artifact in self.sanitized_artifacts],
            "minimized_witness": self.witness.to_dict(),
            "fixes": list(self.fixes),
            "upstream_patches": [patch.to_dict() for patch in self.upstream_patches],
        }


@dataclass(frozen=True, slots=True)
class PublicBugGalleryReport:
    """A deterministic public bug gallery backed by real analyzer replays."""

    version: int
    promptabi_version: str
    methodology: str
    entries: tuple[PublicBugGalleryEntry, ...]
    all_replayed: bool
    report_sha256: str

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({entry.category for entry in self.entries}))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "promptabi_version": self.promptabi_version,
            "methodology": self.methodology,
            "summary": {
                "entries": len(self.entries),
                "categories": list(self.categories),
                "all_replayed": self.all_replayed,
            },
            "entries": [entry.to_dict() for entry in self.entries],
            "report_sha256": self.report_sha256,
        }


def build_public_bug_gallery(path: str | Path | None = None) -> PublicBugGalleryReport:
    """Replay the real-bug benchmark suite and convert it into a public gallery."""

    try:
        suite = load_real_bug_benchmark_suite(path)
        results = suite.replay()
    except RealBugBenchmarkError as exc:
        raise PublicBugGalleryError(str(exc)) from exc
    by_id = {result.case_id: result for result in results}
    entries = tuple(_entry_from_case(case, by_id[case.case_id]) for case in suite.cases)
    report_without_hash: dict[str, object] = {
        "version": PUBLIC_BUG_GALLERY_VERSION,
        "promptabi_version": __version__,
        "methodology": (
            suite.methodology
            + " The public gallery includes only sanitized reductions, replay hashes, minimized structural witnesses, "
            "root-cause summaries, fixes, and upstream patch links."
        ),
        "summary": {
            "entries": len(entries),
            "categories": sorted({entry.category for entry in entries}),
            "all_replayed": all(entry.replayed for entry in entries),
        },
        "entries": [entry.to_dict() for entry in entries],
    }
    return PublicBugGalleryReport(
        version=PUBLIC_BUG_GALLERY_VERSION,
        promptabi_version=__version__,
        methodology=str(report_without_hash["methodology"]),
        entries=entries,
        all_replayed=all(entry.replayed for entry in entries),
        report_sha256=_stable_json_hash(report_without_hash),
    )


def render_public_bug_gallery_json(report: PublicBugGalleryReport) -> str:
    """Render the public bug gallery as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_public_bug_gallery_markdown(report: PublicBugGalleryReport) -> str:
    """Render the public bug gallery as concise Markdown."""

    lines = [
        "# PromptABI public bug gallery",
        "",
        f"{len(report.entries)} sanitized public reductions replay against local analyzers; "
        f"all replayed: `{str(report.all_replayed).lower()}`.",
        "",
        "| Case | Category | Root cause | Witness | Fix / patch |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in report.entries:
        patch = entry.upstream_patches[0].url if entry.upstream_patches else entry.public_reference
        fix = entry.fixes[0] if entry.fixes else "Review the linked upstream patch."
        lines.append(
            "| "
            + " | ".join(
                _md_cell(value)
                for value in (
                    entry.display_name,
                    entry.category,
                    entry.root_cause,
                    f"{', '.join(entry.witness.rule_ids)}; {entry.witness.evidence}",
                    f"{fix} ({patch})",
                )
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Gallery entries intentionally omit upstream source code and private payloads; each row carries a replay hash "
        "and a minimized structural repro in the JSON form."
    )
    return "\n".join(lines)


def render_public_bug_gallery_text(report: PublicBugGalleryReport) -> str:
    """Render the public bug gallery for terminals."""

    lines = [
        "PromptABI public bug gallery",
        f"entries: {len(report.entries)}",
        f"all replayed: {str(report.all_replayed).lower()}",
    ]
    for entry in report.entries:
        lines.append(
            f"- {entry.case_id}: {entry.display_name} [{entry.category}] "
            f"{','.join(entry.witness.rule_ids)}"
        )
        lines.append(f"  root cause: {entry.root_cause}")
        lines.append(f"  witness: {entry.witness.evidence} (hash {entry.witness.replay_hash[:12]})")
        if entry.fixes:
            lines.append(f"  fix: {entry.fixes[0]}")
        if entry.upstream_patches:
            lines.append(f"  patch: {entry.upstream_patches[0].url}")
    return "\n".join(lines) + "\n"


def write_public_bug_gallery(
    output: str | Path,
    *,
    path: str | Path | None = None,
    output_format: str = "json",
) -> PublicBugGalleryReport:
    """Build and write a public bug-gallery report."""

    report = build_public_bug_gallery(path)
    if output_format == "json":
        rendered = render_public_bug_gallery_json(report)
    elif output_format == "markdown":
        rendered = render_public_bug_gallery_markdown(report) + "\n"
    elif output_format == "text":
        rendered = render_public_bug_gallery_text(report)
    else:
        raise PublicBugGalleryError("output_format must be one of: text, json, markdown")
    Path(output).write_text(rendered, encoding="utf-8")
    return report


def _entry_from_case(case: RealBugBenchmarkCase, result: RealBugBenchmarkResult) -> PublicBugGalleryEntry:
    replay_hash = _stable_json_hash({"id": case.case_id, "replay": case.replay, "observed": result.to_dict()})
    fixes = _fixes(case.upstream_issues)
    minimized_repro = {
        "method": str(case.replay["method"]),
        "expected_rule_ids": list(case.expected_rule_ids),
        "observed_rule_ids": list(result.observed_rule_ids),
        "artifact_digest": _stable_json_hash(_sanitized_replay(case)),
    }
    if "config" in case.replay:
        minimized_repro["config"] = str(case.replay["config"])
    if "case_id" in case.replay:
        minimized_repro["corpus_case_id"] = str(case.replay["case_id"])
    if "entry_id" in case.replay:
        minimized_repro["corpus_entry_id"] = str(case.replay["entry_id"])
    return PublicBugGalleryEntry(
        case_id=case.case_id,
        category=case.category,
        display_name=case.display_name,
        bug_class=case.bug_class,
        public_reference=case.public_reference,
        source_kind=case.source_kind,
        labels=case.labels,
        replayed=result.passed,
        root_cause=_root_cause(case),
        sanitized_artifacts=(
            PublicBugGalleryArtifact(
                kind=str(case.replay["method"]),
                reference=_artifact_reference(case),
                sha256=str(minimized_repro["artifact_digest"]),
            ),
        ),
        witness=PublicBugGalleryWitness(
            rule_ids=result.observed_rule_ids,
            evidence=result.evidence_summary,
            minimized_repro=minimized_repro,
            replay_hash=replay_hash,
        ),
        fixes=fixes,
        upstream_patches=case.upstream_issues,
    )


def _root_cause(case: RealBugBenchmarkCase) -> str:
    by_category = {
        "popular-template": "Template-controlled delimiters were not separated from attacker-controlled fields.",
        "tokenizer": "Tokenizer normalization, added-token, or special-token metadata drifted from the pinned expectation.",
        "tool-schema": "A stop or parser boundary can occur inside an otherwise valid tool argument.",
        "provider-migration": "Provider envelopes disagree on structural fields such as tool IDs, streaming chunks, stops, or response formats.",
        "structured-output-library": "The constrained-output grammar and the application parser accept different languages.",
        "rag-truncation": "Framework packing can drop citations or required context after metadata and template overhead are counted.",
        "training-pipeline": "Training targets, masks, or roles do not match the serving prompt interface contract.",
    }
    return by_category.get(case.category, case.bug_class)


def _fixes(issues: tuple[UpstreamIssueLink, ...]) -> tuple[str, ...]:
    fixes: list[str] = []
    for issue in issues:
        fixes.extend(issue.fixed_versions)
        fixes.extend(issue.workarounds)
    return tuple(dict.fromkeys(fixes))


def _artifact_reference(case: RealBugBenchmarkCase) -> str:
    for key in ("config", "case_id", "entry_id"):
        value = case.replay.get(key)
        if isinstance(value, str) and value:
            return value
    return case.case_id


def _sanitized_replay(case: RealBugBenchmarkCase) -> dict[str, object]:
    allowed = {"method", "config", "case_id", "entry_id", "normalization", "added_tokens", "special_tokens"}
    return {key: value for key, value in sorted(case.replay.items()) if key in allowed}


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
