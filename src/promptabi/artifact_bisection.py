"""Regression bisection for local artifact drift histories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .tokenizer_drift import TokenizerDriftFinding, compare_tokenizer_config_snapshots, load_tokenizer_config_snapshot


ARTIFACT_BISECTION_VERSION = 1


class ArtifactDriftSurface(StrEnum):
    """Artifact families that can be bisected for contract drift."""

    TOKENIZER = "tokenizer"
    TEMPLATE = "template"
    SCHEMA = "schema"
    PROVIDER = "provider"
    FRAMEWORK = "framework"


class ArtifactBisectionStatus(StrEnum):
    """Outcome for one probed revision."""

    GOOD = "good"
    BAD = "bad"


@dataclass(frozen=True, slots=True)
class ArtifactRevision:
    """One chronological artifact revision candidate."""

    label: str
    path: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ArtifactBisectionError("artifact revision labels must be non-empty")
        if not self.path:
            raise ArtifactBisectionError("artifact revision paths must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "path": self.path}


@dataclass(frozen=True, slots=True)
class ArtifactDriftFinding:
    """One field-level drift observed between the baseline and a revision."""

    surface: ArtifactDriftSurface
    field: str
    kind: str
    baseline: object
    current: object
    baseline_path: str
    current_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface.value,
            "field": self.field,
            "kind": self.kind,
            "baseline": self.baseline,
            "current": self.current,
            "baseline_path": self.baseline_path,
            "current_path": self.current_path,
        }


@dataclass(frozen=True, slots=True)
class ArtifactBisectionProbe:
    """One real artifact comparison performed by the bisection search."""

    index: int
    revision: ArtifactRevision
    status: ArtifactBisectionStatus
    finding_count: int
    matched_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "revision": self.revision.to_dict(),
            "status": self.status.value,
            "finding_count": self.finding_count,
            "matched_fields": list(self.matched_fields),
        }


@dataclass(frozen=True, slots=True)
class ArtifactBisectionReport:
    """Complete regression-bisection result for an artifact history."""

    surface: ArtifactDriftSurface
    baseline: ArtifactRevision
    revisions: tuple[ArtifactRevision, ...]
    bad_fields: tuple[str, ...]
    first_bad_index: int | None
    first_bad_revision: ArtifactRevision | None
    previous_good_revision: ArtifactRevision | None
    probes: tuple[ArtifactBisectionProbe, ...]
    findings: tuple[ArtifactDriftFinding, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.first_bad_revision is None

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": ARTIFACT_BISECTION_VERSION,
            "ok": self.ok,
            "surface": self.surface.value,
            "baseline": self.baseline.to_dict(),
            "revisions": [revision.to_dict() for revision in self.revisions],
            "bad_fields": list(self.bad_fields),
            "first_bad_index": self.first_bad_index,
            "first_bad_revision": self.first_bad_revision.to_dict() if self.first_bad_revision is not None else None,
            "previous_good_revision": self.previous_good_revision.to_dict() if self.previous_good_revision is not None else None,
            "probes": [probe.to_dict() for probe in self.probes],
            "findings": [finding.to_dict() for finding in self.findings],
        }


class ArtifactBisectionError(ValueError):
    """Raised when an artifact-bisection request is malformed or unreadable."""


def bisect_artifact_drift(
    surface: ArtifactDriftSurface | str,
    baseline_path: str | Path,
    revisions: tuple[ArtifactRevision, ...] | list[ArtifactRevision],
    *,
    baseline_label: str = "baseline",
    bad_fields: tuple[str, ...] | list[str] = (),
) -> ArtifactBisectionReport:
    """Find the first revision that introduces selected artifact drift.

    ``revisions`` must be ordered from oldest to newest after the baseline. The
    search performs real local artifact comparisons and assumes the requested
    regression predicate is monotonic across that ordered history.
    """

    resolved_surface = ArtifactDriftSurface(surface)
    revision_tuple = tuple(revisions)
    if not revision_tuple:
        raise ArtifactBisectionError("at least one candidate revision is required")
    baseline = ArtifactRevision(baseline_label, str(Path(baseline_path)))
    fields = tuple(dict.fromkeys(field for field in bad_fields if field))

    probes_by_index: dict[int, ArtifactBisectionProbe] = {}
    findings_by_index: dict[int, tuple[ArtifactDriftFinding, ...]] = {}

    def probe(index: int) -> bool:
        if index in probes_by_index:
            return probes_by_index[index].status is ArtifactBisectionStatus.BAD
        revision = revision_tuple[index]
        findings = compare_artifact_revision(resolved_surface, baseline.path, revision.path)
        matched = tuple(finding.field for finding in findings if _matches_requested_field(finding, fields))
        bad = bool(matched)
        probes_by_index[index] = ArtifactBisectionProbe(
            index=index,
            revision=revision,
            status=ArtifactBisectionStatus.BAD if bad else ArtifactBisectionStatus.GOOD,
            finding_count=len(findings),
            matched_fields=matched,
        )
        findings_by_index[index] = findings
        return bad

    low = 0
    high = len(revision_tuple) - 1
    first_bad_index: int | None = None
    while low <= high:
        middle = (low + high) // 2
        if probe(middle):
            first_bad_index = middle
            high = middle - 1
        else:
            low = middle + 1

    if first_bad_index is None:
        final_index = len(revision_tuple) - 1
        if final_index not in probes_by_index:
            probe(final_index)
        return ArtifactBisectionReport(
            surface=resolved_surface,
            baseline=baseline,
            revisions=revision_tuple,
            bad_fields=fields,
            first_bad_index=None,
            first_bad_revision=None,
            previous_good_revision=revision_tuple[-1],
            probes=tuple(probes_by_index[index] for index in sorted(probes_by_index)),
            findings=(),
        )

    if first_bad_index > 0 and first_bad_index - 1 not in probes_by_index:
        probe(first_bad_index - 1)
    first_bad_findings = tuple(
        finding for finding in findings_by_index[first_bad_index] if _matches_requested_field(finding, fields)
    )
    return ArtifactBisectionReport(
        surface=resolved_surface,
        baseline=baseline,
        revisions=revision_tuple,
        bad_fields=fields,
        first_bad_index=first_bad_index,
        first_bad_revision=revision_tuple[first_bad_index],
        previous_good_revision=revision_tuple[first_bad_index - 1] if first_bad_index > 0 else baseline,
        probes=tuple(probes_by_index[index] for index in sorted(probes_by_index)),
        findings=first_bad_findings,
    )


def compare_artifact_revision(
    surface: ArtifactDriftSurface | str,
    baseline_path: str | Path,
    current_path: str | Path,
) -> tuple[ArtifactDriftFinding, ...]:
    """Compare two concrete artifact revisions for bisection."""

    resolved_surface = ArtifactDriftSurface(surface)
    if resolved_surface in (ArtifactDriftSurface.TOKENIZER, ArtifactDriftSurface.TEMPLATE):
        findings = _tokenizer_findings(resolved_surface, baseline_path, current_path)
    else:
        findings = _json_findings(resolved_surface, baseline_path, current_path)
    return tuple(sorted(findings, key=lambda finding: (finding.kind, finding.field)))


def render_artifact_bisection_json(report: ArtifactBisectionReport) -> str:
    """Render an artifact-bisection report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_artifact_bisection_text(report: ArtifactBisectionReport) -> str:
    """Render an artifact-bisection report for release maintainers."""

    lines = [
        "PromptABI artifact drift bisection",
        f"surface: {report.surface.value}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"baseline: {report.baseline.label} ({report.baseline.path})",
        f"candidate revisions: {len(report.revisions)}",
        f"probes: {len(report.probes)}",
    ]
    if report.bad_fields:
        lines.append(f"regression predicate: {', '.join(report.bad_fields)}")
    else:
        lines.append("regression predicate: any contract-relevant drift")
    if report.first_bad_revision is None:
        lines.append(f"first bad revision: <none>; newest checked revision is {report.previous_good_revision.label}")
    else:
        previous = report.previous_good_revision.label if report.previous_good_revision is not None else "<none>"
        lines.append(
            "first bad revision: "
            f"#{report.first_bad_index} {report.first_bad_revision.label} ({report.first_bad_revision.path})"
        )
        lines.append(f"previous good revision: {previous}")
        for finding in report.findings:
            lines.append(
                f"- {finding.kind} {finding.field}: "
                f"{_short_value(finding.baseline)} -> {_short_value(finding.current)}"
            )
    return "\n".join(lines) + "\n"


def artifact_revision_from_cli(value: str) -> ArtifactRevision:
    """Parse ``LABEL=PATH`` into a bisection revision."""

    if "=" not in value:
        raise ArtifactBisectionError("revision must use LABEL=PATH")
    label, path = value.split("=", 1)
    return ArtifactRevision(label.strip(), path.strip())


def _tokenizer_findings(
    surface: ArtifactDriftSurface,
    baseline_path: str | Path,
    current_path: str | Path,
) -> tuple[ArtifactDriftFinding, ...]:
    baseline = load_tokenizer_config_snapshot(baseline_path, revision="baseline")
    current = load_tokenizer_config_snapshot(current_path, revision="current")
    findings = tuple(_from_tokenizer_finding(surface, finding) for finding in compare_tokenizer_config_snapshots(baseline, current))
    if surface is ArtifactDriftSurface.TEMPLATE:
        findings = tuple(finding for finding in findings if finding.field.startswith("chat_template_"))
    return findings


def _from_tokenizer_finding(surface: ArtifactDriftSurface, finding: TokenizerDriftFinding) -> ArtifactDriftFinding:
    return ArtifactDriftFinding(
        surface=surface,
        field=finding.field,
        kind=finding.kind.value,
        baseline=finding.baseline,
        current=finding.current,
        baseline_path=finding.baseline_path,
        current_path=finding.current_path,
    )


def _json_findings(
    surface: ArtifactDriftSurface,
    baseline_path: str | Path,
    current_path: str | Path,
) -> tuple[ArtifactDriftFinding, ...]:
    baseline = _load_json_artifact(Path(baseline_path))
    current = _load_json_artifact(Path(current_path))
    baseline_flat = _flatten_json(baseline.value)
    current_flat = _flatten_json(current.value)
    findings: list[ArtifactDriftFinding] = []
    for field in sorted(set(baseline_flat).union(current_flat)):
        baseline_value = baseline_flat.get(field)
        current_value = current_flat.get(field)
        if baseline_value == current_value:
            continue
        if field not in baseline_flat:
            kind = "json-field-added"
        elif field not in current_flat:
            kind = "json-field-removed"
        else:
            kind = "json-field-changed"
        findings.append(
            ArtifactDriftFinding(
                surface=surface,
                field=field,
                kind=kind,
                baseline=baseline_value,
                current=current_value,
                baseline_path=baseline.path,
                current_path=current.path,
            )
        )
    if baseline.digest != current.digest and not findings:
        findings.append(
            ArtifactDriftFinding(
                surface=surface,
                field="/",
                kind="json-digest-changed",
                baseline=baseline.digest,
                current=current.digest,
                baseline_path=baseline.path,
                current_path=current.path,
            )
        )
    return tuple(findings)


@dataclass(frozen=True, slots=True)
class _JsonArtifactSnapshot:
    path: str
    value: object
    digest: str


def _load_json_artifact(path: Path) -> _JsonArtifactSnapshot:
    try:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return _JsonArtifactSnapshot(str(path), value, hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        if path.is_dir():
            values: dict[str, object] = {}
            for child in sorted(item for item in path.rglob("*.json") if item.is_file()):
                rel = child.relative_to(path).as_posix()
                values[rel] = json.loads(child.read_text(encoding="utf-8"))
            if not values:
                raise ArtifactBisectionError(f"JSON artifact directory has no .json files: {path}")
            canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
            return _JsonArtifactSnapshot(str(path), values, hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    except json.JSONDecodeError as exc:
        raise ArtifactBisectionError(f"artifact revision is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise ArtifactBisectionError(f"artifact revision could not be read at {path}: {exc}") from exc
    raise ArtifactBisectionError(f"artifact revision path is not a file or directory: {path}")


def _flatten_json(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {prefix or "/": {}}
        flattened: dict[str, object] = {}
        for key, item in sorted(value.items()):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            flattened.update(_flatten_json(item, f"{prefix}/{escaped}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return {prefix or "/": []}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_json(item, f"{prefix}/{index}"))
        return flattened
    return {prefix or "/": value}


def _matches_requested_field(finding: ArtifactDriftFinding, fields: tuple[str, ...]) -> bool:
    if not fields:
        return True
    for field in fields:
        if field.endswith("*") and finding.field.startswith(field[:-1]):
            return True
        if finding.field == field or finding.kind == field:
            return True
    return False


def _short_value(value: object) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 96 else text[:93] + "..."
