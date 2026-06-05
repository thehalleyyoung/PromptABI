"""Curated gallery of verified PromptABI configurations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .diagnostics import CheckMode, Diagnostic
from .loaders import LoadedArtifact
from .lockfiles import LockfileError, load_lockfile
from .session import VerificationResult, VerificationSession


class GalleryError(ValueError):
    """Raised when the verified configuration gallery cannot be built."""


@dataclass(frozen=True, slots=True)
class GalleryManifestEntry:
    """One curated gallery entry from the manifest."""

    id: str
    title: str
    summary: str
    config_path: Path
    surfaces: tuple[str, ...]
    lockfile_path: Path | None = None
    real_world: str = ""


@dataclass(frozen=True, slots=True)
class GalleryArtifactSummary:
    """Pinned/resolved state for one artifact in a verified gallery entry."""

    name: str
    kind: str
    location: str
    source_type: str
    pinned: bool
    resolved: bool
    sha256: str | None = None
    manifest_sha256: str | None = None

    @classmethod
    def from_loaded(cls, loaded: LoadedArtifact, *, base_dir: Path) -> "GalleryArtifactSummary":
        artifact = loaded.artifact
        location = artifact.location.ref_path or ""
        try:
            rendered_location = str(Path(location).resolve().relative_to(base_dir.resolve()))
        except (OSError, ValueError):
            rendered_location = location
        return cls(
            name=artifact.name,
            kind=artifact.kind.value,
            location=rendered_location,
            source_type=loaded.source_type,
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            sha256=loaded.actual_sha256 or artifact.provenance.sha256,
            manifest_sha256=loaded.manifest_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "location": self.location,
            "name": self.name,
            "pinned": self.pinned,
            "resolved": self.resolved,
            "source_type": self.source_type,
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.manifest_sha256 is not None:
            data["manifest_sha256"] = self.manifest_sha256
        return data


@dataclass(frozen=True, slots=True)
class GalleryProofSummary:
    """A concise proof or solver observation from a real diagnostic witness."""

    rule_id: str
    message: str
    modes: tuple[str, ...]
    actual_backend: str | None
    outcome: str
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "fingerprint": self.fingerprint,
            "message": self.message,
            "modes": list(self.modes),
            "outcome": self.outcome,
            "rule_id": self.rule_id,
        }
        if self.actual_backend is not None:
            data["actual_backend"] = self.actual_backend
        return data


@dataclass(frozen=True, slots=True)
class GalleryAcceptedRisk:
    """A documented suppression or abstention visible in the gallery."""

    rule_id: str
    message: str
    explanation: str
    owner: str | None = None
    expires_on: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "explanation": self.explanation,
            "message": self.message,
            "rule_id": self.rule_id,
        }
        if self.owner is not None:
            data["owner"] = self.owner
        if self.expires_on is not None:
            data["expires_on"] = self.expires_on
        return data


@dataclass(frozen=True, slots=True)
class GalleryEntryReport:
    """Verification result and badges for one gallery entry."""

    id: str
    title: str
    summary: str
    real_world: str
    config_path: str
    surfaces: tuple[str, ...]
    status: str
    badges: tuple[str, ...]
    severity_counts: tuple[tuple[str, int], ...]
    artifacts: tuple[GalleryArtifactSummary, ...]
    proof_summaries: tuple[GalleryProofSummary, ...]
    abstentions: tuple[GalleryAcceptedRisk, ...]
    accepted_risks: tuple[GalleryAcceptedRisk, ...]
    diagnostics: tuple[Diagnostic, ...]
    lockfile: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "abstentions": [item.to_dict() for item in self.abstentions],
            "accepted_risks": [item.to_dict() for item in self.accepted_risks],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "badges": list(self.badges),
            "config_path": self.config_path,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "id": self.id,
            "proof_summaries": [summary.to_dict() for summary in self.proof_summaries],
            "real_world": self.real_world,
            "severity_counts": dict(self.severity_counts),
            "status": self.status,
            "summary": self.summary,
            "surfaces": list(self.surfaces),
            "title": self.title,
        }
        if self.lockfile is not None:
            data["lockfile"] = self.lockfile
        return data


@dataclass(frozen=True, slots=True)
class GalleryReport:
    """A deterministic report for the whole curated gallery."""

    entries: tuple[GalleryEntryReport, ...]
    root: str

    @property
    def ok(self) -> bool:
        return all(entry.status == "PASS" for entry in self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "ok": self.ok,
            "root": self.root,
            "summary": {
                "entries": len(self.entries),
                "passing": sum(1 for entry in self.entries if entry.status == "PASS"),
                "accepted_risks": sum(len(entry.accepted_risks) for entry in self.entries),
                "abstentions": sum(len(entry.abstentions) for entry in self.entries),
            },
        }


def build_gallery(root: str | Path | None = None) -> GalleryReport:
    """Run the curated gallery configs through the real PromptABI verifier."""

    gallery_root = Path(root).resolve() if root is not None else _default_gallery_root()
    manifest_path = gallery_root / "manifest.json"
    entries = _load_manifest(manifest_path)
    reports = tuple(_build_entry_report(entry, gallery_root=gallery_root) for entry in entries)
    return GalleryReport(entries=reports, root=str(gallery_root))


def render_gallery_json(report: GalleryReport) -> str:
    """Render the gallery report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_gallery_text(report: GalleryReport) -> str:
    """Render a compact human-readable gallery."""

    lines = [
        "PromptABI verified configuration gallery",
        f"root: {report.root}",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        "",
    ]
    for entry in report.entries:
        badge_text = " ".join(f"[{badge}]" for badge in entry.badges)
        lines.append(f"{entry.id}: {entry.title} {badge_text}".rstrip())
        lines.append(f"  status: {entry.status}")
        lines.append(f"  surfaces: {', '.join(entry.surfaces)}")
        lines.append(f"  config: {entry.config_path}")
        if entry.lockfile is not None:
            lines.append(f"  lockfile: {entry.lockfile}")
        lines.append(f"  diagnostics: {_format_counts(entry.severity_counts)}")
        if entry.artifacts:
            pinned = sum(1 for artifact in entry.artifacts if artifact.pinned)
            resolved = sum(1 for artifact in entry.artifacts if artifact.resolved)
            lines.append(f"  artifacts: {pinned}/{len(entry.artifacts)} pinned, {resolved}/{len(entry.artifacts)} resolved")
        for proof in entry.proof_summaries[:3]:
            backend = f", backend={proof.actual_backend}" if proof.actual_backend is not None else ""
            lines.append(f"  proof: {proof.rule_id} -> {proof.outcome}{backend}")
        for risk in entry.accepted_risks:
            owner = f" ({risk.owner})" if risk.owner else ""
            lines.append(f"  accepted risk{owner}: {risk.explanation}")
        for abstention in entry.abstentions:
            lines.append(f"  abstention: {abstention.explanation}")
        lines.append(f"  why it matters: {entry.summary}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_entry_report(entry: GalleryManifestEntry, *, gallery_root: Path) -> GalleryEntryReport:
    try:
        config = load_config(entry.config_path)
    except ConfigError as exc:
        raise GalleryError(f"gallery entry '{entry.id}' has invalid config: {exc}") from exc
    session = VerificationSession(config)
    result = session.run()
    loaded, _load_diagnostics = session.load_artifacts_with_diagnostics()
    lockfile_path = _entry_lockfile(entry)
    lockfile_label = _summarize_lockfile(lockfile_path, gallery_root=gallery_root) if lockfile_path is not None else None
    artifacts = tuple(
        sorted(
            (GalleryArtifactSummary.from_loaded(artifact, base_dir=gallery_root.parent) for artifact in loaded),
            key=lambda artifact: (artifact.kind, artifact.name),
        )
    )
    severity_counts = _severity_counts(result)
    accepted_risks = _accepted_risks(result)
    abstentions = _abstentions(result)
    proof_summaries = _proof_summaries(result)
    badges = _badges(
        result,
        artifacts=artifacts,
        accepted_risks=accepted_risks,
        abstentions=abstentions,
        lockfile_label=lockfile_label,
    )
    return GalleryEntryReport(
        id=entry.id,
        title=entry.title,
        summary=entry.summary,
        real_world=entry.real_world,
        config_path=_relative_to(entry.config_path, gallery_root.parent),
        surfaces=entry.surfaces,
        status="PASS" if result.ok else "FAIL",
        badges=badges,
        severity_counts=severity_counts,
        artifacts=artifacts,
        proof_summaries=proof_summaries,
        abstentions=abstentions,
        accepted_risks=accepted_risks,
        diagnostics=result.diagnostics,
        lockfile=lockfile_label,
    )


def _load_manifest(path: Path) -> tuple[GalleryManifestEntry, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GalleryError(f"gallery manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GalleryError(f"gallery manifest is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise GalleryError("gallery manifest root must be an object")
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise GalleryError("gallery manifest field 'entries' must be a non-empty list")
    entries = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_entries, start=1):
        if not isinstance(item, dict):
            raise GalleryError(f"gallery manifest entry {index} must be an object")
        entry_id = _required_str(item, "id", index=index)
        if entry_id in seen_ids:
            raise GalleryError(f"gallery manifest entry id is duplicated: {entry_id}")
        seen_ids.add(entry_id)
        config = path.parent / _required_str(item, "config", index=index)
        surfaces = item.get("surfaces", [])
        if not isinstance(surfaces, list) or not all(isinstance(surface, str) and surface for surface in surfaces):
            raise GalleryError(f"gallery manifest entry '{entry_id}' surfaces must be a list of strings")
        lockfile = item.get("lockfile")
        if lockfile is not None and not isinstance(lockfile, str):
            raise GalleryError(f"gallery manifest entry '{entry_id}' lockfile must be a string")
        entries.append(
            GalleryManifestEntry(
                id=entry_id,
                title=_required_str(item, "title", index=index),
                summary=_required_str(item, "summary", index=index),
                config_path=config.resolve(),
                surfaces=tuple(surfaces),
                lockfile_path=(path.parent / lockfile).resolve() if lockfile is not None else None,
                real_world=str(item.get("real_world", "")),
            )
        )
    return tuple(entries)


def _required_str(item: dict[str, Any], key: str, *, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GalleryError(f"gallery manifest entry {index} field '{key}' must be a non-empty string")
    return value.strip()


def _entry_lockfile(entry: GalleryManifestEntry) -> Path | None:
    if entry.lockfile_path is not None:
        return entry.lockfile_path
    candidate = entry.config_path.with_name("promptabi.lock.json")
    return candidate if candidate.is_file() else None


def _summarize_lockfile(path: Path, *, gallery_root: Path) -> str:
    try:
        lockfile = load_lockfile(path)
    except LockfileError as exc:
        raise GalleryError(f"gallery lockfile could not be loaded: {path}: {exc}") from exc
    pinned = sum(1 for artifact in lockfile.artifacts if artifact.pinned)
    return f"{_relative_to(path, gallery_root.parent)} ({pinned}/{len(lockfile.artifacts)} pinned)"


def _severity_counts(result: VerificationResult) -> tuple[tuple[str, int], ...]:
    counts = Counter(diagnostic.severity.value for diagnostic in result.diagnostics)
    return tuple((severity, counts.get(severity, 0)) for severity in ("error", "warning", "info"))


def _accepted_risks(result: VerificationResult) -> tuple[GalleryAcceptedRisk, ...]:
    risks = []
    for diagnostic in result.diagnostics:
        properties = dict(diagnostic.properties)
        accepted_risk = properties.get("accepted_risk")
        suppression = properties.get("suppression")
        if not isinstance(accepted_risk, str):
            continue
        owner = suppression.get("owner") if isinstance(suppression, dict) else None
        expires_on = suppression.get("expires_on") if isinstance(suppression, dict) else None
        risks.append(
            GalleryAcceptedRisk(
                rule_id=diagnostic.rule_id,
                message=diagnostic.message,
                explanation=accepted_risk,
                owner=owner if isinstance(owner, str) else None,
                expires_on=expires_on if isinstance(expires_on, str) else None,
            )
        )
    return tuple(risks)


def _abstentions(result: VerificationResult) -> tuple[GalleryAcceptedRisk, ...]:
    abstentions = []
    for diagnostic in result.diagnostics:
        if CheckMode.ABSTAINING not in diagnostic.check_modes and "abstain" not in diagnostic.rule_id:
            continue
        explanation = diagnostic.message
        if diagnostic.witness is not None:
            for step in diagnostic.witness.steps:
                if step.action in {"record solver abstention reason", "record solver unknown reason"} and step.output is not None:
                    explanation = step.output
                    break
        abstentions.append(
            GalleryAcceptedRisk(
                rule_id=diagnostic.rule_id,
                message=diagnostic.message,
                explanation=explanation,
            )
        )
    return tuple(abstentions)


def _proof_summaries(result: VerificationResult) -> tuple[GalleryProofSummary, ...]:
    summaries = []
    for diagnostic in result.diagnostics:
        if not _is_proof_like(diagnostic):
            continue
        backend, outcome = _witness_backend_and_outcome(diagnostic)
        summaries.append(
            GalleryProofSummary(
                rule_id=diagnostic.rule_id,
                message=diagnostic.message,
                modes=tuple(mode.value for mode in diagnostic.check_modes),
                actual_backend=backend,
                outcome=outcome,
                fingerprint=diagnostic.fingerprint,
            )
        )
    return tuple(sorted(summaries, key=lambda item: (item.rule_id, item.fingerprint)))


def _is_proof_like(diagnostic: Diagnostic) -> bool:
    modes = set(diagnostic.check_modes)
    return bool({CheckMode.SOUND, CheckMode.COMPLETE, CheckMode.Z3_BACKED_SMT}.intersection(modes))


def _witness_backend_and_outcome(diagnostic: Diagnostic) -> tuple[str | None, str]:
    backend = None
    outcome = diagnostic.severity.value
    if diagnostic.witness is None:
        return backend, outcome
    for step in diagnostic.witness.steps:
        if step.action == "solve finite contract" and step.input is not None:
            backend = step.input
            outcome = step.output or outcome
        elif step.action == "classify SMT diagnostic" and step.output is not None:
            outcome = step.output
        elif step.action.startswith("extract ") and step.output is not None:
            outcome = step.action.removeprefix("extract ")
    return backend, outcome


def _badges(
    result: VerificationResult,
    *,
    artifacts: tuple[GalleryArtifactSummary, ...],
    accepted_risks: tuple[GalleryAcceptedRisk, ...],
    abstentions: tuple[GalleryAcceptedRisk, ...],
    lockfile_label: str | None,
) -> tuple[str, ...]:
    badges = ["PASS" if result.ok else "FAIL"]
    if artifacts and all(artifact.pinned for artifact in artifacts):
        badges.append("PINNED")
    if artifacts and all(artifact.resolved for artifact in artifacts):
        badges.append("OFFLINE")
    if lockfile_label is not None:
        badges.append("LOCKFILE")
    modes = {mode for diagnostic in result.diagnostics for mode in diagnostic.check_modes}
    if CheckMode.SOUND in modes:
        badges.append("SOUND")
    if CheckMode.Z3_BACKED_SMT in modes:
        badges.append("SMT-CAPABLE")
    if any(summary.actual_backend == "z3" for summary in _proof_summaries(result)):
        badges.append("Z3-RUN")
    if abstentions:
        badges.append("ABSTAINS")
    if accepted_risks:
        badges.append("ACCEPTED-RISK")
    return tuple(dict.fromkeys(badges))


def _format_counts(counts: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{severity}={count}" for severity, count in counts)


def _relative_to(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _default_gallery_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "gallery"
