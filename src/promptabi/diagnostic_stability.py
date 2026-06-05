"""Prove that PromptABI diagnostics are stable under formatting-only changes.

A *formatting-only* change rewrites an artifact's bytes without changing its
meaning: re-indenting JSON, sorting object keys, collapsing or expanding
whitespace, or adding trailing newlines.  A trustworthy checker must report the
**same diagnostics** before and after such edits -- otherwise a cosmetic
reformat would spuriously add, drop, or mutate findings and destabilize CI.

This module proves that property against real verification runs.  It runs the
full :class:`promptabi.session.VerificationSession` on a config, then re-runs it
on semantics-preserving reformatted copies of every JSON artifact, and checks
that the *formatting-invariant identity* of every diagnostic is preserved
exactly (no dropped finding, no fabricated finding).  The identity deliberately
excludes byte spans, because line/column offsets legitimately move when bytes
move; the proof additionally records that spans really did shift so the
perturbation is not vacuous.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace

if TYPE_CHECKING:
    from .diagnostics import Diagnostic


DIAGNOSTIC_STABILITY_VERSION = "promptabi.diagnostic-stability.v1"


def _indent4_sorted(data: object) -> str:
    return json.dumps(data, indent=4, sort_keys=True) + "\n"


def _indent2(data: object) -> str:
    return json.dumps(data, indent=2)


def _compact(data: object) -> str:
    return json.dumps(data, separators=(",", ":"))


def _spacious_sorted(data: object) -> str:
    return "\n" + json.dumps(data, indent=8, sort_keys=True) + "\n\n"


#: Default semantics-preserving JSON reformatters applied as perturbations.
DEFAULT_FORMATTING_VARIANTS: tuple[tuple[str, Callable[[object], str]], ...] = (
    ("indent4-sorted", _indent4_sorted),
    ("indent2", _indent2),
    ("compact", _compact),
    ("spacious-sorted", _spacious_sorted),
)


class DiagnosticStabilityFindingKind(StrEnum):
    """Concrete ways a reformat perturbed the diagnostic set."""

    DROPPED_DIAGNOSTIC = "dropped-diagnostic"
    FABRICATED_DIAGNOSTIC = "fabricated-diagnostic"
    VACUOUS_PERTURBATION = "vacuous-perturbation"


@dataclass(frozen=True, slots=True)
class DiagnosticIdentity:
    """A formatting-invariant identity for a diagnostic (excludes byte spans)."""

    rule_id: str
    severity: str
    message: str
    check_modes: tuple[str, ...]
    artifact_kind: str | None
    artifact_name: str | None

    def to_tuple(self) -> tuple[object, ...]:
        return (
            self.rule_id,
            self.severity,
            self.message,
            self.check_modes,
            self.artifact_kind,
            self.artifact_name,
        )

    def describe(self) -> str:
        scope = f"{self.artifact_kind or '-'}:{self.artifact_name or '-'}"
        return f"{self.rule_id}[{self.severity}] {scope} :: {self.message}"


@dataclass(frozen=True, slots=True)
class DiagnosticStabilityFinding:
    """One stability violation observed for a specific formatting variant."""

    kind: DiagnosticStabilityFindingKind
    variant: str
    identity: str
    witness: WitnessTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "kind": self.kind.value,
            "variant": self.variant,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticStabilityVariantResult:
    """Per-variant comparison of baseline vs reformatted diagnostics."""

    variant: str
    baseline_count: int
    variant_count: int
    spans_shifted: int
    findings: tuple[DiagnosticStabilityFinding, ...]

    @property
    def stable(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_count": self.baseline_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "spans_shifted": self.spans_shifted,
            "stable": self.stable,
            "variant": self.variant,
            "variant_count": self.variant_count,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticStabilityReport:
    """Whole-config diagnostic stability proof across formatting variants."""

    version: str
    baseline_count: int
    variants: tuple[DiagnosticStabilityVariantResult, ...]

    @property
    def findings(self) -> tuple[DiagnosticStabilityFinding, ...]:
        return tuple(finding for variant in self.variants for finding in variant.findings)

    @property
    def ok(self) -> bool:
        return all(variant.stable for variant in self.variants)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_count": self.baseline_count,
            "ok": self.ok,
            "variants": [variant.to_dict() for variant in self.variants],
            "version": self.version,
        }


def formatting_invariant_identity(diagnostic: "Diagnostic") -> DiagnosticIdentity:
    """Return the formatting-invariant identity of a diagnostic (span excluded)."""

    artifact = diagnostic.artifact
    return DiagnosticIdentity(
        rule_id=diagnostic.rule_id,
        severity=_value(diagnostic.severity),
        message=diagnostic.message,
        check_modes=tuple(_value(mode) for mode in diagnostic.check_modes),
        artifact_kind=_value(artifact.kind) if artifact is not None else None,
        artifact_name=artifact.name if artifact is not None else None,
    )


def prove_diagnostic_stability_under_formatting(
    config_path: str | Path,
    *,
    variants: tuple[tuple[str, Callable[[object], str]], ...] = DEFAULT_FORMATTING_VARIANTS,
) -> DiagnosticStabilityReport:
    """Prove diagnostics are invariant under semantics-preserving reformatting."""

    from .config import load_config
    from .session import VerificationSession

    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    baseline = VerificationSession(load_config(config_path)).run()
    baseline_ids = Counter(formatting_invariant_identity(d).to_tuple() for d in baseline.diagnostics)
    id_text = {
        formatting_invariant_identity(d).to_tuple(): formatting_invariant_identity(d).describe()
        for d in baseline.diagnostics
    }
    baseline_spans = _span_multiset(baseline.diagnostics)

    results: list[DiagnosticStabilityVariantResult] = []
    for variant_name, transform in variants:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "repo"
            shutil.copytree(config_dir, tmp_dir)
            _reformat_json_tree(tmp_dir, transform)
            variant_config = tmp_dir / config_path.name
            variant_result = VerificationSession(load_config(variant_config)).run()

        variant_ids = Counter(
            formatting_invariant_identity(d).to_tuple() for d in variant_result.diagnostics
        )
        for tuple_id in variant_ids:
            id_text.setdefault(tuple_id, _describe_tuple(tuple_id))
        variant_spans = _span_multiset(variant_result.diagnostics)
        spans_shifted = sum((baseline_spans - variant_spans).values())

        findings: list[DiagnosticStabilityFinding] = []
        for tuple_id, count in (baseline_ids - variant_ids).items():
            findings.append(
                _finding(
                    DiagnosticStabilityFindingKind.DROPPED_DIAGNOSTIC,
                    variant_name,
                    id_text[tuple_id],
                    count,
                )
            )
        for tuple_id, count in (variant_ids - baseline_ids).items():
            findings.append(
                _finding(
                    DiagnosticStabilityFindingKind.FABRICATED_DIAGNOSTIC,
                    variant_name,
                    id_text[tuple_id],
                    count,
                )
            )
        if baseline_ids == variant_ids and baseline.diagnostics and spans_shifted == 0:
            findings.append(
                _finding(
                    DiagnosticStabilityFindingKind.VACUOUS_PERTURBATION,
                    variant_name,
                    "no diagnostic span moved; reformatting did not perturb byte offsets",
                    0,
                )
            )

        results.append(
            DiagnosticStabilityVariantResult(
                variant=variant_name,
                baseline_count=len(baseline.diagnostics),
                variant_count=len(variant_result.diagnostics),
                spans_shifted=spans_shifted,
                findings=tuple(findings),
            )
        )

    return DiagnosticStabilityReport(
        version=DIAGNOSTIC_STABILITY_VERSION,
        baseline_count=len(baseline.diagnostics),
        variants=tuple(results),
    )


def render_diagnostic_stability_json(report: DiagnosticStabilityReport) -> str:
    """Render the stability proof as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_diagnostic_stability_text(report: DiagnosticStabilityReport) -> str:
    """Render the stability proof for CI logs and reviewers."""

    lines = [
        f"PromptABI diagnostic stability under formatting ({report.version})",
        f"status: {'STABLE' if report.ok else 'UNSTABLE'}",
        f"baseline diagnostics: {report.baseline_count}",
    ]
    for variant in report.variants:
        lines.append("")
        lines.append(
            f"{variant.variant}: {'stable' if variant.stable else 'unstable'} "
            f"(spans shifted: {variant.spans_shifted})"
        )
        for finding in variant.findings:
            lines.append(f"  {finding.kind.value}: {finding.identity}")
    return "\n".join(lines) + "\n"


def _reformat_json_tree(root: Path, transform: Callable[[object], str]) -> None:
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        path.write_text(transform(data), encoding="utf-8")


def _span_multiset(diagnostics: tuple["Diagnostic", ...]) -> Counter:
    counter: Counter = Counter()
    for diagnostic in diagnostics:
        span = diagnostic.span
        if span is None:
            continue
        counter[(diagnostic.rule_id, span.start_line, span.start_column, span.end_line, span.end_column)] += 1
    return counter


def _finding(
    kind: DiagnosticStabilityFindingKind,
    variant: str,
    identity: str,
    count: int,
) -> DiagnosticStabilityFinding:
    if kind is DiagnosticStabilityFindingKind.DROPPED_DIAGNOSTIC:
        summary = f"reformat '{variant}' dropped a diagnostic: {identity}"
        fix = "Make the checker key its finding on semantics, not byte layout, so the diagnostic survives reformatting."
    elif kind is DiagnosticStabilityFindingKind.FABRICATED_DIAGNOSTIC:
        summary = f"reformat '{variant}' fabricated a diagnostic: {identity}"
        fix = "Remove the formatting-sensitive condition so reformatting cannot introduce a new finding."
    else:
        summary = f"reformat '{variant}' did not perturb byte offsets: {identity}"
        fix = "Choose a formatting variant that actually moves byte offsets to exercise span stability."
    return DiagnosticStabilityFinding(
        kind=kind,
        variant=variant,
        identity=identity,
        witness=WitnessTrace(
            summary=summary,
            steps=(
                WitnessStep(action="reformat artifacts (semantics-preserving)", input=variant, output="bytes changed"),
                WitnessStep(action="compare diagnostic identities", input=identity, output=kind.value),
            ),
            artifacts=(ArtifactRef(kind="diagnostic", name=identity[:80], path=None),),
            minimal_fixes=(fix,),
        ),
    )


def _describe_tuple(tuple_id: tuple[object, ...]) -> str:
    rule_id, severity, message, _modes, artifact_kind, artifact_name = tuple_id
    scope = f"{artifact_kind or '-'}:{artifact_name or '-'}"
    return f"{rule_id}[{severity}] {scope} :: {message}"


def _value(item: object) -> str:
    return item.value if hasattr(item, "value") else str(item)
