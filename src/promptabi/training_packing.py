"""Dataset-packing verification for training manifests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .artifacts import (
    LossMaskStrategy,
    PackingStrategy,
    TrainingManifestArtifact,
    TrainingSpanContract,
)


class TrainingPackingFindingKind(StrEnum):
    """Finite packing invariants PromptABI can prove from manifest span facts."""

    MISSING_PACKING_WINDOW = "missing-packing-window"
    BOUNDARY_UNPRESERVED = "boundary-unpreserved"
    BOUNDARY_TOKEN_MISSING = "boundary-token-missing"
    SPAN_CROSSES_BOUNDARY = "span-crosses-boundary"
    SPAN_TRUNCATED = "span-truncated"
    MASK_DROPPED = "mask-dropped"
    ROLE_DELIMITER_DRIFT = "role-delimiter-drift"
    BOS_EOS_AMBIGUOUS = "bos-eos-ambiguous"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class TrainingPackingFinding:
    """One dataset-packing contract outcome for a finite training span."""

    kind: TrainingPackingFindingKind
    manifest_name: str
    message: str
    severity: str
    span_id: str | None = None
    packed_example_id: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingPackingReport:
    """Bounded result for a training manifest's packing contract."""

    manifest_name: str
    packing_strategy: PackingStrategy | None
    max_tokens: int | None
    span_count: int
    findings: tuple[TrainingPackingFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is TrainingPackingFindingKind.VERIFIED for finding in self.findings
        )


def analyze_training_packing(manifest: TrainingManifestArtifact) -> TrainingPackingReport:
    """Check finite packing-window facts declared by a training manifest.

    The checker is intentionally bounded: it proves the provided rendered/tokenized
    span facts are internally compatible with declared packing, truncation, BOS/EOS
    boundary, role-region, and loss-mask policies. It does not materialize private
    datasets or require GPU/model weights.
    """

    findings: list[TrainingPackingFinding] = []
    window = manifest.packing_window
    packing_strategy = window.strategy if window is not None else None
    max_tokens = window.max_tokens if window is not None else None
    active_packing = packing_strategy is not None and packing_strategy is not PackingStrategy.NONE

    if manifest.packed or active_packing or manifest.supervised_spans:
        if window is None:
            findings.append(
                _manifest_finding(
                    manifest,
                    TrainingPackingFindingKind.MISSING_PACKING_WINDOW,
                    "training manifest declares packed/supervised spans without a packing_window",
                    "warning",
                    (("inspect packing policy", None, "missing"),),
                )
            )
        elif active_packing and not window.preserve_example_boundaries:
            findings.append(
                _manifest_finding(
                    manifest,
                    TrainingPackingFindingKind.BOUNDARY_UNPRESERVED,
                    "packing policy does not preserve example boundaries",
                    "error",
                    (
                        ("select packing strategy", None, window.strategy.value),
                        ("inspect preserve_example_boundaries", None, "false"),
                    ),
                )
            )
        elif active_packing and window.boundary_token is None:
            findings.append(
                _manifest_finding(
                    manifest,
                    TrainingPackingFindingKind.BOUNDARY_TOKEN_MISSING,
                    "packing policy has no BOS/EOS or boundary token between packed examples",
                    "warning",
                    (
                        ("select packing strategy", None, window.strategy.value),
                        ("inspect boundary token", None, "missing"),
                    ),
                )
            )

    for span in manifest.supervised_spans:
        findings.extend(_span_findings(manifest, span, max_tokens=max_tokens))

    if not findings and manifest.supervised_spans and window is not None:
        findings.append(
            _manifest_finding(
                manifest,
                TrainingPackingFindingKind.VERIFIED,
                f"training manifest '{manifest.name}' preserves {len(manifest.supervised_spans)} supervised span(s) under packing",
                "info",
                (
                    ("select packing strategy", None, window.strategy.value),
                    ("check supervised spans", None, f"{len(manifest.supervised_spans)} finite span contracts"),
                    ("prove bounds", None, f"max_tokens={window.max_tokens}"),
                ),
            )
        )

    return TrainingPackingReport(
        manifest_name=manifest.name,
        packing_strategy=packing_strategy,
        max_tokens=max_tokens,
        span_count=len(manifest.supervised_spans),
        findings=tuple(findings),
    )


def _span_findings(
    manifest: TrainingManifestArtifact,
    span: TrainingSpanContract,
    *,
    max_tokens: int | None,
) -> tuple[TrainingPackingFinding, ...]:
    findings: list[TrainingPackingFinding] = []
    if span.crosses_packing_boundary:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.SPAN_CROSSES_BOUNDARY,
                f"supervised span '{span.span_id}' crosses a packed-example boundary",
                "error",
                (
                    ("select supervised span", span.span_id, _span_range(span)),
                    ("inspect packing boundary flag", None, "crosses_packing_boundary=true"),
                ),
            )
        )
    if max_tokens is not None and span.region_end_token > max_tokens:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.SPAN_TRUNCATED,
                f"supervised span '{span.span_id}' can be truncated by the packing window",
                "error",
                (
                    ("select supervised span", span.span_id, _span_range(span)),
                    ("compare region end to packing window", str(span.region_end_token), f"max_tokens={max_tokens}"),
                ),
            )
        )
    if span.start_token < span.region_start_token or span.end_token > span.region_end_token:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.ROLE_DELIMITER_DRIFT,
                f"supervised span '{span.span_id}' is outside its rendered role region",
                "error",
                (
                    ("select supervised span", span.span_id, _span_range(span)),
                    ("compare role region", None, f"{span.region_start_token}:{span.region_end_token}"),
                ),
            )
        )
    if span.supervised_target and not span.loss_masked:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.MASK_DROPPED,
                f"supervised target span '{span.span_id}' is not covered by the loss mask",
                "error",
                (
                    ("select supervised span", span.span_id, _span_range(span)),
                    ("inspect loss mask", None, "loss_masked=false"),
                ),
            )
        )
    loss_policy = manifest.loss_mask_policy
    if (
        loss_policy is not None
        and loss_policy.strategy in {LossMaskStrategy.ASSISTANT_ONLY, LossMaskStrategy.COMPLETION_ONLY, LossMaskStrategy.EXPLICIT}
        and loss_policy.target_roles
        and span.target_role not in loss_policy.target_roles
    ):
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.MASK_DROPPED,
                f"supervised target span '{span.span_id}' role is absent from loss_mask_policy.target_roles",
                "error",
                (
                    ("select supervised span", span.span_id, f"target_role={span.target_role}"),
                    ("inspect loss mask target roles", None, ", ".join(loss_policy.target_roles)),
                ),
            )
        )
    if span.target_role != span.rendered_region_role:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.ROLE_DELIMITER_DRIFT,
                f"supervised span '{span.span_id}' target role differs from rendered role region",
                "error",
                (
                    ("select supervised span", span.span_id, f"target_role={span.target_role}"),
                    ("inspect rendered role region", None, span.rendered_region_role),
                ),
            )
        )
    if span.region_start_token == span.start_token or span.end_token == span.region_end_token:
        findings.append(
            _span_finding(
                manifest,
                span,
                TrainingPackingFindingKind.BOS_EOS_AMBIGUOUS,
                f"supervised span '{span.span_id}' touches a role boundary or BOS/EOS delimiter",
                "warning",
                (
                    ("select supervised span", span.span_id, _span_range(span)),
                    ("compare role delimiter margins", None, f"region={span.region_start_token}:{span.region_end_token}"),
                ),
            )
        )
    return tuple(findings)


def _manifest_finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingPackingFindingKind,
    message: str,
    severity: str,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingPackingFinding:
    return TrainingPackingFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        witness=witness,
    )


def _span_finding(
    manifest: TrainingManifestArtifact,
    span: TrainingSpanContract,
    kind: TrainingPackingFindingKind,
    message: str,
    severity: str,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingPackingFinding:
    return TrainingPackingFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        span_id=span.span_id,
        packed_example_id=span.packed_example_id,
        witness=witness,
    )


def _span_range(span: TrainingSpanContract) -> str:
    return f"{span.start_token}:{span.end_token}"
