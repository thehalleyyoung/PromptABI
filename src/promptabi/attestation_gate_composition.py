"""Compose runtime attestations with deployment gates into an admission decision.

A runtime attestation says *what a service is actually running*; a deployment gate
says *what was approved for release*.  Neither alone proves the running service is
the approved one.  This module composes the two: it admits a deployment only when
the live attestation's signed bundle identity, reproducibility hash, signing key,
contract families, and environment match the gate that guards the release -- and
otherwise emits a replayable witness for every admission-denying mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .deployment_gates import DeploymentGateReport, build_deployment_gate_report
from .diagnostics import ArtifactRef, WitnessStep, WitnessTrace
from .runtime_attestation import (
    RUNTIME_CONTRACT_FAMILIES,
    RuntimeAttestationReport,
    build_runtime_attestation_report,
)


ATTESTATION_GATE_VERSION = "promptabi.attestation-gate.v1"


class AttestationGateError(ValueError):
    """Raised when an attestation/gate composition cannot be evaluated."""


class AttestationGateDecision(StrEnum):
    """The composed admission decision."""

    ADMIT = "admit"
    DENY = "deny"


class AttestationGateFindingKind(StrEnum):
    """Concrete reasons a live attestation fails the deployment gate."""

    ATTESTATION_NOT_OK = "attestation-not-ok"
    ATTESTATION_BLOCKED = "attestation-blocked"
    GATE_NOT_OK = "gate-not-ok"
    BUNDLE_HASH_MISMATCH = "bundle-hash-mismatch"
    REPRODUCIBILITY_HASH_MISMATCH = "reproducibility-hash-mismatch"
    SIGNING_KEY_MISMATCH = "signing-key-mismatch"
    MISSING_REQUIRED_FAMILY = "missing-required-family"
    ENVIRONMENT_MISMATCH = "environment-mismatch"


@dataclass(frozen=True, slots=True)
class AttestationGateFinding:
    """One admission-denying mismatch with a replayable witness."""

    kind: AttestationGateFindingKind
    message: str
    expected: str
    actual: str
    witness: WitnessTrace
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "actual": self.actual,
            "expected": self.expected,
            "kind": self.kind.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AttestationGateReport:
    """Composed admission decision for a live attestation against a deployment gate."""

    service: str
    environment: str
    decision: AttestationGateDecision
    attestation_bundle_hash: str
    gate_bundle_hash: str
    findings: tuple[AttestationGateFinding, ...]

    @property
    def admitted(self) -> bool:
        return self.decision is AttestationGateDecision.ADMIT

    def to_dict(self) -> dict[str, object]:
        return {
            "attestation_bundle_hash": self.attestation_bundle_hash,
            "decision": self.decision.value,
            "environment": self.environment,
            "findings": [finding.to_dict() for finding in self.findings],
            "gate_bundle_hash": self.gate_bundle_hash,
            "service": self.service,
            "version": ATTESTATION_GATE_VERSION,
        }


def compose_attestation_with_gate(
    attestation: RuntimeAttestationReport,
    gate: DeploymentGateReport,
    *,
    required_families: tuple[str, ...] = RUNTIME_CONTRACT_FAMILIES,
    expected_environment: str | None = None,
) -> AttestationGateReport:
    """Admit a deployment only if the live attestation matches the approving gate."""

    attestation_ref = ArtifactRef(
        kind="runtime-attestation", name=attestation.service, path=attestation.source_config
    )
    gate_ref = ArtifactRef(kind="deployment-gate", name=gate.source_config, path=gate.source_config)
    refs = (attestation_ref, gate_ref)
    findings: list[AttestationGateFinding] = []

    if not attestation.ok:
        findings.append(
            _finding(
                AttestationGateFindingKind.ATTESTATION_NOT_OK,
                "live attestation reports a failing verification result",
                expected="ok=True",
                actual="ok=False",
                refs=refs,
                suggestion="Re-verify the running config and redeploy a passing bundle before admission.",
            )
        )
    for blocker in attestation.blockers:
        findings.append(
            _finding(
                AttestationGateFindingKind.ATTESTATION_BLOCKED,
                f"attestation carries an unresolved blocker: {blocker}",
                expected="no blockers",
                actual=blocker,
                refs=refs,
                suggestion="Resolve the attestation blocker; the gate must not admit blocked services.",
            )
        )
    if not gate.ok:
        findings.append(
            _finding(
                AttestationGateFindingKind.GATE_NOT_OK,
                "deployment gate did not approve the candidate bundle",
                expected="gate.ok=True",
                actual="gate.ok=False",
                refs=refs,
                suggestion="Fix the diagnostics that keep the deployment gate red before composing it with attestation.",
            )
        )

    _compare(
        findings,
        AttestationGateFindingKind.BUNDLE_HASH_MISMATCH,
        "running bundle hash does not match the approved deployment bundle",
        expected=gate.bundle_hash,
        actual=attestation.bundle_hash,
        refs=refs,
        suggestion="Roll the running service to the exact bundle the gate approved.",
    )
    _compare(
        findings,
        AttestationGateFindingKind.REPRODUCIBILITY_HASH_MISMATCH,
        "running reproducibility hash diverges from the approved bundle",
        expected=gate.reproducibility_hash,
        actual=attestation.reproducibility_hash,
        refs=refs,
        suggestion="Rebuild from the approved sources; a reproducibility drift means the artifacts changed.",
    )
    _compare(
        findings,
        AttestationGateFindingKind.SIGNING_KEY_MISMATCH,
        "running bundle was signed by a different key than the gate trusts",
        expected=gate.signing_key_id,
        actual=attestation.signing_key_id,
        refs=refs,
        suggestion="Sign release bundles with the key the deployment gate is configured to trust.",
    )

    present_families = {contract.family for contract in attestation.contracts}
    for family in required_families:
        if family not in present_families:
            findings.append(
                _finding(
                    AttestationGateFindingKind.MISSING_REQUIRED_FAMILY,
                    f"attestation does not cover required contract family '{family}'",
                    expected=family,
                    actual=", ".join(sorted(present_families)) or "none",
                    refs=refs,
                    suggestion=f"Attest the verified '{family}' contract before the gate can admit the service.",
                )
            )

    if expected_environment is not None and attestation.environment != expected_environment:
        findings.append(
            _finding(
                AttestationGateFindingKind.ENVIRONMENT_MISMATCH,
                "attestation environment does not match the gate's protected environment",
                expected=expected_environment,
                actual=attestation.environment,
                refs=refs,
                suggestion="Compose attestation and gate for the same environment.",
            )
        )

    decision = AttestationGateDecision.ADMIT if not findings else AttestationGateDecision.DENY
    return AttestationGateReport(
        service=attestation.service,
        environment=attestation.environment,
        decision=decision,
        attestation_bundle_hash=attestation.bundle_hash,
        gate_bundle_hash=gate.bundle_hash,
        findings=tuple(findings),
    )


def compose_attestation_gate_from_config(
    attestation_config: str | Path,
    gate_config: str | Path | None = None,
    *,
    attestation_key: str | bytes,
    gate_key: str | bytes | None = None,
    bundle_key_id: str = "attestation-gate",
    service: str = "promptabi-service",
    environment: str = "production",
    required_families: tuple[str, ...] = RUNTIME_CONTRACT_FAMILIES,
) -> AttestationGateReport:
    """End-to-end composition: build both reports from real configs, then compose.

    When ``gate_config``/``gate_key`` differ from the attestation inputs the
    composition denies admission with concrete bundle/key mismatch witnesses.
    """

    attestation = build_runtime_attestation_report(
        attestation_config,
        bundle_key=attestation_key,
        bundle_key_id=_key_id(attestation_key, bundle_key_id),
        service=service,
        environment=environment,
    )
    resolved_gate_key = gate_key if gate_key is not None else attestation_key
    gate = build_deployment_gate_report(
        gate_config if gate_config is not None else attestation_config,
        bundle_key=resolved_gate_key,
        bundle_key_id=_key_id(resolved_gate_key, bundle_key_id),
    )
    return compose_attestation_with_gate(
        attestation,
        gate,
        required_families=required_families,
        expected_environment=environment,
    )


def _key_id(key: str | bytes, fallback: str) -> str:
    """Derive a stable signing-key identity so different keys gate differently."""

    if isinstance(key, str) and key:
        return key
    return fallback


def render_attestation_gate_json(report: AttestationGateReport) -> str:
    """Render a composed admission report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_attestation_gate_text(report: AttestationGateReport) -> str:
    """Render a composed admission report for CLI users."""

    lines = [
        "PromptABI attestation x deployment-gate composition",
        f"service: {report.service}",
        f"environment: {report.environment}",
        f"decision: {report.decision.value.upper()}",
        f"attestation_bundle_hash: {report.attestation_bundle_hash}",
        f"gate_bundle_hash: {report.gate_bundle_hash}",
    ]
    if report.admitted:
        lines.append("findings: none")
        return "\n".join(lines) + "\n"
    lines.append(f"findings: {len(report.findings)}")
    for finding in report.findings:
        lines.append(f"DENY {finding.kind.value}: {finding.message}")
        lines.append(f"  expected: {finding.expected}")
        lines.append(f"  actual:   {finding.actual}")
        lines.append(f"  suggestion: {finding.suggestion}")
    return "\n".join(lines) + "\n"


def _compare(
    findings: list[AttestationGateFinding],
    kind: AttestationGateFindingKind,
    message: str,
    *,
    expected: str,
    actual: str,
    refs: tuple[ArtifactRef, ...],
    suggestion: str,
) -> None:
    if expected != actual:
        findings.append(
            _finding(kind, message, expected=expected, actual=actual, refs=refs, suggestion=suggestion)
        )


def _finding(
    kind: AttestationGateFindingKind,
    message: str,
    *,
    expected: str,
    actual: str,
    refs: tuple[ArtifactRef, ...],
    suggestion: str,
) -> AttestationGateFinding:
    return AttestationGateFinding(
        kind=kind,
        message=message,
        expected=expected,
        actual=actual,
        witness=WitnessTrace(
            summary=f"deployment gate denies admission: {kind.value}",
            steps=(
                WitnessStep(action="read live runtime attestation", input="attestation", output=actual),
                WitnessStep(action="read approving deployment gate", input="gate", output=expected),
                WitnessStep(action="emit minimal admission fix", input=kind.value, output=suggestion),
            ),
            artifacts=refs,
            minimal_fixes=(suggestion,),
        ),
        suggestion=suggestion,
    )
