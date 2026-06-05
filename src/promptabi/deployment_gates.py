"""Deployment-gate examples backed by signed PromptABI verification bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._version import __version__
from .integration_api import IntegrationGate, build_integration_report
from .bundles import _stable_json_hash
from .safe_deployment_cores import MinimalUnsatCoreCertificate, derive_safe_deployment_cores


DEPLOYMENT_GATE_MANIFEST_VERSION = "promptabi.deployment-gates.v1"


class DeploymentGateSurface(StrEnum):
    """Deployment systems that can enforce a current PromptABI bundle."""

    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    GITHUB_ENVIRONMENTS = "github-environments"
    INTERNAL_RELEASE_SYSTEM = "internal-release-system"


DEPLOYMENT_GATE_SURFACES: tuple[DeploymentGateSurface, ...] = (
    DeploymentGateSurface.KUBERNETES,
    DeploymentGateSurface.TERRAFORM,
    DeploymentGateSurface.GITHUB_ENVIRONMENTS,
    DeploymentGateSurface.INTERNAL_RELEASE_SYSTEM,
)


class DeploymentGateError(ValueError):
    """Raised when deployment gates cannot be built from trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class DeploymentGateExample:
    """One deploy-time gate that requires the current signed PromptABI bundle."""

    surface: DeploymentGateSurface
    title: str
    path: str
    command: str
    policy: tuple[str, ...]
    content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface.value,
            "title": self.title,
            "path": self.path,
            "command": self.command,
            "policy": list(self.policy),
            "content": self.content,
        }


@dataclass(frozen=True, slots=True)
class DeploymentGateReport:
    """Deployment-gate bundle with machine-readable evidence and rendered examples."""

    source_config: str
    gate: IntegrationGate
    ok: bool
    bundle_hash: str
    reproducibility_hash: str
    signing_key_id: str
    diagnostic_counts: Mapping[str, int]
    safe_deployment_cores: tuple[MinimalUnsatCoreCertificate, ...]
    examples: tuple[DeploymentGateExample, ...]
    blockers: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": DEPLOYMENT_GATE_MANIFEST_VERSION,
            "promptabi_version": __version__,
            "source_config": self.source_config,
            "gate": self.gate.value,
            "ok": self.ok,
            "bundle": {
                "bundle_hash": self.bundle_hash,
                "reproducibility_hash": self.reproducibility_hash,
                "signing_key_id": self.signing_key_id,
            },
            "diagnostic_counts": dict(sorted(self.diagnostic_counts.items())),
            "safe_deployment_cores": [certificate.to_dict() for certificate in self.safe_deployment_cores],
            "blockers": list(self.blockers),
            "examples": [example.to_dict() for example in self.examples],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class DeploymentGateWriteResult:
    """Files written by the deployment-gate example writer."""

    output_dir: Path
    report: DeploymentGateReport
    written_files: tuple[Path, ...]


def build_deployment_gate_report(
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "deployment-gate",
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
) -> DeploymentGateReport:
    """Build deploy-time gates from a live verification run and signed bundle evidence."""

    config_path = Path(config).expanduser().resolve()
    report = build_integration_report(
        config_path,
        surfaces=("ci-provider", "model-registry", "internal-platform"),
        fail_on=fail_on,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        workspace_root=workspace_root,
    )
    registry_surface = _mapping(report.surfaces.get("model-registry"), "model-registry surface")
    signed_bundle = _mapping(registry_surface.get("signed_bundle"), "signed bundle evidence")
    if signed_bundle.get("available") is not True:
        reason = signed_bundle.get("reason", "signed bundle evidence is unavailable")
        raise DeploymentGateError(
            f"deployment gates require a current signed verification bundle; provide a bundle signing key ({reason})"
        )
    bundle_hash = _required_str(signed_bundle, "bundle_hash")
    signing_key_id = _required_str(signed_bundle, "signing_key_id")
    reproducibility_hash = _required_str(registry_surface, "reproducibility_hash")
    blockers = _deployment_blockers(report.ok, report.gate, report.diagnostic_counts)
    safe_core_report = derive_safe_deployment_cores()
    source_config = report.request.config_path or _relative_to_cwd(config_path)
    examples = _build_examples(
        source_config=source_config,
        bundle_hash=bundle_hash,
        reproducibility_hash=reproducibility_hash,
        signing_key_id=signing_key_id,
        fail_on=fail_on,
    )
    payload = {
        "manifest_version": DEPLOYMENT_GATE_MANIFEST_VERSION,
        "promptabi_version": __version__,
        "source_config": source_config,
        "gate": report.gate.value,
        "ok": not blockers,
        "bundle_hash": bundle_hash,
        "reproducibility_hash": reproducibility_hash,
        "signing_key_id": signing_key_id,
        "diagnostic_counts": dict(sorted(report.diagnostic_counts.items())),
        "safe_deployment_cores": [certificate.to_dict() for certificate in safe_core_report.certificates],
        "blockers": blockers,
        "examples": [example.to_dict() for example in examples],
    }
    return DeploymentGateReport(
        source_config=source_config,
        gate=report.gate,
        ok=not blockers,
        bundle_hash=bundle_hash,
        reproducibility_hash=reproducibility_hash,
        signing_key_id=signing_key_id,
        diagnostic_counts=report.diagnostic_counts,
        safe_deployment_cores=safe_core_report.certificates,
        examples=examples,
        blockers=blockers,
        manifest_sha256=_stable_json_hash(payload),
    )


def render_deployment_gate_json(report: DeploymentGateReport) -> str:
    """Render deployment gates as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_deployment_gate_text(report: DeploymentGateReport) -> str:
    """Render a compact deploy-gate summary for release engineers."""

    lines = [
        "PromptABI deployment gates",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"config: {report.source_config}",
        f"gate: {report.gate.value}",
        f"bundle_hash: {report.bundle_hash}",
        f"reproducibility_hash: {report.reproducibility_hash}",
        f"safe_deployment_cores: {len(report.safe_deployment_cores)}",
        f"examples: {len(report.examples)}",
    ]
    for certificate in report.safe_deployment_cores:
        lines.append(f"- unsat-core {certificate.case_id}: {', '.join(certificate.core)}")
    for example in report.examples:
        lines.append(f"- {example.surface.value}: {example.path}")
        lines.append(f"  command: {example.command}")
    if report.blockers:
        lines.append("blockers:")
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    return "\n".join(lines) + "\n"


def write_deployment_gate_examples(
    output_dir: str | Path,
    config: str | Path,
    *,
    bundle_key: str | bytes | None = None,
    bundle_key_id: str = "deployment-gate",
    fail_on: str = "error",
    workspace_root: str | Path | None = None,
    force: bool = False,
) -> DeploymentGateWriteResult:
    """Write deployment-gate examples and a deterministic manifest."""

    destination = Path(output_dir)
    report = build_deployment_gate_report(
        config,
        bundle_key=bundle_key,
        bundle_key_id=bundle_key_id,
        fail_on=fail_on,
        workspace_root=workspace_root,
    )
    filenames = tuple(["deployment-gates.json", *(example.path for example in report.examples)])
    _prepare_output_dir(destination, expected_filenames=filenames, force=force)
    written: list[Path] = []
    manifest_path = destination / "deployment-gates.json"
    manifest_path.write_text(render_deployment_gate_json(report), encoding="utf-8")
    written.append(manifest_path)
    for example in report.examples:
        path = destination / example.path
        path.write_text(example.content, encoding="utf-8")
        written.append(path)
    return DeploymentGateWriteResult(output_dir=destination, report=report, written_files=tuple(written))


def render_deployment_gate_write_summary(result: DeploymentGateWriteResult) -> str:
    return (
        "PromptABI deployment gates\n"
        f"output: {result.output_dir}\n"
        f"files: {len(result.written_files)}\n"
        f"bundle_hash: {result.report.bundle_hash}\n"
        f"reproducibility_hash: {result.report.reproducibility_hash}\n"
        f"manifest: {result.output_dir / 'deployment-gates.json'}\n"
    )


def _build_examples(
    *,
    source_config: str,
    bundle_hash: str,
    reproducibility_hash: str,
    signing_key_id: str,
    fail_on: str,
) -> tuple[DeploymentGateExample, ...]:
    verify_command = (
        "promptabi deployment-gates "
        f"--config {source_config} --bundle-key $PROMPTABI_BUNDLE_KEY --fail-on {fail_on} --format json"
    )
    return (
        DeploymentGateExample(
            surface=DeploymentGateSurface.KUBERNETES,
            title="Kubernetes admission gate",
            path="kubernetes.yaml",
            command=verify_command,
            policy=(
                "Reject deployments that omit the PromptABI bundle hash annotation.",
                "Require the annotation to match the currently verified bundle hash.",
            ),
            content=_kubernetes_yaml(bundle_hash, reproducibility_hash, signing_key_id),
        ),
        DeploymentGateExample(
            surface=DeploymentGateSurface.TERRAFORM,
            title="Terraform plan gate",
            path="terraform.hcl",
            command=verify_command,
            policy=(
                "Fail plan/apply when service metadata does not carry the current bundle hash.",
                "Store the reproducibility hash beside the release artifact for audit replay.",
            ),
            content=_terraform_hcl(bundle_hash, reproducibility_hash, signing_key_id),
        ),
        DeploymentGateExample(
            surface=DeploymentGateSurface.GITHUB_ENVIRONMENTS,
            title="GitHub Environments deployment gate",
            path="github-environments.yml",
            command=verify_command,
            policy=(
                "Require the deployment environment job to regenerate PromptABI gate evidence.",
                "Upload the gate manifest before the protected environment approval step.",
            ),
            content=_github_environment_yaml(source_config, bundle_hash, reproducibility_hash, signing_key_id, fail_on),
        ),
        DeploymentGateExample(
            surface=DeploymentGateSurface.INTERNAL_RELEASE_SYSTEM,
            title="Internal release-system policy",
            path="internal-release-system.json",
            command=verify_command,
            policy=(
                "Release coordinators compare the declared bundle hash with the signed PromptABI manifest.",
                "The release is blocked unless gate, bundle, and reproducibility hashes all match.",
            ),
            content=_internal_release_json(bundle_hash, reproducibility_hash, signing_key_id),
        ),
    )


def _kubernetes_yaml(bundle_hash: str, reproducibility_hash: str, signing_key_id: str) -> str:
    return f"""apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-current-promptabi-bundle
  annotations:
    promptabi.dev/bundle_hash: "{bundle_hash}"
    promptabi.dev/reproducibility_hash: "{reproducibility_hash}"
    promptabi.dev/signing_key_id: "{signing_key_id}"
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups: ["apps"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["deployments"]
  validations:
    - expression: "object.metadata.annotations['promptabi.dev/bundle_hash'] == '{bundle_hash}'"
      message: "Deployment must carry the current PromptABI signed verification bundle hash."
    - expression: "object.metadata.annotations['promptabi.dev/reproducibility_hash'] == '{reproducibility_hash}'"
      message: "Deployment must carry the current PromptABI reproducibility hash."
"""


def _terraform_hcl(bundle_hash: str, reproducibility_hash: str, signing_key_id: str) -> str:
    return f"""variable "promptabi_bundle_hash" {{
  type    = string
  default = "{bundle_hash}"
}}

variable "promptabi_reproducibility_hash" {{
  type    = string
  default = "{reproducibility_hash}"
}}

resource "terraform_data" "promptabi_deployment_gate" {{
  input = {{
    bundle_hash         = var.promptabi_bundle_hash
    reproducibility_hash = var.promptabi_reproducibility_hash
    signing_key_id      = "{signing_key_id}"
  }}

  lifecycle {{
    precondition {{
      condition     = var.promptabi_bundle_hash == "{bundle_hash}"
      error_message = "PromptABI bundle hash is stale; regenerate deployment gate evidence."
    }}
    precondition {{
      condition     = var.promptabi_reproducibility_hash == "{reproducibility_hash}"
      error_message = "PromptABI reproducibility hash is stale; rerun verification before deploy."
    }}
  }}
}}
"""


def _github_environment_yaml(
    source_config: str,
    bundle_hash: str,
    reproducibility_hash: str,
    signing_key_id: str,
    fail_on: str,
) -> str:
    return f"""name: promptabi-deployment-gate
on:
  workflow_dispatch:
  deployment:
jobs:
  promptabi-environment-gate:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: python -m pip install -e ".[dev]"
      - name: Regenerate current PromptABI deployment gate
        env:
          PROMPTABI_BUNDLE_KEY: ${{{{ secrets.PROMPTABI_BUNDLE_KEY }}}}
        run: |
          promptabi deployment-gates --config {source_config} --fail-on {fail_on} --format json > promptabi-deployment-gates.json
          test "$(jq -r '.bundle.bundle_hash' promptabi-deployment-gates.json)" = "{bundle_hash}"
          test "$(jq -r '.bundle.reproducibility_hash' promptabi-deployment-gates.json)" = "{reproducibility_hash}"
          test "$(jq -r '.bundle.signing_key_id' promptabi-deployment-gates.json)" = "{signing_key_id}"
      - uses: actions/upload-artifact@v4
        with:
          name: promptabi-deployment-gates
          path: promptabi-deployment-gates.json
"""


def _internal_release_json(bundle_hash: str, reproducibility_hash: str, signing_key_id: str) -> str:
    return json.dumps(
        {
            "release_system": "internal",
            "gate": "promptabi-current-verification-bundle",
            "required": {
                "bundle_hash": bundle_hash,
                "reproducibility_hash": reproducibility_hash,
                "signing_key_id": signing_key_id,
                "verification_gate": "pass",
            },
            "block_if": [
                "bundle_hash_missing",
                "bundle_hash_mismatch",
                "reproducibility_hash_mismatch",
                "promptabi_gate_not_pass",
            ],
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def _deployment_blockers(ok: bool, gate: IntegrationGate, diagnostic_counts: Mapping[str, int]) -> tuple[str, ...]:
    blockers: list[str] = []
    if not ok:
        blockers.append("PromptABI verification has error diagnostics")
    if gate is IntegrationGate.FAIL:
        blockers.append("integration gate is fail")
    if diagnostic_counts.get("error", 0):
        blockers.append(f"{diagnostic_counts['error']} error diagnostics must be resolved before deployment")
    return tuple(dict.fromkeys(blockers))


def _prepare_output_dir(destination: Path, *, expected_filenames: tuple[str, ...], force: bool) -> None:
    if destination.exists():
        existing = [destination / filename for filename in expected_filenames if (destination / filename).exists()]
        if existing and not force:
            names = ", ".join(path.name for path in existing)
            raise DeploymentGateError(f"deployment-gate files already exist ({names}); pass --force to overwrite")
    destination.mkdir(parents=True, exist_ok=True)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentGateError(f"{context} is missing from integration evidence")
    return value


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise DeploymentGateError(f"signed bundle evidence field {key!r} is required")
    return value


def _relative_to_cwd(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
