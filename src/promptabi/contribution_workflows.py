"""Structured community contribution workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass


CONTRIBUTION_WORKFLOW_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContributionWorkflow:
    """One externally actionable contribution path."""

    id: str
    title: str
    issue_template: str
    label: str
    accepted_artifacts: tuple[str, ...]
    required_evidence: tuple[str, ...]
    validation_commands: tuple[str, ...]
    privacy_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "issue_template": self.issue_template,
            "label": self.label,
            "accepted_artifacts": list(self.accepted_artifacts),
            "required_evidence": list(self.required_evidence),
            "validation_commands": list(self.validation_commands),
            "privacy_checks": list(self.privacy_checks),
        }


def build_contribution_workflows() -> tuple[ContributionWorkflow, ...]:
    """Return the canonical community contribution lanes."""

    return (
        ContributionWorkflow(
            id="sanitized-bug-fixture",
            title="Sanitized bug fixture",
            issue_template="corpus_fixture.yml",
            label="area: corpus",
            accepted_artifacts=(
                "minimal PromptABI config or artifact bundle",
                "expected diagnostic IDs and fingerprints",
                "upstream bug, fixed version, or local workaround link when available",
            ),
            required_evidence=(
                "provenance, license, and exact revision/hash",
                "root-cause note explaining why the finding is structural",
                "fixture minimized enough to replay offline",
            ),
            validation_commands=(
                "promptabi corpus verify --format text",
                "promptabi contribute validate --format text",
            ),
            privacy_checks=(
                "no secrets, private prompts, customer data, or live credentials",
                "bounded excerpts only when raw payloads are necessary for reproduction",
            ),
        ),
        ContributionWorkflow(
            id="minimized-witness",
            title="Minimized witness",
            issue_template="witness_minimization.yml",
            label="area: witness",
            accepted_artifacts=(
                "minimizer case JSON",
                "preserved oracle substring, rule ID, or diagnostic fingerprint",
                "shrunk witness trace with stable source spans or hash-only payloads",
            ),
            required_evidence=(
                "before/after size and accepted shrink steps",
                "oracle command proving the reduced case still fails",
                "sanitization note for any witness payloads",
            ),
            validation_commands=(
                "promptabi minimize repro.json --keep-substring '<structural-boundary>' --format json",
                "promptabi contribute validate --format text",
            ),
            privacy_checks=(
                "prefer hash-only witness privacy for private prompts",
                "remove credentials, tokens, account IDs, and customer-specific strings",
            ),
        ),
        ContributionWorkflow(
            id="prompt-pack-metadata",
            title="Prompt-pack metadata",
            issue_template="prompt_pack_metadata.yml",
            label="area: prompt-pack",
            accepted_artifacts=(
                "prompt-pack JSON",
                "lockfile, registry entry, provenance envelope, or upgrade-gate report",
                "model-family, role, tool, stop, RAG, and truncation compatibility notes",
            ),
            required_evidence=(
                "package provenance and maintainer/review status",
                "lockfile drift or upgrade impact report",
                "prompt-pack registry metadata with privacy behavior",
            ),
            validation_commands=(
                "promptabi prompt-pack lock --config examples/prompt-packs/promptabi.json --write",
                "promptabi prompt-pack registry --config examples/prompt-packs/promptabi.json",
                "promptabi prompt-pack upgrade --config examples/prompt-packs/promptabi.json --baseline-lockfile prompt-pack.lock.json",
            ),
            privacy_checks=(
                "do not submit private prompt text unless intentionally public",
                "include hash-only or bounded excerpts for sensitive prompt libraries",
            ),
        ),
        ContributionWorkflow(
            id="provider-fixture",
            title="Provider fixture",
            issue_template="provider_fixture.yml",
            label="area: provider",
            accepted_artifacts=(
                "offline request/response shape",
                "tool-call streaming, parallel-call, JSON-mode, stop, or error fixture",
                "provider version, adapter, and context-window metadata",
            ),
            required_evidence=(
                "recording provenance and exact provider/API version",
                "expected conformance status and diagnostic fingerprints",
                "explanation of compatibility impact across adapters",
            ),
            validation_commands=(
                "promptabi corpus provider-conformance --format text",
                "promptabi contribute validate --format text",
            ),
            privacy_checks=(
                "fixtures must be recorded offline and non-sensitive",
                "strip API keys, organization IDs, account IDs, and raw user prompts",
            ),
        ),
        ContributionWorkflow(
            id="training-manifest-adapter",
            title="Training-manifest adapter",
            issue_template="training_adapter.yml",
            label="area: training",
            accepted_artifacts=(
                "training manifest or loader adapter fixture",
                "dataset schema, redaction policy, loss-mask, packing, and tokenizer pins",
                "streaming-shard or sidecar proof summary when private corpora are involved",
            ),
            required_evidence=(
                "adapter format and library/version compatibility",
                "invalid-role/tool/JSON/stop examples covered by the manifest",
                "hashes or synthetic rows proving deterministic replay",
            ),
            validation_commands=(
                "promptabi verify-training --manifest examples/end-to-end/training-quickstart/fixed.training-manifest.json",
                "promptabi contribute validate --format text",
            ),
            privacy_checks=(
                "do not submit raw private training rows",
                "use hash-only fingerprints for shard witnesses and private source spans",
            ),
        ),
    )


def render_contribution_workflows_json(workflows: tuple[ContributionWorkflow, ...] | None = None) -> str:
    """Render contribution workflows as stable JSON."""

    workflows = build_contribution_workflows() if workflows is None else workflows
    payload = {
        "manifest_version": CONTRIBUTION_WORKFLOW_VERSION,
        "workflow_count": len(workflows),
        "workflows": [workflow.to_dict() for workflow in workflows],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_contribution_workflows_text(workflows: tuple[ContributionWorkflow, ...] | None = None) -> str:
    """Render contribution workflows for contributors and CI logs."""

    workflows = build_contribution_workflows() if workflows is None else workflows
    lines = [
        f"PromptABI contribution workflows v{CONTRIBUTION_WORKFLOW_VERSION}",
        f"workflows: {len(workflows)}",
    ]
    for workflow in workflows:
        lines.extend(
            (
                "",
                f"{workflow.id}: {workflow.title}",
                f"  template: .github/ISSUE_TEMPLATE/{workflow.issue_template}",
                f"  label: {workflow.label}",
                f"  accepts: {'; '.join(workflow.accepted_artifacts)}",
                f"  evidence: {'; '.join(workflow.required_evidence)}",
                f"  validate: {' && '.join(workflow.validation_commands)}",
                f"  privacy: {'; '.join(workflow.privacy_checks)}",
            )
        )
    return "\n".join(lines) + "\n"
