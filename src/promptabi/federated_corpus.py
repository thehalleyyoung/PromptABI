"""Federated corpus validation manifests (step 278).

In a federated setting, each node validates its *local* shard of a training
corpus and emits a signed validation manifest: the node id, the number of
examples, the contract-pass status, and an HMAC over the canonical manifest.  A
coordinator aggregates these manifests and proves the federation is sound: every
node's signature verifies, no node reports a failed contract, and there are no
duplicate or missing nodes relative to the expected roster.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import StrEnum

FEDERATED_CORPUS_VERSION = "promptabi.federated-corpus.v1"


class FederatedFindingKind(StrEnum):
    BAD_SIGNATURE = "bad-signature"
    CONTRACT_FAILED = "node-contract-failed"
    DUPLICATE_NODE = "duplicate-node"
    MISSING_NODE = "missing-node"
    UNEXPECTED_NODE = "unexpected-node"


@dataclass(frozen=True, slots=True)
class NodeManifest:
    node_id: str
    example_count: int
    contract_passed: bool

    def canonical_bytes(self) -> bytes:
        payload = {
            "version": FEDERATED_CORPUS_VERSION,
            "node_id": self.node_id,
            "example_count": self.example_count,
            "contract_passed": self.contract_passed,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SignedNodeManifest:
    manifest: NodeManifest
    signature: str


def sign_node_manifest(manifest: NodeManifest, key: bytes) -> SignedNodeManifest:
    sig = hmac.new(key, manifest.canonical_bytes(), hashlib.sha256).hexdigest()
    return SignedNodeManifest(manifest=manifest, signature=sig)


@dataclass(frozen=True, slots=True)
class FederatedFinding:
    kind: FederatedFindingKind
    node_id: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "node_id": self.node_id, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FederatedAggregate:
    version: str
    valid: bool
    total_examples: int
    node_count: int
    findings: tuple[FederatedFinding, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "valid": self.valid,
            "total_examples": self.total_examples,
            "node_count": self.node_count,
            "findings": [f.to_dict() for f in self.findings],
        }


def aggregate_manifests(
    signed: tuple[SignedNodeManifest, ...],
    key: bytes,
    expected_roster: frozenset[str],
) -> FederatedAggregate:
    findings: list[FederatedFinding] = []
    seen: set[str] = set()
    total = 0

    for sm in signed:
        node = sm.manifest.node_id
        expected_sig = hmac.new(
            key, sm.manifest.canonical_bytes(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, sm.signature):
            findings.append(
                FederatedFinding(
                    FederatedFindingKind.BAD_SIGNATURE, node, "signature mismatch"
                )
            )
        if node in seen:
            findings.append(
                FederatedFinding(
                    FederatedFindingKind.DUPLICATE_NODE, node, "node reported twice"
                )
            )
        seen.add(node)
        if node not in expected_roster:
            findings.append(
                FederatedFinding(
                    FederatedFindingKind.UNEXPECTED_NODE,
                    node,
                    "node is not in the expected roster",
                )
            )
        if not sm.manifest.contract_passed:
            findings.append(
                FederatedFinding(
                    FederatedFindingKind.CONTRACT_FAILED,
                    node,
                    "node reported a failed training contract",
                )
            )
        total += sm.manifest.example_count

    for missing in sorted(expected_roster - seen):
        findings.append(
            FederatedFinding(
                FederatedFindingKind.MISSING_NODE,
                missing,
                "expected node did not report a manifest",
            )
        )

    return FederatedAggregate(
        version=FEDERATED_CORPUS_VERSION,
        valid=not findings,
        total_examples=total,
        node_count=len(seen),
        findings=tuple(findings),
    )


def render_federated_text(result: FederatedAggregate) -> str:
    lines = [
        f"PromptABI federated corpus validation ({result.version})",
        f"result: {'VALID' if result.valid else 'INVALID'} "
        f"({result.node_count} nodes, {result.total_examples} examples)",
    ]
    for f in result.findings:
        lines.append(f"  ! {f.kind.value} [{f.node_id}]: {f.detail}")
    return "\n".join(lines) + "\n"
