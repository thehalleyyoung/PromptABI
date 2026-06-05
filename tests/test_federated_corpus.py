"""Tests for federated corpus validation manifests (step 278)."""

from __future__ import annotations

import dataclasses

from promptabi.federated_corpus import (
    FederatedFindingKind,
    NodeManifest,
    aggregate_manifests,
    render_federated_text,
    sign_node_manifest,
)

KEY = b"federation-key"
ROSTER = frozenset({"node-a", "node-b"})


def _signed(node: str, count: int, passed: bool = True):
    return sign_node_manifest(NodeManifest(node, count, passed), KEY)


def test_valid_federation() -> None:
    signed = (_signed("node-a", 100), _signed("node-b", 200))
    result = aggregate_manifests(signed, KEY, ROSTER)
    assert result.valid
    assert result.total_examples == 300
    assert result.node_count == 2


def test_bad_signature_detected() -> None:
    signed = list((_signed("node-a", 100), _signed("node-b", 200)))
    signed[0] = dataclasses.replace(signed[0], signature="forged")
    result = aggregate_manifests(tuple(signed), KEY, ROSTER)
    assert any(f.kind is FederatedFindingKind.BAD_SIGNATURE for f in result.findings)


def test_node_contract_failed() -> None:
    signed = (_signed("node-a", 100, passed=False), _signed("node-b", 200))
    result = aggregate_manifests(signed, KEY, ROSTER)
    assert any(f.kind is FederatedFindingKind.CONTRACT_FAILED for f in result.findings)


def test_missing_node() -> None:
    signed = (_signed("node-a", 100),)
    result = aggregate_manifests(signed, KEY, ROSTER)
    assert any(f.kind is FederatedFindingKind.MISSING_NODE for f in result.findings)


def test_unexpected_node() -> None:
    signed = (_signed("node-a", 100), _signed("node-b", 200), _signed("node-x", 1))
    result = aggregate_manifests(signed, KEY, ROSTER)
    assert any(f.kind is FederatedFindingKind.UNEXPECTED_NODE for f in result.findings)


def test_render_text_smoke() -> None:
    signed = (_signed("node-a", 100), _signed("node-b", 200))
    assert "federated" in render_federated_text(aggregate_manifests(signed, KEY, ROSTER))
