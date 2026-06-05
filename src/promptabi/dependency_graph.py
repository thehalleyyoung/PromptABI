"""Artifact-to-check dependency graph visualization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from .artifacts import Artifact, ArtifactKind
from .config import VerificationConfig
from .plugins import PluginRegistry
from .session import VerificationSession


@dataclass(frozen=True, slots=True)
class DependencyGraphNode:
    """A node in the PromptABI artifact/check dependency graph."""

    id: str
    label: str
    kind: str
    configured: bool = True
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata, key=lambda item: item[0])))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "configured": self.configured,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True, slots=True)
class DependencyGraphEdge:
    """A directed dependency edge."""

    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class DependencyGraphReport:
    """Deterministic dependency graph for a PromptABI config."""

    config_name: str
    checks: tuple[str, ...]
    nodes: tuple[DependencyGraphNode, ...] = field(default_factory=tuple)
    edges: tuple[DependencyGraphEdge, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_name": self.config_name,
            "checks": list(self.checks),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def build_dependency_graph(
    config: VerificationConfig,
    *,
    plugin_registry: PluginRegistry | None = None,
    include_all_checks: bool = False,
) -> DependencyGraphReport:
    """Build the artifact/check dependency graph from real scheduler metadata."""

    session = VerificationSession(config, plugin_registry=plugin_registry)
    checks = tuple(sorted(session.checks)) if include_all_checks else tuple(config.checks)
    artifacts_by_kind: dict[ArtifactKind, list[Artifact]] = {}
    for artifact in config.artifact_bundle:
        artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)

    nodes: dict[str, DependencyGraphNode] = {}
    edges: set[DependencyGraphEdge] = set()

    for artifact in sorted(config.artifact_bundle, key=lambda item: (item.kind.value, item.name)):
        node = _artifact_node(artifact)
        nodes[node.id] = node

    for check_name in checks:
        check_node = DependencyGraphNode(
            id=_check_id(check_name),
            label=check_name,
            kind="check",
            metadata=(("source", "registered" if check_name in session.checks else "unknown"),),
        )
        nodes[check_node.id] = check_node
        dependency = session.check_dependencies.get(check_name)
        if dependency is None:
            continue
        for artifact_kind in dependency.artifact_kinds:
            configured_artifacts = artifacts_by_kind.get(artifact_kind, [])
            if configured_artifacts:
                for artifact in sorted(configured_artifacts, key=lambda item: item.name):
                    edges.add(
                        DependencyGraphEdge(
                            source=_artifact_id(artifact),
                            target=check_node.id,
                            relation="feeds",
                        )
                    )
            else:
                kind_node = _artifact_kind_node(artifact_kind)
                nodes.setdefault(kind_node.id, kind_node)
                edges.add(
                    DependencyGraphEdge(
                        source=kind_node.id,
                        target=check_node.id,
                        relation="requires-kind",
                    )
                )
        for prerequisite in dependency.after:
            prerequisite_node = DependencyGraphNode(
                id=_check_id(prerequisite),
                label=prerequisite,
                kind="check",
                configured=prerequisite in checks,
                metadata=(("source", "scheduler-prerequisite"),),
            )
            nodes.setdefault(prerequisite_node.id, prerequisite_node)
            edges.add(
                DependencyGraphEdge(
                    source=prerequisite_node.id,
                    target=check_node.id,
                    relation="runs-before",
                )
            )
        for resource in dependency.resources:
            resource_node = DependencyGraphNode(
                id=_resource_id(resource),
                label=resource,
                kind="analysis-resource",
            )
            nodes.setdefault(resource_node.id, resource_node)
            edges.add(
                DependencyGraphEdge(
                    source=resource_node.id,
                    target=check_node.id,
                    relation="uses",
                )
            )

    return DependencyGraphReport(
        config_name=config.name,
        checks=checks,
        nodes=tuple(sorted(nodes.values(), key=lambda node: (node.kind, node.id))),
        edges=tuple(sorted(edges, key=lambda edge: (edge.target, edge.relation, edge.source))),
    )


def render_dependency_graph_text(report: DependencyGraphReport) -> str:
    """Render a compact human-readable dependency graph."""

    lines = [
        f"PromptABI dependency graph: {report.config_name}",
        f"checks: {len(report.checks)}",
        "",
    ]
    for check in report.checks:
        check_id = _check_id(check)
        incoming = [edge for edge in report.edges if edge.target == check_id]
        lines.append(f"{check}:")
        if not incoming:
            lines.append("  inputs: none declared")
            continue
        for edge in incoming:
            source = _node_label(report.nodes, edge.source)
            lines.append(f"  {edge.relation}: {source}")
    return "\n".join(lines) + "\n"


def render_dependency_graph_json(report: DependencyGraphReport) -> str:
    """Render the graph as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_dependency_graph_mermaid(report: DependencyGraphReport) -> str:
    """Render a Mermaid flowchart suitable for GitHub Markdown."""

    lines = ["flowchart LR"]
    for node in report.nodes:
        shape = _mermaid_shape(_escape_mermaid(node.label), node.kind, node.configured)
        lines.append(f"  {_mermaid_id(node.id)}{shape}")
    for edge in report.edges:
        lines.append(
            f"  {_mermaid_id(edge.source)} -->|{_escape_mermaid(edge.relation)}| {_mermaid_id(edge.target)}"
        )
    return "\n".join(lines) + "\n"


def _artifact_node(artifact: Artifact) -> DependencyGraphNode:
    metadata: list[tuple[str, object]] = [
        ("artifact_kind", artifact.kind.value),
        ("location", artifact.location.ref_path or ""),
    ]
    if artifact.provenance.ref_version is not None:
        metadata.append(("version", artifact.provenance.ref_version))
    return DependencyGraphNode(
        id=_artifact_id(artifact),
        label=f"{artifact.name} ({artifact.kind.value})",
        kind="artifact",
        metadata=tuple(metadata),
    )


def _artifact_kind_node(kind: ArtifactKind) -> DependencyGraphNode:
    return DependencyGraphNode(
        id=f"kind:{kind.value}",
        label=f"{kind.value} artifacts",
        kind="artifact-kind",
        configured=False,
        metadata=(("artifact_kind", kind.value),),
    )


def _artifact_id(artifact: Artifact) -> str:
    return f"artifact:{artifact.name}"


def _check_id(check: str) -> str:
    return f"check:{check}"


def _resource_id(resource: str) -> str:
    return f"resource:{resource}"


def _node_label(nodes: Iterable[DependencyGraphNode], node_id: str) -> str:
    for node in nodes:
        if node.id == node_id:
            suffix = "" if node.configured else " (not configured)"
            return f"{node.label}{suffix}"
    return node_id


def _mermaid_id(value: str) -> str:
    return "n_" + "".join(character if character.isalnum() else "_" for character in value)


def _escape_mermaid(value: str) -> str:
    return value.replace('"', "'")


def _mermaid_shape(label: str, kind: str, configured: bool) -> str:
    if kind == "check":
        return f'["{label}"]'
    if kind == "analysis-resource":
        return f'{{"{label}"}}'
    if configured:
        return f'("{label}")'
    return f'["{label}<br/>not configured"]'
