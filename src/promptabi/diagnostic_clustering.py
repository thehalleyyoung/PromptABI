"""Deterministic grouping of related PromptABI diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .diagnostics import ArtifactRef, Diagnostic, DiagnosticSeverity
from .fix_suggestions import RankedFixSuggestion, rank_fix_suggestions


class DiagnosticClusterStrategy(StrEnum):
    """Supported explanations for why diagnostics are related."""

    ROOT_CAUSE = "root-cause"
    ARTIFACT_EDGE = "artifact-edge"
    RULE = "rule"
    PROVIDER_BEHAVIOR = "provider-behavior"
    SHARED_WITNESS = "shared-witness"


DEFAULT_CLUSTER_STRATEGIES: tuple[DiagnosticClusterStrategy, ...] = tuple(DiagnosticClusterStrategy)


@dataclass(frozen=True, slots=True)
class DiagnosticClusterMember:
    """A diagnostic summarized inside a cluster."""

    fingerprint: str
    rule_id: str
    severity: DiagnosticSeverity
    message: str
    artifact: ArtifactRef | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.artifact is not None:
            payload["artifact"] = self.artifact.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class DiagnosticCluster:
    """A deterministic group of diagnostics that share a useful triage key."""

    cluster_id: str
    strategy: DiagnosticClusterStrategy
    title: str
    key: str
    members: tuple[DiagnosticClusterMember, ...]
    worst_severity: DiagnosticSeverity
    rules: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    ranked_suggestions: tuple[RankedFixSuggestion, ...] = ()
    evidence: tuple[str, ...] = ()
    root_cause: str | None = None
    artifact_edge: str | None = None
    provider_behavior: str | None = None
    shared_witness_digest: str | None = None

    @property
    def count(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cluster_id": self.cluster_id,
            "strategy": self.strategy.value,
            "title": self.title,
            "key": self.key,
            "count": self.count,
            "worst_severity": self.worst_severity.value,
            "rules": list(self.rules),
            "artifacts": list(self.artifacts),
            "suggestions": list(self.suggestions),
            "ranked_suggestions": [suggestion.to_dict() for suggestion in self.ranked_suggestions],
            "evidence": list(self.evidence),
            "members": [member.to_dict() for member in self.members],
        }
        if self.root_cause is not None:
            payload["root_cause"] = self.root_cause
        if self.artifact_edge is not None:
            payload["artifact_edge"] = self.artifact_edge
        if self.provider_behavior is not None:
            payload["provider_behavior"] = self.provider_behavior
        if self.shared_witness_digest is not None:
            payload["shared_witness_digest"] = self.shared_witness_digest
        return payload


@dataclass(frozen=True, slots=True)
class DiagnosticClusterReport:
    """A clustering report with coverage metadata for safe triage summaries."""

    clusters: tuple[DiagnosticCluster, ...]
    total_diagnostics: int
    clustered_fingerprints: tuple[str, ...] = field(default_factory=tuple)
    unclustered_fingerprints: tuple[str, ...] = field(default_factory=tuple)
    strategies: tuple[DiagnosticClusterStrategy, ...] = DEFAULT_CLUSTER_STRATEGIES
    min_cluster_size: int = 2

    @property
    def clustered_diagnostic_count(self) -> int:
        return len(self.clustered_fingerprints)

    @property
    def unclustered_diagnostic_count(self) -> int:
        return len(self.unclustered_fingerprints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "cluster_count": len(self.clusters),
            "clustered_diagnostic_count": self.clustered_diagnostic_count,
            "clustered_fingerprints": list(self.clustered_fingerprints),
            "min_cluster_size": self.min_cluster_size,
            "strategies": [strategy.value for strategy in self.strategies],
            "total_diagnostics": self.total_diagnostics,
            "unclustered_diagnostic_count": self.unclustered_diagnostic_count,
            "unclustered_fingerprints": list(self.unclustered_fingerprints),
        }


def build_diagnostic_clusters(
    diagnostics: Sequence[Diagnostic] | Iterable[Diagnostic],
    *,
    strategies: Sequence[DiagnosticClusterStrategy | str] = DEFAULT_CLUSTER_STRATEGIES,
    min_cluster_size: int = 2,
) -> DiagnosticClusterReport:
    """Group diagnostics by deterministic triage keys such as rule or witness."""

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    diagnostics_tuple = tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.sort_key))
    strategy_tuple = tuple(_coerce_strategy(strategy) for strategy in strategies)
    if not strategy_tuple:
        raise ValueError("at least one diagnostic clustering strategy is required")

    clusters: list[DiagnosticCluster] = []
    for strategy in strategy_tuple:
        buckets: dict[str, list[Diagnostic]] = defaultdict(list)
        evidence_by_key: dict[str, tuple[str, ...]] = {}
        for diagnostic in diagnostics_tuple:
            grouping = _grouping_key(diagnostic, strategy)
            if grouping is None:
                continue
            key, evidence = grouping
            buckets[key].append(diagnostic)
            evidence_by_key.setdefault(key, evidence)
        for key, grouped in sorted(buckets.items(), key=lambda item: (item[0], _fingerprints(item[1]))):
            unique_by_fingerprint = {diagnostic.fingerprint: diagnostic for diagnostic in grouped}
            unique_diagnostics = tuple(unique_by_fingerprint[fingerprint] for fingerprint in sorted(unique_by_fingerprint))
            if len(unique_diagnostics) < min_cluster_size:
                continue
            clusters.append(_build_cluster(strategy, key, unique_diagnostics, evidence_by_key[key]))

    clusters_tuple = tuple(sorted(clusters, key=_cluster_sort_key))
    clustered = tuple(
        sorted({member.fingerprint for cluster in clusters_tuple for member in cluster.members})
    )
    all_fingerprints = tuple(sorted(diagnostic.fingerprint for diagnostic in diagnostics_tuple))
    clustered_set = set(clustered)
    unclustered = tuple(fingerprint for fingerprint in all_fingerprints if fingerprint not in clustered_set)
    return DiagnosticClusterReport(
        clusters=clusters_tuple,
        total_diagnostics=len(diagnostics_tuple),
        clustered_fingerprints=clustered,
        unclustered_fingerprints=unclustered,
        strategies=strategy_tuple,
        min_cluster_size=min_cluster_size,
    )


def render_diagnostic_clusters_json(report: DiagnosticClusterReport) -> str:
    """Render a diagnostic clustering report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_diagnostic_clusters_text(report: DiagnosticClusterReport) -> str:
    """Render diagnostic clusters for terminal triage."""

    lines = [
        "PromptABI diagnostic clusters",
        f"diagnostics: {report.total_diagnostics}",
        f"clusters: {len(report.clusters)}",
        f"clustered diagnostics: {report.clustered_diagnostic_count}",
        f"unclustered diagnostics: {report.unclustered_diagnostic_count}",
    ]
    for cluster in report.clusters:
        lines.append(f"{cluster.strategy.value} {cluster.cluster_id}: {cluster.title}")
        lines.append(f"  severity: {cluster.worst_severity.value}")
        lines.append(f"  findings: {cluster.count}")
        if cluster.rules:
            lines.append(f"  rules: {', '.join(cluster.rules)}")
        if cluster.artifacts:
            lines.append(f"  artifacts: {', '.join(cluster.artifacts)}")
        for item in cluster.evidence:
            lines.append(f"  evidence: {item}")
        for suggestion in cluster.ranked_suggestions[:3]:
            lines.append(f"  suggestion[{suggestion.rank}]: {suggestion.text}")
            lines.append(
                "    rank: "
                f"score={suggestion.score}, safety={suggestion.safety.value}, "
                f"compatibility={suggestion.compatibility.value}, blast_radius={suggestion.blast_radius.value}, "
                f"user_visible_prompt_change={str(suggestion.changes_user_visible_prompt_behavior).lower()}"
            )
        for member in cluster.members:
            lines.append(f"    - {member.severity.value.upper()} {member.rule_id} {member.fingerprint}: {member.message}")
    return "\n".join(lines) + "\n"


def _build_cluster(
    strategy: DiagnosticClusterStrategy,
    key: str,
    diagnostics: tuple[Diagnostic, ...],
    evidence: tuple[str, ...],
) -> DiagnosticCluster:
    members = tuple(_member(diagnostic) for diagnostic in diagnostics)
    worst_severity = min((diagnostic.severity for diagnostic in diagnostics), key=lambda severity: severity.rank)
    rules = tuple(sorted({diagnostic.rule_id for diagnostic in diagnostics}))
    artifacts = tuple(sorted(_artifact_label(diagnostic.artifact) for diagnostic in diagnostics if diagnostic.artifact is not None))
    ranked_suggestions = rank_fix_suggestions(diagnostics)
    suggestions = tuple(suggestion.text for suggestion in ranked_suggestions)
    cluster_id = _stable_digest(
        {
            "key": key,
            "members": [diagnostic.fingerprint for diagnostic in diagnostics],
            "strategy": strategy.value,
        },
        length=16,
    )
    title = _cluster_title(strategy, key, diagnostics)
    kwargs: dict[str, str | None] = {
        "root_cause": key if strategy is DiagnosticClusterStrategy.ROOT_CAUSE else None,
        "artifact_edge": key if strategy is DiagnosticClusterStrategy.ARTIFACT_EDGE else None,
        "provider_behavior": key if strategy is DiagnosticClusterStrategy.PROVIDER_BEHAVIOR else None,
        "shared_witness_digest": key if strategy is DiagnosticClusterStrategy.SHARED_WITNESS else None,
    }
    return DiagnosticCluster(
        cluster_id=cluster_id,
        strategy=strategy,
        title=title,
        key=key,
        members=members,
        worst_severity=worst_severity,
        rules=rules,
        artifacts=artifacts,
        suggestions=suggestions,
        ranked_suggestions=ranked_suggestions,
        evidence=evidence,
        **kwargs,
    )


def _grouping_key(diagnostic: Diagnostic, strategy: DiagnosticClusterStrategy) -> tuple[str, tuple[str, ...]] | None:
    properties = dict(diagnostic.properties)
    if strategy is DiagnosticClusterStrategy.RULE:
        return diagnostic.rule_id, (f"rule_id={diagnostic.rule_id}",)
    if strategy is DiagnosticClusterStrategy.ROOT_CAUSE:
        explicit = _first_property(properties, "root_cause_id", "root_cause", "cause")
        if explicit is not None:
            return explicit, (f"declared root cause={explicit}",)
        normalized_message = _normalize_text(diagnostic.message)
        if not normalized_message:
            return None
        suggestion = _normalize_text(diagnostic.suggestions[0]) if diagnostic.suggestions else "no-suggestion"
        key = f"{diagnostic.rule_id}:{normalized_message}:{suggestion}"
        return key, ("message-and-suggestion fallback",)
    if strategy is DiagnosticClusterStrategy.ARTIFACT_EDGE:
        explicit = _first_property(properties, "artifact_edge", "edge", "source_target_edge")
        if explicit is not None:
            return explicit, (f"declared artifact edge={explicit}",)
        source = _first_property(properties, "source_artifact", "source")
        target = _first_property(properties, "target_artifact", "target")
        if source is not None or target is not None:
            key = f"{source or '?'}->{target or '?'}"
            return key, (f"source={source or '?'}", f"target={target or '?'}")
        if diagnostic.artifact is None:
            return None
        key = _artifact_label(diagnostic.artifact)
        return key, (f"artifact={key}",)
    if strategy is DiagnosticClusterStrategy.PROVIDER_BEHAVIOR:
        explicit = _first_property(properties, "provider_behavior", "provider_semantics", "behavior")
        provider = _first_property(properties, "provider_family", "provider", "provider_name")
        if explicit is not None or provider is not None:
            key = "::".join(item for item in (provider, explicit) if item is not None)
            return key, tuple(
                item for item in (f"provider={provider}" if provider else None, f"behavior={explicit}" if explicit else None) if item
            )
        if diagnostic.artifact is not None and "provider" in diagnostic.artifact.kind.lower():
            key = _artifact_label(diagnostic.artifact)
            return key, (f"provider artifact={key}",)
        return None
    if strategy is DiagnosticClusterStrategy.SHARED_WITNESS:
        if diagnostic.witness is None:
            return None
        payload: dict[str, Any] = {
            "minimal_fixes": diagnostic.witness.minimal_fixes,
            "parser_states": diagnostic.witness.parser_states,
            "rendered_strings": diagnostic.witness.rendered_strings,
            "role_regions": diagnostic.witness.role_regions,
            "solver_assignments": diagnostic.witness.solver_assignments,
            "token_ids": diagnostic.witness.token_ids,
            "truncation_decisions": diagnostic.witness.truncation_decisions,
        }
        if not any(payload.values()):
            last_output = next((step.output for step in reversed(diagnostic.witness.steps) if step.output is not None), None)
            payload = {
                "summary": diagnostic.witness.summary,
                "last_output": last_output,
            }
        digest = _stable_digest(payload)
        return digest, (f"witness_digest={digest}",)
    return None


def _member(diagnostic: Diagnostic) -> DiagnosticClusterMember:
    return DiagnosticClusterMember(
        fingerprint=diagnostic.fingerprint,
        rule_id=diagnostic.rule_id,
        severity=diagnostic.severity,
        message=diagnostic.message,
        artifact=diagnostic.artifact,
    )


def _cluster_title(strategy: DiagnosticClusterStrategy, key: str, diagnostics: tuple[Diagnostic, ...]) -> str:
    rule_count = len({diagnostic.rule_id for diagnostic in diagnostics})
    if strategy is DiagnosticClusterStrategy.RULE:
        return f"{len(diagnostics)} findings from rule {key}"
    if strategy is DiagnosticClusterStrategy.ARTIFACT_EDGE:
        return f"{len(diagnostics)} findings on artifact edge {key}"
    if strategy is DiagnosticClusterStrategy.PROVIDER_BEHAVIOR:
        return f"{len(diagnostics)} findings tied to provider behavior {key}"
    if strategy is DiagnosticClusterStrategy.SHARED_WITNESS:
        return f"{len(diagnostics)} findings share witness digest {key}"
    return f"{len(diagnostics)} findings share a likely root cause across {rule_count} rule(s)"


def _artifact_label(artifact: ArtifactRef) -> str:
    location = artifact.location_uri
    if location is not None:
        return f"{artifact.kind}:{artifact.name}@{location}"
    return f"{artifact.kind}:{artifact.name}"


def _first_property(properties: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = properties.get(name)
        if value is not None and str(value):
            return str(value)
    return None


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"\b[0-9a-f]{8,}\b", "<hash>", value.lower())
    normalized = re.sub(r"\d+", "<n>", normalized)
    normalized = re.sub(r"[^a-z0-9._:-]+", " ", normalized).strip()
    return " ".join(normalized.split())[:160]


def _stable_digest(payload: object, *, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _fingerprints(diagnostics: Sequence[Diagnostic]) -> tuple[str, ...]:
    return tuple(sorted(diagnostic.fingerprint for diagnostic in diagnostics))


def _cluster_sort_key(cluster: DiagnosticCluster) -> tuple[int, str, str]:
    return (cluster.worst_severity.rank, cluster.strategy.value, cluster.key)


def _coerce_strategy(strategy: DiagnosticClusterStrategy | str) -> DiagnosticClusterStrategy:
    if isinstance(strategy, DiagnosticClusterStrategy):
        return strategy
    try:
        return DiagnosticClusterStrategy(strategy)
    except ValueError as exc:
        choices = ", ".join(item.value for item in DiagnosticClusterStrategy)
        raise ValueError(f"unknown diagnostic clustering strategy: {strategy!r}; expected one of {choices}") from exc
