"""Layered composition for PromptABI static contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .artifacts import (
    ArtifactKind,
    ArtifactLocation,
    StaticContractArtifact,
    StaticContractInvariant,
    StaticContractRule,
    StaticContractSchemaObligation,
    StaticContractStopPolicy,
)
from .contract_language import CONTRACT_LANGUAGE_VERSION
from .diagnostics import SourceSpan


class ContractLayer(str, Enum):
    """Explicit precedence layers for composable static contracts."""

    ORGANIZATION_POLICY = "organization-policy"
    PROMPT_PACK = "prompt-pack"
    APP_CONFIG = "app-config"
    TRAINING_MANIFEST = "training-manifest"


_LAYER_RANK = {
    ContractLayer.ORGANIZATION_POLICY: 0,
    ContractLayer.PROMPT_PACK: 1,
    ContractLayer.APP_CONFIG: 2,
    ContractLayer.TRAINING_MANIFEST: 3,
}
_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


@dataclass(frozen=True, slots=True)
class ContractRuleContribution:
    """One rule contributed by a known contract layer."""

    layer: ContractLayer
    contract_name: str
    rule: StaticContractRule
    source_span: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class ContractCompositionConflict:
    """A deterministic conflict detected while composing layered rules."""

    rule_name: str
    field: str
    message: str
    left_layer: ContractLayer
    left_contract: str
    right_layer: ContractLayer
    right_contract: str
    left_span: SourceSpan | None = None
    right_span: SourceSpan | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rule_name": self.rule_name,
            "field": self.field,
            "message": self.message,
            "left_layer": self.left_layer.value,
            "left_contract": self.left_contract,
            "right_layer": self.right_layer.value,
            "right_contract": self.right_contract,
        }
        if self.left_span is not None:
            data["left_span"] = _span_to_dict(self.left_span)
        if self.right_span is not None:
            data["right_span"] = _span_to_dict(self.right_span)
        return data


@dataclass(frozen=True, slots=True)
class ContractCompositionResult:
    """The composed contract plus any non-fatal ambiguity diagnostics."""

    artifact: StaticContractArtifact
    conflicts: tuple[ContractCompositionConflict, ...]
    contributions: tuple[ContractRuleContribution, ...]

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "contract": self.artifact.to_dict(),
            "contributions": [
                {
                    "layer": contribution.layer.value,
                    "contract": contribution.contract_name,
                    "rule": contribution.rule.name,
                    **(
                        {"span": _span_to_dict(contribution.source_span)}
                        if contribution.source_span is not None
                        else {}
                    ),
                }
                for contribution in self.contributions
            ],
        }


def compose_static_contracts(
    layered_contracts: tuple[tuple[ContractLayer, StaticContractArtifact], ...],
    *,
    name: str = "composed-contract",
) -> ContractCompositionResult:
    """Compose static contracts under an explicit, deterministic layer order.

    Organization policies act as governance floors: later layers can specialize
    scopes, add requirements, and tighten guarantees, but cannot silently widen
    permissions or downgrade severity. Conflicting attempts are reported while a
    deterministic strongest contract is still produced for review.
    """

    contributions: list[ContractRuleContribution] = []
    for layer, contract in layered_contracts:
        spans = _rule_spans(contract)
        for rule in contract.rules:
            contributions.append(
                ContractRuleContribution(
                    layer=layer,
                    contract_name=contract.name,
                    rule=rule,
                    source_span=spans.get(rule.name),
                )
            )
    return compose_static_contract_contributions(tuple(contributions), name=name)


def compose_static_contract_contributions(
    contributions: tuple[ContractRuleContribution, ...],
    *,
    name: str = "composed-contract",
) -> ContractCompositionResult:
    """Compose already-extracted rule contributions."""

    grouped: dict[str, list[ContractRuleContribution]] = {}
    for contribution in contributions:
        grouped.setdefault(contribution.rule.name, []).append(contribution)

    conflicts: list[ContractCompositionConflict] = []
    rules: list[StaticContractRule] = []
    for rule_name in sorted(grouped):
        group = tuple(sorted(grouped[rule_name], key=_contribution_sort_key))
        rule, rule_conflicts = _compose_rule_group(group)
        rules.append(rule)
        conflicts.extend(rule_conflicts)

    artifact = StaticContractArtifact(
        kind=ArtifactKind.STATIC_CONTRACT,
        name=name,
        location=ArtifactLocation(uri=f"memory://{name}"),
        contract_version=CONTRACT_LANGUAGE_VERSION,
        rules=tuple(rules),
        metadata=(
            ("composition", "layered-static-contracts"),
            ("layer_precedence", tuple(layer.value for layer in sorted(_LAYER_RANK, key=_LAYER_RANK.__getitem__))),
            ("source_contracts", tuple(sorted({item.contract_name for item in contributions}))),
            ("conflict_count", len(conflicts)),
        ),
    )
    return ContractCompositionResult(
        artifact=artifact,
        conflicts=tuple(sorted(conflicts, key=lambda item: (item.rule_name, item.field, item.message))),
        contributions=tuple(sorted(contributions, key=_contribution_sort_key)),
    )


def contract_layer_from_string(value: str) -> ContractLayer:
    """Parse a CLI/user layer name."""

    normalized = value.strip().replace("_", "-").lower()
    for layer in ContractLayer:
        if layer.value == normalized:
            return layer
    allowed = ", ".join(layer.value for layer in ContractLayer)
    raise ValueError(f"unknown contract layer {value!r}; expected one of: {allowed}")


def render_contract_composition_json(result: ContractCompositionResult) -> str:
    """Render a composed contract and conflicts as deterministic JSON."""

    return json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"


def render_contract_composition_text(result: ContractCompositionResult) -> str:
    """Render a concise human-readable composition report."""

    lines = [
        f"PromptABI contract composition: {result.artifact.name}",
        f"rules: {len(result.artifact.rules)}",
        f"conflicts: {len(result.conflicts)}",
    ]
    for rule in result.artifact.rules:
        lines.append(f"  {rule.name}: severity={rule.severity} type={rule.rule_type}")
    if result.conflicts:
        lines.append("conflict details:")
        for conflict in result.conflicts:
            location = ""
            if conflict.left_span is not None or conflict.right_span is not None:
                left = _format_span(conflict.left_span)
                right = _format_span(conflict.right_span)
                location = f" ({left} vs {right})"
            lines.append(f"  {conflict.rule_name}.{conflict.field}: {conflict.message}{location}")
    return "\n".join(lines) + "\n"


def contract_contributions_from_files(specs: tuple[str, ...]) -> tuple[tuple[ContractLayer, Path], ...]:
    """Parse CLI specs of the form ``layer=path`` without reading files."""

    parsed: list[tuple[ContractLayer, Path]] = []
    for spec in specs:
        layer_name, separator, raw_path = spec.partition("=")
        if not separator or not layer_name or not raw_path:
            raise ValueError("contract inputs must use LAYER=PATH, for example organization-policy=policy.pabi")
        parsed.append((contract_layer_from_string(layer_name), Path(raw_path)))
    return tuple(parsed)


def _compose_rule_group(
    group: tuple[ContractRuleContribution, ...],
) -> tuple[StaticContractRule, tuple[ContractCompositionConflict, ...]]:
    base = group[0].rule
    conflicts: list[ContractCompositionConflict] = []
    for earlier_index, earlier in enumerate(group):
        for later in group[earlier_index + 1 :]:
            if earlier.layer == later.layer and earlier.rule != later.rule:
                conflicts.append(
                    _conflict(
                        earlier,
                        later,
                        "same-layer",
                        "same precedence layer contributes divergent rule bodies",
                    )
                )

    severity = max((item.rule.severity for item in group), key=_SEVERITY_RANK.__getitem__)
    for earlier in group:
        for later in group:
            if _LAYER_RANK[later.layer] <= _LAYER_RANK[earlier.layer]:
                continue
            if _SEVERITY_RANK[later.rule.severity] < _SEVERITY_RANK[earlier.rule.severity]:
                conflicts.append(
                    _conflict(
                        earlier,
                        later,
                        "severity",
                        f"{later.layer.value} attempts to weaken severity to {later.rule.severity!r}; kept {severity!r}",
                    )
                )

    rule_type_owner = _highest_specific(lambda rule: rule.rule_type, group)
    description_owner = _highest_specific(lambda rule: rule.description, group)
    applies_to = _intersect_optional_sets(tuple(item.rule.applies_to for item in group))
    allowed_roles = _intersect_optional_sets(tuple(item.rule.allowed_roles for item in group))
    if any(item.rule.allowed_roles for item in group) and not allowed_roles:
        conflicts.append(
            _conflict(
                group[0],
                group[-1],
                "allowed_roles",
                "layered allowed_roles constraints have an empty intersection",
            )
        )
    required_regions = _union_sets(item.rule.required_regions for item in group)
    forbidden_delimiters = _union_sets(item.rule.forbidden_delimiters for item in group)
    schema_obligations = _merge_schema_obligations(group)
    stop_policies, stop_conflicts = _merge_stop_policies(group)
    invariants, invariant_conflicts = _merge_invariants(group)
    conflicts.extend(stop_conflicts)
    conflicts.extend(invariant_conflicts)

    return (
        StaticContractRule(
            name=base.name,
            rule_type=rule_type_owner.rule.rule_type,
            severity=severity,
            description=description_owner.rule.description,
            applies_to=applies_to,
            allowed_roles=allowed_roles,
            required_regions=required_regions,
            forbidden_delimiters=forbidden_delimiters,
            schema_obligations=schema_obligations,
            stop_policies=stop_policies,
            invariants=invariants,
        ),
        tuple(conflicts),
    )


def _merge_schema_obligations(group: tuple[ContractRuleContribution, ...]) -> tuple[StaticContractSchemaObligation, ...]:
    required_by_schema: dict[str, set[str]] = {}
    for contribution in group:
        for obligation in contribution.rule.schema_obligations:
            required_by_schema.setdefault(obligation.schema, set()).update(obligation.requires)
    return tuple(
        StaticContractSchemaObligation(schema=schema, requires=tuple(sorted(requires)))
        for schema, requires in sorted(required_by_schema.items())
    )


def _merge_stop_policies(
    group: tuple[ContractRuleContribution, ...],
) -> tuple[tuple[StaticContractStopPolicy, ...], tuple[ContractCompositionConflict, ...]]:
    stops_by_name: dict[str, set[str]] = {}
    forbid_by_name: dict[str, tuple[str, ContractRuleContribution]] = {}
    conflicts: list[ContractCompositionConflict] = []
    for contribution in group:
        for policy in contribution.rule.stop_policies:
            stops_by_name.setdefault(policy.name, set()).update(policy.stops)
            if policy.forbid_inside is None:
                continue
            current = forbid_by_name.get(policy.name)
            if current is not None and current[0] != policy.forbid_inside:
                conflicts.append(
                    _conflict(
                        current[1],
                        contribution,
                        f"stop_policies.{policy.name}.forbid_inside",
                        f"conflicting forbid_inside regions {current[0]!r} and {policy.forbid_inside!r}",
                    )
                )
                if _LAYER_RANK[contribution.layer] >= _LAYER_RANK[current[1].layer]:
                    forbid_by_name[policy.name] = (policy.forbid_inside, contribution)
            else:
                forbid_by_name[policy.name] = (policy.forbid_inside, contribution)
    policies = tuple(
        StaticContractStopPolicy(
            name=name,
            stops=tuple(sorted(stops)),
            forbid_inside=forbid_by_name.get(name, (None, group[0]))[0],
        )
        for name, stops in sorted(stops_by_name.items())
    )
    return policies, tuple(conflicts)


def _merge_invariants(
    group: tuple[ContractRuleContribution, ...],
) -> tuple[tuple[StaticContractInvariant, ...], tuple[ContractCompositionConflict, ...]]:
    invariant_by_name: dict[str, tuple[StaticContractInvariant, ContractRuleContribution]] = {}
    conflicts: list[ContractCompositionConflict] = []
    for contribution in group:
        for invariant in contribution.rule.invariants:
            current = invariant_by_name.get(invariant.name)
            if current is not None and current[0] != invariant:
                conflicts.append(
                    _conflict(
                        current[1],
                        contribution,
                        f"invariants.{invariant.name}",
                        "same invariant name has different finite expressions",
                    )
                )
                if _LAYER_RANK[contribution.layer] >= _LAYER_RANK[current[1].layer]:
                    invariant_by_name[invariant.name] = (invariant, contribution)
            else:
                invariant_by_name[invariant.name] = (invariant, contribution)
    return tuple(item[0] for _, item in sorted(invariant_by_name.items())), tuple(conflicts)


def _highest_specific(
    getter,
    group: tuple[ContractRuleContribution, ...],
) -> ContractRuleContribution:
    candidates = tuple(item for item in group if getter(item.rule) is not None)
    return max(candidates or group, key=_contribution_sort_key)


def _intersect_optional_sets(values: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    non_empty = [set(value) for value in values if value]
    if not non_empty:
        return ()
    current = non_empty[0]
    for value in non_empty[1:]:
        current &= value
    return tuple(sorted(current))


def _union_sets(values: object) -> tuple[str, ...]:
    merged: set[str] = set()
    for value in values:
        merged.update(value)
    return tuple(sorted(merged))


def _contribution_sort_key(contribution: ContractRuleContribution) -> tuple[int, str, str]:
    return (_LAYER_RANK[contribution.layer], contribution.contract_name, contribution.rule.name)


def _conflict(
    left: ContractRuleContribution,
    right: ContractRuleContribution,
    field: str,
    message: str,
) -> ContractCompositionConflict:
    return ContractCompositionConflict(
        rule_name=left.rule.name,
        field=field,
        message=message,
        left_layer=left.layer,
        left_contract=left.contract_name,
        right_layer=right.layer,
        right_contract=right.contract_name,
        left_span=left.source_span,
        right_span=right.source_span,
    )


def _rule_spans(contract: StaticContractArtifact) -> dict[str, SourceSpan]:
    line_by_rule: dict[str, int] = {}
    for key, value in contract.metadata:
        if key != "rule_source_lines" or not isinstance(value, tuple):
            continue
        for item in value:
            if not isinstance(item, str) or ":" not in item:
                continue
            rule_name, _, line = item.partition(":")
            if line.isdigit():
                line_by_rule[rule_name] = int(line)
    path = contract.location.path or contract.location.uri or f"{contract.name}.pabi"
    return {
        rule.name: SourceSpan(path=path, start_line=line_by_rule.get(rule.name, 1), start_column=1)
        for rule in contract.rules
    }


def _span_to_dict(span: SourceSpan) -> dict[str, object]:
    return {
        "path": span.path,
        "start_line": span.start_line,
        "start_column": span.start_column,
        **({"end_line": span.end_line} if span.end_line is not None else {}),
        **({"end_column": span.end_column} if span.end_column is not None else {}),
    }


def _format_span(span: SourceSpan | None) -> str:
    if span is None:
        return "unknown"
    path = span.path or "unknown"
    return f"{path}:{span.start_line}:{span.start_column}"
