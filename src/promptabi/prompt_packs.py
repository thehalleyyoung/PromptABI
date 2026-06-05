"""Prompt-pack parsing and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .artifacts import (
    Artifact,
    ChatTemplateArtifact,
    PromptPackArtifact,
    PromptPackStopPolicy,
    PromptPackTemplate,
    PromptPackToolSchema,
    PromptSegmentArtifact,
    ProviderConfigArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
)
from .diagnostics import SourceSpan


class PromptPackFindingKind(StrEnum):
    """Prompt-pack compatibility outcomes."""

    EMPTY_PACK = "empty-pack"
    TEMPLATE_ROLE_MISMATCH = "template-role-mismatch"
    APP_ROLE_MISSING = "app-role-missing"
    TOOL_MISSING = "tool-missing"
    STOP_MISSING = "stop-missing"
    MODEL_FAMILY_UNSUPPORTED = "model-family-unsupported"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class PromptPackFinding:
    """One prompt-pack finding against a downstream app contract."""

    kind: PromptPackFindingKind
    pack: PromptPackArtifact
    message: str
    template: PromptPackTemplate | None = None
    tool: PromptPackToolSchema | None = None
    stop_policy: PromptPackStopPolicy | None = None
    expected: tuple[str, ...] = ()
    observed: tuple[str, ...] = ()
    span: SourceSpan | None = None


@dataclass(frozen=True, slots=True)
class PromptPackReport:
    """Compatibility report for reusable prompt-pack artifacts."""

    findings: tuple[PromptPackFinding, ...]


def analyze_prompt_pack_contracts(
    prompt_pack: PromptPackArtifact,
    artifacts: tuple[Artifact, ...],
    *,
    source_spans: dict[str, SourceSpan] | None = None,
) -> PromptPackReport:
    """Check a reusable prompt pack against its own and app-level ABI promises."""

    source_spans = source_spans or {}
    findings: list[PromptPackFinding] = []
    if not prompt_pack.exported_templates:
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.EMPTY_PACK,
                pack=prompt_pack,
                message=f"prompt pack '{prompt_pack.pack_name}' exports no templates",
                span=source_spans.get("exported_templates"),
            )
        )
        return PromptPackReport(tuple(findings))

    expected_roles = set(prompt_pack.expected_roles)
    for template in prompt_pack.exported_templates:
        if expected_roles and not set(template.roles).issubset(expected_roles):
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.TEMPLATE_ROLE_MISMATCH,
                    pack=prompt_pack,
                    template=template,
                    message=f"template '{template.name}' exports roles outside the pack expected role set",
                    expected=tuple(sorted(expected_roles)),
                    observed=template.roles,
                    span=source_spans.get(f"exported_templates.{template.name}.roles"),
                )
            )

    app_roles = _application_roles(artifacts)
    if app_roles and expected_roles:
        missing_roles = tuple(sorted(expected_roles - app_roles))
        if missing_roles:
            findings.append(
                PromptPackFinding(
                    kind=PromptPackFindingKind.APP_ROLE_MISSING,
                    pack=prompt_pack,
                    message=f"downstream app is missing prompt-pack role(s): {', '.join(missing_roles)}",
                    expected=tuple(sorted(expected_roles)),
                    observed=tuple(sorted(app_roles)),
                    span=source_spans.get("expected_roles"),
                )
            )

    app_tools = _application_tools(artifacts)
    if prompt_pack.tool_schemas:
        for tool in prompt_pack.tool_schemas:
            if tool.required and app_tools and tool.name not in app_tools:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.TOOL_MISSING,
                        pack=prompt_pack,
                        tool=tool,
                        message=f"downstream app does not provide required prompt-pack tool '{tool.name}'",
                        expected=(tool.name,),
                        observed=tuple(sorted(app_tools)),
                        span=source_spans.get(f"tool_schemas.{tool.name}"),
                    )
                )
            elif tool.required and not app_tools:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.TOOL_MISSING,
                        pack=prompt_pack,
                        tool=tool,
                        message=f"prompt pack requires tool '{tool.name}' but no downstream tool-definition artifact is configured",
                        expected=(tool.name,),
                        span=source_spans.get(f"tool_schemas.{tool.name}"),
                    )
                )

    app_stops = _application_stop_sequences(artifacts)
    if prompt_pack.stop_policies:
        for stop_policy in prompt_pack.stop_policies:
            missing_stops = tuple(sequence for sequence in stop_policy.stop_sequences if sequence not in app_stops)
            if missing_stops and app_stops:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.STOP_MISSING,
                        pack=prompt_pack,
                        stop_policy=stop_policy,
                        message=f"downstream stop policy omits prompt-pack stop sequence(s): {', '.join(missing_stops)}",
                        expected=stop_policy.stop_sequences,
                        observed=tuple(sorted(app_stops)),
                        span=source_spans.get(f"stop_policies.{stop_policy.name}"),
                    )
                )
            elif stop_policy.stop_sequences and not app_stops:
                findings.append(
                    PromptPackFinding(
                        kind=PromptPackFindingKind.STOP_MISSING,
                        pack=prompt_pack,
                        stop_policy=stop_policy,
                        message=f"prompt pack declares stop policy '{stop_policy.name}' but no downstream stop-policy artifact is configured",
                        expected=stop_policy.stop_sequences,
                        span=source_spans.get(f"stop_policies.{stop_policy.name}"),
                    )
                )

    app_model_families = _application_model_families(artifacts)
    supported_families = set(prompt_pack.supported_model_families)
    if app_model_families and supported_families and app_model_families.isdisjoint(supported_families):
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.MODEL_FAMILY_UNSUPPORTED,
                pack=prompt_pack,
                message=f"downstream model/provider family is outside prompt-pack support: {', '.join(sorted(app_model_families))}",
                expected=tuple(sorted(supported_families)),
                observed=tuple(sorted(app_model_families)),
                span=source_spans.get("supported_model_families"),
            )
        )

    if not findings:
        findings.append(
            PromptPackFinding(
                kind=PromptPackFindingKind.VERIFIED,
                pack=prompt_pack,
                message=(
                    f"prompt pack '{prompt_pack.pack_name}' exports {len(prompt_pack.exported_templates)} "
                    "template contract(s) compatible with configured app artifacts"
                ),
                expected=tuple(sorted(expected_roles)),
                observed=tuple(sorted(app_roles)),
            )
        )
    return PromptPackReport(tuple(findings))


def _application_roles(artifacts: tuple[Artifact, ...]) -> set[str]:
    roles: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ChatTemplateArtifact):
            roles.update(artifact.roles)
        elif isinstance(artifact, PromptSegmentArtifact):
            roles.update(segment.role for segment in artifact.segments if segment.role is not None)
    return roles


def _application_tools(artifacts: tuple[Artifact, ...]) -> set[str]:
    tools: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ToolDefinitionArtifact):
            tools.update(artifact.tool_names)
    return tools


def _application_stop_sequences(artifacts: tuple[Artifact, ...]) -> set[str]:
    stops: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, StopPolicyArtifact):
            stops.update(artifact.stop_sequences)
    return stops


def _application_model_families(artifacts: tuple[Artifact, ...]) -> set[str]:
    families: set[str] = set()
    for artifact in artifacts:
        if isinstance(artifact, ProviderConfigArtifact):
            families.add(artifact.provider)
            if artifact.api_family is not None:
                families.add(artifact.api_family)
        elif isinstance(artifact, TokenizerArtifact) and artifact.family is not None:
            families.add(artifact.family)
    return families
