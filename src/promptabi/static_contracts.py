"""Z3-backed finite static contracts over PromptABI artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import (
    ArtifactKind,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptSegmentArtifact,
    SpecialTokenMapArtifact,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TrainingTextSourceKind,
)
from .config import VerificationConfig
from .formal import (
    BoundedStringDomain,
    Contains,
    EnumDomain,
    Eq,
    FiniteContractProblem,
    Gt,
    InSet,
    IntRangeDomain,
    NamedConstraint,
    Ne,
    Not,
    Or,
    SolverResult,
    SolverStatus,
    Value,
    Var,
)
from .loaders import LoadedArtifact


@dataclass(frozen=True, slots=True)
class StaticContractFinding:
    """One finite SMT obligation derived from real artifact surfaces."""

    name: str
    status: SolverStatus
    result: SolverResult | None
    problem: FiniteContractProblem | None
    message: str
    suggestion: str
    severity: str
    evidence: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StaticContractReport:
    """All static-contract obligations for one verification run."""

    findings: tuple[StaticContractFinding, ...]

    @property
    def violations(self) -> tuple[StaticContractFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "error")


def analyze_static_contracts(
    config: VerificationConfig,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    *,
    prefer_z3: bool = True,
) -> StaticContractReport:
    """Lower cross-artifact finite obligations to the existing SMT core."""

    findings: list[StaticContractFinding] = []
    artifacts = tuple(loaded.artifact for loaded in loaded_artifacts)
    prompt_segments = tuple(
        artifact for artifact in artifacts
        if isinstance(artifact, PromptSegmentArtifact)
    )
    truncation_configs = tuple(
        artifact for artifact in artifacts
        if isinstance(artifact, FrameworkTruncationConfigArtifact)
    )
    stop_policies = tuple(artifact for artifact in artifacts if isinstance(artifact, StopPolicyArtifact))
    special_maps = tuple(artifact for artifact in artifacts if isinstance(artifact, SpecialTokenMapArtifact))
    tokenizers = tuple(artifact for artifact in artifacts if isinstance(artifact, TokenizerArtifact))
    tools = tuple(artifact for artifact in artifacts if isinstance(artifact, ToolDefinitionArtifact))
    templates = tuple(artifact for artifact in artifacts if isinstance(artifact, ChatTemplateArtifact))
    training_manifests = tuple(
        artifact for artifact in artifacts
        if isinstance(artifact, TrainingManifestArtifact)
    )

    findings.extend(_budget_obligation(config, prompt_segments, truncation_configs, prefer_z3=prefer_z3))
    findings.extend(
        _role_region_nonforgeability_obligation(
            prompt_segments,
            templates,
            special_maps,
            stop_policies,
            tokenizers,
            prefer_z3=prefer_z3,
        )
    )
    findings.extend(_stop_control_token_obligation(stop_policies, special_maps, tokenizers, prefer_z3=prefer_z3))
    findings.extend(_tool_provider_obligation(tools, loaded_artifacts, prefer_z3=prefer_z3))
    findings.extend(_tool_schema_precondition_obligation(loaded_artifacts, prefer_z3=prefer_z3))
    findings.extend(_training_target_obligation(training_manifests, templates, prefer_z3=prefer_z3))
    findings.extend(_training_span_region_obligation(training_manifests, templates, prefer_z3=prefer_z3))
    findings.extend(_training_source_leakage_obligation(training_manifests, prefer_z3=prefer_z3))

    if not findings:
        findings.append(
            StaticContractFinding(
                name="static-contract-abstained",
                status=SolverStatus.UNKNOWN,
                result=None,
                problem=None,
                severity="warning",
                message="no finite cross-artifact static contract could be derived",
                suggestion="Declare at least two related artifacts, such as prompt segments with a truncation config or stop policies with special tokens.",
                evidence=(("artifact_count", str(len(loaded_artifacts))),),
            )
        )
    return StaticContractReport(findings=tuple(findings))


def _budget_obligation(
    config: VerificationConfig,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    truncation_configs: tuple[FrameworkTruncationConfigArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    if not prompt_segments:
        return ()
    required = tuple(
        segment
        for artifact in prompt_segments
        for segment in artifact.segments
        if segment.required
    )
    if not required:
        return ()
    budget = truncation_configs[0] if truncation_configs else None
    max_context = (
        budget.max_context_tokens
        if budget is not None and budget.max_context_tokens is not None
        else config.max_context_tokens
    )
    if max_context is None:
        return (
            StaticContractFinding(
                name="prompt-segment-budget",
                status=SolverStatus.UNKNOWN,
                result=None,
                problem=None,
                severity="warning",
                message="required prompt segments exist but no finite context budget is declared",
                suggestion="Declare max_context_tokens or a framework-truncation-config artifact so SMT budget obligations can be solved.",
                evidence=(("required_segments", ", ".join(segment.name for segment in required)),),
                artifacts=tuple(artifact.name for artifact in prompt_segments),
            ),
        )
    unknown = tuple(segment.name for segment in required if segment.token_count is None and segment.content is None)
    if unknown:
        return (
            StaticContractFinding(
                name="prompt-segment-budget",
                status=SolverStatus.UNKNOWN,
                result=None,
                problem=None,
                severity="warning",
                message="required prompt segments have unknown token counts",
                suggestion="Add token_count or content for every required prompt segment before proving finite budget survival.",
                evidence=(("unknown_required_segments", ", ".join(unknown)),),
                artifacts=tuple(artifact.name for artifact in prompt_segments),
            ),
        )
    reserved = 0
    if budget is not None:
        reserved = (
            budget.reserve_output_tokens
            + budget.reserved_tool_tokens
            + budget.generation_prompt_tokens
            + budget.special_token_overhead
        )
    input_budget = max_context - reserved
    required_tokens = sum((segment.token_count if segment.token_count is not None else len(segment.content or "")) + segment.overhead_tokens for segment in required)
    upper = max(input_budget, required_tokens, 0)
    problem = FiniteContractProblem(
        name="prompt-segment-survival-violation",
        variables=(
            IntRangeDomain("input_budget_tokens", input_budget, input_budget),
            IntRangeDomain("required_prompt_tokens", required_tokens, required_tokens),
        ),
        constraints=(
            NamedConstraint("required-tokens-exceed-input-budget", Gt(Var("required_prompt_tokens"), Var("input_budget_tokens"))),
            NamedConstraint("finite-budget-domain", InSet(Var("input_budget_tokens"), range(min(input_budget, upper), upper + 1))),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    if result.sat:
        return (
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity="error",
                message="required prompt segments exceed the finite modeled input budget",
                suggestion="Increase the context budget, lower reserved output/tool overhead, or shorten required segments.",
                evidence=(
                    ("required_prompt_tokens", str(required_tokens)),
                    ("input_budget_tokens", str(input_budget)),
                    ("required_segments", ", ".join(segment.name for segment in required)),
                ),
                artifacts=tuple(artifact.name for artifact in (*prompt_segments, *(truncation_configs[:1] if budget else ()))),
            ),
        )
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity="info",
            message="required prompt segments fit within the finite modeled input budget",
            suggestion="Keep token counts and truncation reserves pinned so this proof remains reproducible.",
            evidence=(
                ("required_prompt_tokens", str(required_tokens)),
                ("input_budget_tokens", str(input_budget)),
                ("required_segments", ", ".join(segment.name for segment in required)),
            ),
            artifacts=tuple(artifact.name for artifact in (*prompt_segments, *(truncation_configs[:1] if budget else ()))),
        ),
    )


def _role_region_nonforgeability_obligation(
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    templates: tuple[ChatTemplateArtifact, ...],
    special_maps: tuple[SpecialTokenMapArtifact, ...],
    stop_policies: tuple[StopPolicyArtifact, ...],
    tokenizers: tuple[TokenizerArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    controlled = tuple(
        segment
        for artifact in prompt_segments
        for segment in artifact.segments
        if (segment.role or "").lower() in {"user", "tool", "function", "retrieval"}
    )
    markers = _role_boundary_markers(templates, special_maps, stop_policies, tokenizers)
    if not controlled or not markers:
        return ()

    region_candidates = tuple(
        sorted(
            {
                _region_candidate(segment.name, content)
                for segment in controlled
                for content in _controlled_content_candidates(segment.content, markers)
            }
        )
    )
    if not region_candidates:
        return ()
    problem = FiniteContractProblem(
        name="role-region-nonforgeability",
        variables=(
            EnumDomain("controlled_region", region_candidates),
            EnumDomain("boundary_marker", markers),
        ),
        constraints=(
            NamedConstraint("controlled-region-contains-boundary-marker", Contains(Var("controlled_region"), Var("boundary_marker"))),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    if result.sat:
        assignment = result.assignment or {}
        region = str(assignment.get("controlled_region", ""))
        marker = str(assignment.get("boundary_marker", "<unknown>"))
        segment_name, content = _split_region_candidate(region)
        return (
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity="error",
                message=f"controlled prompt region {segment_name!r} can contain boundary marker {marker!r}",
                suggestion="Escape, JSON-encode, wrap, or reject user/tool content that can render provider or template control delimiters.",
                evidence=(
                    ("controlled_region", segment_name),
                    ("boundary_marker", marker),
                    ("malicious_content", content),
                    ("controlled_segments", ", ".join(segment.name for segment in controlled)),
                ),
                artifacts=tuple(artifact.name for artifact in (*prompt_segments, *templates, *special_maps, *stop_policies, *tokenizers)),
            ),
        )
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity="info",
            message="controlled prompt regions are disjoint from finite role/control boundary markers",
            suggestion="Keep sanitizer assumptions and template/control-token markers pinned with the prompt artifact.",
            evidence=(
                ("controlled_segments", ", ".join(segment.name for segment in controlled)),
                ("boundary_markers", ", ".join(markers)),
            ),
            artifacts=tuple(artifact.name for artifact in (*prompt_segments, *templates, *special_maps, *stop_policies, *tokenizers)),
        ),
    )


def _stop_control_token_obligation(
    stop_policies: tuple[StopPolicyArtifact, ...],
    special_maps: tuple[SpecialTokenMapArtifact, ...],
    tokenizers: tuple[TokenizerArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    stop_sequences = tuple(sorted({sequence for policy in stop_policies for sequence in policy.stop_sequences}))
    control_tokens = tuple(
        sorted(
            {
                token.text
                for mapping in special_maps
                for token in mapping.tokens
            }.union({token for tokenizer in tokenizers for token in tokenizer.added_tokens})
        )
    )
    if not stop_sequences or not control_tokens:
        return ()
    prefix_alphabet = tuple(sorted({value[0] for value in (*stop_sequences, *control_tokens) if value}))
    problem = FiniteContractProblem(
        name="stop-control-token-collision",
        variables=(
            EnumDomain("stop_sequence", stop_sequences),
            EnumDomain("control_token", control_tokens),
            BoundedStringDomain("shared_prefix", prefix_alphabet, min_length=1, max_length=1),
        ),
        constraints=(
            NamedConstraint("stop-equals-control-token", Eq(Var("stop_sequence"), Var("control_token"))),
            NamedConstraint("prefix-from-stop", Contains(Var("stop_sequence"), Var("shared_prefix"))),
            NamedConstraint("prefix-from-control-token", Contains(Var("control_token"), Var("shared_prefix"))),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    if result.sat:
        stop = str((result.assignment or {}).get("stop_sequence", "<unknown>"))
        return (
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity="error",
                message=f"stop sequence {stop!r} is also a tokenizer control token",
                suggestion="Choose an application stop that cannot be emitted as a special/added control token, or document the provider-specific behavior.",
                evidence=(("stop_sequences", ", ".join(stop_sequences)), ("control_tokens", ", ".join(control_tokens))),
                artifacts=tuple(artifact.name for artifact in (*stop_policies, *special_maps, *tokenizers)),
            ),
        )
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity="info",
            message="stop sequences are disjoint from declared special and added control tokens",
            suggestion="Keep stop policies and tokenizer special-token maps version-pinned.",
            evidence=(("stop_sequences", ", ".join(stop_sequences)), ("control_tokens", ", ".join(control_tokens))),
            artifacts=tuple(artifact.name for artifact in (*stop_policies, *special_maps, *tokenizers)),
        ),
    )


def _tool_provider_obligation(
    tools: tuple[ToolDefinitionArtifact, ...],
    loaded_artifacts: tuple[LoadedArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    tool_providers = tuple(sorted({tool.provider for tool in tools if tool.provider}))
    provider_configs = tuple(
        sorted(
            {
                str(getattr(loaded.artifact, "provider"))
                for loaded in loaded_artifacts
                if loaded.artifact.kind is ArtifactKind.PROVIDER_CONFIG
            }
        )
    )
    if not tool_providers or not provider_configs:
        return ()
    problem = FiniteContractProblem(
        name="tool-provider-mismatch",
        variables=(
            EnumDomain("tool_provider", tool_providers),
            EnumDomain("active_provider", provider_configs),
        ),
        constraints=(
            NamedConstraint("tool-provider-differs-from-active-provider", Ne(Var("tool_provider"), Var("active_provider"))),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    severity = "error" if result.sat else "info"
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=severity,
            message=(
                "tool definitions can bind to a different provider family than the active provider config"
                if result.sat
                else "tool definition provider families match the active provider config"
            ),
            suggestion=(
                "Align tool-definition provider metadata with the selected provider config."
                if result.sat
                else "Keep provider metadata explicit when migrating tool-call envelopes."
            ),
            evidence=(("tool_providers", ", ".join(tool_providers)), ("provider_configs", ", ".join(provider_configs))),
            artifacts=tuple(artifact.name for artifact in tools),
        ),
    )


def _tool_schema_precondition_obligation(
    loaded_artifacts: tuple[LoadedArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    findings: list[StaticContractFinding] = []
    for loaded in loaded_artifacts:
        artifact = loaded.artifact
        if not isinstance(artifact, ToolDefinitionArtifact):
            continue
        metadata = dict(loaded.metadata)
        tool_count = int(metadata.get("tool_count", 0))
        for index in range(tool_count):
            tool_name = str(metadata.get(f"tool_{index}_name", ""))
            required = _metadata_tuple(metadata.get(f"tool_{index}_required"))
            if not tool_name or not required:
                continue
            properties = _metadata_tuple(metadata.get(f"tool_{index}_properties"))
            constraints = [NamedConstraint("required-parameter-not-declared", Not(InSet(Var("required_parameter"), properties)))]
            if properties:
                constraints.append(
                    NamedConstraint(
                        "declared-parameter-domain-nonempty",
                        Or(*(Eq(Var("declared_parameter"), Value(property_name)) for property_name in properties)),
                    )
                )
                variables = (
                    EnumDomain("required_parameter", required),
                    EnumDomain("declared_parameter", properties),
                )
            else:
                variables = (EnumDomain("required_parameter", required),)
            problem = FiniteContractProblem(
                name="tool-schema-precondition-satisfiability",
                variables=variables,
                constraints=tuple(constraints),
            )
            result = problem.solve(prefer_z3=prefer_z3)
            if result.sat:
                missing = str((result.assignment or {}).get("required_parameter", "<unknown>"))
                findings.append(
                    StaticContractFinding(
                        name=problem.name,
                        status=result.status,
                        result=result,
                        problem=problem,
                        severity="error",
                        message=f"tool {tool_name!r} requires parameter {missing!r} that is absent from its declared properties",
                        suggestion="Declare every required tool parameter under properties, or remove it from the required list.",
                        evidence=(
                            ("tool_name", tool_name),
                            ("required_parameters", ", ".join(required)),
                            ("declared_properties", ", ".join(properties) or "<none>"),
                        ),
                        artifacts=(artifact.name,),
                    )
                )
            else:
                findings.append(
                    StaticContractFinding(
                        name=problem.name,
                        status=result.status,
                        result=result,
                        problem=problem,
                        severity="info",
                        message=f"tool {tool_name!r} declares every required parameter property",
                        suggestion="Keep required parameter lists and schemas generated from the same typed source.",
                        evidence=(
                            ("tool_name", tool_name),
                            ("required_parameters", ", ".join(required)),
                            ("declared_properties", ", ".join(properties)),
                        ),
                        artifacts=(artifact.name,),
                    )
                )
    return tuple(findings)


def _training_target_obligation(
    manifests: tuple[TrainingManifestArtifact, ...],
    templates: tuple[ChatTemplateArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    target_roles = tuple(sorted({role for manifest in manifests for role in manifest.target_roles}))
    template_roles = tuple(sorted({role for template in templates for role in template.roles}))
    if not target_roles:
        return ()
    if not template_roles:
        return (
            StaticContractFinding(
                name="training-target-role-alignment",
                status=SolverStatus.UNKNOWN,
                result=None,
                problem=None,
                severity="warning",
                message="training target roles are declared but no chat-template role universe is available",
                suggestion="Declare a chat-template artifact with roles so training targets can be checked statically.",
                evidence=(("target_roles", ", ".join(target_roles)),),
                artifacts=tuple(manifest.name for manifest in manifests),
            ),
        )

    problem = FiniteContractProblem(
        name="training-target-role-alignment",
        variables=(EnumDomain("target_role", target_roles),),
        constraints=(
            NamedConstraint("target-role-outside-template-roles", Not(InSet(Var("target_role"), template_roles))),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    severity = "error" if result.sat else "info"
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=severity,
            message=(
                "training target roles include a role outside the chat-template role universe"
                if result.sat
                else "training target roles are members of the chat-template role universe"
            ),
            suggestion=(
                "Update the training manifest roles or the chat template roles so supervised targets align with rendered message regions."
                if result.sat
                else "Keep training manifests pinned to the same template role contract used for serving."
            ),
            evidence=(("target_roles", ", ".join(target_roles)), ("template_roles", ", ".join(template_roles))),
            artifacts=tuple(artifact.name for artifact in (*manifests, *templates)),
        ),
    )


def _training_span_region_obligation(
    manifests: tuple[TrainingManifestArtifact, ...],
    templates: tuple[ChatTemplateArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    supervised_spans = tuple(
        (manifest, span)
        for manifest in manifests
        for span in manifest.supervised_spans
        if span.supervised_target
    )
    if not supervised_spans:
        return ()

    template_roles = tuple(sorted({role for template in templates for role in template.roles}))
    bad_reasons: dict[str, tuple[str, ...]] = {}
    for manifest, span in supervised_spans:
        intended_roles = _training_intended_target_roles(manifest)
        ignored_roles = set(manifest.loss_mask_policy.ignored_roles) if manifest.loss_mask_policy is not None else set()
        reasons: list[str] = []
        if span.target_role not in intended_roles:
            reasons.append("target-role-not-supervised")
        if span.target_role in ignored_roles:
            reasons.append("target-role-loss-ignored")
        if template_roles and span.rendered_region_role not in template_roles:
            reasons.append("rendered-region-role-not-in-template")
        if span.rendered_region_role != span.target_role:
            reasons.append("target-span-outside-intended-rendered-role")
        if not (span.region_start_token <= span.start_token <= span.end_token <= span.region_end_token):
            reasons.append("tokenized-span-outside-rendered-region-bounds")
        if not span.loss_masked:
            reasons.append("supervised-target-not-selected-by-loss-mask")
        if (
            manifest.packing_window is not None
            and manifest.packing_window.preserve_example_boundaries
            and span.crosses_packing_boundary
        ):
            reasons.append("packed-span-crosses-example-boundary")
        if reasons:
            bad_reasons[span.span_id] = tuple(reasons)

    span_ids = tuple(span.span_id for _manifest, span in supervised_spans)
    problem = FiniteContractProblem(
        name="training-supervised-span-region-alignment",
        variables=(EnumDomain("span_id", span_ids),),
        constraints=(
            NamedConstraint("supervised-span-violates-render-token-pack-mask-contract", InSet(Var("span_id"), bad_reasons)),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    severity = "error" if result.sat else "info"
    selected_id = str(result.assignment.get("span_id")) if result.assignment else None
    selected_manifest = None
    selected_span = None
    if selected_id is not None:
        for manifest, span in supervised_spans:
            if span.span_id == selected_id:
                selected_manifest = manifest
                selected_span = span
                break
    evidence: list[tuple[str, str]] = [
        ("span_count", str(len(supervised_spans))),
        ("intended_target_roles", ", ".join(sorted({role for manifest in manifests for role in _training_intended_target_roles(manifest)}))),
        ("template_roles", ", ".join(template_roles)),
        ("render", "compare target_role with observed rendered_region_role"),
        ("tokenize", "require target token bounds to be inside rendered region bounds"),
        ("pack", "reject supervised spans crossing preserved example boundaries"),
        ("loss_mask", "require supervised spans to be selected by the loss mask"),
    ]
    if selected_manifest is not None and selected_span is not None:
        evidence.extend(
            (
                ("span_id", selected_span.span_id),
                ("manifest", selected_manifest.name),
                ("target_role", selected_span.target_role),
                ("rendered_region_role", selected_span.rendered_region_role),
                ("token_span", f"{selected_span.start_token}:{selected_span.end_token}"),
                ("rendered_region_bounds", f"{selected_span.region_start_token}:{selected_span.region_end_token}"),
                ("loss_masked", str(selected_span.loss_masked)),
                ("reasons", ", ".join(bad_reasons[selected_span.span_id])),
            )
        )
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=severity,
            message=(
                "a supervised target span can fall outside its intended rendered assistant region, token bounds, packing boundary, or loss mask"
                if result.sat
                else "all declared supervised target spans stay inside their intended rendered role regions, token bounds, packing boundaries, and loss masks"
            ),
            suggestion=(
                "Regenerate the dataset span manifest from the same renderer, tokenizer, packer, and loss-mask builder used for fine-tuning."
                if result.sat
                else "Keep supervised span manifests emitted by the training data builder and checked before fine-tuning."
            ),
            evidence=tuple(evidence),
            artifacts=tuple(manifest.name for manifest in manifests),
        ),
    )


def _training_intended_target_roles(manifest: TrainingManifestArtifact) -> tuple[str, ...]:
    if manifest.loss_mask_policy is not None and manifest.loss_mask_policy.target_roles:
        return manifest.loss_mask_policy.target_roles
    return manifest.target_roles


def _training_source_leakage_obligation(
    manifests: tuple[TrainingManifestArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    source_facts = tuple(
        (manifest, span, index, contribution)
        for manifest in manifests
        for span in manifest.supervised_spans
        if span.supervised_target and span.loss_masked
        for index, contribution in enumerate(span.source_contributions)
    )
    if not source_facts:
        return ()

    leaking_kinds = {
        TrainingTextSourceKind.USER,
        TrainingTextSourceKind.TOOL,
        TrainingTextSourceKind.RETRIEVAL,
        TrainingTextSourceKind.PREFERENCE,
    }
    leaking_fields = {"user", "tool", "retrieval", "context", "document", "preference", "prompt", "chosen", "rejected"}
    leak_reasons: dict[str, tuple[str, ...]] = {}
    source_details: dict[str, tuple[TrainingManifestArtifact, object, object]] = {}
    for manifest, span, index, contribution in source_facts:
        leak_id = f"{span.span_id}:{index}:{contribution.source_id}"
        source_details[leak_id] = (manifest, span, contribution)
        reasons: list[str] = []
        if contribution.source_kind in leaking_kinds:
            reasons.append(f"{contribution.source_kind.value}-text-overlaps-supervised-target")
        if contribution.source_field is not None and contribution.source_field.lower() in leaking_fields:
            reasons.append(f"{contribution.source_field.lower()}-field-overlaps-supervised-target")
        overlaps_target = contribution.start_token <= span.end_token and span.start_token <= contribution.end_token
        if not overlaps_target:
            reasons.clear()
        if reasons:
            leak_reasons[leak_id] = tuple(reasons)

    leak_ids = tuple(source_details)
    problem = FiniteContractProblem(
        name="training-supervised-source-leakage",
        variables=(EnumDomain("source_contribution", leak_ids),),
        constraints=(
            NamedConstraint("non-target-source-overlaps-supervised-target", InSet(Var("source_contribution"), leak_reasons)),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    severity = "error" if result.sat else "info"
    selected_id = str(result.assignment.get("source_contribution")) if result.assignment else None
    evidence: list[tuple[str, str]] = [
        ("source_contribution_count", str(len(source_facts))),
        ("leaking_source_kinds", ", ".join(kind.value for kind in sorted(leaking_kinds, key=lambda item: item.value))),
        ("range_model", "closed token intervals must not overlap supervised/loss-masked target spans"),
    ]
    if selected_id is not None:
        manifest, span, contribution = source_details[selected_id]
        evidence.extend(
            (
                ("manifest", manifest.name),
                ("span_id", span.span_id),
                ("target_role", span.target_role),
                ("target_token_span", f"{span.start_token}:{span.end_token}"),
                ("source_id", contribution.source_id),
                ("source_kind", contribution.source_kind.value),
                ("source_token_span", f"{contribution.start_token}:{contribution.end_token}"),
                ("transform", contribution.transform),
                ("reasons", ", ".join(leak_reasons[selected_id])),
            )
        )
        if contribution.source_field is not None:
            evidence.append(("source_field", contribution.source_field))
        if contribution.text_sha256 is not None:
            evidence.append(("text_sha256", contribution.text_sha256))

    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=severity,
            message=(
                "user, tool, retrieval, or preference text can overlap a supervised/loss-masked target span after a dataset transform"
                if result.sat
                else "declared source contributions do not place user, tool, retrieval, or preference text inside supervised target spans"
            ),
            suggestion=(
                "Regenerate target spans after transforms, keep non-assistant source ranges loss-masked, and store only hashes for leaked text witnesses."
                if result.sat
                else "Keep transform source-contribution manifests emitted by the dataset builder and verify them before fine-tuning."
            ),
            evidence=tuple(evidence),
            artifacts=tuple(manifest.name for manifest in manifests),
        ),
    )


def _role_boundary_markers(
    templates: tuple[ChatTemplateArtifact, ...],
    special_maps: tuple[SpecialTokenMapArtifact, ...],
    stop_policies: tuple[StopPolicyArtifact, ...],
    tokenizers: tuple[TokenizerArtifact, ...],
) -> tuple[str, ...]:
    roles = tuple(sorted({role for template in templates for role in template.roles if role}))
    role_markers = {
        marker
        for role in roles
        for marker in (
            f"{role}:",
            f"<|im_start|>{role}",
            f"<|start_header_id|>{role}<|end_header_id|>",
            f"### {role}",
            f"[/{role.upper()}]",
        )
    }
    control_markers = {
        token.text
        for mapping in special_maps
        for token in mapping.tokens
    }.union(
        {token for tokenizer in tokenizers for token in tokenizer.added_tokens},
        {sequence for policy in stop_policies for sequence in policy.stop_sequences},
    )
    return tuple(sorted(marker for marker in role_markers.union(control_markers) if marker))


def _controlled_content_candidates(content: str | None, markers: tuple[str, ...]) -> tuple[str, ...]:
    if content is not None:
        return (content,)
    return ()


def _region_candidate(segment_name: str, content: str) -> str:
    return f"{segment_name}\x1f{content}"


def _split_region_candidate(value: str) -> tuple[str, str]:
    if "\x1f" not in value:
        return value, value
    name, content = value.split("\x1f", 1)
    return name, content


def _metadata_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        return (value,)
    return ()
