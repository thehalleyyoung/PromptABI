"""Z3-backed finite static contracts over PromptABI artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import (
    ArtifactKind,
    ChatTemplateArtifact,
    FrameworkTruncationConfigArtifact,
    PromptPackArtifact,
    PreferencePairContract,
    PromptSegmentArtifact,
    SpecialTokenMapArtifact,
    SchemaArtifact,
    StaticContractArtifact,
    StaticContractInvariant,
    StaticContractRule,
    StopPolicyArtifact,
    TokenizerArtifact,
    ToolDefinitionArtifact,
    TrainingManifestArtifact,
    TrainingTextSourceKind,
)
from .config import VerificationConfig
from .diagnostics import SourceSpan
from .formal import (
    BoundedStringDomain,
    Contains,
    EnumDomain,
    Eq,
    Ge,
    FiniteContractProblem,
    Gt,
    InSet,
    IntRangeDomain,
    Le,
    Lt,
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
    source_span: SourceSpan | None = None


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
    schemas = tuple(artifact for artifact in artifacts if isinstance(artifact, SchemaArtifact))
    prompt_packs = tuple(artifact for artifact in artifacts if isinstance(artifact, PromptPackArtifact))
    static_contracts = tuple(artifact for artifact in artifacts if isinstance(artifact, StaticContractArtifact))
    static_contract_source_spans = {
        loaded.artifact.name: {name: span for name, span in loaded.source_spans}
        for loaded in loaded_artifacts
        if isinstance(loaded.artifact, StaticContractArtifact)
    }
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
    findings.extend(_training_stage_consistency_obligation(training_manifests, prefer_z3=prefer_z3))
    findings.extend(_preference_pair_contract_obligation(training_manifests, prefer_z3=prefer_z3))
    findings.extend(
        _explicit_static_contract_obligations(
            config,
            static_contracts,
            static_contract_source_spans,
            loaded_artifacts,
            prompt_segments,
            truncation_configs,
            templates,
            schemas,
            stop_policies,
            training_manifests,
            prompt_packs,
            prefer_z3=prefer_z3,
        )
    )

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


def _training_stage_consistency_obligation(
    manifests: tuple[TrainingManifestArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    stage_facts = tuple(
        (manifest, stage)
        for manifest in manifests
        for stage in manifest.pipeline_stages
    )
    if not stage_facts:
        return ()

    expected_stages = ("dataset-preparation", "training", "evaluation", "serving")
    mismatch_ids: dict[str, tuple[str, ...]] = {}
    component_details: dict[str, tuple[TrainingManifestArtifact, str, tuple[tuple[str, str], ...]]] = {}
    incomplete: list[StaticContractFinding] = []

    for manifest in manifests:
        if not manifest.pipeline_stages:
            continue
        observed_stages = tuple(stage.stage for stage in manifest.pipeline_stages)
        missing_stages = tuple(stage for stage in expected_stages if stage not in observed_stages)
        incomplete_components: list[str] = []
        for component in ("tokenizer", "chat-template"):
            fingerprints = tuple(
                (stage.stage, _training_stage_fingerprint(stage, component))
                for stage in manifest.pipeline_stages
            )
            comparable = tuple((stage, fingerprint) for stage, fingerprint in fingerprints if fingerprint is not None)
            if len(comparable) != len(fingerprints):
                incomplete_components.append(component)
            unique_fingerprints = {fingerprint for _stage, fingerprint in comparable}
            if len(unique_fingerprints) > 1:
                mismatch_id = f"{manifest.name}:{component}"
                mismatch_ids[mismatch_id] = (
                    f"{component}-differs-across-dataset-preparation-training-evaluation-serving",
                )
                component_details[mismatch_id] = (
                    manifest,
                    component,
                    tuple((stage, fingerprint) for stage, fingerprint in comparable),
                )
        if missing_stages or incomplete_components:
            evidence: list[tuple[str, str]] = [
                ("manifest", manifest.name),
                ("observed_stages", ", ".join(observed_stages)),
                ("expected_stages", ", ".join(expected_stages)),
            ]
            if missing_stages:
                evidence.append(("missing_stages", ", ".join(missing_stages)))
            if incomplete_components:
                evidence.append(("incomplete_components", ", ".join(sorted(set(incomplete_components)))))
            incomplete.append(
                StaticContractFinding(
                    name="training-tokenizer-template-stage-consistency",
                    status=SolverStatus.UNKNOWN,
                    result=None,
                    problem=None,
                    severity="warning",
                    message="training pipeline stage pins are incomplete for tokenizer/template consistency proof",
                    suggestion="Record dataset-preparation, training, evaluation, and serving tokenizer/template names plus version, revision, or sha256 pins in the training manifest.",
                    evidence=tuple(evidence),
                    artifacts=(manifest.name,),
                )
            )

    if not component_details:
        complete_stage_count = sum(len(manifest.pipeline_stages) for manifest in manifests if manifest.pipeline_stages)
        if incomplete:
            return tuple(incomplete)
        return (
            StaticContractFinding(
                name="training-tokenizer-template-stage-consistency",
                status=SolverStatus.UNSAT,
                result=None,
                problem=None,
                severity="info",
                message="tokenizer and chat-template pins match across fine-tuning preparation, training, evaluation, and serving stages",
                suggestion="Keep stage pins generated by the data-preparation job and checked before fine-tuning or serving rollout.",
                evidence=(("stage_count", str(complete_stage_count)), ("expected_stages", ", ".join(expected_stages))),
                artifacts=tuple(manifest.name for manifest in manifests if manifest.pipeline_stages),
            ),
        )

    problem = FiniteContractProblem(
        name="training-tokenizer-template-stage-consistency",
        variables=(EnumDomain("stage_component", tuple(component_details)),),
        constraints=(
            NamedConstraint("fine-tuning-stage-tokenizer-or-template-drift", InSet(Var("stage_component"), mismatch_ids)),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    selected_id = str(result.assignment.get("stage_component")) if result.assignment else None
    evidence: list[tuple[str, str]] = [
        ("stage_count", str(len(stage_facts))),
        ("expected_stages", ", ".join(expected_stages)),
        ("compared_components", "tokenizer, chat-template"),
    ]
    artifacts = tuple(manifest.name for manifest in manifests if manifest.pipeline_stages)
    if selected_id is not None:
        manifest, component, details = component_details[selected_id]
        evidence.extend(
            (
                ("manifest", manifest.name),
                ("component", component),
                ("stage_fingerprints", "; ".join(f"{stage}={fingerprint}" for stage, fingerprint in details)),
                ("reasons", ", ".join(mismatch_ids[selected_id])),
            )
        )
        artifacts = (manifest.name,)

    return (
        *tuple(incomplete),
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity="error" if result.sat else "info",
            message=(
                "fine-tuning dataset preparation, training, evaluation, and serving can use different tokenizer or chat-template pins"
                if result.sat
                else "tokenizer and chat-template pins match across fine-tuning preparation, training, evaluation, and serving stages"
            ),
            suggestion=(
                "Regenerate the dataset with the serving tokenizer/template or update training, evaluation, and serving lockfiles to one shared pinned contract."
                if result.sat
                else "Keep stage pins generated by the data-preparation job and checked before fine-tuning or serving rollout."
            ),
            evidence=tuple(evidence),
            artifacts=artifacts,
        ),
    )


def _training_stage_fingerprint(stage: object, component: str) -> str | None:
    if component == "tokenizer":
        fields = (
            ("name", getattr(stage, "tokenizer_name")),
            ("version", getattr(stage, "tokenizer_version")),
            ("revision", getattr(stage, "tokenizer_revision")),
            ("sha256", getattr(stage, "tokenizer_sha256")),
        )
    elif component == "chat-template":
        fields = (
            ("name", getattr(stage, "chat_template_name")),
            ("version", getattr(stage, "chat_template_version")),
            ("revision", getattr(stage, "chat_template_revision")),
            ("sha256", getattr(stage, "chat_template_sha256")),
            ("add_generation_prompt", getattr(stage, "add_generation_prompt")),
        )
    else:
        raise ValueError(f"unknown training stage component: {component}")
    populated = tuple((name, value) for name, value in fields if value is not None)
    if not populated:
        return None
    return ",".join(f"{name}={value}" for name, value in populated)


def _preference_pair_contract_obligation(
    manifests: tuple[TrainingManifestArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    pair_facts = tuple(
        (manifest, pair)
        for manifest in manifests
        for pair in manifest.preference_pairs
    )
    if not pair_facts:
        return ()

    bad_reasons: dict[str, tuple[str, ...]] = {}
    pair_details: dict[str, tuple[TrainingManifestArtifact, PreferencePairContract]] = {}
    for manifest, pair in pair_facts:
        pair_key = f"{manifest.name}:{pair.pair_id}"
        pair_details[pair_key] = (manifest, pair)
        reasons: list[str] = []
        branch_prompt_hashes = tuple(
            value
            for value in (pair.chosen_prompt_sha256, pair.rejected_prompt_sha256)
            if value is not None
        )
        if branch_prompt_hashes and (
            len(set(branch_prompt_hashes)) != 1
            or any(value != pair.prompt_sha256 for value in branch_prompt_hashes)
        ):
            reasons.append("prompt-prefix-hash-mismatch")
        if pair.chosen_role_layout != pair.rejected_role_layout:
            reasons.append("role-layout-mismatch")
        if pair.chosen_tokenizer != pair.rejected_tokenizer:
            reasons.append("tokenizer-version-mismatch")
        if pair.chosen_mask_policy != pair.rejected_mask_policy:
            reasons.append("mask-policy-mismatch")
        if pair.chosen_prompt_tokens != pair.rejected_prompt_tokens:
            reasons.append("prompt-prefix-token-length-mismatch")
        if pair.chosen_response_start_token != pair.rejected_response_start_token:
            reasons.append("response-start-token-mismatch")
        if pair.chosen_truncated or pair.rejected_truncated:
            reasons.append("preference-branch-truncated")
        if (
            manifest.packing_window is not None
            and manifest.packing_window.preserve_example_boundaries
            and pair.chosen_packed_example_id is not None
            and pair.rejected_packed_example_id is not None
            and pair.chosen_packed_example_id != pair.rejected_packed_example_id
        ):
            reasons.append("packed-example-boundary-mismatch")
        if reasons:
            bad_reasons[pair_key] = tuple(reasons)

    problem = FiniteContractProblem(
        name="training-preference-pair-contract",
        variables=(EnumDomain("preference_pair", tuple(pair_details)),),
        constraints=(
            NamedConstraint("preference-pair-branches-diverge-before-compared-response", InSet(Var("preference_pair"), bad_reasons)),
        ),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    selected_id = str(result.assignment.get("preference_pair")) if result.assignment else None
    evidence: list[tuple[str, str]] = [
        ("preference_pair_count", str(len(pair_facts))),
        ("checked_invariants", "prompt prefix hash/length, role layout, tokenizer, mask policy, response start, truncation, packing boundary"),
    ]
    artifacts = tuple(manifest.name for manifest in manifests if manifest.preference_pairs)
    if selected_id is not None:
        manifest, pair = pair_details[selected_id]
        evidence.extend(
            (
                ("manifest", manifest.name),
                ("pair_id", pair.pair_id),
                ("prompt_sha256", pair.prompt_sha256),
                ("chosen_sha256", pair.chosen_sha256),
                ("rejected_sha256", pair.rejected_sha256),
                ("chosen_role_layout", " > ".join(pair.chosen_role_layout)),
                ("rejected_role_layout", " > ".join(pair.rejected_role_layout)),
                ("chosen_tokenizer", pair.chosen_tokenizer),
                ("rejected_tokenizer", pair.rejected_tokenizer),
                ("chosen_mask_policy", pair.chosen_mask_policy),
                ("rejected_mask_policy", pair.rejected_mask_policy),
                ("chosen_prompt_tokens", str(pair.chosen_prompt_tokens)),
                ("rejected_prompt_tokens", str(pair.rejected_prompt_tokens)),
                ("chosen_response_start_token", str(pair.chosen_response_start_token)),
                ("rejected_response_start_token", str(pair.rejected_response_start_token)),
                ("chosen_response_token_span", f"{pair.chosen_response_start_token}:{pair.chosen_response_end_token}"),
                ("rejected_response_token_span", f"{pair.rejected_response_start_token}:{pair.rejected_response_end_token}"),
                ("reasons", ", ".join(bad_reasons[selected_id])),
            )
        )
        if pair.chosen_packed_example_id is not None:
            evidence.append(("chosen_packed_example_id", pair.chosen_packed_example_id))
        if pair.rejected_packed_example_id is not None:
            evidence.append(("rejected_packed_example_id", pair.rejected_packed_example_id))
        if pair.chosen_prompt_sha256 is not None:
            evidence.append(("chosen_prompt_sha256", pair.chosen_prompt_sha256))
        if pair.rejected_prompt_sha256 is not None:
            evidence.append(("rejected_prompt_sha256", pair.rejected_prompt_sha256))
        artifacts = (manifest.name,)

    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity="error" if result.sat else "info",
            message=(
                "a chosen/rejected preference pair can diverge in prompt prefix, role layout, tokenizer, masking, truncation, or packing before the compared response"
                if result.sat
                else "all declared chosen/rejected preference pairs share prompt prefix, role layout, tokenizer version, masking policy, and finite packing/truncation invariants"
            ),
            suggestion=(
                "Regenerate preference-pair manifests from the DPO/RLHF data builder and require shared prompt hashes, layout fingerprints, tokenizer pins, mask policy, response starts, and packing boundaries."
                if result.sat
                else "Keep preference-pair contracts emitted by the data-preparation job and verify them before DPO/RLHF training."
            ),
            evidence=tuple(evidence),
            artifacts=artifacts,
        ),
    )


def _explicit_static_contract_obligations(
    config: VerificationConfig,
    contracts: tuple[StaticContractArtifact, ...],
    contract_source_spans: dict[str, dict[str, SourceSpan]],
    loaded_artifacts: tuple[LoadedArtifact, ...],
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    truncation_configs: tuple[FrameworkTruncationConfigArtifact, ...],
    templates: tuple[ChatTemplateArtifact, ...],
    schemas: tuple[SchemaArtifact, ...],
    stop_policies: tuple[StopPolicyArtifact, ...],
    training_manifests: tuple[TrainingManifestArtifact, ...],
    prompt_packs: tuple[PromptPackArtifact, ...],
    *,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    findings: list[StaticContractFinding] = []
    for contract in contracts:
        for rule in contract.rules:
            rule_span = _static_contract_rule_span(contract, rule, contract_source_spans)
            findings.extend(_explicit_allowed_roles_obligation(contract, rule, prompt_segments, templates, training_manifests, rule_span=rule_span, prefer_z3=prefer_z3))
            findings.extend(_explicit_required_regions_obligation(contract, rule, prompt_segments, prompt_packs, rule_span=rule_span, prefer_z3=prefer_z3))
            findings.extend(_explicit_forbidden_delimiters_obligation(contract, rule, prompt_segments, rule_span=rule_span, prefer_z3=prefer_z3))
            findings.extend(_explicit_schema_obligation(contract, rule, loaded_artifacts, schemas, rule_span=rule_span, prefer_z3=prefer_z3))
            findings.extend(_explicit_stop_policy_obligation(contract, rule, stop_policies, rule_span=rule_span, prefer_z3=prefer_z3))
            findings.extend(_explicit_invariant_obligation(contract, rule, config, prompt_segments, truncation_configs, stop_policies, rule_span=rule_span, prefer_z3=prefer_z3))
    return tuple(findings)


def _explicit_allowed_roles_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    templates: tuple[ChatTemplateArtifact, ...],
    training_manifests: tuple[TrainingManifestArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    if not rule.allowed_roles:
        return ()
    observed_roles = tuple(
        sorted(
            {
                role
                for template in templates
                for role in template.roles
            }.union(
                {
                    segment.role
                    for artifact in prompt_segments
                    for segment in artifact.segments
                    if segment.role is not None
                },
                {role for manifest in training_manifests for role in (*manifest.message_roles, *manifest.target_roles)},
            )
        )
    )
    if not observed_roles:
        return (_explicit_unknown(contract, rule, "static-contract-allowed-roles", "allowed roles are declared but no role-bearing artifacts are loaded", "Load chat-template, prompt-segment, or training-manifest artifacts so declared role policy can be checked.", (("allowed_roles", ", ".join(rule.allowed_roles)),), rule_span=rule_span),)
    problem = FiniteContractProblem(
        name="static-contract-allowed-roles",
        variables=(EnumDomain("observed_role", observed_roles),),
        constraints=(NamedConstraint("observed-role-outside-declared-allowed-roles", Not(InSet(Var("observed_role"), rule.allowed_roles))),),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=_explicit_severity(rule, result.sat),
            message=(
                f"static contract rule {rule.name!r} found an observed role outside its allowed role set"
                if result.sat
                else f"static contract rule {rule.name!r} allows every observed role"
            ),
            suggestion=(
                "Update the contract allowed_roles or normalize the artifact role labels before verification."
                if result.sat
                else "Keep role labels pinned in the contract when templates, data, or providers change."
            ),
            evidence=(
                ("contract", contract.name),
                ("rule", rule.name),
                ("allowed_roles", ", ".join(rule.allowed_roles)),
                ("observed_roles", ", ".join(observed_roles)),
            ),
            artifacts=(contract.name,),
            source_span=rule_span,
        ),
    )


def _explicit_required_regions_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    prompt_packs: tuple[PromptPackArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    if not rule.required_regions:
        return ()
    observed_regions = tuple(
        sorted(
            {segment.name for artifact in prompt_segments for segment in artifact.segments}.union(
                {
                    region
                    for pack in prompt_packs
                    for template in pack.exported_templates
                    for region in template.required_regions
                }
            )
        )
    )
    if not observed_regions:
        return (_explicit_unknown(contract, rule, "static-contract-required-regions", "required prompt regions are declared but no region-bearing artifacts are loaded", "Load prompt-segment or prompt-pack artifacts so required regions can be checked.", (("required_regions", ", ".join(rule.required_regions)),), rule_span=rule_span),)
    problem = FiniteContractProblem(
        name="static-contract-required-regions",
        variables=(EnumDomain("required_region", rule.required_regions),),
        constraints=(NamedConstraint("declared-required-region-missing-from-artifacts", Not(InSet(Var("required_region"), observed_regions))),),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=_explicit_severity(rule, result.sat),
            message=(
                f"static contract rule {rule.name!r} requires a prompt region missing from loaded artifacts"
                if result.sat
                else f"static contract rule {rule.name!r} finds every required prompt region"
            ),
            suggestion=(
                "Add the missing segment/pack region, or remove it from required_regions if it is no longer part of the interface."
                if result.sat
                else "Keep required region names stable across prompt-pack and application updates."
            ),
            evidence=(
                ("contract", contract.name),
                ("rule", rule.name),
                ("required_regions", ", ".join(rule.required_regions)),
                ("observed_regions", ", ".join(observed_regions)),
            ),
            artifacts=(contract.name,),
            source_span=rule_span,
        ),
    )


def _explicit_forbidden_delimiters_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    if not rule.forbidden_delimiters:
        return ()
    controlled = tuple(
        segment
        for artifact in prompt_segments
        for segment in artifact.segments
        if (segment.role or "").lower() in {"user", "tool", "function", "retrieval"} and segment.content is not None
    )
    if not controlled:
        return (_explicit_unknown(contract, rule, "static-contract-forbidden-delimiters", "forbidden delimiters are declared but no controlled prompt content is materialized", "Provide content or bounded examples for user/tool/retrieval prompt segments so delimiter exclusion can be checked.", (("forbidden_delimiters", ", ".join(rule.forbidden_delimiters)),), rule_span=rule_span),)
    region_candidates = tuple(sorted(_region_candidate(segment.name, segment.content or "") for segment in controlled))
    problem = FiniteContractProblem(
        name="static-contract-forbidden-delimiters",
        variables=(
            EnumDomain("controlled_region", region_candidates),
            EnumDomain("forbidden_delimiter", rule.forbidden_delimiters),
        ),
        constraints=(NamedConstraint("controlled-region-contains-forbidden-delimiter", Contains(Var("controlled_region"), Var("forbidden_delimiter"))),),
    )
    result = problem.solve(prefer_z3=prefer_z3)
    evidence: list[tuple[str, str]] = [
        ("contract", contract.name),
        ("rule", rule.name),
        ("forbidden_delimiters", ", ".join(rule.forbidden_delimiters)),
        ("controlled_regions", ", ".join(segment.name for segment in controlled)),
    ]
    if result.assignment:
        segment_name, content = _split_region_candidate(str(result.assignment.get("controlled_region", "")))
        evidence.extend((("controlled_region", segment_name), ("malicious_content", content)))
    return (
        StaticContractFinding(
            name=problem.name,
            status=result.status,
            result=result,
            problem=problem,
            severity=_explicit_severity(rule, result.sat),
            message=(
                f"static contract rule {rule.name!r} found a forbidden delimiter inside controlled prompt content"
                if result.sat
                else f"static contract rule {rule.name!r} excludes forbidden delimiters from controlled prompt content"
            ),
            suggestion=(
                "Escape, reject, or encode controlled fields before rendering them through the chat template."
                if result.sat
                else "Keep delimiter and sanitizer assumptions in the checked contract."
            ),
            evidence=tuple(evidence),
            artifacts=(contract.name,),
            source_span=rule_span,
        ),
    )


def _explicit_schema_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    schemas: tuple[SchemaArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    findings: list[StaticContractFinding] = []
    schema_by_name = {schema.name: schema for schema in schemas}
    metadata_by_name = {loaded.artifact.name: dict(loaded.metadata) for loaded in loaded_artifacts if isinstance(loaded.artifact, SchemaArtifact)}
    for obligation in rule.schema_obligations:
        schema = schema_by_name.get(obligation.schema)
        if schema is None:
            findings.append(_explicit_unknown(contract, rule, "static-contract-schema-obligations", f"schema obligation {obligation.schema!r} targets no loaded schema artifact", "Load the referenced schema artifact or update the contract schema name.", (("schema", obligation.schema), ("requires", ", ".join(obligation.requires))), rule_span=rule_span))
            continue
        required_names = _metadata_tuple(metadata_by_name.get(schema.name, {}).get("required_property_names"))
        if not required_names:
            findings.append(_explicit_unknown(contract, rule, "static-contract-schema-obligations", f"schema obligation {obligation.schema!r} cannot be checked because required properties were not extracted", "Use a JSON Schema object with a supported required list so PromptABI can prove schema obligations.", (("schema", obligation.schema), ("requires", ", ".join(obligation.requires))), rule_span=rule_span))
            continue
        problem = FiniteContractProblem(
            name="static-contract-schema-obligations",
            variables=(EnumDomain("required_property", obligation.requires),),
            constraints=(NamedConstraint("contract-required-schema-property-missing", Not(InSet(Var("required_property"), required_names))),),
        )
        result = problem.solve(prefer_z3=prefer_z3)
        findings.append(
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity=_explicit_severity(rule, result.sat),
                message=(
                    f"static contract rule {rule.name!r} requires a schema property missing from {schema.name!r}"
                    if result.sat
                    else f"static contract rule {rule.name!r} finds all required schema properties in {schema.name!r}"
                ),
                suggestion=(
                    "Add the missing property to the schema required list or relax the contract obligation."
                    if result.sat
                    else "Keep schema required lists and contract obligations generated from the same interface source."
                ),
                evidence=(
                    ("contract", contract.name),
                    ("rule", rule.name),
                    ("schema", schema.name),
                    ("requires", ", ".join(obligation.requires)),
                    ("schema_required_properties", ", ".join(required_names)),
                ),
                artifacts=(contract.name, schema.name),
                source_span=rule_span,
            )
        )
    return tuple(findings)


def _explicit_stop_policy_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    stop_policies: tuple[StopPolicyArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    findings: list[StaticContractFinding] = []
    observed_stops = tuple(sorted({sequence for policy in stop_policies for sequence in policy.stop_sequences}))
    for policy in rule.stop_policies:
        if not policy.stops:
            findings.append(_explicit_unknown(contract, rule, "static-contract-stop-policies", f"static contract stop policy {policy.name!r} declares no stops", "Add concrete stop strings to the contract stop policy before checking it.", (("stop_policy", policy.name),), rule_span=rule_span))
            continue
        if not observed_stops:
            findings.append(_explicit_unknown(contract, rule, "static-contract-stop-policies", "stop policy obligations are declared but no stop-policy artifact is loaded", "Load a stop-policy artifact so declared stops can be checked against real request/generation settings.", (("declared_stops", ", ".join(policy.stops)),), rule_span=rule_span))
            continue
        problem = FiniteContractProblem(
            name="static-contract-stop-policies",
            variables=(EnumDomain("declared_stop", policy.stops),),
            constraints=(NamedConstraint("contract-stop-missing-from-loaded-stop-policies", Not(InSet(Var("declared_stop"), observed_stops))),),
        )
        result = problem.solve(prefer_z3=prefer_z3)
        findings.append(
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity=_explicit_severity(rule, result.sat),
                message=(
                    f"static contract rule {rule.name!r} declares a stop string missing from loaded stop policies"
                    if result.sat
                    else f"static contract rule {rule.name!r} matches declared stop strings to loaded stop policies"
                ),
                suggestion=(
                    "Update stop-policy artifacts or the contract so provider request stops and interface policy agree."
                    if result.sat
                    else "Keep stop declarations pinned across provider and framework adapters."
                ),
                evidence=(
                    ("contract", contract.name),
                    ("rule", rule.name),
                    ("stop_policy", policy.name),
                    ("declared_stops", ", ".join(policy.stops)),
                    ("observed_stops", ", ".join(observed_stops)),
                    ("forbid_inside", policy.forbid_inside or "<unspecified>"),
                ),
                artifacts=(contract.name,),
                source_span=rule_span,
            )
        )
    return tuple(findings)


def _explicit_invariant_obligation(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    config: VerificationConfig,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    truncation_configs: tuple[FrameworkTruncationConfigArtifact, ...],
    stop_policies: tuple[StopPolicyArtifact, ...],
    *,
    rule_span: SourceSpan | None,
    prefer_z3: bool,
) -> tuple[StaticContractFinding, ...]:
    findings: list[StaticContractFinding] = []
    metrics, missing_reason = _static_contract_metrics(config, prompt_segments, truncation_configs, stop_policies)
    for invariant in rule.invariants:
        left = _resolve_static_contract_operand(invariant.left, metrics)
        right = _resolve_static_contract_operand(invariant.right, metrics)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append(invariant.left)
            if right is None:
                missing.append(invariant.right)
            reason = missing_reason or f"unknown metric operand(s): {', '.join(missing)}"
            findings.append(
                _explicit_unknown(
                    contract,
                    rule,
                    "static-contract-invariant",
                    f"static contract invariant {invariant.name!r} could not be resolved",
                    "Use supported finite metrics such as required_prompt_tokens, input_budget_tokens, reserved_tokens, max_context_tokens, required_region_count, and stop_policy_count, or integer constants.",
                    (("invariant", invariant.name), ("expression", f"{invariant.left} {invariant.op} {invariant.right}"), ("reason", reason)),
                    rule_span=rule_span,
                )
            )
            continue
        problem = FiniteContractProblem(
            name="static-contract-invariant",
            variables=(
                IntRangeDomain("left_value", left, left),
                IntRangeDomain("right_value", right, right),
            ),
            constraints=(NamedConstraint("contract-invariant-violated", _invariant_violation_expression(invariant)),),
        )
        result = problem.solve(prefer_z3=prefer_z3)
        findings.append(
            StaticContractFinding(
                name=problem.name,
                status=result.status,
                result=result,
                problem=problem,
                severity=_explicit_severity(rule, result.sat),
                message=(
                    f"static contract invariant {invariant.name!r} is violated by loaded artifact metrics"
                    if result.sat
                    else f"static contract invariant {invariant.name!r} holds over loaded artifact metrics"
                ),
                suggestion=(
                    "Adjust the contract, prompt region sizes, or truncation budget so the finite invariant is true."
                    if result.sat
                    else "Keep finite contract metrics regenerated from the same artifacts that CI verifies."
                ),
                evidence=(
                    ("contract", contract.name),
                    ("rule", rule.name),
                    ("invariant", invariant.name),
                    ("expression", f"{invariant.left} {invariant.op} {invariant.right}"),
                    ("left_value", str(left)),
                    ("right_value", str(right)),
                ),
                artifacts=(contract.name,),
                source_span=rule_span,
            )
        )
    return tuple(findings)


def _explicit_unknown(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    name: str,
    message: str,
    suggestion: str,
    evidence: tuple[tuple[str, str], ...],
    *,
    rule_span: SourceSpan | None,
) -> StaticContractFinding:
    return StaticContractFinding(
        name=name,
        status=SolverStatus.UNKNOWN,
        result=None,
        problem=None,
        severity="warning",
        message=message,
        suggestion=suggestion,
        evidence=(("contract", contract.name), ("rule", rule.name), *evidence),
        artifacts=(contract.name,),
        source_span=rule_span,
    )


def _static_contract_rule_span(
    contract: StaticContractArtifact,
    rule: StaticContractRule,
    source_spans: dict[str, dict[str, SourceSpan]],
) -> SourceSpan | None:
    return source_spans.get(contract.name, {}).get(f"rules.{rule.name}") or contract.source_span


def _explicit_severity(rule: StaticContractRule, violated: bool) -> str:
    if not violated:
        return "info"
    return rule.severity


def _static_contract_metrics(
    config: VerificationConfig,
    prompt_segments: tuple[PromptSegmentArtifact, ...],
    truncation_configs: tuple[FrameworkTruncationConfigArtifact, ...],
    stop_policies: tuple[StopPolicyArtifact, ...],
) -> tuple[dict[str, int], str | None]:
    required = tuple(segment for artifact in prompt_segments for segment in artifact.segments if segment.required)
    unknown_required = tuple(segment.name for segment in required if segment.token_count is None and segment.content is None)
    if unknown_required:
        return {}, f"required prompt segments have unknown token counts: {', '.join(unknown_required)}"
    budget = truncation_configs[0] if truncation_configs else None
    max_context = budget.max_context_tokens if budget is not None and budget.max_context_tokens is not None else config.max_context_tokens
    if max_context is None:
        return {}, "no finite max_context_tokens or framework truncation budget is declared"
    reserved = 0
    if budget is not None:
        reserved = budget.reserve_output_tokens + budget.reserved_tool_tokens + budget.generation_prompt_tokens + budget.special_token_overhead
    required_tokens = sum((segment.token_count if segment.token_count is not None else len(segment.content or "")) + segment.overhead_tokens for segment in required)
    return {
        "required_prompt_tokens": required_tokens,
        "input_budget_tokens": max_context - reserved,
        "reserved_tokens": reserved,
        "max_context_tokens": max_context,
        "required_region_count": len(required),
        "prompt_segment_count": sum(len(artifact.segments) for artifact in prompt_segments),
        "stop_policy_count": len(stop_policies),
    }, None


def _resolve_static_contract_operand(operand: str, metrics: dict[str, int]) -> int | None:
    try:
        return int(operand)
    except ValueError:
        return metrics.get(operand)


def _invariant_violation_expression(invariant: StaticContractInvariant):
    if invariant.op == "<=":
        return Gt(Var("left_value"), Var("right_value"))
    if invariant.op == "<":
        return Ge(Var("left_value"), Var("right_value"))
    if invariant.op == ">=":
        return Lt(Var("left_value"), Var("right_value"))
    if invariant.op == ">":
        return Le(Var("left_value"), Var("right_value"))
    if invariant.op == "==":
        return Ne(Var("left_value"), Var("right_value"))
    if invariant.op == "!=":
        return Eq(Var("left_value"), Var("right_value"))
    raise ValueError(f"unsupported static contract invariant op: {invariant.op}")


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
