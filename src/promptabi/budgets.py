"""Context-window and token-budget modeling for PromptABI."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import (
    ArtifactKind,
    FrameworkTruncationConfigArtifact,
    PromptSegment,
    PromptSegmentArtifact,
    TokenizerArtifact,
)
from .config import VerificationConfig
from .loaders import LoadedArtifact
from .tokenizers import TokenizerAdapter, TokenizerError


@dataclass(frozen=True, slots=True)
class TokenBudgetReservation:
    """Reserved portions of a model context window before prompt input is packed."""

    max_context_tokens: int
    reserve_output_tokens: int = 0
    reserved_tool_tokens: int = 0
    generation_prompt_tokens: int = 0
    special_token_overhead: int = 0

    @property
    def reserved_total(self) -> int:
        return (
            self.reserve_output_tokens
            + self.reserved_tool_tokens
            + self.generation_prompt_tokens
            + self.special_token_overhead
        )

    @property
    def input_budget_tokens(self) -> int:
        return self.max_context_tokens - self.reserved_total

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("max_context_tokens", self.max_context_tokens),
            ("reserve_output_tokens", self.reserve_output_tokens),
            ("reserved_tool_tokens", self.reserved_tool_tokens),
            ("generation_prompt_tokens", self.generation_prompt_tokens),
            ("special_token_overhead", self.special_token_overhead),
            ("reserved_total", self.reserved_total),
            ("input_budget_tokens", self.input_budget_tokens),
        )


@dataclass(frozen=True, slots=True)
class TokenBudgetSegment:
    """A prompt segment with a tokenizer-relative or declared token count."""

    index: int
    name: str
    role: str | None
    required: bool
    token_count: int | None
    source: str
    max_tokens: int | None = None
    overhead_tokens: int = 0
    chunk_id: str | None = None
    document_id: str | None = None
    chunk_tokenizer: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    chunk_start: int | None = None
    chunk_end: int | None = None
    expected_overlap_tokens: int | None = None
    actual_overlap_tokens: int | None = None
    citation: str | None = None
    citation_required: bool = False
    metadata_tokens: int = 0
    template_overhead_tokens: int = 0
    retrieval_payload_limit_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.token_count is None:
            return None
        return self.token_count + self.overhead_tokens + self.metadata_tokens + self.template_overhead_tokens

    @property
    def is_retrieval_chunk(self) -> bool:
        return (
            self.chunk_id is not None
            or self.document_id is not None
            or self.chunk_tokenizer is not None
            or self.source_start is not None
            or self.source_end is not None
            or self.chunk_start is not None
            or self.chunk_end is not None
            or self.expected_overlap_tokens is not None
            or self.actual_overlap_tokens is not None
            or self.citation is not None
            or self.citation_required
            or self.metadata_tokens > 0
            or self.template_overhead_tokens > 0
            or self.retrieval_payload_limit_tokens is not None
        )


@dataclass(frozen=True, slots=True)
class TruncationPolicy:
    """A normalized framework policy for bounded prompt-packing simulation."""

    framework: str
    strategy: str
    preserve_system: bool = False
    preserve_tools: bool = False
    drop_roles: tuple[str, ...] = ()
    supported: bool = True
    source: str = "declared"

    def to_metadata(self) -> tuple[tuple[str, str], ...]:
        return (
            ("framework", self.framework),
            ("strategy", self.strategy),
            ("preserve_system", str(self.preserve_system)),
            ("preserve_tools", str(self.preserve_tools)),
            ("drop_roles", ", ".join(self.drop_roles) or "<none>"),
            ("source", self.source),
        )


@dataclass(frozen=True, slots=True)
class TruncationDecision:
    """The bounded result of applying one normalized truncation policy."""

    policy: TruncationPolicy
    kept_segments: tuple[TokenBudgetSegment, ...]
    dropped_segments: tuple[TokenBudgetSegment, ...]
    overflow_tokens: int
    unknown_segments: tuple[TokenBudgetSegment, ...] = ()

    @property
    def kept_tokens(self) -> int | None:
        if any(segment.total_tokens is None for segment in self.kept_segments):
            return None
        return sum(segment.total_tokens or 0 for segment in self.kept_segments)


@dataclass(frozen=True, slots=True)
class MustSurviveProof:
    """Bounded proof or counterexample for must-survive prompt segments."""

    status: str
    required_segments: tuple[str, ...]
    survived_segments: tuple[str, ...]
    dropped_segments: tuple[str, ...]
    input_budget_tokens: int
    policy: TruncationPolicy
    minimal_counterexample: tuple[TokenBudgetSegment, ...] = ()
    reason: str | None = None

    @property
    def counterexample_tokens(self) -> int | None:
        if not self.minimal_counterexample:
            return 0
        if any(segment.total_tokens is None for segment in self.minimal_counterexample):
            return None
        return sum(segment.total_tokens or 0 for segment in self.minimal_counterexample)

    def to_metadata(self) -> tuple[tuple[str, str], ...]:
        counterexample_tokens = self.counterexample_tokens
        return (
            ("must_survive_status", self.status),
            ("required_segments", ", ".join(self.required_segments) or "<none>"),
            ("survived_required", ", ".join(self.survived_segments) or "<none>"),
            ("dropped_required", ", ".join(self.dropped_segments) or "<none>"),
            (
                "minimal_counterexample",
                ", ".join(_segment_token_summary(segment) for segment in self.minimal_counterexample) or "<none>",
            ),
            (
                "minimal_counterexample_tokens",
                str(counterexample_tokens) if counterexample_tokens is not None else "unknown",
            ),
            ("input_budget_tokens", str(self.input_budget_tokens)),
            ("framework", self.policy.framework),
            ("strategy", self.policy.strategy),
            ("reason", self.reason or "<none>"),
        )


@dataclass(frozen=True, slots=True)
class TokenBudgetVisualizationRow:
    """One prompt-region row in a deterministic budget visualization."""

    index: int
    name: str
    role: str | None
    required: bool
    token_count: int | None
    overhead_tokens: int
    metadata_tokens: int
    template_overhead_tokens: int
    total_tokens: int | None
    source: str
    start_token: int | None
    end_token: int | None
    status: str
    survival: str

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "index": self.index,
            "name": self.name,
            "required": self.required,
            "overhead_tokens": self.overhead_tokens,
            "metadata_tokens": self.metadata_tokens,
            "template_overhead_tokens": self.template_overhead_tokens,
            "source": self.source,
            "status": self.status,
            "survival": self.survival,
        }
        if self.role is not None:
            data["role"] = self.role
        if self.token_count is not None:
            data["token_count"] = self.token_count
        if self.total_tokens is not None:
            data["total_tokens"] = self.total_tokens
        if self.start_token is not None:
            data["start_token"] = self.start_token
        if self.end_token is not None:
            data["end_token"] = self.end_token
        return data

    def compact_line(self) -> str:
        role = self.role or "unknown-role"
        total = "unknown" if self.total_tokens is None else str(self.total_tokens)
        if self.start_token is None or self.end_token is None:
            span = "?:?"
        else:
            span = f"{self.start_token}:{self.end_token}"
        requirement = "must-survive" if self.required else "optional"
        extras = []
        if self.overhead_tokens:
            extras.append(f"overhead={self.overhead_tokens}")
        if self.metadata_tokens:
            extras.append(f"metadata={self.metadata_tokens}")
        if self.template_overhead_tokens:
            extras.append(f"template={self.template_overhead_tokens}")
        extras.append(f"source={self.source}")
        return (
            f"{self.index}. {self.name} ({role}, {requirement}) "
            f"tokens={total} span={span} status={self.status} survival={self.survival}; "
            + ", ".join(extras)
        )


@dataclass(frozen=True, slots=True)
class TokenBudgetVisualization:
    """Stable prompt-budget visualization shared by text, JSON, and SARIF witnesses."""

    budget_source: str
    framework: str
    strategy: str
    max_context_tokens: int
    reserved_total: int
    input_budget_tokens: int
    total_prompt_tokens: int | None
    required_prompt_tokens: int | None
    overflow_tokens: int | None
    truncation_boundary_tokens: int | None
    must_survive_status: str
    rows: tuple[TokenBudgetVisualizationRow, ...]

    @property
    def dropped_fields(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows if row.status == "dropped")

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows if row.total_tokens is None)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "budget_source": self.budget_source,
            "framework": self.framework,
            "strategy": self.strategy,
            "max_context_tokens": self.max_context_tokens,
            "reserved_total": self.reserved_total,
            "input_budget_tokens": self.input_budget_tokens,
            "must_survive_status": self.must_survive_status,
            "rows": [row.to_dict() for row in self.rows],
            "dropped_fields": list(self.dropped_fields),
            "unknown_fields": list(self.unknown_fields),
        }
        if self.total_prompt_tokens is not None:
            data["total_prompt_tokens"] = self.total_prompt_tokens
        if self.required_prompt_tokens is not None:
            data["required_prompt_tokens"] = self.required_prompt_tokens
        if self.overflow_tokens is not None:
            data["overflow_tokens"] = self.overflow_tokens
        if self.truncation_boundary_tokens is not None:
            data["truncation_boundary_tokens"] = self.truncation_boundary_tokens
        return data

    def render_text(self) -> str:
        total = "unknown" if self.total_prompt_tokens is None else str(self.total_prompt_tokens)
        required = "unknown" if self.required_prompt_tokens is None else str(self.required_prompt_tokens)
        overflow = "unknown" if self.overflow_tokens is None else str(self.overflow_tokens)
        boundary = (
            "unknown"
            if self.truncation_boundary_tokens is None
            else str(self.truncation_boundary_tokens)
        )
        header = (
            f"budget={self.budget_source} framework={self.framework}:{self.strategy} "
            f"context={self.max_context_tokens} reserved={self.reserved_total} "
            f"input={self.input_budget_tokens} total={total} required={required} "
            f"overflow={overflow} boundary={boundary} must_survive={self.must_survive_status}"
        )
        dropped = ", ".join(self.dropped_fields) or "<none>"
        unknown = ", ".join(self.unknown_fields) or "<none>"
        rows = " | ".join(row.compact_line() for row in self.rows) or "<no prompt segments>"
        return f"{header}; dropped={dropped}; unknown={unknown}; rows: {rows}"


@dataclass(frozen=True, slots=True)
class TokenBudgetFinding:
    """A budget-model observation that should become a diagnostic."""

    rule_id: str
    severity: str
    message: str
    suggestion: str
    evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TokenBudgetReport:
    """The modeled context-window budget for one verification run."""

    budget_source: str | None
    framework: str | None
    strategy: str | None
    model: str | None
    reservation: TokenBudgetReservation | None
    policy: TruncationPolicy | None
    truncation: TruncationDecision | None
    must_survive_proof: MustSurviveProof | None
    segments: tuple[TokenBudgetSegment, ...]
    findings: tuple[TokenBudgetFinding, ...]
    visualization: TokenBudgetVisualization | None = None

    @property
    def known_segments(self) -> tuple[TokenBudgetSegment, ...]:
        return tuple(segment for segment in self.segments if segment.total_tokens is not None)

    @property
    def unknown_segments(self) -> tuple[TokenBudgetSegment, ...]:
        return tuple(segment for segment in self.segments if segment.total_tokens is None)

    @property
    def total_prompt_tokens(self) -> int | None:
        if self.unknown_segments:
            return None
        return sum(segment.total_tokens or 0 for segment in self.segments)

    @property
    def required_prompt_tokens(self) -> int | None:
        required = tuple(segment for segment in self.segments if segment.required)
        if any(segment.total_tokens is None for segment in required):
            return None
        return sum(segment.total_tokens or 0 for segment in required)


def analyze_token_budget(
    config: VerificationConfig,
    loaded_artifacts: tuple[LoadedArtifact, ...],
    *,
    tokenizers: tuple[tuple[TokenizerArtifact, TokenizerAdapter], ...] = (),
) -> TokenBudgetReport:
    """Build a bounded context-window model from declared artifacts and real tokenizers."""

    segment_artifacts = [
        loaded.artifact
        for loaded in loaded_artifacts
        if loaded.artifact.kind is ArtifactKind.PROMPT_SEGMENT and isinstance(loaded.artifact, PromptSegmentArtifact)
    ]
    budget_artifacts = [
        loaded.artifact
        for loaded in loaded_artifacts
        if loaded.artifact.kind is ArtifactKind.FRAMEWORK_TRUNCATION_CONFIG
        and isinstance(loaded.artifact, FrameworkTruncationConfigArtifact)
    ]
    findings: list[TokenBudgetFinding] = []
    if not segment_artifacts:
        return TokenBudgetReport(
            budget_source=None,
            framework=None,
            strategy=None,
            model=None,
            reservation=None,
            policy=None,
            truncation=None,
            must_survive_proof=None,
            segments=(),
            findings=(
                TokenBudgetFinding(
                    rule_id="token-budget-abstained",
                    severity="warning",
                    message="no prompt-segment artifact is available for token-budget modeling",
                    suggestion="Declare a prompt-segment artifact with named prompt regions and must-survive flags.",
                ),
            ),
        )

    segments = tuple(
        _segment_with_count(index, segment, tokenizers)
        for index, segment in enumerate(
            segment
            for artifact in sorted(segment_artifacts, key=lambda item: item.name)
            for segment in artifact.segments
        )
    )
    budget = _select_budget(config, budget_artifacts, findings)
    if budget is None:
        return TokenBudgetReport(
            budget_source=None,
            framework=None,
            strategy=None,
            model=None,
            reservation=None,
            policy=None,
            truncation=None,
            must_survive_proof=None,
            segments=segments,
            findings=(
                *findings,
                TokenBudgetFinding(
                    rule_id="token-budget-abstained",
                    severity="warning",
                    message="no max_context_tokens value is available for token-budget modeling",
                    suggestion="Set max_context_tokens in the config or add a framework-truncation-config artifact.",
                ),
            ),
        )

    budget_source, budget_artifact, reservation = budget
    policy = _normalize_policy(budget_artifact)
    if reservation.input_budget_tokens <= 0:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-invalid",
                severity="error",
                message=(
                    "reserved output, tool, generation-prompt, and special-token overhead "
                    "consume the entire context window"
                ),
                suggestion="Reduce reserved tokens or increase the declared model context window.",
                evidence=_reservation_evidence(reservation),
            )
        )

    unknown_required = tuple(segment for segment in segments if segment.required and segment.total_tokens is None)
    if unknown_required:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-abstained",
                severity="warning",
                message="one or more required prompt segments lack a declared or tokenizer-derived token count",
                suggestion="Add token_count values or include a supported tokenizer and segment content.",
                evidence=(("segments", ", ".join(segment.name for segment in unknown_required)),),
            )
        )

    for segment in sorted(segments, key=lambda item: item.name):
        if segment.total_tokens is not None and segment.total_tokens > reservation.input_budget_tokens:
            findings.append(
                TokenBudgetFinding(
                    rule_id="token-budget-segment-overflow",
                    severity="error" if segment.required else "warning",
                    message=(
                        f"prompt segment '{segment.name}' requires {segment.total_tokens} token(s), "
                        f"exceeding the modeled input budget of {reservation.input_budget_tokens}"
                    ),
                    suggestion="Shorten that segment, move part of it outside the prompt, or increase the context budget.",
                    evidence=(
                        ("segment", segment.name),
                        ("required", str(segment.required)),
                        ("token_source", segment.source),
                    ),
                )
            )

    required_total = sum(segment.total_tokens or 0 for segment in segments if segment.required)
    if not unknown_required and required_total > reservation.input_budget_tokens:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-required-overflow",
                severity="error",
                message=(
                    f"required prompt segments need {required_total} token(s), exceeding "
                    f"the modeled input budget of {reservation.input_budget_tokens}"
                ),
                suggestion="Lower must-survive prompt budgets or reserve fewer context-window tokens.",
                evidence=(
                    ("required_segments", ", ".join(segment.name for segment in segments if segment.required)),
                    ("required_tokens", str(required_total)),
                    ("input_budget_tokens", str(reservation.input_budget_tokens)),
                ),
            )
        )

    if not any(segment.total_tokens is None for segment in segments):
        total_tokens = sum(segment.total_tokens or 0 for segment in segments)
        if total_tokens > reservation.input_budget_tokens:
            findings.append(
                TokenBudgetFinding(
                    rule_id="token-budget-total-overflow",
                    severity="warning",
                    message=(
                        f"all modeled prompt segments need {total_tokens} token(s), exceeding "
                        f"the modeled input budget of {reservation.input_budget_tokens}"
                    ),
                    suggestion="Add an explicit truncation policy before relying on optional segments fitting.",
                    evidence=(
                        ("total_prompt_tokens", str(total_tokens)),
                        ("input_budget_tokens", str(reservation.input_budget_tokens)),
                    ),
                )
            )

    truncation = _apply_truncation_policy(policy, segments, reservation)
    must_survive_proof = _prove_must_survive(policy, segments, reservation, truncation)
    if truncation.unknown_segments:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-truncation-abstained",
                severity="warning",
                message="the framework truncation policy cannot be simulated because some prompt segments lack token counts",
                suggestion="Add token_count/max_tokens values or include a supported tokenizer and segment content.",
                evidence=(
                    ("framework", policy.framework),
                    ("strategy", policy.strategy),
                    ("segments", ", ".join(segment.name for segment in truncation.unknown_segments)),
                ),
            )
        )
    elif policy.strategy == "none" and truncation.overflow_tokens > 0:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-policy-overflow",
                severity="error",
                message=(
                    f"framework policy '{policy.framework}' declares no truncation, but the prompt exceeds "
                    f"the input budget by {truncation.overflow_tokens} token(s)"
                ),
                suggestion="Choose an explicit framework truncation strategy or reduce prompt/reservation tokens.",
                evidence=(
                    ("framework", policy.framework),
                    ("strategy", policy.strategy),
                    ("overflow_tokens", str(truncation.overflow_tokens)),
                ),
            )
        )
    else:
        dropped_required = tuple(segment for segment in truncation.dropped_segments if segment.required)
        if dropped_required:
            findings.append(
                TokenBudgetFinding(
                    rule_id="token-budget-required-truncated",
                    severity="error",
                    message=(
                        f"framework policy '{policy.framework}:{policy.strategy}' can drop required "
                        f"prompt segment(s): {', '.join(segment.name for segment in dropped_required)}"
                    ),
                    suggestion="Mark required segments as preserved, change truncation strategy, or reduce earlier context.",
                    evidence=(
                        ("framework", policy.framework),
                        ("strategy", policy.strategy),
                        ("dropped_required", ", ".join(segment.name for segment in dropped_required)),
                        ("kept_segments", ", ".join(segment.name for segment in truncation.kept_segments) or "<none>"),
                        ("dropped_segments", ", ".join(segment.name for segment in truncation.dropped_segments) or "<none>"),
                        *(
                            must_survive_proof.to_metadata()
                            if must_survive_proof is not None
                            else ()
                        ),
                    ),
                )
            )
        elif truncation.dropped_segments:
            findings.append(
                TokenBudgetFinding(
                    rule_id="token-budget-framework-truncation",
                    severity="info",
                    message=(
                        f"framework policy '{policy.framework}:{policy.strategy}' drops optional "
                        f"prompt segment(s): {', '.join(segment.name for segment in truncation.dropped_segments)}"
                    ),
                    suggestion="Review optional segment loss if retrieval/citation quality depends on those regions.",
                    evidence=(
                        ("framework", policy.framework),
                        ("strategy", policy.strategy),
                        ("kept_segments", ", ".join(segment.name for segment in truncation.kept_segments) or "<none>"),
                        ("dropped_segments", ", ".join(segment.name for segment in truncation.dropped_segments) or "<none>"),
                    ),
                )
            )

    findings.extend(_rag_chunking_findings(segments, truncation, tokenizers))

    report = TokenBudgetReport(
        budget_source=budget_source,
        framework=budget_artifact.framework if budget_artifact is not None else "config",
        strategy=budget_artifact.strategy.value if budget_artifact is not None else "none",
        model=budget_artifact.model if budget_artifact is not None else None,
        reservation=reservation,
        policy=policy,
        truncation=truncation,
        must_survive_proof=must_survive_proof,
        segments=segments,
        findings=tuple(findings),
    )
    return TokenBudgetReport(
        budget_source=report.budget_source,
        framework=report.framework,
        strategy=report.strategy,
        model=report.model,
        reservation=report.reservation,
        policy=report.policy,
        truncation=report.truncation,
        must_survive_proof=report.must_survive_proof,
        segments=report.segments,
        findings=report.findings,
        visualization=build_token_budget_visualization(report),
    )


def build_token_budget_visualization(report: TokenBudgetReport) -> TokenBudgetVisualization | None:
    """Render prompt budget arithmetic and truncation decisions as stable data."""

    if report.reservation is None:
        return None
    truncation = report.truncation
    policy = report.policy
    dropped_names = {segment.name for segment in truncation.dropped_segments} if truncation is not None else set()
    unknown_names = {segment.name for segment in truncation.unknown_segments} if truncation is not None else set()
    proof = report.must_survive_proof
    proof_dropped = set(proof.dropped_segments) if proof is not None else set()
    proof_survived = set(proof.survived_segments) if proof is not None else set()
    rows: list[TokenBudgetVisualizationRow] = []
    running_start: int | None = 0
    for segment in sorted(report.segments, key=lambda item: item.index):
        total = segment.total_tokens
        if running_start is None or total is None:
            start_token = None
            end_token = None
            running_start = None
        else:
            start_token = running_start
            end_token = running_start + total
            running_start = end_token
        rows.append(
            TokenBudgetVisualizationRow(
                index=segment.index,
                name=segment.name,
                role=segment.role,
                required=segment.required,
                token_count=segment.token_count,
                overhead_tokens=segment.overhead_tokens,
                metadata_tokens=segment.metadata_tokens,
                template_overhead_tokens=segment.template_overhead_tokens,
                total_tokens=total,
                source=segment.source,
                start_token=start_token,
                end_token=end_token,
                status=_visualization_status(segment.name, dropped_names, unknown_names, total),
                survival=_visualization_survival(segment, dropped_names, proof_dropped, proof_survived, proof),
            )
        )
    overflow = truncation.overflow_tokens if truncation is not None else None
    return TokenBudgetVisualization(
        budget_source=report.budget_source or "missing",
        framework=policy.framework if policy is not None else report.framework or "config",
        strategy=policy.strategy if policy is not None else report.strategy or "none",
        max_context_tokens=report.reservation.max_context_tokens,
        reserved_total=report.reservation.reserved_total,
        input_budget_tokens=report.reservation.input_budget_tokens,
        total_prompt_tokens=report.total_prompt_tokens,
        required_prompt_tokens=report.required_prompt_tokens,
        overflow_tokens=overflow,
        truncation_boundary_tokens=report.reservation.input_budget_tokens,
        must_survive_status=proof.status if proof is not None else "not-modeled",
        rows=tuple(rows),
    )


def _visualization_status(
    name: str,
    dropped_names: set[str],
    unknown_names: set[str],
    total_tokens: int | None,
) -> str:
    if total_tokens is None or name in unknown_names:
        return "unknown"
    if name in dropped_names:
        return "dropped"
    return "kept"


def _visualization_survival(
    segment: TokenBudgetSegment,
    dropped_names: set[str],
    proof_dropped: set[str],
    proof_survived: set[str],
    proof: MustSurviveProof | None,
) -> str:
    if not segment.required:
        return "optional-dropped" if segment.name in dropped_names else "optional-kept"
    if proof is None:
        return "not-modeled"
    if proof.status == "abstained":
        return "abstained"
    if segment.name in proof_dropped:
        return "violated"
    if segment.name in proof_survived:
        return "guaranteed"
    return proof.status


def _select_budget(
    config: VerificationConfig,
    budget_artifacts: list[FrameworkTruncationConfigArtifact],
    findings: list[TokenBudgetFinding],
) -> tuple[str, FrameworkTruncationConfigArtifact | None, TokenBudgetReservation] | None:
    artifact = budget_artifacts[0] if budget_artifacts else None
    if len(budget_artifacts) > 1:
        findings.append(
            TokenBudgetFinding(
                rule_id="token-budget-context-conflict",
                severity="warning",
                message="multiple framework-truncation-config artifacts are declared; using the first deterministic artifact",
                suggestion="Run separate PromptABI configs when comparing multiple context-window policies.",
                evidence=(("artifacts", ", ".join(artifact.name for artifact in budget_artifacts)),),
            )
        )
    if artifact is not None and artifact.max_context_tokens is not None:
        if config.max_context_tokens is not None and config.max_context_tokens != artifact.max_context_tokens:
            findings.append(
                TokenBudgetFinding(
                    rule_id="token-budget-context-conflict",
                    severity="warning",
                    message=(
                        f"framework budget '{artifact.name}' max_context_tokens={artifact.max_context_tokens} "
                        f"overrides config max_context_tokens={config.max_context_tokens}"
                    ),
                    suggestion="Keep the config and framework budget artifact aligned to avoid ambiguous CI results.",
                    evidence=(
                        ("config.max_context_tokens", str(config.max_context_tokens)),
                        (f"artifacts.{artifact.name}.max_context_tokens", str(artifact.max_context_tokens)),
                    ),
                )
            )
        return (
            artifact.name,
            artifact,
            TokenBudgetReservation(
                max_context_tokens=artifact.max_context_tokens,
                reserve_output_tokens=artifact.reserve_output_tokens,
                reserved_tool_tokens=artifact.reserved_tool_tokens,
                generation_prompt_tokens=artifact.generation_prompt_tokens,
                special_token_overhead=artifact.special_token_overhead,
            ),
        )
    if config.max_context_tokens is None:
        return None
    return (
        "config.max_context_tokens",
        None,
        TokenBudgetReservation(max_context_tokens=config.max_context_tokens),
    )


def _segment_with_count(
    index: int,
    segment: PromptSegment,
    tokenizers: tuple[tuple[TokenizerArtifact, TokenizerAdapter], ...],
) -> TokenBudgetSegment:
    if segment.token_count is not None:
        return TokenBudgetSegment(
            index=index,
            name=segment.name,
            role=segment.role,
            required=segment.required,
            token_count=segment.token_count,
            max_tokens=segment.max_tokens,
            overhead_tokens=segment.overhead_tokens,
            source="declared token_count",
            **_segment_chunk_fields(segment),
        )
    if segment.max_tokens is not None:
        return TokenBudgetSegment(
            index=index,
            name=segment.name,
            role=segment.role,
            required=segment.required,
            token_count=segment.max_tokens,
            max_tokens=segment.max_tokens,
            overhead_tokens=segment.overhead_tokens,
            source="declared max_tokens",
            **_segment_chunk_fields(segment),
        )
    if segment.content is not None and tokenizers:
        tokenizer_artifact, tokenizer = tokenizers[0]
        try:
            count = len(tokenizer.encode(segment.content, add_special_tokens=False).tokens)
        except TokenizerError:
            count = None
        if count is not None:
            return TokenBudgetSegment(
                index=index,
                name=segment.name,
                role=segment.role,
                required=segment.required,
                token_count=count,
                max_tokens=segment.max_tokens,
                overhead_tokens=segment.overhead_tokens,
                source=f"tokenizer:{tokenizer_artifact.name}",
                **_segment_chunk_fields(segment),
            )
    return TokenBudgetSegment(
        index=index,
        name=segment.name,
        role=segment.role,
        required=segment.required,
        token_count=None,
        max_tokens=segment.max_tokens,
        overhead_tokens=segment.overhead_tokens,
        source="missing token count",
        **_segment_chunk_fields(segment),
    )


def _segment_chunk_fields(segment: PromptSegment) -> dict[str, object]:
    return {
        "chunk_id": segment.chunk_id,
        "document_id": segment.document_id,
        "chunk_tokenizer": segment.chunk_tokenizer,
        "source_start": segment.source_start,
        "source_end": segment.source_end,
        "chunk_start": segment.chunk_start,
        "chunk_end": segment.chunk_end,
        "expected_overlap_tokens": segment.expected_overlap_tokens,
        "actual_overlap_tokens": segment.actual_overlap_tokens,
        "citation": segment.citation,
        "citation_required": segment.citation_required,
        "metadata_tokens": segment.metadata_tokens,
        "template_overhead_tokens": segment.template_overhead_tokens,
        "retrieval_payload_limit_tokens": segment.retrieval_payload_limit_tokens,
    }


def _rag_chunking_findings(
    segments: tuple[TokenBudgetSegment, ...],
    truncation: TruncationDecision,
    tokenizers: tuple[tuple[TokenizerArtifact, TokenizerAdapter], ...],
) -> tuple[TokenBudgetFinding, ...]:
    chunks = tuple(segment for segment in segments if segment.is_retrieval_chunk)
    if not chunks:
        return ()
    findings: list[TokenBudgetFinding] = []
    tokenizer_names = {
        value
        for artifact, _tokenizer in tokenizers
        for value in (artifact.name, artifact.family)
        if value is not None
    }
    dropped_names = {segment.name for segment in truncation.dropped_segments}
    for chunk in chunks:
        if chunk.chunk_tokenizer is not None and tokenizer_names and chunk.chunk_tokenizer not in tokenizer_names:
            selected = ", ".join(sorted(tokenizer_names))
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-tokenizer-mismatch",
                    severity="warning",
                    message=(
                        f"retrieval chunk '{chunk.name}' was budgeted with tokenizer "
                        f"'{chunk.chunk_tokenizer}', but verification loaded {selected}"
                    ),
                    suggestion="Recompute retrieval chunk token counts with the serving tokenizer revision.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("chunk_tokenizer", chunk.chunk_tokenizer),
                        ("selected_tokenizers", selected),
                    ),
                )
            )
        if _has_boundary_drift(chunk):
            assert chunk.source_start is not None and chunk.source_end is not None
            assert chunk.chunk_start is not None and chunk.chunk_end is not None
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-chunk-boundary-drift",
                    severity="warning",
                    message=(
                        f"retrieval chunk '{chunk.name}' declares source boundary "
                        f"{chunk.source_start}:{chunk.source_end} but packed boundary "
                        f"{chunk.chunk_start}:{chunk.chunk_end}"
                    ),
                    suggestion="Regenerate chunks after tokenizer, splitter, or document-normalization changes.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("source_boundary", f"{chunk.source_start}:{chunk.source_end}"),
                        ("packed_boundary", f"{chunk.chunk_start}:{chunk.chunk_end}"),
                    ),
                )
            )
        if chunk.citation_required and not chunk.citation:
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-citation-loss",
                    severity="error",
                    message=f"retrieval chunk '{chunk.name}' is citation-required but has no citation label",
                    suggestion="Attach stable citation metadata before rendering retrieved context.",
                    evidence=(("chunk", chunk.name), ("citation", "<missing>")),
                )
            )
        if chunk.citation_required and chunk.name in dropped_names:
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-citation-loss",
                    severity="error",
                    message=f"framework truncation can drop citation-required retrieval chunk '{chunk.name}'",
                    suggestion="Reserve retrieval payload budget or make citation-bearing chunks must-survive.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("citation", chunk.citation or "<missing>"),
                        ("dropped_segments", ", ".join(segment.name for segment in truncation.dropped_segments)),
                    ),
                )
            )
        if chunk.retrieval_payload_limit_tokens is not None and chunk.total_tokens is not None:
            if chunk.total_tokens > chunk.retrieval_payload_limit_tokens:
                findings.append(
                    TokenBudgetFinding(
                        rule_id="rag-payload-truncation",
                        severity="error" if chunk.citation_required else "warning",
                        message=(
                            f"retrieval chunk '{chunk.name}' needs {chunk.total_tokens} token(s), "
                            f"exceeding its payload limit of {chunk.retrieval_payload_limit_tokens}"
                        ),
                        suggestion="Lower chunk size, metadata, or template overhead, or raise the retrieval payload limit.",
                        evidence=(
                            ("chunk", chunk.name),
                            ("total_tokens", str(chunk.total_tokens)),
                            ("retrieval_payload_limit_tokens", str(chunk.retrieval_payload_limit_tokens)),
                        ),
                    )
                )
        if chunk.name in dropped_names:
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-payload-truncation",
                    severity="warning",
                    message=f"framework truncation can remove retrieval chunk '{chunk.name}' from the prompt payload",
                    suggestion="Reduce retrieval tokens or configure the framework to preserve required retrieval payloads.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("policy", f"{truncation.policy.framework}:{truncation.policy.strategy}"),
                    ),
                )
            )
        if chunk.metadata_tokens and chunk.token_count is not None and chunk.metadata_tokens > max(8, chunk.token_count // 2):
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-metadata-inflation",
                    severity="warning",
                    message=(
                        f"retrieval chunk '{chunk.name}' spends {chunk.metadata_tokens} metadata token(s) "
                        f"for {chunk.token_count} content token(s)"
                    ),
                    suggestion="Compress metadata keys, omit unused fields, or account for metadata outside retrieved content.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("content_tokens", str(chunk.token_count)),
                        ("metadata_tokens", str(chunk.metadata_tokens)),
                    ),
                )
            )
        if _template_overhead_exceeds_chunk_budget(chunk):
            assert chunk.max_tokens is not None
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-template-overhead",
                    severity="warning",
                    message=(
                        f"retrieval chunk '{chunk.name}' exceeds max_tokens={chunk.max_tokens} after "
                        f"metadata and prompt-template overhead are included"
                    ),
                    suggestion="Include retrieval wrapper text in chunk sizing or reduce per-chunk template overhead.",
                    evidence=(
                        ("chunk", chunk.name),
                        ("max_tokens", str(chunk.max_tokens)),
                        ("content_tokens", str(chunk.token_count) if chunk.token_count is not None else "unknown"),
                        ("metadata_tokens", str(chunk.metadata_tokens)),
                        ("template_overhead_tokens", str(chunk.template_overhead_tokens)),
                    ),
                )
            )
    findings.extend(_rag_overlap_findings(chunks))
    return tuple(findings)


def _has_boundary_drift(chunk: TokenBudgetSegment) -> bool:
    return (
        chunk.source_start is not None
        and chunk.source_end is not None
        and chunk.chunk_start is not None
        and chunk.chunk_end is not None
        and (chunk.source_start != chunk.chunk_start or chunk.source_end != chunk.chunk_end)
    )


def _template_overhead_exceeds_chunk_budget(chunk: TokenBudgetSegment) -> bool:
    if chunk.max_tokens is None or chunk.token_count is None:
        return False
    return chunk.token_count + chunk.metadata_tokens + chunk.template_overhead_tokens > chunk.max_tokens


def _rag_overlap_findings(chunks: tuple[TokenBudgetSegment, ...]) -> tuple[TokenBudgetFinding, ...]:
    findings: list[TokenBudgetFinding] = []
    by_document: dict[str, list[TokenBudgetSegment]] = {}
    for chunk in chunks:
        if chunk.document_id is not None:
            by_document.setdefault(chunk.document_id, []).append(chunk)
    for document_id, document_chunks in sorted(by_document.items()):
        ordered = sorted(
            document_chunks,
            key=lambda chunk: (
                chunk.source_start if chunk.source_start is not None else chunk.index,
                chunk.index,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            expected = current.expected_overlap_tokens
            if expected is None:
                continue
            actual = _actual_overlap(previous, current)
            if actual is None or actual == expected:
                continue
            findings.append(
                TokenBudgetFinding(
                    rule_id="rag-overlap-accounting",
                    severity="warning",
                    message=(
                        f"retrieval chunks '{previous.name}' and '{current.name}' for document "
                        f"'{document_id}' have overlap {actual}, expected {expected}"
                    ),
                    suggestion="Recompute chunk overlaps with the same splitter and tokenizer used at serving time.",
                    evidence=(
                        ("document_id", document_id),
                        ("previous_chunk", previous.name),
                        ("current_chunk", current.name),
                        ("expected_overlap_tokens", str(expected)),
                        ("actual_overlap_tokens", str(actual)),
                    ),
                )
            )
    return tuple(findings)


def _actual_overlap(previous: TokenBudgetSegment, current: TokenBudgetSegment) -> int | None:
    if current.actual_overlap_tokens is not None:
        return current.actual_overlap_tokens
    if previous.source_end is None or current.source_start is None:
        return None
    return max(0, previous.source_end - current.source_start)


def _reservation_evidence(reservation: TokenBudgetReservation) -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value)) for key, value in reservation.to_metadata())


def _normalize_policy(artifact: FrameworkTruncationConfigArtifact | None) -> TruncationPolicy:
    if artifact is None:
        return TruncationPolicy(framework="config", strategy="none", source="config.max_context_tokens")
    framework = _canonical_framework(artifact.framework)
    strategy = artifact.strategy.value
    preserve_system = artifact.preserve_system
    preserve_tools = artifact.preserve_tools
    drop_roles = artifact.drop_roles
    source = "declared"
    if strategy == "none":
        strategy, preserve_system, preserve_tools, source = _framework_default_policy(
            framework,
            preserve_system=preserve_system,
            preserve_tools=preserve_tools,
        )
    return TruncationPolicy(
        framework=framework,
        strategy=strategy,
        preserve_system=preserve_system,
        preserve_tools=preserve_tools,
        drop_roles=drop_roles,
        supported=strategy != "custom",
        source=source,
    )


def _canonical_framework(framework: str) -> str:
    normalized = framework.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "openai": "openai-compatible",
        "openai-compatible-server": "openai-compatible",
        "openai-compatible-servers": "openai-compatible",
        "hf-transformers": "transformers",
        "huggingface-transformers": "transformers",
        "llama-cpp": "llama.cpp",
        "llamacpp": "llama.cpp",
        "custom-rag-pipeline": "custom-rag",
        "custom-rag-pipelines": "custom-rag",
        "message-dropping": "message-dropping",
        "message-dropping-strategy": "message-dropping",
    }
    return aliases.get(normalized, normalized)


def _framework_default_policy(
    framework: str,
    *,
    preserve_system: bool,
    preserve_tools: bool,
) -> tuple[str, bool, bool, str]:
    if framework in {"langchain", "llamaindex", "llama-index"}:
        return "oldest-message", True if not preserve_system else preserve_system, preserve_tools, "framework-default"
    if framework in {"vllm", "transformers", "llama.cpp", "ollama", "openai-compatible", "litellm"}:
        return "left", preserve_system, preserve_tools, "framework-default"
    if framework in {"custom-rag", "rag"}:
        return "priority", True if not preserve_system else preserve_system, preserve_tools, "framework-default"
    if framework == "message-dropping":
        return "oldest-message", preserve_system, preserve_tools, "framework-default"
    return "none", preserve_system, preserve_tools, "declared"


def _apply_truncation_policy(
    policy: TruncationPolicy,
    segments: tuple[TokenBudgetSegment, ...],
    reservation: TokenBudgetReservation,
) -> TruncationDecision:
    unknown = tuple(segment for segment in segments if segment.total_tokens is None)
    if unknown or not policy.supported:
        return TruncationDecision(
            policy=policy,
            kept_segments=segments,
            dropped_segments=(),
            overflow_tokens=0,
            unknown_segments=unknown,
        )
    total = sum(segment.total_tokens or 0 for segment in segments)
    overflow = max(0, total - reservation.input_budget_tokens)
    if overflow <= 0:
        return TruncationDecision(policy=policy, kept_segments=segments, dropped_segments=(), overflow_tokens=0)
    if policy.strategy == "none":
        return TruncationDecision(
            policy=policy,
            kept_segments=segments,
            dropped_segments=(),
            overflow_tokens=overflow,
        )
    if policy.strategy in {"left", "oldest-message", "sliding-window"}:
        kept, dropped = _drop_until_fits(
            segments,
            reservation.input_budget_tokens,
            candidates=segments,
            keep_system=policy.preserve_system or policy.strategy == "oldest-message",
            keep_tools=policy.preserve_tools,
        )
    elif policy.strategy == "right":
        kept, dropped = _drop_until_fits(
            segments,
            reservation.input_budget_tokens,
            candidates=tuple(reversed(segments)),
            keep_system=policy.preserve_system,
            keep_tools=policy.preserve_tools,
        )
    elif policy.strategy == "middle":
        kept, dropped = _drop_until_fits(
            segments,
            reservation.input_budget_tokens,
            candidates=_middle_out_candidates(segments),
            keep_system=policy.preserve_system,
            keep_tools=policy.preserve_tools,
        )
    elif policy.strategy == "priority":
        role_candidates = tuple(
            segment
            for role in policy.drop_roles
            for segment in segments
            if (segment.role or "").lower() == role.lower()
        )
        candidates = role_candidates or tuple(segment for segment in segments if not segment.required)
        kept, dropped = _drop_until_fits(
            segments,
            reservation.input_budget_tokens,
            candidates=candidates,
            keep_system=policy.preserve_system,
            keep_tools=policy.preserve_tools,
        )
    else:
        kept, dropped = segments, ()
    return TruncationDecision(policy=policy, kept_segments=kept, dropped_segments=dropped, overflow_tokens=overflow)


def _prove_must_survive(
    policy: TruncationPolicy,
    segments: tuple[TokenBudgetSegment, ...],
    reservation: TokenBudgetReservation,
    truncation: TruncationDecision,
) -> MustSurviveProof:
    required = tuple(segment for segment in segments if segment.required)
    required_names = tuple(segment.name for segment in required)
    if not required:
        return MustSurviveProof(
            status="proven",
            required_segments=(),
            survived_segments=(),
            dropped_segments=(),
            input_budget_tokens=reservation.input_budget_tokens,
            policy=policy,
            reason="no must-survive prompt segments declared",
        )
    if truncation.unknown_segments or not policy.supported:
        return MustSurviveProof(
            status="abstained",
            required_segments=required_names,
            survived_segments=(),
            dropped_segments=(),
            input_budget_tokens=reservation.input_budget_tokens,
            policy=policy,
            reason="segment token counts or truncation policy are outside the bounded model",
        )
    dropped_required = tuple(segment for segment in truncation.dropped_segments if segment.required)
    if not dropped_required:
        return MustSurviveProof(
            status="proven",
            required_segments=required_names,
            survived_segments=tuple(segment.name for segment in required),
            dropped_segments=(),
            input_budget_tokens=reservation.input_budget_tokens,
            policy=policy,
            reason="simulated truncation keeps every must-survive segment",
        )
    counterexample = _minimize_survival_counterexample(
        policy,
        segments,
        reservation,
        target_names=tuple(segment.name for segment in dropped_required),
    )
    dropped_names = tuple(segment.name for segment in dropped_required)
    survived_names = tuple(segment.name for segment in required if segment.name not in dropped_names)
    return MustSurviveProof(
        status="violated",
        required_segments=required_names,
        survived_segments=survived_names,
        dropped_segments=dropped_names,
        input_budget_tokens=reservation.input_budget_tokens,
        policy=policy,
        minimal_counterexample=counterexample,
        reason="framework truncation can remove a must-survive segment",
    )


def _minimize_survival_counterexample(
    policy: TruncationPolicy,
    segments: tuple[TokenBudgetSegment, ...],
    reservation: TokenBudgetReservation,
    *,
    target_names: tuple[str, ...],
) -> tuple[TokenBudgetSegment, ...]:
    counterexample = list(segments)
    target_name_set = set(target_names)
    for candidate in segments:
        if candidate.name in target_name_set:
            continue
        trial = tuple(segment for segment in counterexample if segment != candidate)
        trial_decision = _apply_truncation_policy(policy, trial, reservation)
        trial_dropped = {
            segment.name for segment in trial_decision.dropped_segments if segment.required
        }
        if trial_dropped.intersection(target_name_set):
            counterexample = list(trial)
    return tuple(sorted(counterexample, key=lambda segment: segment.index))


def _drop_until_fits(
    segments: tuple[TokenBudgetSegment, ...],
    budget: int,
    *,
    candidates: tuple[TokenBudgetSegment, ...],
    keep_system: bool,
    keep_tools: bool,
) -> tuple[tuple[TokenBudgetSegment, ...], tuple[TokenBudgetSegment, ...]]:
    dropped: list[TokenBudgetSegment] = []
    kept = list(segments)
    total = sum(segment.total_tokens or 0 for segment in kept)
    for candidate in candidates:
        if total <= budget:
            break
        if _is_preserved(candidate, keep_system=keep_system, keep_tools=keep_tools):
            continue
        if candidate not in kept:
            continue
        kept.remove(candidate)
        dropped.append(candidate)
        total -= candidate.total_tokens or 0
    return tuple(sorted(kept, key=lambda segment: segment.index)), tuple(sorted(dropped, key=lambda segment: segment.index))


def _is_preserved(segment: TokenBudgetSegment, *, keep_system: bool, keep_tools: bool) -> bool:
    role = (segment.role or "").lower()
    return (keep_system and role in {"system", "developer"}) or (keep_tools and role in {"tool", "function"})


def _middle_out_candidates(segments: tuple[TokenBudgetSegment, ...]) -> tuple[TokenBudgetSegment, ...]:
    if not segments:
        return ()
    center = (len(segments) - 1) / 2
    return tuple(sorted(segments, key=lambda segment: (abs(segment.index - center), segment.index)))


def _segment_token_summary(segment: TokenBudgetSegment) -> str:
    tokens = segment.total_tokens
    suffix = "unknown" if tokens is None else str(tokens)
    return f"{segment.name}={suffix}"
