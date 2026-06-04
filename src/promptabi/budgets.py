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
    overhead_tokens: int = 0

    @property
    def total_tokens(self) -> int | None:
        if self.token_count is None:
            return None
        return self.token_count + self.overhead_tokens


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
    segments: tuple[TokenBudgetSegment, ...]
    findings: tuple[TokenBudgetFinding, ...]

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

    return TokenBudgetReport(
        budget_source=budget_source,
        framework=budget_artifact.framework if budget_artifact is not None else "config",
        strategy=budget_artifact.strategy.value if budget_artifact is not None else "none",
        model=budget_artifact.model if budget_artifact is not None else None,
        reservation=reservation,
        policy=policy,
        truncation=truncation,
        segments=segments,
        findings=tuple(findings),
    )


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
            overhead_tokens=segment.overhead_tokens,
            source="declared token_count",
        )
    if segment.max_tokens is not None:
        return TokenBudgetSegment(
            index=index,
            name=segment.name,
            role=segment.role,
            required=segment.required,
            token_count=segment.max_tokens,
            overhead_tokens=segment.overhead_tokens,
            source="declared max_tokens",
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
                overhead_tokens=segment.overhead_tokens,
                source=f"tokenizer:{tokenizer_artifact.name}",
            )
    return TokenBudgetSegment(
        index=index,
        name=segment.name,
        role=segment.role,
        required=segment.required,
        token_count=None,
        overhead_tokens=segment.overhead_tokens,
        source="missing token count",
    )


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
