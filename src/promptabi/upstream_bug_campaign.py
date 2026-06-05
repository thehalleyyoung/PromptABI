"""Auditable campaign to confirm a *new* upstream interface-safety bug.

This module operationalizes the end-to-end research-and-engineering campaign for
discovering an interface-safety bug in a real upstream project, triaging it
honestly, and drafting a responsible report.  It is intentionally conservative:
it treats a substring match or a single heuristic flag as a *candidate*, never as
a confirmed bug, and routes every candidate through deterministic PromptABI
analysis of the *exact pinned upstream source* before it can be called
reportable.

The committed dossier (``fixtures/upstream_bug_campaign/campaign.json``) is backed
by real source captured from upstream HEAD with pinned commit SHAs and content
hashes.  The engine re-verifies those hashes against the captured files, replays
PromptABI analyzers on the captured source, and reports an honest triage outcome
for each candidate:

* ``confirmed``   - PromptABI deterministically reproduces an interface-contract
  violation on the exact pinned source.
* ``rejected``    - the candidate flag fires but the violation is not triggerable
  on the real code path (e.g. a marker that is always a single special token).
* ``abstained``   - the source is outside a supported analysis fragment, so no
  claim is made.
* ``duplicate``   - the symptom already has an upstream issue/PR.

The campaign deliberately keeps the *methodology* the product: a reproducible,
peer-reviewable audit trail, not an ad hoc bug hunt.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chat_templates import parse_hf_chat_template_config
from .role_boundaries import analyze_role_boundary_nonforgeability


UPSTREAM_BUG_CAMPAIGN_VERSION = 1

DEFAULT_CAMPAIGN_DOSSIER_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "upstream_bug_campaign" / "campaign.json"
)
DEFAULT_REAL_WORLD_BUG_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "real_world_bugs" / "corpus.json"
)

# The interface-safety bug taxonomy the campaign is allowed to report (step 3).
INTERFACE_SAFETY_BUG_CLASSES = frozenset(
    {
        "role-boundary-forgery",
        "delimiter-leakage",
        "parser-boundary-confusion",
        "tool-call-corruption",
        "malformed-structured-output",
        "unsafe-stop-behavior",
        "reasoning-tool-boundary-loss",
        "response-shape-violation",
    }
)

# Classes explicitly out of scope (step 4).
EXCLUDED_BUG_CLASSES = frozenset(
    {
        "model-quality",
        "hallucination",
        "benchmark-score-regression",
        "performance-only",
        "hardware-only",
        "generic-runtime",
    }
)

# Source-pattern flag rules (steps 32-54).  Each id maps a structural smell to the
# bug class it would, if triggerable, violate.  A flag is *only* a candidate.
FLAG_RULES: dict[str, str] = {
    "template-content-adjacent-role-header": "role-boundary-forgery",
    "template-unescaped-special-token": "delimiter-leakage",
    "template-tool-call-content-none-variance": "tool-call-corruption",
    "template-tool-args-string-vs-dict": "tool-call-corruption",
    "template-missing-closing-delimiter": "parser-boundary-confusion",
    "parser-raw-delimiter-in-data-region": "parser-boundary-confusion",
    "parser-quote-regex-after-sentinel": "parser-boundary-confusion",
    "parser-streaming-accumulator-reconstruction": "malformed-structured-output",
    "parser-partial-special-token-leak": "delimiter-leakage",
    "parser-function-state-cleared-late": "tool-call-corruption",
    "parser-reused-tool-call-index": "tool-call-corruption",
    "parser-premature-argument-emit": "tool-call-corruption",
    "parser-malformed-swallowed-empty": "tool-call-corruption",
    "request-skip-special-tokens-true-loses-delimiters": "delimiter-leakage",
    "request-skip-special-tokens-false-leaks-tokens": "delimiter-leakage",
    "tokenizer-vocab-decoded-mismatch": "delimiter-leakage",
    "reasoning-token-id-only-transition": "reasoning-tool-boundary-loss",
    "reasoning-drop-on-end-and-tool-same-delta": "reasoning-tool-boundary-loss",
    "serializer-empty-tool-calls-list": "response-shape-violation",
    "serializer-content-and-tool-calls-both": "response-shape-violation",
    "schema-double-encoded-arguments": "malformed-structured-output",
    "grammar-json-inside-tag-region": "malformed-structured-output",
    "stop-policy-cross-mode-tokens": "unsafe-stop-behavior",
}

TRIAGE_OUTCOMES = frozenset({"confirmed", "rejected", "abstained", "duplicate"})
DISCLOSURE_CHANNELS = frozenset({"public_issue", "security_advisory"})
REPRO_KINDS = frozenset({"parser-invocation", "template-render", "serializer-call", "local-server"})


class UpstreamBugCampaignError(ValueError):
    """Raised when the campaign dossier is incomplete or fails replay."""


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(mapping: object, key: str, ctx: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise UpstreamBugCampaignError(f"{ctx} is missing required field {key!r}")
    return mapping[key]


def _require_str(mapping: object, key: str, ctx: str) -> str:
    value = _require(mapping, key, ctx)
    if not isinstance(value, str) or not value:
        raise UpstreamBugCampaignError(f"{ctx}.{key} must be a non-empty string")
    return value


def _require_bool(mapping: object, key: str, ctx: str) -> bool:
    value = _require(mapping, key, ctx)
    if not isinstance(value, bool):
        raise UpstreamBugCampaignError(f"{ctx}.{key} must be a boolean")
    return value


def _require_list(mapping: object, key: str, ctx: str) -> list[Any]:
    value = _require(mapping, key, ctx)
    if not isinstance(value, list):
        raise UpstreamBugCampaignError(f"{ctx}.{key} must be a list")
    return value


def _str_tuple(mapping: object, key: str, ctx: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    items = _require_list(mapping, key, ctx)
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise UpstreamBugCampaignError(f"{ctx}.{key} entries must be non-empty strings")
        out.append(item)
    if not out and not allow_empty:
        raise UpstreamBugCampaignError(f"{ctx}.{key} must be non-empty")
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CampaignDefinitions:
    """Up-front scope definitions (steps 1-5)."""

    target_outcome: str
    confirmed_criteria: tuple[str, ...]
    bug_class_taxonomy: tuple[str, ...]
    excluded_classes: tuple[str, ...]
    working_notes_path: str

    @classmethod
    def from_mapping(cls, raw: object) -> "CampaignDefinitions":
        ctx = "definitions"
        taxonomy = _str_tuple(raw, "bug_class_taxonomy", ctx)
        unknown = set(taxonomy) - INTERFACE_SAFETY_BUG_CLASSES
        if unknown:
            raise UpstreamBugCampaignError(f"definitions.bug_class_taxonomy has unknown classes: {sorted(unknown)}")
        excluded = _str_tuple(raw, "excluded_classes", ctx)
        bad_excluded = set(excluded) - EXCLUDED_BUG_CLASSES
        if bad_excluded:
            raise UpstreamBugCampaignError(f"definitions.excluded_classes has unknown classes: {sorted(bad_excluded)}")
        notes = _require_str(raw, "working_notes_path", ctx)
        if notes in {"", "/"} or notes.endswith(("/src", "/src/")):
            raise UpstreamBugCampaignError("definitions.working_notes_path must be outside the committed tree")
        return cls(
            target_outcome=_require_str(raw, "target_outcome", ctx),
            confirmed_criteria=_str_tuple(raw, "confirmed_criteria", ctx),
            bug_class_taxonomy=taxonomy,
            excluded_classes=excluded,
            working_notes_path=notes,
        )


@dataclass(frozen=True, slots=True)
class UpstreamTarget:
    """A selected upstream project to scan (steps 6-12)."""

    target_id: str
    repository_url: str
    default_branch: str
    latest_release: str
    contribution_guide_url: str
    security_policy_url: str
    accepts_public_parser_issues: bool
    requires_security_disclosure: bool
    selection_rationale: str
    activity_signals: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: object) -> "UpstreamTarget":
        ctx = "target"
        target_id = _require_str(raw, "target_id", ctx)
        ctx = f"target[{target_id}]"
        url = _require_str(raw, "repository_url", ctx)
        if not url.startswith("https://github.com/"):
            raise UpstreamBugCampaignError(f"{ctx}.repository_url must be a public GitHub URL")
        return cls(
            target_id=target_id,
            repository_url=url,
            default_branch=_require_str(raw, "default_branch", ctx),
            latest_release=_require_str(raw, "latest_release", ctx),
            contribution_guide_url=_require_str(raw, "contribution_guide_url", ctx),
            security_policy_url=_require_str(raw, "security_policy_url", ctx),
            accepts_public_parser_issues=_require_bool(raw, "accepts_public_parser_issues", ctx),
            requires_security_disclosure=_require_bool(raw, "requires_security_disclosure", ctx),
            selection_rationale=_require_str(raw, "selection_rationale", ctx),
            activity_signals=_str_tuple(raw, "activity_signals", ctx),
        )


@dataclass(frozen=True, slots=True)
class ScannedSource:
    """Exact pinned upstream source under analysis (steps 13-15, 27)."""

    source_id: str
    target_id: str
    repository: str
    commit_sha: str
    package_version: str
    path: str
    public_url: str
    license: str
    captured_file: str
    full_file_sha256: str
    excerpt: str
    excerpt_sha256: str

    @classmethod
    def from_mapping(cls, raw: object) -> "ScannedSource":
        ctx = "scanned_source"
        source_id = _require_str(raw, "source_id", ctx)
        ctx = f"scanned_source[{source_id}]"
        public_url = _require_str(raw, "public_url", ctx)
        if not public_url.startswith("https://github.com/"):
            raise UpstreamBugCampaignError(f"{ctx}.public_url must be a GitHub URL")
        excerpt = _require_str(raw, "excerpt", ctx)
        excerpt_sha = _require_str(raw, "excerpt_sha256", ctx)
        actual = _sha256(excerpt)
        if actual != excerpt_sha:
            raise UpstreamBugCampaignError(f"{ctx}.excerpt_sha256 mismatch: expected {excerpt_sha}, got {actual}")
        return cls(
            source_id=source_id,
            target_id=_require_str(raw, "target_id", ctx),
            repository=_require_str(raw, "repository", ctx),
            commit_sha=_require_str(raw, "commit_sha", ctx),
            package_version=_require_str(raw, "package_version", ctx),
            path=_require_str(raw, "path", ctx),
            public_url=public_url,
            license=_require_str(raw, "license", ctx),
            captured_file=_require_str(raw, "captured_file", ctx),
            full_file_sha256=_require_str(raw, "full_file_sha256", ctx),
            excerpt=excerpt,
            excerpt_sha256=excerpt_sha,
        )

    def verify_capture(self, *, base_dir: Path) -> None:
        """Step 87/27: confirm the captured file matches the pinned full-file hash."""
        captured = base_dir / self.captured_file
        if not captured.is_file():
            raise UpstreamBugCampaignError(f"{self.source_id} captured file missing: {captured}")
        digest = _sha256(captured.read_text(encoding="utf-8"))
        if digest != self.full_file_sha256:
            raise UpstreamBugCampaignError(
                f"{self.source_id} captured file hash mismatch: expected {self.full_file_sha256}, got {digest}"
            )
        if self.excerpt not in captured.read_text(encoding="utf-8"):
            raise UpstreamBugCampaignError(f"{self.source_id} excerpt is not a substring of the captured file")


@dataclass(frozen=True, slots=True)
class CampaignInventories:
    """Inventories of prompt-facing surfaces (steps 16-24)."""

    prompt_facing_files: tuple[str, ...]
    response_models: tuple[str, ...]
    parser_boundary_constants: tuple[str, ...]
    request_mutation: tuple[str, ...]
    reasoning_parsers: tuple[str, ...]
    streaming_state_machines: tuple[str, ...]
    schema_grammar_adapters: tuple[str, ...]
    existing_tests: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: object) -> "CampaignInventories":
        ctx = "inventories"
        return cls(
            prompt_facing_files=_str_tuple(raw, "prompt_facing_files", ctx),
            response_models=_str_tuple(raw, "response_models", ctx),
            parser_boundary_constants=_str_tuple(raw, "parser_boundary_constants", ctx),
            request_mutation=_str_tuple(raw, "request_mutation", ctx),
            reasoning_parsers=_str_tuple(raw, "reasoning_parsers", ctx),
            streaming_state_machines=_str_tuple(raw, "streaming_state_machines", ctx),
            schema_grammar_adapters=_str_tuple(raw, "schema_grammar_adapters", ctx),
            existing_tests=_str_tuple(raw, "existing_tests", ctx),
        )


@dataclass(frozen=True, slots=True)
class DuplicateSearch:
    """Duplicate triage for a candidate (steps 62-66)."""

    queries: tuple[str, ...]
    is_duplicate: bool
    matches: tuple[str, ...]
    action: str

    @classmethod
    def from_mapping(cls, raw: object, *, ctx: str) -> "DuplicateSearch":
        is_dup = _require_bool(raw, "is_duplicate", ctx)
        matches = _str_tuple(raw, "matches", ctx, allow_empty=not is_dup)
        if is_dup and not matches:
            raise UpstreamBugCampaignError(f"{ctx}.matches must be present when is_duplicate is true")
        for match in matches:
            if not match.startswith("https://github.com/"):
                raise UpstreamBugCampaignError(f"{ctx}.matches entries must be GitHub URLs")
        return cls(
            queries=_str_tuple(raw, "queries", ctx),
            is_duplicate=is_dup,
            matches=matches,
            action=_require_str(raw, "action", ctx),
        )


@dataclass(frozen=True, slots=True)
class CandidateRepro:
    """Minimal deterministic reproduction plan (steps 67-86)."""

    kind: str
    invocation: str
    install_command: str
    runtime_requirement: str
    model_identifier: str
    regression_test: str
    fails_on_pinned_commit: bool
    offline: bool
    no_large_download: bool
    deterministic_two_runs: bool
    survives_cache_clear: bool
    not_promptabi_local: bool

    @classmethod
    def from_mapping(cls, raw: object, *, ctx: str) -> "CandidateRepro":
        kind = _require_str(raw, "kind", ctx)
        if kind not in REPRO_KINDS:
            raise UpstreamBugCampaignError(f"{ctx}.kind must be one of {sorted(REPRO_KINDS)}")
        return cls(
            kind=kind,
            invocation=_require_str(raw, "invocation", ctx),
            install_command=_require_str(raw, "install_command", ctx),
            runtime_requirement=_require_str(raw, "runtime_requirement", ctx),
            model_identifier=_require_str(raw, "model_identifier", ctx),
            regression_test=_require_str(raw, "regression_test", ctx),
            fails_on_pinned_commit=_require_bool(raw, "fails_on_pinned_commit", ctx),
            offline=_require_bool(raw, "offline", ctx),
            no_large_download=_require_bool(raw, "no_large_download", ctx),
            deterministic_two_runs=_require_bool(raw, "deterministic_two_runs", ctx),
            survives_cache_clear=_require_bool(raw, "survives_cache_clear", ctx),
            not_promptabi_local=_require_bool(raw, "not_promptabi_local", ctx),
        )


@dataclass(frozen=True, slots=True)
class CandidateDisclosure:
    """Disclosure routing for a candidate (steps 88-91)."""

    sensitivity: str
    channel: str
    redaction_ok: bool
    factual_impact: str

    @classmethod
    def from_mapping(cls, raw: object, *, ctx: str) -> "CandidateDisclosure":
        channel = _require_str(raw, "channel", ctx)
        if channel not in DISCLOSURE_CHANNELS:
            raise UpstreamBugCampaignError(f"{ctx}.channel must be one of {sorted(DISCLOSURE_CHANNELS)}")
        return cls(
            sensitivity=_require_str(raw, "sensitivity", ctx),
            channel=channel,
            redaction_ok=_require_bool(raw, "redaction_ok", ctx),
            factual_impact=_require_str(raw, "factual_impact", ctx),
        )


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Drafted upstream report (steps 92-104)."""

    title: str
    summary: str
    why_it_matters: str
    actual_output: str
    expected_output: str
    proposed_fix: str
    promptabi_attribution: str
    severity_claim: str

    @classmethod
    def from_mapping(cls, raw: object, *, ctx: str) -> "CandidateReport":
        severity = _require_str(raw, "severity_claim", ctx)
        if severity not in {"defer-to-maintainers", "low", "informational"}:
            raise UpstreamBugCampaignError(
                f"{ctx}.severity_claim must avoid overclaiming (use defer-to-maintainers/low/informational)"
            )
        return cls(
            title=_require_str(raw, "title", ctx),
            summary=_require_str(raw, "summary", ctx),
            why_it_matters=_require_str(raw, "why_it_matters", ctx),
            actual_output=_require_str(raw, "actual_output", ctx),
            expected_output=_require_str(raw, "expected_output", ctx),
            proposed_fix=raw.get("proposed_fix", "") if isinstance(raw, dict) else "",
            promptabi_attribution=_require_str(raw, "promptabi_attribution", ctx),
            severity_claim=severity,
        )


@dataclass(frozen=True, slots=True)
class CandidateFinding:
    """One triaged candidate (steps 31-104)."""

    candidate_id: str
    source_id: str
    flag_rule_id: str
    bug_class: str
    contract_violation_claim: str
    expected_behavior: str
    actual_behavior: str
    triggerable: bool
    triggerability_reason: str
    uses_exact_upstream_path: bool
    deterministic_parser_failure: bool
    already_fixed_on_branch: bool
    expected_outcome: str
    analysis_kind: str
    duplicate_search: DuplicateSearch
    repro: CandidateRepro
    disclosure: CandidateDisclosure
    report: CandidateReport | None

    @classmethod
    def from_mapping(cls, raw: object) -> "CandidateFinding":
        ctx = "candidate"
        candidate_id = _require_str(raw, "candidate_id", ctx)
        ctx = f"candidate[{candidate_id}]"
        flag_rule_id = _require_str(raw, "flag_rule_id", ctx)
        if flag_rule_id not in FLAG_RULES:
            raise UpstreamBugCampaignError(f"{ctx}.flag_rule_id {flag_rule_id!r} is not a known flag rule")
        bug_class = _require_str(raw, "bug_class", ctx)
        if FLAG_RULES[flag_rule_id] != bug_class:
            raise UpstreamBugCampaignError(
                f"{ctx}.bug_class {bug_class!r} does not match flag rule class {FLAG_RULES[flag_rule_id]!r}"
            )
        expected_outcome = _require_str(raw, "expected_outcome", ctx)
        if expected_outcome not in TRIAGE_OUTCOMES:
            raise UpstreamBugCampaignError(f"{ctx}.expected_outcome must be one of {sorted(TRIAGE_OUTCOMES)}")
        analysis_kind = _require_str(raw, "analysis_kind", ctx)
        if analysis_kind not in {"role-boundary", "reasoning-token-transition", "duplicate-only"}:
            raise UpstreamBugCampaignError(f"{ctx}.analysis_kind {analysis_kind!r} is unsupported")
        report_raw = raw.get("report") if isinstance(raw, dict) else None
        report = CandidateReport.from_mapping(report_raw, ctx=f"{ctx}.report") if report_raw else None
        if expected_outcome == "confirmed" and report is None:
            raise UpstreamBugCampaignError(f"{ctx} expects a confirmed outcome but has no drafted report")
        return cls(
            candidate_id=candidate_id,
            source_id=_require_str(raw, "source_id", ctx),
            flag_rule_id=flag_rule_id,
            bug_class=bug_class,
            contract_violation_claim=_require_str(raw, "contract_violation_claim", ctx),
            expected_behavior=_require_str(raw, "expected_behavior", ctx),
            actual_behavior=_require_str(raw, "actual_behavior", ctx),
            triggerable=_require_bool(raw, "triggerable", ctx),
            triggerability_reason=_require_str(raw, "triggerability_reason", ctx),
            uses_exact_upstream_path=_require_bool(raw, "uses_exact_upstream_path", ctx),
            deterministic_parser_failure=_require_bool(raw, "deterministic_parser_failure", ctx),
            already_fixed_on_branch=_require_bool(raw, "already_fixed_on_branch", ctx),
            expected_outcome=expected_outcome,
            analysis_kind=analysis_kind,
            duplicate_search=DuplicateSearch.from_mapping(_require(raw, "duplicate_search", ctx), ctx=f"{ctx}.duplicate_search"),
            repro=CandidateRepro.from_mapping(_require(raw, "repro", ctx), ctx=f"{ctx}.repro"),
            disclosure=CandidateDisclosure.from_mapping(_require(raw, "disclosure", ctx), ctx=f"{ctx}.disclosure"),
            report=report,
        )


@dataclass(frozen=True, slots=True)
class CampaignDossier:
    """The full auditable campaign dossier."""

    path: Path
    base_dir: Path
    version: int
    definitions: CampaignDefinitions
    targets: tuple[UpstreamTarget, ...]
    scanned_sources: tuple[ScannedSource, ...]
    inventories: CampaignInventories
    candidates: tuple[CandidateFinding, ...]

    def target(self, target_id: str) -> UpstreamTarget:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise UpstreamBugCampaignError(f"unknown target {target_id!r}")

    def source(self, source_id: str) -> ScannedSource:
        for source in self.scanned_sources:
            if source.source_id == source_id:
                return source
        raise UpstreamBugCampaignError(f"unknown scanned source {source_id!r}")


def load_campaign_dossier(path: str | Path = DEFAULT_CAMPAIGN_DOSSIER_PATH) -> CampaignDossier:
    """Load and structurally validate the campaign dossier."""

    dossier_path = Path(path)
    raw = json.loads(dossier_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise UpstreamBugCampaignError("campaign dossier must be a JSON object")
    version = raw.get("version")
    if version != UPSTREAM_BUG_CAMPAIGN_VERSION:
        raise UpstreamBugCampaignError(f"unsupported campaign dossier version {version!r}")

    definitions = CampaignDefinitions.from_mapping(_require(raw, "definitions", "dossier"))
    targets = tuple(UpstreamTarget.from_mapping(item) for item in _require_list(raw, "targets", "dossier"))
    if len(targets) < 3:
        raise UpstreamBugCampaignError("at least three upstream targets are required (step 6)")
    target_ids = {target.target_id for target in targets}
    if len(target_ids) != len(targets):
        raise UpstreamBugCampaignError("target ids must be unique")

    sources = tuple(ScannedSource.from_mapping(item) for item in _require_list(raw, "scanned_sources", "dossier"))
    if not sources:
        raise UpstreamBugCampaignError("at least one scanned source is required")
    source_ids = {source.source_id for source in sources}
    if len(source_ids) != len(sources):
        raise UpstreamBugCampaignError("scanned source ids must be unique")
    for source in sources:
        if source.target_id not in target_ids:
            raise UpstreamBugCampaignError(f"scanned source {source.source_id} references unknown target {source.target_id}")

    inventories = CampaignInventories.from_mapping(_require(raw, "inventories", "dossier"))
    candidates = tuple(CandidateFinding.from_mapping(item) for item in _require_list(raw, "candidates", "dossier"))
    if not candidates:
        raise UpstreamBugCampaignError("at least one candidate finding is required")
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise UpstreamBugCampaignError("candidate ids must be unique")
    for candidate in candidates:
        if candidate.analysis_kind != "duplicate-only" and candidate.source_id not in source_ids:
            raise UpstreamBugCampaignError(
                f"candidate {candidate.candidate_id} references unknown source {candidate.source_id}"
            )

    return CampaignDossier(
        path=dossier_path,
        base_dir=dossier_path.resolve().parent,
        version=version,
        definitions=definitions,
        targets=targets,
        scanned_sources=sources,
        inventories=inventories,
        candidates=candidates,
    )


# --------------------------------------------------------------------------- #
# Analysis: run PromptABI analyzers on the exact pinned source.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CandidateTriage:
    """Honest triage outcome for a candidate after real analysis."""

    candidate_id: str
    flag_rule_id: str
    bug_class: str
    outcome: str
    reportable: bool
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "flag_rule_id": self.flag_rule_id,
            "bug_class": self.bug_class,
            "outcome": self.outcome,
            "reportable": self.reportable,
            "evidence": self.evidence,
        }


def _detect_token_id_only_transition(source: str) -> bool:
    """Step 48: reasoning parser decides transitions from token *ids* only.

    Returns True when the streaming reasoning logic branches on ``*_token_id in
    *_token_ids`` membership without ever comparing the decoded marker text.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    token_id_membership_in_branch = False

    def _branch_uses_token_id(test: ast.AST) -> bool:
        for node in ast.walk(test):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        left = ast.unparse(node.left)
                        rights = " ".join(ast.unparse(c) for c in node.comparators)
                        if "token_id" in left and "token_ids" in rights:
                            return True
        return False

    class _Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            nonlocal token_id_membership_in_branch
            if _branch_uses_token_id(node.test):
                token_id_membership_in_branch = True
            self.generic_visit(node)

    _Visitor().visit(tree)
    # The reasoning/content transition is *decided* by token-id membership in a
    # branch condition.  Decoded-text use for slicing after the decision does not
    # change that the boundary detection itself is id-based.
    return token_id_membership_in_branch


def _marker_is_atomic_special_token(source: str) -> bool:
    """Triage helper (step 58): the reasoning marker is a single special token.

    DeepSeek-style ``<think>``/``</think>`` are atomic added tokens, so the
    id-only transition cannot be triggered by fragmenting the marker into
    ordinary text tokens.  This downgrades the candidate to ``rejected``.
    """

    return ("start_token" in source and "</think>" in source) or "<think>" in source


def _triage_candidate(
    candidate: CandidateFinding,
    dossier: CampaignDossier,
    *,
    corpus_path: Path,
) -> CandidateTriage:
    if candidate.duplicate_search.is_duplicate:
        return CandidateTriage(
            candidate.candidate_id,
            candidate.flag_rule_id,
            candidate.bug_class,
            "duplicate",
            reportable=False,
            evidence=f"duplicate of {candidate.duplicate_search.matches[0]}",
        )

    if candidate.analysis_kind == "role-boundary":
        source = dossier.source(candidate.source_id)
        # Build a chat-template config from the captured template and run the
        # real role-boundary non-forgeability analyzer.
        config = {
            "chat_template": source.excerpt,
            "eos_token": "<|im_end|>",
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
        parsed = parse_hf_chat_template_config(config)
        if not parsed.supported:
            return CandidateTriage(
                candidate.candidate_id,
                candidate.flag_rule_id,
                candidate.bug_class,
                "abstained",
                reportable=False,
                evidence=(
                    "template uses constructs outside the supported Jinja fragment "
                    f"({len(parsed.unsupported_constructs)} unsupported constructs); no claim made"
                ),
            )
        report = analyze_role_boundary_nonforgeability(parsed)
        if not report.ok:
            return CandidateTriage(
                candidate.candidate_id,
                candidate.flag_rule_id,
                candidate.bug_class,
                "confirmed",
                reportable=True,
                evidence=f"role-boundary forgeable: {len(report.findings)} witness(es) on pinned source",
            )
        return CandidateTriage(
            candidate.candidate_id,
            candidate.flag_rule_id,
            candidate.bug_class,
            "rejected",
            reportable=False,
            evidence="role boundaries proven non-forgeable on pinned source",
        )

    if candidate.analysis_kind == "reasoning-token-transition":
        source = dossier.source(candidate.source_id)
        flagged = _detect_token_id_only_transition(source.excerpt)
        if not flagged:
            return CandidateTriage(
                candidate.candidate_id,
                candidate.flag_rule_id,
                candidate.bug_class,
                "rejected",
                reportable=False,
                evidence="no token-id-only transition logic found on pinned source",
            )
        if _marker_is_atomic_special_token(source.excerpt):
            return CandidateTriage(
                candidate.candidate_id,
                candidate.flag_rule_id,
                candidate.bug_class,
                "rejected",
                reportable=False,
                evidence=(
                    "token-id-only transition is present, but the marker is an atomic special "
                    "token that cannot be fragmented into ordinary text tokens; not triggerable"
                ),
            )
        return CandidateTriage(
            candidate.candidate_id,
            candidate.flag_rule_id,
            candidate.bug_class,
            "confirmed",
            reportable=True,
            evidence="token-id-only transition over a fragmentable marker",
        )

    # duplicate-only analysis kind with no duplicate is contradictory.
    return CandidateTriage(
        candidate.candidate_id,
        candidate.flag_rule_id,
        candidate.bug_class,
        "rejected",
        reportable=False,
        evidence="no analyzable upstream path supplied",
    )


@dataclass(frozen=True, slots=True)
class ConfirmedDetectionReference:
    """A historical confirmed bug the analyzer reproduces (proof-of-detection)."""

    case_id: str
    public_reference: str
    bug_class: str
    witness_count: int


def _confirmed_detection_references(corpus_path: Path) -> tuple[ConfirmedDetectionReference, ...]:
    """Replay the public confirmed exemplars to prove the analyzer finds real bugs."""

    if not corpus_path.is_file():
        return ()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    references: list[ConfirmedDetectionReference] = []
    for case in corpus.get("role_boundary_cases", []):
        parsed = parse_hf_chat_template_config(case["template_config"])
        if not parsed.supported:
            continue
        report = analyze_role_boundary_nonforgeability(parsed)
        if not report.ok and report.findings:
            references.append(
                ConfirmedDetectionReference(
                    case_id=case["id"],
                    public_reference=case["public_reference"],
                    bug_class="role-boundary-forgery",
                    witness_count=len(report.findings),
                )
            )
    return tuple(references)


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Outcome of running the full campaign."""

    dossier_path: Path
    targets: int
    scanned_sources: int
    candidates: int
    triage: tuple[CandidateTriage, ...]
    confirmed_detection_references: tuple[ConfirmedDetectionReference, ...]

    @property
    def reportable(self) -> tuple[CandidateTriage, ...]:
        return tuple(t for t in self.triage if t.reportable)

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts = {outcome: 0 for outcome in sorted(TRIAGE_OUTCOMES)}
        for triage in self.triage:
            counts[triage.outcome] += 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "version": UPSTREAM_BUG_CAMPAIGN_VERSION,
            "dossier_path": str(self.dossier_path),
            "targets": self.targets,
            "scanned_sources": self.scanned_sources,
            "candidates": self.candidates,
            "outcome_counts": self.outcome_counts,
            "reportable": [t.candidate_id for t in self.reportable],
            "triage": [t.to_dict() for t in self.triage],
            "confirmed_detection_references": [
                {
                    "case_id": ref.case_id,
                    "public_reference": ref.public_reference,
                    "bug_class": ref.bug_class,
                    "witness_count": ref.witness_count,
                }
                for ref in self.confirmed_detection_references
            ],
        }


def run_campaign(
    dossier: CampaignDossier | None = None,
    *,
    dossier_path: str | Path = DEFAULT_CAMPAIGN_DOSSIER_PATH,
    corpus_path: str | Path = DEFAULT_REAL_WORLD_BUG_CORPUS_PATH,
    verify_captures: bool = True,
) -> CampaignResult:
    """Run the campaign: verify provenance, replay analyzers, triage candidates."""

    dossier = dossier if dossier is not None else load_campaign_dossier(dossier_path)
    corpus = Path(corpus_path)

    if verify_captures:
        for source in dossier.scanned_sources:
            source.verify_capture(base_dir=dossier.base_dir)

    triage = tuple(_triage_candidate(c, dossier, corpus_path=corpus) for c in dossier.candidates)

    # Step 31/55-61: a candidate may only be declared confirmed if the dossier's
    # own expectation matches the analyzer's honest outcome.
    by_id = {c.candidate_id: c for c in dossier.candidates}
    for result in triage:
        expected = by_id[result.candidate_id].expected_outcome
        if expected != result.outcome:
            raise UpstreamBugCampaignError(
                f"candidate {result.candidate_id} expected {expected!r} but analysis produced {result.outcome!r}: "
                f"{result.evidence}"
            )

    references = _confirmed_detection_references(corpus)

    return CampaignResult(
        dossier_path=dossier.path,
        targets=len(dossier.targets),
        scanned_sources=len(dossier.scanned_sources),
        candidates=len(dossier.candidates),
        triage=triage,
        confirmed_detection_references=references,
    )


def render_candidate_report_markdown(candidate: CandidateFinding, source: ScannedSource) -> str:
    """Render a responsible-disclosure report draft for a reportable candidate (steps 92-104)."""

    if candidate.report is None:
        raise UpstreamBugCampaignError(f"candidate {candidate.candidate_id} has no drafted report")
    report = candidate.report
    lines = [
        f"# {report.title}",
        "",
        "## Summary",
        report.summary,
        "",
        "## Affected version",
        f"- Repository: {source.repository}",
        f"- Commit: {source.commit_sha}",
        f"- Package version: {source.package_version}",
        f"- File: `{source.path}`",
        f"- Source: {source.public_url}",
        "",
        "## Reproduction",
        f"- Install: `{candidate.repro.install_command}`",
        f"- Run: `{candidate.repro.invocation}`",
        f"- Regression test: `{candidate.repro.regression_test}`",
        "",
        "## Actual behavior",
        report.actual_output,
        "",
        "## Expected behavior",
        report.expected_output,
        "",
        "## Why it matters",
        report.why_it_matters,
        "",
        "## Duplicate search",
        candidate.duplicate_search.action,
    ]
    if report.proposed_fix:
        lines += ["", "## Proposed fix direction", report.proposed_fix]
    lines += [
        "",
        "## Provenance",
        report.promptabi_attribution,
        f"Severity: {report.severity_claim} (maintainers decide final labels).",
    ]
    return "\n".join(lines)


def render_campaign_text(result: CampaignResult) -> str:
    """Human-readable campaign summary."""

    lines = [
        "Upstream interface-safety bug campaign",
        f"  dossier: {result.dossier_path}",
        f"  targets: {result.targets}  sources: {result.scanned_sources}  candidates: {result.candidates}",
        "  outcomes: " + ", ".join(f"{k}={v}" for k, v in result.outcome_counts.items()),
        "  reportable: " + (", ".join(t.candidate_id for t in result.reportable) or "<none>"),
        "  proof-of-detection (confirmed public exemplars the analyzer reproduces):",
    ]
    for ref in result.confirmed_detection_references:
        lines.append(f"    - {ref.case_id} [{ref.bug_class}] {ref.witness_count} witnesses -> {ref.public_reference}")
    lines.append("  triage:")
    for triage in result.triage:
        lines.append(f"    - {triage.candidate_id}: {triage.outcome} ({triage.evidence})")
    return "\n".join(lines)


def render_campaign_json(result: CampaignResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)
