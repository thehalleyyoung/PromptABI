"""Static lint checks for PromptABI contract-language artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .artifacts import StaticContractArtifact, StaticContractInvariant, StaticContractRule
from .diagnostics import SourceSpan
from .policies import Suppression, VerificationPolicy, empty_policy, load_policy_file


SUPPORTED_RULE_TYPES = frozenset(
    {
        "budget",
        "evaluation",
        "interface",
        "llm-app",
        "prompt-pack",
        "rag",
        "role-boundary",
        "solver",
        "static-contract",
        "stop-policy",
        "tool-schema",
        "training",
    }
)
SUPPORTED_APPLIES_TO = frozenset(
    {
        "chat-template",
        "evaluation-harness",
        "framework-truncation-config",
        "grammar",
        "prompt-pack",
        "prompt-segment",
        "provider-config",
        "schema",
        "special-token-map",
        "static-contract",
        "stop-policy",
        "tokenizer",
        "tool-definition",
        "training-manifest",
    }
)
SUPPORTED_INVARIANT_SYMBOLS = frozenset(
    {
        "assistant_prompt_tokens",
        "completion_budget_tokens",
        "context_window_tokens",
        "input_budget_tokens",
        "max_context_tokens",
        "max_output_tokens",
        "prompt_tokens",
        "required_prompt_tokens",
        "reserved_tool_tokens",
        "system_prompt_tokens",
        "tool_argument_tokens",
        "user_prompt_tokens",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_BROAD_RULE_IDS = frozenset({"*", "all", "promptabi.*", "static-contract-*"})


@dataclass(frozen=True, slots=True)
class ContractLintFinding:
    """One deterministic contract-lint finding."""

    code: str
    severity: str
    message: str
    suggestion: str
    rule_name: str | None = None
    span: SourceSpan | None = None
    evidence: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }
        if self.rule_name is not None:
            data["rule_name"] = self.rule_name
        if self.span is not None:
            data["span"] = self.span.to_dict()
        if self.evidence:
            data["evidence"] = dict(self.evidence)
        return data


@dataclass(frozen=True, slots=True)
class ContractLintReport:
    """Contract lint findings plus aggregate status."""

    contract_name: str
    findings: tuple[ContractLintFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "contract_name": self.contract_name,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def lint_static_contract(
    contract: StaticContractArtifact,
    *,
    policy: VerificationPolicy | None = None,
) -> ContractLintReport:
    """Lint one parsed static contract for review hazards and finite contradictions."""

    spans = _rule_spans(contract)
    findings: list[ContractLintFinding] = []
    seen_names: set[str] = set()
    for rule in contract.rules:
        span = spans.get(rule.name)
        if rule.name in seen_names:
            findings.append(
                _finding(
                    "contradictory-policy",
                    "error",
                    f"rule {rule.name!r} is declared more than once",
                    "Compose repeated rules explicitly or rename one rule so precedence is reviewable.",
                    rule,
                    span,
                )
            )
        seen_names.add(rule.name)
        findings.extend(_lint_rule_shape(rule, span))
        findings.extend(_lint_rule_semantics(rule, span))
        findings.extend(_lint_invariants(rule, span))
    for suppression in (policy or empty_policy()).suppressions:
        findings.extend(_lint_suppression(suppression))
    return ContractLintReport(
        contract_name=contract.name,
        findings=tuple(sorted(findings, key=lambda item: (item.severity, item.code, item.rule_name or "", item.message))),
    )


def load_contract_lint_policy(paths: tuple[str | Path, ...]) -> VerificationPolicy:
    """Load optional policy files whose suppressions should be linted with a contract."""

    policy = empty_policy()
    if not paths:
        return policy
    from .policies import merge_policies

    for path in paths:
        policy = merge_policies(policy, load_policy_file(path))
    return policy


def render_contract_lint_json(report: ContractLintReport) -> str:
    """Render a deterministic JSON lint report."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_contract_lint_text(report: ContractLintReport) -> str:
    """Render a concise human-readable lint report."""

    lines = [
        f"PromptABI contract lint: {report.contract_name}",
        f"errors: {report.error_count}",
        f"warnings: {report.warning_count}",
    ]
    if not report.findings:
        lines.append("status: PASS")
        return "\n".join(lines) + "\n"
    lines.append("findings:")
    for finding in report.findings:
        location = f" ({_format_span(finding.span)})" if finding.span is not None else ""
        rule = f" {finding.rule_name}" if finding.rule_name is not None else ""
        lines.append(f"  {finding.severity.upper()} {finding.code}{rule}{location}: {finding.message}")
        lines.append(f"    suggestion: {finding.suggestion}")
        for key, value in finding.evidence:
            lines.append(f"    {key}: {value}")
    return "\n".join(lines) + "\n"


def _lint_rule_shape(rule: StaticContractRule, span: SourceSpan | None) -> tuple[ContractLintFinding, ...]:
    findings: list[ContractLintFinding] = []
    if rule.rule_type not in SUPPORTED_RULE_TYPES:
        findings.append(
            _finding(
                "unsupported-fragment",
                "warning",
                f"rule type {rule.rule_type!r} is not in the supported contract-lint taxonomy",
                "Use a supported rule type or add an explicit checker/plugin before relying on this rule in CI.",
                rule,
                span,
                evidence=(("supported_rule_types", ", ".join(sorted(SUPPORTED_RULE_TYPES))),),
            )
        )
    unsupported_surfaces = tuple(surface for surface in rule.applies_to if surface not in SUPPORTED_APPLIES_TO)
    if unsupported_surfaces:
        findings.append(
            _finding(
                "unsupported-fragment",
                "warning",
                f"applies_to references unsupported artifact surfaces: {', '.join(unsupported_surfaces)}",
                "Spell artifact surfaces using PromptABI artifact kind names so dependency analysis can wire the rule.",
                rule,
                span,
            )
        )
    for field_name, values in (
        ("allowed_roles", rule.allowed_roles),
        ("required_regions", rule.required_regions),
        ("schema.requires", tuple(value for obligation in rule.schema_obligations for value in obligation.requires)),
        ("stop.forbid_inside", tuple(policy.forbid_inside for policy in rule.stop_policies if policy.forbid_inside is not None)),
        ("assume.artifact", tuple(assumption.artifact for assumption in rule.assumptions)),
        ("assume.requires", tuple(value for assumption in rule.assumptions for value in assumption.requires)),
        ("guarantee.artifact", tuple(guarantee.artifact for guarantee in rule.guarantees)),
        ("guarantee.provides", tuple(value for guarantee in rule.guarantees for value in guarantee.provides)),
    ):
        invalid = tuple(value for value in values if not _IDENTIFIER_RE.fullmatch(value))
        if invalid:
            findings.append(
                _finding(
                    "unsupported-fragment",
                    "warning",
                    f"{field_name} contains values outside the finite identifier fragment: {', '.join(invalid)}",
                    "Use finite symbolic names and put raw delimiters only in forbid_delimiters or stop sequences.",
                    rule,
                    span,
                )
            )
    provided = {
        provided
        for guarantee in rule.guarantees
        for provided in guarantee.provides
    }
    for assumption in rule.assumptions:
        if not assumption.requires:
            findings.append(
                _finding(
                    "vacuous-guarantee",
                    "warning",
                    f"assumption for {assumption.artifact!r} requires no guarantee facts",
                    "List at least one finite guarantee fact after `requires`.",
                    rule,
                    span,
                )
            )
            continue
        unresolved = tuple(fact for fact in assumption.requires if fact not in provided)
        if unresolved and rule.guarantees:
            findings.append(
                _finding(
                    "unresolved-assumption",
                    "warning",
                    f"assumption for {assumption.artifact!r} has no matching guarantee fact in this contract: {', '.join(unresolved)}",
                    "Add a matching guarantee directive, or compose this contract with the provider contract before enforcing it.",
                    rule,
                    span,
                    evidence=(("available_guarantees", ", ".join(sorted(provided)) or "<none>"),),
                )
            )
    for guarantee in rule.guarantees:
        if not guarantee.provides:
            findings.append(
                _finding(
                    "vacuous-guarantee",
                    "warning",
                    f"guarantee for {guarantee.artifact!r} provides no facts",
                    "List at least one finite guarantee fact after `provides`.",
                    rule,
                    span,
                )
            )
    return tuple(findings)


def _lint_rule_semantics(rule: StaticContractRule, span: SourceSpan | None) -> tuple[ContractLintFinding, ...]:
    findings: list[ContractLintFinding] = []
    if not any(
        (
            rule.allowed_roles,
            rule.required_regions,
            rule.forbidden_delimiters,
            rule.schema_obligations,
            rule.stop_policies,
            rule.invariants,
            rule.assumptions,
            rule.guarantees,
        )
    ):
        findings.append(
            _finding(
                "vacuous-guarantee",
                "warning",
                "rule has severity and metadata but no enforceable obligations",
                "Add roles, required regions, forbidden delimiters, schema requirements, stop policies, invariants, assumptions, or guarantees.",
                rule,
                span,
            )
        )
    if rule.allowed_roles:
        missing_regions = tuple(region for region in rule.required_regions if region not in rule.allowed_roles)
        if missing_regions:
            findings.append(
                _finding(
                    "impossible-rule",
                    "error",
                    f"required regions are forbidden by allowed_roles: {', '.join(missing_regions)}",
                    "Either allow the required role regions or remove them from required_regions.",
                    rule,
                    span,
                )
            )
    forbidden_required = tuple(delimiter for delimiter in rule.forbidden_delimiters if delimiter in rule.required_regions)
    if forbidden_required:
        findings.append(
            _finding(
                "contradictory-policy",
                "error",
                f"values are both required regions and forbidden delimiters: {', '.join(forbidden_required)}",
                "Keep structural region names separate from literal forbidden delimiters.",
                rule,
                span,
            )
        )
    for policy in rule.stop_policies:
        empty_stops = tuple(stop for stop in policy.stops if stop == "")
        if empty_stops:
            findings.append(
                _finding(
                    "impossible-rule",
                    "error",
                    f"stop policy {policy.name!r} contains an empty stop sequence",
                    "Remove empty stops; every stop sequence must consume at least one byte or token.",
                    rule,
                    span,
                )
            )
        delimiter_collisions = tuple(stop for stop in policy.stops if stop in rule.forbidden_delimiters)
        if delimiter_collisions:
            findings.append(
                _finding(
                    "contradictory-policy",
                    "warning",
                    f"stop policy {policy.name!r} requires stops that are also forbidden delimiters: {', '.join(delimiter_collisions)}",
                    "Model control delimiters and output stop sequences separately, or justify the overlap with a narrower rule.",
                    rule,
                    span,
                )
            )
    return tuple(findings)


def _lint_invariants(rule: StaticContractRule, span: SourceSpan | None) -> tuple[ContractLintFinding, ...]:
    findings: list[ContractLintFinding] = []
    for invariant in rule.invariants:
        unknown = tuple(
            symbol
            for symbol in (invariant.left, invariant.right)
            if not _is_int(symbol) and symbol not in SUPPORTED_INVARIANT_SYMBOLS
        )
        if unknown:
            findings.append(
                _finding(
                    "unsupported-fragment",
                    "warning",
                    f"invariant {invariant.name!r} references unsupported finite symbols: {', '.join(unknown)}",
                    "Use known finite contract symbols or add a plugin that provides domains for the custom symbols.",
                    rule,
                    span,
                )
            )
        if _literal_relation_impossible(invariant):
            findings.append(
                _finding(
                    "impossible-rule",
                    "error",
                    f"invariant {invariant.name!r} is unsatisfiable over integer literals",
                    "Fix the literal comparison or replace it with a symbolic invariant that can be checked against artifacts.",
                    rule,
                    span,
                    evidence=(("expression", f"{invariant.left} {invariant.op} {invariant.right}"),),
                )
            )
    findings.extend(_pairwise_invariant_contradictions(rule, span))
    return tuple(findings)


def _pairwise_invariant_contradictions(
    rule: StaticContractRule,
    span: SourceSpan | None,
) -> tuple[ContractLintFinding, ...]:
    findings: list[ContractLintFinding] = []
    by_left: dict[str, list[StaticContractInvariant]] = {}
    for invariant in rule.invariants:
        if _is_int(invariant.right):
            by_left.setdefault(invariant.left, []).append(invariant)
    for left, invariants in by_left.items():
        lowers = tuple(item for item in invariants if item.op in {">", ">="})
        uppers = tuple(item for item in invariants if item.op in {"<", "<="})
        equals = tuple(item for item in invariants if item.op == "==")
        not_equals = tuple(item for item in invariants if item.op == "!=")
        for equal in equals:
            equal_value = int(equal.right)
            if any(_violates_bound(equal_value, bound) for bound in (*lowers, *uppers)):
                findings.append(_contradictory_invariants(rule, span, left, equal, (*lowers, *uppers)))
            if any(int(item.right) == equal_value for item in not_equals):
                findings.append(_contradictory_invariants(rule, span, left, equal, not_equals))
        for lower in lowers:
            for upper in uppers:
                if _bounds_do_not_overlap(lower, upper):
                    findings.append(_contradictory_invariants(rule, span, left, lower, (upper,)))
    return tuple(findings)


def _lint_suppression(suppression: Suppression) -> tuple[ContractLintFinding, ...]:
    broad_reasons: list[str] = []
    if suppression.rule_id in _BROAD_RULE_IDS or "*" in suppression.rule_id:
        broad_reasons.append("wildcard rule_id")
    if suppression.fingerprint is None:
        broad_reasons.append("missing fingerprint")
    if suppression.artifact is None and suppression.path is None:
        broad_reasons.append("missing artifact/path scope")
    if not broad_reasons:
        return ()
    return (
        ContractLintFinding(
            code="overly-broad-suppression",
            severity="warning",
            message=f"suppression for {suppression.rule_id!r} is broad: {', '.join(broad_reasons)}",
            suggestion="Scope suppressions to a stable fingerprint and a specific artifact or path before relying on them in CI.",
            span=suppression.span,
            evidence=(
                ("rule_id", suppression.rule_id),
                ("has_fingerprint", str(suppression.fingerprint is not None).lower()),
            ),
        ),
    )


def _contradictory_invariants(
    rule: StaticContractRule,
    span: SourceSpan | None,
    left: str,
    invariant: StaticContractInvariant,
    others: tuple[StaticContractInvariant, ...],
) -> ContractLintFinding:
    expressions = (invariant, *others)
    return _finding(
        "contradictory-policy",
        "error",
        f"invariants over {left!r} have no shared integer assignment",
        "Relax or remove one of the contradictory numeric bounds.",
        rule,
        span,
        evidence=(("invariants", "; ".join(f"{item.left} {item.op} {item.right}" for item in expressions)),),
    )


def _finding(
    code: str,
    severity: str,
    message: str,
    suggestion: str,
    rule: StaticContractRule,
    span: SourceSpan | None,
    *,
    evidence: tuple[tuple[str, str], ...] = (),
) -> ContractLintFinding:
    return ContractLintFinding(
        code=code,
        severity=severity,
        message=message,
        suggestion=suggestion,
        rule_name=rule.name,
        span=span,
        evidence=evidence,
    )


def _literal_relation_impossible(invariant: StaticContractInvariant) -> bool:
    if not (_is_int(invariant.left) and _is_int(invariant.right)):
        return False
    left = int(invariant.left)
    right = int(invariant.right)
    return not _compare(left, invariant.op, right)


def _bounds_do_not_overlap(lower: StaticContractInvariant, upper: StaticContractInvariant) -> bool:
    lower_value = int(lower.right)
    upper_value = int(upper.right)
    if lower_value > upper_value:
        return True
    if lower_value == upper_value and (lower.op == ">" or upper.op == "<"):
        return True
    return False


def _violates_bound(value: int, bound: StaticContractInvariant) -> bool:
    return not _compare(value, bound.op, int(bound.right))


def _compare(left: int, op: str, right: int) -> bool:
    if op == "<=":
        return left <= right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == ">":
        return left > right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    raise ValueError(f"unsupported invariant op {op!r}")


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _rule_spans(contract: StaticContractArtifact) -> dict[str, SourceSpan]:
    line_by_rule: dict[str, int] = {}
    for key, value in contract.metadata:
        if key != "rule_source_lines" or not isinstance(value, tuple):
            continue
        for item in value:
            if not isinstance(item, str) or ":" not in item:
                continue
            rule_name, raw_line = item.rsplit(":", 1)
            try:
                line_by_rule[rule_name] = int(raw_line)
            except ValueError:
                continue
    path = contract.location.path or contract.location.uri or f"{contract.name}.pabi"
    return {
        rule.name: contract.source_span or SourceSpan(path=path, start_line=line_by_rule.get(rule.name, 1), start_column=1)
        for rule in contract.rules
    }


def _format_span(span: SourceSpan | None) -> str:
    if span is None:
        return "unknown"
    return f"{span.path}:{span.start_line}:{span.start_column}"
