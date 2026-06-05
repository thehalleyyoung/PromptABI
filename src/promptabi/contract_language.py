"""Parser and formatter for the PromptABI static contract DSL."""

from __future__ import annotations

import csv
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    ArtifactKind,
    ArtifactLocation,
    StaticContractArtifact,
    StaticContractAssumption,
    StaticContractGuarantee,
    StaticContractInvariant,
    StaticContractRule,
    StaticContractSchemaObligation,
    StaticContractStopPolicy,
)
from .diagnostics import SourceSpan


CONTRACT_LANGUAGE_VERSION = "promptabi.contract/v1"
_RULE_RE = re.compile(r"^rule\s+([A-Za-z0-9_.:/-]+)(?P<attrs>.*):$")
_INVARIANT_RE = re.compile(r"^([A-Za-z0-9_.:/-]+)\s*(<=|<|>=|>|==|!=)\s*([A-Za-z0-9_.:/-]+|[-+]?\d+)$")


class ContractLanguageError(ValueError):
    """A parse error with a source location suitable for diagnostics."""

    def __init__(self, message: str, line: int, column: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"line {self.line}:{self.column}: {self.message}"

    def to_source_span(self, path: str) -> SourceSpan:
        return SourceSpan(path=path, start_line=self.line, start_column=self.column)


def parse_static_contract_text(
    text: str,
    *,
    name: str = "contract",
    path: str | None = None,
) -> StaticContractArtifact:
    """Parse a human-authored ``.pabi`` contract into the typed artifact model.

    The DSL is intentionally line-oriented so it can be reviewed in diffs and
    round-tripped by the formatter. It lowers to the same JSON artifact shape
    used by the verifier, preserving rule names and source line metadata.
    """

    version = CONTRACT_LANGUAGE_VERSION
    rules: list[StaticContractRule] = []
    rule_lines: list[tuple[str, int]] = []
    current: _RuleBuilder | None = None
    current_line = 0

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("contract "):
            if current is not None:
                rule, start_line = current.build()
                rules.append(rule)
                rule_lines.append((rule.name, start_line))
                current = None
            version = line.split(None, 1)[1].strip()
            if version != CONTRACT_LANGUAGE_VERSION:
                raise ContractLanguageError(
                    f"unsupported contract version {version!r}; expected {CONTRACT_LANGUAGE_VERSION!r}",
                    line_number,
                )
            continue
        match = _RULE_RE.match(line)
        if match:
            if current is not None:
                rule, start_line = current.build()
                rules.append(rule)
                rule_lines.append((rule.name, start_line))
            current_line = line_number
            current = _RuleBuilder(name=match.group(1), line=line_number)
            _parse_rule_attributes(match.group("attrs"), current, line_number)
            continue
        if current is None:
            raise ContractLanguageError("expected 'contract ...' or 'rule <name> ...:'", line_number)
        _parse_rule_body_line(line, current, line_number)

    if current is not None:
        rule, start_line = current.build()
        rules.append(rule)
        rule_lines.append((rule.name, start_line))
    if not rules:
        raise ContractLanguageError("contract must declare at least one rule", current_line or 1)

    metadata: tuple[tuple[str, object], ...] = (
        ("language", "pabi"),
        ("rule_source_lines", tuple(f"{rule_name}:{line}" for rule_name, line in rule_lines)),
    )
    return StaticContractArtifact(
        kind=ArtifactKind.STATIC_CONTRACT,
        name=name,
        location=ArtifactLocation(path=path or f"{name}.pabi"),
        contract_version=version,
        rules=tuple(rules),
        metadata=metadata,
    )


def parse_static_contract_file(path: Path, *, name: str | None = None) -> StaticContractArtifact:
    """Read and parse a ``.pabi`` contract file."""

    return parse_static_contract_text(path.read_text(encoding="utf-8"), name=name or path.stem, path=str(path))


def format_static_contract(contract: StaticContractArtifact) -> str:
    """Render a canonical ``.pabi`` representation for a static contract."""

    lines = [f"contract {contract.contract_version}", ""]
    for rule_index, rule in enumerate(contract.rules):
        if rule_index:
            lines.append("")
        attrs = [f"type {rule.rule_type}", f"severity {rule.severity}"]
        if rule.applies_to:
            attrs.append("applies_to " + _format_values(rule.applies_to))
        lines.append(f"rule {rule.name} {' '.join(attrs)}:")
        if rule.description is not None:
            lines.append(f"  description {_quote(rule.description)}")
        if rule.allowed_roles:
            lines.append(f"  allowed_roles {_format_values(rule.allowed_roles)}")
        if rule.required_regions:
            lines.append(f"  required_regions {_format_values(rule.required_regions)}")
        if rule.forbidden_delimiters:
            lines.append(f"  forbid_delimiters {_format_values(rule.forbidden_delimiters)}")
        for obligation in rule.schema_obligations:
            lines.append(f"  schema {obligation.schema} requires {_format_values(obligation.requires)}")
        for policy in rule.stop_policies:
            suffix = ""
            if policy.forbid_inside is not None:
                suffix = f" forbid_inside {policy.forbid_inside}"
            lines.append(f"  stop {policy.name} stops {_format_values(policy.stops)}{suffix}")
        for invariant in rule.invariants:
            lines.append(f"  invariant {invariant.name}: {invariant.left} {invariant.op} {invariant.right}")
        for assumption in rule.assumptions:
            lines.append(f"  assume {assumption.artifact} requires {_format_values(assumption.requires)}")
        for guarantee in rule.guarantees:
            lines.append(f"  guarantee {guarantee.artifact} provides {_format_values(guarantee.provides)}")
    return "\n".join(lines) + "\n"


def render_static_contract_json(contract: StaticContractArtifact) -> str:
    """Render the artifact JSON shape produced by the DSL parser."""

    return json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n"


def _parse_rule_attributes(attrs: str, builder: "_RuleBuilder", line_number: int) -> None:
    tokens = shlex.split(attrs)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "type":
            index += 1
            builder.rule_type = _expect_token(tokens, index, line_number, "type value")
        elif token == "severity":
            index += 1
            builder.severity = _expect_token(tokens, index, line_number, "severity value")
        elif token == "applies_to":
            index += 1
            value = _expect_token(tokens, index, line_number, "applies_to list")
            builder.applies_to = _parse_values(value)
        else:
            raise ContractLanguageError(f"unknown rule attribute {token!r}", line_number)
        index += 1


def _parse_rule_body_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    if line.startswith("description "):
        builder.description = line.split(None, 1)[1].strip().strip('"')
        return
    if line.startswith("allowed_roles "):
        builder.allowed_roles = _parse_values(line.split(None, 1)[1])
        return
    if line.startswith("required_regions "):
        builder.required_regions = _parse_values(line.split(None, 1)[1])
        return
    if line.startswith("forbid_delimiters ") or line.startswith("forbidden_delimiters "):
        builder.forbidden_delimiters = _parse_values(line.split(None, 1)[1])
        return
    if line.startswith("schema "):
        _parse_schema_line(line, builder, line_number)
        return
    if line.startswith("stop "):
        _parse_stop_line(line, builder, line_number)
        return
    if line.startswith("invariant "):
        _parse_invariant_line(line, builder, line_number)
        return
    if line.startswith("assume "):
        _parse_assumption_line(line, builder, line_number)
        return
    if line.startswith("guarantee "):
        _parse_guarantee_line(line, builder, line_number)
        return
    raise ContractLanguageError(f"unknown rule directive {line.split()[0]!r}", line_number)


def _parse_schema_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    tokens = shlex.split(line)
    if len(tokens) != 4 or tokens[2] != "requires":
        raise ContractLanguageError("schema directives must be: schema <name> requires <field[,field...]>", line_number)
    builder.schema_obligations.append(StaticContractSchemaObligation(schema=tokens[1], requires=_parse_values(tokens[3])))


def _parse_stop_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    tokens = shlex.split(line)
    if len(tokens) not in {4, 6} or tokens[2] != "stops":
        raise ContractLanguageError(
            "stop directives must be: stop <name> stops <stop[,stop...]> [forbid_inside <region>]",
            line_number,
        )
    forbid_inside = None
    if len(tokens) == 6:
        if tokens[4] != "forbid_inside":
            raise ContractLanguageError("expected forbid_inside after stop list", line_number)
        forbid_inside = tokens[5]
    builder.stop_policies.append(StaticContractStopPolicy(name=tokens[1], stops=_parse_values(tokens[3]), forbid_inside=forbid_inside))


def _parse_invariant_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    head, _, expr = line.partition(":")
    tokens = shlex.split(head)
    if len(tokens) == 2 and expr.strip():
        name = tokens[1]
        expression = expr.strip()
    elif len(tokens) == 4 and not expr.strip():
        name = f"{tokens[1]}-{tokens[2]}-{tokens[3]}"
        expression = " ".join(tokens[1:])
    else:
        raise ContractLanguageError("invariant directives must be: invariant <name>: <left> <op> <right>", line_number)
    match = _INVARIANT_RE.match(expression)
    if match is None:
        raise ContractLanguageError("invariant expression must use <=, <, >=, >, ==, or !=", line_number)
    builder.invariants.append(
        StaticContractInvariant(name=name, left=match.group(1), op=match.group(2), right=match.group(3))
    )


def _parse_assumption_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    tokens = shlex.split(line)
    if len(tokens) != 4 or tokens[2] != "requires":
        raise ContractLanguageError("assume directives must be: assume <artifact-or-kind> requires <fact[,fact...]>", line_number)
    builder.assumptions.append(StaticContractAssumption(artifact=tokens[1], requires=_parse_values(tokens[3])))


def _parse_guarantee_line(line: str, builder: "_RuleBuilder", line_number: int) -> None:
    tokens = shlex.split(line)
    if len(tokens) != 4 or tokens[2] != "provides":
        raise ContractLanguageError("guarantee directives must be: guarantee <artifact-or-kind> provides <fact[,fact...]>", line_number)
    builder.guarantees.append(StaticContractGuarantee(artifact=tokens[1], provides=_parse_values(tokens[3])))


def _parse_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in next(csv.reader([raw], skipinitialspace=True)) if value.strip())
    if not values:
        raise ValueError("value lists must contain at least one item")
    return values


def _format_values(values: tuple[str, ...]) -> str:
    return ",".join(_quote(value) if _needs_quote(value) else value for value in values)


def _quote(value: str) -> str:
    return json.dumps(value)


def _needs_quote(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_.:/-]+", value) is None


def _strip_comment(line: str) -> str:
    in_quote = False
    escaped = False
    result: list[str] = []
    for char in line:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_quote:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
        if char == "#" and not in_quote:
            break
        result.append(char)
    return "".join(result)


def _expect_token(tokens: list[str], index: int, line_number: int, expected: str) -> str:
    if index >= len(tokens):
        raise ContractLanguageError(f"expected {expected}", line_number)
    return tokens[index]


@dataclass(slots=True)
class _RuleBuilder:
    name: str
    line: int
    rule_type: str = "interface"
    severity: str = "error"
    description: str | None = None
    applies_to: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ()
    required_regions: tuple[str, ...] = ()
    forbidden_delimiters: tuple[str, ...] = ()
    schema_obligations: list[StaticContractSchemaObligation] | None = None
    stop_policies: list[StaticContractStopPolicy] | None = None
    invariants: list[StaticContractInvariant] | None = None
    assumptions: list[StaticContractAssumption] | None = None
    guarantees: list[StaticContractGuarantee] | None = None

    def __post_init__(self) -> None:
        self.schema_obligations = []
        self.stop_policies = []
        self.invariants = []
        self.assumptions = []
        self.guarantees = []

    def build(self) -> tuple[StaticContractRule, int]:
        assert self.schema_obligations is not None
        assert self.stop_policies is not None
        assert self.invariants is not None
        assert self.assumptions is not None
        assert self.guarantees is not None
        return (
            StaticContractRule(
                name=self.name,
                rule_type=self.rule_type,
                severity=self.severity,
                description=self.description,
                applies_to=self.applies_to,
                allowed_roles=self.allowed_roles,
                required_regions=self.required_regions,
                forbidden_delimiters=self.forbidden_delimiters,
                schema_obligations=tuple(self.schema_obligations),
                stop_policies=tuple(self.stop_policies),
                invariants=tuple(self.invariants),
                assumptions=tuple(self.assumptions),
                guarantees=tuple(self.guarantees),
            ),
            self.line,
        )
