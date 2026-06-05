"""Migration support for deprecated PromptABI static-contract syntax."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .contract_language import (
    CONTRACT_LANGUAGE_VERSION,
    format_static_contract,
    parse_static_contract_text,
)
from .contract_linting import ContractLintFinding, lint_static_contract
from .diagnostics import SourceSpan


_CONTRACT_VERSION_RE = re.compile(r"^(\s*)contract\s+promptabi\.contract/v0\s*$")
_RULE_RE = re.compile(r"^(?P<indent>\s*)rule\s+(?P<name>[A-Za-z0-9_.:/-]+)(?P<attrs>.*):\s*$")
_SCHEMA_FIELDS_RE = re.compile(r"^(\s*)(requires_schema|schema_requires)\s+(\S+)\s+fields\s+(.+)$")
_STOP_POLICY_RE = re.compile(r"^(\s*)stop_policy\s+(\S+)\s+(sequences?|stops)\s+(.+)$")
_ASSERT_RE = re.compile(r"^(\s*)assert\s+(.+)$")

_RULE_ATTR_ALIASES = {
    "kind": ("type", "automata+solver"),
    "level": ("severity", "solver"),
    "artifacts": ("applies_to", "automata"),
    "surfaces": ("applies_to", "automata"),
}
_BODY_PREFIX_ALIASES = {
    "roles": ("allowed_roles", "automata"),
    "role_set": ("allowed_roles", "automata"),
    "requires_regions": ("required_regions", "automata"),
    "regions_required": ("required_regions", "automata"),
    "forbid_tokens": ("forbid_delimiters", "automata"),
    "forbidden_tokens": ("forbid_delimiters", "automata"),
    "forbidden_delimiters": ("forbid_delimiters", "automata"),
}


@dataclass(frozen=True, slots=True)
class ContractMigrationEdit:
    """One source rewrite and its verification impact."""

    code: str
    line: int
    before: str
    after: str
    backend: str
    behavior_change: str

    @property
    def span(self) -> SourceSpan:
        return SourceSpan(path="<memory>", start_line=self.line, start_column=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "line": self.line,
            "before": self.before,
            "after": self.after,
            "backend": self.backend,
            "behavior_change": self.behavior_change,
        }


@dataclass(frozen=True, slots=True)
class ContractMigrationReport:
    """Result of migrating one static-contract source."""

    contract_name: str
    changed: bool
    migrated_text: str
    edits: tuple[ContractMigrationEdit, ...]
    lint_findings: tuple[ContractLintFinding, ...]

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.lint_findings if finding.severity == "warning")

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.lint_findings if finding.severity == "error")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "contract_name": self.contract_name,
            "changed": self.changed,
            "edit_count": len(self.edits),
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "edits": [edit.to_dict() for edit in self.edits],
            "lint_findings": [finding.to_dict() for finding in self.lint_findings],
            "migrated_text": self.migrated_text,
        }


def migrate_static_contract_text(
    text: str,
    *,
    name: str = "contract",
    path: str | None = None,
) -> ContractMigrationReport:
    """Rewrite deprecated contract syntax and validate the migrated artifact."""

    rewritten_lines: list[str] = []
    edits: list[ContractMigrationEdit] = []
    invariant_index = 1
    for line_number, line in enumerate(text.splitlines(), start=1):
        migrated, line_edits, invariant_index = _migrate_line(line, line_number, invariant_index)
        rewritten_lines.append(migrated)
        edits.extend(line_edits)

    rewritten = "\n".join(rewritten_lines)
    if text.endswith("\n"):
        rewritten += "\n"
    contract = parse_static_contract_text(rewritten, name=name, path=path)
    canonical = format_static_contract(contract)
    if canonical != rewritten:
        edits.append(
            ContractMigrationEdit(
                code="canonical-format",
                line=1,
                before="source order/spacing",
                after="canonical .pabi order/spacing",
                backend="automata+solver",
                behavior_change="No semantic change; canonical formatting makes parser, automata, and SMT lowering deterministic.",
            )
        )
    lint = lint_static_contract(contract)
    return ContractMigrationReport(
        contract_name=contract.name,
        changed=bool(edits),
        migrated_text=canonical,
        edits=tuple(edits),
        lint_findings=lint.findings,
    )


def migrate_static_contract_file(path: Path, *, name: str | None = None) -> ContractMigrationReport:
    """Read and migrate a ``.pabi`` file."""

    return migrate_static_contract_text(path.read_text(encoding="utf-8"), name=name or path.stem, path=str(path))


def render_contract_migration_json(report: ContractMigrationReport) -> str:
    """Render a deterministic JSON migration report."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_contract_migration_text(report: ContractMigrationReport) -> str:
    """Render a human-readable migration report with backend impact notes."""

    lines = [
        f"PromptABI contract migration: {report.contract_name}",
        f"changed: {'yes' if report.changed else 'no'}",
        f"edits: {len(report.edits)}",
        f"lint_errors: {report.error_count}",
        f"lint_warnings: {report.warning_count}",
    ]
    if report.edits:
        lines.append("edits:")
        for edit in report.edits:
            lines.append(f"  {edit.code} line {edit.line} [{edit.backend}]")
            lines.append(f"    before: {edit.before}")
            lines.append(f"    after: {edit.after}")
            lines.append(f"    behavior: {edit.behavior_change}")
    if report.lint_findings:
        lines.append("post-migration lint:")
        for finding in report.lint_findings:
            rule = f" {finding.rule_name}" if finding.rule_name else ""
            lines.append(f"  {finding.severity.upper()} {finding.code}{rule}: {finding.message}")
            lines.append(f"    suggestion: {finding.suggestion}")
    return "\n".join(lines) + "\n"


def _migrate_line(
    line: str,
    line_number: int,
    invariant_index: int,
) -> tuple[str, tuple[ContractMigrationEdit, ...], int]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line, (), invariant_index

    edits: list[ContractMigrationEdit] = []
    version_match = _CONTRACT_VERSION_RE.match(line)
    if version_match:
        migrated = f"{version_match.group(1)}contract {CONTRACT_LANGUAGE_VERSION}"
        edits.append(
            _edit(
                "contract-version",
                line_number,
                line,
                migrated,
                "automata+solver",
                "Updates the file header to the supported v1 parser; rule semantics are preserved before automata or SMT lowering.",
            )
        )
        return migrated, tuple(edits), invariant_index

    rule_match = _RULE_RE.match(line)
    if rule_match:
        migrated, attr_edits = _migrate_rule_header(rule_match, line, line_number)
        return migrated, attr_edits, invariant_index

    body_migrated = _migrate_body_prefix(line, line_number, invariant_index)
    if body_migrated is not None:
        return body_migrated

    schema_match = _SCHEMA_FIELDS_RE.match(line)
    if schema_match:
        migrated = f"{schema_match.group(1)}schema {schema_match.group(3)} requires {schema_match.group(4)}"
        edits.append(
            _edit(
                "schema-fields",
                line_number,
                line,
                migrated,
                "solver",
                "Preserves required-field obligations while using the v1 schema directive consumed by finite precondition checks.",
            )
        )
        return migrated, tuple(edits), invariant_index

    stop_match = _STOP_POLICY_RE.match(line)
    if stop_match:
        tail = stop_match.group(4)
        migrated_tail = tail.replace(" inside ", " forbid_inside ", 1)
        migrated = f"{stop_match.group(1)}stop {stop_match.group(2)} stops {migrated_tail}"
        edits.append(
            _edit(
                "stop-policy",
                line_number,
                line,
                migrated,
                "automata",
                "Maps legacy stop_policy syntax to v1 stop directives; stop reachability and overreachability checks receive the same sequences.",
            )
        )
        return migrated, tuple(edits), invariant_index

    assert_match = _ASSERT_RE.match(line)
    if assert_match:
        migrated = f"{assert_match.group(1)}invariant migrated-{invariant_index}: {assert_match.group(2)}"
        edits.append(
            _edit(
                "assert-invariant",
                line_number,
                line,
                migrated,
                "solver",
                "Turns an unnamed assertion into a named invariant so SMT witnesses and unsat-core notes remain attributable.",
            )
        )
        return migrated, tuple(edits), invariant_index + 1

    return line, (), invariant_index


def _migrate_rule_header(
    match: re.Match[str],
    original: str,
    line_number: int,
) -> tuple[str, tuple[ContractMigrationEdit, ...]]:
    tokens = shlex.split(match.group("attrs"))
    if not tokens:
        return original, ()
    migrated_tokens: list[str] = []
    edits: list[ContractMigrationEdit] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        replacement = _RULE_ATTR_ALIASES.get(token)
        if replacement is None:
            migrated_tokens.append(token)
        else:
            replacement_token, backend = replacement
            migrated_tokens.append(replacement_token)
            edits.append(
                _edit(
                    f"rule-attribute-{token}",
                    line_number,
                    token,
                    replacement_token,
                    backend,
                    f"Renames rule attribute {token!r} to {replacement_token!r}; the parsed value and downstream check obligations are unchanged.",
                )
            )
        if index + 1 < len(tokens):
            migrated_tokens.append(tokens[index + 1])
        index += 2
    attrs = f" {' '.join(_quote_token(token) for token in migrated_tokens)}" if migrated_tokens else ""
    migrated = f"{match.group('indent')}rule {match.group('name')}{attrs}:"
    if edits:
        edits.append(
            _edit(
                "rule-header",
                line_number,
                original,
                migrated,
                "automata+solver",
                "Only directive names changed; severity, surface selection, and rule type feed the same v1 lowering.",
            )
        )
    return migrated, tuple(edits)


def _migrate_body_prefix(
    line: str,
    line_number: int,
    invariant_index: int,
) -> tuple[str, tuple[ContractMigrationEdit, ...], int] | None:
    indent_length = len(line) - len(line.lstrip())
    indent = line[:indent_length]
    stripped = line[indent_length:]
    if not stripped:
        return None
    head, _, tail = stripped.partition(" ")
    replacement = _BODY_PREFIX_ALIASES.get(head)
    if replacement is None:
        return None
    replacement_head, backend = replacement
    migrated = f"{indent}{replacement_head} {tail}" if tail else f"{indent}{replacement_head}"
    return (
        migrated,
        (
            _edit(
                f"directive-{head}",
                line_number,
                line,
                migrated,
                backend,
                f"Renames body directive {head!r} to {replacement_head!r}; the finite language or solver constraint is unchanged.",
            ),
        ),
        invariant_index,
    )


def _edit(
    code: str,
    line: int,
    before: str,
    after: str,
    backend: str,
    behavior_change: str,
) -> ContractMigrationEdit:
    return ContractMigrationEdit(
        code=code,
        line=line,
        before=before.strip(),
        after=after.strip(),
        backend=backend,
        behavior_change=behavior_change,
    )


def _quote_token(token: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:/,<>|`\"'{}\[\]-]+", token):
        return token
    return json.dumps(token)
