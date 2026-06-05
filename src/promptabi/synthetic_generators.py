"""Static checks for synthetic-data generator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .artifacts import SyntheticGeneratorSpec, SyntheticTruncationContract, TrainingManifestArtifact


class SyntheticGeneratorFindingKind(StrEnum):
    """Finite synthetic-generator contract outcomes."""

    ROLE_CONTRACT_VIOLATION = "role-contract-violation"
    SCHEMA_CONTRACT_VIOLATION = "schema-contract-violation"
    TOOL_CALL_CONTRACT_VIOLATION = "tool-call-contract-violation"
    TRUNCATION_CONTRACT_VIOLATION = "truncation-contract-violation"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class SyntheticGeneratorFinding:
    """One bounded synthetic-data generator contract finding."""

    kind: SyntheticGeneratorFindingKind
    manifest_name: str
    generator_name: str
    message: str
    severity: str
    subject: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class SyntheticGeneratorReport:
    """Static report for synthetic-data generator preflight checks."""

    manifest_name: str
    findings: tuple[SyntheticGeneratorFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is SyntheticGeneratorFindingKind.VERIFIED for finding in self.findings
        )


_STANDARD_STRUCTURAL_ROLES = (
    "assistant",
    "developer",
    "function",
    "system",
    "tool",
    "user",
)
_CONTRACT_METADATA_KEYS = ("synthetic_data_contract", "training_interface_contract", "interface_contract")


def analyze_synthetic_generators(manifest: TrainingManifestArtifact) -> SyntheticGeneratorReport:
    """Verify finite synthetic-data generator summaries before materialization.

    The analyzer never samples or opens private generated rows. It checks the
    generator's declared finite output summaries against role, schema, tool-call,
    and truncation contracts that can be proven from manifest metadata.
    """

    if not manifest.synthetic_generators:
        return SyntheticGeneratorReport(manifest_name=manifest.name, findings=())

    contract = _interface_contract(manifest)
    allowed_roles = _allowed_roles(contract)
    findings: list[SyntheticGeneratorFinding] = []
    for generator in manifest.synthetic_generators:
        findings.extend(_role_findings(manifest, generator, allowed_roles=allowed_roles))
        findings.extend(_schema_findings(manifest, generator))
        findings.extend(_tool_call_findings(manifest, generator))
        findings.extend(_truncation_findings(manifest, generator))

    if not findings:
        findings.append(
            _finding(
                manifest,
                manifest.synthetic_generators[0],
                SyntheticGeneratorFindingKind.VERIFIED,
                f"{len(manifest.synthetic_generators)} synthetic-data generator(s) satisfy finite role, schema, tool-call, and truncation contracts",
                "info",
                subject="synthetic_generators",
                witness=(
                    ("select synthetic generators", None, str(len(manifest.synthetic_generators))),
                    ("select allowed structural roles", None, ", ".join(sorted(allowed_roles))),
                    ("check generator output roles", None, "all finite roles allowed"),
                    ("check schema outputs", None, _count_summary(manifest.synthetic_generators, "schema")),
                    ("check tool-call outputs", None, _count_summary(manifest.synthetic_generators, "tool")),
                    ("check truncation cases", None, _count_summary(manifest.synthetic_generators, "truncation")),
                ),
            )
        )
    return SyntheticGeneratorReport(manifest_name=manifest.name, findings=tuple(findings))


def _role_findings(
    manifest: TrainingManifestArtifact,
    generator: SyntheticGeneratorSpec,
    *,
    allowed_roles: set[str],
) -> tuple[SyntheticGeneratorFinding, ...]:
    findings: list[SyntheticGeneratorFinding] = []
    output_roles = {_normalize_role(role) for role in generator.output_roles}
    for role in sorted(output_roles):
        if role not in allowed_roles:
            findings.append(
                _finding(
                    manifest,
                    generator,
                    SyntheticGeneratorFindingKind.ROLE_CONTRACT_VIOLATION,
                    f"synthetic generator '{generator.name}' can emit role '{role}' outside the declared role contract",
                    "error",
                    subject=f"synthetic_generators.{generator.name}.output_roles",
                    witness=(
                        ("select synthetic generator", generator.name, generator.generator_type),
                        ("inspect generated role", None, role),
                        ("select allowed roles", None, ", ".join(sorted(allowed_roles))),
                        ("prove role admissibility", None, "not allowed"),
                    ),
                )
            )
    for role in sorted({_normalize_role(role) for role in generator.forbidden_roles}.intersection(output_roles)):
        findings.append(
            _finding(
                manifest,
                generator,
                SyntheticGeneratorFindingKind.ROLE_CONTRACT_VIOLATION,
                f"synthetic generator '{generator.name}' emits forbidden role '{role}'",
                "error",
                subject=f"synthetic_generators.{generator.name}.forbidden_roles",
                witness=(
                    ("select synthetic generator", generator.name, generator.generator_type),
                    ("inspect forbidden role", None, role),
                    ("inspect generated roles", None, ", ".join(sorted(output_roles))),
                ),
            )
        )
    for role in sorted({_normalize_role(role) for role in generator.required_roles}.difference(output_roles)):
        findings.append(
            _finding(
                manifest,
                generator,
                SyntheticGeneratorFindingKind.ROLE_CONTRACT_VIOLATION,
                f"synthetic generator '{generator.name}' omits required role '{role}'",
                "error",
                subject=f"synthetic_generators.{generator.name}.required_roles",
                witness=(
                    ("select synthetic generator", generator.name, generator.generator_type),
                    ("inspect required role", None, role),
                    ("inspect generated roles", None, ", ".join(sorted(output_roles)) or "<none>"),
                ),
            )
        )
    return tuple(findings)


def _schema_findings(
    manifest: TrainingManifestArtifact,
    generator: SyntheticGeneratorSpec,
) -> tuple[SyntheticGeneratorFinding, ...]:
    findings: list[SyntheticGeneratorFinding] = []
    for case in generator.schema_outputs:
        if case.valid is not False and case.parses is not False and case.schema_valid is not False:
            continue
        findings.append(
            _finding(
                manifest,
                generator,
                SyntheticGeneratorFindingKind.SCHEMA_CONTRACT_VIOLATION,
                f"synthetic generator '{generator.name}' has schema output '{case.case_id}' that violates the declared parser/schema contract",
                "error",
                subject=f"synthetic_generators.{generator.name}.schema_outputs.{case.case_id}",
                witness=(
                    ("select schema output case", case.case_id, _display(case.reason)),
                    ("inspect valid flag", None, _display_bool(case.valid)),
                    ("inspect parse result", None, _display_bool(case.parses)),
                    ("inspect schema result", None, _display_bool(case.schema_valid)),
                ),
            )
        )
    return tuple(findings)


def _tool_call_findings(
    manifest: TrainingManifestArtifact,
    generator: SyntheticGeneratorSpec,
) -> tuple[SyntheticGeneratorFinding, ...]:
    findings: list[SyntheticGeneratorFinding] = []
    for case in generator.tool_calls:
        if case.valid is not False and case.malformed is not True:
            continue
        findings.append(
            _finding(
                manifest,
                generator,
                SyntheticGeneratorFindingKind.TOOL_CALL_CONTRACT_VIOLATION,
                f"synthetic generator '{generator.name}' has malformed tool-call output '{case.case_id}'",
                "error",
                subject=f"synthetic_generators.{generator.name}.tool_calls.{case.case_id}",
                witness=(
                    ("select tool-call output case", case.case_id, _display(case.reason)),
                    ("inspect valid flag", None, _display_bool(case.valid)),
                    ("inspect malformed flag", None, _display_bool(case.malformed)),
                ),
            )
        )
    return tuple(findings)


def _truncation_findings(
    manifest: TrainingManifestArtifact,
    generator: SyntheticGeneratorSpec,
) -> tuple[SyntheticGeneratorFinding, ...]:
    findings: list[SyntheticGeneratorFinding] = []
    required_roles = {_normalize_role(role) for role in generator.required_roles}
    for case in generator.truncation_cases:
        reasons = _truncation_violation_reasons(case, required_roles=required_roles)
        if not reasons:
            continue
        findings.append(
            _finding(
                manifest,
                generator,
                SyntheticGeneratorFindingKind.TRUNCATION_CONTRACT_VIOLATION,
                f"synthetic generator '{generator.name}' can truncate required contract content in case '{case.case_id}'",
                "error",
                subject=f"synthetic_generators.{generator.name}.truncation_cases.{case.case_id}",
                witness=(
                    ("select truncation case", case.case_id, _display(case.reason)),
                    ("inspect token budget", None, _token_budget_summary(case)),
                    ("inspect required roles", None, ", ".join(sorted(required_roles)) or "<none>"),
                    ("classify truncation contract", None, "; ".join(reasons)),
                ),
            )
        )
    return tuple(findings)


def _truncation_violation_reasons(
    case: SyntheticTruncationContract,
    *,
    required_roles: set[str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    truncated = {_normalize_role(role) for role in case.truncated_required_roles}
    if truncated:
        reasons.append(f"truncated required roles: {', '.join(sorted(truncated))}")
    preserved = {_normalize_role(role) for role in case.preserved_required_roles}
    if required_roles and preserved and not required_roles.issubset(preserved):
        missing = required_roles.difference(preserved)
        reasons.append(f"required roles not preserved: {', '.join(sorted(missing))}")
    if (
        case.input_tokens is not None
        and case.output_tokens is not None
        and case.max_context_tokens is not None
        and case.input_tokens + case.output_tokens > case.max_context_tokens
    ):
        reasons.append("input_tokens + output_tokens exceeds max_context_tokens")
    return tuple(reasons)


def _interface_contract(manifest: TrainingManifestArtifact) -> Mapping[str, object]:
    metadata = dict(manifest.metadata)
    for key in _CONTRACT_METADATA_KEYS:
        value = _mapping(metadata.get(key))
        if value:
            return value
    return {}


def _allowed_roles(contract: Mapping[str, object]) -> set[str]:
    roles = {_normalize_role(role) for role in _STANDARD_STRUCTURAL_ROLES}
    for key in ("allowed_roles", "roles", "canonical_roles"):
        value = contract.get(key)
        if isinstance(value, list):
            roles.update(_normalize_role(item) for item in value if isinstance(item, str) and item)
    return roles


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _normalize_role(role: str) -> str:
    return role.strip().lower().replace("_", "-")


def _display(value: str | None) -> str:
    return value if value is not None else "<missing>"


def _display_bool(value: bool | None) -> str:
    if value is None:
        return "<missing>"
    return str(value).lower()


def _token_budget_summary(case: SyntheticTruncationContract) -> str:
    return f"input={_display_int(case.input_tokens)}, output={_display_int(case.output_tokens)}, max={_display_int(case.max_context_tokens)}"


def _display_int(value: int | None) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _count_summary(generators: tuple[SyntheticGeneratorSpec, ...], surface: str) -> str:
    if surface == "schema":
        count = sum(len(generator.schema_outputs) for generator in generators)
    elif surface == "tool":
        count = sum(len(generator.tool_calls) for generator in generators)
    else:
        count = sum(len(generator.truncation_cases) for generator in generators)
    return f"{count} finite case(s)"


def _finding(
    manifest: TrainingManifestArtifact,
    generator: SyntheticGeneratorSpec,
    kind: SyntheticGeneratorFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> SyntheticGeneratorFinding:
    return SyntheticGeneratorFinding(
        kind=kind,
        manifest_name=manifest.name,
        generator_name=generator.name,
        message=message,
        severity=severity,
        subject=subject,
        witness=witness,
    )
