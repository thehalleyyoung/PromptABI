"""Trained-on-invalid-interface checks for training manifests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .artifacts import TrainingManifestArtifact


class TrainingInvalidInterfaceFindingKind(StrEnum):
    """Bounded invalid-interface outcomes for training data."""

    CONTRACT_MISSING = "contract-missing"
    IMPOSSIBLE_ROLE = "impossible-role"
    MALFORMED_TOOL_CALL = "malformed-tool-call"
    INVALID_JSON_OUTPUT = "invalid-json-output"
    UNREACHABLE_STOP_SEQUENCE = "unreachable-stop-sequence"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class TrainingInvalidInterfaceFinding:
    """One finite invalid-interface finding for a training manifest."""

    kind: TrainingInvalidInterfaceFindingKind
    manifest_name: str
    message: str
    severity: str
    subject: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingInvalidInterfaceReport:
    """Bounded report for invalid prompt-interface facts in training data."""

    manifest_name: str
    findings: tuple[TrainingInvalidInterfaceFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is TrainingInvalidInterfaceFindingKind.VERIFIED for finding in self.findings
        )


_CONTRACT_METADATA_KEYS = ("training_interface_contract", "trained_interface_contract", "interface_contract")
_STANDARD_STRUCTURAL_ROLES = (
    "assistant",
    "developer",
    "function",
    "system",
    "tool",
    "user",
)


def analyze_training_invalid_interface(manifest: TrainingManifestArtifact) -> TrainingInvalidInterfaceReport:
    """Detect finite evidence that training data encodes invalid interfaces.

    The analyzer only inspects manifest-level structural facts and optional
    ``metadata.training_interface_contract`` summaries. It does not open private
    dataset rows, but it can prove that declared roles, tool-call examples, JSON
    outputs, or stop examples are impossible under the finite facts supplied by
    a data-preparation job.
    """

    contract = _interface_contract(manifest)
    if not contract:
        return TrainingInvalidInterfaceReport(
            manifest_name=manifest.name,
            findings=(
                _finding(
                    manifest,
                    TrainingInvalidInterfaceFindingKind.CONTRACT_MISSING,
                    f"training manifest '{manifest.name}' has no training_interface_contract metadata",
                    "info",
                    subject="metadata.training_interface_contract",
                    witness=(
                        ("select training manifest", manifest.name, _dataset_summary(manifest)),
                        ("inspect invalid-interface evidence", None, "missing"),
                    ),
                ),
            ),
        )

    findings: list[TrainingInvalidInterfaceFinding] = []
    allowed_roles = _allowed_roles(contract)
    findings.extend(_role_findings(manifest, contract, allowed_roles=allowed_roles))
    findings.extend(_tool_call_findings(manifest, contract))
    findings.extend(_json_output_findings(manifest, contract))
    findings.extend(_stop_sequence_findings(manifest, contract))

    if not findings:
        findings.append(
            _finding(
                manifest,
                TrainingInvalidInterfaceFindingKind.VERIFIED,
                f"training manifest '{manifest.name}' contains no finite invalid-interface examples",
                "info",
                subject="metadata.training_interface_contract",
                witness=(
                    ("select allowed structural roles", None, ", ".join(sorted(allowed_roles))),
                    ("check declared role usages", None, "all finite roles allowed"),
                    ("check tool-call examples", None, _list_summary(contract, "tool_calls")),
                    ("check JSON-output examples", None, _list_summary(contract, "json_outputs")),
                    ("check stop-sequence examples", None, _list_summary(contract, "stop_sequences")),
                ),
            )
        )
    return TrainingInvalidInterfaceReport(manifest_name=manifest.name, findings=tuple(findings))


def _role_findings(
    manifest: TrainingManifestArtifact,
    contract: Mapping[str, object],
    *,
    allowed_roles: set[str],
) -> tuple[TrainingInvalidInterfaceFinding, ...]:
    findings: list[TrainingInvalidInterfaceFinding] = []
    role_values: list[tuple[str, str]] = []
    role_values.extend((f"message_roles.{index}", role) for index, role in enumerate(manifest.message_roles))
    role_values.extend((f"target_roles.{index}", role) for index, role in enumerate(manifest.target_roles))
    role_values.extend(
        (f"role_labels.{label.source_role}.canonical_role", label.canonical_role)
        for label in manifest.role_labels
    )
    if manifest.loss_mask_policy is not None:
        role_values.extend(
            (f"loss_mask_policy.target_roles.{index}", role)
            for index, role in enumerate(manifest.loss_mask_policy.target_roles)
        )
        role_values.extend(
            (f"loss_mask_policy.ignored_roles.{index}", role)
            for index, role in enumerate(manifest.loss_mask_policy.ignored_roles)
        )
    for span in manifest.supervised_spans:
        role_values.append((f"supervised_spans.{span.span_id}.target_role", span.target_role))
        role_values.append((f"supervised_spans.{span.span_id}.rendered_region_role", span.rendered_region_role))
    for pair in manifest.preference_pairs:
        role_values.extend(
            (f"preference_pairs.{pair.pair_id}.chosen_role_layout.{index}", role)
            for index, role in enumerate(pair.chosen_role_layout)
        )
        role_values.extend(
            (f"preference_pairs.{pair.pair_id}.rejected_role_layout.{index}", role)
            for index, role in enumerate(pair.rejected_role_layout)
        )
    for item in _list_of_mappings(contract.get("role_examples")):
        role = _string_value(item.get("role"))
        subject = _string_value(item.get("id")) or _string_value(item.get("example_id")) or "role_examples"
        if role is not None:
            role_values.append((f"metadata.training_interface_contract.role_examples.{subject}", role))

    for subject, role in sorted(role_values):
        if _normalize_role(role) in allowed_roles:
            continue
        findings.append(
            _finding(
                manifest,
                TrainingInvalidInterfaceFindingKind.IMPOSSIBLE_ROLE,
                f"training data uses role '{role}' outside the declared prompt-interface role set",
                "error",
                subject=subject,
                witness=(
                    ("select role usage", subject, role),
                    ("select allowed roles", None, ", ".join(sorted(allowed_roles))),
                    ("prove role admissibility", None, "not allowed"),
                ),
            )
        )
    return tuple(findings)


def _tool_call_findings(
    manifest: TrainingManifestArtifact,
    contract: Mapping[str, object],
) -> tuple[TrainingInvalidInterfaceFinding, ...]:
    findings: list[TrainingInvalidInterfaceFinding] = []
    for index, item in enumerate(_list_of_mappings(contract.get("tool_calls"))):
        valid = _bool_value(item.get("valid"))
        malformed = _bool_value(item.get("malformed"))
        if valid is not False and malformed is not True:
            continue
        example_id = _example_id(item, index)
        reason = _reason(item)
        findings.append(
            _finding(
                manifest,
                TrainingInvalidInterfaceFindingKind.MALFORMED_TOOL_CALL,
                f"training tool-call example '{example_id}' is malformed under the declared tool-call contract",
                "error",
                subject=f"metadata.training_interface_contract.tool_calls.{example_id}",
                witness=(
                    ("select tool-call example", example_id, _display(reason)),
                    ("inspect valid flag", None, _display_bool(valid)),
                    ("inspect malformed flag", None, _display_bool(malformed)),
                ),
            )
        )
    return tuple(findings)


def _json_output_findings(
    manifest: TrainingManifestArtifact,
    contract: Mapping[str, object],
) -> tuple[TrainingInvalidInterfaceFinding, ...]:
    findings: list[TrainingInvalidInterfaceFinding] = []
    for index, item in enumerate(_list_of_mappings(contract.get("json_outputs"))):
        valid = _bool_value(item.get("valid"))
        parses = _bool_value(item.get("parses"))
        schema_valid = _bool_value(item.get("schema_valid"))
        if valid is not False and parses is not False and schema_valid is not False:
            continue
        example_id = _example_id(item, index)
        reason = _reason(item)
        findings.append(
            _finding(
                manifest,
                TrainingInvalidInterfaceFindingKind.INVALID_JSON_OUTPUT,
                f"training JSON output example '{example_id}' is invalid under the declared parser/schema contract",
                "error",
                subject=f"metadata.training_interface_contract.json_outputs.{example_id}",
                witness=(
                    ("select JSON output example", example_id, _display(reason)),
                    ("inspect parse result", None, _display_bool(parses)),
                    ("inspect schema result", None, _display_bool(schema_valid)),
                ),
            )
        )
    return tuple(findings)


def _stop_sequence_findings(
    manifest: TrainingManifestArtifact,
    contract: Mapping[str, object],
) -> tuple[TrainingInvalidInterfaceFinding, ...]:
    findings: list[TrainingInvalidInterfaceFinding] = []
    for index, item in enumerate(_list_of_mappings(contract.get("stop_sequences"))):
        reachable = _bool_value(item.get("reachable"))
        matching_examples = _int_value(item.get("matching_examples"))
        if reachable is not False and matching_examples != 0:
            continue
        sequence = _string_value(item.get("sequence")) or _string_value(item.get("stop")) or f"stop-{index}"
        findings.append(
            _finding(
                manifest,
                TrainingInvalidInterfaceFindingKind.UNREACHABLE_STOP_SEQUENCE,
                f"training stop sequence '{sequence}' is unreachable in the declared training interface",
                "error",
                subject=f"metadata.training_interface_contract.stop_sequences.{sequence}",
                witness=(
                    ("select stop sequence", None, sequence),
                    ("inspect reachability", None, _display_bool(reachable)),
                    ("inspect matching examples", None, _display_int(matching_examples)),
                ),
            )
        )
    return tuple(findings)


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


def _list_of_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_mapping(item) for item in value if isinstance(item, dict))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _normalize_role(role: str) -> str:
    return role.strip().lower().replace("_", "-")


def _example_id(item: Mapping[str, object], index: int) -> str:
    return (
        _string_value(item.get("id"))
        or _string_value(item.get("example_id"))
        or _string_value(item.get("call_id"))
        or f"{index}"
    )


def _reason(item: Mapping[str, object]) -> str | None:
    return (
        _string_value(item.get("reason"))
        or _string_value(item.get("parser_error"))
        or _string_value(item.get("schema_error"))
        or _string_value(item.get("error"))
    )


def _list_summary(contract: Mapping[str, object], key: str) -> str:
    return f"{len(_list_of_mappings(contract.get(key)))} finite example(s)"


def _dataset_summary(manifest: TrainingManifestArtifact) -> str:
    if not manifest.datasets:
        return "0 dataset declarations"
    return ", ".join(f"{dataset.name}:{dataset.kind.value}" for dataset in manifest.datasets)


def _display(value: str | None) -> str:
    return value if value is not None else "<missing>"


def _display_bool(value: bool | None) -> str:
    if value is None:
        return "<missing>"
    return str(value).lower()


def _display_int(value: int | None) -> str:
    if value is None:
        return "<missing>"
    return str(value)


def _finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingInvalidInterfaceFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingInvalidInterfaceFinding:
    return TrainingInvalidInterfaceFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        subject=subject,
        witness=witness,
    )
