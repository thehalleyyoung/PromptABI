"""Static redaction checks for training-manifest witnesses and reports."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from .artifacts import (
    TrainingManifestArtifact,
    TrainingRedactionMode,
    TrainingRedactionPolicy,
    TrainingSourceContribution,
    TrainingSpanContract,
)


class TrainingRedactionFindingKind(StrEnum):
    """Training redaction outcomes over finite manifest metadata."""

    POLICY_MISSING = "policy-missing"
    RAW_WITNESS_FIELD = "raw-witness-field"
    HASH_MISSING = "hash-missing"
    SECRET_MATERIAL = "secret-material"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class TrainingRedactionFinding:
    """One non-sensitive redaction finding for a training manifest."""

    kind: TrainingRedactionFindingKind
    manifest_name: str
    message: str
    severity: str
    subject: str | None = None
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingRedactionReport:
    """Static privacy report for training-manifest evidence."""

    manifest_name: str
    findings: tuple[TrainingRedactionFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is TrainingRedactionFindingKind.VERIFIED for finding in self.findings
        )


_DEFAULT_RESTRICTED_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "credential",
    "email",
    "password",
    "provider_key",
    "secret",
    "token",
)
_DEFAULT_ALLOWED_REPORT_FIELDS = (
    "dataset_count",
    "example_count",
    "pair_id",
    "source_field",
    "source_id",
    "source_kind",
    "span_id",
    "target_role",
    "text_sha256",
    "token_range",
)
_DEFAULT_FORBIDDEN_REPORT_FIELDS = (
    "api_key",
    "authorization",
    "content",
    "prompt",
    "raw",
    "raw_text",
    "secret",
    "system_message",
    "text",
)
_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer-token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
    "github-token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    "openai-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    "anthropic-key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    "provider-key": re.compile(
        r"\b(?:sk-(?:proj-)?|sk-ant-|AKIA|gh[pousr]_)[A-Za-z0-9_-]{12,}\b",
        re.IGNORECASE,
    ),
}


def analyze_training_redaction(manifest: TrainingManifestArtifact) -> TrainingRedactionReport:
    """Verify training evidence can be persisted without raw private content.

    The analysis is deliberately static and local: it inspects only manifest
    declarations, source-contribution facts, hashes, field names, and artifact
    metadata. It never opens dataset rows or provider fixtures.
    """

    findings: list[TrainingRedactionFinding] = []
    policy = manifest.redaction_policy
    if policy is None:
        findings.append(
            _finding(
                manifest,
                TrainingRedactionFindingKind.POLICY_MISSING,
                f"training manifest '{manifest.name}' has no redaction_policy for stored witnesses or reports",
                "warning",
                subject="redaction_policy",
                witness=(
                    ("inspect redaction policy", None, "missing"),
                    ("classify persisted evidence", None, _evidence_summary(manifest)),
                ),
            )
        )
        policy = TrainingRedactionPolicy()

    findings.extend(_policy_findings(manifest, policy))
    findings.extend(_source_contribution_findings(manifest, policy))
    findings.extend(_preference_pair_findings(manifest, policy))
    findings.extend(_metadata_findings(manifest, policy))

    if not findings:
        findings.append(
            _finding(
                manifest,
                TrainingRedactionFindingKind.VERIFIED,
                f"training manifest '{manifest.name}' stores only structural or hashed training evidence",
                "info",
                subject="redaction_policy",
                witness=(
                    ("inspect redaction policy", None, policy.mode.value),
                    ("verify source contribution hashes", None, f"{_source_contribution_count(manifest)} contribution(s)"),
                    ("scan report field allowlist", None, "no raw text fields"),
                    ("scan structural metadata", None, "no restricted keys or provider-key patterns"),
                ),
            )
        )
    return TrainingRedactionReport(manifest_name=manifest.name, findings=tuple(findings))


def _policy_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    forbidden_fields = set(_DEFAULT_FORBIDDEN_REPORT_FIELDS).union(policy.forbidden_report_fields)
    allowed_fields = set(_DEFAULT_ALLOWED_REPORT_FIELDS).union(policy.allowed_report_fields)
    if policy.allow_raw_text_in_witnesses:
        findings.append(
            _finding(
                manifest,
                TrainingRedactionFindingKind.RAW_WITNESS_FIELD,
                "redaction_policy allows raw text in persisted training witnesses",
                "error",
                subject="redaction_policy.allow_raw_text_in_witnesses",
                witness=(("inspect witness text policy", None, "allow_raw_text_in_witnesses=true"),),
            )
        )
    if policy.mode is TrainingRedactionMode.STRUCTURAL and policy.require_text_hashes:
        findings.append(
            _finding(
                manifest,
                TrainingRedactionFindingKind.RAW_WITNESS_FIELD,
                "redaction_policy mixes structural mode with required text hashes; use hash-only or metadata-only",
                "warning",
                subject="redaction_policy.mode",
                witness=(("inspect redaction mode", None, "structural"),),
            )
        )
    for field_name in sorted(allowed_fields.intersection(forbidden_fields)):
        findings.append(
            _finding(
                manifest,
                TrainingRedactionFindingKind.RAW_WITNESS_FIELD,
                f"report field '{field_name}' is both allowed and forbidden by the redaction policy",
                "error",
                subject=f"redaction_policy.allowed_report_fields.{field_name}",
                witness=(
                    ("compare report field allowlist", field_name, "conflicts with forbidden raw field"),
                ),
            )
        )
    for field_name in sorted(allowed_fields):
        if _field_name_is_restricted(field_name, policy):
            findings.append(
                _finding(
                    manifest,
                    TrainingRedactionFindingKind.RAW_WITNESS_FIELD,
                    f"report field '{field_name}' is a restricted metadata or raw-content field",
                    "error",
                    subject=f"redaction_policy.allowed_report_fields.{field_name}",
                    witness=(("classify report field", field_name, "restricted"),),
                )
            )
    return tuple(findings)


def _source_contribution_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    for span in manifest.supervised_spans:
        findings.extend(_text_value_findings(manifest, policy, f"supervised_spans.{span.span_id}.span_id", span.span_id))
        for index, contribution in enumerate(span.source_contributions):
            subject = f"supervised_spans.{span.span_id}.source_contributions.{index}"
            if policy.require_text_hashes and not _is_hash_ref(contribution.text_sha256):
                findings.append(
                    _finding(
                        manifest,
                        TrainingRedactionFindingKind.HASH_MISSING,
                        f"source contribution {index} in span '{span.span_id}' lacks a reproducible text_sha256",
                        "error",
                        subject=f"{subject}.text_sha256",
                        witness=(
                            ("select source contribution", subject, _contribution_range(contribution)),
                            ("inspect text hash", None, "missing or non-sha256"),
                        ),
                    )
                )
            findings.extend(_source_field_findings(manifest, policy, span, contribution, index))
    return tuple(findings)


def _source_field_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
    span: TrainingSpanContract,
    contribution: TrainingSourceContribution,
    contribution_index: int,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    values = (
        ("source_id", contribution.source_id),
        ("source_field", contribution.source_field),
        ("transform", contribution.transform),
        ("text_sha256", contribution.text_sha256),
    )
    for field_name, value in values:
        if value is None:
            continue
        subject = f"supervised_spans.{span.span_id}.source_contributions.{contribution_index}.{field_name}"
        findings.extend(_text_value_findings(manifest, policy, subject, value))
        if field_name == "source_field" and _field_name_is_restricted(value, policy):
            findings.append(
                _finding(
                    manifest,
                    TrainingRedactionFindingKind.SECRET_MATERIAL,
                    f"source contribution {contribution_index} in span '{span.span_id}' references a restricted metadata field",
                    "error",
                    subject=subject,
                    witness=(("classify source field", value, "restricted metadata key"),),
                )
            )
    return tuple(findings)


def _preference_pair_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    for index, pair in enumerate(manifest.preference_pairs):
        hash_fields = (
            ("prompt_sha256", pair.prompt_sha256),
            ("chosen_sha256", pair.chosen_sha256),
            ("rejected_sha256", pair.rejected_sha256),
            ("chosen_prompt_sha256", pair.chosen_prompt_sha256),
            ("rejected_prompt_sha256", pair.rejected_prompt_sha256),
        )
        for field_name, value in hash_fields:
            if value is not None and not _is_hash_ref(value):
                findings.append(
                    _preference_hash_finding(manifest, index, field_name)
                )
        for field_name, value in (
            ("pair_id", pair.pair_id),
            ("chosen_tokenizer", pair.chosen_tokenizer),
            ("rejected_tokenizer", pair.rejected_tokenizer),
            ("chosen_mask_policy", pair.chosen_mask_policy),
            ("rejected_mask_policy", pair.rejected_mask_policy),
        ):
            findings.extend(_text_value_findings(manifest, policy, f"preference_pairs.{index}.{field_name}", value))
    return tuple(findings)


def _metadata_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    for key, value in manifest.metadata:
        subject = f"metadata.{key}"
        if _field_name_is_restricted(key, policy):
            findings.append(
                _finding(
                    manifest,
                    TrainingRedactionFindingKind.SECRET_MATERIAL,
                    f"training manifest metadata key '{key}' is restricted from reports",
                    "error",
                    subject=subject,
                    witness=(("classify metadata key", key, "restricted"),),
                )
            )
        findings.extend(_text_value_findings(manifest, policy, subject, str(value)))
    return tuple(findings)


def _preference_hash_finding(
    manifest: TrainingManifestArtifact,
    pair_index: int,
    field_name: str,
) -> TrainingRedactionFinding:
    return _finding(
        manifest,
        TrainingRedactionFindingKind.HASH_MISSING,
        f"preference pair {pair_index} field '{field_name}' is not a sha256 reference",
        "error",
        subject=f"preference_pairs.{pair_index}.{field_name}",
        witness=(
            ("select preference pair", f"preference_pairs.{pair_index}", "hash-only evidence required"),
            ("inspect hash field", field_name, "missing sha256 prefix or digest"),
        ),
    )


def _text_value_findings(
    manifest: TrainingManifestArtifact,
    policy: TrainingRedactionPolicy,
    subject: str,
    value: str,
) -> tuple[TrainingRedactionFinding, ...]:
    findings: list[TrainingRedactionFinding] = []
    for pattern_name, pattern in _active_secret_patterns(policy):
        if pattern.search(value):
            findings.append(
                _finding(
                    manifest,
                    TrainingRedactionFindingKind.SECRET_MATERIAL,
                    f"training manifest field '{subject}' matches secret pattern '{pattern_name}'",
                    "error",
                    subject=subject,
                    witness=(
                        ("scan structural field", subject, "secret-like value detected"),
                        ("redact matched value", None, _redacted_value(value)),
                    ),
                )
            )
    return tuple(findings)


def _field_name_is_restricted(field_name: str, policy: TrainingRedactionPolicy) -> bool:
    normalized = _normalize_field(field_name)
    restricted = {_normalize_field(key) for key in (*_DEFAULT_RESTRICTED_KEYS, *policy.restricted_metadata_keys)}
    forbidden = {_normalize_field(key) for key in (*_DEFAULT_FORBIDDEN_REPORT_FIELDS, *policy.forbidden_report_fields)}
    return any(part in restricted or part in forbidden for part in normalized.split("."))


def _active_secret_patterns(policy: TrainingRedactionPolicy) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if not policy.secret_patterns:
        names = ("provider-key", "bearer-token")
    else:
        names = policy.secret_patterns
    return tuple((name, _SECRET_PATTERNS[name]) for name in names if name in _SECRET_PATTERNS)


def _is_hash_ref(value: str | None) -> bool:
    if value is None:
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)


def _normalize_field(value: str) -> str:
    return re.sub(r"[^a-z0-9_.]+", "_", value.lower()).strip("_.")


def _redacted_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"len={len(value)},sha256-prefix={digest}"


def _source_contribution_count(manifest: TrainingManifestArtifact) -> int:
    return sum(len(span.source_contributions) for span in manifest.supervised_spans)


def _evidence_summary(manifest: TrainingManifestArtifact) -> str:
    return (
        f"{len(manifest.supervised_spans)} span(s), "
        f"{_source_contribution_count(manifest)} source contribution(s), "
        f"{len(manifest.preference_pairs)} preference pair(s)"
    )


def _contribution_range(contribution: TrainingSourceContribution) -> str:
    return f"{contribution.source_kind.value}:{contribution.start_token}:{contribution.end_token}"


def _finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingRedactionFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingRedactionFinding:
    return TrainingRedactionFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        subject=subject,
        witness=witness,
    )
