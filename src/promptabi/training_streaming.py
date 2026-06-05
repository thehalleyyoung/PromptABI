"""Streaming-dataset verification for large or private training manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .artifacts import TrainingDatasetSpec, TrainingManifestArtifact


class TrainingStreamingFindingKind(StrEnum):
    """Bounded streaming dataset verification outcomes."""

    CONTRACT_MISSING = "contract-missing"
    STREAM_UNREADABLE = "stream-unreadable"
    SAMPLE_EMPTY = "sample-empty"
    FIELD_MISSING = "field-missing"
    FORBIDDEN_FIELD = "forbidden-field"
    ROLE_VIOLATION = "role-violation"
    CHUNK_OVERSIZED = "chunk-oversized"
    COUNT_MISMATCH = "count-mismatch"
    HASH_MISSING = "hash-missing"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class StreamingDatasetSpec:
    """One bounded streaming sample declared by a training manifest."""

    name: str
    path: str
    dataset: str | None
    sample_rows: int
    chunk_size: int
    content_fields: tuple[str, ...]
    preference_fields: tuple[str, ...]
    hash_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    messages_field: str
    max_row_bytes: int | None
    expected_sample_rows: int | None


@dataclass(frozen=True, slots=True)
class TrainingStreamingFinding:
    """One streaming sample invariant violation or proof."""

    kind: TrainingStreamingFindingKind
    manifest_name: str
    message: str
    severity: str
    subject: str | None = None
    sample_name: str | None = None
    rows_sampled: int = 0
    chunks_sampled: int = 0
    witness: tuple[tuple[str, str | None, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingStreamingReport:
    """Bounded streaming verification report for one training manifest."""

    manifest_name: str
    findings: tuple[TrainingStreamingFinding, ...]

    @property
    def verified(self) -> bool:
        return bool(self.findings) and all(
            finding.kind is TrainingStreamingFindingKind.VERIFIED for finding in self.findings
        )


_STREAMING_METADATA_KEYS = (
    "streaming_dataset_verification",
    "streaming_datasets",
    "streaming_dataset_samples",
)
_STANDARD_ROLES = ("assistant", "developer", "function", "system", "tool", "user")


def analyze_training_streaming(
    manifest: TrainingManifestArtifact,
    *,
    base_dir: str | Path | None = None,
) -> TrainingStreamingReport:
    """Sample streaming datasets without materializing the corpus.

    The checker reads only the declared JSONL prefix sample and validates chunk
    shape, required structural fields, role labels, hash-only evidence, and row
    byte bounds. It never accumulates the full corpus in memory.
    """

    dataset_index = {dataset.name: dataset for dataset in manifest.datasets}
    try:
        specs = _streaming_specs(manifest, dataset_index=dataset_index)
    except ValueError as exc:
        return TrainingStreamingReport(
            manifest_name=manifest.name,
            findings=(
                _finding(
                    manifest,
                    TrainingStreamingFindingKind.STREAM_UNREADABLE,
                    f"training manifest streaming dataset contract is malformed: {exc}",
                    "error",
                    subject="metadata.streaming_dataset_verification",
                    witness=(("parse streaming dataset contract", None, str(exc)),),
                ),
            ),
        )

    if not specs:
        return TrainingStreamingReport(
            manifest_name=manifest.name,
            findings=(
                _finding(
                    manifest,
                    TrainingStreamingFindingKind.CONTRACT_MISSING,
                    f"training manifest '{manifest.name}' has no streaming dataset verification contract",
                    "info",
                    subject="metadata.streaming_dataset_verification",
                    witness=(
                        ("select training manifest", manifest.name, f"{len(manifest.datasets)} dataset declaration(s)"),
                        ("inspect streaming contract", None, "missing"),
                    ),
                ),
            ),
        )

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    findings: list[TrainingStreamingFinding] = []
    verified_rows = 0
    verified_chunks = 0
    for spec in specs:
        sample_findings, rows_sampled, chunks_sampled = _verify_streaming_spec(
            manifest,
            spec,
            base_dir=root,
        )
        findings.extend(sample_findings)
        if not sample_findings:
            verified_rows += rows_sampled
            verified_chunks += chunks_sampled
            findings.append(
                _finding(
                    manifest,
                    TrainingStreamingFindingKind.VERIFIED,
                    f"streaming dataset sample '{spec.name}' preserves chunk-level invariants",
                    "info",
                    subject=f"metadata.streaming_dataset_verification.{spec.name}",
                    sample_name=spec.name,
                    rows_sampled=rows_sampled,
                    chunks_sampled=chunks_sampled,
                    witness=(
                        ("open dataset stream", spec.path, "jsonl iterator"),
                        ("sample rows", None, str(rows_sampled)),
                        ("validate chunks", None, f"{chunks_sampled} chunk(s) of <= {spec.chunk_size} row(s)"),
                        ("validate fields", None, ", ".join((*spec.content_fields, *spec.preference_fields)) or "<none>"),
                        ("validate roles and hashes", None, "all sampled rows satisfy declared invariants"),
                    ),
                )
            )
    return TrainingStreamingReport(manifest_name=manifest.name, findings=tuple(findings))


def _verify_streaming_spec(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    *,
    base_dir: Path,
) -> tuple[list[TrainingStreamingFinding], int, int]:
    findings: list[TrainingStreamingFinding] = []
    path = Path(spec.path)
    if not path.is_absolute():
        path = base_dir / path
    rows_sampled = 0
    chunks_sampled = 0
    chunk_rows = 0
    required_fields = tuple(dict.fromkeys((*spec.content_fields, *spec.preference_fields)))
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [
            _finding(
                manifest,
                TrainingStreamingFindingKind.STREAM_UNREADABLE,
                f"streaming dataset sample '{spec.name}' could not be opened: {exc}",
                "error",
                subject=f"metadata.streaming_dataset_verification.{spec.name}.path",
                sample_name=spec.name,
                witness=(("open dataset stream", str(path), exc.__class__.__name__),),
            )
        ], 0, 0

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if rows_sampled >= spec.sample_rows:
                break
            if not line.strip():
                continue
            row_bytes = len(line.encode("utf-8"))
            if spec.max_row_bytes is not None and row_bytes > spec.max_row_bytes:
                findings.append(
                    _sample_finding(
                        manifest,
                        spec,
                        TrainingStreamingFindingKind.CHUNK_OVERSIZED,
                        f"streaming dataset sample '{spec.name}' row {line_number} exceeds max_row_bytes",
                        "error",
                        f"row[{line_number}]",
                        rows_sampled,
                        chunks_sampled,
                        (
                            ("read JSONL row", str(line_number), f"{row_bytes} byte(s)"),
                            ("compare row byte bound", None, f"max_row_bytes={spec.max_row_bytes}"),
                        ),
                    )
                )
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                findings.append(
                    _sample_finding(
                        manifest,
                        spec,
                        TrainingStreamingFindingKind.STREAM_UNREADABLE,
                        f"streaming dataset sample '{spec.name}' row {line_number} is not valid JSONL",
                        "error",
                        f"row[{line_number}]",
                        rows_sampled,
                        chunks_sampled,
                        (("parse JSONL row", str(line_number), exc.msg),),
                    )
                )
                continue
            if not isinstance(payload, Mapping):
                findings.append(
                    _sample_finding(
                        manifest,
                        spec,
                        TrainingStreamingFindingKind.STREAM_UNREADABLE,
                        f"streaming dataset sample '{spec.name}' row {line_number} is not a JSON object",
                        "error",
                        f"row[{line_number}]",
                        rows_sampled,
                        chunks_sampled,
                        (("parse JSONL row", str(line_number), type(payload).__name__),),
                    )
                )
                continue

            rows_sampled += 1
            chunk_rows += 1
            if chunk_rows == 1:
                chunks_sampled += 1
            if chunk_rows > spec.chunk_size:
                findings.append(
                    _sample_finding(
                        manifest,
                        spec,
                        TrainingStreamingFindingKind.CHUNK_OVERSIZED,
                        f"streaming dataset sample '{spec.name}' chunk {chunks_sampled} exceeds declared chunk_size",
                        "error",
                        f"chunk[{chunks_sampled}]",
                        rows_sampled,
                        chunks_sampled,
                        (
                            ("advance stream chunk", str(chunks_sampled), f"{chunk_rows} row(s)"),
                            ("compare chunk bound", None, f"chunk_size={spec.chunk_size}"),
                        ),
                    )
                )
            if chunk_rows >= spec.chunk_size:
                chunk_rows = 0

            findings.extend(_field_findings(manifest, spec, payload, line_number, rows_sampled, chunks_sampled, required_fields))
            findings.extend(_role_findings(manifest, spec, payload, line_number, rows_sampled, chunks_sampled))
            findings.extend(_hash_findings(manifest, spec, payload, line_number, rows_sampled, chunks_sampled))
            findings.extend(_forbidden_field_findings(manifest, spec, payload, line_number, rows_sampled, chunks_sampled))

    if rows_sampled == 0:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SAMPLE_EMPTY,
                f"streaming dataset sample '{spec.name}' produced no rows",
                "error",
                "sample_rows",
                rows_sampled,
                chunks_sampled,
                (("sample dataset stream", spec.path, "0 row(s)"),),
            )
        )
    if spec.expected_sample_rows is not None and rows_sampled != spec.expected_sample_rows:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.COUNT_MISMATCH,
                f"streaming dataset sample '{spec.name}' sampled {rows_sampled} row(s), expected {spec.expected_sample_rows}",
                "error",
                "expected_sample_rows",
                rows_sampled,
                chunks_sampled,
                (
                    ("sample dataset stream", spec.path, f"{rows_sampled} row(s)"),
                    ("compare expected sample rows", None, str(spec.expected_sample_rows)),
                ),
            )
        )
    return findings, rows_sampled, chunks_sampled


def _field_findings(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    row: Mapping[str, object],
    line_number: int,
    rows_sampled: int,
    chunks_sampled: int,
    required_fields: tuple[str, ...],
) -> tuple[TrainingStreamingFinding, ...]:
    missing = tuple(field for field in required_fields if field not in row)
    if not missing:
        return ()
    return (
        _sample_finding(
            manifest,
            spec,
            TrainingStreamingFindingKind.FIELD_MISSING,
            f"streaming dataset sample '{spec.name}' row {line_number} is missing required fields: {', '.join(missing)}",
            "error",
            f"row[{line_number}]",
            rows_sampled,
            chunks_sampled,
            (
                ("select sampled row", str(line_number), f"fields={', '.join(sorted(row))}"),
                ("compare required fields", None, ", ".join(required_fields)),
            ),
        ),
    )


def _role_findings(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    row: Mapping[str, object],
    line_number: int,
    rows_sampled: int,
    chunks_sampled: int,
) -> tuple[TrainingStreamingFinding, ...]:
    roles = _row_roles(row, messages_field=spec.messages_field)
    allowed = {_normalize_role(role) for role in spec.allowed_roles}
    findings: list[TrainingStreamingFinding] = []
    for role in roles:
        if _normalize_role(role) in allowed:
            continue
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.ROLE_VIOLATION,
                f"streaming dataset sample '{spec.name}' row {line_number} uses undeclared role '{role}'",
                "error",
                f"row[{line_number}].role",
                rows_sampled,
                chunks_sampled,
                (
                    ("extract sampled row roles", str(line_number), ", ".join(roles)),
                    ("compare allowed roles", None, ", ".join(sorted(allowed))),
                ),
            )
        )
    return tuple(findings)


def _hash_findings(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    row: Mapping[str, object],
    line_number: int,
    rows_sampled: int,
    chunks_sampled: int,
) -> tuple[TrainingStreamingFinding, ...]:
    findings: list[TrainingStreamingFinding] = []
    for field in spec.hash_fields:
        value = row.get(field)
        if isinstance(value, str) and _is_sha256_ref(value):
            continue
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.HASH_MISSING,
                f"streaming dataset sample '{spec.name}' row {line_number} field '{field}' is not sha256 evidence",
                "error",
                f"row[{line_number}].{field}",
                rows_sampled,
                chunks_sampled,
                (
                    ("select hash-only field", field, _display_value(value)),
                    ("verify hash reference", None, "missing sha256:<digest>"),
                ),
            )
        )
    return tuple(findings)


def _forbidden_field_findings(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    row: Mapping[str, object],
    line_number: int,
    rows_sampled: int,
    chunks_sampled: int,
) -> tuple[TrainingStreamingFinding, ...]:
    present = tuple(field for field in spec.forbidden_fields if field in row)
    if not present:
        return ()
    return tuple(
        _sample_finding(
            manifest,
            spec,
            TrainingStreamingFindingKind.FORBIDDEN_FIELD,
            f"streaming dataset sample '{spec.name}' row {line_number} contains forbidden raw field '{field}'",
            "error",
            f"row[{line_number}].{field}",
            rows_sampled,
            chunks_sampled,
            (
                ("select sampled row", str(line_number), f"field={field}"),
                ("enforce raw-field exclusion", None, "forbidden"),
            ),
        )
        for field in present
    )


def _streaming_specs(
    manifest: TrainingManifestArtifact,
    *,
    dataset_index: Mapping[str, TrainingDatasetSpec],
) -> tuple[StreamingDatasetSpec, ...]:
    metadata = dict(manifest.metadata)
    raw_specs: object = ()
    for key in _STREAMING_METADATA_KEYS:
        if key in metadata:
            raw_specs = metadata[key]
            break
    if raw_specs in (None, (), []):
        return ()
    if isinstance(raw_specs, Mapping):
        raw_items = []
        for name, value in sorted(raw_specs.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("streaming dataset contract keys must be non-empty strings")
            if not isinstance(value, Mapping):
                raise ValueError(f"streaming dataset entry '{name}' must be an object")
            raw_items.append({"name": name, **dict(value)})
    elif isinstance(raw_specs, list):
        raw_items = raw_specs
    else:
        raise ValueError("streaming dataset contract must be an object or list")

    specs: list[StreamingDatasetSpec] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("streaming dataset entries must be objects")
        dataset_name = _optional_string(item.get("dataset"), "dataset")
        dataset = dataset_index.get(dataset_name or "") if dataset_name else None
        path = _optional_string(item.get("path"), "path") or (dataset.path if dataset is not None else None)
        if path is None:
            raise ValueError("streaming dataset entry must define path or dataset")
        content_fields = _strings(item.get("content_fields"), "content_fields")
        preference_fields = _strings(item.get("preference_fields"), "preference_fields")
        if dataset is not None:
            content_fields = content_fields or dataset.content_fields
            preference_fields = preference_fields or dataset.preference_fields
        allowed_roles = _strings(item.get("allowed_roles"), "allowed_roles") or _manifest_roles(manifest)
        specs.append(
            StreamingDatasetSpec(
                name=_optional_string(item.get("name"), "name") or dataset_name or f"stream-{index}",
                path=path,
                dataset=dataset_name,
                sample_rows=_positive_int(item.get("sample_rows", item.get("max_sample_rows", 16)), "sample_rows"),
                chunk_size=_positive_int(item.get("chunk_size", 4), "chunk_size"),
                content_fields=content_fields,
                preference_fields=preference_fields,
                hash_fields=_strings(item.get("hash_fields"), "hash_fields"),
                forbidden_fields=_strings(item.get("forbidden_fields"), "forbidden_fields"),
                allowed_roles=allowed_roles,
                messages_field=_optional_string(item.get("messages_field"), "messages_field") or "messages",
                max_row_bytes=_optional_positive_int(item.get("max_row_bytes"), "max_row_bytes"),
                expected_sample_rows=_optional_positive_int(item.get("expected_sample_rows"), "expected_sample_rows"),
            )
        )
    return tuple(specs)


def _row_roles(row: Mapping[str, object], *, messages_field: str) -> tuple[str, ...]:
    roles: list[str] = []
    direct = row.get("role")
    if isinstance(direct, str) and direct:
        roles.append(direct)
    messages = row.get(messages_field)
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role")
                if isinstance(role, str) and role:
                    roles.append(role)
    return tuple(roles)


def _manifest_roles(manifest: TrainingManifestArtifact) -> tuple[str, ...]:
    roles = set(_STANDARD_ROLES)
    roles.update(manifest.message_roles)
    roles.update(manifest.target_roles)
    roles.update(label.canonical_role for label in manifest.role_labels)
    return tuple(sorted(_normalize_role(role) for role in roles if role))


def _sample_finding(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    kind: TrainingStreamingFindingKind,
    message: str,
    severity: str,
    subject: str,
    rows_sampled: int,
    chunks_sampled: int,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingStreamingFinding:
    return _finding(
        manifest,
        kind,
        message,
        severity,
        subject=f"metadata.streaming_dataset_verification.{spec.name}.{subject}",
        sample_name=spec.name,
        rows_sampled=rows_sampled,
        chunks_sampled=chunks_sampled,
        witness=witness,
    )


def _finding(
    manifest: TrainingManifestArtifact,
    kind: TrainingStreamingFindingKind,
    message: str,
    severity: str,
    *,
    subject: str | None,
    sample_name: str | None = None,
    rows_sampled: int = 0,
    chunks_sampled: int = 0,
    witness: tuple[tuple[str, str | None, str | None], ...],
) -> TrainingStreamingFinding:
    return TrainingStreamingFinding(
        kind=kind,
        manifest_name=manifest.name,
        message=message,
        severity=severity,
        subject=subject,
        sample_name=sample_name,
        rows_sampled=rows_sampled,
        chunks_sampled=chunks_sampled,
        witness=witness,
    )


def _strings(value: object, field: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"streaming dataset field '{field}' must be a list of non-empty strings")
    return tuple(dict.fromkeys(value))


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"streaming dataset field '{field}' must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"streaming dataset field '{field}' must be a positive integer")
    return value


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _normalize_role(role: str) -> str:
    return role.strip().lower().replace("_", "-")


def _is_sha256_ref(value: str) -> bool:
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)


def _display_value(value: object) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, str):
        return value if len(value) <= 32 else f"len={len(value)}"
    return type(value).__name__
