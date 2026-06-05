"""Streaming-dataset verification for large or private training manifests."""

from __future__ import annotations

import hashlib
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
    SHARD_PROOF_MISSING = "shard-proof-missing"
    SHARD_PROOF_INVALID = "shard-proof-invalid"
    SHARD_PROOF_HASH_MISMATCH = "shard-proof-hash-mismatch"
    SHARD_PROOF_SUMMARY_MISMATCH = "shard-proof-summary-mismatch"
    SHARD_PROOF_UNSAFE_FINGERPRINT = "shard-proof-unsafe-fingerprint"
    SHARD_PROOF_VERIFIED = "shard-proof-verified"
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
    proof_sidecars: tuple[str, ...] = ()


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
            finding.kind in _SUCCESS_KINDS for finding in self.findings
        )


_STREAMING_METADATA_KEYS = (
    "streaming_dataset_verification",
    "streaming_datasets",
    "streaming_dataset_samples",
)
_STANDARD_ROLES = ("assistant", "developer", "function", "system", "tool", "user")
_SHARD_PROOF_SCHEMA_VERSION = "promptabi.dataset-shard-proof.v1"
_SHARD_PROOF_KIND = "promptabi-dataset-shard-proof"
_SUCCESS_KINDS = frozenset(
    {
        TrainingStreamingFindingKind.VERIFIED,
        TrainingStreamingFindingKind.SHARD_PROOF_VERIFIED,
    }
)


def analyze_training_streaming(
    manifest: TrainingManifestArtifact,
    *,
    base_dir: str | Path | None = None,
) -> TrainingStreamingReport:
    """Sample streaming datasets without materializing the corpus.

    The checker reads only the declared JSONL prefix sample and validates chunk
    shape, required structural fields, role labels, hash-only evidence, and row
    byte bounds. It never accumulates the full corpus in memory. When a contract
    opts into proof sidecars, PromptABI additionally streams the declared shard
    once to verify its proof-carrying artifact hash without materializing it.
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
            proof_findings = _proof_sidecar_findings(
                manifest,
                spec,
                base_dir=root,
                rows_sampled=rows_sampled,
                chunks_sampled=chunks_sampled,
            )
            findings.extend(proof_findings)
            if any(finding.kind not in _SUCCESS_KINDS for finding in proof_findings):
                continue
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
                proof_sidecars=_proof_sidecars(item),
            )
        )
    return tuple(specs)


def build_dataset_shard_proof(
    *,
    shard_path: str | Path,
    sample_name: str,
    rows_sampled: int,
    chunks_sampled: int,
    hash_fields: tuple[str, ...] | list[str] = (),
    allowed_roles: tuple[str, ...] | list[str] = (),
    chunk_size: int | None = None,
    sample_rows: int | None = None,
    counterexample_fingerprints: tuple[str, ...] | list[str] = (),
    base_dir: str | Path | None = None,
) -> dict[str, object]:
    """Build a proof-carrying shard sidecar payload for local verification.

    The returned mapping is intentionally non-sensitive: it contains the shard
    path, a streamed SHA-256 digest, bounded verification counters, contract
    names, and optional sha256-shaped counterexample fingerprints.
    """

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    shard = Path(shard_path)
    digest = _file_sha256_ref(_resolve_under_root(shard, root))
    summary: dict[str, object] = {
        "sample_name": sample_name,
        "rows_sampled": rows_sampled,
        "chunks_sampled": chunks_sampled,
        "hash_fields": sorted(dict.fromkeys(hash_fields)),
        "allowed_roles": sorted(_normalize_role(role) for role in dict.fromkeys(allowed_roles)),
    }
    if chunk_size is not None:
        summary["chunk_size"] = chunk_size
    if sample_rows is not None:
        summary["sample_rows"] = sample_rows
    return {
        "schema_version": _SHARD_PROOF_SCHEMA_VERSION,
        "kind": _SHARD_PROOF_KIND,
        "shard": str(shard),
        "sample_name": sample_name,
        "artifact_hashes": {"dataset": digest},
        "verification_summary": summary,
        "counterexample_fingerprints": list(counterexample_fingerprints),
    }


def _proof_sidecars(item: Mapping[str, object]) -> tuple[str, ...]:
    direct = _optional_string(item.get("proof_sidecar"), "proof_sidecar")
    many = _strings(item.get("proof_sidecars"), "proof_sidecars")
    values = []
    if direct is not None:
        values.append(direct)
    values.extend(many)
    return tuple(dict.fromkeys(values))


def _proof_sidecar_findings(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    *,
    base_dir: Path,
    rows_sampled: int,
    chunks_sampled: int,
) -> tuple[TrainingStreamingFinding, ...]:
    if not spec.proof_sidecars:
        return ()
    findings: list[TrainingStreamingFinding] = []
    shard_path = _resolve_under_root(Path(spec.path), base_dir)
    expected_hash: str | None = None
    for sidecar in spec.proof_sidecars:
        sidecar_path = _resolve_under_root(Path(sidecar), base_dir)
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except OSError as exc:
            findings.append(
                _sample_finding(
                    manifest,
                    spec,
                    TrainingStreamingFindingKind.SHARD_PROOF_MISSING,
                    f"streaming dataset sample '{spec.name}' proof sidecar could not be opened: {exc}",
                    "error",
                    f"proof_sidecars.{sidecar}",
                    rows_sampled,
                    chunks_sampled,
                    (("open proof sidecar", str(sidecar_path), exc.__class__.__name__),),
                )
            )
            continue
        except json.JSONDecodeError as exc:
            findings.append(
                _sample_finding(
                    manifest,
                    spec,
                    TrainingStreamingFindingKind.SHARD_PROOF_INVALID,
                    f"streaming dataset sample '{spec.name}' proof sidecar is not valid JSON: {exc.msg}",
                    "error",
                    f"proof_sidecars.{sidecar}",
                    rows_sampled,
                    chunks_sampled,
                    (("parse proof sidecar", str(sidecar_path), exc.msg),),
                )
            )
            continue
        if not isinstance(payload, Mapping):
            findings.append(
                _sample_finding(
                    manifest,
                    spec,
                    TrainingStreamingFindingKind.SHARD_PROOF_INVALID,
                    f"streaming dataset sample '{spec.name}' proof sidecar is not a JSON object",
                    "error",
                    f"proof_sidecars.{sidecar}",
                    rows_sampled,
                    chunks_sampled,
                    (("parse proof sidecar", str(sidecar_path), type(payload).__name__),),
                )
            )
            continue
        if expected_hash is None:
            try:
                expected_hash = _file_sha256_ref(shard_path)
            except OSError as exc:
                findings.append(
                    _sample_finding(
                        manifest,
                        spec,
                        TrainingStreamingFindingKind.STREAM_UNREADABLE,
                        f"streaming dataset sample '{spec.name}' could not be hashed for proof sidecar verification: {exc}",
                        "error",
                        "path",
                        rows_sampled,
                        chunks_sampled,
                        (("stream shard digest", str(shard_path), exc.__class__.__name__),),
                    )
                )
                continue
        findings.extend(
            _validate_proof_payload(
                manifest,
                spec,
                payload,
                sidecar_path=sidecar_path,
                shard_path=shard_path,
                expected_hash=expected_hash,
                rows_sampled=rows_sampled,
                chunks_sampled=chunks_sampled,
            )
        )
    return tuple(findings)


def _validate_proof_payload(
    manifest: TrainingManifestArtifact,
    spec: StreamingDatasetSpec,
    payload: Mapping[str, object],
    *,
    sidecar_path: Path,
    shard_path: Path,
    expected_hash: str,
    rows_sampled: int,
    chunks_sampled: int,
) -> tuple[TrainingStreamingFinding, ...]:
    findings: list[TrainingStreamingFinding] = []
    schema_version = payload.get("schema_version")
    kind = payload.get("kind")
    if schema_version != _SHARD_PROOF_SCHEMA_VERSION or kind != _SHARD_PROOF_KIND:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_INVALID,
                f"streaming dataset sample '{spec.name}' proof sidecar has unsupported schema metadata",
                "error",
                "proof_sidecars.schema_version",
                rows_sampled,
                chunks_sampled,
                (("check proof schema", str(sidecar_path), f"{schema_version!r}/{kind!r}"),),
            )
        )
    sidecar_shard = payload.get("shard")
    if not isinstance(sidecar_shard, str) or _resolve_under_root(Path(sidecar_shard), shard_path.parent) != shard_path:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_INVALID,
                f"streaming dataset sample '{spec.name}' proof sidecar points at a different shard",
                "error",
                "proof_sidecars.shard",
                rows_sampled,
                chunks_sampled,
                (("compare proof shard", _display_value(sidecar_shard), str(shard_path)),),
            )
        )
    artifact_hashes = payload.get("artifact_hashes")
    dataset_hash = artifact_hashes.get("dataset") if isinstance(artifact_hashes, Mapping) else None
    if dataset_hash != expected_hash:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_HASH_MISMATCH,
                f"streaming dataset sample '{spec.name}' proof sidecar dataset hash does not match the local shard",
                "error",
                "proof_sidecars.artifact_hashes.dataset",
                rows_sampled,
                chunks_sampled,
                (("stream shard digest", str(shard_path), expected_hash), ("compare proof digest", None, _display_value(dataset_hash))),
            )
        )
    summary = payload.get("verification_summary")
    if not isinstance(summary, Mapping) or _proof_summary_mismatches(spec, summary, rows_sampled, chunks_sampled):
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_SUMMARY_MISMATCH,
                f"streaming dataset sample '{spec.name}' proof sidecar summary does not match the current bounded verification",
                "error",
                "proof_sidecars.verification_summary",
                rows_sampled,
                chunks_sampled,
                (
                    ("compare proof rows", None, str(rows_sampled)),
                    ("compare proof chunks", None, str(chunks_sampled)),
                    ("compare proof contract", None, spec.name),
                ),
            )
        )
    fingerprints = payload.get("counterexample_fingerprints", ())
    if not isinstance(fingerprints, list) or any(not isinstance(item, str) or not _is_sha256_ref(item) for item in fingerprints):
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_UNSAFE_FINGERPRINT,
                f"streaming dataset sample '{spec.name}' proof sidecar counterexample fingerprints are not hash-only",
                "error",
                "proof_sidecars.counterexample_fingerprints",
                rows_sampled,
                chunks_sampled,
                (("inspect proof fingerprints", None, _display_value(fingerprints)),),
            )
        )
    if not findings:
        findings.append(
            _sample_finding(
                manifest,
                spec,
                TrainingStreamingFindingKind.SHARD_PROOF_VERIFIED,
                f"streaming dataset sample '{spec.name}' proof sidecar matches the local shard and bounded verification summary",
                "info",
                "proof_sidecars",
                rows_sampled,
                chunks_sampled,
                (
                    ("open proof sidecar", str(sidecar_path), "json"),
                    ("stream shard digest", str(shard_path), expected_hash),
                    ("compare bounded summary", None, f"{rows_sampled} row(s), {chunks_sampled} chunk(s)"),
                    ("verify counterexample fingerprints", None, "hash-only"),
                ),
            )
        )
    return tuple(findings)


def _proof_summary_mismatches(
    spec: StreamingDatasetSpec,
    summary: Mapping[str, object],
    rows_sampled: int,
    chunks_sampled: int,
) -> bool:
    expected: dict[str, object] = {
        "sample_name": spec.name,
        "rows_sampled": rows_sampled,
        "chunks_sampled": chunks_sampled,
        "hash_fields": sorted(spec.hash_fields),
        "allowed_roles": sorted(_normalize_role(role) for role in spec.allowed_roles),
        "chunk_size": spec.chunk_size,
        "sample_rows": spec.sample_rows,
    }
    return any(summary.get(key) != value for key, value in expected.items())


def _resolve_under_root(path: Path, root: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    return resolved.resolve(strict=False)


def _file_sha256_ref(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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
