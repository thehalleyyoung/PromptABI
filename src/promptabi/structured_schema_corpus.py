"""Offline structured-output and tool-calling schema corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    GrammarArtifact,
    SchemaArtifact,
    ToolDefinitionArtifact,
)
from .parser_compatibility import ParserCompatibilityStatus, analyze_parser_compatibility


DEFAULT_STRUCTURED_SCHEMA_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "structured_schemas"
STRUCTURED_SCHEMA_CORPUS_MANIFEST_VERSION = 1
REQUIRED_STRUCTURED_SCHEMA_SOURCES = frozenset(
    {"open-source-agent-reduction", "anonymized-production-pattern", "synthetic-stress"}
)


class StructuredSchemaCorpusError(ValueError):
    """Raised when the structured-schema corpus is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class StructuredSchemaCorpusEntry:
    """One labeled structured-output or tool-calling schema fixture."""

    entry_id: str
    entry_type: str
    source_category: str
    path: Path
    metadata: dict[str, object]
    artifact_path: Path
    artifact_sha256: str
    metadata_sha256: str
    config_sha256: str | None

    @property
    def labels(self) -> tuple[str, ...]:
        return _tuple_from_list(self.metadata["labels"])

    @property
    def expected_status(self) -> str:
        value = self.metadata["expected_parser_compatibility_status"]
        assert isinstance(value, str)
        return value

    @property
    def expected_rule_ids(self) -> tuple[str, ...]:
        return _tuple_from_list(self.metadata["expected_rule_ids"])

    @property
    def promptabi_config_path(self) -> Path:
        return self.path / "promptabi.json"

    def artifact(self) -> Artifact:
        """Expose the corpus fixture as a typed PromptABI artifact."""

        provenance = ArtifactProvenance(
            version=str(self.metadata["fixture_revision"]),
            sha256=self.artifact_sha256,
            license=str(self.metadata["license"]),
            source=str(self.metadata["source"]),
        )
        metadata = (
            ("corpus_entry", self.entry_id),
            ("corpus_labels", self.labels),
            ("parser_format", str(self.metadata["parser_format"])),
            ("parser_compatibility", self.metadata["parser_compatibility"]),
            ("source_category", self.source_category),
        )
        if self.entry_type == "schema":
            return SchemaArtifact(
                kind=ArtifactKind.SCHEMA,
                name=f"{self.entry_id}-schema",
                location=ArtifactLocation(path=str(self.artifact_path)),
                provenance=provenance,
                metadata=metadata,
                dialect=str(self.metadata.get("dialect", "json-schema")),
            )
        if self.entry_type == "grammar":
            return GrammarArtifact(
                kind=ArtifactKind.GRAMMAR,
                name=f"{self.entry_id}-grammar",
                location=ArtifactLocation(path=str(self.artifact_path)),
                provenance=provenance,
                metadata=metadata,
                grammar_type=str(self.metadata["grammar_type"]),
            )
        if self.entry_type == "tool-definition":
            return ToolDefinitionArtifact(
                kind=ArtifactKind.TOOL_DEFINITION,
                name=f"{self.entry_id}-tool",
                location=ArtifactLocation(path=str(self.artifact_path)),
                provenance=provenance,
                metadata=metadata,
                provider=str(self.metadata["provider"]),
                tool_names=_tuple_from_list(self.metadata["tool_names"]),
            )
        raise StructuredSchemaCorpusError(f"{self.entry_id} has unsupported entry_type {self.entry_type!r}")

    def to_manifest_entry(self) -> dict[str, object]:
        return {
            "id": self.entry_id,
            "entry_type": self.entry_type,
            "source_category": self.source_category,
            "display_name": self.metadata["display_name"],
            "source": self.metadata["source"],
            "license": self.metadata["license"],
            "fixture_revision": self.metadata["fixture_revision"],
            "upstream_reference": self.metadata["upstream_reference"],
            "upstream_revision": self.metadata["upstream_revision"],
            "download_required": self.metadata["download_required"],
            "anonymized": self.metadata["anonymized"],
            "labels": list(self.labels),
            "expected_parser_compatibility_status": self.expected_status,
            "expected_rule_ids": list(self.expected_rule_ids),
            "metadata_sha256": self.metadata_sha256,
            "artifact_sha256": self.artifact_sha256,
            "config_sha256": self.config_sha256,
            "fixture_sha256": _stable_json_hash(
                {
                    "artifact_sha256": self.artifact_sha256,
                    "config_sha256": self.config_sha256,
                    "metadata_sha256": self.metadata_sha256,
                }
            ),
            "files": {
                "metadata": "metadata.json",
                "artifact": self.artifact_path.name,
                "promptabi_config": "promptabi.json" if self.promptabi_config_path.is_file() else None,
            },
        }


@dataclass(frozen=True, slots=True)
class StructuredSchemaCorpus:
    """Deterministic collection of labeled structured schema/tool fixtures."""

    root: Path
    entries: tuple[StructuredSchemaCorpusEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda entry: entry.entry_id)))
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise StructuredSchemaCorpusError("structured schema corpus contains duplicate entry ids")

    @property
    def source_categories(self) -> tuple[str, ...]:
        return tuple(sorted({entry.source_category for entry in self.entries}))

    @property
    def entry_types(self) -> tuple[str, ...]:
        return tuple(sorted({entry.entry_type for entry in self.entries}))

    def by_id(self, entry_id: str) -> StructuredSchemaCorpusEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(entry_id)

    def artifact_bundle(self) -> ArtifactBundle:
        return ArtifactBundle(tuple(entry.artifact() for entry in self.entries))

    def manifest(self) -> dict[str, object]:
        entries = [entry.to_manifest_entry() for entry in self.entries]
        manifest: dict[str, object] = {
            "manifest_version": STRUCTURED_SCHEMA_CORPUS_MANIFEST_VERSION,
            "root": str(self.root),
            "entry_count": len(entries),
            "entry_types": list(self.entry_types),
            "source_categories": list(self.source_categories),
            "required_source_categories": sorted(REQUIRED_STRUCTURED_SCHEMA_SOURCES),
            "entries": entries,
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_structured_schema_corpus(root: str | Path | None = None) -> StructuredSchemaCorpus:
    """Load and validate the offline structured-output/tool schema corpus."""

    corpus_root = Path(root) if root is not None else DEFAULT_STRUCTURED_SCHEMA_CORPUS_ROOT
    if not corpus_root.is_dir():
        raise StructuredSchemaCorpusError(f"structured schema corpus root does not exist: {corpus_root}")
    entries = tuple(_load_entry(path) for path in sorted(corpus_root.iterdir()) if path.is_dir())
    corpus = StructuredSchemaCorpus(root=corpus_root, entries=entries)
    missing_sources = REQUIRED_STRUCTURED_SCHEMA_SOURCES.difference(corpus.source_categories)
    if missing_sources:
        raise StructuredSchemaCorpusError(
            "structured schema corpus is missing required source categories: "
            + ", ".join(sorted(missing_sources))
        )
    if not {"schema", "grammar", "tool-definition"}.issubset(corpus.entry_types):
        raise StructuredSchemaCorpusError(
            "structured schema corpus must include schema, grammar, and tool-definition entries"
        )
    return corpus


def build_structured_schema_corpus_manifest(root: str | Path | None = None) -> dict[str, object]:
    """Validate the corpus and return its deterministic manifest."""

    return load_structured_schema_corpus(root).manifest()


def write_structured_schema_corpus_manifest(
    output: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Write the deterministic structured-schema corpus manifest."""

    manifest = build_structured_schema_corpus_manifest(root)
    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_structured_schema_entry(entry: StructuredSchemaCorpusEntry) -> ParserCompatibilityStatus | None:
    """Replay the entry's labeled parser-compatibility evidence when applicable."""

    artifact = entry.artifact()
    if not isinstance(artifact, (SchemaArtifact, GrammarArtifact)):
        return None
    report = analyze_parser_compatibility(artifact)
    expected = ParserCompatibilityStatus(entry.expected_status)
    if report.status is not expected:
        raise StructuredSchemaCorpusError(
            f"{entry.entry_id} expected parser compatibility {expected.value}, got {report.status.value}: "
            f"{report.reason or 'no reason'}"
        )
    return report.status


def _load_entry(path: Path) -> StructuredSchemaCorpusEntry:
    metadata_path = path / "metadata.json"
    metadata = _read_json_object(metadata_path)
    _validate_metadata(path.name, metadata)
    artifact_name = str(metadata["artifact"])
    artifact_path = path / artifact_name
    if not artifact_path.is_file():
        raise StructuredSchemaCorpusError(f"{path.name} artifact file is missing: {artifact_name}")
    config_path = path / "promptabi.json"
    entry = StructuredSchemaCorpusEntry(
        entry_id=str(metadata["id"]),
        entry_type=str(metadata["entry_type"]),
        source_category=str(metadata["source_category"]),
        path=path,
        metadata=metadata,
        artifact_path=artifact_path,
        artifact_sha256=_sha256(artifact_path),
        metadata_sha256=_sha256(metadata_path),
        config_sha256=_sha256(config_path) if config_path.is_file() else None,
    )
    validate_structured_schema_entry(entry)
    return entry


def _validate_metadata(dirname: str, metadata: dict[str, object]) -> None:
    required_strings = (
        "id",
        "entry_type",
        "source_category",
        "display_name",
        "source",
        "license",
        "fixture_revision",
        "upstream_reference",
        "upstream_revision",
        "reproducibility_notes",
        "artifact",
        "parser_format",
        "expected_parser_compatibility_status",
    )
    for key in required_strings:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise StructuredSchemaCorpusError(f"{dirname}/metadata.json field '{key}' must be a non-empty string")
    if metadata["id"] != dirname:
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json id must match its directory name")
    if metadata["entry_type"] not in {"schema", "grammar", "tool-definition"}:
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json has unsupported entry_type")
    if metadata["source_category"] not in REQUIRED_STRUCTURED_SCHEMA_SOURCES:
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json has unsupported source_category")
    if metadata.get("download_required") is not False:
        raise StructuredSchemaCorpusError(
            f"{dirname}/metadata.json field 'download_required' must be false for CPU-only fixtures"
        )
    if not isinstance(metadata.get("anonymized"), bool):
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json field 'anonymized' must be boolean")
    for key in ("labels", "expected_rule_ids"):
        _require_non_empty_strings(dirname, metadata.get(key), key)
    status = metadata["expected_parser_compatibility_status"]
    if status not in {item.value for item in ParserCompatibilityStatus}:
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json has unsupported parser compatibility status")
    if not isinstance(metadata.get("parser_compatibility"), dict):
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json field 'parser_compatibility' must be an object")
    if metadata["entry_type"] == "grammar":
        value = metadata.get("grammar_type")
        if not isinstance(value, str) or not value:
            raise StructuredSchemaCorpusError(f"{dirname}/metadata.json grammar entries require grammar_type")
    if metadata["entry_type"] == "tool-definition":
        provider = metadata.get("provider")
        if not isinstance(provider, str) or not provider:
            raise StructuredSchemaCorpusError(f"{dirname}/metadata.json tool-definition entries require provider")
        _require_non_empty_strings(dirname, metadata.get("tool_names"), "tool_names")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StructuredSchemaCorpusError(f"structured schema corpus file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StructuredSchemaCorpusError(
            f"structured schema corpus file is not valid JSON: {path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(raw, dict):
        raise StructuredSchemaCorpusError(f"structured schema corpus file must contain a JSON object: {path}")
    return raw


def _require_non_empty_strings(dirname: str, value: object, key: str) -> None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise StructuredSchemaCorpusError(f"{dirname}/metadata.json field '{key}' must be a non-empty string list")


def _tuple_from_list(value: object) -> tuple[str, ...]:
    assert isinstance(value, list)
    return tuple(str(item) for item in value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
