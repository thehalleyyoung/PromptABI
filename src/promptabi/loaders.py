"""Offline artifact loading and pin validation for PromptABI."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import tarfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .artifacts import Artifact, ArtifactKind, GrammarArtifact, SchemaArtifact, StopPolicyArtifact
from .chat_templates import ChatTemplateParseError, parse_hf_tokenizer_config_chat_template, symbolically_execute_chat_template
from .diagnostics import SourceSpan
from .grammars import GrammarIngestionError, ingest_grammar_file, ingest_json_schema_mapping
from .json_schema import normalize_json_schema_mapping
from .role_boundaries import build_role_boundary_model
from .source import build_json_source_map
from .stop_policies import StopPolicyParseError, parse_stop_policy_config


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_IMMUTABLE_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A deterministic, lightweight summary of a verification artifact."""

    artifact: Artifact
    source_type: str
    pinned: bool
    resolved: bool
    actual_sha256: str | None = None
    size_bytes: int | None = None
    manifest_sha256: str | None = None
    members: tuple[str, ...] = ()
    metadata: tuple[tuple[str, object], ...] = ()
    source_spans: tuple[tuple[str, SourceSpan], ...] = ()
    warnings: tuple["ArtifactLoadWarning", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(sorted(dict.fromkeys(self.members))))
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata, key=lambda item: item[0])))
        object.__setattr__(self, "source_spans", tuple(sorted(self.source_spans, key=lambda item: item[0])))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "artifact": self.artifact.to_ref().to_dict(),
            "source_type": self.source_type,
            "pinned": self.pinned,
            "resolved": self.resolved,
        }
        if self.actual_sha256 is not None:
            data["actual_sha256"] = self.actual_sha256
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.manifest_sha256 is not None:
            data["manifest_sha256"] = self.manifest_sha256
        if self.members:
            data["members"] = list(self.members)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.source_spans:
            data["source_spans"] = {
                name: span.to_dict() for name, span in self.source_spans
            }
        if self.warnings:
            data["warnings"] = [warning.to_dict() for warning in self.warnings]
        return data


@dataclass(frozen=True, slots=True)
class ArtifactLoadWarning:
    """A non-fatal loader issue that should become a warning diagnostic."""

    rule_id: str
    message: str
    suggestion: str
    steps: tuple[tuple[str, str | None, str | None], ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rule_id": self.rule_id,
            "message": self.message,
            "suggestion": self.suggestion,
        }
        if self.steps:
            data["steps"] = [
                {"action": action, "input": input_value, "output": output_value}
                for action, input_value, output_value in self.steps
            ]
        return data


@dataclass(frozen=True, slots=True)
class ArtifactLoadError(ValueError):
    """A fatal artifact loading problem with diagnostic-ready context."""

    rule_id: str
    message: str
    suggestion: str
    steps: tuple[tuple[str, str | None, str | None], ...] = field(default_factory=tuple)
    span: SourceSpan | None = None

    def __str__(self) -> str:
        return self.message


class ArtifactLoader:
    """Load PromptABI artifacts without network access or heavyweight libraries."""

    def load(self, artifact: Artifact) -> LoadedArtifact:
        if artifact.location.uri is not None:
            return self._load_uri(artifact)

        if artifact.location.path is None:
            raise AssertionError("ArtifactLocation invariant violated")

        path = Path(artifact.location.path)
        if path.is_dir():
            return self._load_directory(artifact, path)
        if not path.is_file():
            raise ArtifactLoadError(
                rule_id="artifact-missing",
                message=f"artifact '{artifact.name}' does not exist",
                suggestion="Check the path relative to the PromptABI config file.",
                steps=(
                    ("resolve artifact path", None, str(path)),
                    ("check local filesystem", None, "missing"),
                ),
            )
        if path.suffix.lower() == ".gguf":
            return self._load_gguf_stub(artifact, path)
        if _is_archive(path):
            return self._load_archive(artifact, path)
        if artifact.kind is ArtifactKind.CHAT_TEMPLATE:
            return self._load_chat_template(artifact, path)
        if artifact.kind is ArtifactKind.STOP_POLICY:
            return self._load_stop_policy(artifact, path)
        if artifact.kind is ArtifactKind.SCHEMA:
            return self._load_schema(artifact, path)
        if artifact.kind is ArtifactKind.GRAMMAR:
            return self._load_grammar(artifact, path)
        if artifact.kind is ArtifactKind.PROVIDER_CONFIG:
            return self._load_provider_snapshot(artifact, path)
        return self._load_file(artifact, path, source_type="local-file")

    def _load_uri(self, artifact: Artifact) -> LoadedArtifact:
        assert artifact.location.uri is not None
        parsed = urlparse(artifact.location.uri)
        if parsed.scheme == "hf":
            return self._load_huggingface_ref(artifact, parsed)
        if parsed.scheme == "memory":
            return LoadedArtifact(
                artifact=artifact,
                source_type="memory",
                pinned=True,
                resolved=True,
                metadata=(("uri_scheme", parsed.scheme),),
            )
        warnings = self._pin_warnings(artifact, source_type="remote-uri", resolved=False)
        return LoadedArtifact(
            artifact=artifact,
            source_type="remote-uri",
            pinned=_has_pin(artifact),
            resolved=False,
            metadata=(("uri_scheme", parsed.scheme),),
            warnings=warnings,
        )

    def _load_huggingface_ref(self, artifact: Artifact, parsed) -> LoadedArtifact:
        repo_and_path = (parsed.netloc + parsed.path).strip("/")
        if "/" not in repo_and_path:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"Hugging Face artifact '{artifact.name}' must include owner and repo",
                suggestion="Use a URI such as hf://org/model?revision=<commit-sha>.",
                steps=(("parse Hugging Face URI", artifact.location.uri, repo_and_path or None),),
            )

        query = parse_qs(parsed.query)
        query_revision = query.get("revision", [None])[0]
        revision = query_revision or artifact.provenance.revision or artifact.provenance.version
        repo_parts = repo_and_path.split("/", 2)
        repo_id = "/".join(repo_parts[:2])
        artifact_path = repo_parts[2] if len(repo_parts) == 3 else None
        warnings: list[ArtifactLoadWarning] = []
        if revision is None:
            warnings.append(
                ArtifactLoadWarning(
                    rule_id="artifact-unpinned",
                    message=f"Hugging Face artifact '{artifact.name}' is not revision-pinned",
                    suggestion="Pin hf:// artifacts to an immutable commit SHA with ?revision=<40-hex-sha>.",
                    steps=(("parse Hugging Face revision", artifact.location.uri, "missing"),),
                )
            )
        elif not _IMMUTABLE_HF_REVISION_RE.fullmatch(revision.lower()):
            warnings.append(
                ArtifactLoadWarning(
                    rule_id="artifact-weak-pin",
                    message=f"Hugging Face artifact '{artifact.name}' uses a movable or non-commit revision",
                    suggestion="Use a full 40-character commit SHA for reproducible offline verification.",
                    steps=(("parse Hugging Face revision", artifact.location.uri, revision),),
                )
            )

        metadata: list[tuple[str, object]] = [("repo_id", repo_id)]
        if artifact_path is not None:
            metadata.append(("artifact_path", artifact_path))
        if revision is not None:
            metadata.append(("revision", revision))
        return LoadedArtifact(
            artifact=artifact,
            source_type="huggingface-model-repo",
            pinned=revision is not None or _has_pin(artifact),
            resolved=False,
            metadata=tuple(metadata),
            warnings=tuple(warnings),
        )

    def _load_directory(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        if artifact.kind is not ArtifactKind.TOKENIZER:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"artifact '{artifact.name}' points to a directory but is not a tokenizer",
                suggestion="Use tokenizer directories only for tokenizer artifacts, or point to a concrete file.",
                steps=(("inspect local artifact path", str(path), "directory"),),
            )
        files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
        members = tuple(item.relative_to(path).as_posix() for item in files)
        if not {"tokenizer.json", "tokenizer_config.json"}.intersection(members):
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"tokenizer directory artifact '{artifact.name}' lacks tokenizer metadata",
                suggestion="Include tokenizer.json or tokenizer_config.json in the tokenizer directory.",
                steps=(("scan tokenizer directory", str(path), f"{len(members)} files"),),
            )
        actual_sha256, size_bytes = _hash_directory_manifest(path, files)
        warnings = self._validate_pin(artifact, actual_sha256, source_type="tokenizer-directory")
        source_spans = _directory_source_spans(artifact, path, members)
        return LoadedArtifact(
            artifact=artifact,
            source_type="tokenizer-directory",
            pinned=_has_pin(artifact),
            resolved=True,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            manifest_sha256=actual_sha256,
            members=members,
            source_spans=source_spans,
            warnings=warnings,
        )

    def _load_file(self, artifact: Artifact, path: Path, *, source_type: str) -> LoadedArtifact:
        actual_sha256, size_bytes = _hash_file(path)
        warnings = self._validate_pin(artifact, actual_sha256, source_type=source_type)
        source_spans = _file_source_spans(artifact, path)
        return LoadedArtifact(
            artifact=artifact,
            source_type=source_type,
            pinned=_has_pin(artifact),
            resolved=True,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            source_spans=source_spans,
            warnings=warnings,
        )

    def _load_provider_snapshot(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"provider snapshot artifact '{artifact.name}' is not valid JSON",
                suggestion="Store provider snapshots as deterministic JSON objects.",
                steps=(("parse provider snapshot", str(path), exc.msg),),
                span=SourceSpan(path=str(path), start_line=exc.lineno, start_column=exc.colno),
            ) from exc
        if not isinstance(raw, dict):
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"provider snapshot artifact '{artifact.name}' must be a JSON object",
                suggestion="Store provider, captured_at, and request/response shape metadata in an object.",
                steps=(("parse provider snapshot", str(path), type(raw).__name__),),
            )
        provider = raw.get("provider")
        if not isinstance(provider, str) or not provider:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"provider snapshot artifact '{artifact.name}' lacks a provider",
                suggestion="Add a non-empty 'provider' field to the provider snapshot.",
                steps=(("validate provider snapshot", str(path), "provider missing"),),
            )
        if not any(key in raw for key in ("request_shape", "response_shape", "streaming_deltas")):
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"provider snapshot artifact '{artifact.name}' lacks captured API shape metadata",
                suggestion="Record request_shape, response_shape, or streaming_deltas in the snapshot.",
                steps=(("validate provider snapshot", str(path), "shape metadata missing"),),
            )
        loaded = self._load_file(artifact, path, source_type="provider-config-snapshot")
        return LoadedArtifact(
            artifact=loaded.artifact,
            source_type=loaded.source_type,
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            actual_sha256=loaded.actual_sha256,
            size_bytes=loaded.size_bytes,
            metadata=(("provider", provider),),
            source_spans=loaded.source_spans,
            warnings=loaded.warnings,
        )

    def _load_chat_template(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        loaded = self._load_file(artifact, path, source_type="chat-template-file")
        if path.suffix.lower() != ".json":
            return loaded
        try:
            parsed = parse_hf_tokenizer_config_chat_template(path)
        except ChatTemplateParseError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"chat-template artifact '{artifact.name}' could not be parsed",
                suggestion="Point Hugging Face chat-template artifacts at tokenizer_config.json with a string chat_template.",
                steps=(("parse Hugging Face chat template", str(path), str(exc)),),
            ) from exc
        symbolic = symbolically_execute_chat_template(parsed)
        role_boundaries = build_role_boundary_model(parsed)
        metadata = (
            ("filters", parsed.filters),
            ("generation_prompt_excerpts", parsed.generation_prompt_excerpts),
            ("message_fields", tuple(field.field for field in parsed.message_fields)),
            ("role_boundary_path_count", len(role_boundaries.paths)),
            ("role_boundary_region_count", sum(len(path.regions) for path in role_boundaries.paths)),
            ("role_boundary_roles", role_boundaries.roles),
            ("role_boundary_supported", role_boundaries.supported),
            ("role_assumptions", parsed.role_assumptions),
            ("special_tokens", tuple(token.text for token in parsed.special_tokens)),
            ("supported_fragment", parsed.supported),
            ("template_format", parsed.template_format),
            ("template_length", len(parsed.template_source)),
            ("tool_fields", tuple(field.field for field in parsed.tool_fields)),
            ("symbolic_abstentions", tuple(item.expression for item in symbolic.abstentions)),
            ("symbolic_path_count", len(symbolic.paths)),
            ("symbolic_supported_fragment", symbolic.supported),
            ("unsupported_constructs", tuple(item.expression for item in parsed.unsupported_constructs)),
            ("uses_generation_prompt", parsed.uses_generation_prompt),
            ("uses_tools", parsed.uses_tools),
            ("uses_whitespace_control", parsed.uses_whitespace_control),
        )
        return LoadedArtifact(
            artifact=loaded.artifact,
            source_type="huggingface-tokenizer-config-chat-template",
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            actual_sha256=loaded.actual_sha256,
            size_bytes=loaded.size_bytes,
            metadata=metadata,
            source_spans=loaded.source_spans,
            warnings=loaded.warnings,
        )

    def _load_stop_policy(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        if path.suffix.lower() != ".json":
            return self._load_file(artifact, path, source_type="stop-policy-file")
        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"stop-policy artifact '{artifact.name}' is not valid JSON",
                suggestion="Store stop policies as deterministic JSON objects.",
                steps=(("parse stop-policy JSON", str(path), exc.msg),),
                span=SourceSpan(path=str(path), start_line=exc.lineno, start_column=exc.colno),
            ) from exc
        if not isinstance(raw, dict):
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"stop-policy artifact '{artifact.name}' must be a JSON object",
                suggestion="Use a provider request snapshot, generation config, or wrapper config object.",
                steps=(("parse stop-policy root", str(path), type(raw).__name__),),
            )
        try:
            source_map = build_json_source_map(text, path)
            parsed = parse_stop_policy_config(
                raw,
                source_map=source_map,
                declared_family=artifact.source_family if isinstance(artifact, StopPolicyArtifact) else None,
            )
        except StopPolicyParseError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"stop-policy artifact '{artifact.name}' could not be parsed",
                suggestion="Use string stop sequences and integer stop token IDs in supported provider/framework fields.",
                steps=(("parse stop-policy fields", ".".join(exc.path) or "<root>", str(exc)),),
                span=exc.span,
            ) from exc
        loaded = self._load_file(artifact, path, source_type="stop-policy-config")
        parsed_artifact = artifact
        if isinstance(artifact, StopPolicyArtifact):
            parsed_artifact = replace(
                artifact,
                stop_sequences=parsed.stop_sequences or artifact.stop_sequences,
                stop_token_ids=parsed.stop_token_ids or artifact.stop_token_ids,
                include_eos=parsed.include_eos,
                source_family=artifact.source_family or parsed.source_family,
            )
        source_metadata = tuple(
            (f"source_{index}_{key}", value)
            for index, source in enumerate(parsed.sources)
            for key, value in source.to_metadata()
        )
        return LoadedArtifact(
            artifact=parsed_artifact,
            source_type="stop-policy-config",
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            actual_sha256=loaded.actual_sha256,
            size_bytes=loaded.size_bytes,
            metadata=(*parsed.to_metadata(), *source_metadata),
            source_spans=loaded.source_spans,
            warnings=loaded.warnings,
        )

    def _load_schema(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        if path.suffix.lower() != ".json":
            return self._load_file(artifact, path, source_type="schema-file")
        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"schema artifact '{artifact.name}' is not valid JSON",
                suggestion="Store JSON Schema artifacts as deterministic JSON objects.",
                steps=(("parse JSON Schema", str(path), exc.msg),),
                span=SourceSpan(path=str(path), start_line=exc.lineno, start_column=exc.colno),
            ) from exc
        if not isinstance(raw, dict):
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"schema artifact '{artifact.name}' must be a JSON object",
                suggestion="Use a JSON Schema object with type, properties, required, enum, const, or constraints.",
                steps=(("parse JSON Schema root", str(path), type(raw).__name__),),
            )
        try:
            source_map = build_json_source_map(text, path)
            declared_type = artifact.dialect if isinstance(artifact, SchemaArtifact) else "json-schema"
            parsed = ingest_json_schema_mapping(raw, declared_type=declared_type, source_map=source_map)
            normalized = normalize_json_schema_mapping(raw, source_map=source_map)
        except GrammarIngestionError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"schema artifact '{artifact.name}' could not be ingested",
                suggestion="Use the supported JSON Schema subset or keep unsupported keywords explicit for later normalization.",
                steps=(("ingest JSON Schema", str(path), str(exc)),),
                span=exc.span,
            ) from exc
        loaded = self._load_file(artifact, path, source_type="json-schema")
        return LoadedArtifact(
            artifact=loaded.artifact,
            source_type="json-schema",
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            actual_sha256=loaded.actual_sha256,
            size_bytes=loaded.size_bytes,
            metadata=_merge_metadata(parsed.to_metadata(), normalized.to_metadata()),
            source_spans=normalized.source_spans or parsed.source_spans or loaded.source_spans,
            warnings=loaded.warnings,
        )

    def _load_grammar(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        declared_type = artifact.grammar_type if isinstance(artifact, GrammarArtifact) else "promptabi"
        try:
            parsed = ingest_grammar_file(path, declared_type=declared_type)
        except GrammarIngestionError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"grammar artifact '{artifact.name}' could not be ingested",
                suggestion="Use a supported JSON Schema, regex, EBNF, Outlines, xgrammar, llguidance, or PromptABI grammar shape.",
                steps=(("ingest grammar", str(path), str(exc)),),
                span=exc.span,
            ) from exc
        loaded = self._load_file(artifact, path, source_type=f"grammar-{parsed.dialect.value}")
        parsed_artifact = artifact
        if isinstance(artifact, GrammarArtifact):
            parsed_artifact = replace(
                artifact,
                grammar_type=parsed.dialect.value,
                start_symbol=artifact.start_symbol or parsed.start_symbol,
                rule_names=parsed.rule_names,
                supported_fragment=parsed.supported_fragment,
            )
        return LoadedArtifact(
            artifact=parsed_artifact,
            source_type=f"grammar-{parsed.dialect.value}",
            pinned=loaded.pinned,
            resolved=loaded.resolved,
            actual_sha256=loaded.actual_sha256,
            size_bytes=loaded.size_bytes,
            metadata=parsed.to_metadata(),
            source_spans=parsed.source_spans or loaded.source_spans,
            warnings=loaded.warnings,
        )

    def _load_gguf_stub(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:4] != b"GGUF":
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"GGUF artifact '{artifact.name}' does not have a valid GGUF header",
                suggestion="Point GGUF artifacts at real GGUF files or minimized GGUF metadata stubs.",
                steps=(("read GGUF header", str(path), "invalid magic or short header"),),
            )
        version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", header[4:24])
        actual_sha256, size_bytes = _hash_file(path)
        warnings = self._validate_pin(artifact, actual_sha256, source_type="gguf-metadata-stub")
        return LoadedArtifact(
            artifact=artifact,
            source_type="gguf-metadata-stub",
            pinned=_has_pin(artifact),
            resolved=True,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            metadata=(
                ("gguf_version", version),
                ("metadata_kv_count", metadata_kv_count),
                ("tensor_count", tensor_count),
            ),
            warnings=warnings,
        )

    def _load_archive(self, artifact: Artifact, path: Path) -> LoadedArtifact:
        members = _archive_members(path)
        actual_sha256, size_bytes = _hash_file(path)
        warnings = self._validate_pin(artifact, actual_sha256, source_type="fixture-bundle-archive")
        return LoadedArtifact(
            artifact=artifact,
            source_type="fixture-bundle-archive",
            pinned=_has_pin(artifact),
            resolved=True,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            members=members,
            source_spans=_archive_source_spans(path, members),
            warnings=warnings,
        )

    def _validate_pin(
        self,
        artifact: Artifact,
        actual_sha256: str,
        *,
        source_type: str,
    ) -> tuple[ArtifactLoadWarning, ...]:
        expected = artifact.provenance.sha256
        if expected is not None:
            if not _SHA256_RE.fullmatch(expected):
                raise ArtifactLoadError(
                    rule_id="artifact-pin-invalid",
                    message=f"artifact '{artifact.name}' has a malformed sha256 pin",
                    suggestion="Use a 64-character hexadecimal sha256 digest.",
                    steps=(("validate sha256 pin", expected, "malformed"),),
                )
            if expected.lower() != actual_sha256:
                raise ArtifactLoadError(
                    rule_id="artifact-hash-mismatch",
                    message=f"artifact '{artifact.name}' does not match its sha256 pin",
                    suggestion="Update the artifact pin only after reviewing the artifact change.",
                    steps=(
                        ("read artifact bytes", artifact.location.ref_path, f"sha256={actual_sha256}"),
                        ("compare sha256 pin", expected.lower(), "mismatch"),
                    ),
                )
        return self._pin_warnings(artifact, source_type=source_type, resolved=True)

    def _pin_warnings(
        self,
        artifact: Artifact,
        *,
        source_type: str,
        resolved: bool,
    ) -> tuple[ArtifactLoadWarning, ...]:
        if _has_pin(artifact):
            return ()
        return (
            ArtifactLoadWarning(
                rule_id="artifact-unpinned",
                message=f"artifact '{artifact.name}' is not version-pinned",
                suggestion="Add sha256, revision, or version provenance to make verification reproducible.",
                steps=(
                    ("inspect artifact provenance", artifact.location.ref_path, "no pin"),
                    ("classify artifact source", source_type, "resolved" if resolved else "metadata-only"),
                ),
            ),
        )


def load_artifact(artifact: Artifact) -> LoadedArtifact:
    """Load one artifact with the default offline loader."""

    return ArtifactLoader().load(artifact)


def _merge_metadata(
    primary: tuple[tuple[str, object], ...],
    secondary: tuple[tuple[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    primary_keys = {key for key, _value in primary}
    return (*primary, *((key, value) for key, value in secondary if key not in primary_keys))


def _has_pin(artifact: Artifact) -> bool:
    return any(
        value is not None
        for value in (artifact.provenance.sha256, artifact.provenance.revision, artifact.provenance.version)
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hash_directory_manifest(root: Path, files: tuple[Path, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        file_hash, file_size = _hash_file(file_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        size += file_size
    return digest.hexdigest(), size


def _is_archive(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return (
        path.suffix.lower() == ".zip"
        or path.suffix.lower() == ".tar"
        or suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"])
    )


def _archive_members(path: Path) -> tuple[str, ...]:
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                return tuple(sorted(item.filename for item in archive.infolist() if not item.is_dir()))
        with tarfile.open(path) as archive:
            return tuple(sorted(item.name for item in archive.getmembers() if item.isfile()))
    except (tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ArtifactLoadError(
            rule_id="artifact-load-failed",
            message=f"fixture bundle archive '{path}' cannot be read",
            suggestion="Use a valid zip, tar, tar.gz, tar.xz, or tar.bz2 fixture bundle.",
            steps=(("read fixture bundle archive", str(path), str(exc)),),
        ) from exc


def _directory_source_spans(
    artifact: Artifact,
    root: Path,
    members: tuple[str, ...],
) -> tuple[tuple[str, SourceSpan], ...]:
    spans: list[tuple[str, SourceSpan]] = []
    if artifact.kind is not ArtifactKind.TOKENIZER:
        return ()
    for member in members:
        if member not in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
            continue
        member_path = root / member
        for name, span in _file_source_spans(artifact, member_path):
            spans.append((f"{member}:{name}", span))
    return tuple(spans)


def _archive_source_spans(path: Path, members: tuple[str, ...]) -> tuple[tuple[str, SourceSpan], ...]:
    spans: list[tuple[str, SourceSpan]] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in members:
                if not member.endswith(".json"):
                    continue
                try:
                    text = archive.read(member).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ArtifactLoadError(
                        rule_id="artifact-load-failed",
                        message=f"fixture bundle member '{member}' is not UTF-8 JSON",
                        suggestion="Store fixture JSON members as UTF-8 text.",
                        steps=(("decode fixture bundle member", member, str(exc)),),
                    ) from exc
                spans.extend(_json_source_spans(text, f"{path}!/{member}", prefix=member))
        return tuple(spans)
    with tarfile.open(path) as archive:
        for member in members:
            if not member.endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                text = extracted.read().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactLoadError(
                    rule_id="artifact-load-failed",
                    message=f"fixture bundle member '{member}' is not UTF-8 JSON",
                    suggestion="Store fixture JSON members as UTF-8 text.",
                    steps=(("decode fixture bundle member", member, str(exc)),),
                ) from exc
            spans.extend(_json_source_spans(text, f"{path}!/{member}", prefix=member))
    return tuple(spans)


def _file_source_spans(artifact: Artifact, path: Path) -> tuple[tuple[str, SourceSpan], ...]:
    if path.suffix.lower() == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(
                rule_id="artifact-load-failed",
                message=f"{artifact.kind.value} artifact '{artifact.name}' is not valid JSON",
                suggestion="Use valid UTF-8 JSON for tokenizer configs, schemas, tools, and PromptABI metadata.",
                steps=(("parse JSON artifact", str(path), exc.msg),),
                span=SourceSpan(path=str(path), start_line=exc.lineno, start_column=exc.colno),
            ) from exc
        return _json_source_spans(text, str(path))
    if artifact.kind is ArtifactKind.CHAT_TEMPLATE:
        return (("template", _whole_file_span(path)),)
    return ()


def _json_source_spans(
    text: str,
    path: str,
    *,
    prefix: str | None = None,
) -> tuple[tuple[str, SourceSpan], ...]:
    try:
        source_map = build_json_source_map(text, path)
    except ValueError as exc:
        raise ArtifactLoadError(
            rule_id="artifact-load-failed",
            message=f"JSON artifact '{path}' could not be source-mapped",
            suggestion="Use standard JSON syntax so PromptABI can attach precise diagnostics.",
            steps=(("source-map JSON artifact", path, str(exc)),),
        ) from exc
    spans = source_map.prefixed(())
    if prefix is None:
        return spans
    return tuple((f"{prefix}:{name}", span) for name, span in spans)


def _whole_file_span(path: Path) -> SourceSpan:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines() or [""]
    return SourceSpan(
        path=str(path),
        start_line=1,
        start_column=1,
        end_line=len(lines),
        end_column=max(1, len(lines[-1])),
    )
