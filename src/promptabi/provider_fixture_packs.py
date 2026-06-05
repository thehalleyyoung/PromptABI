"""Recorded, secret-free provider fixture packs for offline API-contract checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactBundle, ArtifactKind, ArtifactLocation, ArtifactProvenance, ProviderConfigArtifact
from .provider_migration import SUPPORTED_PROVIDER_FAMILIES, canonical_provider_family


DEFAULT_PROVIDER_FIXTURE_PACK_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "provider_fixture_packs"
PROVIDER_FIXTURE_PACK_MANIFEST_VERSION = 1
REQUIRED_PROVIDER_FIXTURE_FAMILIES = frozenset(
    {
        "anthropic",
        "bedrock",
        "gemini",
        "litellm",
        "openai",
        "vllm-openai-server",
    }
)

_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "x-api-key",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class ProviderFixturePackError(ValueError):
    """Raised when a provider fixture pack is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProviderFixturePackEntry:
    """One recorded provider API-contract fixture pack."""

    entry_id: str
    provider_family: str
    path: Path
    metadata: dict[str, object]
    pack: dict[str, object]
    metadata_sha256: str
    pack_sha256: str

    @property
    def edge_case_ids(self) -> tuple[str, ...]:
        edge_cases = self.pack["edge_cases"]
        assert isinstance(edge_cases, list)
        return tuple(str(item["id"]) for item in edge_cases if isinstance(item, dict))

    @property
    def captured_surfaces(self) -> tuple[str, ...]:
        return (
            "request",
            "response",
            "tool_calls",
            "stops",
            "streaming",
            "errors",
            "limits",
        )

    def artifact(self) -> ProviderConfigArtifact:
        """Expose this pack as a PromptABI provider-config artifact."""

        return ProviderConfigArtifact(
            kind=ArtifactKind.PROVIDER_CONFIG,
            name=f"{self.entry_id}-provider-fixture",
            location=ArtifactLocation(path=str(self.path / "pack.json")),
            provenance=ArtifactProvenance(
                version=str(self.metadata["fixture_revision"]),
                sha256=self.pack_sha256,
                license=str(self.metadata["license"]),
                source=str(self.metadata["source"]),
            ),
            provider=str(self.pack["provider"]),
            api_family=self.provider_family,
            metadata=(
                ("corpus_entry", self.entry_id),
                ("fixture_pack", True),
                ("edge_cases", self.edge_case_ids),
            ),
        )

    def to_manifest_entry(self) -> dict[str, object]:
        return {
            "id": self.entry_id,
            "provider_family": self.provider_family,
            "display_name": self.metadata["display_name"],
            "source": self.metadata["source"],
            "license": self.metadata["license"],
            "fixture_revision": self.metadata["fixture_revision"],
            "upstream_reference": self.metadata["upstream_reference"],
            "upstream_revision": self.metadata["upstream_revision"],
            "download_required": self.metadata["download_required"],
            "secrets_included": self.metadata["secrets_included"],
            "anonymized": self.metadata["anonymized"],
            "reproducibility_notes": self.metadata["reproducibility_notes"],
            "captured_surfaces": list(self.captured_surfaces),
            "edge_cases": list(self.edge_case_ids),
            "metadata_sha256": self.metadata_sha256,
            "pack_sha256": self.pack_sha256,
            "fixture_sha256": _stable_json_hash(
                {
                    "metadata_sha256": self.metadata_sha256,
                    "pack_sha256": self.pack_sha256,
                }
            ),
            "files": {
                "metadata": "metadata.json",
                "pack": "pack.json",
            },
        }


@dataclass(frozen=True, slots=True)
class ProviderFixturePackCorpus:
    """Deterministic collection of recorded provider fixture packs."""

    root: Path
    entries: tuple[ProviderFixturePackEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda entry: entry.entry_id)))
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ProviderFixturePackError("provider fixture pack corpus contains duplicate entry ids")

    @property
    def provider_families(self) -> tuple[str, ...]:
        return tuple(sorted({entry.provider_family for entry in self.entries}))

    def by_id(self, entry_id: str) -> ProviderFixturePackEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(entry_id)

    def artifact_bundle(self) -> ArtifactBundle:
        return ArtifactBundle(tuple(entry.artifact() for entry in self.entries))

    def manifest(self) -> dict[str, object]:
        entries = [entry.to_manifest_entry() for entry in self.entries]
        manifest: dict[str, object] = {
            "manifest_version": PROVIDER_FIXTURE_PACK_MANIFEST_VERSION,
            "root": str(self.root),
            "entry_count": len(entries),
            "provider_families": list(self.provider_families),
            "required_provider_families": sorted(REQUIRED_PROVIDER_FIXTURE_FAMILIES),
            "entries": entries,
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_provider_fixture_pack_corpus(root: str | Path | None = None) -> ProviderFixturePackCorpus:
    """Load and validate recorded provider fixture packs without network access."""

    corpus_root = Path(root) if root is not None else DEFAULT_PROVIDER_FIXTURE_PACK_ROOT
    if not corpus_root.is_dir():
        raise ProviderFixturePackError(f"provider fixture pack root does not exist: {corpus_root}")
    entries = tuple(_load_entry(path) for path in sorted(corpus_root.iterdir()) if path.is_dir())
    corpus = ProviderFixturePackCorpus(root=corpus_root, entries=entries)
    missing = REQUIRED_PROVIDER_FIXTURE_FAMILIES.difference(corpus.provider_families)
    if missing:
        raise ProviderFixturePackError(
            "provider fixture pack corpus is missing required provider families: " + ", ".join(sorted(missing))
        )
    return corpus


def build_provider_fixture_pack_manifest(root: str | Path | None = None) -> dict[str, object]:
    """Validate provider fixture packs and return a deterministic manifest."""

    return load_provider_fixture_pack_corpus(root).manifest()


def write_provider_fixture_pack_manifest(
    output: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Write the deterministic provider fixture pack manifest."""

    manifest = build_provider_fixture_pack_manifest(root)
    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_entry(path: Path) -> ProviderFixturePackEntry:
    metadata_path = path / "metadata.json"
    pack_path = path / "pack.json"
    metadata = _read_json_object(metadata_path)
    pack = _read_json_object(pack_path)
    _validate_metadata(path.name, metadata)
    _validate_pack(path.name, pack)
    provider_family = canonical_provider_family(str(pack["provider_family"]))
    assert provider_family is not None
    if metadata["provider_family"] != provider_family:
        raise ProviderFixturePackError(
            f"{path.name}/metadata.json provider_family must match pack.json provider_family"
        )
    _reject_secret_like_values(path.name, metadata)
    _reject_secret_like_values(path.name, pack)
    return ProviderFixturePackEntry(
        entry_id=str(metadata["id"]),
        provider_family=provider_family,
        path=path,
        metadata=metadata,
        pack=pack,
        metadata_sha256=_sha256(metadata_path),
        pack_sha256=_sha256(pack_path),
    )


def _validate_metadata(dirname: str, metadata: dict[str, object]) -> None:
    required_strings = (
        "id",
        "provider_family",
        "display_name",
        "source",
        "license",
        "fixture_revision",
        "upstream_reference",
        "upstream_revision",
        "reproducibility_notes",
    )
    for key in required_strings:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ProviderFixturePackError(f"{dirname}/metadata.json field '{key}' must be a non-empty string")
    if metadata["id"] != dirname:
        raise ProviderFixturePackError(f"{dirname}/metadata.json id must match its directory name")
    if canonical_provider_family(str(metadata["provider_family"])) not in SUPPORTED_PROVIDER_FAMILIES:
        raise ProviderFixturePackError(f"{dirname}/metadata.json has unsupported provider_family")
    if metadata.get("download_required") is not False:
        raise ProviderFixturePackError(
            f"{dirname}/metadata.json field 'download_required' must be false for offline fixtures"
        )
    if metadata.get("secrets_included") is not False:
        raise ProviderFixturePackError(
            f"{dirname}/metadata.json field 'secrets_included' must be false"
        )
    if not isinstance(metadata.get("anonymized"), bool):
        raise ProviderFixturePackError(f"{dirname}/metadata.json field 'anonymized' must be boolean")


def _validate_pack(dirname: str, pack: dict[str, object]) -> None:
    for key in ("provider", "provider_family"):
        value = pack.get(key)
        if not isinstance(value, str) or not value:
            raise ProviderFixturePackError(f"{dirname}/pack.json field '{key}' must be a non-empty string")
    if canonical_provider_family(str(pack["provider_family"])) not in SUPPORTED_PROVIDER_FAMILIES:
        raise ProviderFixturePackError(f"{dirname}/pack.json has unsupported provider_family")
    for section in ("request", "response", "stops", "streaming", "errors", "limits"):
        if not isinstance(pack.get(section), dict):
            raise ProviderFixturePackError(f"{dirname}/pack.json section '{section}' must be an object")
    request = _mapping(pack["request"])
    response = _mapping(pack["response"])
    tool_calls = _mapping(response.get("tool_calls"))
    _require_non_empty_strings(dirname, request.get("fields"), "request.fields")
    _require_non_empty_strings(dirname, response.get("fields"), "response.fields")
    for key in ("name_path", "arguments_path", "argument_encoding"):
        value = tool_calls.get(key)
        if not isinstance(value, str) or not value:
            raise ProviderFixturePackError(f"{dirname}/pack.json response.tool_calls.{key} must be a string")
    _require_non_empty_strings(dirname, _mapping(pack["stops"]).get("sequences"), "stops.sequences")
    streaming = _mapping(pack["streaming"])
    if not isinstance(streaming.get("emits_argument_fragments"), bool):
        raise ProviderFixturePackError(
            f"{dirname}/pack.json streaming.emits_argument_fragments must be boolean"
        )
    errors = _mapping(pack["errors"])
    for key in ("code_path", "message_path", "rate_limit_path"):
        value = errors.get(key)
        if not isinstance(value, str) or not value:
            raise ProviderFixturePackError(f"{dirname}/pack.json errors.{key} must be a string")
    limits = _mapping(pack["limits"])
    for key in ("max_input_tokens", "max_output_tokens"):
        value = limits.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ProviderFixturePackError(f"{dirname}/pack.json limits.{key} must be a positive integer")
    edge_cases = pack.get("edge_cases")
    if not isinstance(edge_cases, list) or not edge_cases:
        raise ProviderFixturePackError(f"{dirname}/pack.json edge_cases must be a non-empty list")
    for index, edge_case in enumerate(edge_cases):
        if not isinstance(edge_case, dict):
            raise ProviderFixturePackError(f"{dirname}/pack.json edge_cases[{index}] must be an object")
        for key in ("id", "surface", "expected_behavior"):
            value = edge_case.get(key)
            if not isinstance(value, str) or not value:
                raise ProviderFixturePackError(f"{dirname}/pack.json edge_cases[{index}].{key} must be a string")


def _reject_secret_like_values(dirname: str, value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if normalized in _SECRET_KEY_NAMES:
                raise ProviderFixturePackError(
                    f"{dirname} contains secret-like field at {path}.{key_text}; fixture packs must be redacted"
                )
            _reject_secret_like_values(dirname, child, f"{path}.{key_text}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_like_values(dirname, child, f"{path}[{index}]")
        return
    if isinstance(value, str):
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                raise ProviderFixturePackError(
                    f"{dirname} contains secret-like value at {path}; fixture packs must be redacted"
                )


def reject_secret_like_values(dirname: str, value: object, path: str = "$") -> None:
    """Reject secret-shaped provider fixture content using the shared fixture scanner."""

    _reject_secret_like_values(dirname, value, path)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProviderFixturePackError(f"provider fixture pack file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderFixturePackError(
            f"provider fixture pack file is not valid JSON: {path}:{exc.lineno}:{exc.colno}"
        ) from exc
    if not isinstance(raw, dict):
        raise ProviderFixturePackError(f"provider fixture pack file must contain a JSON object: {path}")
    return raw


def _require_non_empty_strings(dirname: str, value: object, key: str) -> None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ProviderFixturePackError(f"{dirname}/pack.json field '{key}' must be a non-empty string list")


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
