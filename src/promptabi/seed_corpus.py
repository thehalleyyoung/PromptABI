"""Curated offline seed corpus for instruct tokenizer and chat-template fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactBundle,
    ArtifactKind,
    ArtifactLocation,
    ArtifactProvenance,
    ChatTemplateArtifact,
    TokenizerArtifact,
)


DEFAULT_SEED_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "seed_corpus"

REQUIRED_FAMILIES = frozenset(
    {
        "chatml",
        "deepseek",
        "fine-tune",
        "gemma",
        "llama",
        "mistral",
        "openai-compatible",
        "phi",
        "qwen",
        "zephyr",
    }
)
MANIFEST_VERSION = 1


class SeedCorpusError(ValueError):
    """Raised when a seed-corpus fixture is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SeedCorpusEntry:
    """One minimized, CPU-only tokenizer/chat-template fixture."""

    entry_id: str
    family: str
    path: Path
    metadata: dict[str, object]
    tokenizer_config: dict[str, object]
    metadata_sha256: str
    tokenizer_config_sha256: str

    @property
    def chat_template(self) -> str:
        template = self.tokenizer_config["chat_template"]
        assert isinstance(template, str)
        return template

    @property
    def roles(self) -> tuple[str, ...]:
        roles = self.metadata["roles"]
        assert isinstance(roles, list)
        return tuple(str(role) for role in roles)

    @property
    def sentinels(self) -> tuple[str, ...]:
        sentinels = self.metadata["sentinels"]
        assert isinstance(sentinels, list)
        return tuple(str(sentinel) for sentinel in sentinels)

    @property
    def expected_behaviors(self) -> tuple[str, ...]:
        behaviors = self.metadata["expected_behaviors"]
        assert isinstance(behaviors, list)
        return tuple(str(behavior) for behavior in behaviors)

    def artifacts(self) -> tuple[TokenizerArtifact, ChatTemplateArtifact]:
        """Expose this fixture as concrete PromptABI artifacts."""

        provenance = ArtifactProvenance(
            version=str(self.metadata["fixture_revision"]),
            sha256=self.tokenizer_config_sha256,
            license=str(self.metadata["license"]),
            source=str(self.metadata["source"]),
        )
        tokenizer = TokenizerArtifact(
            kind=ArtifactKind.TOKENIZER,
            name=f"{self.entry_id}-tokenizer",
            location=ArtifactLocation(path=str(self.path)),
            provenance=ArtifactProvenance(
                version=str(self.metadata["fixture_revision"]),
                source=str(self.metadata["source"]),
            ),
            family=self.family,
            added_tokens=self.sentinels,
            metadata=(("corpus_entry", self.entry_id),),
        )
        template = ChatTemplateArtifact(
            kind=ArtifactKind.CHAT_TEMPLATE,
            name=f"{self.entry_id}-chat-template",
            location=ArtifactLocation(path=str(self.path / "tokenizer_config.json")),
            provenance=provenance,
            roles=self.roles,
            add_generation_prompt=bool(self.metadata["supports_generation_prompt"]),
            metadata=(
                ("corpus_entry", self.entry_id),
                ("template_family", self.family),
            ),
        )
        return tokenizer, template

    def to_manifest_entry(self) -> dict[str, object]:
        """Render deterministic metadata for corpus update manifests."""

        return {
            "id": self.entry_id,
            "family": self.family,
            "display_name": self.metadata["display_name"],
            "source": self.metadata["source"],
            "license": self.metadata["license"],
            "fixture_revision": self.metadata["fixture_revision"],
            "upstream_reference": self.metadata["upstream_reference"],
            "upstream_revision": self.metadata["upstream_revision"],
            "download_required": self.metadata["download_required"],
            "reproducibility_notes": self.metadata["reproducibility_notes"],
            "roles": list(self.roles),
            "sentinels": list(self.sentinels),
            "supports_generation_prompt": self.metadata["supports_generation_prompt"],
            "expected_behaviors": list(self.expected_behaviors),
            "metadata_sha256": self.metadata_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "fixture_sha256": _entry_manifest_hash(self),
            "files": {
                "metadata": "metadata.json",
                "tokenizer_config": "tokenizer_config.json",
            },
        }


@dataclass(frozen=True, slots=True)
class SeedCorpus:
    """A deterministic collection of curated seed-corpus entries."""

    root: Path
    entries: tuple[SeedCorpusEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda entry: entry.entry_id)))
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise SeedCorpusError("seed corpus contains duplicate entry ids")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({entry.family for entry in self.entries}))

    def by_id(self, entry_id: str) -> SeedCorpusEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(entry_id)

    def by_family(self, family: str) -> tuple[SeedCorpusEntry, ...]:
        return tuple(entry for entry in self.entries if entry.family == family)

    def artifact_bundle(self) -> ArtifactBundle:
        artifacts = []
        for entry in self.entries:
            artifacts.extend(entry.artifacts())
        return ArtifactBundle(tuple(artifacts))

    def manifest(self) -> dict[str, object]:
        """Build the deterministic corpus update manifest."""

        entries = [entry.to_manifest_entry() for entry in self.entries]
        manifest: dict[str, object] = {
            "manifest_version": MANIFEST_VERSION,
            "root": str(self.root),
            "entry_count": len(entries),
            "families": list(self.families),
            "required_families": sorted(REQUIRED_FAMILIES),
            "entries": entries,
        }
        manifest["manifest_sha256"] = _stable_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        return manifest


def load_seed_corpus(root: str | Path | None = None) -> SeedCorpus:
    """Load and validate the curated CPU-only seed corpus."""

    corpus_root = Path(root) if root is not None else DEFAULT_SEED_CORPUS_ROOT
    if not corpus_root.is_dir():
        raise SeedCorpusError(f"seed corpus root does not exist: {corpus_root}")
    entries = tuple(_load_entry(path) for path in sorted(corpus_root.iterdir()) if path.is_dir())
    corpus = SeedCorpus(root=corpus_root, entries=entries)
    missing = REQUIRED_FAMILIES.difference(corpus.families)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SeedCorpusError(f"seed corpus is missing required families: {missing_text}")
    return corpus


def build_seed_corpus_manifest(root: str | Path | None = None) -> dict[str, object]:
    """Validate the seed corpus and return a deterministic update manifest."""

    return load_seed_corpus(root).manifest()


def write_seed_corpus_manifest(
    output: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Write the deterministic seed-corpus manifest to disk."""

    manifest = build_seed_corpus_manifest(root)
    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_entry(path: Path) -> SeedCorpusEntry:
    metadata_path = path / "metadata.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    metadata = _read_json_object(metadata_path)
    tokenizer_config = _read_json_object(tokenizer_config_path)
    _validate_metadata(path.name, metadata)
    _validate_tokenizer_config(path.name, tokenizer_config)
    _validate_consistency(path.name, metadata, tokenizer_config)
    return SeedCorpusEntry(
        entry_id=str(metadata["id"]),
        family=str(metadata["family"]),
        path=path,
        metadata=metadata,
        tokenizer_config=tokenizer_config,
        metadata_sha256=_sha256(metadata_path),
        tokenizer_config_sha256=_sha256(tokenizer_config_path),
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SeedCorpusError(f"seed corpus file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SeedCorpusError(f"seed corpus file is not valid JSON: {path}:{exc.lineno}:{exc.colno}") from exc
    if not isinstance(raw, dict):
        raise SeedCorpusError(f"seed corpus file must contain a JSON object: {path}")
    return raw


def _validate_metadata(dirname: str, metadata: dict[str, object]) -> None:
    required_strings = (
        "id",
        "family",
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
            raise SeedCorpusError(f"{dirname}/metadata.json field '{key}' must be a non-empty string")
    if metadata["id"] != dirname:
        raise SeedCorpusError(f"{dirname}/metadata.json id must match its directory name")
    if metadata["family"] not in REQUIRED_FAMILIES:
        raise SeedCorpusError(f"{dirname}/metadata.json has unsupported family '{metadata['family']}'")
    for key in ("roles", "sentinels", "expected_behaviors"):
        _require_non_empty_strings(dirname, metadata.get(key), key)
    if not isinstance(metadata.get("supports_generation_prompt"), bool):
        raise SeedCorpusError(
            f"{dirname}/metadata.json field 'supports_generation_prompt' must be boolean"
        )
    if metadata.get("download_required") is not False:
        raise SeedCorpusError(
            f"{dirname}/metadata.json field 'download_required' must be false for CPU-only fixtures"
        )


def _validate_tokenizer_config(dirname: str, tokenizer_config: dict[str, object]) -> None:
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or "{{" not in chat_template or "messages" not in chat_template:
        raise SeedCorpusError(
            f"{dirname}/tokenizer_config.json must contain a minimized Jinja chat_template over messages"
        )
    specials = _special_token_values(tokenizer_config)
    if len(specials) < 2:
        raise SeedCorpusError(f"{dirname}/tokenizer_config.json must declare at least two special tokens")


def _validate_consistency(
    dirname: str,
    metadata: dict[str, object],
    tokenizer_config: dict[str, object],
) -> None:
    template = tokenizer_config["chat_template"]
    assert isinstance(template, str)
    for sentinel in _iter_strings(metadata["sentinels"]):
        if sentinel not in template and sentinel not in _special_token_values(tokenizer_config):
            raise SeedCorpusError(
                f"{dirname} sentinel {sentinel!r} must appear in the template or tokenizer special tokens"
            )
    if metadata["supports_generation_prompt"] and "add_generation_prompt" not in template:
        raise SeedCorpusError(
            f"{dirname} declares generation-prompt support but template does not branch on add_generation_prompt"
        )


def _require_non_empty_strings(dirname: str, value: object, key: str) -> None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise SeedCorpusError(f"{dirname}/metadata.json field '{key}' must be a non-empty string list")


def _iter_strings(value: object) -> Iterable[str]:
    assert isinstance(value, list)
    return (str(item) for item in value)


def _special_token_values(tokenizer_config: dict[str, object]) -> set[str]:
    values: set[str] = set()
    for key, value in tokenizer_config.items():
        if key.endswith("_token") and isinstance(value, str) and value:
            values.add(value)
    additional = tokenizer_config.get("additional_special_tokens", [])
    if isinstance(additional, list):
        values.update(item for item in additional if isinstance(item, str) and item)
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry_manifest_hash(entry: SeedCorpusEntry) -> str:
    return _stable_json_hash(
        {
            "metadata_sha256": entry.metadata_sha256,
            "tokenizer_config_sha256": entry.tokenizer_config_sha256,
        }
    )


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
