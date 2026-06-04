"""Tokenizer and generation-config drift detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKind, TokenizerArtifact
from .loaders import LoadedArtifact


class TokenizerDriftKind(StrEnum):
    """Kinds of tokenizer/config changes that can break prompt-interface contracts."""

    ADDED_TOKENS = "added-token-change"
    BOS_EOS = "bos-eos-change"
    CHAT_TEMPLATE = "chat-template-change"
    NORMALIZATION = "normalization-change"
    SPECIAL_TOKEN_ID = "special-token-id-change"
    STOP_POLICY = "stop-policy-change"


@dataclass(frozen=True, slots=True)
class TokenizerConfigSnapshot:
    """Stable summary of tokenizer files used for drift comparison."""

    path: str
    revision: str | None
    special_tokens: tuple[tuple[str, str | int | None, int | None], ...] = ()
    added_tokens: tuple[tuple[str, int | None, bool], ...] = ()
    normalizer_signature: str | None = None
    chat_template_sha256: str | None = None
    chat_template_length: int | None = None
    bos_token: str | None = None
    bos_token_id: int | None = None
    eos_token: str | None = None
    eos_token_id: int | None = None
    add_bos_token: bool | None = None
    add_eos_token: bool | None = None
    stop_sequences: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()

    def value_for(self, field: str) -> object:
        return getattr(self, field)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"path": self.path}
        for field in _SNAPSHOT_FIELDS:
            value = getattr(self, field)
            if value not in (None, (), ""):
                data[field] = value
        return data


@dataclass(frozen=True, slots=True)
class TokenizerDriftFinding:
    """One baseline/current tokenizer difference."""

    kind: TokenizerDriftKind
    field: str
    baseline: object
    current: object
    baseline_path: str
    current_path: str
    baseline_revision: str | None = None
    current_revision: str | None = None

    @property
    def breaking(self) -> bool:
        return self.kind in {
            TokenizerDriftKind.BOS_EOS,
            TokenizerDriftKind.CHAT_TEMPLATE,
            TokenizerDriftKind.SPECIAL_TOKEN_ID,
            TokenizerDriftKind.STOP_POLICY,
        }


@dataclass(frozen=True, slots=True)
class TokenizerDriftAbstention:
    """A tokenizer drift baseline could not be compared deterministically."""

    artifact_name: str
    reason: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class TokenizerDriftReport:
    """Drift comparison result for configured tokenizer artifacts."""

    findings: tuple[TokenizerDriftFinding, ...]
    abstentions: tuple[TokenizerDriftAbstention, ...]
    compared: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.findings and not self.abstentions


_SNAPSHOT_FIELDS = (
    "revision",
    "special_tokens",
    "added_tokens",
    "normalizer_signature",
    "chat_template_sha256",
    "chat_template_length",
    "bos_token",
    "bos_token_id",
    "eos_token",
    "eos_token_id",
    "add_bos_token",
    "add_eos_token",
    "stop_sequences",
    "stop_token_ids",
)

_FIELD_KINDS: dict[str, TokenizerDriftKind] = {
    "special_tokens": TokenizerDriftKind.SPECIAL_TOKEN_ID,
    "added_tokens": TokenizerDriftKind.ADDED_TOKENS,
    "normalizer_signature": TokenizerDriftKind.NORMALIZATION,
    "chat_template_sha256": TokenizerDriftKind.CHAT_TEMPLATE,
    "chat_template_length": TokenizerDriftKind.CHAT_TEMPLATE,
    "bos_token": TokenizerDriftKind.BOS_EOS,
    "bos_token_id": TokenizerDriftKind.BOS_EOS,
    "eos_token": TokenizerDriftKind.BOS_EOS,
    "eos_token_id": TokenizerDriftKind.BOS_EOS,
    "add_bos_token": TokenizerDriftKind.BOS_EOS,
    "add_eos_token": TokenizerDriftKind.BOS_EOS,
    "stop_sequences": TokenizerDriftKind.STOP_POLICY,
    "stop_token_ids": TokenizerDriftKind.STOP_POLICY,
}


def analyze_tokenizer_config_drift(loaded_artifacts: tuple[LoadedArtifact, ...]) -> TokenizerDriftReport:
    """Compare current tokenizer artifacts against configured baseline snapshots.

    A tokenizer artifact opts in with ``metadata.drift_baseline_path``. The path is
    resolved relative to the current tokenizer file's parent, or relative to the
    current tokenizer directory's parent. Both sides are parsed from real
    tokenizer files (``tokenizer_config.json``, ``tokenizer.json``,
    ``special_tokens_map.json``, and ``generation_config.json``).
    """

    findings: list[TokenizerDriftFinding] = []
    abstentions: list[TokenizerDriftAbstention] = []
    compared: list[tuple[str, str]] = []

    for loaded in loaded_artifacts:
        artifact = loaded.artifact
        if artifact.kind is not ArtifactKind.TOKENIZER or not isinstance(artifact, TokenizerArtifact):
            continue
        current_path_text = artifact.location.path
        if current_path_text is None:
            continue
        baseline_value = _metadata_value(artifact, "drift_baseline_path")
        if baseline_value is None:
            continue
        if not isinstance(baseline_value, str) or not baseline_value:
            abstentions.append(
                TokenizerDriftAbstention(
                    artifact_name=artifact.name,
                    reason="metadata.drift_baseline_path must be a non-empty string",
                    path=current_path_text,
                )
            )
            continue

        current_path = Path(current_path_text)
        baseline_path = _resolve_baseline_path(current_path, baseline_value)
        try:
            current = load_tokenizer_config_snapshot(
                current_path,
                revision=artifact.provenance.revision or artifact.provenance.version,
            )
            baseline = load_tokenizer_config_snapshot(baseline_path, revision=_metadata_str(artifact, "drift_baseline_revision"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            abstentions.append(
                TokenizerDriftAbstention(
                    artifact_name=artifact.name,
                    reason=str(exc),
                    path=str(baseline_path),
                )
            )
            continue

        compared.append((baseline.path, current.path))
        findings.extend(_compare_snapshots(baseline, current))

    return TokenizerDriftReport(
        findings=tuple(findings),
        abstentions=tuple(abstentions),
        compared=tuple(compared),
    )


def load_tokenizer_config_snapshot(path: str | Path, *, revision: str | None = None) -> TokenizerConfigSnapshot:
    """Load a stable tokenizer/config snapshot from a tokenizer file or directory."""

    root = Path(path)
    if not root.exists():
        raise ValueError(f"tokenizer drift baseline does not exist: {root}")
    files = _snapshot_files(root)
    tokenizer_config = _read_json_object(files.get("tokenizer_config.json"))
    tokenizer_json = _read_json_object(files.get("tokenizer.json"))
    special_tokens_map = _read_json_object(files.get("special_tokens_map.json"))
    generation_config = _read_json_object(files.get("generation_config.json"))

    special_tokens = _special_tokens(tokenizer_config, special_tokens_map, tokenizer_json)
    added_tokens = _added_tokens(tokenizer_config, tokenizer_json)
    chat_template = _optional_str_value(tokenizer_config.get("chat_template"))
    normalizer_signature = _normalizer_signature(tokenizer_config, tokenizer_json)
    bos_token, bos_token_id = _token_and_id("bos_token", "bos_token_id", tokenizer_config, special_tokens_map, tokenizer_json)
    eos_token, eos_token_id = _token_and_id("eos_token", "eos_token_id", tokenizer_config, special_tokens_map, tokenizer_json)
    stop_sequences = _string_tuple(generation_config.get("stop_strings") or tokenizer_config.get("stop_strings"))
    stop_token_ids = _int_tuple(generation_config.get("eos_token_id") or tokenizer_config.get("eos_token_id"))

    return TokenizerConfigSnapshot(
        path=str(root),
        revision=revision,
        special_tokens=special_tokens,
        added_tokens=added_tokens,
        normalizer_signature=normalizer_signature,
        chat_template_sha256=_sha256_text(chat_template) if chat_template is not None else None,
        chat_template_length=len(chat_template) if chat_template is not None else None,
        bos_token=bos_token,
        bos_token_id=bos_token_id,
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        add_bos_token=_optional_bool(tokenizer_config.get("add_bos_token")),
        add_eos_token=_optional_bool(tokenizer_config.get("add_eos_token")),
        stop_sequences=stop_sequences,
        stop_token_ids=stop_token_ids,
    )


def _compare_snapshots(
    baseline: TokenizerConfigSnapshot,
    current: TokenizerConfigSnapshot,
) -> tuple[TokenizerDriftFinding, ...]:
    findings: list[TokenizerDriftFinding] = []
    for field in _SNAPSHOT_FIELDS:
        if field == "revision":
            continue
        baseline_value = baseline.value_for(field)
        current_value = current.value_for(field)
        if baseline_value in (None, ()) and current_value in (None, ()):
            continue
        if baseline_value == current_value:
            continue
        findings.append(
            TokenizerDriftFinding(
                kind=_FIELD_KINDS[field],
                field=field,
                baseline=baseline_value,
                current=current_value,
                baseline_path=baseline.path,
                current_path=current.path,
                baseline_revision=baseline.revision,
                current_revision=current.revision,
            )
        )
    return tuple(findings)


def _snapshot_files(root: Path) -> dict[str, Path]:
    if root.is_file():
        return {root.name: root}
    if not root.is_dir():
        raise ValueError(f"tokenizer snapshot path is neither a file nor a directory: {root}")
    files = {
        name: root / name
        for name in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "generation_config.json")
        if (root / name).is_file()
    }
    if not files:
        raise ValueError(f"tokenizer snapshot directory has no supported tokenizer/config files: {root}")
    return files


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"tokenizer drift file must be a JSON object: {path}")
    return raw


def _special_tokens(
    tokenizer_config: dict[str, Any],
    special_tokens_map: dict[str, Any],
    tokenizer_json: dict[str, Any],
) -> tuple[tuple[str, str | int | None, int | None], ...]:
    tokens: dict[str, tuple[str | int | None, int | None]] = {}
    for name in ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"):
        text, token_id = _token_and_id(name, f"{name}_id", tokenizer_config, special_tokens_map, tokenizer_json)
        if text is not None or token_id is not None:
            tokens[name] = (text, token_id)
    for item in _json_added_tokens(tokenizer_config) + _json_added_tokens(tokenizer_json):
        if item.get("special") is True:
            content = _optional_str_value(item.get("content"))
            token_id = _optional_int(item.get("id"))
            if content is not None:
                tokens.setdefault(content, (content, token_id))
    return tuple((name, value, token_id) for name, (value, token_id) in sorted(tokens.items()))


def _added_tokens(
    tokenizer_config: dict[str, Any],
    tokenizer_json: dict[str, Any],
) -> tuple[tuple[str, int | None, bool], ...]:
    tokens: dict[str, tuple[int | None, bool]] = {}
    for item in _json_added_tokens(tokenizer_config) + _json_added_tokens(tokenizer_json):
        content = _optional_str_value(item.get("content"))
        if content is None:
            continue
        token_id = _optional_int(item.get("id"))
        special = item.get("special") is True
        tokens[content] = (token_id, special)
    return tuple((content, token_id, special) for content, (token_id, special) in sorted(tokens.items()))


def _json_added_tokens(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    value = mapping.get("added_tokens")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalizer_signature(tokenizer_config: dict[str, Any], tokenizer_json: dict[str, Any]) -> str | None:
    for value in (tokenizer_config.get("normalizer"), tokenizer_json.get("normalizer")):
        if value is None:
            continue
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    for key in ("do_lower_case", "clean_up_tokenization_spaces", "strip_accents"):
        if key in tokenizer_config:
            return json.dumps(
                {name: tokenizer_config[name] for name in ("do_lower_case", "clean_up_tokenization_spaces", "strip_accents") if name in tokenizer_config},
                sort_keys=True,
                separators=(",", ":"),
            )
    return None


def _token_and_id(
    token_key: str,
    id_key: str,
    tokenizer_config: dict[str, Any],
    special_tokens_map: dict[str, Any],
    tokenizer_json: dict[str, Any],
) -> tuple[str | None, int | None]:
    token_value = tokenizer_config.get(token_key, special_tokens_map.get(token_key))
    token_text = _special_token_text(token_value)
    token_id = _optional_int(tokenizer_config.get(id_key))
    if token_id is None and token_text is not None:
        token_id = _id_for_added_token(token_text, tokenizer_config) or _id_for_added_token(token_text, tokenizer_json)
    return token_text, token_id


def _special_token_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        return _optional_str_value(value.get("content"))
    return None


def _id_for_added_token(content: str, mapping: dict[str, Any]) -> int | None:
    for item in _json_added_tokens(mapping):
        if item.get("content") == content:
            return _optional_int(item.get("id"))
    return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list):
        return tuple(sorted(dict.fromkeys(item for item in value if isinstance(item, str) and item)))
    return ()


def _int_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, list):
        return tuple(sorted(dict.fromkeys(item for item in value if isinstance(item, int) and not isinstance(item, bool))))
    return ()


def _optional_str_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_value(artifact: TokenizerArtifact, key: str) -> object | None:
    return dict(artifact.metadata).get(key)


def _metadata_str(artifact: TokenizerArtifact, key: str) -> str | None:
    value = _metadata_value(artifact, key)
    return value if isinstance(value, str) and value else None


def _resolve_baseline_path(current_path: Path, baseline_value: str) -> Path:
    baseline_path = Path(baseline_value)
    if baseline_path.is_absolute():
        return baseline_path
    primary = (current_path.parent / baseline_path).resolve()
    if primary.exists() or current_path.is_file():
        return primary
    secondary = (current_path / baseline_path).resolve()
    return secondary if secondary.exists() else primary
